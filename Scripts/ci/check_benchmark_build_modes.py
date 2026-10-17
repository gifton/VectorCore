#!/usr/bin/env python3
"""Four actual benchmark products and linked INT8 forwarding controls (V02).

Works under both SwiftPM build systems. Product, object and module discovery
goes through Scripts/ci/swiftpm_products.py, which accepts exactly the native
and the swiftbuild layouts and fails on a missing or ambiguous product.

Builds may run longer than executions. Every executable child has file capture,
a 15 second deadline and at most two seconds of owned-process-group cleanup.
No ordinary production object is instrumented or overwritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import swiftpm_products as products  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "Scripts/ci/fixtures/V02"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run(argv, directory, name, timeout=15, expected=0):
    directory.mkdir(parents=True, exist_ok=True)
    record = directory / (name + ".json")
    assert not record.exists(), record
    stdout, stderr = directory / (name + ".stdout"), directory / (name + ".stderr")
    env = {key: value for key, value in os.environ.items() if not key.startswith("VC_")}
    provenance = {"argv": list(map(str, argv)), "cwd": str(ROOT), "deadlineSeconds": timeout,
                  "environment": {key: env[key] for key in ["PATH", "DEVELOPER_DIR", "SDKROOT", "SWIFT_EXEC", "LANG"] if key in env},
                  "timeout": False, "status": "starting"}
    write_json(record, provenance)
    process = None
    old_handlers = {}
    def stop(signum, frame):
        if process is not None:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        raise KeyboardInterrupt("driver signal " + str(signum))
    # Handlers are installed before launch; the parent owns only this child's group.
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, stop)
    started = time.monotonic()
    try:
        with stdout.open("xb") as out, stderr.open("xb") as err:
            process = subprocess.Popen(list(map(str, argv)), cwd=ROOT, env=env, stdout=out, stderr=err, start_new_session=True)
            provenance.update(pid=process.pid, status="running")
            write_json(record, provenance)
            try:
                status = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                provenance.update(timeout=True, status="timeout", exit=None)
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                try:
                    process.wait(timeout=2)
                    provenance["cleanupCompleted"] = True
                except subprocess.TimeoutExpired:
                    provenance["cleanupCompleted"] = False
                # Do not interpret a timed-out child's return status.
                raise RuntimeError("Execution timeout: " + name)
            provenance.update(status="completed", exit=status)
            allowed = (expected,) if isinstance(expected, int) else tuple(expected)
            if status not in allowed:
                raise RuntimeError(f"{name}: exit {status}, expected {expected}; see {stderr}")
    except BaseException as error:
        provenance["error"] = str(error)
        if process is not None and provenance["status"] == "running":
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            try: process.wait(timeout=2)
            except subprocess.TimeoutExpired: pass
        raise
    finally:
        for sig, handler in old_handlers.items(): signal.signal(sig, handler)
        provenance["elapsedSeconds"] = time.monotonic() - started
        for key, path in [("stdout", stdout), ("stderr", stderr)]:
            if path.exists(): provenance[key] = {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
        write_json(record, provenance)
    return stdout.read_text()


def artifacts(build, output):
    rows = []
    for path in sorted(build.rglob("*")):
        if path.is_file() and (path.suffix in (".o", ".swiftmodule", ".swiftdoc", ".swiftsourceinfo", ".pcm") or path.name == "vectorcore-bench"):
            copy = output / path.relative_to(build)
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, copy)
            rows.append({"buildPath": str(path), "retainedPath": str(copy), "sha256": sha(copy), "bytes": copy.stat().st_size})
    assert rows
    return rows


def clang_command(log, source):
    found = []
    for line in log.splitlines():
        try: parts = shlex.split(line)
        except ValueError: continue
        if parts and Path(parts[0]).name in ("clang", "clang++") and "-c" in parts and parts[parts.index("-c") + 1] == str(source):
            found.append(parts)
    assert len(found) == 1, f"Expected one actual dispatcher compiler command, found {len(found)}"
    return found[0]


def replacement_command(original, source, output):
    args = list(original)
    args[args.index("-c") + 1] = str(source)
    args[args.index("-o") + 1] = str(output)
    for flag in ("-MF", "-MT", "-MQ"):
        if flag in args: args[args.index(flag) + 1] = str(output.with_suffix(".d"))
    if "-index-store-path" in args: args[args.index("-index-store-path") + 1] = str(output.parent / "index")
    # swiftbuild adds per-object side outputs; never overwrite the real build's.
    for flag, path in (("--serialize-diagnostics", output.with_suffix(".dia")), ("-serialize-diagnostics", output.with_suffix(".dia")),
                       ("-index-unit-output-path", output)):
        if flag in args: args[args.index(flag) + 1] = str(path)
    return args


def object_roots(build):
    """Directories holding the real build's per-source objects and their side files (.d, .dia, index units)."""
    return sorted({p.parent for target in ("VectorCore", "VectorCoreC") for p in products.source_objects(build, target)})


def snapshot(roots):
    """sha256 of every file under the real build's object directories (objects and side files)."""
    return {str(p): sha(p) for root in roots for p in sorted(root.rglob("*")) if p.is_file()}


def assert_isolated(command, roots, label):
    """No argument of a rewritten compile command may resolve under the real build's object directories.

    Module-cache writes (-fmodules-cache-path) go to the build's ModuleCache, which is cache-only
    (content-addressed, regenerated on demand) and lies outside these object directories.
    """
    resolved_roots = [r.resolve() for r in roots]
    for argument in map(str, command):
        for candidate in [argument] + argument.split("=", 1)[1:]:
            if candidate.startswith("/"):
                path = Path(candidate).resolve()
                assert not any(path == r or r in path.parents for r in resolved_roots), f"{label}: argument {argument} resolves under the real build"


def find_executable(scratch, configuration):
    # native: <arch>/debug/vectorcore-bench; swiftbuild: out/Products/Debug/vectorcore-bench.
    executables = [p for p in scratch.rglob("vectorcore-bench")
                   if p.is_file() and not p.is_symlink() and p.parent.name.lower() == configuration.lower()]
    assert len(executables) == 1, executables
    return executables[0]


def cell(configuration, enabled, scratch_root, output, jobs):
    name = configuration + ("-c-on" if enabled else "-c-off")
    directory = output / name
    directory.mkdir()
    scratch = scratch_root / name
    if scratch.exists(): raise RuntimeError("Refusing existing scratch: " + str(scratch))
    argv = ["swift", "build", "--product", "vectorcore-bench", "-c", configuration, "--jobs", str(jobs), "--scratch-path", str(scratch), "--verbose"]
    if enabled: argv += ["-Xswiftc", "-DVC_USE_C_KERNELS"]
    run(argv, directory, "build", timeout=1200)
    executable = find_executable(scratch, configuration)
    build = executable.parent
    assert products.find_executable(build, "vectorcore-bench") == executable
    layout = products.detect_layout(build)
    retained = artifacts(build, directory / "artifacts")
    write_json(directory / "artifacts.json", retained)
    command = [executable, "--suites", "distance", "--dims", "512", "--filter", "dist.euclidean.512.optimized", "--repeats", "1", "--samples", "2", "--min-time", "0", "--strict-correctness", "--format", "json"]
    request = "--use-ckernels" if enabled else "--no-use-ckernels"
    for mode, flags in [("absent", []), ("matching", [request])]:
        report = directory / (mode + "-report.json")
        run(command + flags + ["--out", report], directory, "benchmark-" + mode)
        result = json.loads(report.read_text())
        metadata = result["metadata"]
        assert metadata["coreConfiguration"] == configuration and metadata["benchmarkConfiguration"] == configuration
        assert metadata["flags"]["useCKernels"] is enabled
        assert metadata["flags"].get("requestedCKernels") == (enabled if mode == "matching" else None)
        assert metadata["selectedCaseIDs"] == ["dist.euclidean.512.optimized"]
        assert len(result["results"]) == 1 and not result["failures"]
        assert result["results"][0]["iterations"] == 2
    opposite = "--no-use-ckernels" if enabled else "--use-ckernels"
    run(command + [opposite, "--out", directory / "must-not-exist.json"], directory, "benchmark-mismatch", expected=1)
    assert not (directory / "must-not-exist.json").exists()
    assert "does not match compiled" in (directory / "benchmark-mismatch.stderr").read_text()

    # Recompile the one real dispatcher source, changing only the entry symbol.
    original_source = ROOT / "Sources/VectorCoreC/common/vc_dispatch.c"
    text = original_source.read_text()
    assert len(re.findall(r"\bvc_dot_int8\b", text)) == 1
    derivative = directory / "dispatcher-renamed.c"
    derivative.write_text(re.sub(r"\bvc_dot_int8\b", "v02_real_vc_dot_int8", text))
    original_command = clang_command((directory / "build.stdout").read_text() + (directory / "build.stderr").read_text(), original_source)
    dispatcher = directory / "dispatcher-renamed.o"
    roots = object_roots(build)
    real_before = snapshot(roots)
    assert real_before
    dispatcher_command = replacement_command(original_command, derivative, dispatcher)
    assert_isolated(dispatcher_command, roots, "compile-real-dispatcher")
    run(dispatcher_command, directory, "compile-real-dispatcher", timeout=120)
    shim = directory / "forwarding.o"
    shim_command = replacement_command(original_command, FIXTURES / "forwarding.c", shim)
    assert_isolated(shim_command, roots, "compile-forwarding")
    run(shim_command, directory, "compile-forwarding", timeout=120)
    library_objects = products.source_objects(build, "VectorCore")
    c_objects = products.source_objects(build, "VectorCoreC")
    dispatch_name = "vc_dispatch.c.o" if layout == "native" else "vc_dispatch.o"
    replaced = [p for p in c_objects if p.name == dispatch_name]
    assert len(replaced) == 1 and library_objects
    linked = library_objects + [p for p in c_objects if p not in replaced] + [dispatcher, shim]
    consumer = directory / "public-routes"
    link = ["swiftc", "-swift-version", "6", "-parse-as-library", "-O" if configuration == "release" else "-Onone", "-I", products.module_search_path(build), "-Xcc", "-fmodule-map-file=" + str(products.module_map(build, "VectorCoreC")), "-Xcc", "-I" + str(ROOT / "Sources/VectorCoreC/include"), FIXTURES / "public_routes.swift", *linked, "-framework", "Accelerate", "-o", consumer]
    run(link, directory, "link-public-routes", timeout=120)
    route_output = run([consumer, "1" if enabled else "0"], directory, "public-routes")
    route_rows = [json.loads(line[len("V02_ROUTE "):]) for line in route_output.splitlines() if line.startswith("V02_ROUTE ")]
    assert len(route_rows) == 36 and {r["dimension"] for r in route_rows} == {512,768,1536}
    assert len([line for line in route_output.splitlines() if line.startswith("V02_C_TAIL ")]) == 10
    assert "V02_ROUTE_ENTER" in route_output and "V02_ROUTE_COMPLETE widths=3 routes=36 tails=10" in route_output
    for row in route_rows:
        expected = 3 if enabled and (row["mode"] == "equal" or (row["mode"] == "unequal" and row["metric"] != "euclidean")) else 0
        assert row["calls"] == expected and row["expectedCalls"] == expected
    # Verify all ordinary retained compiled bytes stayed untouched during replacement/link.
    assert all(sha(row["buildPath"]) == row["sha256"] for row in retained)
    # Also the per-source objects and side files (.d, .dia, index units) the rewritten commands could have hit.
    assert snapshot(roots) == real_before, "real build object directory changed during replacement compile/link"
    write_json(directory / "real-build-snapshot.json", {"files": len(real_before), "unchanged": True})
    mapping = {"layout": layout, "originalDispatcher": {"path": str(replaced[0]), "sha256": sha(replaced[0])},
               "originalSource": {"path": str(original_source), "sha256": sha(original_source)},
               "derivativeSource": {"path": str(derivative), "sha256": sha(derivative)},
               "consumerSource": {"path": str(FIXTURES / "public_routes.swift"), "sha256": sha(FIXTURES / "public_routes.swift")},
               "shimSource": {"path": str(FIXTURES / "forwarding.c"), "sha256": sha(FIXTURES / "forwarding.c")},
               "linkedObjects": [{"path": str(p), "sha256": sha(p)} for p in linked],
               "consumer": {"path": str(consumer), "sha256": sha(consumer)}, "routeRows": route_rows}
    write_json(directory / "link-provenance.json", mapping)
    return {"name": name, "configuration": configuration, "cKernels": enabled, "actualBenchmarkCases": 1,
            "matchingRequestPassed": True, "absentRequestPassed": True, "mismatchRejected": True,
            "routeRows": 36, "directCTails": 10, "publicRouteCompletion": "V02_ROUTE_COMPLETE widths=3 routes=36 tails=10",
            "benchmarkSHA256": sha(executable), "consumerSHA256": sha(consumer), "artifactManifest": str(directory / "artifacts.json"),
            "linkProvenance": str(directory / "link-provenance.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 6: parser.error("jobs must be in 1...6")
    scratch, output = args.scratch_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    scratch.mkdir(parents=True, exist_ok=True)
    run(["swift", "--version"], output, "toolchain")
    sources = []
    for folder in ["Sources", "Benchmarks", "Scripts"]:
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file(): sources.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path)})
    sources.append({"path": "Package.swift", "sha256": sha(ROOT / "Package.swift")})
    write_json(output / "sources.json", sources)
    cells = []
    try:
        for configuration in ["debug", "release"]:
            for enabled in [False, True]:
                cells.append(cell(configuration, enabled, scratch, output, args.jobs))
                write_json(output / "progress.json", {"cells": cells})
        assert len(cells) == 4
        assert all(sha(ROOT / row["path"]) == row["sha256"] for row in sources)
        write_json(output / "result.json", {"schemaVersion": 1, "marker": "V02_BUILD_MODES_COMPLETE", "cells": cells})
        print("V02_BUILD_MODES_COMPLETE")
    except BaseException as error:
        write_json(output / "failure.json", {"error": str(error), "completedCells": cells})
        raise


if __name__ == "__main__":
    main()
