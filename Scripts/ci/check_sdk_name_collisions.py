#!/usr/bin/env python3
"""Fail when a public top-level VectorCore type name collides with an SDK type.

Why: a newer SDK can add a type whose name equals one of our public top-level
types (Xcode 27 added Foundation.ProgressReporter, colliding with the former
VectorCoreBenchmarking.ProgressReporter). Any file importing both modules then
sees an ambiguous name. This check closes that class mechanically.

It compares the public/open top-level struct, class, enum, protocol and
typealias names of the VectorCore and VectorCoreBenchmarking modules (from
`swift package dump-symbol-graph`) with the same kinds of names in the SDK
modules Foundation, Swift, Accelerate, simd, Dispatch and os (from
`swift-symbolgraph-extract` against `xcrun --show-sdk-path`).

Exit status: 0 no unreviewed collision; 1 unreviewed collision with a non-Swift
SDK module; 2 tool or environment failure. Collisions with the Swift stdlib are
informational only. Unverified rationale (root's stated reasoning, no test
exercises it): declarations in other imported modules shadow stdlib
declarations, so a stdlib-name collision does not make the name ambiguous.
Reviewed exceptions live in Scripts/ci/sdk_name_collision_allowlist.json as a
JSON list of {"name", "module", "reason"} objects. Matching is keyed on
(name, SDK module) only, not on the package module, so one entry also covers
re-exports (VectorCoreBenchmarking re-exports VectorCore.Logger). Allowlist
entries that no longer collide produce a WARNING (not a failure).
Each package module's extracted names are also cross-checked against the public top-level
declarations lexically present in Sources/<Module> (exit 2 on an unreviewed omission; reviewed
omissions live in SOURCE_OMISSION_ALLOWLIST). The extraction is not assumed complete.
Package symbol graphs always come from a fresh per-run scratch directory, so a
failed dump can never be satisfied by graphs left by an earlier run.

CI command (macOS runner with the Xcode under test selected):
    python3 Scripts/ci/check_sdk_name_collisions.py
Only the active SDK is checked; a collision introduced by a different SDK is
not detected by a run against this one.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PACKAGE_MODULES = ("VectorCore", "VectorCoreBenchmarking")
SDK_MODULES = ("Foundation", "Swift", "Accelerate", "simd", "Dispatch", "os")
KINDS = ("swift.struct", "swift.class", "swift.enum", "swift.protocol",
         "swift.typealias", "swift.actor")
PUBLIC_ACCESS = ("public", "open")
DUMP_TIMEOUT_S = 1200   # may include a full package build
EXTRACT_TIMEOUT_S = 300  # per SDK module
QUICK_TIMEOUT_S = 60
# Extraction self-check. Each name must appear in the extracted public
# top-level names of its module. Sentinels must exist in EVERY supported SDK
# (CI's Xcode 26.3 as well as Xcode 27): do not add a name an older SDK lacks
# (Foundation.ProgressReporter is Xcode 27 only and was removed for that reason).
# Verified by running the checker under CLT SDK MacOSX26.0 and Xcode 27 (T03 fix round 1).
SENTINELS = {
    "Foundation": ("URL", "Data", "Date", "UUID", "AttributedString"),
    "Swift": ("Array", "String"),
    "Accelerate": ("DSPComplex", "BNNSFilter"),
    "simd": ("simd_float4", "simd_quatf"),
    "Dispatch": ("DispatchQueue", "DispatchTime"),
    "os": ("Logger", "OSAllocatedUnfairLock"),
}


# Package self-check: stable public top-level types of OUR modules that a complete extraction
# must contain (verified against the real Xcode 27 and CLT 6.2 extractions). Same rule as the
# SDK sentinels: a missing name exits 2. Sentinels only sample the surface; completeness of the
# package graphs is checked separately by the source cross-check.
# VectorCoreBenchmarking also re-exports VectorCore (`@_exported import VectorCore`); toolchains
# differ on whether that module's graph contains the re-exported names (Xcode 27: yes, 254 names;
# CLT 6.2: no, 24 own names), so re-exported names are deliberately NOT sentinels there. The
# re-exported names are VectorCore's own, which are checked through the VectorCore graph. That
# graph is NOT assumed complete: sentinels only sample it, so the source cross-check below
# compares every module's extracted names with the public top-level declarations in
# Sources/<Module> and exits 2 on any omission that is not reviewed in
# SOURCE_OMISSION_ALLOWLIST.
# Decision on a nonzero `dump-symbol-graph` exit: fresh graphs are accepted ONLY if every
# package sentinel is present (the check below runs for every exit status), because a failed
# dump can leave a partial graph.
PACKAGE_SENTINELS = {
    "VectorCore": ("Vector", "DynamicVector", "Vector512Optimized", "DistanceMetric",
                   "VectorError", "EuclideanDistance"),
    "VectorCoreBenchmarking": ("BenchmarkProgressReporter", "JSONWriter", "BenchResult",
                               "EnvCapture", "RunFilters"),
}


# Source cross-check. A public top-level declaration found lexically in Sources/<Module> but absent
# from the extracted graph is an extraction gap and exits 2, unless reviewed here. Keyed on
# (module, name). Entries that no longer omit produce a WARNING.
SOURCE_OMISSION_ALLOWLIST = {
    ("VectorCore", "VectorCoreVersion"): (
        "swift-symbolgraph-extract / dump-symbol-graph drops any top-level type whose name starts with "
        "'<Module>Version' (it treats it like the clang-generated <Module>Version* version symbols). "
        "Evidence: the omission of VectorCoreVersion from the real tree (229 of 230 names) was observed on "
        "BOTH Xcode 27 and CLT 6.2. An isolated variant reproduction (a throwaway module named VectorCore "
        "in which 'VectorCoreVersion', 'VectorCoreVersionNumber' and 'VectorCoreVersion2' were all omitted "
        "while 'VectorCoreVers' was kept) was run once, on Xcode 27 only (its build artifacts name the "
        "XcodeDefault toolchain); it was not run on CLT 6.2. The type is declared plainly at "
        "Sources/VectorCore/Version.swift:27 (no underscore, @_spi, #if or availability). It can "
        "only collide with an SDK name spelled 'VectorCoreVersion', which no SDK declares."),
}
DECLARATION_RE = re.compile(
    r"^(?:@\w+(?:\([^)]*\))?\s+)*(?:public|open)\s+(?:final\s+|indirect\s+)*"
    r"(?:struct|class|enum|protocol|typealias|actor)\s+(\w+)", re.M)


class ToolError(Exception):
    pass


def run(argv, timeout, cwd=None):
    try:
        return subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise ToolError(f"timed out after {timeout}s: {' '.join(argv)}")
    except OSError as exc:
        raise ToolError(f"cannot run {argv[0]}: {exc}")


def public_top_level_names(path: Path) -> set[str]:
    data = json.loads(path.read_text())
    names = set()
    for sym in data.get("symbols", []):
        if (sym.get("kind", {}).get("identifier") in KINDS
                and sym.get("accessLevel") in PUBLIC_ACCESS
                and len(sym.get("pathComponents", [])) == 1):
            names.add(sym["pathComponents"][0])
    return names


def source_public_names(module_dir: Path) -> set[str]:
    """Lexical scan: unindented `public|open struct|class|enum|protocol|typealias|actor Name` declarations."""
    names: set[str] = set()
    for path in sorted(module_dir.rglob("*.swift")):
        names.update(DECLARATION_RE.findall(path.read_text(encoding="utf-8", errors="replace")))
    return names


def check_source_coverage(module: str, extracted: set[str], source: set[str],
                          allowed=None) -> list[str]:
    """Raise ToolError for source-declared public names missing from the graph and not allowlisted.

    Returns the allowlisted omissions that were actually observed (for stale-entry reporting).
    """
    allowed = SOURCE_OMISSION_ALLOWLIST if allowed is None else allowed
    omitted = sorted(source - extracted)
    unreviewed = [n for n in omitted if (module, n) not in allowed]
    if unreviewed:
        raise ToolError(f"source cross-check failed for {module}: public top-level declaration(s) "
                        f"{unreviewed} in Sources/{module} are missing from the extracted symbol graph; "
                        "the extraction is incomplete or the name is silently filtered "
                        "(review, then add to SOURCE_OMISSION_ALLOWLIST with a reason)")
    return [n for n in omitted if (module, n) in allowed]


def package_names(package_path: Path, scratch: Path) -> dict[str, set[str]]:
    """Dump symbol graphs into `scratch`, which the caller guarantees is fresh."""
    proc = run(["swift", "package", "--package-path", str(package_path),
                "--scratch-path", str(scratch), "dump-symbol-graph",
                "--minimum-access-level", "public"], DUMP_TIMEOUT_S)
    out = proc.stdout + proc.stderr
    result: dict[str, set[str]] = {}
    for module in PACKAGE_MODULES:
        graph = next(iter(sorted(scratch.rglob(f"{module}.symbols.json"))), None)
        if graph is None:
            tail = "\n".join(out.splitlines()[-15:])
            raise ToolError(f"no fresh symbol graph for {module} "
                            f"(dump exit {proc.returncode}):\n{tail}")
        result[module] = public_top_level_names(graph)
        missing = [n for n in PACKAGE_SENTINELS.get(module, ()) if n not in result[module]]
        if missing:
            raise ToolError(f"package extraction self-check failed for {module}: sentinel "
                            f"name(s) {missing} missing from {len(result[module])} extracted names "
                            f"(dump exit {proc.returncode}); the symbol graph is incomplete")
        observed = check_source_coverage(module, result[module],
                                         source_public_names(package_path / "Sources" / module))
        for name in sorted(n for (m, n) in SOURCE_OMISSION_ALLOWLIST if m == module and n not in observed):
            print(f"WARNING: stale source-omission allowlist entry {module}.{name} is no longer omitted; "
                  "remove it", file=sys.stderr)
    if proc.returncode != 0:
        # Other package targets (for example the bench executable) may fail to
        # compile; the fresh library graphs above were still produced by this run.
        print(f"warning: dump-symbol-graph exited {proc.returncode}; "
              "using the graphs it produced in this run", file=sys.stderr)
    return result


def sdk_names(sdk: str, triple: str, outdir: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for module in SDK_MODULES:
        target = outdir / module
        target.mkdir(parents=True, exist_ok=True)
        proc = run(["xcrun", "swift-symbolgraph-extract", "-module-name", module,
                    "-target", triple, "-sdk", sdk, "-output-dir", str(target),
                    "-minimum-access-level", "public"], EXTRACT_TIMEOUT_S)
        graph = target / f"{module}.symbols.json"
        if proc.returncode != 0 or not graph.is_file():
            raise ToolError(f"symbolgraph-extract failed for {module} "
                            f"(exit {proc.returncode}): {proc.stderr[-800:]}")
        result[module] = public_top_level_names(graph)
        missing = [n for n in SENTINELS.get(module, ()) if n not in result[module]]
        if missing:
            raise ToolError(f"extraction self-check failed for {module}: sentinel "
                            f"name(s) {missing} missing from {len(result[module])} extracted names")
    return result


def load_allowlist(path: Path) -> set[tuple[str, str]]:
    entries = json.loads(path.read_text())
    if not isinstance(entries, list):
        raise ToolError(f"{path}: expected a JSON list")
    allowed = set()
    for entry in entries:
        if not (isinstance(entry, dict) and entry.get("name") and entry.get("module")
                and entry.get("reason")):
            raise ToolError(f"{path}: entries need name, module and reason: {entry!r}")
        allowed.add((entry["name"], entry["module"]))
    return allowed


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--package-path", type=Path, default=here.parent.parent)
    parser.add_argument("--scratch-path", type=Path,
                        help="parent for a fresh per-run SwiftPM scratch subdirectory (default: temporary directory)")
    parser.add_argument("--allowlist", type=Path,
                        default=here / "sdk_name_collision_allowlist.json")
    parser.add_argument("--sdk", help="SDK path (default: xcrun --show-sdk-path)")
    args = parser.parse_args(argv)
    try:
        sdk = args.sdk
        if not sdk:
            proc = run(["xcrun", "--show-sdk-path"], QUICK_TIMEOUT_S)
            sdk = proc.stdout.strip()
            if proc.returncode != 0 or not sdk:
                raise ToolError(f"xcrun --show-sdk-path failed (exit {proc.returncode}, "
                                f"output {sdk!r}): {proc.stderr.strip()[-300:]}")
        info = run(["xcrun", "swift", "-print-target-info"], QUICK_TIMEOUT_S)
        triple = json.loads(info.stdout)["target"]["triple"]
        allowed = load_allowlist(args.allowlist)
        with tempfile.TemporaryDirectory(prefix="sdk-names-") as tmp:
            if args.scratch_path:
                args.scratch_path.mkdir(parents=True, exist_ok=True)
                scratch = Path(tempfile.mkdtemp(prefix="run-", dir=args.scratch_path))
            else:
                scratch = Path(tmp) / "scratch"
            ours = package_names(args.package_path, scratch)
            theirs = sdk_names(sdk, triple, Path(tmp) / "sdk")
    except (ToolError, ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"SDK: {sdk} ({triple})")
    for module, names in {**ours, **theirs}.items():
        print(f"  {module}: {len(names)} public top-level type names")
    failures, info_only, used = [], [], set()
    for ours_mod, our_names in ours.items():
        for sdk_mod, sdk_set in theirs.items():
            for name in sorted(our_names & sdk_set):
                row = (name, ours_mod, sdk_mod)
                if sdk_mod == "Swift":
                    info_only.append(row)
                elif (name, sdk_mod) in allowed:
                    used.add((name, sdk_mod))
                    print(f"allowed: {ours_mod}.{name} == {sdk_mod}.{name}")
                else:
                    failures.append(row)
    for name, sdk_mod in sorted(allowed - used):
        print(f"WARNING: stale allowlist entry {name} ({sdk_mod}) no longer collides; "
              "remove it")
    for name, ours_mod, sdk_mod in info_only:
        print(f"info (Swift stdlib, not a failure): {ours_mod}.{name} == Swift.{name}")
    for name, ours_mod, sdk_mod in failures:
        print(f"COLLISION: {ours_mod}.{name} == {sdk_mod}.{name}")
    if failures:
        print(f"{len(failures)} unreviewed collision(s); rename the type or add a "
              "justified allowlist entry.")
        return 1
    print("OK: no unreviewed SDK name collisions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
