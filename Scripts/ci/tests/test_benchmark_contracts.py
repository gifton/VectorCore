#!/usr/bin/env python3
"""CLI process contracts; pass the actual built vectorcore-bench executable."""
import json
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import argparse
import importlib.util
from types import SimpleNamespace
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
README = ROOT / "Benchmarks/baselines/README.md"


def load_reporting():
    """The import-safe fixture builders of test_benchmark_reporting.py (a schema-3 report with one valid case)."""
    spec = importlib.util.spec_from_file_location("v02_reporting", Path(__file__).resolve().with_name("test_benchmark_reporting.py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def comparator_contracts(helper, root, product_reports=()):
    """PR #56 review SW4-5 and BM5: the baselines README's two comparator lines run against the stored files (each file as
    its own current run), the two former forms exit 2, and the comparator observes both reports' build configurations and
    prints a WARN line for every side that is not a release build."""
    reporting = load_reporting()
    folder = root / "comparator"; folder.mkdir()
    comparator = folder / "compare-benchmarks"
    helper.run(["swiftc", "-module-cache-path", folder / "module-cache", ROOT / "Scripts/compare_benchmarks.swift", "-o", comparator],
               folder, "compile", timeout=300)
    counter = 0
    def compare(flags, expected, present=(), absent=()):
        nonlocal counter
        counter += 1
        name = f"compare-{counter}"
        stdout = helper.run([comparator, *flags], folder, name, expected=expected)
        text = stdout + (folder / (name + ".stderr")).read_text()
        for fragment in present:
            if fragment not in text:
                raise AssertionError((name, flags, fragment, text))
        for fragment in absent:
            if fragment in text:
                raise AssertionError((name, flags, fragment, text))
        return stdout
    # The documented lines are read from the README itself, so the test follows the text.
    lines = [line.strip()[2:].split() for line in README.read_text().splitlines() if line.strip().startswith("- swift Scripts/compare_benchmarks.swift")]
    if not (len(lines) == 2):
        raise AssertionError(lines)
    metrics = []
    for tokens in lines:
        if not (tokens[:2] == ["swift", "Scripts/compare_benchmarks.swift"]):
            raise AssertionError(tokens)
        flags = tokens[2:]
        baseline = flags[flags.index("--baseline") + 1]
        if not (flags[flags.index("--current") + 1] == ".bench/results.json"):
            raise AssertionError(flags)
        flags[flags.index("--current") + 1] = baseline
        metric = flags[flags.index("--metric") + 1]; metrics.append(metric)
        results = json.loads((ROOT / baseline).read_text())["results"]
        compare(flags, 0, present=(f"Compared {len(results)} cases; metric={metric}; per-unit=True".replace("True", "true"),
                                   "Observation coreConfiguration: baseline=unavailable, current=unavailable"))
        # The former README line lacked --allow-incompatible: the stored files' legacy metadata is refused.
        compare([flag for flag in flags if flag != "--allow-incompatible"], 2, present=("Incompatible benchmark metadata",))
        if metric == "nsPerOp":
            # The former line also asked for the median, which the local file does not carry; the row reader refuses first.
            compare([("median" if flag == "nsPerOp" else flag) for flag in flags], 2,
                    present=("Invalid or missing numeric field", "medianNsPerOp"))
    if not (sorted(metrics) == ["median", "nsPerOp"]):
        raise AssertionError(metrics)
    def report(core="release", bench="release", legacy=None):
        value = reporting.run_report()
        if legacy is not None: value["metadata"] = {"buildConfiguration": legacy}
        else: value["metadata"].update(coreConfiguration=core, benchmarkConfiguration=bench)
        return value
    def pair(name, before, after):
        a, b = folder / (name + "-before.json"), folder / (name + "-after.json")
        helper.write_json(a, before); helper.write_json(b, after)
        return ["--baseline", a, "--current", b]
    compare(pair("release", report(), report()), 0, absent=("WARN",),
            present=("Observation coreConfiguration: baseline=release, current=release",
                     "Observation benchmarkConfiguration: baseline=release, current=release"))
    compare(pair("debug", report("debug", "debug"), report("debug", "debug")), 0,
            present=("WARN baseline.coreConfiguration=debug: timings are not release numbers", "WARN current.benchmarkConfiguration=debug",
                     "Observation coreConfiguration: baseline=debug, current=debug"))
    # Mixed configurations are incompatible; the warning for the debug side is printed before the refusal.
    compare(pair("mixed", report(), report("debug", "debug")), 2, absent=("WARN baseline",),
            present=("Metadata mismatch: coreConfiguration: baseline=release, current=debug", "WARN current.coreConfiguration=debug",
                     "Incompatible benchmark metadata"))
    # Legacy (schema 2) reports carry buildConfiguration only; it is flagged the same way.
    compare(pair("legacy", report(legacy="debug"), report(legacy="debug")) + ["--allow-incompatible"], 0,
            present=("WARN baseline.buildConfiguration=debug", "WARN current.buildConfiguration=debug"))
    if product_reports:
        actual_before, actual_after = [json.loads(path.read_text()) for path in product_reports]
        for actual in (actual_before, actual_after):
            if not (actual["metadata"]["schemaVersion"] == 3 and len(actual["results"]) == 2 and not actual["failures"]
                    and all(row["correctness"]["status"] == "valid" for row in actual["results"])):
                raise AssertionError(actual)
        # Self-comparison proves a successful current-schema execution without host timing assumptions.
        compare(["--baseline", product_reports[0], "--current", product_reports[0], "--per-unit"], 0,
                present=("Compared 2 cases; metric=median; per-unit=true; regressions=0",))
        # A distinct actual run may regress on a loaded host; retain that exit, never accept it as performance evidence.
        compare(["--baseline", product_reports[0], "--current", product_reports[1], "--per-unit"], (0, 1),
                present=("Compared 2 cases; metric=median; per-unit=true;",))
        for side in (0, 1):
            for label, mutate, expected, fragment in [
                ("identity", lambda r: r["metadata"].__setitem__("arch", "incompatible-control"), 2, "Metadata mismatch: arch"),
                ("missing", lambda r: r["results"].pop(), 1, "Missing in"),
                ("numeric", lambda r: r["results"][0].__setitem__("medianNsPerOp", -1), 2, "Invalid or missing numeric field"),
                ("failure", lambda r: r["failures"].append({"name": "control", "reason": "injected failure"}), 2, "Execution/correctness failures"),
            ]:
                # Start from one report on both sides: missing-case exit 1 cannot be a timing regression.
                reports = [copy.deepcopy(actual_before), copy.deepcopy(actual_before)]
                mutate(reports[side])
                compare(pair(f"product-{label}-{side}", *reports), expected, present=(fragment,))
    return counter


def reporting_contracts(helper, root):
    """Execute the standalone reporting table; import-safe unittest discovery alone does not run it."""
    output = root / "reporting"
    text = helper.run([sys.executable, "-B", Path(__file__).with_name("test_benchmark_reporting.py"), "--output", output],
                      root, "reporting-contracts", timeout=300)
    if "V02_REPORTING_COMPLETE" not in text:
        raise AssertionError("Reporting driver did not complete: " + text)
    result = json.loads((output / "result.json").read_text())
    if not (result["marker"] == "V02_REPORTING_COMPLETE" and len(result["cases"]) > 1):
        raise AssertionError(result)


def repaired_observation_contracts(helper, root, binary):
    """Every generic/optimized width and copy fixture must produce independent strict evidence."""
    path = root / "repaired-family-report.json"
    helper.run([binary, "--suites", "dot,normalize,memory", "--dims", "512,768,1536", "--no-ab",
                "--filter", "dot.*,normalize.*,mem.copy.*", "--repeats", "1", "--samples", "2",
                "--strict-correctness", "--format", "json", "--out", path], root, "repaired-families", timeout=30)
    report = json.loads(path.read_text())
    expected = {f"dot.{dim}.{variant}": 1 for dim in [512, 768, 1536] for variant in ["generic", "optimized"]}
    expected.update({f"normalize.{outcome}.{dim}.{variant}": dim if outcome == "success" else 1
                     for dim in [512, 768, 1536] for outcome in ["success", "zeroFail"] for variant in ["generic", "optimized"]})
    expected.update({f"mem.copy.{dim}": dim * 1024 for dim in [512, 768, 1536]})
    rows = report["results"]
    if report["failures"] or len(rows) != len(expected) or {row["name"] for row in rows} != set(expected):
        raise AssertionError(report)
    for row in rows:
        observation = row["correctness"]
        if not (observation["status"] == "valid" and observation["samples"] == expected[row["name"]]
                and row["boundary"]["inclusions"]["correctnessValidation"] == "excluded"
                and len(row["sampleNsPerInvocation"]) == 2 and len(row["sampleIterations"]) == 2):
            raise AssertionError(row)
        if row["name"].startswith("mem.copy.") and observation["maxAbsError"] != 0:
            raise AssertionError(row)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    binary = str(args.binary.resolve())
    root = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="V01-cli-contracts-"))
    root.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("v02_capture", Path(__file__).resolve().parents[1] / "check_benchmark_build_modes.py")
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    counter = 0
    env = {k: v for k, v in os.environ.items() if not k.startswith("VC_")}
    base = [binary, "--suites", "batch", "--dims", "512", "--batch-ns", "2", "--repeats", "1", "--no-ab", "--filter", "batch.euclidean2.512.N2.generic.sequential"]

    def run(args, expected):
        nonlocal counter
        counter += 1
        name = "case-" + str(counter)
        stdout = helper.run(args, root, name, expected=expected)
        stderr = (root / (name + ".stderr")).read_text()
        return SimpleNamespace(returncode=expected, stdout=stdout, stderr=stderr)

    for flag in ["--max-rel-error", "--max-abs-error"]:
        for value in ["-1", "nan", "inf", "-inf", "not-a-number"]:
            result = run(base + [flag, value], 1)
            if not ("finite" in result.stderr or "Invalid" in result.stderr):
                raise AssertionError
    for fmt in ["json", "csv"]:
        result = run(base + ["--format", fmt, "--out", str(root)], 1)
        if not ("Failed to write" in result.stderr):
            raise AssertionError
    artifact = root / "finite.json"
    finite = run(base + ["--strict-correctness", "--format", "json", "--out", str(artifact)], 0)
    report = json.loads(artifact.read_text())
    if not (len(report["results"]) == 1 and not report["failures"]):
        raise AssertionError
    if not (report["results"][0]["correctness"]["status"] == "valid"):
        raise AssertionError
    if not (report["results"][0]["correctness"]["samples"] >= 3):
        raise AssertionError
    # PR #56 review BM5: the pretty banner names the compiled configuration; only a release build is unflagged.
    configuration = report["metadata"]["benchmarkConfiguration"]
    if not (f"configuration: {configuration}" in finite.stdout):
        raise AssertionError(finite.stdout)
    if not ((f"[WARN {configuration} build" in finite.stdout) == (configuration != "release")):
        raise AssertionError(finite.stdout)
    # PR #56 review BM8: an explicit suite list other than batch contradicts --ab-only and is rejected by name, before the
    # dimension check that used to blame the batch suite the user never asked for.
    for arguments in [["--suites", "memory", "--ab-only"], ["--suites", "memory", "--dims", "100", "--ab-only"]]:
        result = run([binary, *arguments], 1)
        if not ("--ab-only" in result.stderr and "Unsupported dimension" not in result.stderr):
            raise AssertionError(result.stderr)
    product_command = list(base)
    product_command[-1] = "batch.euclidean2.512.N2.generic.sequential,batch.euclidean2.512.N2.optimized.sequential"
    product_reports = [root / ("product-" + side + ".json") for side in ("before", "after")]
    for path in product_reports:
        run(product_command + ["--samples", "2", "--strict-correctness", "--format", "json", "--out", str(path)], 0)
    repaired = repaired_observation_contracts(helper, root, binary)
    comparisons = comparator_contracts(helper, root, product_reports)
    reporting_contracts(helper, root)
    # Mechanical family guard: every provider await in BatchBench propagates throws;
    # the shared case boundary is exercised by the Swift process table.
    source = (Path(__file__).resolve().parents[3] / "Benchmarks/VectorCoreBench/Suites/BatchBench.swift").read_text()
    if "try? await provider" in source:
        raise AssertionError
    provider_calls = source.count("try await provider.parallelReduce")
    if not (provider_calls == 28):
        raise AssertionError(provider_calls)
    if not (source.count("return await BatchExecution.run(name: name)") == 14):
        raise AssertionError
    if not (source.count("observations.observe(observedValue)") == 28):
        raise AssertionError
    print(f"PASS: 10 invalid threshold configurations, JSON/CSV write failures, finite strict real benchmark with its configuration banner, "
          f"2 rejected --ab-only suite lists, {comparisons} comparator invocations (README, configuration, actual-product controls), "
          f"{repaired} repaired-family strict observations, executed current-schema reporting table, "
          f"28 provider calls / 14 case boundaries. Artifacts: {root}")


if __name__ == "__main__":
    main()
