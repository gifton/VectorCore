#!/usr/bin/env python3
"""CLI process contracts; pass the actual built vectorcore-bench executable."""
import json
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


def comparator_contracts(helper, root):
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
        for fragment in present: assert fragment in text, (name, flags, fragment, text)
        for fragment in absent: assert fragment not in text, (name, flags, fragment, text)
        return stdout
    # The documented lines are read from the README itself, so the test follows the text.
    lines = [line.strip()[2:].split() for line in README.read_text().splitlines() if line.strip().startswith("- swift Scripts/compare_benchmarks.swift")]
    assert len(lines) == 2, lines
    metrics = []
    for tokens in lines:
        assert tokens[:2] == ["swift", "Scripts/compare_benchmarks.swift"], tokens
        flags = tokens[2:]
        baseline = flags[flags.index("--baseline") + 1]
        assert flags[flags.index("--current") + 1] == ".bench/results.json", flags
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
    assert sorted(metrics) == ["median", "nsPerOp"], metrics
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
    return counter


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
            assert "finite" in result.stderr or "Invalid" in result.stderr
    for fmt in ["json", "csv"]:
        result = run(base + ["--format", fmt, "--out", str(root)], 1)
        assert "Failed to write" in result.stderr
    artifact = root / "finite.json"
    finite = run(base + ["--strict-correctness", "--format", "json", "--out", str(artifact)], 0)
    report = json.loads(artifact.read_text())
    assert len(report["results"]) == 1 and not report["failures"]
    assert report["results"][0]["correctness"]["status"] == "valid"
    assert report["results"][0]["correctness"]["samples"] >= 3
    # PR #56 review BM5: the pretty banner names the compiled configuration; only a release build is unflagged.
    configuration = report["metadata"]["benchmarkConfiguration"]
    assert f"configuration: {configuration}" in finite.stdout, finite.stdout
    assert (f"[WARN {configuration} build" in finite.stdout) == (configuration != "release"), finite.stdout
    # PR #56 review BM8: an explicit suite list other than batch contradicts --ab-only and is rejected by name, before the
    # dimension check that used to blame the batch suite the user never asked for.
    for arguments in [["--suites", "memory", "--ab-only"], ["--suites", "memory", "--dims", "100", "--ab-only"]]:
        result = run([binary, *arguments], 1)
        assert "--ab-only" in result.stderr and "Unsupported dimension" not in result.stderr, result.stderr
    comparisons = comparator_contracts(helper, root)
    # Mechanical family guard: every provider await in BatchBench propagates throws;
    # the shared case boundary is exercised by the Swift process table.
    source = (Path(__file__).resolve().parents[3] / "Benchmarks/VectorCoreBench/Suites/BatchBench.swift").read_text()
    assert "try? await provider" not in source
    provider_calls = source.count("try await provider.parallelReduce")
    assert provider_calls == 28, provider_calls
    assert source.count("return await BatchExecution.run(name: name)") == 14
    assert source.count("observations.observe(observedValue)") == 28
    print(f"PASS: 10 invalid threshold configurations, JSON/CSV write failures, finite strict real benchmark with its configuration banner, "
          f"2 rejected --ab-only suite lists, {comparisons} comparator invocations (README lines, former forms, configuration warnings), "
          f"28 provider calls / 14 case boundaries. Artifacts: {root}")


if __name__ == "__main__":
    main()
