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
    run(base + ["--strict-correctness", "--format", "json", "--out", str(artifact)], 0)
    report = json.loads(artifact.read_text())
    assert len(report["results"]) == 1 and not report["failures"]
    assert report["results"][0]["correctness"]["status"] == "valid"
    assert report["results"][0]["correctness"]["samples"] >= 3
    # Mechanical family guard: every provider await in BatchBench propagates throws;
    # the shared case boundary is exercised by the Swift process table.
    source = (Path(__file__).resolve().parents[3] / "Benchmarks/VectorCoreBench/Suites/BatchBench.swift").read_text()
    assert "try? await provider" not in source
    provider_calls = source.count("try await provider.parallelReduce")
    assert provider_calls == 28, provider_calls
    assert source.count("return await BatchExecution.run(name: name)") == 14
    assert source.count("observations.observe(observedValue)") == 28
    print(f"PASS: 10 invalid threshold configurations, JSON/CSV write failures, finite strict real benchmark, 28 provider calls / 14 case boundaries. Artifacts: {root}")


if __name__ == "__main__":
    main()
