#!/usr/bin/env python3
"""Compare the paired raw runs in results/ (Python standard library only)."""
import csv
from pathlib import Path
from statistics import median

root = Path(__file__).parent / "results"
data = {}
for revision in ("baseline", "final"):
    cases = {}
    for run in (1, 2):
        lines = (root / f"{revision}-{run}.csv").read_text().splitlines()
        assert lines[0] == "downstream_inlinable_smoke,passed"
        for row in csv.DictReader(lines[2:-1]):
            key = (row["dataset"], row["api"], int(row["k"]))
            entry = cases.setdefault(key, {"samples": [], "checksum": row["checksum"]})
            assert entry["checksum"] == row["checksum"]
            entry["samples"].extend(map(int, row["sample_ns"].split(";")))
    data[revision] = cases

assert data["baseline"].keys() == data["final"].keys()
print("| Data | API | k | Baseline ms | Final ms | Delta |")
print("|---|---|---:|---:|---:|---:|")
for key, baseline in data["baseline"].items():
    final = data["final"][key]
    assert baseline["checksum"] == final["checksum"], key
    b, f = median(baseline["samples"]), median(final["samples"])
    print(f"| {key[0]} | {key[1]} | {key[2]:,} | {b / 1e6:.4f} | {f / 1e6:.4f} | {(f / b - 1) * 100:+.1f}% |")
for revision, cases in data.items():
    for dataset in ("mixed", "duplicates"):
        for k in (10, 20_000):
            assert cases[(dataset, "array", k)]["checksum"] == cases[(dataset, "pointer", k)]["checksum"]
print("\nAll eight baseline/final checksums and all array/pointer checksums agree.")
