#!/usr/bin/env python3
"""Bounded source/manifest contract checks for the repository's current CI shape."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_TEST_SOURCES = {'BenchmarkContractTests.swift', 'BenchmarkSelectionContractTests.swift',
                          'RunnerContractTests.swift', 'SwiftTestingEntryTests.swift'}


def sanitizer_manifest(root=ROOT):
    data = json.loads((root / 'Scripts/ci/sanitizer_bodies.json').read_text())
    expected = {'address': [], 'thread': []}
    for line in (root / 'Scripts/ci/sanitizer_expected_bodies.txt').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        mode, body = line.split()
        expected[mode].append(body)
    if set(data['sanitizers']) != set(expected):
        raise ValueError('Sanitizer modes must be address and thread')
    for mode, rows in data['sanitizers'].items():
        if not rows or Counter(r['id'] for r in rows) != Counter(expected[mode]) or len(set(expected[mode])) != len(expected[mode]):
            raise ValueError('Missing/duplicate required instrumented body: ' + mode)
        for row in rows:
            suite, method = row['id'].split('/')
            exact = '^ComprehensiveTests\\.' + suite + '/' + method + r'\(\)'
            if row['filter'] != exact:
                raise ValueError('Missing or inexact sanitizer filter: ' + row['id'])
            text = (root / row['source']).read_text()
            if not re.search(r'@Test(?:\([^\n]*\))?\s*func\s+' + re.escape(method) + r'\(', text):
                raise ValueError('Missing actual sanitizer test declaration: ' + row['id'])
            q03 = row['id'] == 'FP16DiagnosticsBoundaryTests/isolatedConcurrency'
            if row['deadlineSeconds'] != (49 if q03 else 15):
                raise ValueError('Unexpected execution deadline: ' + row['id'])
    return data


def coverage_text(data):
    counts = {mode: len(rows) for mode, rows in data['sanitizers'].items()}
    return """# Executed CI contracts

This table describes configured remote commands. Local verification is recorded separately in V03 evidence; YAML and policy success do not establish a hosted run or branch protection.

| Surface | Target / command | Activation and coverage |
| --- | --- | --- |
| Ordinary tests | `swift test -c "$CONFIGURATION"` | Debug/Release, Xcode 16.2/26.3; extended=0 and strict=0. Guarded child entry functions alone are not child coverage. |
| Performance opt-out | `python3 Scripts/ci/check_performance_gates.py --configuration "$CONFIGURATION" --scratch-root DIR --output DIR --jobs 3` | Fresh environments; lazy Swift Testing/XCTest failures, live numeric controls, one-shot observations and named real extended tests. |
| Benchmark contracts | `swift build -c release --product vectorcore-bench`, actual `--show-bin-path`, then `python3 Scripts/ci/tests/test_benchmark_contracts.py BINARY` | Actual built CLI invalid-input/output/strict-correctness cases. Python discovery is separate and is not CLI execution. |
| C modes and routes | `python3 Scripts/ci/check_benchmark_build_modes.py --scratch-root DIR --output DIR --jobs 3` | Four distinct Debug/Release × optional-C off/on products; compiled `-DVC_USE_C_KERNELS`, actual public forwarding counts and `V02_BUILD_MODES_COMPLETE`. |
| Address sanitizer | `python3 Scripts/ci/check_sanitizers.py --sanitizer address --scratch-root DIR --output DIR --jobs 3` | COUNT_ADDRESS pinned valid bodies; O05 borrows/ownership, existing selected suites and E02 cancellation/reductions. Leak detection is disabled. |
| Thread sanitizer | Same shared driver with `--sanitizer thread` | COUNT_THREAD entries; Q03 parent requires its actual instrumented child, six × 2000 loops and all result/join markers. Parent deadline 49s, child 45s, cleanup ≤2s; other executions 15s. |
| Repository policy | `python3 -B -m unittest discover -s Scripts/ci/tests -v`; `ruby Scripts/ci/validate_github.rb` | Exact seven-job aggregate, semantic commands, independent sanitizer body set, maintained timing sites, target membership and mutation controls. Narrow lexical/workflow checks, not arbitrary Swift/shell dataflow proof. |
| Lint | Pinned SwiftLint 0.65.1 `lint --config .swiftlint.yml` | Sources, Tests and Benchmarks; no `analyze`, unused-declaration or unused-import analysis claim. |
| Documentation consumers | `python3 Scripts/ci/consumer_smoke.py` | Build and execute extracted supported public examples; compile the nine must-not-compile contract fixtures and their positive controls. |
| Apple platforms | `xcodebuild` library destinations; Intel `swift build --build-tests --triple x86_64-apple-macosx14.0` | Compilation only for these destinations; no device or downstream runtime claim. |
| CodeQL | Separate scheduled/PR CodeQL workflow | Library and extracted dependencies; outside CI Required, not all tests/benchmarks/every file. |
| Perf smoke / full benchmarks | Separate reporting jobs | Outside CI Required; executable failure still fails their own jobs. No speed target. |

CI Required has exactly `test`, `lint`, `repository-checks`, `consumer`, `platforms`, `benchmark-contracts`, `sanitizers`; `always()` runs the exact-result aggregate. Missing, skipped, cancelled and failed results reject.

BenchmarkContractTests was already registered before V03. Its default source membership remains `BenchmarkContractTests.swift`, `BenchmarkSelectionContractTests.swift`, `RunnerContractTests.swift`, `SwiftTestingEntryTests.swift`, with dependencies VectorCoreBenchSupport and VectorCoreBenchmarking. The policy rejects target/source exclusions.

`Scripts/ci/performance_sites.json` inventories 71 existing timing assertions and two intentional gate probes, plus 29 existing measurement bodies and one observation probe. Strict is enabled only by the exact string `1`; disabled lazy predicates do not evaluate. Fixed injected tuning metrics, numerical assertions, allocation counts and liveness deadlines remain active. The unregistered obsolete NSW benchmark source is retired, not registered.

`Scripts/ci/sanitizer_bodies.json` is the exact runtime body manifest; `sanitizer_expected_bodies.txt` is its independently maintained required set. Completion requires actual body events/output and host/bundle/runtime identity, with Q03 child evidence retained separately. This is selected-path coverage, not whole-suite sanitizer coverage.
""".replace('COUNT_ADDRESS', str(counts['address'])).replace('COUNT_THREAD', str(counts['thread']))


def validate(root=ROOT):
    data = sanitizer_manifest(root)
    if (root / 'Docs/executed-ci-contracts.md').read_text() != coverage_text(data):
        raise ValueError('Executable coverage table differs from checked manifest')
    package = (root / 'Package.swift').read_text()
    registration = re.findall(r'\.testTarget\(\s*name:\s*"BenchmarkContractTests"\s*,\s*dependencies:\s*\["VectorCoreBenchSupport",\s*"VectorCoreBenchmarking"\]\s*\)', package)
    if len(registration) != 1:
        raise ValueError('Preserve default BenchmarkContractTests target/dependencies without source exclusion')
    actual = {p.name for p in (root / 'Tests/BenchmarkContractTests').glob('*.swift')}
    if actual != BENCHMARK_TEST_SOURCES:
        raise ValueError('Benchmark test source membership changed; update reviewed contract')
    for relative in ['.swiftlint.yml', 'Sources/.swiftlint.yml']:
        text = (root / relative).read_text()
        if re.search(r'(?m)^analyzer_rules:', text):
            raise ValueError('Unexecuted analyzer rules: ' + relative)
        include = re.search(r'(?m)^included:\s*\n((?:[ \t]+.*\n)+)', text)
        if not include or not all(re.search(r'(?m)^\s*-\s*'+name+r'\s*$', include.group(1)) for name in ['Sources', 'Tests', 'Benchmarks']):
            raise ValueError('Lint must include Sources, Tests, Benchmarks')
    return {'sanitizerBodies': {mode: len(rows) for mode, rows in data['sanitizers'].items()},
            'benchmarkTestSources': sorted(actual)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    print(json.dumps(validate(parser.parse_args().root), sort_keys=True))
