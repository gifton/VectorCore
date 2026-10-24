#!/usr/bin/env python3
"""Actual fresh-process strict-gate, numeric-control and observation contracts."""
import argparse
from pathlib import Path
import re
import shutil

from swiftpm_products import (BUILD_SYSTEMS, build_system_flags, check_no_rebuild, find_swift,
                              find_test_bundles, in_log, resolve_bin_path, swift_testing_runs, xctest_executed)
from v03_process import retain_artifacts, run, sha, test_environment, write_json

ROOT = Path(__file__).resolve().parents[2]


def representative_executed(log):
    """True when XCTest or Swift Testing reports at least one executed test.

    `swift test` runs both frameworks, so an XCTest representative is followed by a Swift Testing
    summary of 0 tests (and the reverse). Swift 6.0 renders that as `Test run with 0 tests passed`,
    which the former `'0 tests passed' in log` check misread as "not executed" on Xcode 16.2.
    Pinned by test_test_output_counts.RepresentativeExecuted.
    """
    return xctest_executed(log) >= 1 or any(count >= 1 for count, _ in swift_testing_runs(log))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration', choices=['debug', 'release'], required=True)
    parser.add_argument('--scratch-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--swift-build-system', choices=BUILD_SYSTEMS, default=None,
                        help='Pass --build-system to every swift invocation; default is the toolchain default.')
    args = parser.parse_args()
    if not 1 <= args.jobs <= 6:
        raise ValueError('jobs must be in 1...6')
    scratch = args.scratch_root.resolve() / args.configuration
    output = args.output.resolve()
    if scratch.exists() or output.exists():
        raise ValueError('Use fresh gate scratch/output directories')
    output.mkdir(parents=True)
    env = test_environment()
    # One flag list serves build, bin-path and `swift test --skip-build`, so the test runs reuse the built products.
    # `swift build` (unlike `swift test`) does not enable testability in Release (T01); @testable import needs it.
    flags = ['-c', args.configuration, '--jobs', args.jobs, '--scratch-path', scratch,
             *build_system_flags(args.swift_build_system)]
    if args.configuration == 'release':
        flags += ['-Xswiftc', '-enable-testing']
    swift_path = find_swift(output, cwd=ROOT, environment=env)
    run([swift_path, 'build', '--build-tests', *flags, '--verbose'], output, 'build', cwd=ROOT, environment=env, timeout=1200)
    bin_path = resolve_bin_path(swift_path, flags, output, 'bin-path', cwd=ROOT, environment=env)
    write_json(output / 'artifacts.json', retain_artifacts(scratch, output / 'artifacts'))
    toolchain = Path(swift_path).resolve().parents[1]
    helper = toolchain / 'libexec/swift/pm/swiftpm-testing-helper'
    # Observed on Swift 6.4 / Xcode 27: dyld loaded <platform>/Developer/Library/Xcode/Agents/xctest while
    # `xcrun --find xctest` named Developer/usr/bin/xctest, so the former is the host checked in the dyld log
    # (v5/failed-attempt2: the shim path was absent from xctest-00 logs; v5/gates-debug: Agents/xctest present).
    _, platform_path = run(['xcrun', '--sdk', 'macosx', '--show-sdk-platform-path'], output, 'find-platform',
                           cwd=ROOT, environment=env)
    xctest = Path(platform_path.strip()) / 'Developer/Library/Xcode/Agents/xctest'
    if not xctest.is_file():
        raise ValueError('XCTest host is unavailable: ' + str(xctest))
    hosts = [helper, xctest]
    for host in hosts:
        shutil.copy2(host, output / host.name)
    bundles = find_test_bundles(bin_path, ['ComprehensiveTests', 'MinimalTests'])
    env['DYLD_PRINT_LIBRARIES'] = '1'
    # Observed on Swift 6.4 / Xcode 27 (v5 evidence, 2026-10-03): the same `swift test` with DYLD_PRINT_LIBRARIES=1
    # printed 0 dyld lines via /usr/bin/swift and 6045 via the xcrun-resolved toolchain swift. Unverified on
    # other toolchains; all invocations here use the xcrun-resolved binary.
    base = [swift_path, 'test', '--skip-build', *flags]
    records = []
    for framework, selector in [('swift', r'^ComprehensiveTests\.PerformanceGateContractTests/isolatedProbe\(\)'),
                                ('xctest', r'^ComprehensiveTests\.XCTestPerformanceGateContractTests/testIsolatedProbe$')]:
        for index, (strict, mode, expected, evaluated) in enumerate([
            (None, 'false', 0, False), ('0', 'false', 0, False), ('', 'false', 0, False),
            ('true', 'false', 0, False), ('01', 'false', 0, False), ('1', 'false', 1, True),
            ('1', 'true', 0, True), ('0', 'numeric', 1, False)]):
            child_env = dict(env, VECTORCORE_TEST_EXTENDED='1', V03_GATE_CASE=mode)
            if strict is None:
                child_env.pop('VECTORCORE_STRICT_PERF', None)
            else:
                child_env['VECTORCORE_STRICT_PERF'] = strict
            name = f'{framework}-{index:02d}'
            command, log = run(base + ['--filter', selector], output, name, cwd=ROOT, environment=child_env, expected=(expected,))
            if f'V03_GATE_BODY {framework} {mode} strict={str(strict == "1").lower()}' not in log or f'V03_GATE_COMPLETE {framework}' not in log:
                raise ValueError('Missing actual gate body markers: ' + name)
            if ('V03_PERFORMANCE_PREDICATE_EVALUATED' in log) != evaluated:
                raise ValueError('Incorrect lazy predicate evaluation: ' + name)
            if expected and ('V03 numerical control' if mode == 'numeric' else 'V03.probe.' + framework) not in log:
                raise ValueError('Expected failure was not the controlled assertion: ' + name)
            if not in_log(bundles['ComprehensiveTests'], log) or str(hosts[0 if framework == 'swift' else 1]) not in log:
                raise ValueError('Actual host/loaded-bundle image evidence missing')
            check_no_rebuild(log, name)
            records.append({'name': name, 'expectedExit': expected, 'predicateEvaluated': evaluated,
                            'command': name + '.json', 'kind': 'intentional failure control' if expected else 'success control'})
    for strict, mode in [('0', 'valid'), ('0', 'numeric'), ('1', 'valid')]:
        name = 'observation-' + strict + '-' + mode
        expected = 1 if mode == 'numeric' else 0
        command, log = run(base + ['--filter', r'^MinimalTests\.PerformanceObservationContractTests/testIsolatedObservation$'],
                           output, name, cwd=ROOT, environment=dict(env, VECTORCORE_TEST_EXTENDED='1',
                           VECTORCORE_STRICT_PERF=strict, V03_OBSERVATION_CASE=mode), expected=(expected,))
        if not in_log(bundles['MinimalTests'], log):
            raise ValueError('Observation loaded-bundle image evidence missing: ' + name)
        check_no_rebuild(log, name)
        counts = re.findall(r'V03_OBSERVATION_COMPLETE bodies=(\d+)', log)
        if len(counts) != 1 or (int(counts[0]) != 1 if strict == '0' else int(counts[0]) <= 1):
            raise ValueError('Observation body count mismatch: ' + name)
        if strict == '0' and 'measured [Time' in log:
            raise ValueError('Disabled observation used XCTest.measure')
        if strict == '1' and 'measured [Time' not in log:
            raise ValueError('Enabled observation did not use XCTest.measure')
        if mode == 'numeric' and 'V03 body numerical control' not in log:
            raise ValueError('Missing observation numerical failure control')
        records.append({'name': name, 'expectedExit': expected, 'actualBodyCount': int(counts[0]), 'command': name + '.json'})
    # These are actual formerly ungated tests, separate from deterministic gate fixtures.
    representatives = [('normal-simd', 'MixedPrecisionKernelTests.*testSIMDUtilization'),
                       ('extended-latency', 'MixedPrecisionPerformanceBenchmark.*benchmarkEuclideanLatency512'),
                       ('measured-profile', 'KernelAutoTunerTests/testProfileWorkload512')]
    for name, selector in representatives:
        _, log = run(base + ['--filter', selector], output, name, cwd=ROOT,
                     environment=dict(env, VECTORCORE_TEST_EXTENDED='1', VECTORCORE_STRICT_PERF='0'))
        if not representative_executed(log):
            raise ValueError('Representative test did not execute: ' + name)
        if not in_log(bundles['ComprehensiveTests'], log):
            raise ValueError('Representative loaded-bundle image evidence missing: ' + name)
        check_no_rebuild(log, name)
        records.append({'name': name, 'command': name + '.json', 'kind': 'actual representative test'})
    write_json(output / 'result.json', {'configuration': args.configuration, 'cases': records,
               'hosts': [{'path': str(h), 'sha256': sha(h)} for h in hosts],
               'buildSystem': args.swift_build_system or 'toolchain default', 'binPath': str(bin_path),
               'loadedBundles': {t: {'path': str(b), 'sha256': sha(b)} for t, b in bundles.items()},
               'scope': 'Controlled gates and named representative tests, not every extended test.'})
    print('V03_PERFORMANCE_GATES_COMPLETE', args.configuration, len(records))


if __name__ == '__main__':
    main()
