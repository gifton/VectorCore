#!/usr/bin/env python3
"""Build instrumented tests and execute each pinned valid body in a bounded process.

The Q03 parent has the specifically approved outer budget for its own 45-second
instrumented child; all other test executions have a 15-second deadline.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil

from check_ci_contracts import sanitizer_manifest
from swiftpm_products import (BUILD_SYSTEMS, build_system_flags, check_no_rebuild, find_swift,
                              find_test_bundles, in_log, resolve_bin_path, target_of_filter)
from v03_process import retain_artifacts, run, sha, test_environment, write_json

ROOT = Path(__file__).resolve().parents[2]
Q03_MARKERS = ['Q03_BODY_BEGIN concurrency', 'Q03_WORKERS_JOINED completedLoops=[2000, 2000, 2000, 2000, 2000, 2000]',
               'Q03_STABLE_TOTALS_JOINED', 'Q03_ATOMIC_BATCHES_JOINED', 'Q03_RESULT concurrency', 'Q03_BODY_END concurrency']


def verify_body(row, log):
    label = row['displayName']
    rendered = label if label.endswith('()') else '"' + label + '"'
    # Newer Swift Testing appends the function identity to custom display names: Test "name" (aka 'fn()') started.
    method = row['id'].split('/')[-1] + '()'
    aka = '' if label.endswith('()') else r"(?: \(aka '" + re.escape(method) + r"'\))?"
    body = 'Test ' + re.escape(rendered) + aka
    if not re.search(body + ' started\\.', log) or not re.search(body + ' passed', log):
        raise ValueError('Missing actual started/passed body: ' + row['id'])
    if not re.search(r'Test run with 1 test in \d+ suites? passed', log) or ' skipped' in log:
        raise ValueError('Expected one executed, unskipped body: ' + row['id'])


def verify_q03(directory, bundle_hash, host_hash):
    records = list(directory.glob('Q03-child-concurrency-*.json'))
    if len(records) != 1:
        raise ValueError('Expected exactly one actual Q03 child capture')
    data = json.loads(records[0].read_text())
    if data.get('timeout') or data.get('exit') != 0 or not data.get('cleaned'):
        raise ValueError('Q03 child did not complete successfully')
    if data['loadedBundleSHA256'] != bundle_hash or data['hostSHA256'] != host_hash:
        raise ValueError('Q03 child host/bundle identity mismatch')
    if data['executionDeadlineSeconds'] != 45 or data['cleanupDeadlineSeconds'] != 2:
        raise ValueError('Q03 child deadline changed')
    if data['environment'].get('Q03_TSAN') != '1' or data['environment'].get('Q03_UNSAFE_BASELINE') == '1':
        raise ValueError('Q03 fixed TSan mode missing')
    if 'libclang_rt.tsan' not in data['environment'].get('DYLD_INSERT_LIBRARIES', ''):
        raise ValueError('Q03 actual TSan runtime missing')
    log = Path(data['log']).read_text()
    if sha(data['log']) != data['logSHA256'] or not all(marker in log for marker in Q03_MARKERS):
        raise ValueError('Q03 actual concurrent workload evidence missing')
    return {'record': str(records[0]), 'sha256': sha(records[0]), 'markers': Q03_MARKERS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sanitizer', choices=['address', 'thread'], required=True)
    parser.add_argument('--scratch-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--swift-build-system', choices=BUILD_SYSTEMS, default=None,
                        help='Pass --build-system to every swift invocation; default is the toolchain default.')
    args = parser.parse_args()
    if not 1 <= args.jobs <= 6:
        raise ValueError('jobs must be in 1...6')
    rows = sanitizer_manifest(ROOT)['sanitizers'][args.sanitizer]
    scratch = args.scratch_root.resolve() / args.sanitizer
    output = args.output.resolve()
    if scratch.exists() or output.exists():
        raise ValueError('Use fresh sanitizer scratch/output directories')
    output.mkdir(parents=True)
    env = test_environment()
    env['ASAN_OPTIONS'] = 'detect_leaks=0:halt_on_error=1'
    env['TSAN_OPTIONS'] = 'halt_on_error=1'
    flags = ['--jobs', args.jobs, '--scratch-path', scratch, '--sanitize', args.sanitizer,
             *build_system_flags(args.swift_build_system)]
    swift_path = find_swift(output, cwd=ROOT, environment=env)
    run([swift_path, 'build', '--build-tests', *flags, '--verbose'], output, 'build', cwd=ROOT, environment=env, timeout=1200)
    bin_path = resolve_bin_path(swift_path, flags, output, 'bin-path', cwd=ROOT, environment=env)
    targets = {row['id']: target_of_filter(row['filter']) for row in rows}
    bundles = find_test_bundles(bin_path, targets.values())
    artifacts = retain_artifacts(scratch, output / 'artifacts')
    write_json(output / 'artifacts.json', artifacts)
    swift = Path(swift_path).resolve()
    # Observed on Swift 6.4 / Xcode 27 (v5 evidence): 0 dyld lines via /usr/bin/swift vs 6045 via the xcrun-resolved
    # swift with DYLD_PRINT_LIBRARIES=1; unverified elsewhere. All invocations use the xcrun-resolved binary.
    host = swift.parents[1] / 'libexec/swift/pm/swiftpm-testing-helper'
    if not host.is_file():
        raise ValueError('SwiftPM test host is unavailable')
    shutil.copy2(host, output / host.name)
    env['DYLD_PRINT_LIBRARIES'] = '1'
    if args.sanitizer == 'thread':
        env['Q03_TSAN'] = '1'
        env['Q03_CHILD_LOG_DIR'] = str(output)
    completed = []
    for index, row in enumerate(rows):
        bundle = bundles[targets[row['id']]]
        _, log = run([swift_path, 'test', '--skip-build', '--disable-xctest', *flags, '--filter', row['filter'], '--verbose'],
                     output, f'body-{index:03d}', cwd=ROOT, environment=env, timeout=row['deadlineSeconds'])
        verify_body(row, log)
        check_no_rebuild(log, row['id'])
        # dyld reports actual loaded image paths; these are separate from the swift launcher.
        if str(host) not in log or not in_log(bundle, log):
            raise ValueError('Actual SwiftPM host/loaded bundle image evidence missing')
        runtime = 'libclang_rt.' + ('asan' if args.sanitizer == 'address' else 'tsan')
        if runtime not in log:
            raise ValueError('Actual sanitizer image missing')
        completed.append({'id': row['id'], 'target': targets[row['id']], 'command': f'body-{index:03d}.json', 'kind': row['kind']})
    # The Q03 parent runs in its row's bundle; the child must report that bundle's identity.
    q03_targets = {targets[row['id']] for row in rows if row['kind'] == 'isolated parent and child'}
    if args.sanitizer == 'thread' and len(q03_targets) != 1:
        raise ValueError('Expected exactly one isolated parent/child row')
    q03 = verify_q03(output, sha(bundles[q03_targets.pop()]), sha(host)) if args.sanitizer == 'thread' else None
    write_json(output / 'result.json', {'sanitizer': args.sanitizer, 'completed': completed,
               'manifestSHA256': sha(ROOT / 'Scripts/ci/sanitizer_bodies.json'), 'host': str(host), 'hostSHA256': sha(host),
               'buildSystem': args.swift_build_system or 'toolchain default', 'binPath': str(bin_path),
               'loadedBundles': {t: {'path': str(b), 'sha256': sha(b)} for t, b in bundles.items()}, 'q03Child': q03,
               'leakDetection': False, 'scope': 'Pinned valid bodies only; no whole-suite or hosted-CI claim.'})
    print('V03_SANITIZERS_COMPLETE', args.sanitizer, len(completed))


if __name__ == '__main__':
    main()
