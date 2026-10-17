#!/usr/bin/env python3
"""P02 strict Swift 6 public consumers (`import VectorCore`, never @testable) and their immutable evidence.
Evidence script, not wired into CI (its name does not match unittest discovery's `test*.py`).

usage: P02RunConsumers.py BIN_PATH OUTPUT baseline|final
  BIN_PATH  a SwiftPM bin path of the tree under test (`swift build --show-bin-path`; native or swiftbuild layout)
  OUTPUT    a new directory (must not exist)
  baseline  expectations before P02 (no BatchSearchCPUStrategy); final: expectations after P02.

Fixtures, Tests/P02Consumers/ (phase 1 check map S4, P1, P2):
  PublicBatchStrategy.swift  final: typecheck rc 0, link against the snapshot objects, run rc 0, print its marker;
                             baseline: typecheck rc 1 naming the missing type `BatchSearchCPUStrategy`.
  ParameterOrder.swift       final: typecheck rc 1 with exactly one error, which names `metric` and `cpuStrategy`
                             (cpuStrategy is the final parameter); baseline: rc 1 (no `cpuStrategy` parameter).
Every compile is `swiftc -swift-version 6 -parse-as-library` from `xcrun --find swiftc` against the snapshot's module
and objects (Scripts/ci/swiftpm_products.snapshot_consumer_inputs), as D02RunConsumers.py does. Children run in their
own process group: compiles 120 s, executables 15 s, cleanup 2 s, then a throw; only the child's own group is
signalled. Filesystem only; no git.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / 'Scripts/ci'))
import swiftpm_products as products  # noqa: E402

FIXTURES = root / 'Tests/P02Consumers'
COMPILE_DEADLINE, RUN_DEADLINE, CLEANUP = 120, 15, 2
MARKER = 'P02_PUBLIC_BATCH_STRATEGY_COMPLETE'
# fixture -> {mode: (kind, detail)}
TABLE = {
    'PublicBatchStrategy.swift': {'baseline': ('fails', "cannot find type 'BatchSearchCPUStrategy' in scope"),
                                  'final': ('positive', MARKER)},
    'ParameterOrder.swift': {'baseline': ('fails', None),
                             'final': ('negative', ("'metric'", "'cpuStrategy'"))},
}

active = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def terminate(signum, frame):
    if active is not None and active.poll() is None:
        os.killpg(active.pid, signal.SIGKILL)
        try:
            active.wait(timeout=CLEANUP)
        except subprocess.TimeoutExpired:
            pass
    raise SystemExit(128 + signum)


def main():
    build, output, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    assert mode in ('baseline', 'final')
    output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    snap = products.snapshot_consumer_inputs(build, output)
    fixtures = sorted(FIXTURES.glob('*.swift'))
    assert sorted(p.name for p in fixtures) == sorted(TABLE), 'fixture table and directory differ'
    for source in fixtures + [Path(__file__)]:
        (output / source.name).write_bytes(source.read_bytes())
    record = {'mode': mode, 'bin': str(build),
              'sourceHashes': {str(p.relative_to(root)): sha(p) for p in sorted(root.glob('Sources/**/*.swift'))},
              'fixtureHashes': {p.name: sha(p) for p in fixtures + [Path(__file__)]},
              'artifacts': {str(dst.relative_to(output)): sha(dst) for _, dst in snap['records']}, 'runs': [],
              'problems': []}
    swiftc = subprocess.run(['xcrun', '--find', 'swiftc'], capture_output=True, text=True, timeout=60).stdout.strip()
    sdk = subprocess.run(['xcrun', '--sdk', 'macosx', '--show-sdk-path'], capture_output=True, text=True,
                         timeout=60).stdout.strip()
    assert swiftc and sdk, 'xcrun could not resolve swiftc or the macOS SDK'
    record['swiftc'], record['sdk'] = swiftc, sdk
    record['swiftcVersion'] = subprocess.run([swiftc, '--version'], capture_output=True, text=True,
                                             timeout=60).stdout.strip()
    includes = ['-sdk', sdk, '-I', str(snap['modules']), '-Xcc', '-fmodule-map-file=' + str(snap['modulemap']),
                '-Xcc', '-I' + str(root / 'Sources/VectorCoreC/include')]

    def run(name, command, deadline):
        global active
        path = output / (name + '.log')
        with path.open('xb') as log:
            active = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            pid = active.pid
            try:
                status = active.wait(timeout=deadline)
            except subprocess.TimeoutExpired:
                os.killpg(pid, signal.SIGKILL)
                try:
                    active.wait(timeout=CLEANUP)
                except subprocess.TimeoutExpired:
                    pass
                raise SystemExit(f'{name}: timed out after {deadline} s')
        active = None
        record['runs'].append({'name': name, 'command': command, 'exit': status, 'pid': pid,
                               'deadlineSeconds': deadline, 'cleanupSeconds': CLEANUP, 'logSHA256': sha(path)})
        return status, path.read_text(errors='replace')

    results = {}
    for fixture in fixtures:
        kind, detail = TABLE[fixture.name][mode]
        copy = output / fixture.name
        stem = fixture.stem
        status, text = run(f'typecheck-{stem}', [swiftc, '-swift-version', '6', '-parse-as-library', '-typecheck',
                                                 *includes, str(copy)], COMPILE_DEADLINE)
        errors = re.findall(rf'{re.escape(fixture.name)}:(\d+):\d+: error: ([^\n]+)', text)
        warnings = re.findall(rf'{re.escape(fixture.name)}:(\d+):\d+: warning: ([^\n]+)', text)
        outcome = {'kind': kind, 'typecheckExit': status, 'errors': [f'{n}: {m}' for n, m in errors],
                   'warnings': [f'{n}: {m}' for n, m in warnings]}
        ok = False
        if kind == 'positive':
            ok = status == 0 and not warnings
            if ok:
                exe = output / f'bin-{stem}'
                status, _ = run(f'link-{stem}', [swiftc, '-swift-version', '6', '-parse-as-library', *includes,
                                                 str(copy), *map(str, snap['objects']), '-o', str(exe)],
                                COMPILE_DEADLINE)
                outcome['linkExit'] = status
                if status == 0:
                    status, text = run(f'run-{stem}', [str(exe)], RUN_DEADLINE)
                    outcome['runExit'] = status
                    outcome['marker'] = detail in text
                    outcome['output'] = text.strip().split('\n')[-1][:400]
                    record.setdefault('executables', {})[exe.name] = sha(exe)
                ok = outcome.get('runExit') == 0 and outcome.get('marker', False)
        elif kind == 'negative':
            ok = status == 1 and len(errors) == 1 and all(name in errors[0][1] for name in detail)
            outcome['requiredInTheOneError'] = list(detail)
        elif kind == 'fails':
            ok = status == 1 and (detail is None or detail in text)
            outcome['requiredDiagnostic'] = detail
        outcome['passed'] = ok
        results[fixture.name] = outcome
        print(fixture.name, kind, 'passed' if ok else 'FAILED', flush=True)
        if not ok:
            record['problems'].append(f'{fixture.name}: {outcome}')
    record['results'] = results
    (output / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    for problem in record['problems']:
        print('PROBLEM', problem)
    print(f'P02_CONSUMERS_{mode.upper()}_{"COMPLETE" if not record["problems"] else "FAILED"}', flush=True)
    return 0 if not record['problems'] else 1


if __name__ == '__main__':
    sys.exit(main())
