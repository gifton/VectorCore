#!/usr/bin/env python3
"""Run standalone lost-store probes with ordinary optimization and bounded evidence.

Exit 1 means at least one non-negative probe failed numerical acceptance. The
negative control must print the stale bits and be rejected in every configuration.
"""
import argparse
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'Scripts/ci'))
from v03_process import run, sha, test_environment, write_json


def accepts(code, output):
    return code == 0 and output == 'LOST_STORE observed=1.0 bits=3f800000 expected=1.0\n'


def probe_sources(new, retained):
    paths = [new / name for name in (
        'ByValueReadAfter.swift', 'CopyInoutReturnInside.swift', 'NoWriteNegative.swift',
        'NonLiteralReadAfter.swift', 'OutOfLineRead.swift')]
    paths += [retained / name for name in (
        'ByValueReturnInside.swift', 'LostStoreObserved.swift', 'ReturnInsideInout.swift')]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError('Required compiler probe: ' + str(path))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--swiftc', type=Path, required=True)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frontend-sha256', required=True)
    parser.add_argument('--deadline', type=float, default=60)
    args = parser.parse_args()
    if not math.isfinite(args.deadline) or args.deadline <= 0:
        parser.error('--deadline must be finite and positive')
    swiftc = args.swiftc.absolute()
    frontend = swiftc.parent / 'swift-frontend'
    if sha(frontend) != args.frontend_sha256:
        parser.error('frontend hash does not match requested identity')
    sdk = args.sdk.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = test_environment()
    environment.update(SWIFT_EXEC=str(swiftc), SDKROOT=str(sdk),
                       PATH=str(swiftc.parent) + ':' + os.environ.get('PATH', ''))
    sources = output / 'sources'
    sources.mkdir()
    retained = ROOT / 'Docs/reviews/2026-11-01-compiler-qualification/probes'
    probes = probe_sources(Path(__file__).with_name('probes'), retained)
    for probe in probes:
        shutil.copy2(probe, sources / probe.name)
    shutil.copy2(__file__, output / 'driver.py')
    inputs = {str(p): sha(p) for p in [swiftc.resolve(), frontend, sdk / 'SDKSettings.json',
                                     Path(__file__), ROOT / 'Scripts/ci/v03_process.py', *probes]}
    _, version = run([swiftc, '--version'], output, 'compiler-version', cwd=ROOT,
                     environment=environment, timeout=args.deadline)
    _, revision = run(['git', 'rev-parse', 'HEAD'], output, 'source-revision', cwd=ROOT, timeout=10)
    _, state = run(['git', 'status', '--short'], output, 'source-state', cwd=ROOT, timeout=10)
    _, host = run(['/usr/bin/sw_vers'], output, 'host', cwd=ROOT, timeout=10)
    summary = {'inputs': inputs, 'compilerVersion': version.strip(), 'revision': revision.strip(),
               'sourceState': state, 'host': host, 'target': 'arm64-apple-macosx14.0',
               'sdk': str(sdk), 'cells': []}
    write_json(output / 'summary.json', summary)
    common = ['-swift-version', '6', '-target', summary['target'], '-sdk', sdk,
              '-module-cache-path', output / 'module-cache']
    for probe in probes:
        for configuration, flags in [('debug', ['-Onone', '-D', 'DEBUG']), ('release', ['-O'])]:
            label = probe.stem + '-' + configuration
            binary = output / label
            compiled, _ = run([swiftc, *common, *flags, sources / probe.name, '-o', binary],
                              output, label + '-compile', cwd=ROOT, environment=environment,
                              timeout=args.deadline, expected=None)
            cell = {'probe': probe.name, 'configuration': configuration, 'compileExit': compiled['exit'],
                    'sourceSHA256': sha(sources / probe.name), 'accepted': False}
            if compiled['exit'] == 0:
                executed, observation = run([binary], output, label + '-run', cwd=ROOT,
                                           environment=environment, timeout=10, expected=None)
                cell.update(runExit=executed['exit'], observation=observation.strip(),
                            binarySHA256=sha(binary), accepted=accepts(executed['exit'], observation))
                if probe.name == 'NoWriteNegative.swift':
                    cell['negativeControlPassed'] = (executed['exit'] == 0 and observation ==
                        'LOST_STORE observed=2.0 bits=40000000 expected=1.0\n' and not cell['accepted'])
            summary['cells'].append(cell)
            write_json(output / 'summary.json', summary)
        if probe.name in ('LostStoreObserved.swift', 'ReturnInsideInout.swift', 'ByValueReturnInside.swift', 'CopyInoutReturnInside.swift'):
            for action, extension in [('-emit-sil', 'sil'), ('-emit-ir', 'll')]:
                path = output / (probe.stem + '.' + extension)
                run([swiftc, *common, '-O', action, sources / probe.name, '-o', path], output,
                    probe.stem + '-' + extension, cwd=ROOT, environment=environment, timeout=args.deadline)
    summary['artifacts'] = {p.name: sha(p) for p in sorted(output.iterdir())
                            if p.is_file() and p.name != 'summary.json'}
    write_json(output / 'summary.json', summary)
    failures = [c for c in summary['cells'] if not c.get('negativeControlPassed', c['accepted'])]
    print('cells=' + str(len(summary['cells'])) + ' numerical_or_compile_failures=' + str(len(failures)))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
