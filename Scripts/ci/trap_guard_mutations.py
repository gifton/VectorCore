#!/usr/bin/env python3
"""Bounded runtime guard-deletion checks in a disposable copy; never edit the input tree.

Example: python3 Scripts/ci/trap_guard_mutations.py --compiler-bin BIN --sdk SDK \
    --output /tmp/new-evidence --configuration release --targets basis,min
Each deletion is restored before the next build. The nearest guard is redundant with
MinHeap's rejection and is reported as equivalent, not counted as a detected deletion.
"""
import argparse
import atexit
import hashlib
import io
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tarfile

from v03_process import run, write_json


def delete_guard(source, target):
    text = source.decode()
    needle = target['remove']
    occurrence = target.get('occurrence', 0)
    count = text.count(needle)
    expected_count = 2 if target['id'].endswith('-count') else 1
    if count != expected_count:
        raise ValueError(f"{target['id']}: expected {expected_count} exact guards, found {count}")
    index = -1
    for _ in range(occurrence + 1):
        index = text.index(needle, index + 1)
    return (text[:index] + text[index + len(needle):]).encode()


def validate_reused_source(source_hashes, reused_input):
    if reused_input is not None and source_hashes != reused_input['sourceHashes']:
        raise ValueError('Reused disposable snapshot does not match its original production hashes')


def executed_test_count(log):
    swift = re.findall(r'Test run with (\d+) tests?', log)
    xctest = re.findall(r'Executed (\d+) tests?', log)
    # XCTest bundles also emit a zero-test Swift Testing summary; retain the nonzero XCTest count.
    return max(map(int, swift + xctest), default=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-bin', type=Path, required=True)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--configuration', choices=['debug', 'release'], required=True)
    parser.add_argument('--targets', help='Comma-separated finite target IDs; default all')
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--focused-build', action='store_true', help='In disposable copy only, compile complete owned suites and real helper dependencies')
    parser.add_argument('--source-ref', help='Snapshot this git revision instead of working-tree files')
    parser.add_argument('--reuse-snapshot', type=Path, help='Reuse a previous disposable source/build pair after a setup failure; it must not be in use')
    parser.add_argument('--observe-gap', action='store_true', help='Record original weak-test outcomes without requiring detection')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    manifest = json.loads(Path(__file__).with_suffix('.json').read_text())
    selected = args.targets.split(',') if args.targets else [t['id'] for t in manifest['targets']]
    if len(selected) != len(set(selected)) or set(selected) - {t['id'] for t in manifest['targets']}:
        parser.error('targets must be distinct known IDs')
    targets = [t for t in manifest['targets'] if t['id'] in selected]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tree = output / 'source'
    reused_input = None
    if args.reuse_snapshot:
        previous = args.reuse_snapshot.resolve()
        reused_input = json.loads((previous / 'input.json').read_text())
        if reused_input['configuration'] != args.configuration:
            parser.error('reused snapshot configuration differs')
        tree = previous / 'source'
    elif args.source_ref:
        tree.mkdir()
        archive = subprocess.check_output(['git', 'archive', args.source_ref], cwd=repo)
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(tree, filter='data')
    else:
        shutil.copytree(repo, tree, ignore=shutil.ignore_patterns('.git', '.build', '__pycache__', '.DS_Store'))
    scratch = args.reuse_snapshot.resolve() / 'build' if args.reuse_snapshot else output / 'build'
    env = dict(os.environ)
    env.update(PATH=str(args.compiler_bin) + ':' + env.get('PATH', ''), SWIFT_EXEC=str(args.compiler_bin / 'swiftc'),
               VECTORCORE_TEST_EXTENDED='0', VECTORCORE_STRICT_PERF='0',
               CLANG_MODULE_CACHE_PATH=str(output / 'modules'), SWIFT_BACKTRACE='enable=no')
    developer = '/Applications/Xcode.app/Contents/Developer'
    env['DYLD_LIBRARY_PATH'] = str(args.compiler_bin.parent / 'lib/swift/macosx/testing') + ':' + developer + '/Platforms/MacOSX.platform/Developer/usr/lib:' + developer + '/usr/lib'
    env['DYLD_FRAMEWORK_PATH'] = developer + '/Platforms/MacOSX.platform/Developer/Library/Frameworks:' + developer + '/Library/Frameworks'
    # Retain each child capture alongside the parent log, including expected failures.
    env.update(P01_CAPTURE_DIRECTORY=str(output / 'captures'), R04_CAPTURE_DIRECTORY=str(output / 'captures'),
               TRAP_CAPTURE_DIRECTORY=str(output / 'captures'))
    build_command = [args.compiler_bin / 'swift', 'test', 'list', '--build-system', 'native', '--jobs', '2',
                     '--sdk', args.sdk, '-c', args.configuration, '--scratch-path', scratch]
    restore_package = None
    if args.focused_build:
        package_path = tree / 'Package.swift'
        package_original = package_path.read_bytes()
        members = [
            'D02DimensionReachabilityTests.swift', 'D02DegenerateWidthTests.swift',
            'HeapMembershipContractTests.swift', 'ExecutionBoundaryTests.swift',
            'P01TopKBufferRedTests.swift', 'P01RedSupport.swift',
            'TrapObservation.swift', 'EntropyRangeContractTests.swift',
            'E06ConfigurationTestHelpers.swift', 'BatchConfigurationContractTests.swift',
            # Exists only on the pre-migration base used for original-gap controls.
            'ChildProbeLauncher.swift',
        ]
        members = [name for name in members if (tree / 'Tests/ComprehensiveTests' / name).is_file()]
        package = package_original.decode()
        pattern = r'(\.testTarget\(\s*name: "ComprehensiveTests",\s*dependencies: \[[^\]]*\],)'
        addition = '\n            sources: [' + ', '.join(json.dumps(name) for name in members) + '],'
        package, count = re.subn(pattern, lambda m: m[0] + addition, package)
        if count != 1:
            raise RuntimeError('Expected one unchanged ComprehensiveTests target')
        package_path.write_text(package)
        write_json(output / 'focused-package.json', {'members': members,
                   'originalSHA256': hashlib.sha256(package_original).hexdigest(),
                   'focusedSHA256': hashlib.sha256(package_path.read_bytes()).hexdigest()})
        (output / 'package-original.swift').write_bytes(package_original)
        def restore_package():
            package_path.write_bytes(package_original)
            write_json(output / 'focused-package-restoration.json',
                       {'restoredSHA256': hashlib.sha256(package_path.read_bytes()).hexdigest(),
                        'matchesOriginal': package_path.read_bytes() == package_original})
        atexit.register(restore_package)

    source_hashes = {str(p.relative_to(tree)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (tree / 'Sources').rglob('*') if p.is_file()}
    validate_reused_source(source_hashes, reused_input)
    write_json(output / 'input.json', {'revision': subprocess.check_output(['git', 'rev-parse', args.source_ref or 'HEAD'], cwd=repo, text=True).strip(),
                                     'configuration': args.configuration, 'targets': targets, 'sourceHashes': source_hashes, 'reusedInput': reused_input,
                                     'testHashes': {str(p.relative_to(tree)): hashlib.sha256(p.read_bytes()).hexdigest()
                                                    for p in (tree / 'Tests').rglob('*') if p.is_file()}})

    def build(name):
        run(build_command, output, name, cwd=tree, environment=env, timeout=900)

    build('baseline-build')
    bundles = list(scratch.glob('*/' + args.configuration + '/VectorCorePackageTests.xctest'))
    if len(bundles) != 1:
        raise RuntimeError('Expected exactly one test bundle: ' + str(bundles))
    bundle = bundles[0]
    binary = bundle / 'Contents/MacOS/VectorCorePackageTests'
    helper = args.compiler_bin.parent / 'libexec/swift/pm/swiftpm-testing-helper'

    def execute(target, name):
        filt = target['filter']
        if args.observe_gap and target.get('runner') == 'xctest':
            filt = 'ComprehensiveTests.ExecutionBoundaryTests/testDangerousBoundariesInChildren'
        if target.get('runner') == 'xctest':
            argv = [developer + '/usr/bin/xctest', '-XCTest', filt, bundle]
        else:
            argv = [helper, '--test-bundle-path', binary, '--filter', filt, binary, '--testing-library', 'swift-testing']
        metadata, log = run(argv, output, name, cwd=tree, environment=env, timeout=120, expected=None)
        captures = output / 'child-captures' / name
        records = []
        for raw in sorted(set(re.findall(r'/[^\s]+\.log', log))):
            path = Path(raw)
            if path.is_file():
                captures.mkdir(parents=True, exist_ok=True)
                destination = captures / path.name
                shutil.copyfile(path, destination)
                lines = path.read_text(errors='replace').splitlines()
                records.append({'capture': str(destination), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                'markers': [line for line in lines if any(marker in line for marker in
                                            ['TRAP_', 'D02_', 'R04_', 'P01_', 'E01_BODY_REACHED'])]})
        write_json(output / (name + '-observations.json'), {'captures': records,
                   'childResults': [line for line in log.splitlines() if 'CHILD_RESULT ' in line]})
        return metadata, log

    # Selection must execute real tests and succeed before the mutation can count.
    for target in targets:
        metadata, log = execute(target, 'baseline-' + target['id'])
        if metadata['exit'] != 0 or executed_test_count(log) == 0:
            raise RuntimeError('Baseline did not pass: ' + target['id'])
    rows = []
    if not args.baseline_only:
        for target in targets:
            path = tree / target['file']
            original = path.read_bytes()
            mutation = {'id': target['id'], 'originalSHA256': hashlib.sha256(original).hexdigest()}
            try:
                mutant = delete_guard(original, target)
                mutation['mutantSHA256'] = hashlib.sha256(mutant).hexdigest()
                path.write_bytes(mutant)
                build('mutant-build-' + target['id'])
                metadata, log = execute(target, 'mutant-' + target['id'])
                detected = metadata['exit'] != 0 and target.get('failure', '') in log
                row = {'id': target['id'], 'exit': metadata['exit'], 'detected': detected,
                       'equivalentDownstream': target.get('equivalentDownstream', False)}
                rows.append(row)
                write_json(output / 'results.json', rows)
                if not args.observe_gap:
                    if target.get('equivalentDownstream'):
                        if metadata['exit'] != 0 or executed_test_count(log) == 0:
                            raise RuntimeError('Equivalent guard unexpectedly changed behavior: ' + target['id'])
                    elif not detected:
                        raise RuntimeError('Missing expected guard evidence: ' + target['id'])
            finally:
                path.write_bytes(original)
                mutation['restoredSHA256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                write_json(output / ('mutation-' + target['id'] + '-restoration.json'), mutation)
                if path.read_bytes() != original:
                    raise RuntimeError('Restoration failed: ' + str(path))
        # Do not leave a mutant binary as the final artifact.
        build('restored-build')
        for target in targets:
            metadata, _ = execute(target, 'restored-' + target['id'])
            if metadata['exit'] != 0:
                raise RuntimeError('Restored test failed: ' + target['id'])
    final_hashes = {str(p.relative_to(tree)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (tree / 'Sources').rglob('*') if p.is_file()}
    if final_hashes != source_hashes:
        raise RuntimeError('Production source changed')
    write_json(output / 'restoration.json', {'sourceHashesMatch': True})
    if restore_package:
        restore_package()
        atexit.unregister(restore_package)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
