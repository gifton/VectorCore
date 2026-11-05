#!/usr/bin/env python3
"""Run full discovery against one already-built, source-verified Release bundle.

Every subprocess uses v03_process.run; no compilation or mandatory-driver change.
The result keeps failed/skipped/pending selectors, including a timed-out partial run.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'Scripts/ci'))
from check_sanitizers import sanitizer_host_environment
from swiftpm_products import check_no_rebuild, in_log
from v03_process import run, sha, test_environment, write_json
from scope_check import swift_testing, xctest, qualifies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver-result', type=Path, required=True)
    parser.add_argument('--source-identity', type=Path, required=True,
                        help='Pre-build JSON: inputs path/hash map and packageSHA256 (phase-one identity format).')
    parser.add_argument('--scope', required=True, help='Exact source scope, e.g. baseline or combined commits.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists(): raise ValueError('Use a fresh output directory')
    output.mkdir(parents=True)
    identity = json.loads(args.source_identity.read_text())
    inputs = identity['inputs']
    if not inputs or not any(p.startswith('Tests/') for p in inputs):
        raise ValueError('Source identity must include test inputs')
    _, tracked = run(['/usr/bin/git', 'ls-files', 'Sources', 'Tests', 'Benchmarks'], output, 'tracked-inputs',
                     cwd=ROOT, environment=test_environment(), timeout=15)
    if set(tracked.splitlines()) != set(inputs):
        raise ValueError('Source identity must include every tracked Sources/Tests/Benchmarks input')
    mismatches = [p for p, digest in inputs.items() if sha(ROOT / p) != digest]
    if mismatches or sha(ROOT / 'Package.swift') != identity['packageSHA256']:
        raise ValueError('Source identity mismatch: ' + str(mismatches))
    data = json.loads(args.driver_result.read_text())
    if data['configuration'] != 'release' or data['buildSystem'] != 'native':
        raise ValueError('This experiment requires the verified native Release layout')
    sanitizer = data['sanitizer']
    bundles = {Path(b['path']) for b in data['loadedBundles'].values()}
    inventory = sorted(Path(data['binPath']).rglob('*.xctest'))
    if len(bundles) != 1 or len(inventory) != 1:
        raise ValueError('Expected one complete native aggregate bundle; inventory=' + str(inventory))
    bundle = bundles.pop()
    for record in data['loadedBundles'].values():
        if sha(record['path']) != record['sha256']: raise ValueError('Bundle hash mismatch')
    for key in ['host', 'runtime']:
        if sha(data[key]) != data[key + 'SHA256']: raise ValueError(key + ' hash mismatch')
    swift = Path(data['host']).parents[3] / 'bin/swift'
    env = test_environment()
    env.update(ASAN_OPTIONS='detect_leaks=0:halt_on_error=1', TSAN_OPTIONS='halt_on_error=1')
    host, runtime, env = sanitizer_host_environment(swift,
        '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform', sanitizer, env)
    if str(host) != data['host'] or sha(runtime) != data['runtimeSHA256']:
        raise ValueError('Selected helper/runtime differs from build qualification')
    shutil.copytree(bundle.parents[2], output / bundle.parents[2].name)
    for path in [host, runtime, args.driver_result, args.source_identity]:
        shutil.copy2(path, output / path.name)
    write_json(output / 'identity.json', {'scope': args.scope, 'sourceIdentitySHA256': sha(args.source_identity),
        'driverResultSHA256': sha(args.driver_result), 'bundleSHA256': sha(bundle), 'inventory': list(map(str, inventory)),
        'hostSHA256': sha(host), 'runtimeSHA256': sha(runtime), 'sourceInputs': inputs,
        'frontendSHA256': sha(swift.parent / 'swift-frontend'), 'packageSHA256': sha(ROOT / 'Package.swift')})
    flags = ['--configuration', 'release', '--jobs', '2', '--scratch-path', Path(data['binPath']).parents[1],
             '--sanitize', sanitizer, '--build-system', 'native', '-Xswiftc', '-enable-testing']
    _, log = run([swift, 'test', 'list', '--skip-build', *flags], output, 'swift-list', cwd=ROOT, environment=env, timeout=60)
    check_no_rebuild(log, 'full discovery')
    full = set((output / 'swift-list.stdout').read_text().splitlines())
    _, log = run([host, '--test-bundle-path', bundle, '--list-tests', bundle, '--testing-library', 'swift-testing'],
                 output, 'testing-list', cwd=ROOT, environment=env, timeout=30)
    testing = set((output / 'testing-list.stdout').read_text().splitlines())
    if not testing or not testing <= full: raise ValueError('Missing or inconsistent discovery')
    xctests = full - testing
    if not xctests: raise ValueError('Missing XCTest discovery')
    for name, selectors in [('testing', testing), ('xctest', xctests)]:
        (output / (name + '-discovery.txt')).write_text('\n'.join(sorted(selectors)) + '\n')
    if sanitizer == 'thread':
        env.update(Q03_TSAN='1', Q03_CHILD_LOG_DIR=str(output / 'q03-children'))
        Path(env['Q03_CHILD_LOG_DIR']).mkdir()
    results = {}
    for framework in ['testing', 'xctest']:
        name = 'full-' + framework
        if framework == 'testing':
            argv = [host, '--test-bundle-path', bundle, '--no-parallel', '--event-stream-output-path',
                    output / 'testing.events.jsonl', '--event-stream-version', '0', bundle, '--testing-library', 'swift-testing']
            deadline = 600
            loaded_host = host
        else:
            loaded_host = Path('/Applications/Xcode.app/Contents/Developer/usr/bin/xctest').resolve()
            argv = [loaded_host, bundle.parents[2]]
            deadline = 600
        try:
            run(argv, output, name, cwd=ROOT, environment=env, timeout=deadline, expected=None)
        except RuntimeError:
            # run() retains timeout/cleanup metadata; the partial scope remains unqualified.
            pass
        command = json.loads((output / (name + '.json')).read_text())
        log = (output / (name + '.stdout')).read_text() + (output / (name + '.stderr')).read_text()
        if framework == 'testing':
            events = output / 'testing.events.jsonl'
            result = swift_testing(events.read_text() if events.exists() else '', testing)
        else:
            result = xctest(log, xctests)
        result['imagesVerified'] = all(in_log(p, log) for p in [loaded_host, runtime, bundle])
        result['qualified'] = qualifies(result, command) and result['imagesVerified']
        result['counts'] = {k: len(result[k]) for k in ['discovered', 'started', 'passed', 'failed', 'skipped', 'pending', 'unknown']}
        write_json(output / (framework + '-scope.json'), result)
        results[framework] = {'counts': result['counts'], 'qualified': result['qualified'], 'exit': command.get('exit'),
                              'timeout': command['timeout'], 'imagesVerified': result['imagesVerified']}
        print(framework, results[framework], flush=True)
    if any(sha(ROOT / p) != digest for p, digest in inputs.items()) or sha(bundle) != data['loadedBundles'][next(iter(data['loadedBundles']))]['sha256']:
        raise ValueError('Source or executed bundle changed during execution')
    write_json(output / 'summary.json', {'scope': args.scope, 'sanitizer': sanitizer, 'results': results,
        'note': 'XCTest host also runs Swift Testing in this toolchain; its extra text is retained as overlapping execution, not added to XCTest counts.'})


if __name__ == '__main__': main()
