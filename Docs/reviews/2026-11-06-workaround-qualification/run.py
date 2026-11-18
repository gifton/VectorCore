#!/usr/bin/env python3
"""Collect residual-exposure observations; never admit a compiler or release."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Scripts/ci'))
from consumer_buffer_value import judge
import swiftpm_products as products
from v03_process import run, sha, test_environment, write_json

PRIOR = ROOT.parent / 'VectorCore-buffer-api-migration-2026-11-06-evidence'
NAMES = [
    'new.concrete.value', 'new.concrete.copied-inout', 'new.inside.copied-inout',
    'new.aggregate.inout', 'new.generic.value', 'new.generic.copied-inout',
    'custom.value-bridge', 'custom.inout-bridge', 'optimized.storage.stdlib-inout',
    'optimized.storage.value', 'optimized.storage.copied-inout', 'optimized.float.value',
    'optimized.float.copied-inout', 'fixed.value', 'fixed.copied-inout',
    'dynamic.value', 'dynamic.copied-inout',
]
EXPECTED = {name: '3f800000' for name in NAMES}
OMITTED = {name: '40000000' for name in NAMES}
BY_VALUE = {name for name in NAMES if name.endswith('.value') or name == 'custom.value-bridge'}


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name('ResidualExposure.swift')
    source_hashes = {str(source): sha(source), str(Path(__file__).resolve()): sha(__file__)}
    frozen = read(PRIOR / 'compiled-source-inputs.json')
    assert all(sha(ROOT / path) == digest for path, digest in frozen.items())
    write_json(out / 'compiled-source-inputs.json', frozen)
    write_json(out / 'probe-inputs.json', source_hashes)
    shutil.copy2(source, out / source.name)
    shutil.copy2(__file__, out / 'run.py')
    summary = {'sourceBase': 'bdf990847babf567be1cbce22d2606e7102b1a47',
               'productReuse': 'Matching retained Release products from the completed by-value migration; no new library build.',
               'compilerGateChanged': False, 'releaseAdmitted': False, 'caseCount': len(NAMES),
               'expected': EXPECTED, 'omittedWriteExpected': OMITTED,
               'toolchains': [], 'collectionComplete': False}
    for label in ['xcode-27.2', 'swiftorg-6.4.2', 'xcode-27.3-beta']:
        prior = PRIOR / label
        target = out / label
        target.mkdir()
        identity = read(prior / 'identity.json')
        assert identity['sourceInputs'] == frozen
        control = read(prior / 'compiler-controls/summary.json')
        frontend_path = next(Path(path) for path in control['inputs'] if Path(path).name == 'swift-frontend')
        assert sha(frontend_path) == identity['frontendSHA256']
        swiftc = frontend_path.with_name('swiftc')
        sdk = Path(control['sdk'])
        assert sha(sdk / 'SDKSettings.json') == identity['sdkSettingsSHA256']
        binding = read(prior / 'release-consumer-binding.json')
        assert sha(binding['summary']) == binding['summarySHA256']
        assert sha(binding['buildReceipt']) == binding['buildReceiptSHA256']
        consumer = read(Path(binding['summary']))
        assert consumer['libraryConfiguration'] == 'release' and consumer['validationPassed']
        retained_inputs = read(prior / 'consumers-release/inputs.json')
        assert all(sha(path) == digest for path, digest in retained_inputs.items())
        build = Path(consumer['binPath'])
        objects = products.link_objects(build, ['VectorCore', 'VectorCoreC'])
        modules = products.module_search_path(build)
        modulemap = products.module_map(build, 'VectorCoreC')
        assert all(str(path) in retained_inputs for path in objects)
        inputs = {**retained_inputs, **source_hashes}
        write_json(target / 'inputs.json', inputs)
        write_json(target / 'producer-binding.json', {
            'priorBinding': str(prior / 'release-consumer-binding.json'),
            'priorBindingSHA256': sha(prior / 'release-consumer-binding.json'),
            'producerReceipt': binding['buildReceipt'], 'producerReceiptSHA256': binding['buildReceiptSHA256'],
            'frontendSHA256': identity['frontendSHA256'], 'sdkSettingsSHA256': identity['sdkSettingsSHA256'],
            'compiledSourceManifestSHA256': sha(PRIOR / 'compiled-source-inputs.json'),
        })
        env = test_environment()
        env.update(SWIFT_EXEC=str(swiftc), SDKROOT=str(sdk), PATH=str(swiftc.parent) + ':' + env['PATH'])
        _, version = run([swiftc, '--version'], target, 'compiler-version', cwd=ROOT, environment=env)
        common = [swiftc, '-swift-version', '6', '-target', 'arm64-apple-macosx14.0', '-sdk', sdk,
                  '-module-cache-path', target / 'module-cache', '-I', modules,
                  '-Xcc', '-fmodule-map-file=' + str(modulemap),
                  '-Xcc', '-I' + str(ROOT / 'Sources/VectorCoreC/include'), '-parse-as-library']
        row = {'toolchain': label, 'compilerVersion': version.strip(), 'frontendSHA256': identity['frontendSHA256'],
               'target': 'arm64-apple-macosx14.0', 'sdk': str(sdk), 'producerConfiguration': 'release', 'cases': []}
        summary['toolchains'].append(row)
        for mode, optimization in [('debug', '-Onone'), ('release', '-O')]:
            for omitted in [False, True]:
                name = mode + ('-omitted-write' if omitted else '-positive')
                directory = target / name
                directory.mkdir()
                binary = directory / 'consumer'
                defines = ['-D', 'OMIT_WRITES'] if omitted else []
                compiled, _ = run([*common, optimization, *defines, out / source.name, *objects,
                                    '-framework', 'Accelerate', '-o', binary], directory, 'compile',
                                   cwd=ROOT, environment=env, timeout=120, expected=None)
                case = {'clientMode': mode, 'omittedWrite': omitted, 'compileExit': compiled['exit']}
                row['cases'].append(case)
                write_json(out / 'summary.json', summary)
                assert compiled['exit'] == 0, 'Compilation failed; retained as setup failure: ' + str(directory)
                executed, text = run([binary], directory, 'run', cwd=ROOT, environment=env, timeout=30, expected=None)
                verdict = judge(executed['exit'], text, EXPECTED, negative_witness=OMITTED if omitted else None)
                case.update(verdict)
                case.update(runExit=executed['exit'], binarySHA256=sha(binary))
                complete = not verdict['problems'] and executed['exit'] == int(bool(verdict['failures']))
                case['observationComplete'] = complete
                case['byValueControlsPassed'] = all(verdict['observations'].get(name) == EXPECTED[name] for name in BY_VALUE) if not omitted else None
                write_json(out / 'summary.json', summary)
                assert complete and (not omitted or verdict['passed']), 'Incomplete observation or incorrect omitted-write control'
                print(label, mode, 'omitted' if omitted else 'positive', 'failures:', sorted(verdict['failures']), flush=True)
            sil = target / (mode + '.sil')
            emitted, _ = run([*common, optimization, '-emit-sil', out / source.name, '-o', sil],
                              target, mode + '-sil', cwd=ROOT, environment=env, timeout=120, expected=(0,))
        row['inputsUnchanged'] = all(sha(path) == digest for path, digest in inputs.items())
        assert row['inputsUnchanged']
        row['artifacts'] = {str(path.relative_to(target)): sha(path) for path in target.rglob('*')
                            if path.is_file() and 'module-cache' not in path.parts}
        write_json(out / 'summary.json', summary)
    assert all(sha(ROOT / path) == digest for path, digest in frozen.items())
    assert all(sha(path) == digest for path, digest in source_hashes.items())
    summary['compiledSourcesUnchanged'] = True
    summary['collectionComplete'] = True
    summary['byValueControlsPassed'] = all(case['byValueControlsPassed'] for row in summary['toolchains']
                                          for case in row['cases'] if not case['omittedWrite'])
    summary['allPositiveShapesCorrect'] = all(case['passed'] for row in summary['toolchains']
                                            for case in row['cases'] if not case['omittedWrite'])
    summary['limits'] = ['Native macOS arm64, matching Release producer products only.',
                         'Client-local custom conformer and generic helpers; no new separate witness module.',
                         'First-lane observation only; no new tail/COW/empty/throw/descriptor-replacement coverage.',
                         'No arbitrary pointer-bearing state, external generated source, physical-device or performance qualification.']
    write_json(out / 'summary.json', summary)
    print('Collection complete; compiler policy unchanged.', flush=True)


if __name__ == '__main__':
    main()
