#!/usr/bin/env python3
"""Compile, link and run public inherited/explicit Collection witnesses in both client modes.

Supply a fresh Debug or Release SwiftPM bin path built with the named compiler.
The driver never builds or changes the package. Every case must pass; failures are
retained and the driver exits nonzero after collecting the remaining controls.
"""
import argparse
import math
from pathlib import Path
import shutil

import swiftpm_products as products
from v03_process import run, sha, test_environment, write_json


def positive_deadline(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('deadline must be finite and positive')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--swiftc', type=Path, required=True)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--bin-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=positive_deadline, default=60)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    swiftc = args.swiftc.absolute()
    sdk = args.sdk.resolve()
    build = args.bin_path.resolve()
    fixture = root / 'Scripts/ci/fixtures/compile-contracts/CollectionWitnessConsumer.swift'
    shutil.copy2(fixture, output / fixture.name)
    objects = products.link_objects(build, ['VectorCore', 'VectorCoreC'])
    modules = products.module_search_path(build)
    modulemap = products.module_map(build, 'VectorCoreC')
    inputs = [Path(__file__).resolve(), fixture, swiftc.resolve(), swiftc.parent / 'swift-frontend',
              sdk / 'SDKSettings.json', modulemap, *objects]
    for module in modules.glob('VectorCore.*swiftmodule*'):
        inputs.extend(p for p in module.rglob('*') if p.is_file()) if module.is_dir() else inputs.append(module)
    write_json(output / 'inputs.json', {str(p): sha(p) for p in inputs})
    env = test_environment()
    env.update(SWIFT_EXEC=str(swiftc), PATH=str(swiftc.parent) + ':' + env['PATH'],
               CLANG_MODULE_CACHE_PATH=str(output / 'module-cache'), TMPDIR=str(output / 'tmp'))
    Path(env['TMPDIR']).mkdir()
    write_json(output / 'environment.json', {key: env[key] for key in (
        'SWIFT_EXEC', 'PATH', 'CLANG_MODULE_CACHE_PATH', 'TMPDIR',
        'VECTORCORE_TEST_EXTENDED', 'VECTORCORE_STRICT_PERF')})
    run([swiftc, '--version'], output, 'compiler', cwd=root, environment=env, timeout=args.deadline)
    common = [swiftc, '-swift-version', '6', '-target', 'arm64-apple-macosx14.0', '-sdk', sdk,
              '-module-cache-path', env['CLANG_MODULE_CACHE_PATH'], '-I', modules,
              '-Xcc', '-fmodule-map-file=' + str(modulemap),
              '-Xcc', '-I' + str(root / 'Sources/VectorCoreC/include'), '-parse-as-library']
    results = []
    for mode, optimization in [('debug', '-Onone'), ('release', '-O')]:
        for witness in ['inherited', 'explicit']:
            name = mode + '-' + witness
            binary = output / name
            defines = ['-D', 'EXPLICIT_WITNESS'] if witness == 'explicit' else []
            record, _ = run([*common, optimization, *defines, fixture, *objects,
                             '-framework', 'Accelerate', '-o', binary], output, name + '-compile',
                            cwd=root, environment=env, timeout=args.deadline, expected=None)
            result = {'mode': mode, 'witness': witness, 'compileExit': record['exit'], 'passed': False}
            if record['exit'] == 0:
                execution, text = run([binary], output, name + '-run', cwd=root,
                                      environment=env, timeout=args.deadline, expected=None)
                result.update(runExit=execution['exit'], binarySHA256=sha(binary),
                              passed=execution['exit'] == 0 and 'COLLECTION_WITNESS_OK' in text)
            results.append(result)
            write_json(output / 'summary.json', results)
    print('\n'.join(str(result) for result in results))
    return 0 if all(result['passed'] for result in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
