#!/usr/bin/env python3
"""Run by-value consumers, migration contracts and concrete stdlib diagnostics.

Supply fresh, matching compiler/library products. Both client optimization modes
and omitted-write controls are executed. Exit 0 requires the by-value checks,
negative controls and migration contracts to pass, and the concrete diagnostic
to execute completely. It does NOT require the concrete stdlib inout route to
return the right number or qualify the compiler. Those numerical failures stay
explicit in summary.json. No compiler admission or known-issue policy is changed.
"""
import argparse
import math
from pathlib import Path
import re
import shutil

import consumer_smoke
import swiftpm_products as products
from v03_process import run, sha, test_environment, write_json

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / 'Scripts/ci/fixtures/compile-contracts'


def expected_observations(*, omit_writes=False):
    """Manual exact-value oracles; never generated from the observed program."""
    one, two = ['3f800000'], ['40000000']
    double_one, double_two = ['3ff0000000000000'], ['4000000000000000']

    def frame(initial, replacement):
        # calls, callback return, inside-element count, after count, copy count;
        # then inside, entire mutated owner and entire retained-copy contents.
        n = len(initial)
        flatten = lambda values: [lane for element in values for lane in element]
        if omit_writes and n:
            replacement = initial[0]
        after = [replacement] + initial[1:] if n else []
        return ','.join(['1', '731', str(int(n > 0)), str(n), str(n)]
                        + (replacement if n else []) + flatten(after) + flatten(initial))

    empty = frame([], one)
    tail = frame([two, ['40400000'], ['40800000']], one)
    simd_float = frame([['40000000', '40800000', '40c00000', '41000000']] * 3,
                       ['3f800000', '40400000', '40a00000', '40e00000'])
    first = two[0] if omit_writes else one[0]
    thrown = '1,1,3,3,' + first + ',40400000,40800000,40000000,40400000,40800000'
    return {
        'concrete.float.singleton': first, 'concrete.float.empty': empty, 'concrete.float.tail': tail,
        'generic.float.singleton': first, 'generic.float.empty': empty, 'generic.float.tail': tail,
        'generic.double.one': frame([double_two], double_one),
        'generic.simd4float.tail': simd_float,
        'generic.simd4int8.tail': frame([['2', '4', '6', '8']] * 3, ['1', '3', '5', '7']),
        'external.float.tail': tail, 'external.simd4float.tail': simd_float,
        'generic.float.throw': thrown, 'external.float.throw': thrown,
        'storage.float.empty': empty,
        'storage.float.inline16': frame([two] * 16, one), 'storage.float.heap17': frame([two] * 17, one),
        'storage.double.inline8': frame([double_two] * 8, double_one),
        'storage.double.heap9': frame([double_two] * 9, double_one),
        'vector.dim3': frame([two] * 3, one), 'vector.dim512': frame([two] * 512, one),
        'vector.dynamic.empty': empty, 'vector.dynamic17': frame([two] * 17, one),
        'vector.optimized512': frame([two] * 512, one),
        'vector.inlinable.arithmetic': '40c00000,41000000,41200000,40000000,40400000,40800000',
    }


def judge(code, text, expected, *, negative_witness=None):
    """Require every registered numerical row and an honest completion record.

    The deliberate omitted-write build must exit 1 and match its entire expected
    observation table, including unchanged rows. A crash, a missing case or an
    arbitrary diagnostic is not that control. ``negative_witness`` is a complete
    numerical oracle, not a subset of witness rows.
    """
    observations = {}
    problems = []
    lines = text.splitlines()
    completion = None
    for index, line in enumerate(lines):
        row = re.fullmatch(r'BUFFER_VALUE ([a-z0-9.-]+) observed=([^\s]+)', line)
        end = re.fullmatch(r'BUFFER_VALUE_COMPLETE count=(\d+) failures=(\d+)', line)
        if row:
            name, value = row.groups()
            if name in observations:
                problems.append('duplicate case: ' + name)
            observations[name] = value
        elif end and index == len(lines) - 1:
            completion = tuple(map(int, end.groups()))
        else:
            problems.append('unexpected output: ' + line)
    if set(observations) != set(expected):
        problems.append('case inventory differs')
    failures = {name: {'observed': observations.get(name), 'expected': value}
                for name, value in expected.items() if observations.get(name) != value}
    if completion != (len(expected), len(failures)):
        problems.append('missing or inconsistent completion record')
    if negative_witness is None:
        passed = code == 0 and not problems and not failures
    else:
        passed = (code == 1 and not problems and bool(failures)
                  and set(negative_witness) == set(expected) and observations == negative_witness)
    return {'passed': passed, 'observations': observations, 'failures': failures, 'problems': problems}


def positive_deadline(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('deadline must be finite and positive')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--swiftc', required=True, type=Path)
    parser.add_argument('--frontend-sha256', required=True)
    parser.add_argument('--sdk', required=True, type=Path)
    parser.add_argument('--bin-path', required=True, type=Path)
    parser.add_argument('--library-configuration', required=True, choices=['debug', 'release'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--deadline', type=positive_deadline, default=90)
    args = parser.parse_args()
    swiftc, sdk = args.swiftc.absolute(), args.sdk.resolve()
    frontend = swiftc.parent / 'swift-frontend'
    if sha(frontend) != args.frontend_sha256:
        parser.error('frontend identity differs from requested SHA256')
    expected = expected_observations()
    if not expected:
        raise RuntimeError('No numerical cases registered')
    source = FIXTURES / 'BufferValueConsumer.swift'
    witness = FIXTURES / 'BufferValueWitness.swift'
    exposure = FIXTURES / 'BufferInoutExposure.swift'
    contracts = [row for row in consumer_smoke.CONTRACTS if row['id'].startswith('SIMDStorage-')]
    if {row['id'] for row in contracts} != {'SIMDStorage-new-conformer', 'SIMDStorage-legacy-call', 'SIMDStorage-legacy-conformer'}:
        raise RuntimeError('Incomplete SIMDStorage migration contract inventory')
    fixtures = [source, witness, exposure, *[ROOT / row['path'] for row in contracts]]
    for path in fixtures:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    build = args.bin_path.resolve()
    objects = products.link_objects(build, ['VectorCore', 'VectorCoreC'])
    modules = products.module_search_path(build)
    modulemap = products.module_map(build, 'VectorCoreC')
    inputs = [Path(__file__), ROOT / 'Scripts/ci/v03_process.py', ROOT / 'Scripts/ci/swiftpm_products.py',
              ROOT / 'Scripts/ci/consumer_smoke.py', *fixtures,
              swiftc.resolve(), frontend, sdk / 'SDKSettings.json', modulemap, *objects]
    for module in modules.glob('VectorCore.*'):
        inputs.extend(p for p in module.rglob('*') if p.is_file()) if module.is_dir() else inputs.append(module)
    inputs.extend(p for p in (ROOT / 'Sources/VectorCoreC/include').rglob('*') if p.is_file())
    hashes = {str(path): sha(path) for path in inputs}
    write_json(output / 'inputs.json', hashes)
    for path in fixtures:
        shutil.copy2(path, output / path.name)
    env = test_environment()
    env.update(SWIFT_EXEC=str(swiftc), SDKROOT=str(sdk), PATH=str(swiftc.parent) + ':' + env['PATH'])
    _, version = run([swiftc, '--version'], output, 'compiler-version', cwd=ROOT, environment=env)
    common = [swiftc, '-swift-version', '6', '-target', 'arm64-apple-macosx14.0', '-sdk', sdk,
              '-module-cache-path', output / 'module-cache', '-I', modules,
              '-Xcc', '-fmodule-map-file=' + str(modulemap),
              '-Xcc', '-I' + str(ROOT / 'Sources/VectorCoreC/include'), '-parse-as-library']
    summary = {'compilerVersion': version.strip(), 'libraryConfiguration': args.library_configuration,
               'binPath': str(build), 'compilerQualification': 'not determined by this driver',
               'cases': [], 'legacyExposure': [], 'migrationContracts': []}
    for mode, optimization in [('debug', '-Onone'), ('release', '-O')]:
        for contract in contracts:
            path = output / Path(contract['path']).name
            record, diagnostics = run([*common, optimization, '-typecheck', path], output,
                                      mode + '-' + contract['id'], cwd=ROOT, environment=env,
                                      timeout=args.deadline, expected=None)
            passed, detail = consumer_smoke.judge_contract(contract, record['exit'], diagnostics,
                                                         path.read_text(), path.name)
            summary['migrationContracts'].append({'id': contract['id'], 'clientMode': mode,
                                                 'exit': record['exit'], 'passed': passed, 'detail': detail})
        for negative in [False, True]:
            label = mode + ('-omitted-write' if negative else '-by-value')
            directory = output / label
            directory.mkdir()
            defines = ['-D', 'OMIT_WRITES'] if negative else []
            module_path, object_path = directory / 'BufferValueWitness.swiftmodule', directory / 'witness.o'
            c, _ = run([*common, optimization, *defines, '-module-name', 'BufferValueWitness',
                         '-emit-module', '-emit-module-path', module_path, '-emit-object',
                         output / witness.name, '-o', object_path], directory, 'witness-compile',
                        cwd=ROOT, environment=env, timeout=args.deadline, expected=None)
            result = {'clientMode': mode, 'negativeControl': negative, 'witnessCompileExit': c['exit'], 'passed': False}
            if c['exit'] == 0:
                binary = directory / 'consumer'
                compiled, _ = run([*common, optimization, '-I', directory, output / source.name,
                                    object_path, *objects, '-framework', 'Accelerate', '-o', binary],
                                   directory, 'consumer-compile', cwd=ROOT, environment=env,
                                   timeout=args.deadline, expected=None)
                result['compileExit'] = compiled['exit']
                if compiled['exit'] == 0:
                    executed, text = run([binary], directory, 'consumer-run', cwd=ROOT,
                                         environment=env, timeout=30, expected=None)
                    result.update(runExit=executed['exit'], binarySHA256=sha(binary))
                    result.update(judge(executed['exit'], text, expected,
                                        negative_witness=expected_observations(omit_writes=True) if negative else None))
            if not negative and c['exit'] == 0:
                # This separate emission records the retained generic witness
                # body. It is not a pass trace of the executed binary.
                sil = directory / 'witness.sil'
                emitted, _ = run([*common, optimization, '-module-name', 'BufferValueWitness',
                                   '-emit-sil', output / witness.name, '-o', sil], directory, 'witness-sil',
                                  cwd=ROOT, environment=env, timeout=args.deadline, expected=None)
                result['silExit'] = emitted['exit']
                calls = []
                if emitted['exit'] == 0:
                    calls = [line.strip() for line in sil.read_text().splitlines()
                             if 'witness_method' in line and '#SIMDStorage.withUnsafeMutableElements' in line]
                    result['silSHA256'] = sha(sil)
                result['genericWitnessCalls'] = calls
                result['passed'] = result['passed'] and bool(calls)
            summary['cases'].append(result)
            write_json(output / 'summary.json', summary)
        directory = output / (mode + '-legacy-exposure')
        directory.mkdir()
        binary = directory / 'consumer'
        compiled, _ = run([*common, optimization, output / exposure.name, *objects,
                            '-framework', 'Accelerate', '-o', binary], directory, 'compile',
                           cwd=ROOT, environment=env, timeout=args.deadline, expected=None)
        result = {'clientMode': mode, 'compileExit': compiled['exit'], 'passed': False}
        if compiled['exit'] == 0:
            executed, text = run([binary], directory, 'run', cwd=ROOT, environment=env, timeout=30, expected=None)
            result.update(runExit=executed['exit'], binarySHA256=sha(binary))
            result.update(judge(executed['exit'], text, {
                'concrete-inout': '3f800000', 'concrete-value': '3f800000'}))
        summary['legacyExposure'].append(result)
        write_json(output / 'summary.json', summary)
    summary['inputsUnchanged'] = all(sha(Path(path)) == digest for path, digest in hashes.items())
    summary['byValueAndNegativeControlsPassed'] = summary['inputsUnchanged'] and all(c['passed'] for c in summary['cases'])
    summary['legacyExposurePassed'] = all(c['passed'] for c in summary['legacyExposure'])
    summary['migrationContractsPassed'] = all(c['passed'] for c in summary['migrationContracts'])
    summary['concreteDiagnosticsCompleted'] = all(c.get('runExit') in (0, 1) and c.get('problems') == []
                                                  for c in summary['legacyExposure'])
    summary['validationPassed'] = (summary['byValueAndNegativeControlsPassed']
                                   and summary['migrationContractsPassed'] and summary['concreteDiagnosticsCompleted'])
    summary['artifacts'] = {str(p.relative_to(output)): sha(p) for p in output.rglob('*')
                            if p.is_file() and 'module-cache' not in p.parts and p.name != 'summary.json'}
    write_json(output / 'summary.json', summary)
    print('byValueAndNegativeControlsPassed=' + str(summary['byValueAndNegativeControlsPassed'])
          + ' migrationContractsPassed=' + str(summary['migrationContractsPassed'])
          + ' concreteDiagnosticsCompleted=' + str(summary['concreteDiagnosticsCompleted'])
          + ' legacyExposurePassed=' + str(summary['legacyExposurePassed']))
    return 0 if summary['validationPassed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
