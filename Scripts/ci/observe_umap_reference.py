#!/usr/bin/env python3
"""Measure fixed-budget reference drift on one CI host/configuration.

Runs unmodified alpha, retained accepted snapshot and current candidate through
an independent public-API fixture. A miss fails; this script cannot adjust budgets
or admit a new exact-reference profile.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import zipfile

import umap_reference as reference


def decode_word(word):
    if type(word) is not int or not 0 <= word <= 0xffffffff:
        raise ValueError('invalid Float word')
    value = struct.unpack('>f', struct.pack('>I', word))[0]
    if not math.isfinite(value):
        raise ValueError('nonfinite reference/candidate')
    return value


def ordered_word(word):
    return (~word & 0xffffffff) if word & 0x80000000 else word | 0x80000000


def deviation(actual, expected):
    if not actual or len(actual) != len(expected):
        raise ValueError('reference/candidate shape mismatch')
    errors = [abs(decode_word(a) - decode_word(b)) for a, b in zip(actual, expected)]
    worst = max(range(len(errors)), key=errors.__getitem__)
    return {'maxAbsolute': errors[worst], 'worstIndex': worst,
            'maxULP': max(abs(ordered_word(a) - ordered_word(b)) for a, b in zip(actual, expected))}


def compare(candidate, control, profile):
    wanted = set(profile['cases'])
    if set(candidate['cases']) != wanted or set(control['cases']) != wanted:
        raise ValueError('capture inventory mismatch')
    result = {'passed': True, 'cases': {}, 'structureMatches': True}
    for field in ('rowOffsets', 'neighborIndices'):
        if candidate[field] != profile[field] or control[field] != profile[field]:
            result['structureMatches'] = result['passed'] = False
    for name, case in profile['cases'].items():
        historical = deviation(candidate['cases'][name], profile['words'][case['reference']])
        paired = deviation(candidate['cases'][name], control['cases'][name])
        passed = historical['maxAbsolute'] <= case['tolerance'] and paired['maxAbsolute'] <= case['tolerance']
        result['cases'][name] = {'historical': historical, 'sameRunner': paired,
                                 'ceiling': case['tolerance'], 'passed': passed}
        result['passed'] &= passed
    return result


def checkout_revision(revision, output):
    # Fetch only a missing pinned commit. Preserve the working tree and branch stack.
    exists = subprocess.run(['git', 'cat-file', '-e', revision + '^{commit}'], cwd=reference.ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    if exists.returncode:
        subprocess.run(['git', 'fetch', '--no-tags', 'origin', revision], cwd=reference.ROOT, check=True, timeout=180)
    observed = reference.command_text(['git', 'rev-parse', revision + '^{commit}'])
    if observed != revision:
        raise ValueError('pinned reference revision mismatch')
    output.mkdir(parents=True)
    archive = output.parent / (output.parent.name + '.zip')
    subprocess.run(['git', 'archive', '--format=zip', '--output', str(archive), revision],
                   cwd=reference.ROOT, check=True, timeout=120)
    with zipfile.ZipFile(archive) as packed:
        packed.extractall(output)
    return output


def capture(dependency, label, configuration, output, modern):
    client = output / label
    (client / 'Sources/Capture').mkdir(parents=True)
    defines = '[.define("VECTORCORE_CURRENT_API")]' if modern else '[]'
    manifest = ('// swift-tools-version: 6.0\nimport PackageDescription\n'
                'let package = Package(name: "UMAPReferenceCapture", platforms: [.macOS(.v14)],\n'
                'dependencies: [.package(name: "VectorCore", path: ' + json.dumps(str(dependency)) + ')],\n'
                'targets: [.executableTarget(name: "Capture", dependencies: [.product(name: "VectorCore", '
                'package: "VectorCore")], swiftSettings: ' + defines + ')])\n')
    (client / 'Package.swift').write_text(manifest)
    fixtures = reference.ROOT / 'Scripts/ci/fixtures/umap-reference'
    shutil.copyfile(fixtures / 'LegacyCapture.swift', client / 'Sources/Capture/main.swift')
    if modern:
        shutil.copyfile(fixtures / 'CurveCapture.swift', client / 'Sources/Capture/CurveCapture.swift')
    env = reference.test_environment(os.environ, {}, output)
    env.pop('VECTORCORE_UMAP_CHECK_BASELINE_BITS')
    env.pop('VECTORCORE_UMAP_REFERENCE_CONTEXT')
    command = ['swift', 'build', '--package-path', str(client), '-c', configuration, '--jobs', '3']
    print(f'Capturing {label} ({configuration})', flush=True)
    if reference.bounded_run(command, output / (label + '-build.log'), env=env):
        raise ValueError(label + ' did not build unchanged')
    binary_dir = reference.command_text(command + ['--show-bin-path'])
    records = []
    for attempt in range(2):
        log = output / f'{label}-run-{attempt}.log'
        if reference.bounded_run([str(Path(binary_dir) / 'Capture')], log, env=env, timeout=60):
            raise ValueError(label + ' capture process failed')
        rows = [line.removeprefix('UMAP_CAPTURE_JSON ') for line in log.read_text().splitlines()
                if line.startswith('UMAP_CAPTURE_JSON ')]
        if len(rows) != 1:
            raise ValueError(label + ' capture marker missing or duplicated')
        records.append(json.loads(rows[0]))
    if records[0] != records[1]:
        raise ValueError(label + ' is not repeatable across fresh processes')
    (output / (label + '.json')).write_text(json.dumps(records[0], indent=2) + '\n')
    return records[0]


def alpha_control(alpha, accepted, profile):
    # Alpha legacy routes anchor both default and explicit Euclidean candidates.
    result = dict(alpha, cases={})
    expected_alpha = {name.replace('-explicit', '-legacy') for name in profile['cases']
                      if name != 'default-coefficients'}
    if set(alpha['cases']) != expected_alpha or set(accepted['cases']) != set(profile['cases']):
        raise ValueError('independent control inventory mismatch')
    for name in profile['cases']:
        result['cases'][name] = (accepted['cases'][name] if name == 'default-coefficients'
                                 else alpha['cases'][name.replace('-explicit', '-legacy')])
    return result


def markdown(report):
    lines = ['| Case | Max abs vs historical | Max abs vs same-runner control | Worst historical index | Result |',
             '|---|---:|---:|---:|---|']
    for name, row in report['comparison']['cases'].items():
        lines.append(f"| {name} | {row['historical']['maxAbsolute']:.9g} | {row['sameRunner']['maxAbsolute']:.9g} | "
                     f"{row['historical']['worstIndex']} | {'pass' if row['passed'] else 'FAIL'} |")
    lines += ['', 'Discrete graph structure: ' + ('pass' if report['comparison']['structureMatches'] else 'FAIL'),
              'The coefficient control uses the retained public-API snapshot; all graph/layout controls use the alpha.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration', required=True, choices=['debug', 'release'])
    parser.add_argument('--output', required=True, type=Path, help='new scratch/evidence directory')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile = reference.load_profile()
    report = {'schema': 1, 'passed': False, 'configuration': args.configuration, 'profile': profile['id'],
              'alphaRevision': profile['alphaRevision'], 'acceptedRevision': profile['acceptedRevision']}
    try:
        report['host'] = reference.host_identity()
        report['source'] = reference.source_identity()
        report['profileSHA256'] = hashlib.sha256(reference.PROFILE.read_bytes()).hexdigest()
        report['clientSHA256'] = reference.digest_files(reference.ROOT, [
            'Scripts/ci/fixtures/umap-reference/LegacyCapture.swift', 'Scripts/ci/fixtures/umap-reference/CurveCapture.swift'])
        alpha = checkout_revision(profile['alphaRevision'], output / 'alpha-source/VectorCore')
        accepted = checkout_revision(profile['acceptedRevision'], output / 'accepted-source/VectorCore')
        old = capture(alpha, 'alpha', args.configuration, output, False)
        prior = capture(accepted, 'accepted', args.configuration, output, True)
        current = capture(reference.ROOT, 'candidate', args.configuration, output, True)
        report['providers'] = {key: value['provider'] for key, value in [('alpha', old), ('accepted', prior), ('candidate', current)]}
        if len(set(report['providers'].values())) != 1:
            raise ValueError('reference/candidate provider mismatch')
        report['comparison'] = compare(current, alpha_control(old, prior, profile), profile)
        report['passed'] = report['comparison']['passed']
        if reference.source_identity() != report['source']:
            raise ValueError('source or harness changed during observation')
        (output / 'deviations.md').write_text(markdown(report))
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        report['passed'] = False
        report['error'] = str(error)
    finally:
        (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise SystemExit('UMAP observation failed: ' + report.get('error', 'a route missed its fixed budget; see deviations.md'))
    print('All 35 measured routes within predeclared budgets; captures repeated twice.', flush=True)


if __name__ == '__main__':
    main()
