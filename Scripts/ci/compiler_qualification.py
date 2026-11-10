#!/usr/bin/env python3
"""Assess reviewed, candidate-bound numerical evidence without release admission.

The profile's literal oracles and diagnostic exceptions require human review.
This checker does not discover a safe subset of arbitrary Swift programs.
"""
import argparse
import json
from pathlib import Path
import re
import struct

from qualification_records import POLICY, named, read_json, require, verify_capture


def observations(case, command):
    correct, expected = case['correct'], case['observed']
    require(type(correct) is dict and bool(correct) and type(expected) is dict
            and correct.keys() == expected.keys(), 'oracle inventory differs')
    require(all(type(v) is str and re.fullmatch('[0-9a-f,]+', v)
                for table in (correct, expected) for v in table.values()), 'invalid oracle value')
    text = command['text']
    code = command['record']['exit']
    if case['format'] == 'buffer-value':
        lines = text.splitlines()
        require(bool(lines), 'empty numerical output')
        found = {}
        for line in lines[:-1]:
            match = re.fullmatch(r'BUFFER_VALUE ([a-z0-9.-]+) observed=([0-9a-f,]+)', line)
            require(match is not None, 'malformed numerical row')
            name, value = match.groups()
            require(name not in found, 'duplicate numerical row: ' + name)
            found[name] = value
        require(found == expected, 'numerical observations differ')
        failures = sorted(k for k in correct if found[k] != correct[k])
        require(lines[-1] == f'BUFFER_VALUE_COMPLETE count={len(correct)} failures={len(failures)}',
                'incomplete numerical output')
        require(code == int(bool(failures)), 'numerical exit differs')
    elif case['format'] == 'lost-store':
        require(set(correct) == {'value'} and correct['value'] == '3f800000', 'original probe oracle differs')
        bits = expected['value']
        require(bits in ('3f800000', '40000000'), 'unreviewed original probe value')
        value = struct.unpack('>f', bytes.fromhex(bits))[0]
        require(text == f'LOST_STORE observed={value:.1f} bits={bits} expected=1.0\n' and code == 0,
                'original probe observation differs')
        found = {'value': bits}
        failures = ['value'] if bits != correct['value'] else []
    else:
        raise ValueError('unsupported observation format: ' + str(case['format']))
    return {'observations': found, 'failures': failures, 'numericalCorrect': not failures,
            'role': case['role']}


def check(profile_path: Path, evidence_root: Path, source_root: Path) -> dict:
    report = {'schemaVersion': 1, 'policyId': POLICY, 'supportPolicyAdopted': True,
              'compilerProfile': 'unverified', 'vectorCoreQualification': 'rejected',
              'sourceManifestSHA256': None, 'scope': [], 'problems': [],
              'diagnosticFailures': {}, 'releaseAdmitted': False}
    try:
        profile = read_json(profile_path)
        verified = verify_capture(profile, evidence_root, source_root)
        for item in verified['checks']:
            spec, command = item['spec'], item['command']
            if spec['format'] == 'native-suite':
                from native_qualification import assess_suite, suite_observations
                actual = suite_observations(command['text'], item['xml'])
                assess_suite(actual, spec['expected']['outcomes'], command['record']['exit'])
                require(item['listing'].splitlines() == spec['expected']['testList'], 'native test discovery differs')
            elif spec['format'] == 'migration-contract':
                errors = re.findall(r'^.*?\.swift:(\d+):\d+: error: (.+)$', command['text'], re.M)
                require([list(row) for row in errors] == spec['expected']['errors'], 'migration contract diagnostic differs')
                require(command['record']['exit'] == spec['expected']['exit'], 'migration contract exit differs')
        cases = named(profile['cases'], 'profile cases')
        outcomes = {}
        supported, controls, compiler_failures = set(), set(), set()
        for name, case in cases.items():
            role = case['role']
            require(role in ('supported', 'diagnostic', 'negative-control'), 'unknown observation role')
            require(type(case['compilerProbe']) is bool, 'invalid compiler-probe marker')
            result = observations(case, verified['commands'][case['command']])
            required = case.get('requiredCorrectRows', [])
            require(type(required) is list and all(type(v) is str for v in required)
                    and len(required) == len(set(required)) and set(required) <= set(case['correct']),
                    'invalid required-correct rows')
            require(not (set(required) & set(result['failures'])), 'required supported row failed: ' + name)
            outcomes[name] = result
            if role == 'supported':
                require(not result['failures'], 'supported-route failure: ' + name)
                require(case['compilerProbe'] is False, 'supported route cannot be a diagnostic exception')
                supported.add(name)
            elif role == 'negative-control':
                positive = case['negativeFor']
                require(positive in cases and cases[positive]['role'] in ('supported', 'diagnostic'), 'invalid negative control pair')
                require(case['correct'] == cases[positive]['correct'] and bool(result['failures']),
                        'ineffective negative control')
                require(case['producer'] == cases[positive]['producer'] and case['client'] == cases[positive]['client'],
                        'negative control configuration differs')
                if cases[positive]['role'] == 'supported':
                    controls.add(positive)
            elif result['failures']:
                report['diagnosticFailures'][name] = result['failures']
                if case['compilerProbe']:
                    compiler_failures.add(name)
        require(bool(supported) and supported == controls, 'missing supported checks or negative controls')
        require(bool(compiler_failures), 'no verified known-failing compiler observation')
        report.update(compilerProfile='known-failing', vectorCoreQualification='accepted',
                      sourceManifestSHA256=verified['sourceManifestSHA256'], scope=profile['scope'],
                      verifiedFiles=verified['verifiedFiles'], observations=outcomes,
                      invocationId=verified['capture']['invocationId'])
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
        report['problems'].append(str(error))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('profile', 'evidence-root', 'source-root', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    try:
        with args.output.open('x') as output:
            report = check(args.profile, args.evidence_root, args.source_root)
            json.dump(report, output, indent=2, sort_keys=True)
            output.write('\n')
    except OSError as error:
        parser.exit(2, str(error) + '\n')
    print('vectorCoreQualification=' + report['vectorCoreQualification']
          + '; compilerProfile=' + report['compilerProfile'] + '; releaseAdmitted=false')
    return 0 if report['vectorCoreQualification'] == 'accepted' else 1


if __name__ == '__main__':
    raise SystemExit(main())
