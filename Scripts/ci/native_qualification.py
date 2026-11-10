"""Data-only native suite and routing assessment; never imports capture helpers."""
import json
import math
import re
import struct
import xml.etree.ElementTree as ET

from qualification_records import option, require, same, unique

CANARY = 'ComprehensiveTests.LostStoreCanaryTests/literalArrayStoreIsObservedAfterOutOfLineClosure()'


def suite_observations(text, xml_text):
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as error:
        raise ValueError('incomplete/malformed native XML') from error
    require(root.tag == 'testsuites', 'native XML root differs')
    swift = {}
    for row in root.findall('.//testcase'):
        key = row.attrib['classname'] + '/' + row.attrib['name']
        require(key not in swift, 'duplicate native test identity: ' + key)
        require(all(child.tag in ('skipped', 'failure', 'error', 'system-out', 'system-err') for child in row),
                'unrecognized native test status')
        issues = [dict(kind=child.tag, message=child.get('message', ''), text=(child.text or '').strip())
                  for child in row if child.tag in ('failure', 'error')]
        skipped = row.findall('skipped')
        require(len(skipped) <= 1 and not (skipped and issues), 'conflicting native test status')
        swift[key] = {'status': 'failed' if issues else 'skipped' if skipped else 'passed', 'issues': issues}
    require(bool(swift), 'empty native XML inventory')
    starts = re.findall(r"Test Case '(-\[[^\n']+\])' started\.", text)
    outcomes = re.findall(r"Test Case '(-\[[^\n']+\])' (passed|failed|skipped) \([^\n]+?\)\.", text)
    xctest = {}
    for key, state in outcomes:
        require(key not in xctest, 'duplicate XCTest outcome')
        xctest[key] = state
    require(bool(starts) and len(starts) == len(set(starts)) and set(starts) == set(xctest),
            'XCTest individual outcomes incomplete')
    totals = [(int(a), int(b or 0), int(c)) for a, b, c in re.findall(
        r'Executed (\d+) tests?, with (?:(\d+) tests? skipped and )?(\d+) failures?', text)]
    require(bool(totals), 'XCTest completion missing')
    require(max(totals) == (len(xctest), sum(v == 'skipped' for v in xctest.values()),
                            sum(v == 'failed' for v in xctest.values())), 'XCTest totals differ from outcomes')
    runs = [(int(n), outcome) for n, outcome in re.findall(
        r'Test run with (\d+) tests? in \d+ suites? (passed|failed) after[^\n]*', text)]
    require(bool(runs), 'Swift Testing completion missing')
    maximum = max(n for n, _ in runs)
    main = [row for row in runs if row[0] == maximum]
    require(main == [(len(swift), 'failed' if any(v['status'] == 'failed' for v in swift.values()) else 'passed')],
            'Swift Testing completion differs from individual outcomes')
    issues = [line[line.index('Test '):].strip() for line in text.splitlines()
              if 'Test ' in line and 'recorded an issue at' in line]
    require(not any('recorded a known issue' in line for line in text.splitlines()), 'unreviewed known-issue outcome')
    return {'xctest': xctest, 'swiftTesting': swift, 'issues': sorted(issues)}


def assess_suite(actual, expected, code):
    require(same(actual, expected), 'native individual outcome profile differs')
    require(all(value in ('passed', 'skipped') for value in actual['xctest'].values()), 'XCTest failed')
    failed = {key for key, value in actual['swiftTesting'].items() if value['status'] == 'failed'}
    require(failed <= {CANARY}, 'supported native test failed')
    issue_count = sum(len(value['issues']) for value in actual['swiftTesting'].values())
    require(len(actual['issues']) == issue_count == len(failed), 'additional or missing native issue')
    if failed:
        require('literalArrayStoreIsObservedAfterOutOfLineClosure() recorded an issue at LostStoreCanaryTests.swift:47:9:'
                in actual['issues'][0], 'native diagnostic identity differs')
    require(type(code) is int and code == int(bool(failed)), 'native process exit differs')


def route_references():
    """Scalar integer sums for the unchanged V02 fixture, with literal calibration.

    Unit/equal uses scale 1, unequal uses 1 and 2; asymmetric [-30,50]
    uses Float32(80/255), zero point -32; [0,255] uses 1 and -128.
    NativeAssessmentTests checks integer-dot agreement and rejection boundaries.
    """
    asym_scale = struct.unpack('f', struct.pack('f', 80 / 255))[0]
    calibrations = {'equal': (1.0, 1.0, 0), 'unequal': (1.0, 2.0, 0),
                    'asymmetric': (asym_scale, asym_scale, -32), 'nonzero-zero-point': (1.0, 1.0, -128)}
    result = {}
    for dimension in (512, 768, 1536):
        for mode, (qs, cs, zero) in calibrations.items():
            q = [i % 15 - 7 - zero for i in range(dimension)]
            c = [(i * 5 + 3) % 19 - 9 - zero for i in range(dimension)]
            dot = sum(a * b for a, b in zip(q, c))
            normq, normc = sum(a * a for a in q), sum(b * b for b in c)
            result[dimension, mode, 'dot'] = dot * qs * cs
            result[dimension, mode, 'euclidean'] = math.sqrt(sum((a * qs - b * cs) ** 2 for a, b in zip(q, c)))
            result[dimension, mode, 'cosine'] = 1 - max(-1, min(1, dot / (math.sqrt(normq) * math.sqrt(normc))))
    return result


def assess_routes(text, enabled):
    require(type(enabled) is bool, 'C route feature flag type')
    lines = text.splitlines()
    require(len(lines) == 48 and lines[0] == 'V02_ROUTE_ENTER'
            and lines[-1] == 'V02_ROUTE_COMPLETE widths=3 routes=36 tails=10', 'incomplete C route observation')
    refs, found = route_references(), set()
    for line in lines[1:37]:
        require(line.startswith('V02_ROUTE '), 'missing route row')
        row = json.loads(line[len('V02_ROUTE '):], object_pairs_hook=unique)
        require(set(row) == {'dimension', 'mode', 'metric', 'calls', 'expectedCalls', 'actual', 'reference'},
                'C route fields differ')
        key = row['dimension'], row['mode'], row['metric']
        require(type(row['dimension']) is int and key in refs and key not in found, 'C route inventory differs')
        found.add(key)
        reference = refs[key]
        for field in ('actual', 'reference'):
            require(type(row[field]) in (int, float) and math.isfinite(row[field]), 'nonfinite C route value')
        require(abs(row['reference'] - reference) <= max(1e-12, abs(reference) * 1e-12), 'C scalar reference differs')
        require(abs(row['actual'] - reference) <= max(1e-5, abs(reference) * 2e-6), 'C route numerical failure')
        calls = 3 if enabled and (key[1] == 'equal' or (key[1] == 'unequal' and key[2] != 'euclidean')) else 0
        require(all(type(row[k]) is int and row[k] == calls for k in ('calls', 'expectedCalls')), 'C forwarding count differs')
    require(found == set(refs), 'C route coverage differs')
    require(lines[37:47] == [f'V02_C_TAIL {n} calls=1' for n in (0, 1, 3, 4, 15, 16, 17, 31, 32, 33)],
            'C tail control inventory differs')


def route_link_binding(link, executed, provenance, mode, enabled):
    wanted = [row['path'] for row in provenance['linkedObjects']]
    actual = [arg for arg in link['argv'] if arg.endswith('.o')]
    require(bool(wanted) and len(wanted) == len(set(wanted)) and actual == wanted, 'C route linked objects differ')
    binary = provenance['consumer']['path']
    require(option(link['argv'], '-o') == binary, 'C route output binary differs')
    require(executed['argv'] == [binary, '1' if enabled else '0'], 'C route executed binary/feature differs')
    require([arg for arg in link['argv'] if arg.startswith('-O')] == ['-O' if mode == 'release' else '-Onone'],
            'C route client optimization differs')
