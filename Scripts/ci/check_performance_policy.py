#!/usr/bin/env python3
"""Check the explicit clock-assertion inventory; this is not Swift dataflow analysis."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def compact(text):
    return re.sub(r'\s+', '', text)


def argument(text):
    """First balanced Swift argument in the deliberately narrow inventoried syntax."""
    depth, quoted, escaped = 0, False, False
    for i, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == '\\' and quoted:
            escaped = True
            continue
        if char == '"':
            quoted = not quoted
        elif not quoted:
            if char in '([{':
                depth += 1
            elif char in ')]}':
                if depth == 0:
                    return text[:i].strip()
                depth -= 1
            elif char == ',' and depth == 0:
                return text[:i].strip()
    raise ValueError('Unclosed inventoried argument')


def validate(root=ROOT):
    data = json.loads((root / 'Scripts/ci/performance_sites.json').read_text())
    expected = {row['id']: row for row in data['assertions']}
    if len(expected) != len(data['assertions']):
        raise ValueError('Duplicate assertion inventory ID')
    for row in data['assertions']:
        if re.fullmatch(r'!?\s*[A-Za-z_][A-Za-z0-9_]*', row['expression']):
            raise ValueError('Inventory cannot bless a precomputed timing Boolean: ' + row['id'])
    seen = []
    for path in sorted((root / 'Tests').rglob('*.swift')):
        text = path.read_text()
        for match in re.finditer(r'\b(expectPerformance|expectXCTestPerformance)\s*\(\s*"([^"]+)"\s*,', text):
            helper, site = match.groups()
            if site not in expected:
                raise ValueError('Unknown performance site: ' + site)
            row = expected[site]
            if row['file'] != str(path.relative_to(root)):
                raise ValueError('Wrong performance-site owner: ' + site)
            if helper != ('expectXCTestPerformance' if row['framework'] == 'XCTest' else 'expectPerformance'):
                raise ValueError('Wrong framework helper: ' + site)
            if compact(argument(text[match.end():])) != compact(row['expression']):
                raise ValueError('Changed or pre-evaluated performance condition: ' + site)
            if 'entryCondition' in row:
                prefix = text[:match.start()]
                entry = re.search(r'if\s+([^{}]+)\{\s*$', prefix)
                if (not entry or compact(entry.group(1)) != compact(row['entryCondition'])
                        or not compact(entry.group(1)).startswith('strictPerfGatesEnabled&&')):
                    raise ValueError('Clock-derived assertion entry must gate strict mode first: ' + site)
            seen.append(site)
        owned = [row for row in data['assertions'] if row['file'] == str(path.relative_to(root))]
        if owned:
            direct_conditions = {compact(argument(text[match.end():])) for match in
                                 re.finditer(r'(?:#expect|XCTAssert(?:True|GreaterThan|LessThan))\s*\(', text)}
            for row in owned:
                if compact(row['expression']) in direct_conditions:
                    raise ValueError('Direct inventoried timing assertion: ' + row['id'])
    if Counter(seen) != Counter(expected.keys()):
        raise ValueError('Missing or duplicate performance helper call')
    helper = (root / 'Tests/ComprehensiveTests/PerfGate.swift').read_text()
    if 'environment["VECTORCORE_STRICT_PERF"] == "1"' not in helper:
        raise ValueError('Missing exact-1 gate')
    for name, reporting in [('expectPerformance', 'Issue.record('), ('expectXCTestPerformance', 'XCTFail(')]:
        match = re.search(r'func '+name+r'\((.*?)\n\}', helper, re.S)
        if not match:
            raise ValueError('Missing performance helper: ' + name)
        body = match.group(1)
        if '_ condition: @autoclosure () -> Bool' not in body or 'guard strictPerfGatesEnabled else { return }' not in body:
            raise ValueError('Missing lazy enforcement: ' + name)
        if body.index('guard strictPerfGatesEnabled') > body.index('condition()') or reporting not in body:
            raise ValueError('Incorrect performance enforcement: ' + name)
    expected_measure = {row['id']: row for row in data['measurements']}
    if len(expected_measure) != len(data['measurements']):
        raise ValueError('Duplicate observation inventory ID')
    measured = []
    for path in (root / 'Tests').rglob('*.swift'):
        text = path.read_text()
        relative = str(path.relative_to(root))
        if re.search(r'(?m)^\s*(?:self\.)?measure\s*(?:\(|\{)', text):
            if relative != 'Tests/MinimalTests/PerformanceObservation.swift':
                raise ValueError('Direct XCTest measure outside wrapper: ' + relative)
        for site in re.findall(r'\bobservePerformance\s*\(\s*"([^"]+)"', text):
            if site not in expected_measure or expected_measure[site]['file'] != relative:
                raise ValueError('Unknown observation site: ' + site)
            measured.append(site)
    if Counter(measured) != Counter(expected_measure.keys()):
        raise ValueError('Missing or duplicate observation site')
    wrapper = (root / 'Tests/MinimalTests/PerformanceObservation.swift').read_text()
    if not re.search(r'if ProcessInfo.processInfo.environment\["VECTORCORE_STRICT_PERF"\] == "1" \{\s*measure\(block\)\s*\} else \{', wrapper):
        raise ValueError('Observation wrapper must reserve XCTest measure for exact-1')
    if wrapper.count('block()') != 1:
        raise ValueError('Observation wrapper must call body once when disabled')
    if (root / 'Tests/PerformanceTests/NSWOptimizationBenchmark.swift').exists():
        raise ValueError('Obsolete unregistered NSW test reintroduced')
    return {'assertionSites': len(seen), 'observationSites': len(measured)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    print(json.dumps(validate(parser.parse_args().root), sort_keys=True))
