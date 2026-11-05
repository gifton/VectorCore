#!/usr/bin/env python3
"""Describe full-discovery execution, rejecting partial evidence as qualification.

ScopeControls exercises missing/empty/truncated/failing/skipped output controls.
This standalone phase-two experiment does not alter the mandatory CI driver.
"""
import argparse
import json
from pathlib import Path
import re


def qualifies(scope, command):
    return (scope['accepted'] and command.get('status') == 'completed'
            and command.get('exit') == 0 and command.get('timeout') is False)


def swift_testing(text, discovery=None):
    functions, suites, started, ended, skipped, failed, unknown = set(), set(), set(), set(), set(), set(), set()
    begun = complete = False
    errors = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if row['version'] != 0:
                raise ValueError('unsupported event version')
            payload = row['payload']
            kind = payload['kind']
            if row['kind'] == 'test':
                (functions if kind == 'function' else suites).add(payload['id'])
            elif row['kind'] == 'event':
                ident = payload.get('testID')
                if ident and ident not in functions | suites:
                    unknown.add(ident)
                if kind == 'runStarted': begun = True
                if kind == 'runEnded': complete = True
                if kind == 'testStarted': started.add(ident)
                if kind == 'testEnded':
                    ended.add(ident)
                    if not any(m.get('symbol') == 'pass' for m in payload.get('messages', [])):
                        failed.add(ident)
                if kind == 'testSkipped': skipped.add(ident)
                if kind == 'issueRecorded' and not payload.get('issue', {}).get('isKnown', False):
                    # Unattributed issues remain visible and prevent acceptance.
                    failed.add(ident or '<unattributed issue>')
        except (ValueError, TypeError, KeyError) as error:
            errors.append(str(error))
    inherited_skips = {f for f in functions if any(f == s or f.startswith(s + '/') for s in skipped if s)}
    failed_functions = functions & failed
    passed = (functions & started & ended) - failed_functions - inherited_skips
    pending = functions - passed - failed_functions - inherited_skips
    result = {'framework': 'swift-testing', 'discovered': sorted(functions), 'started': sorted(functions & started),
              'passed': sorted(passed), 'failed': sorted(failed_functions), 'skipped': sorted(inherited_skips),
              'pending': sorted(pending), 'unknown': sorted(unknown), 'issues': sorted(failed), 'errors': errors,
              'runStarted': begun, 'runEnded': complete}
    if discovery is not None:
        normalized = {f.rsplit('/', 1)[0] for f in functions}
        result['discoveryMissing'] = sorted(set(discovery) - normalized)
        result['discoveryUnexpected'] = sorted(normalized - set(discovery))
        if normalized != set(discovery): errors.append('independent discovery differs from event inventory')
    result['accepted'] = bool(functions) and begun and complete and not (pending or failed or inherited_skips or unknown or errors)
    return result


_CASE = re.compile(r"^Test Case '-\[([^ ]+) ([^\]]+)\]' (started|passed|failed|skipped)\b", re.M)
_SUMMARY = re.compile(r'^\s*Executed (\d+) tests?, with (?:(\d+) tests? skipped and )?(\d+) failures?', re.M)


def xctest(text, discovery):
    discovered = set(discovery)
    started, passed, failed, skipped, unknown = set(), set(), set(), set(), set()
    sets = {'started': started, 'passed': passed, 'failed': failed, 'skipped': skipped}
    for owner, method, status in _CASE.findall(text):
        ident = owner + '/' + method
        sets[status].add(ident)
        if ident not in discovered: unknown.add(ident)
    pending = discovered - passed - failed - skipped
    summaries = _SUMMARY.findall(text)
    counts = tuple(map(int, (summaries[-1][0], summaries[-1][1] or 0, summaries[-1][2]))) if summaries else None
    complete = bool(re.search(r"^Test Suite 'All tests' (passed|failed) at ", text, re.M))
    result = {'framework': 'xctest', 'discovered': sorted(discovered), 'started': sorted(started),
              'passed': sorted(passed), 'failed': sorted(failed), 'skipped': sorted(skipped),
              'pending': sorted(pending), 'unknown': sorted(unknown), 'summaryCounts': counts, 'runEnded': complete}
    result['accepted'] = (bool(discovered) and complete and counts == (len(discovered), 0, 0)
                          and passed == discovered == started and not (failed or skipped or unknown or pending))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework', required=True, choices=['swift-testing', 'xctest'])
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--command', type=Path, required=True)
    parser.add_argument('--discovery', type=Path, required=True, help='Independent selectors, one per line')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Refusing to overwrite scope evidence')
    text = args.input.read_text()
    result = swift_testing(text, args.discovery.read_text().splitlines()) if args.framework == 'swift-testing' else xctest(text, args.discovery.read_text().splitlines())
    result['qualified'] = qualifies(result, json.loads(args.command.read_text()))
    result['counts'] = {key: len(result[key]) for key in ['discovered', 'started', 'passed', 'failed', 'skipped', 'pending', 'unknown']}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result['counts'], sort_keys=True), 'qualified=' + str(result['qualified']))


if __name__ == '__main__': main()
