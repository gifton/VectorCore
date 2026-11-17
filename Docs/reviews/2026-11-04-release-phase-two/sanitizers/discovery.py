"""Typed discovery for this qualification's ASCII Swift/XCTest selector records.

DiscoveryControls covers top-level/nested entries, the exact observed warning
pair, malformed/duplicate/ambiguous records, truncation and both partitions.
Recognizing a warning retains it; it does not establish runtime reliability.
"""
import re

PARSER_VERSION = 1
_IDENT = r'[A-Za-z_][A-Za-z_0-9]*'
_OWNER = _IDENT + r'(?:\.' + _IDENT + r')+'
_ARGS = r'\((?:' + _IDENT + r':)*\)'
_XCTEST = re.compile(_OWNER + '/' + _IDENT)
_SWIFT = re.compile(r'(?:' + _IDENT + r'\.' + _IDENT + _ARGS + r'|' +
                    _OWNER + r'(?:/' + _IDENT + r')*/' + _IDENT + _ARGS + r')')
_WARNING = re.compile(r'==([1-9][0-9]*)==WARN: (.+)')
_KNOWN = {
    'Invalid dyld module map detected. This is most likely a bug in the sanitizer.': 'invalid-dyld-module-map',
    'Backtraces may be unreliable.': 'unreliable-backtraces',
}


class DiscoveryError(ValueError):
    """Discovery is empty, incomplete, untyped, duplicated or inconsistent."""


def _warning_record(line, source, number):
    match = _WARNING.fullmatch(line)
    if match and match[2] in _KNOWN:
        return {'source': source, 'line': number, 'text': line, 'pid': int(match[1]), 'kind': _KNOWN[match[2]]}
    return None


def known_warning_records(text, *, source):
    """Index only the exact pair in retained stderr; never construct selectors.

    Unknown stdout is rejected by parse_discovery. Original stderr is retained
    verbatim; this index is not a generic warning suppression or acceptance rule.
    """
    return [record for number, line in enumerate(text.splitlines(), 1)
            if (record := _warning_record(line, source, number)) is not None]


def parse_discovery(text, *, source):
    if not text or not text.endswith('\n'):
        raise DiscoveryError(source + ': empty or unterminated discovery')
    groups = {'xctest': set(), 'swiftTesting': set()}
    diagnostics = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line:
            continue
        warning = _warning_record(line, source, number)
        if warning is not None:
            diagnostics.append(warning)
            continue
        kind = 'xctest' if _XCTEST.fullmatch(line) else 'swiftTesting' if _SWIFT.fullmatch(line) else None
        if kind is None:
            # Do not echo arbitrary untrusted stdout into the error channel.
            raise DiscoveryError(f'{source}:{number}: unrecognized discovery record')
        if line in groups[kind]:
            raise DiscoveryError(f'{source}:{number}: duplicate selector')
        groups[kind].add(line)
    if not any(groups.values()):
        raise DiscoveryError(source + ': no selector records')
    # The same no-argument member cannot ambiguously name both frameworks.
    if groups['xctest'] & {s[:-2] for s in groups['swiftTesting'] if s.endswith('()')}:
        raise DiscoveryError(source + ': ambiguous dual-framework selector')
    return {**{kind: sorted(values) for kind, values in groups.items()}, 'diagnostics': diagnostics}


def partition_discovery(full_text, helper_text, *, full_source='swift-list.stdout', helper_source='testing-list.stdout'):
    full = parse_discovery(full_text, source=full_source)
    helper = parse_discovery(helper_text, source=helper_source)
    if helper['xctest'] or full['swiftTesting'] != helper['swiftTesting'] or not full['xctest']:
        raise DiscoveryError('Missing or inconsistent framework inventories')
    return {'xctest': full['xctest'], 'swiftTesting': helper['swiftTesting'],
            'diagnostics': full['diagnostics'] + helper['diagnostics']}


def xctest_command(host, bundle, selectors):
    values = list(selectors)
    if not values or len(set(values)) != len(values) or any(not _XCTEST.fullmatch(s) for s in values):
        raise DiscoveryError('XCTest invocation requires unique nonempty typed XCTest selectors')
    return [str(host), '-XCTest', ','.join(sorted(values)), str(bundle)]
