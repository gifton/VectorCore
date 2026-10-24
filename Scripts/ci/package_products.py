#!/usr/bin/env python3
"""Enumerate Package.swift library products and declared platforms (text parse, no toolchain).

CLI: prints one library product name per line, minus any `--exclude NAME`.
Scripts/ci/tests/test_platform_ci_coverage.py uses the same functions to prove the
CI platform job builds every library product on every declared platform.
"""
import argparse
import re
import sys
from pathlib import Path

PLATFORM_CASES = {'macOS': 'macOS', 'iOS': 'iOS', 'tvOS': 'tvOS', 'watchOS': 'watchOS', 'visionOS': 'visionOS',
                  'macCatalyst': 'macCatalyst'}


def _strip_comments(text):
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    return re.sub(r'//[^\n]*', '', text)


def _balanced(text, open_index):
    """Return the text between the bracket at open_index and its match (string-literal aware)."""
    pairs = {'(': ')', '[': ']'}
    stack, in_string, i = [], False, open_index
    while i < len(text):
        c = text[i]
        if in_string:
            if c == '\\':
                i += 1  # skip the escaped character
            elif c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif c in pairs:
            stack.append(pairs[c])
        elif stack and c == stack[-1]:
            stack.pop()
            if not stack:
                return text[open_index + 1:i]
        i += 1
    raise ValueError('unbalanced brackets in Package.swift')


def _section(text, label):
    matches = list(re.finditer(r'(?m)^[ \t]*(?:\w+\s*\(\s*)?' + label + r'\s*:\s*\[', text))
    if len(matches) != 1:
        raise ValueError(f'Package.swift must have exactly one literal {label}: [...] section')
    open_index = matches[0].end() - 1
    body = _balanced(text, open_index)
    if re.match(r'\s*[+\.]', text[open_index + len(body) + 2:]) or re.search(r'\]\s*\+|\+\s*\[', body):
        raise ValueError(f'{label} is not a single literal array (concatenation is not supported)')
    return body


def library_products(package_text):
    body = _section(_strip_comments(package_text), 'products')
    names = re.findall(r'\.library\s*\(\s*name\s*:\s*"([^"]+)"', body)
    declared = len(re.findall(r'\.library\s*\(', body))
    if declared != len(names):
        raise ValueError(f'{declared} .library declarations but {len(names)} literal names parsed')
    if not names:
        raise ValueError('Package.swift declares no library products')
    if len(set(names)) != len(names):
        raise ValueError('duplicate library product names')
    return names


def declared_platforms(package_text):
    body = _section(_strip_comments(package_text), 'platforms')
    names = re.findall(r'\.(\w+)\s*\(\s*\.', body)
    unknown = [n for n in names if n not in PLATFORM_CASES]
    if len(re.findall(r'\.\w+\s*\(\s*\.', body)) != len(re.findall(r'\.\w+\s*\(', body)):
        raise ValueError('platforms contain a non-literal entry')
    if unknown or not names:
        raise ValueError(f'unrecognized or empty platforms: {unknown}')
    return names


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', default=str(Path(__file__).resolve().parents[2] / 'Package.swift'))
    parser.add_argument('--exclude', action='append', default=[])
    args = parser.parse_args(argv)
    products = library_products(Path(args.package).read_text())
    for name in args.exclude:
        if name not in products:
            parser.error(f'--exclude {name} is not a library product')
    print('\n'.join(p for p in products if p not in args.exclude))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
