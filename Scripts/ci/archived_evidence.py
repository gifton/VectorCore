#!/usr/bin/env python3
"""Read exact archived JSON without keeping machine-specific captures in the tree.

ArchivedEvidenceTests covers digest/size checks, immutable Git references, shallow
clones, duplicate keys, and rejection of nested references. Only data is read;
historical Python helpers are never imported or executed.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def parse(data):
    return json.loads(data, object_pairs_hook=unique)


def validate_reference(reference):
    require(set(reference) == {'artifactReference', 'gitCommit', 'gitPath', 'bytes', 'sha256'},
            'invalid archive reference fields')
    require(type(reference['artifactReference']) is int and reference['artifactReference'] == 1,
            'unsupported archive reference version')
    require(type(reference['gitCommit']) is str
            and re.fullmatch(r'[0-9a-f]{40}', reference['gitCommit']), 'invalid archive commit')
    require(type(reference['sha256']) is str
            and re.fullmatch(r'[0-9a-f]{64}', reference['sha256']), 'invalid archive digest')
    require(type(reference['bytes']) is int and reference['bytes'] > 0, 'invalid archive size')
    name = reference['gitPath']
    require(type(name) is str and name and not any(ord(c) < 32 for c in name)
            and '\\' not in name, 'invalid archive path')
    path = PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts and path.as_posix() == name
            and path.suffix == '.json', 'invalid archive path')


def git(directory, *args):
    try:
        return subprocess.check_output(['git', '--no-replace-objects', '--no-lazy-fetch',
                                        '-C', str(directory), *args],
                                       stderr=subprocess.PIPE, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError('archive Git object unavailable; fetch its pinned commit or use the retained original JSON') from error


def read_bytes(path, *, fetch=False):
    path = Path(path)
    data = path.read_bytes()
    reference = parse(data)
    if type(reference) is not dict or 'artifactReference' not in reference:
        return data
    validate_reference(reference)
    spec = reference['gitCommit'] + ':' + reference['gitPath']
    try:
        data = git(path.parent, 'cat-file', 'blob', spec)
    except ValueError:
        if not fetch:
            raise
        # Explicit opt-in, exact commit only; never check out or execute old code.
        git(path.parent, 'fetch', '--no-tags', '--no-recurse-submodules', '--depth=1',
            '--no-filter', '--refetch',
            'origin', reference['gitCommit'])
        data = git(path.parent, 'cat-file', 'blob', spec)
    require(len(data) == reference['bytes'], 'archived JSON size differs')
    require(hashlib.sha256(data).hexdigest() == reference['sha256'], 'archived JSON digest differs')
    payload = parse(data)
    require(type(payload) is not dict or 'artifactReference' not in payload, 'nested archive reference')
    return data


def read_json(path, *, fetch=False):
    return parse(read_bytes(path, fetch=fetch))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fetch', action='store_true', help='allow fetching the pinned commit from origin')
    args = parser.parse_args()
    try:
        data = read_bytes(args.reference, fetch=args.fetch)
        with args.output.open('xb') as stream:
            stream.write(data)
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + '\n')
    print('Archived JSON verified and materialized; no qualification or release admission performed.')


if __name__ == '__main__':
    main()
