#!/usr/bin/env python3
"""Run mandatory paired UMAP development checks; see Docs/UMAP_Reference_Checks.md."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

import umap_reference as reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new private evidence directory')
    parser.add_argument('--scratch-path', type=Path, help='optional isolated SwiftPM build cache')
    parser.add_argument('--profile', type=Path, default=reference.PROFILE, help='reviewed, versioned profile')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'schema': 1, 'status': 'failed', 'configurations': {}}
    try:
        profile = reference.load_profile(args.profile)
        result['profile'] = profile['id']
        result['profileSHA256'] = hashlib.sha256(args.profile.read_bytes()).hexdigest()
        result['host'] = reference.host_identity()
        reference.require_host(result['host'], profile)
        result['source'] = reference.source_identity()
        if result['source']['dirty']:
            raise ValueError('commit the candidate before recording mandatory exact-reference evidence')
        swift = reference.command_text(['xcrun', '--find', 'swift'])
        for configuration in ('debug', 'release'):
            receipts = args.output / configuration
            receipts.mkdir()
            context = {'runID': str(uuid.uuid4()), 'configuration': configuration,
                       'profile': profile['id'], 'fixtureSHA256': result['source']['fixtureSHA256']}
            argv = [swift, 'test', '-c', configuration, '--filter', reference.FILTER]
            if args.scratch_path:
                argv += ['--scratch-path', str(args.scratch_path.resolve())]
            print(f'Running exact UMAP references: {configuration}', flush=True)
            exit_code = reference.bounded_run(argv, args.output / (configuration + '.log'),
                                             env=reference.test_environment(os.environ, context, receipts))
            cell = dict(context, command=argv, exitCode=exit_code)
            result['configurations'][configuration] = cell
            cell.update(reference.validate_receipts(receipts, profile, context, exit_code))
            print(f"{configuration}: {cell['cases']} exact cases / {cell['words']} words", flush=True)
        if reference.source_identity() != result['source']:
            raise ValueError('source or harness changed during reference execution')
        result['status'] = 'passed'
    except (ValueError, OSError, KeyError) as error:
        result['error'] = str(error)
    finally:
        (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    if result['status'] != 'passed':
        raise SystemExit('UMAP reference check failed: ' + result.get('error', 'see retained logs'))


if __name__ == '__main__':
    main()
