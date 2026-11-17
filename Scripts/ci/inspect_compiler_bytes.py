#!/usr/bin/env python3
"""Record byte-region hashes for thin arm64 Mach-O compilers, without admission."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile


def require(value, message):
    if not value:
        raise ValueError(message)


def sha_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def region(data, start, end, label, padding=False):
    require(0 <= start <= end <= len(data), 'region outside file: ' + label)
    digest = hashlib.sha256()
    zero = True
    for position in range(start, end, 1024 * 1024):
        block = data[position:min(position + 1024 * 1024, end)]
        digest.update(block)
        if padding:
            zero = zero and not any(block)
    result = dict(label=label, offset=start, bytes=end-start, sha256=digest.hexdigest())
    if padding:
        result['allZero'] = zero
    return result


def command(argv):
    env = dict(os.environ)
    for name in ('GH_TOKEN', 'GITHUB_TOKEN'):
        env.pop(name, None)
    result = subprocess.run([str(v) for v in argv], env=env, capture_output=True,
                            text=True, errors='replace', timeout=30, check=False)
    return dict(argv=[str(v) for v in argv], exitCode=result.returncode,
                stdout=result.stdout, stderr=result.stderr)


def inspect(path):
    before = sha_file(path)
    with path.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        require(len(data) >= 32, 'missing Mach-O header')
        magic, cpu, subtype, kind, count, commands_size, flags, reserved = struct.unpack_from('<8I', data)
        require(magic == 0xfeedfacf and cpu == 0x0100000c, 'expected little-endian thin arm64 Mach-O')
        commands_end = 32 + commands_size
        require(commands_end <= len(data) and count <= commands_size // 8, 'invalid load-command bounds')
        position, signatures = 32, []
        for _ in range(count):
            require(position + 8 <= commands_end, 'missing load-command header')
            cmd, size = struct.unpack_from('<II', data, position)
            require(size >= 8 and size % 8 == 0 and position + size <= commands_end,
                    'invalid load-command size')
            if cmd == 0x1d:  # LC_CODE_SIGNATURE, mach-o/loader.h.
                require(size == 16, 'invalid linkedit_data_command size')
                signatures.append(struct.unpack_from('<II', data, position + 8))
            position += size
        require(position == commands_end and len(signatures) == 1, 'expected exactly one code-signature command')
        offset, size = signatures[0]
        end = offset + size
        require(commands_end <= offset and size >= 12 and end <= len(data), 'code signature outside file')
        # CS_SuperBlob and CS_BlobIndex: apple-oss-distributions/xnu, osfmk/kern/cs_blobs.h.
        blob_magic, blob_size, blob_count = struct.unpack_from('>III', data, offset)
        index_end = 12 + 8 * blob_count
        require(blob_magic == 0xfade0cc0 and index_end <= blob_size <= size,
                'invalid signature superblob')
        require(blob_count <= 256, 'unexpected signature component count')
        components = []
        for index in range(blob_count):
            slot, relative = struct.unpack_from('>II', data, offset + 12 + 8 * index)
            require(index_end <= relative and relative + 8 <= blob_size, 'signature component header outside blob')
            child_magic, child_size = struct.unpack_from('>II', data, offset + relative)
            require(child_size >= 8 and relative + child_size <= blob_size, 'signature component outside blob')
            components.append(dict(slot=slot, magic=hex(child_magic), offset=offset+relative,
                                   bytes=child_size))
        require(len({c['slot'] for c in components}) == len(components), 'duplicate signature slots')
        parts = [region(data, offset, offset + index_end, 'superblob-header-and-index')]
        previous = offset + index_end
        cms = []
        for component in sorted(components, key=lambda item: item['offset']):
            start, stop = component['offset'], component['offset'] + component['bytes']
            require(start >= previous, 'overlapping signature components')
            if start > previous:
                parts.append(region(data, previous, start, 'gap-before-slot-' + str(component['slot']), True))
            component.update(region(data, start, stop, 'slot-' + str(component['slot'])))
            component['header'] = region(data, start, start + 8, 'component-header')
            component['payload'] = region(data, start + 8, stop, 'component-payload')
            parts.append(component)
            if component['slot'] == 0x10000:
                require(component['magic'] == '0xfade0b01', 'unexpected CMS wrapper magic')
                require(stop - start <= 1024 * 1024, 'unexpected CMS payload size')
                with tempfile.TemporaryDirectory(prefix='vectorcore-cms-metadata-') as scratch:
                    cms_path = Path(scratch) / 'signature.der'
                    cms_path.write_bytes(data[start + 8:stop])
                    parsed = command(['/usr/bin/openssl', 'asn1parse', '-inform', 'DER', '-i', '-in', cms_path])
                    require(parsed['exitCode'] == 0, 'CMS ASN.1 inspection failed: ' + parsed['stderr'])
                    cms.append(dict(slot=component['slot'], payloadOffset=start+8,
                                    payloadBytes=stop-start-8, asn1=parsed))
            previous = stop
        if previous < offset + blob_size:
            parts.append(region(data, previous, offset + blob_size, 'superblob-tail', True))
        if offset + blob_size < end:
            parts.append(region(data, offset + blob_size, end, 'signature-allocation-padding', True))
        require(sum(p['bytes'] for p in parts) == size, 'signature partition does not cover allocation')
        require(len(cms) == 1, 'expected exactly one CMS component')
        result = dict(path=str(path), resolvedPath=str(path.resolve()), bytes=len(data),
                      sha256=before, machHeader=dict(cpu=cpu, subtype=subtype, kind=kind,
                      loadCommands=count, loadCommandBytes=commands_size, flags=flags, reserved=reserved),
                      headerAndLoadCommands=region(data, 0, commands_end, 'header-and-load-commands'),
                      outsideSignature=[region(data, 0, offset, 'prefix'), region(data, end, len(data), 'suffix')],
                      codeSignature=region(data, offset, end, 'code-signature-allocation'),
                      superblob=dict(magic=hex(blob_magic), bytes=blob_size, count=blob_count),
                      signaturePartitions=parts, cms=cms)
        require(region(data, 0, len(data), 'whole-file')['sha256'] == before, 'open binary changed during inspection')
    require(sha_file(path) == before, 'binary path changed during inspection')
    result['wholeFileHashRechecked'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--developer-dir', type=Path, required=True)
    parser.add_argument('--candidate-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.developer_dir.is_absolute(), 'developer directory must be absolute')
    require(re.fullmatch('[0-9a-f]{40}', args.candidate_commit), 'full candidate SHA required')
    report = dict(schemaVersion=1, purpose='compiler byte-region metadata only',
                  vectorCoreQualification='unassessed', releaseAdmitted=False,
                  requestedCandidateCommit=args.candidate_commit,
                  startedAt=datetime.now(timezone.utc).isoformat(),
                  scriptSHA256=sha_file(Path(__file__)),
                  run={name: os.environ.get(name) for name in (
                      'GITHUB_REPOSITORY', 'GITHUB_SHA', 'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT',
                      'GITHUB_REF', 'ImageOS', 'ImageVersion')}, binaries={}, problems=[])
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write('\n')
    report['opensslVersion'] = command(['/usr/bin/openssl', 'version'])
    for name in ('swift-frontend', 'swift-driver'):
        try:
            path = args.developer_dir / 'Toolchains/XcodeDefault.xctoolchain/usr/bin' / name
            report['binaries'][name] = inspect(path)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            report['problems'].append(name + ': ' + str(error))
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    report['finishedAt'] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(f'Byte-region metadata retained: {args.output}')
    print('vectorCoreQualification=unassessed; releaseAdmitted=false')
    if report['problems']:
        raise SystemExit('; '.join(report['problems']))


if __name__ == '__main__':
    main()
