#!/usr/bin/env python3
"""Collect unqualified CI evidence, or verify a separately reviewed archive.

HostedEvidenceTests cover provenance, tracked profiles, archive extraction and
admission separation. Reviewed profiles are trusted policy data; archived code
is never executed by the verify operation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import tarfile

from qualification_records import read_json, require, sha
from v03_process import run, test_environment, write_json

FRONTEND = '448457da9a8a56741f2c8c4b8931e85b7377fc009c16918cd8ac1a8a3210ea1f'
DEVELOPER = Path('/Applications/Xcode_27.2.app/Contents/Developer')


def full_commit(value):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{40}', value) is not None,
            'candidate/policy revision must be a full lowercase commit SHA')
    return value


def git(root, *args):
    try:
        return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.PIPE, timeout=30)
    except subprocess.CalledProcessError as error:
        raise ValueError('cannot verify Git provenance') from error


def checkout_identity(root, revision):
    require(git(root, 'rev-parse', 'HEAD').decode().strip() == full_commit(revision), 'checkout revision differs')
    require(not git(root, 'status', '--porcelain=v1').strip(), 'checkout must be clean')


def reviewed_profile(policy_root, relative, revision):
    root = Path(policy_root).resolve()
    path = Path(relative)
    require(not path.is_absolute() and '..' not in path.parts and path.suffix == '.json', 'invalid profile path')
    prefix = Path('Scripts/ci/qualification_profiles')
    require(path.is_relative_to(prefix) and len(path.parts) > len(prefix.parts), 'profile outside reviewed directory')
    resolved = root / path
    require(resolved.resolve() == resolved and resolved.is_file(), 'profile missing or symlinked')
    require(git(root, 'rev-parse', 'HEAD').decode().strip() == full_commit(revision), 'policy revision differs')
    require(git(root, 'show', revision + ':' + path.as_posix()) == resolved.read_bytes(), 'profile is not tracked unchanged')
    return resolved


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def extract_reviewed_bundle(archive, destination, review, expected):
    require(type(review) is dict and all(review.get(k) == v for k,v in expected.items()), 'artifact provenance differs')
    require(review.get('sha256') == file_sha(archive), 'archive digest differs')
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        seen = set()
        for member in members:
            path = PurePosixPath(member.name)
            require(not path.is_absolute() and '..' not in path.parts and path.parts
                    and path.as_posix() == member.name.rstrip('/'), 'unsafe archive path')
            require(path.as_posix() not in seen, 'duplicate archive member')
            seen.add(path.as_posix())
            require(member.isfile() or member.isdir(), 'archive links/special files are not permitted')
        require('collection.json' in seen and 'evidence/capture.json' in seen, 'missing collection/capture')
        destination.mkdir(parents=True, exist_ok=False)
        tar.extractall(destination, members=members, filter='data')
    require(read_json(destination/'collection.json') == expected, 'collection metadata differs')
    return destination/'evidence'


def require_accepted(result):
    require(result.get('vectorCoreQualification') == 'accepted'
            and result.get('compilerProfile') == 'known-failing'
            and result.get('releaseAdmitted') is False and result.get('problems') == [],
            'reviewed evidence did not satisfy scoped qualification')


def metadata(args):
    require(re.fullmatch(r'[1-9][0-9]*', args.run_id) is not None, 'invalid evidence run ID')
    require(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repository) is not None, 'invalid repository')
    return dict(runId=args.run_id, repository=args.repository,
        candidateCommit=full_commit(args.candidate_commit), policyCommit=full_commit(args.policy_commit),
        platform=args.platform, name=args.platform+'-evidence')


def collect(args):
    meta = metadata(args)
    checkout_identity(args.source_root, args.candidate_commit)
    checkout_identity(args.policy_root, args.policy_commit)
    require(Path(os.environ.get('DEVELOPER_DIR','')) == DEVELOPER, 'explicit developer directory differs')
    swiftc = DEVELOPER/'Toolchains/XcodeDefault.xctoolchain/usr/bin/swiftc'
    require(sha(swiftc.with_name('swift-frontend')) == FRONTEND, 'frontend fingerprint differs')
    require(platform.machine() == 'arm64', 'host architecture differs')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'collection.json', meta)
    env = test_environment()
    # Collection needs no repository token, including when nested tools inherit env.
    for key in ('GH_TOKEN','GITHUB_TOKEN'):
        env.pop(key, None)
    env.pop('SDKROOT', None)
    receipts = args.output/'commands'
    def command(name, argv, timeout=60):
        return run(argv, receipts, name, cwd=args.source_root, environment=env, timeout=timeout)
    _, xcode = command('xcode-version', [DEVELOPER/'usr/bin/xcodebuild','-version'])
    require(xcode.strip() == 'Xcode 27.2\nBuild version 27B52', 'Xcode build differs')
    target = args.output/'evidence'
    if args.platform == 'native':
        argv = [sys.executable,'-B',args.source_root/'Scripts/ci/capture_compiler_qualification.py',
            '--source-root',args.source_root,'--swiftc',swiftc,'--frontend-sha256',FRONTEND,
            '--sdk',DEVELOPER/'Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk',
            '--output',target,'--jobs','4']
        command('collect',argv,7200)
    else:
        require(args.simulator_runtime and re.fullmatch(r'com\.apple\.CoreSimulator\.SimRuntime\.iOS-[0-9-]+', args.simulator_runtime),
                'explicit iOS simulator runtime required')
        _, inventory = command('runtime-inventory', ['/usr/bin/xcrun','simctl','list','runtimes','-j'])
        runtimes = json.loads(inventory)['runtimes']
        require(any(r['identifier']==args.simulator_runtime and r['isAvailable'] for r in runtimes), 'requested runtime unavailable')
        _, created = command('create-simulator', ['/usr/bin/xcrun','simctl','create',
            'VectorCore-qualification-'+args.run_id,'com.apple.CoreSimulator.SimDeviceType.iPhone-16-Pro-Max',args.simulator_runtime])
        device = created.strip()
        require(re.fullmatch(r'[0-9A-Fa-f-]{36}',device) is not None, 'invalid created simulator ID')
        try:
            command('boot-simulator',['/usr/bin/xcrun','simctl','boot',device])
            command('boot-ready',['/usr/bin/xcrun','simctl','bootstatus',device,'-b'],180)
            command('collect',[sys.executable,'-B',args.source_root/'Scripts/ci/qualification_ios.py',
                '--source-root',args.source_root,'--developer-dir',DEVELOPER,
                '--destination','platform=iOS Simulator,id='+device,'--matrix','--output',target],7200)
        finally:
            command('shutdown-simulator',['/usr/bin/xcrun','simctl','shutdown',device])
            command('delete-simulator',['/usr/bin/xcrun','simctl','delete',device])
    checkout_identity(args.source_root,args.candidate_commit)
    require(read_json(target/'capture.json')['collectionComplete'] is True,'collection incomplete')
    print('collectionComplete=true; qualification=unassessed; releaseAdmitted=false')


def bundle(args):
    require(args.output.is_dir(), 'no collected evidence to retain')
    args.upload.mkdir(parents=True, exist_ok=False)
    archive = args.upload/'evidence.tar.gz'
    # Dereference build-product aliases so extraction never creates links.
    with tarfile.open(archive, 'w:gz', dereference=True) as tar:
        for child in sorted(args.output.iterdir()):
            tar.add(child, arcname=child.name)
    write_json(args.upload/'archive.json', dict(sha256=file_sha(archive),bytes=archive.stat().st_size,
        vectorCoreQualification='unassessed',releaseAdmitted=False))


def verify(args):
    from compiler_qualification import check
    checkout_identity(args.source_root,args.candidate_commit)
    checkout_identity(args.policy_root,args.policy_commit)
    profile = reviewed_profile(args.policy_root,args.profile,args.policy_commit)
    reviewed = read_json(profile)
    artifact = reviewed.get('reviewedArtifact',{})
    expected = metadata(args)
    # The trusted profile identifies the earlier collection policy revision;
    # args.policy_commit identifies the current verifier/profile revision.
    expected['policyCommit'] = full_commit(artifact.get('policyCommit'))
    evidence = extract_reviewed_bundle(args.archive,args.output,artifact,expected)
    require(read_json(evidence/'capture.json').get('sourceCommit') == args.candidate_commit,'captured candidate differs')
    report = check(profile,evidence,args.source_root)
    report['hostedProvenance'] = dict(expected,reviewedPolicyCommit=args.policy_commit,
        reviewedProfileSHA256=sha(profile),archiveSHA256=artifact['sha256'])
    write_json(args.output/'verification.json',report)
    require_accepted(report)
    print('vectorCoreQualification=accepted; compilerProfile=known-failing; releaseAdmitted=false')


def main():
    if sys.version_info < (3, 12):
        raise SystemExit('Hosted qualification requires Python 3.12 or later')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation',required=True)
    for name in ('collect','verify'):
        p = sub.add_parser(name)
        for arg in ('source-root','policy-root','output'):
            p.add_argument('--'+arg,type=Path,required=True)
        for arg in ('candidate-commit','policy-commit','run-id','repository'):
            p.add_argument('--'+arg,required=True)
        p.add_argument('--platform',choices=('native','ios-simulator'),required=True)
        if name == 'collect': p.add_argument('--simulator-runtime')
        else:
            p.add_argument('--profile',required=True)
            p.add_argument('--archive',type=Path,required=True)
    p = sub.add_parser('bundle')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--upload',type=Path,required=True)
    args = parser.parse_args()
    {'collect':collect,'verify':verify,'bundle':bundle}[args.operation](args)


if __name__ == '__main__':
    main()
