#!/usr/bin/env python3
"""Record compiler identity metadata for diagnosis; this does not qualify a build."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import re
import subprocess
import tempfile


LOCAL_FRONTEND = '448457da9a8a56741f2c8c4b8931e85b7377fc009c16918cd8ac1a8a3210ea1f'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def command(argv, env):
    argv = [str(arg) for arg in argv]
    result = {'argv': argv}
    try:
        process = subprocess.run(argv, env=env, capture_output=True, text=True,
                                 errors='replace', timeout=30, check=False)
        result.update(exitCode=process.returncode, stdout=process.stdout,
                      stderr=process.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        result.update(exitCode=None, error=str(error))
    return result


def file_info(path):
    record = {'path': str(path), 'resolvedPath': str(path.resolve())}
    try:
        record.update(bytes=path.stat().st_size, sha256=sha(path))
    except OSError as error:
        record['error'] = str(error)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--developer-dir', type=Path, required=True)
    parser.add_argument('--candidate-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.developer_dir.is_absolute():
        parser.error('developer directory must be absolute')
    if re.fullmatch(r'[0-9a-f]{40}', args.candidate_commit) is None:
        parser.error('requested candidate must be a full lowercase commit SHA')
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / 'inventory.json'
    developer = args.developer_dir
    toolchain = developer / 'Toolchains/XcodeDefault.xctoolchain'
    binaries = toolchain / 'usr/bin'
    env = dict(os.environ, DEVELOPER_DIR=str(developer))
    for name in ('GH_TOKEN', 'GITHUB_TOKEN', 'SDKROOT'):
        env.pop(name, None)
    report = {
        'schemaVersion': 1,
        'purpose': 'compiler identity diagnosis only; no product compilation or tests',
        'vectorCoreQualification': 'unassessed',
        'releaseAdmitted': False,
        'requestedCandidateCommit': args.candidate_commit,
        'startedAt': datetime.now(timezone.utc).isoformat(),
        'host': {'system': platform.system(), 'machine': platform.machine()},
        'developerDirectory': str(developer),
        'resolvedDeveloperDirectory': str(developer.resolve()),
        'script': file_info(Path(__file__)),
        'run': {name: os.environ.get(name) for name in (
            'GITHUB_REPOSITORY', 'GITHUB_SHA', 'GITHUB_REF', 'GITHUB_WORKFLOW_REF',
            'GITHUB_WORKFLOW_SHA', 'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT',
            'RUNNER_OS', 'RUNNER_ARCH', 'ImageOS', 'ImageVersion')},
        'commands': {}, 'files': {}, 'plists': {},
    }

    def save():
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + '\n')

    def record(name, argv):
        report['commands'][name] = result = command(argv, env)
        save()
        return result

    save()
    root = Path(__file__).resolve().parents[2]
    record('checkout-head', ['git', '-C', root, 'rev-parse', 'HEAD'])
    record('checkout-status', ['git', '-C', root, 'status', '--porcelain=v1'])
    record('os-version', ['/usr/bin/sw_vers'])
    record('xcode-version', [developer / 'usr/bin/xcodebuild', '-version'])
    record('sdk-inventory', [developer / 'usr/bin/xcodebuild', '-showsdks'])
    record('runtime-inventory', ['/usr/bin/xcrun', 'simctl', 'list', 'runtimes', '-j'])
    for name in ('swift-frontend', 'swift-driver', 'swiftc', 'swift'):
        path = binaries / name
        report['files'][name] = file_info(path)
        if name != 'swift-driver':
            record(name + '-version', [path, '--version'])
        record(name + '-file', ['/usr/bin/file', '-L', path])
        record(name + '-archs', ['/usr/bin/lipo', '-archs', path])
        record(name + '-uuid', ['/usr/bin/dwarfdump', '--uuid', path])
        record(name + '-signature', ['/usr/bin/codesign', '-d', '--verbose=4', path])
        record(name + '-signature-verification',
               ['/usr/bin/codesign', '--verify', '--strict', '--verbose=2', path])
    for name, path in {
        'xcode-version': developer.parent / 'version.plist',
        'xcode-info': developer.parent / 'Info.plist',
        'toolchain-info': toolchain / 'ToolchainInfo.plist',
    }.items():
        result = file_info(path)
        try:
            with path.open('rb') as stream:
                result['contents'] = plistlib.load(stream)
        except (OSError, plistlib.InvalidFileException, ValueError) as error:
            result['error'] = str(error)
        report['plists'][name] = result
        save()

    frontend = binaries / 'swift-frontend'
    arch_result = report['commands']['swift-frontend-archs']
    arches = arch_result.get('stdout', '').split() if arch_result.get('exitCode') == 0 else []
    slice_hash = None
    if arches == ['arm64']:
        slice_hash = report['files']['swift-frontend'].get('sha256')
    elif 'arm64' in arches:
        with tempfile.TemporaryDirectory(prefix='vectorcore-frontend-slice-') as scratch:
            sliced = Path(scratch) / 'swift-frontend-arm64'
            result = record('frontend-arm64-extraction',
                            ['/usr/bin/lipo', frontend, '-thin', 'arm64', '-output', sliced])
            if result.get('exitCode') == 0:
                slice_hash = sha(sliced)
    report['frontendComparison'] = {
        'localReferenceFullSHA256': LOCAL_FRONTEND,
        'observedFullSHA256': report['files']['swift-frontend'].get('sha256'),
        'observedArm64SliceSHA256': slice_hash,
        'arm64SliceEqualsLocalReference': None if slice_hash is None else slice_hash == LOCAL_FRONTEND,
        'interpretation': 'Hash observations only; no compiler identity is accepted by this inventory.',
    }
    report['commandsWithNonzeroOrMissingExit'] = [
        name for name, result in report['commands'].items() if result.get('exitCode') != 0]
    report['finishedAt'] = datetime.now(timezone.utc).isoformat()
    save()
    print(f'Metadata retained: {report_path}')
    print('vectorCoreQualification=unassessed; releaseAdmitted=false')


if __name__ == '__main__':
    main()
