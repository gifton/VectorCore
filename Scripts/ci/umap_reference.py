"""Small development-reference checks. These are not release qualification."""
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PROFILE = Path(__file__).with_name('umap-reference') / 'xcode27.2-m3max.json'
REFERENCE_FILES = [
    'Tests/ComprehensiveTests/' + name + '.swift' for name in (
        'UMAPReferenceAssertions', 'UMAPProgressFixtures', 'UMAPProgressReferenceTests',
        'KNNMetricTests', 'UMAPPublicCurveTests')]
FILTER = 'UMAPProgressReferenceTests|KNNMetricTests/euclideanPreservesReferenceWords|UMAPPublicCurveTests'


def load_profile(path=PROFILE):
    profile = json.loads(Path(path).read_text())
    if profile['schema'] != 1 or not profile['cases'] or not profile['words']:
        raise ValueError('unsupported or empty profile')
    for case in profile['cases'].values():
        words = profile['words'][case['reference']]
        if not words or any(type(word) is not int or not 0 <= word <= 0xffffffff for word in words):
            raise ValueError('invalid profile words')
    return profile


def command_text(argv, *, cwd=ROOT):
    return subprocess.check_output(argv, cwd=cwd, text=True, stderr=subprocess.STDOUT, timeout=30).strip()


def host_identity():
    return {
        'xcode': command_text(['xcodebuild', '-version']),
        'swift': command_text(['xcrun', 'swift', '--version']),
        'architecture': platform.machine(),
        'macOS': command_text(['sw_vers', '-productVersion']),
        'macOSBuild': command_text(['sw_vers', '-buildVersion']),
        'model': command_text(['sysctl', '-n', 'hw.model']),
        'cpu': command_text(['sysctl', '-n', 'machdep.cpu.brand_string']),
    }


def require_host(actual, profile):
    mismatches = [key for key, value in profile['host'].items() if actual.get(key) != value]
    if mismatches:
        raise ValueError('unreviewed exact-reference host: ' + ', '.join(mismatches))


def digest_files(root, names):
    digest = hashlib.sha256()
    for name in sorted(names):
        data = (root / name).read_bytes()
        digest.update(name.encode() + b'\0' + len(data).to_bytes(8, 'big') + data)
    return digest.hexdigest()


def source_identity(root=ROOT):
    names = command_text(['git', 'ls-files'], cwd=root).splitlines()
    production = [name for name in names if name.startswith('Sources/') or name == 'Package.swift']
    return {
        'revision': command_text(['git', 'rev-parse', 'HEAD'], cwd=root),
        'dirty': bool(command_text(['git', 'status', '--porcelain'], cwd=root)),
        'productionSHA256': digest_files(root, production),
        'fixtureSHA256': digest_files(root, REFERENCE_FILES),
        'runnerSHA256': digest_files(root, ['Scripts/ci/umap_reference.py', 'Scripts/ci/check_umap_reference.py']),
    }


def test_environment(inherited, context, directory, *, exact=True, profile=None):
    # Start from ordinary process prerequisites, not arbitrary test/provider controls.
    keep = {'PATH', 'HOME', 'TMPDIR', 'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'DEVELOPER_DIR', 'SDKROOT'}
    env = {key: value for key, value in inherited.items() if key in keep}
    env.update(VECTORCORE_TEST_EXTENDED='0', VECTORCORE_STRICT_PERF='0', SWIFT_TESTING_ENABLED='1')
    if exact:
        env['VECTORCORE_UMAP_CHECK_BASELINE_BITS'] = '1'
    child_context = dict(context, directory=str(directory))
    if profile is not None:
        child_context['oracle'] = {name: profile['words'][case['reference']]
                                   for name, case in profile['cases'].items()}
    env['VECTORCORE_UMAP_REFERENCE_CONTEXT'] = json.dumps(child_context)
    return env


def bounded_run(argv, log, *, cwd=ROOT, env=None, timeout=900):
    """Terminate our compiler/test process group at the deadline, retaining its log."""
    with Path(log).open('x') as output:
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
            raise


def validate_receipts(directory, profile, context, exit_code):
    if exit_code != 0:
        raise ValueError(f'Swift test process failed: {exit_code}')
    seen = set()
    word_count = 0
    for path in sorted(Path(directory).iterdir()):
        if path.suffix != '.json' or not path.is_file() or path.is_symlink():
            raise ValueError('malformed receipt path')
        record = json.loads(path.read_text())
        if not isinstance(record, dict):
            raise ValueError('malformed receipt')
        case_id = record.get('caseID')
        if case_id not in profile['cases']:
            raise ValueError(f'unexpected case: {case_id}')
        if case_id in seen:
            raise ValueError(f'duplicate case: {case_id}')
        seen.add(case_id)
        for key, value in dict(context, mode='exact', provider=profile['provider']).items():
            if record.get(key) != value:
                raise ValueError(f'{case_id}: incorrect {key}')
        words = profile['words'][profile['cases'][case_id]['reference']]
        if type(record.get('wordCount')) is not int or record['wordCount'] != len(words):
            raise ValueError(f'{case_id}: incorrect wordCount')
        actual = record.get('words')
        if not isinstance(actual, list) or any(type(word) is not int for word in actual) or actual != words:
            raise ValueError(f'{case_id}: incorrect words')
        word_count += len(words)
    if seen != set(profile['cases']):
        raise ValueError('incomplete comparison inventory: ' + ', '.join(sorted(set(profile['cases']) - seen)))
    return {'cases': len(seen), 'words': word_count}
