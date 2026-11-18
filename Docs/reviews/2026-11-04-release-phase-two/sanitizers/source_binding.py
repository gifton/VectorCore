"""Fresh-build source association for the phase-two sanitizer experiment.

SourceBindingControls exercises the real producer with a fixture driver. The
association is provenance from this producer, not an authenticity signature or
proof against a source edit that is reverted between snapshots. Source inputs
must stay quiescent for the entire build; pre/post drift prevents admission.
"""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'Scripts/ci'))
from v03_process import run, sha, test_environment, write_json


DRIVER_FLAGS = ['--configuration', 'release', '--swift-build-system', 'native', '--jobs', '2']


def reference(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def capture_source(root, output, name, environment):
    """Capture source files even when gitignored, excluding build/runtime caches.

    test_ignored_source_is_captured_and_generated_bytecode_is_excluded covers the
    distinction. The package's declared source directories are the inventory;
    source-directory symlinks are refused rather than silently left unwalked.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for tree in ['Sources', 'Tests', 'Benchmarks', 'Scripts/ci']:
        for directory, dirs, files in os.walk(root / tree):
            dirs[:] = sorted(d for d in dirs if d not in ['.build', '.swiftpm', '.git', '__pycache__'])
            if any((Path(directory) / d).is_symlink() for d in dirs):
                raise ValueError('Source directory symlinks require an explicit inventory contract')
            paths.extend(str((Path(directory) / f).relative_to(root)) for f in sorted(files))
    paths.sort()
    (output / (name + '-files.txt')).write_text('\n'.join(paths) + '\n')
    inputs = {p: sha(root / p) for p in paths if not p.startswith('Scripts/ci/')}
    drivers = {p: sha(root / p) for p in paths if p.startswith('Scripts/ci/')}
    if not any(p.startswith('Tests/') for p in inputs) or 'Scripts/ci/check_sanitizers.py' not in drivers:
        raise ValueError('Source inventory must contain tests and the mandatory driver')
    identity = {'inputs': inputs, 'driverInputs': drivers, 'packageSHA256': sha(root / 'Package.swift'),
                'producerInputs': {str(p): sha(p) for p in [HERE / 'source_binding.py', HERE / 'build_bound_sanitizers.py']}}
    write_json(output / (name + '-source.json'), identity)
    return identity


def driver_argv(root, scratch, output, sanitizer, python):
    return [str(python), str(Path(root) / 'Scripts/ci/check_sanitizers.py'), '--sanitizer', sanitizer,
            *DRIVER_FLAGS, '--scratch-root', str(scratch), '--output', str(output)]


def read_reference(record):
    path = Path(record['path'])
    if not path.is_absolute() or sha(path) != record['sha256']:
        raise ValueError('Associated record hash mismatch: ' + str(path))
    return path


def check_command(path, root):
    record = json.loads(Path(path).read_text())
    if record.get('status') != 'completed' or record.get('exit') != 0 or record.get('timeout') is not False:
        raise ValueError('Associated command did not complete successfully: ' + str(path))
    if Path(record['cwd']).resolve() != Path(root).resolve():
        raise ValueError('Associated command source root mismatch')
    if sha(record['executable']) != record['executableSHA256']:
        raise ValueError('Associated command executable changed')
    for key in ['stdout', 'stderr']:
        stream = record[key]
        if sha(stream['path']) != stream['sha256'] or Path(stream['path']).stat().st_size != stream['bytes']:
            raise ValueError('Associated command output changed')
    return record


def inside(path, parent):
    if not Path(path).resolve().is_relative_to(Path(parent).resolve()):
        raise ValueError('Associated artifact is outside its fresh build/output root')


def admit_association(association, root, current):
    """Reject stale-source, independently swapped-result and altered-artifact controls."""
    association, root = Path(association).resolve(), Path(root).resolve()
    binding = json.loads(association.read_text())
    if binding['schema'] != 1 or binding['kind'] != 'fresh-mandatory-sanitizer-build':
        raise ValueError('Unsupported build association')
    if Path(binding['sourceRoot']).resolve() != root:
        raise ValueError('Source root differs from associated build')
    output = association.parent
    scratch = Path(binding['scratchRoot']).resolve()
    sanitizer = binding['sanitizer']
    if sanitizer not in ['address', 'thread']:
        raise ValueError('Unsupported associated sanitizer')
    paths = {key: read_reference(value) for key, value in binding['records'].items()}
    required = {'sourceBefore', 'sourceAfter', 'freshness', 'driverCommand', 'buildCommand', 'result', 'artifacts', 'findSwift'}
    if set(paths) != required:
        raise ValueError('Missing or unexpected association record')
    expected_paths = {'sourceBefore': output / 'before-source.json', 'sourceAfter': output / 'after-source.json',
                      'freshness': output / 'freshness.json', 'driverCommand': output / 'mandatory-driver.json',
                      'buildCommand': output / 'mandatory/build.json', 'result': output / 'mandatory/result.json',
                      'artifacts': output / 'mandatory/artifacts.json', 'findSwift': output / 'mandatory/find-swift.stdout'}
    if paths != expected_paths:
        raise ValueError('Source/build association records must belong to one producer output')
    before = json.loads(paths['sourceBefore'].read_text())
    after = json.loads(paths['sourceAfter'].read_text())
    if before != after or before != current:
        raise ValueError('Source identity differs before/after build or current admission')
    source_digest = sha(paths['sourceBefore'])
    scratch_base = Path(binding['scratchBase']).resolve()
    if scratch != scratch_base / source_digest:
        raise ValueError('Source identity does not name the associated source-bound build root')
    freshness = json.loads(paths['freshness'].read_text())
    if freshness != {'scratchRoot': str(scratch), 'scratchBase': str(scratch_base), 'sourceSHA256': source_digest,
                     'mandatoryOutput': str(output / 'mandatory'), 'scratchExisted': False, 'mandatoryOutputExisted': False}:
        raise ValueError('Missing fresh-build admission evidence')
    command = check_command(paths['driverCommand'], root)
    if command['argv'] != driver_argv(root, scratch, output / 'mandatory', sanitizer, command['argv'][0]):
        raise ValueError('Associated mandatory-driver invocation differs from producer contract')
    build = check_command(paths['buildCommand'], root)
    swift = paths['findSwift'].read_text().strip()
    if not Path(swift).is_absolute():
        raise ValueError('Missing selected compiler')
    flags = ['--configuration', 'release', '--jobs', '2', '--scratch-path', str(scratch / sanitizer),
             '--sanitize', sanitizer, '--build-system', 'native', '-Xswiftc', '-enable-testing']
    if build['argv'] != [swift, 'build', '--build-tests', *flags, '--verbose']:
        raise ValueError('Associated build invocation differs from Release sanitizer contract')
    if Path(build['executable']).resolve() != Path(swift).resolve():
        raise ValueError('Associated compiler executable mismatch')
    for key, value in {'ASAN_OPTIONS':'detect_leaks=0:halt_on_error=1', 'TSAN_OPTIONS':'halt_on_error=1',
                       'VECTORCORE_TEST_EXTENDED':'0', 'VECTORCORE_STRICT_PERF':'0'}.items():
        if build['environment'].get(key) != value:
            raise ValueError('Associated build diagnostic environment differs: ' + key)
    data = json.loads(paths['result'].read_text())
    if (data['configuration'], data['buildSystem'], data['sanitizer']) != ('release', 'native', sanitizer):
        raise ValueError('Associated result is not the matching native Release sanitizer build')
    if data['manifestSHA256'] != sha(root / 'Scripts/ci/sanitizer_bodies.json'):
        raise ValueError('Associated mandatory manifest differs')
    rows = json.loads((root / 'Scripts/ci/sanitizer_bodies.json').read_text())['sanitizers'][sanitizer]
    if [r['id'] for r in data['completed']] != [r['id'] for r in rows]:
        raise ValueError('Associated mandatory workloads did not complete exactly')
    inside(data['binPath'], scratch / sanitizer)
    bundles = data['loadedBundles']
    if not bundles: raise ValueError('Associated result contains no bundles')
    artifacts = json.loads(paths['artifacts'].read_text())
    for bundle in bundles.values():
        inside(bundle['path'], data['binPath'])
        if sha(bundle['path']) != bundle['sha256']:
            raise ValueError('Associated bundle changed')
        retained = [r for r in artifacts if r['built'] == bundle['path']]
        if len(retained) != 1: raise ValueError('Associated bundle has no unique retained build artifact')
        artifact = retained[0]
        inside(artifact['retained'], output / 'mandatory/artifacts')
        if artifact['sha256'] != bundle['sha256'] or sha(artifact['retained']) != bundle['sha256']:
            raise ValueError('Associated retained bundle does not match the build result')
    return {'identity': before, 'data': data, 'driverResult': paths['result'], 'sourceIdentity': paths['sourceBefore'],
            'bundle': Path(next(iter(bundles.values()))['path']), 'records': paths}


def produce(root, output, scratch_root, sanitizer, environment=None):
    root, output, scratch = Path(root).resolve(), Path(output).resolve(), Path(scratch_root).resolve()
    if sanitizer not in ['address', 'thread']: raise ValueError('Unsupported sanitizer')
    if output == scratch or output.is_relative_to(scratch) or scratch.is_relative_to(output):
        raise ValueError('Use separate scratch and evidence roots')
    if output.exists() or scratch.exists(): raise ValueError('Use fresh output and scratch roots')
    output.mkdir(parents=True)
    environment = test_environment() if environment is None else dict(environment)
    before = capture_source(root, output, 'before', environment)
    scratch_base = scratch
    source_digest = sha(output / 'before-source.json')
    scratch = scratch_base / source_digest
    write_json(output / 'freshness.json', {'scratchRoot': str(scratch), 'scratchBase': str(scratch_base),
                                          'sourceSHA256': source_digest, 'mandatoryOutput': str(output / 'mandatory'),
                                          'scratchExisted': False, 'mandatoryOutputExisted': False})
    try:
        command, _ = run(driver_argv(root, scratch, output / 'mandatory', sanitizer, sys.executable),
                         output, 'mandatory-driver', cwd=root, environment=environment, timeout=2400, expected=None)
    finally:
        after = capture_source(root, output, 'after', environment)
    if before != after: raise ValueError('Source changed during mandatory driver execution')
    if command.get('exit') != 0 or command.get('timeout'):
        raise ValueError('Mandatory driver did not succeed; no association emitted')
    paths = {'sourceBefore': output / 'before-source.json', 'sourceAfter': output / 'after-source.json',
             'freshness': output / 'freshness.json', 'driverCommand': output / 'mandatory-driver.json',
             'buildCommand': output / 'mandatory/build.json', 'result': output / 'mandatory/result.json',
             'artifacts': output / 'mandatory/artifacts.json', 'findSwift': output / 'mandatory/find-swift.stdout'}
    binding = {'schema': 1, 'kind': 'fresh-mandatory-sanitizer-build', 'sourceRoot': str(root),
               'scratchRoot': str(scratch), 'scratchBase': str(scratch_base), 'sanitizer': sanitizer, 'records': {k: reference(p) for k, p in paths.items()}}
    # Validate the complete association before publishing its accepted filename.
    candidate = output / 'association-candidate.json'
    write_json(candidate, binding)
    admit_association(candidate, root, after)
    candidate.rename(output / 'association.json')
    return output / 'association.json'
