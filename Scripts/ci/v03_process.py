"""File-backed owned-process execution for the V03 contract drivers."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def run(argv, output, name, *, cwd, environment=None, timeout=15, expected=(0,)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    record = output / (name + '.json')
    if record.exists():
        raise ValueError('Refusing to overwrite command evidence: ' + str(record))
    env = dict(os.environ if environment is None else environment)
    command = list(map(str, argv))
    executable = Path(shutil.which(command[0], path=env.get('PATH')) or command[0]).resolve()
    metadata = {'argv': command, 'cwd': str(cwd), 'status': 'starting', 'deadlineSeconds': timeout,
                'cleanupDeadlineSeconds': 2, 'timeout': False,
                'executable': str(executable), 'executableSHA256': sha(executable),
                'environment': {k: v for k, v in env.items() if k.startswith(('VECTORCORE_', 'Q03_', 'V03_', 'ASAN_', 'TSAN_', 'DYLD_'))
                                or k in ['PATH', 'DEVELOPER_DIR', 'SDKROOT', 'SWIFT_EXEC']}}
    write_json(record, metadata)
    stdout, stderr = output / (name + '.stdout'), output / (name + '.stderr')
    process = None
    handlers = {}

    def kill_owned():
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def interrupted(signum, frame):
        kill_owned()
        raise KeyboardInterrupt('Driver interrupted: ' + str(signum))

    for sig in (signal.SIGINT, signal.SIGTERM):
        handlers[sig] = signal.signal(sig, interrupted)
    start = time.monotonic()
    try:
        with stdout.open('xb') as out, stderr.open('xb') as err:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True)
            metadata.update(status='running', pid=process.pid)
            write_json(record, metadata)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                metadata.update(status='timeout', timeout=True, exit=None)
                kill_owned()
                try:
                    process.wait(timeout=2)
                    metadata['cleanupCompleted'] = True
                except subprocess.TimeoutExpired:
                    metadata['cleanupCompleted'] = False
                raise RuntimeError('Execution deadline: ' + name)
            metadata.update(status='completed', exit=code)
            if expected is not None and code not in expected:
                raise RuntimeError(f'{name}: exit {code}, expected {expected}; see {stderr}')
    except BaseException as error:
        metadata['error'] = str(error)
        if process is not None and metadata['status'] == 'running':
            kill_owned()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        metadata['elapsedSeconds'] = time.monotonic() - start
        for name, path in [('stdout', stdout), ('stderr', stderr)]:
            if path.exists():
                metadata[name] = {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
        write_json(record, metadata)
    return metadata, stdout.read_text(errors='replace') + stderr.read_text(errors='replace')


def test_environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith(('VECTORCORE_', 'Q03_', 'V03_', 'ASAN_', 'TSAN_', 'DYLD_'))}
    env.update(VECTORCORE_TEST_EXTENDED='0', VECTORCORE_STRICT_PERF='0')
    return env


def retain_artifacts(build, output):
    records = []
    for path in sorted(Path(build).rglob('*')):
        if path.is_file() and not path.is_symlink() and (path.suffix in ['.o', '.swiftmodule', '.swiftdoc', '.swiftsourceinfo', '.pcm']
                                                        or path.name == 'vectorcore-bench'
                                                        or (path.parent.name == 'MacOS' and path.parent.parent.name == 'Contents'
                                                            and path.parent.parent.parent.name == path.name + '.xctest')):
            copy = Path(output) / path.relative_to(build)
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, copy)
            records.append({'built': str(path), 'retained': str(copy), 'sha256': sha(copy), 'bytes': copy.stat().st_size})
    if not records:
        raise ValueError('No compiled artifacts found')
    return records
