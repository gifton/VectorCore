#!/usr/bin/env python3
"""Bounded, file-backed compile experiments; output directories must be new."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def run(argv, output, label, env, deadline, inputs):
    record = {'argv': list(map(str, argv)), 'cwd': str(output), 'deadlineSeconds': deadline,
              'inputs': {str(p): sha(p) for p in inputs},
              'environment': {k: env[k] for k in ['SWIFT_EXEC', 'PATH', 'CLANG_MODULE_CACHE_PATH']},
              'status': 'starting', 'exit': None, 'timeout': False}
    metadata = output / (label + '.json')
    write(metadata, record)
    started = time.monotonic()
    process = None
    handlers = {}

    def kill():
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def interrupted(signum, frame):
        kill()
        raise KeyboardInterrupt('Interrupted: ' + str(signum))

    try:
        for sig in [signal.SIGINT, signal.SIGTERM]:
            handlers[sig] = signal.signal(sig, interrupted)
        with (output / (label + '.stdout')).open('xb') as out, (output / (label + '.stderr')).open('xb') as err:
            process = subprocess.Popen(record['argv'], cwd=output, env=env, stdout=out, stderr=err,
                                       start_new_session=True)
            record.update(status='running', pid=process.pid)
            write(metadata, record)
            try:
                record.update(exit=process.wait(timeout=deadline), status='completed')
            except subprocess.TimeoutExpired:
                record.update(timeout=True, status='timeout')
                kill()
                try:
                    record['reapedExit'] = process.wait(timeout=2)
                    record['cleanupCompleted'] = True
                except subprocess.TimeoutExpired:
                    record['cleanupCompleted'] = False
    except BaseException as error:
        record.update(status='error', error=str(error))
        kill()
        if process is not None:
            try:
                record['reapedExit'] = process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                record['cleanupCompleted'] = False
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        record['elapsedSeconds'] = time.monotonic() - started
        record['outputs'] = {p.name: sha(p) for p in output.glob(label + '.*') if p != metadata and p.is_file()}
        write(metadata, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--swiftc', type=Path, required=True)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--importer', type=Path, required=True)
    parser.add_argument('--module-dir', type=Path, action='append', default=[])
    parser.add_argument('--library-source', type=Path, action='append', default=[])
    parser.add_argument('--module-name', default='MinimalLibrary')
    parser.add_argument('--witness', choices=['inherited', 'explicit', 'both'], default='both')
    parser.add_argument('--profile', choices=['release', 'basic', 'optimized'], default='release')
    parser.add_argument('--action', choices=['object', 'silgen', 'sil'], default='object')
    parser.add_argument('--extra-flag', action='append', default=[])
    parser.add_argument('--deadline', type=float, default=60)
    parser.add_argument('--allow-failure', action='store_true')
    args = parser.parse_args()
    if args.deadline <= 0:
        parser.error('--deadline must be positive')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args.swiftc = args.swiftc.absolute()
    args.sdk = args.sdk.resolve()
    args.importer = args.importer.resolve()
    args.library_source = [p.resolve() for p in args.library_source]
    dirs = [p.resolve() for p in args.module_dir]
    env = dict(os.environ, SWIFT_EXEC=str(args.swiftc),
               PATH=str(args.swiftc.parent) + ':' + os.environ.get('PATH', ''),
               CLANG_MODULE_CACHE_PATH=str(args.output / 'module-cache'))
    flags = ['-swift-version', '6', '-target', 'arm64-apple-macosx14.0', '-sdk', str(args.sdk),
             '-module-cache-path', env['CLANG_MODULE_CACHE_PATH']]
    if args.profile == 'release':
        flags += ['-O', '-whole-module-optimization', '-enable-testing',
                  '-enable-experimental-feature', 'StrictConcurrency']
    elif args.profile == 'optimized':
        flags += ['-O']
    else:
        flags += ['-Onone']
    flags += args.extra_flag
    snapshots = args.output / 'sources'
    snapshots.mkdir()
    for source in [args.importer, *args.library_source]:
        shutil.copy2(source, snapshots / (sha(source) + '-' + source.name))
    inputs = [args.swiftc.resolve(), args.importer, *args.library_source]
    frontend = args.swiftc.parent / 'swift-frontend'
    if frontend.is_file():
        inputs.append(frontend.resolve())
    settings = args.sdk / 'SDKSettings.json'
    if settings.exists():
        inputs.append(settings)
    version = run([args.swiftc, '--version'], args.output, 'compiler', env, args.deadline, [args.swiftc.resolve()])
    records = [version]
    if args.library_source and version['exit'] == 0:
        records.append(run([args.swiftc, *flags, '-parse-as-library', '-c', '-o', args.output / 'library.o', '-emit-module', '-emit-module-path',
                            args.output / (args.module_name + '.swiftmodule'), '-module-name', args.module_name,
                            *args.library_source], args.output, 'library', env, args.deadline, inputs))
        dirs.insert(0, args.output)
    if all(r['exit'] == 0 for r in records):
        for d in dirs:
            flags += ['-I', str(d)]
            inputs += sorted(d.glob('*.swiftmodule'))
        witnesses = ['inherited', 'explicit'] if args.witness == 'both' else [args.witness]
        for witness in witnesses:
            action = {'object': ['-c', '.o'], 'silgen': ['-emit-silgen', '.sil'], 'sil': ['-emit-sil', '.sil']}[args.action]
            output = args.output / (witness + action[1])
            define = ['-D', 'EXPLICIT_WITNESS'] if witness == 'explicit' else []
            records.append(run([args.swiftc, *flags, '-parse-as-library', '-module-name', 'WitnessProbe',
                                *define, action[0], args.importer, '-o', output], args.output, witness, env,
                               args.deadline, inputs))
    summary = [{k: r.get(k) for k in ['argv', 'status', 'exit', 'timeout', 'elapsedSeconds']} for r in records]
    write(args.output / 'summary.json', summary)
    print(json.dumps({'output': str(args.output), 'results': [(r['status'], r['exit']) for r in records]}))
    return 0 if args.allow_failure or all(r['exit'] == 0 for r in records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
