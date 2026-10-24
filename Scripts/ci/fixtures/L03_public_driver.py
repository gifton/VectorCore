"""Bounded Swift6 external UMAP numerical-boundary controls; no @testable import.

Works under both SwiftPM layouts (native and swiftbuild) through Scripts/ci/swiftpm_products.py.
"""
import argparse
import hashlib
import json
import signal
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import swiftpm_products as products  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv, output, name, expect_success=True):
    log = output / (name + '.log')
    # Install before launch. File capture cannot fill an unread pipe.
    previous = signal.signal(signal.SIGCHLD, lambda *_: None)
    record = {'argv': list(map(str, argv)), 'cwd': str(Path.cwd()), 'deadline_seconds': 15,
              'cleanup_seconds': 2, 'log': str(log)}
    try:
        with log.open('xb') as capture:
            child = subprocess.Popen(argv, stdout=capture, stderr=subprocess.STDOUT)
            record['pid'] = child.pid
            try:
                status = child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                record['timed_out'] = True
                child.terminate()
                try:
                    child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    child.kill()
                    try:
                        child.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        record['cleanup_timed_out'] = True
                raise RuntimeError(name + ': deadline exceeded')
        record['returncode'] = status
        if (status == 0) != expect_success:
            raise RuntimeError(name + ': unexpected exit ' + str(status))
    finally:
        signal.signal(signal.SIGCHLD, previous)
        record['log_sha256'] = sha(log)
        (output / (name + '.json')).write_text(json.dumps(record, indent=2) + '\n')
    return log.read_text()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bin-path', '--artifacts', dest='bin_path', type=Path, required=True,
                   help='SwiftPM bin path (swift build --show-bin-path), or a retained native copy of it')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    bin_path = a.bin_path.resolve()
    fixture = Path(__file__).resolve().parent
    root = fixture.parents[2]
    objects = products.link_objects(bin_path, ['VectorCore', 'VectorCoreC'])
    assert any('vc_lapack' in x.name for x in products.source_objects(bin_path, 'VectorCoreC'))
    modules = products.module_search_path(bin_path)
    modulemap = products.module_map(bin_path, 'VectorCoreC')
    common = ['/usr/bin/swiftc', '-swift-version', '6', '-strict-concurrency=complete',
              '-I', str(modules), '-Xcc', '-fmodule-map-file=' + str(modulemap),
              '-Xcc', '-I' + str(root / 'Sources/VectorCoreC/include')]
    executable = a.output / 'L03PublicConsumer'
    run(common + ['-parse-as-library', str(fixture / 'L03PublicConsumer.swift'),
                  *map(str, objects), '-framework', 'Accelerate', '-o', str(executable)], a.output, 'positive-compile')
    result = run([str(executable)], a.output, 'positive-run')
    assert 'L03_PUBLIC_COMPLETE' in result
    files = objects + [modulemap]
    for entry in sorted(modules.glob('VectorCore.*swiftmodule*')):
        files += sorted(x for x in entry.rglob('*') if x.is_file()) if entry.is_dir() else [entry]
    files += [fixture / 'L03PublicConsumer.swift', Path(__file__).resolve(), executable]
    (a.output / 'inputs.json').write_text(json.dumps({str(x): sha(x) for x in files}, indent=2) + '\n')
    print('L03_PUBLIC_DRIVER_PASS Swift 6 compile/link/run')

if __name__ == '__main__':
    main()
