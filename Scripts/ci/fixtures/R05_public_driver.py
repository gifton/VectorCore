"""R05 bounded Swift6 external relevance-score consumer; no @testable import.

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
    p.add_argument('--bin-path', '--products', dest='bin_path', type=Path, required=True,
                   help='SwiftPM bin path (swift build --show-bin-path): native <arch>/<config> or swiftbuild out/Products/<Config>')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    bin_path = a.bin_path.resolve()
    fixture = Path(__file__).resolve().parent
    root = fixture.parents[2]
    objects = products.link_objects(bin_path, ['VectorCore', 'VectorCoreC'])
    modules = products.module_search_path(bin_path)
    modulemap = products.module_map(bin_path, 'VectorCoreC')
    include = root / 'Sources/VectorCoreC/include'
    common = ['/usr/bin/swiftc', '-swift-version', '6', '-strict-concurrency=complete',
              '-I', str(modules), '-Xcc', '-fmodule-map-file=' + str(modulemap),
              '-Xcc', '-I' + str(include)]
    executable = a.output / 'R05PublicConsumer'
    run(common + ['-parse-as-library', str(fixture / 'R05PublicConsumer.swift'),
                  *map(str, objects), '-framework', 'Accelerate', '-o', str(executable)], a.output, 'positive-compile')
    result = run([str(executable)], a.output, 'positive-run')
    assert 'R05_PUBLIC_COMPLETE' in result
    result = run(common + ['-typecheck', str(fixture / 'R05OldPropertyNegative.swift')],
                 a.output, 'negative-old-property', expect_success=False)
    # Check diagnostic lines only; the compiler also echoes source snippets.
    diagnostics = [x for x in result.splitlines() if ': error:' in x]
    if not any('cannot convert value of type' in x and '(any DistanceMetric<Float>) -> Float?' in x for x in diagnostics):
        raise RuntimeError('negative-old-property: expected method-vs-property type diagnostic, got ' + repr(diagnostics))
    module_files = []
    for entry in sorted(modules.glob('VectorCore.*swiftmodule*')):
        module_files += sorted(x for x in entry.rglob('*') if x.is_file()) if entry.is_dir() else [entry]
    files = objects + [modulemap, *module_files, fixture / 'R05PublicConsumer.swift',
                       fixture / 'R05OldPropertyNegative.swift', Path(__file__).resolve(), executable]
    (a.output / 'inputs.json').write_text(json.dumps({str(x): sha(x) for x in files}, indent=2) + '\n')
    print('R05_PUBLIC_DRIVER_PASS Swift 6 positive compile/run; removed-property negative control')

if __name__ == '__main__':
    main()
