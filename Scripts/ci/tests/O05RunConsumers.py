#!/usr/bin/env python3
"""Bounded independent public Swift 6 migration fixtures and immutable evidence."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / 'Scripts/ci'))
import swiftpm_products as products
# argv[1] is the SwiftPM bin path (swift build -c release --show-bin-path); native and swiftbuild layouts both work.
build = Path(sys.argv[1])
output = Path(sys.argv[2])
mode = sys.argv[3]
assert mode in ['baseline', 'final']
output.mkdir(parents=True, exist_ok=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
snap = products.snapshot_consumer_inputs(build, output)
objects = snap['objects']
artifacts = [dst for _, dst in snap['records']] + [snap['modulemap']]
assert objects
fixtures = sorted((root / 'Scripts/ci/tests/O05').glob('*.swift'))
for source in fixtures + [Path(__file__)]:
    (output / source.name).write_bytes(source.read_bytes())
record = {'mode': mode, 'sourceHashes': {str(p.relative_to(root)): sha(p) for p in root.glob('Sources/**/*.swift')},
          'fixtureHashes': {p.name: sha(p) for p in fixtures + [Path(__file__)]},
          'artifacts': {str(p.relative_to(output)): sha(p) for p in artifacts}, 'runs': []}
active = None

def terminate(signum, frame):
    if active is not None and active.poll() is None:
        os.killpg(active.pid, signal.SIGKILL)
        try:
            active.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    raise SystemExit(128 + signum)

signal.signal(signal.SIGTERM, terminate)
signal.signal(signal.SIGINT, terminate)

def run(name, command, expected, diagnostic=None, marker=None):
    global active
    path = output / (name + '.log')
    timed_out = False
    cleanup_timed_out = False
    with path.open('wb') as log:
        active = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        pid = active.pid
        try:
            status = active.wait(timeout=15)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(pid, signal.SIGKILL)
            try:
                status = active.wait(timeout=2)
            except subprocess.TimeoutExpired:
                status = None
                cleanup_timed_out = True
    active = None
    text = path.read_text()
    passed = not timed_out and status == expected and (diagnostic is None or diagnostic in text) and (marker is None or marker in text)
    record['runs'].append({'name': name, 'command': command, 'exit': status, 'expectedExit': expected,
                           'diagnostic': diagnostic, 'marker': marker, 'passed': passed, 'pid': pid,
                           'timedOut': timed_out, 'cleanupTimedOut': cleanup_timed_out,
                           'executionDeadlineSeconds': 15, 'cleanupDeadlineSeconds': 2,
                           'logSHA256': sha(path)})
    record['executables'] = {p.name: sha(p) for p in output.glob('scoped-*') if p.is_file() and p.suffix == ''}
    (output / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    print(name, 'exit', status, 'passed', passed, flush=True)
    assert passed, name

includes = ['-I', str(snap['modules']), '-Xcc', '-fmodule-map-file=' + str(snap['modulemap']),
            '-Xcc', '-I' + str(root / 'Sources/VectorCoreC/include')]
for width in [512, 768, 1536]:
    legacy = output / f'Legacy{width}.swift'
    scoped = output / f'Scoped{width}.swift'
    run(f'legacy-{width}', ['swiftc', '-swift-version', '6', '-typecheck', *includes, str(legacy)],
        0 if mode == 'baseline' else 1, None if mode == 'baseline' else "has no member 'fp16Storage'")
    run(f'new-typecheck-{width}', ['swiftc', '-swift-version', '6', '-typecheck', *includes, str(scoped)],
        1 if mode == 'baseline' else 0, "has no member 'withUnsafeFP16BufferPointer'" if mode == 'baseline' else None)
    if mode == 'final':
        executable = output / f'scoped-{width}'
        run(f'new-link-{width}', ['swiftc', '-O', '-swift-version', '6', *includes, str(scoped),
            *map(str, objects), '-o', str(executable)], 0)
        run(f'new-run-{width}', [str(executable)], 0, marker=f'O05_SCOPED_{width}_COMPLETE')
if mode == 'final':
    forbidden = re.compile(r'\b(?:struct|class|enum)\s+(?:StorageSlice|MoveOnlyStorage)\b|\bMOVE_ONLY_EXPERIMENTAL\b|\bpublic\s+var\s+fp16Storage\b|\.fp16Storage\b|->\s*StorageSlice\b')
    policy_cases = [
        ("local let", "let fp16Storage = values; return fp16Storage", False),
        ("local var", "var fp16Storage = values; fp16Storage.append(0)", False),
        ("removed public getter", "public var fp16Storage: UnsafePointer<UInt16>? { nil }", True),
        ("removed member access", "let pointer = vector.fp16Storage", True),
        ("obsolete slice declaration", "internal struct StorageSlice<Element> {}", True),
        ("obsolete slice helper", "func slice() -> StorageSlice<Float> {}", True),
        ("obsolete move-only declaration", "internal struct MoveOnlyStorage<Element>: ~Copyable {}", True),
        ("obsolete compilation gate", "#if MOVE_ONLY_EXPERIMENTAL", True),
    ]
    record['scannerPolicyCases'] = []
    for name, source, expected in policy_cases:
        actual = forbidden.search(source) is not None
        record['scannerPolicyCases'].append({'name': name, 'source': source,
                                            'expectedMatch': expected, 'actualMatch': actual})
        assert actual == expected, name
    matches = [str(p.relative_to(root)) for p in (root / 'Sources').rglob('*.swift') if forbidden.search(p.read_text())]
    record['obsoleteSourceMatches'] = matches
    (output / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    assert not matches, matches
print('O05_CONSUMERS_COMPLETE', flush=True)
