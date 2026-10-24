#!/usr/bin/env python3
"""D02 ordinary Swift 6 consumers (`import VectorCore`, never @testable) and their immutable evidence.
Evidence script, not wired into CI.

usage: D02RunConsumers.py BIN_PATH OUTPUT baseline|final
  BIN_PATH  a SwiftPM bin path of the tree under test (`swift build --show-bin-path`; native or swiftbuild layout)
  OUTPUT    a new directory (must not exist)
  baseline  expectations at the pre-D02 base; final: expectations after D02 (phases 2-5).

Each fixture in Scripts/ci/tests/D02/ has one expectation per mode:
  positive  typecheck rc 0, link against the snapshot objects, run rc 0 and print its marker;
  negative  typecheck rc 1 with exactly one `error:` on every code line of the file (one per removed entry);
  fails     typecheck rc 1 containing a named diagnostic (a positive fixture that cannot compile at the base).
The count guard checks that Reachability.swift calls every symbol of D02/gated-public-base.tsv in code (comments and
strings masked). In final mode the binding check proves, from the SILGen `function_ref` targets, that each of those
calls binds to Vector's own unconstrained member and that no VectorProtocol/IndexableVector fallback is referenced.
Also in final mode, the member backstop extracts the public symbol graph of the build and requires the member set of
VectorProtocol, VectorFactory and IndexableVector to equal the live lexical scan (test_d02_surface_guards.py), the base
TSV (D02/protocol-members-base.tsv) and the D02WitnessDispatchMatrixTests rows; it prints
`member backstop passed graph=N scan=N tsv=N rows=N` (N = 77 today).
Children run in their own process group: compiles 120 s, executables 15 s, cleanup 2 s, then a throw.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / 'Scripts/ci'))
import swiftpm_products as products  # noqa: E402
from swift_lexing import _mask  # noqa: E402  (comments and string literals blanked, same length)

FIXTURES = Path(__file__).resolve().parent / 'D02'
COMPILE_DEADLINE, RUN_DEADLINE, CLEANUP = 120, 15, 2
# fixture -> {mode: (kind, detail)}; detail is the run marker (positive) or a required diagnostic (fails).
TABLE = {
    'Reachability.swift': {'baseline': ('fails', "does not conform to protocol 'StaticDimension'"),
                           'final': ('positive', 'D02_REACHABILITY_COMPLETE')},
    'StoragePositive.swift': {'baseline': ('fails', "does not conform to protocol 'StaticDimension'"),
                              'final': ('positive', 'D02_STORAGE_POSITIVE_COMPLETE')},
    'StorageNegative.swift': {'baseline': ('typechecks', None), 'final': ('negative', None)},
    'DeviceSupport.swift': {'baseline': ('fails', "has no member 'isSupportedByCore'"),
                            'final': ('positive', 'D02_DEVICE_SUPPORT_COMPLETE')},
    'DeviceNegative.swift': {'baseline': ('typechecks', None), 'final': ('negative', None)},
    'Factory.swift': {'baseline': ('positive', 'D02_FACTORY_COMPLETE'),
                      'final': ('positive', 'D02_FACTORY_COMPLETE')},
    # Phase 5: the hook and Configuration removals (H1/H2/H7), the routed removals (RD-R8/RD-R2), and the Logger
    # qualification Logger.swift documents (RD-R5; unchanged by D02, so the same outcome in both modes).
    'HooksNegative.swift': {'baseline': ('typechecks', None), 'final': ('negative', None)},
    'RoutedNegative.swift': {'baseline': ('typechecks', None), 'final': ('negative', None)},
    'LoggerQualification.swift': {'baseline': ('positive', 'D02_LOGGER_QUALIFICATION_COMPLETE'),
                                  'final': ('positive', 'D02_LOGGER_QUALIFICATION_COMPLETE')},
    'LoggerNegative.swift': {'baseline': ('negative', None), 'final': ('negative', None)},
}

active = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def terminate(signum, frame):
    if active is not None and active.poll() is None:
        os.killpg(active.pid, signal.SIGKILL)
        try:
            active.wait(timeout=CLEANUP)
        except subprocess.TimeoutExpired:
            pass
    raise SystemExit(128 + signum)


def gated_symbol_patterns():
    rows = [line.split('\t') for line in (FIXTURES / 'gated-public-base.tsv').read_text().splitlines()
            if line and not line.startswith('#') and not line.startswith('access\t')]
    patterns = {}
    for _, _, path in rows:
        owner, member = path.split('.', 1)
        if member.startswith('.'):                                   # operators: Vector..<(_:_:)
            operator = member[1:].split('(')[0]
            patterns[path] = r'\s' + re.escape('.' + operator) + r'\s'
        elif '(' in member:
            name, labels = member[:-1].split('(', 1)
            first = labels.split(':')[0] if labels else ''
            prefix = r'VectorMath\.' if owner == 'VectorMath' else r'\.'
            if name == 'init':
                patterns[path] = r'\(' + re.escape(first) + ':'
            elif first == '':
                patterns[path] = prefix + re.escape(name) + r'\(\)'
            elif first == '_':
                patterns[path] = prefix + re.escape(name) + r'\((?!\w+:)'
            else:
                patterns[path] = prefix + re.escape(name) + r'\(' + re.escape(first) + ':'
        else:
            patterns[path] = r'\.' + re.escape(member) + r'\b'
    return patterns


def count_guard_missing(source, patterns=None):
    """TSV paths not called in CODE: comments and string literals are masked first (A2-M3), so a mention in a
    comment or a string does not count as a call."""
    patterns = patterns or gated_symbol_patterns()
    code = _mask(source)
    return sorted(path for path, pattern in patterns.items() if not re.search(pattern, code))


def _split_top_level(text):
    parts, depth, current = [], 0, ''
    for ch in text:
        if ch in '([<':
            depth += 1
        elif ch in ')]>':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(current.strip())
            current = ''
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    return parts


def _demangled_member(demangled):
    """(owner, name, labels or None for an accessor) for an unconstrained member of `VectorCore.Vector` or
    `VectorCore.VectorMath`, else None. A constrained extension or a protocol extension demangles with a different
    prefix (`(extension in VectorCore):...`, `VectorCore.VectorProtocol...`) and is rejected here."""
    text = demangled[len('static '):] if demangled.startswith('static ') else demangled
    for owner in ('VectorCore.Vector.', 'VectorCore.VectorMath.'):
        if not text.startswith(owner):
            continue
        rest = text[len(owner):]
        short = owner.split('.')[1]
        operator = re.match(r'(\.[<>]) infix\(', rest)
        if operator:
            return short, operator.group(1), ['_', '_']
        accessor = re.match(r'(\w+)\.(?:getter|setter|modify|unsafeMutableAddressor) :', rest)
        if accessor:
            return short, accessor.group(1), None
        method = re.match(r'(\w+)', rest)
        if not method:
            return None
        name, i = method.group(1), method.end()
        if i < len(rest) and rest[i] == '<':          # generic clause
            depth = 0
            while i < len(rest):
                depth += {'<': 1, '>': -1}.get(rest[i], 0)
                i += 1
                if depth == 0:
                    break
        if i >= len(rest) or rest[i] != '(':
            return None
        depth, j = 0, i
        while j < len(rest):
            depth += {'(': 1, ')': -1}.get(rest[j], 0)
            if depth == 0:
                break
            j += 1
        args = _split_top_level(rest[i + 1:j])
        labels = [m.group(1) if (m := re.match(r'(\w+): ', a)) else '_' for a in args]
        return short, name, labels
    return None


def _tsv_member(path):
    """(owner, name, labels or None) of a gated-public-base.tsv path such as `Vector.range(from:)`."""
    owner, member = path.split('.', 1)
    if member.startswith('.'):
        return owner, member.split('(')[0], ['_', '_']      # `.<` / `.>`, as the demangler spells them
    if '(' not in member:
        return owner, member, None
    name, labels = member[:-1].split('(', 1)
    return owner, name, [label for label in labels.split(':') if label] if labels else []


def binding_failures(demangled_refs, witness_methods=()):
    """Paths of gated-public-base.tsv whose consumer call does not bind to Vector's (or VectorMath's) own,
    unconstrained member, judged from the SILGen `function_ref` targets of the consumer (A2-I2), plus any
    reference to a VectorProtocol / IndexableVector fallback for the shadowed names."""
    members = [_demangled_member(d) for d in demangled_refs]
    keyed = {(o, n, tuple(l) if l is not None else None) for o, n, l in (m for m in members if m)}
    missing = []
    for path in gated_symbol_patterns():
        owner, name, labels = _tsv_member(path)
        if (owner, name, tuple(labels) if labels is not None else None) not in keyed:
            missing.append(path)
    shadowed = ('l1Norm', 'lInfinityNorm', 'manhattanDistance', 'chebyshevDistance', 'isFinite', 'isNormalized')
    fallbacks = sorted({d for d in demangled_refs for n in shadowed
                        if re.search(rf'VectorCore\.(?:VectorProtocol|IndexableVector)\.{n}\b', d)} |
                       {w for w in witness_methods for n in shadowed if w.split('.')[-1].split('!')[0] == n})
    return sorted(missing), fallbacks


def graph_protocol_members(graph_dir):
    """Member paths (`Proto.name(label:)`) of VectorProtocol, VectorFactory and IndexableVector in a
    swift-symbolgraph-extract output: every symbol whose path is `[Proto, member]`, requirements and extension members
    alike (the same rule that generated D02/protocol-members-base.tsv)."""
    paths = set()
    for path in sorted(Path(graph_dir).glob('*.symbols.json')):
        for symbol in json.loads(path.read_text())['symbols']:
            components = symbol.get('pathComponents', [])
            if len(components) == 2 and components[0] in ('VectorProtocol', 'VectorFactory', 'IndexableVector'):
                paths.add('.'.join(components))
    return paths


def member_backstop_differences(graph, scan, tsv, rows):
    """Every pairwise difference between the graph's member set and the live scan, the TSV and the matrix rows."""
    differences = {}
    for name, other in (('scan', scan), ('tsv', tsv), ('rows', rows)):
        if set(other) != set(graph):
            differences[name] = {'onlyInGraph': sorted(set(graph) - set(other)),
                                 f'onlyIn{name.capitalize()}': sorted(set(other) - set(graph))}
    if len(rows) != len(set(rows)):
        differences['rowsDuplicated'] = sorted({r for r in rows if rows.count(r) > 1})
    return differences


def code_lines(text):
    return [n for n, line in enumerate(text.split('\n'), 1)
            if line.strip() and not line.strip().startswith('//') and not line.startswith('import ')]


def main():
    build, output, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    assert mode in ('baseline', 'final')
    output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    snap = products.snapshot_consumer_inputs(build, output)
    fixtures = sorted(FIXTURES.glob('*.swift'))
    assert sorted(p.name for p in fixtures) == sorted(TABLE), 'fixture table and directory differ'
    for source in fixtures + [Path(__file__), FIXTURES / 'gated-public-base.tsv']:
        (output / source.name).write_bytes(source.read_bytes())
    record = {'mode': mode, 'bin': str(build),
              'sourceHashes': {str(p.relative_to(root)): sha(p) for p in sorted(root.glob('Sources/**/*.swift'))},
              'fixtureHashes': {p.name: sha(p) for p in fixtures + [Path(__file__)]},
              'artifacts': {str(dst.relative_to(output)): sha(dst) for _, dst in snap['records']}, 'runs': [],
              'problems': []}
    # The toolchain's swiftc (A2-M8: not whatever `swiftc` is first on PATH) and the macOS SDK it needs.
    swiftc = subprocess.run(['xcrun', '--find', 'swiftc'], capture_output=True, text=True, timeout=60).stdout.strip()
    sdk = subprocess.run(['xcrun', '--sdk', 'macosx', '--show-sdk-path'], capture_output=True, text=True,
                         timeout=60).stdout.strip()
    assert swiftc and sdk, 'xcrun could not resolve swiftc or the macOS SDK'
    record['swiftc'], record['sdk'] = swiftc, sdk
    includes = ['-sdk', sdk, '-I', str(snap['modules']), '-Xcc', '-fmodule-map-file=' + str(snap['modulemap']),
                '-Xcc', '-I' + str(root / 'Sources/VectorCoreC/include')]

    def run(name, command, deadline):
        global active
        path = output / (name + '.log')
        with path.open('xb') as log:
            active = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            pid = active.pid
            try:
                status = active.wait(timeout=deadline)
            except subprocess.TimeoutExpired:
                os.killpg(pid, signal.SIGKILL)
                try:
                    active.wait(timeout=CLEANUP)
                except subprocess.TimeoutExpired:
                    pass
                raise SystemExit(f'{name}: timed out after {deadline} s')
        active = None
        record['runs'].append({'name': name, 'command': command, 'exit': status, 'pid': pid,
                               'deadlineSeconds': deadline, 'cleanupSeconds': CLEANUP, 'logSHA256': sha(path)})
        return status, path.read_text(errors='replace')

    results = {}
    for fixture in fixtures:
        kind, detail = TABLE[fixture.name][mode]
        copy = output / fixture.name
        stem = fixture.stem
        status, text = run(f'typecheck-{stem}', [swiftc, '-swift-version', '6', '-parse-as-library', '-typecheck',
                                                 *includes, str(copy)], COMPILE_DEADLINE)
        errors = re.findall(rf'{re.escape(fixture.name)}:(\d+):\d+: error: ([^\n]+)', text)
        outcome = {'kind': kind, 'typecheckExit': status, 'errors': [f'{n}: {m}' for n, m in errors]}
        ok = False
        if kind in ('positive', 'typechecks'):
            ok = status == 0
            if ok and kind == 'positive':
                exe = output / f'bin-{stem}'
                status, _ = run(f'link-{stem}', [swiftc, '-swift-version', '6', '-parse-as-library', *includes,
                                                 str(copy), *map(str, snap['objects']), '-o', str(exe)],
                                COMPILE_DEADLINE)
                outcome['linkExit'] = status
                if status == 0:
                    status, text = run(f'run-{stem}', [str(exe)], RUN_DEADLINE)
                    outcome['runExit'] = status
                    outcome['marker'] = detail in text
                    outcome['output'] = text.strip().split('\n')[-1][:400]
                    record.setdefault('executables', {})[exe.name] = sha(exe)
                ok = outcome.get('runExit') == 0 and outcome.get('marker', False)
        elif kind == 'negative':
            expected = code_lines(fixture.read_text())
            lines = sorted({int(n) for n, _ in errors})
            outcome['expectedErrorLines'] = expected
            ok = status == 1 and lines == expected and len(errors) == len(expected)
        elif kind == 'fails':
            ok = status == 1 and detail in text
            outcome['requiredDiagnostic'] = detail
        outcome['passed'] = ok
        results[fixture.name] = outcome
        print(fixture.name, kind, 'passed' if ok else 'FAILED', flush=True)
        if not ok:
            record['problems'].append(f'{fixture.name}: {outcome}')
    record['results'] = results
    # Count guard: every formerly gated public symbol is called in CODE of Reachability.swift (comments and string
    # literals masked).
    patterns = gated_symbol_patterns()
    missing = count_guard_missing((FIXTURES / 'Reachability.swift').read_text(), patterns)
    record['countGuard'] = {'symbols': len(patterns), 'missing': missing, 'masked': True}
    if len(patterns) != 61 or missing:
        record['problems'].append(f'count guard: {len(patterns)} symbols, missing {missing}')
    # Binding check (A2-I2), final mode: each of the 61 calls must bind to Vector's (VectorMath's) own unconstrained
    # member. SILGen output is used because it is the type checker's choice before any SIL optimization: no
    # inlining has happened yet (mandatory inlining of @_transparent code runs after SILGen), so every call to an
    # @inlinable member is still a `function_ref` to the member that overload resolution picked. A re-gated member
    # makes the generic call bind to a fallback instead (a VectorProtocol/IndexableVector extension function or a
    # `witness_method`), and its expected `function_ref` disappears.
    if mode == 'final':
        sil = output / 'Reachability.silgen.sil'
        status, _ = run('silgen-Reachability', [swiftc, '-swift-version', '6', '-parse-as-library', '-emit-silgen',
                                                *includes, str(output / 'Reachability.swift'), '-o', str(sil)],
                        COMPILE_DEADLINE)
        binding = {'silgenExit': status}
        if status == 0:
            text = sil.read_text()
            refs = sorted(set(re.findall(r'function_ref @(\$s\w+)', text)))
            witness = sorted(set(re.findall(r'witness_method \$[^,]+, #([\w.!]+)', text)))
            demangle = subprocess.run(['xcrun', 'swift-demangle', '--compact'], input='\n'.join(refs),
                                      capture_output=True, text=True, timeout=120)
            demangled = demangle.stdout.splitlines()
            assert demangle.returncode == 0 and len(demangled) == len(refs), 'swift-demangle failed'
            (output / 'Reachability.function_refs.txt').write_text(
                ''.join(f'{r}\t{d}\n' for r, d in zip(refs, demangled)))
            unbound, fallbacks = binding_failures(demangled, witness)
            binding.update({'functionRefs': len(refs), 'witnessMethods': witness, 'unbound': unbound,
                            'fallbackReferences': fallbacks, 'symbols': len(patterns)})
            if unbound or fallbacks:
                record['problems'].append(f'binding check: unbound {unbound}, fallback references {fallbacks}')
        else:
            record['problems'].append('binding check: SILGen of Reachability.swift failed')
        record['bindingCheck'] = binding
        # Member backstop (RR-I2, phase 5 checklist): the protocol member set of the public symbol graph of THIS build
        # must equal the live lexical scan, the base TSV and the dispatch-matrix rows. It covers what the lexical scan
        # cannot see (members a macro adds, or an extension spelled through a typealias).
        guards_spec = importlib.util.spec_from_file_location('d02_guards', Path(__file__).resolve().parent /
                                                             'test_d02_surface_guards.py')
        guards = importlib.util.module_from_spec(guards_spec)
        guards_spec.loader.exec_module(guards)
        graph_dir = output / 'symbolgraph'
        graph_dir.mkdir()
        arch = 'arm64' if os.uname().machine == 'arm64' else 'x86_64'
        status, _ = run('symbolgraph-VectorCore', ['xcrun', 'swift-symbolgraph-extract', '-module-name', 'VectorCore',
                                                   '-target', f'{arch}-apple-macosx14.0', '-sdk', sdk,
                                                   '-I', str(snap['modules']),
                                                   '-Xcc', '-fmodule-map-file=' + str(snap['modulemap']),
                                                   '-minimum-access-level', 'public', '-output-dir', str(graph_dir)],
                        300)
        backstop = {'extractExit': status}
        if status == 0:
            graph = graph_protocol_members(graph_dir)
            scan = guards.protocol_members(root)
            tsv = [row[0] for row in guards.tsv_paths('protocol-members-base.tsv')]
            rows = [path for path, _ in guards.dispatch_rows(
                (root / 'Tests/ComprehensiveTests/D02WitnessDispatchMatrixTests.swift').read_text())]
            differences = member_backstop_differences(graph, scan, tsv, rows)
            backstop.update({'graph': len(graph), 'scan': len(scan), 'tsv': len(tsv), 'rows': len(rows),
                             'differences': differences})
            if differences:
                record['problems'].append(f'member backstop: {differences}')
            print('member backstop', 'passed' if not differences else 'FAILED',
                  f'graph={len(graph)} scan={len(scan)} tsv={len(tsv)} rows={len(rows)}', flush=True)
        else:
            record['problems'].append('member backstop: swift-symbolgraph-extract failed')
        record['memberBackstop'] = backstop
        print('binding check', 'passed' if status == 0 and not binding.get('unbound') and
              not binding.get('fallbackReferences') else 'FAILED', flush=True)
    (output / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    for problem in record['problems']:
        print('PROBLEM', problem)
    print(f'D02_CONSUMERS_{mode.upper()}_{"COMPLETE" if not record["problems"] else "FAILED"}', flush=True)
    return 0 if not record['problems'] else 1


if __name__ == '__main__':
    sys.exit(main())
