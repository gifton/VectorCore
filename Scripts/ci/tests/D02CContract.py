#!/usr/bin/env python3
"""D02 C-surface and INT8-accumulation evidence script (RD-C2). NOT wired into CI (the D02 brief forbids CI gate
changes); it is committed so a reviewer can reproduce the recorded RED and GREEN runs.

usage: D02CContract.py --tree TREE --out OUT --mode red|green [--compare-to RED_OUT]

TREE is the source tree under test (a repository checkout or an archive of one); its Sources/VectorCoreC files are
compiled BY PATH with clang (no rewritten copies) and hashed before and after every step. Fixtures are this
script's sibling directory D02C/. OUT must not exist. macOS arm64 host only.

What it runs, per compile configuration:
  sdot  the default arm64 macOS 14 target (clang defines __ARM_FEATURE_DOTPROD, so vc_arm64.c compiles SDOT);
  neon  the same target with -march=armv8-a -mcpu=generic (no dot-product extension, so vc_arm64.c compiles the
        NEON fallback; -mcpu=generic also stops the auto-vectorizer from emitting sdot, which the run asserts with
        otool on the neon objects; the subset ISA runs natively on an arm64 Mac),
each as `san` (-O1 -g, AddressSanitizer + UndefinedBehaviorSanitizer, recoverable reports) and `opt` (-O2):
  * the INT8 fixture table (D02C/d02_int8_harness.c) through the dispatcher vc_dot_int8, the internal scalar entry
    vc_scalar_dot_int8 and the internal architecture entry vc_arm64_dot_int8, each against an independent
    Int64-then-modulo oracle; one oracle-mutant run must FAIL (mutation control);
  * the range512 contract fixture (D02C/d02_range512_harness.c) through the same three entries;
  * an ordinary C consumer and the same consumer as C++ (D02C/d02_consumer.c/.cpp) calling every retained function;
  * a generated consumer that takes the address of EVERY function the public headers declare (links iff every
    declaration is defined);
  * a feature probe comparing vc_has_neon/vc_has_dotprod with the configuration's `clang -dM` macros.
Plus: the private helper tests (D02C/d02_helper_tests.c; green only, the header does not exist at the base), an
x86_64 cross-compile and disassembly of x86_64/vc_x86.c (supporting evidence only: NOT executed, no emulator), the
declared/defined/exported C symbol inventory, and lexical checks (old `i + N <= lanes` guards, `(int32_t)` casts in
the INT8 sources, hardware/performance comments in the C files, internal names in public headers).

Mode expectations (the run exits 1 if they do not hold):
  red    UBSan reports signed integer overflow in the scalar INT8 loop and in the SDOT and NEON final additions;
         the every-declaration consumer fails to link (declared-but-undefined proposed functions).
  green  0 sanitizer reports, every harness and consumer passes, the every-declaration consumer links, the lexical
         checks are clean, and (with --compare-to) the exported symbol set is a subset of the red run's.
Every child runs with a 15 s deadline (compiles: 120 s) in its own process group and is killed as a group on
timeout, with a 2 s cleanup bound; output goes to a log file per step. Records: OUT/provenance.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / 'D02C'
C_SOURCES = ['common/vc_scalar.c', 'common/vc_dispatch.c', 'common/vc_features.c', 'common/vc_lapack.c',
             'arm64/vc_arm64.c', 'x86_64/vc_x86.c']
CONFIGS = {'sdot': ['-target', 'arm64-apple-macos14.0'],
           'neon': ['-target', 'arm64-apple-macos14.0', '-march=armv8-a', '-mcpu=generic']}
VARIANTS = {'san': ['-O1', '-g', '-fno-omit-frame-pointer', '-fsanitize=address,undefined'],
            'opt': ['-O2']}
SAN_ENV = {'UBSAN_OPTIONS': 'print_stacktrace=1:report_error_type=1:halt_on_error=0',
           'ASAN_OPTIONS': 'detect_leaks=0:halt_on_error=1:abort_on_error=0'}
RUN_DEADLINE, COMPILE_DEADLINE, CLEANUP = 15, 120, 2
UB_REPORT = re.compile(r'([\w./-]+\.(?:c|h)):(\d+):\d+: runtime error: ([^\n]+)')
ASAN_REPORT = re.compile(r'ERROR: AddressSanitizer')
OLD_GUARD = re.compile(r'\bi\s*\+\s*\d+\s*<=\s*lanes\b')
PROTO = re.compile(r'^\s*(?:VC_EXPORT\s+)?(?:extern\s+)?[A-Za-z_][\w\s\*]*?\b(vc_\w+)\s*\(')
HW_COMMENT = re.compile(r'cycles|\bA1[0-9]\b|\bILP\b|Apple Silicon|M-series', re.IGNORECASE)
# A signed add/sub on i32 values carrying `nsw`, also when LLVM prints `nuw nsw` (both flags).
IR_SIGNED_ADD = re.compile(r'\b(?:add|sub)(?: nuw)? nsw (?:i32|<\d+ x i32>)')
for _line, _flag in (('%5 = add nsw i32 %3, %4', True), ('%5 = add nuw nsw i32 %3, %4', True),
                     ('%5 = sub nsw <4 x i32> %a, %b', True), ('%5 = add i32 %3, %4', False),
                     ('%5 = add nuw i32 %3, %4', False), ('%5 = mul nsw i32 %3, %4', False),
                     ('%5 = add nsw i64 %3, %4', False)):
    assert bool(IR_SIGNED_ADD.search(_line)) == _flag, _line

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


class Runner:
    def __init__(self, out):
        self.out = out
        self.records = []

    def run(self, name, argv, deadline=RUN_DEADLINE, env=None):
        """Run argv in its own process group with a deadline; output to OUT/logs/<name>.log. Never raises on rc."""
        global active
        log = self.out / 'logs' / f'{name}.log'
        timed_out = cleanup_timed_out = False
        full_env = dict(os.environ)
        full_env.update(env or {})
        with log.open('xb') as handle:
            active = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                      start_new_session=True, env=full_env)
            pid = active.pid
            try:
                status = active.wait(timeout=deadline)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(pid, signal.SIGKILL)
                try:
                    status = active.wait(timeout=CLEANUP)
                except subprocess.TimeoutExpired:
                    status, cleanup_timed_out = None, True
        active = None
        if timed_out:
            # Throw immediately after a timeout, before any status is used.
            raise SystemExit(f'{name}: timed out after {deadline} s (cleanup timed out: {cleanup_timed_out})')
        text = log.read_text(errors='replace')
        record = {'name': name, 'argv': argv, 'pid': pid, 'exit': status, 'deadlineSeconds': deadline,
                  'cleanupSeconds': CLEANUP, 'env': env or {}, 'log': str(log.relative_to(self.out)),
                  'logSHA256': sha(log)}
        self.records.append(record)
        return status, text


def source_hashes(tree):
    root = tree / 'Sources/VectorCoreC'
    return {str(p.relative_to(tree)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def declared_functions(tree):
    names = []
    for header in sorted((tree / 'Sources/VectorCoreC/include').glob('*.h')):
        in_block = False
        for line in header.read_text().split('\n'):
            stripped = line.strip()
            if stripped.startswith('/*'):
                in_block = '*/' not in stripped
                continue
            if in_block:
                in_block = '*/' not in stripped
                continue
            if stripped.startswith('//') or stripped.startswith('#') or stripped.startswith('*'):
                continue
            match = PROTO.match(line)
            if match and match.group(1) not in names:
                names.append(match.group(1))
    return names


def nm_exported(objects):
    out = {}
    for obj in objects:
        proc = subprocess.run(['nm', '-gU', '-j', str(obj)], capture_output=True, text=True, timeout=60)
        for name in proc.stdout.split():
            if name.startswith('_vc_'):
                out.setdefault(name[1:], []).append(obj.name)
    return out


def lexical_checks(tree):
    c_root = tree / 'Sources/VectorCoreC'
    old_guards, casts, hw, internal_in_headers = [], [], [], []
    for path in sorted(c_root.rglob('*')):
        if path.suffix not in ('.c', '.h'):
            continue
        rel = str(path.relative_to(tree))
        text = path.read_text()
        for n, line in enumerate(text.split('\n'), 1):
            if OLD_GUARD.search(line):
                old_guards.append(f'{rel}:{n}: {line.strip()}')
            if path.name in ('vc_scalar.c', 'vc_arm64.c', 'vc_x86.c', 'vc_dispatch.c') and '(int32_t)' in line:
                casts.append(f'{rel}:{n}: {line.strip()}')
            if path.name in ('vc_scalar.c', 'vc_arm64.c', 'vc_x86.c', 'VectorCoreC.h') and HW_COMMENT.search(line):
                hw.append(f'{rel}:{n}: {line.strip()}')
            if path.parent.name == 'include' and re.search(r'\bvc_(?:arm64|x86|scalar)_\w+\s*\(', line) \
                    and not line.strip().startswith('//'):
                internal_in_headers.append(f'{rel}:{n}: {line.strip()}')
    helper = c_root / 'common/vc_int8_accum.h'
    helper_casts = []
    if helper.exists():
        # Every `(int32_t)` operand in the helper: int8_t promotions (`a`, `b`) are in range by type; the two
        # uint32_t operands must be the checked mapping's (`bits` behind `bits <= INT32_MAX`, and
        # `(UINT32_MAX - bits)` on the other branch).
        code = '\n'.join(line for line in helper.read_text().split('\n') if not line.strip().startswith('//'))
        helper_casts = re.findall(r'\(int32_t\)\s*(\([^()]*\)|\w+)', code)
    package = (tree / 'Package.swift').read_text()
    return {'oldBlockGuards': old_guards, 'int32CastsInInt8Sources': casts, 'hardwareOrPerformanceComments': hw,
            'internalNamesInPublicHeaders': internal_in_headers, 'helperInt32Casts': helper_casts,
            'signedOverflowFlagsInPackage': [f for f in ('-fwrapv', '-fno-strict-overflow') if f in package]}


def ub_reports(text):
    return sorted({f'{Path(m.group(1)).name}:{m.group(2)}: {m.group(3).strip()}' for m in UB_REPORT.finditer(text)})


def build_config(runner, tree, config, variant, include_dir, sources_ok):
    """Compile the tree's C sources for one configuration; return the object paths."""
    obj_dir = runner.out / f'obj-{config}-{variant}'
    obj_dir.mkdir()
    objects = []
    for rel in C_SOURCES:
        src = tree / 'Sources/VectorCoreC' / rel
        obj = obj_dir / (Path(rel).stem + '.o')
        argv = ['xcrun', 'clang', *CONFIGS[config], *VARIANTS[variant], '-std=c11', '-c',
                '-DACCELERATE_NEW_LAPACK=1', '-I', str(include_dir), str(src), '-o', str(obj)]
        status, _ = runner.run(f'compile-{config}-{variant}-{Path(rel).stem}', argv, COMPILE_DEADLINE)
        sources_ok.append((f'compile-{config}-{variant}-{Path(rel).stem}', status == 0))
        objects.append(obj)
    return objects


def link(runner, name, config, variant, sources, objects, include_dirs, cxx=False):
    exe = runner.out / 'bin' / name
    tool = 'clang++' if cxx else 'clang'
    lang = ['-std=c++17', '-x', 'c++'] if cxx else ['-std=c11']
    argv = ['xcrun', tool, *CONFIGS[config], *VARIANTS[variant], *lang]
    for inc in include_dirs:
        argv += ['-I', str(inc)]
    argv += [str(s) for s in sources]
    if cxx:
        argv += ['-x', 'none']
    argv += [str(o) for o in objects] + ['-framework', 'Accelerate', '-o', str(exe)]
    status, text = runner.run(f'link-{name}', argv, COMPILE_DEADLINE)
    return exe if status == 0 else None, text


def macros(runner, config):
    status, text = runner.run(f'macros-{config}', ['xcrun', 'clang', *CONFIGS[config], '-dM', '-E', '-x', 'c',
                                                   '/dev/null'], COMPILE_DEADLINE)
    assert status == 0, f'macros-{config}'
    defined = set(re.findall(r'#define (\w+)', text))
    return {name: name in defined for name in ('__ARM_NEON', '__ARM_FEATURE_DOTPROD', '__AVX2__', '__aarch64__')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--mode', choices=['red', 'green'], required=True)
    parser.add_argument('--compare-to')
    args = parser.parse_args()
    tree, out = Path(args.tree).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'logs').mkdir()
    (out / 'bin').mkdir()
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    runner = Runner(out)
    include_dir = tree / 'Sources/VectorCoreC/include'
    hashes_before = source_hashes(tree)
    record = {'mode': args.mode, 'tree': str(tree), 'sourceHashes': hashes_before,
              'fixtureHashes': {p.name: sha(p) for p in sorted(FIXTURES.iterdir()) if p.is_file()},
              'scriptSHA256': sha(Path(__file__)), 'host': os.uname().machine, 'cells': [], 'problems': []}
    problems = record['problems']
    toolchain = runner.run('toolchain', ['xcrun', 'clang', '--version'], COMPILE_DEADLINE)[1]
    record['toolchain'] = toolchain.strip().split('\n')[0]
    compiles = []
    declared = declared_functions(tree)
    record['declaredFunctions'] = declared
    # Generated consumer: the address of every declared function (links iff every declaration is defined).
    every = out / 'every_declared.c'
    every.write_text('#include "VectorCoreC.h"\n#include <stdio.h>\nint main(void) {\n'
                     '    void (*volatile fns[])(void) = {\n'
                     + ''.join(f'        (void (*)(void))&{name},\n' for name in declared)
                     + '    };\n    printf("D02_EVERY_DECLARED_LINKED %zu\\n", sizeof fns / sizeof fns[0]);\n'
                     '    return 0;\n}\n')
    ub_by_cell = {}
    for config in CONFIGS:
        record.setdefault('macros', {})[config] = macros(runner, config)
        for variant in VARIANTS:
            env = SAN_ENV if variant == 'san' else {}
            objects = build_config(runner, tree, config, variant, include_dir, compiles)
            tag = f'{config}-{variant}'
            if config == 'neon':
                # The NEON-fallback cells must contain no SDOT instruction anywhere (not even an unreachable one).
                _, disassembly = runner.run(f'disassemble-{tag}', ['otool', '-tV', *map(str, objects)], COMPILE_DEADLINE)
                sdot = len(re.findall(r'\bsdot\b', disassembly))
                record.setdefault('neonSdotInstructions', {})[tag] = sdot
                if sdot:
                    problems.append(f'{tag}: {sdot} sdot instruction(s) in the NEON-fallback objects')
            int8, _ = link(runner, f'int8-{tag}', config, variant, [FIXTURES / 'd02_int8_harness.c'], objects,
                           [include_dir])
            rng, _ = link(runner, f'range-{tag}', config, variant, [FIXTURES / 'd02_range512_harness.c'], objects,
                          [include_dir])
            for entry in ('dispatcher', 'scalar', 'arch'):
                for kind, exe in (('int8', int8), ('range', rng)):
                    if exe is None:
                        problems.append(f'{kind}-{tag}: link failed')
                        continue
                    status, text = runner.run(f'{kind}-{tag}-{entry}', [str(exe), entry], env=env)
                    reports = ub_reports(text)
                    asan = bool(ASAN_REPORT.search(text))
                    summary = re.findall(r'(?:SUMMARY|RANGE_TESTS) [^\n]*', text)
                    cell = {'cell': f'{kind}-{tag}-{entry}', 'exit': status, 'ubsan': reports, 'asan': asan,
                            'summary': summary[-1] if summary else None}
                    record['cells'].append(cell)
                    ub_by_cell[cell['cell']] = reports
                    if args.mode == 'green' and (status != 0 or reports or asan):
                        problems.append(f'{cell["cell"]}: exit {status}, ubsan {reports}, asan {asan}')
            if int8 is not None and variant == 'opt':
                status, text = runner.run(f'int8-{tag}-dispatcher-oracle-mutant', [str(int8), 'dispatcher',
                                                                                   '--oracle-mutant'])
                record['cells'].append({'cell': f'int8-{tag}-oracle-mutant', 'exit': status,
                                        'summary': (re.findall(r'SUMMARY [^\n]*', text) or [None])[-1]})
                if status != 0:
                    problems.append(f'int8-{tag}: the oracle mutant was not caught')
            for lang, src, cxx in (('c', 'd02_consumer.c', False), ('cxx', 'd02_consumer.cpp', True)):
                exe, _ = link(runner, f'consumer-{lang}-{tag}', config, variant, [FIXTURES / src], objects,
                              [include_dir, FIXTURES], cxx=cxx)
                status, text = (None, '') if exe is None else runner.run(f'consumer-{lang}-{tag}', [str(exe)], env=env)
                ok = status == 0 and 'D02_C_CONSUMER_COMPLETE' in text and not ub_reports(text) \
                    and not ASAN_REPORT.search(text)
                record['cells'].append({'cell': f'consumer-{lang}-{tag}', 'linked': exe is not None, 'exit': status,
                                        'passed': ok, 'features': (re.findall(r'features [^\n]*', text) or [None])[-1]})
                if not ok:
                    problems.append(f'consumer-{lang}-{tag}: linked {exe is not None}, exit {status}')
                if ok and lang == 'c':
                    feats = dict(re.findall(r'(\w+)=(\d)', re.findall(r'features [^\n]*', text)[-1]))
                    want = record['macros'][config]
                    probe = {'neon': int(feats['neon']) == int(want['__ARM_NEON']),
                             'dotprod': int(feats['dotprod']) == int(want['__ARM_FEATURE_DOTPROD'])}
                    record.setdefault('featureVersusMacros', {})[tag] = {'queries': feats, 'macros': want,
                                                                         'equal': probe}
                    if not all(probe.values()):
                        problems.append(f'feature queries differ from the {config} macros: {feats} vs {want}')
            exe, link_text = link(runner, f'every-declared-{tag}', config, variant, [every], objects, [include_dir])
            undefined = sorted(set(re.findall(r'"_(vc_\w+)"', link_text)))
            record['cells'].append({'cell': f'every-declared-{tag}', 'linked': exe is not None,
                                    'undefined': undefined})
            if args.mode == 'green' and exe is None:
                problems.append(f'every-declared-{tag}: declared-but-undefined {undefined}')
            if args.mode == 'red' and exe is not None:
                problems.append(f'every-declared-{tag}: linked at the base (expected the proposed functions undefined)')
            if config == 'sdot' and variant == 'opt':
                exported = nm_exported(objects)
                record['symbols'] = {'exported': sorted(exported), 'declared': declared,
                                     'declaredUndefined': sorted(set(declared) - set(exported)),
                                     'exportedUndeclared': sorted(set(exported) - set(declared)),
                                     'objectHashes': {o.name: sha(o) for o in objects}}
    failed_compiles = [name for name, ok in compiles if not ok]
    if failed_compiles:
        problems.append(f'compile failures: {failed_compiles}')
    # Private helper tests (pure arithmetic; no buffers). The header does not exist at the base.
    helper = tree / 'Sources/VectorCoreC/common/vc_int8_accum.h'
    if helper.exists():
        for variant in VARIANTS:
            exe = out / 'bin' / f'helper-{variant}'
            argv = ['xcrun', 'clang', *CONFIGS['sdot'], *VARIANTS[variant], '-std=c11', '-I', str(helper.parent),
                    str(FIXTURES / 'd02_helper_tests.c'), '-o', str(exe)]
            status, _ = runner.run(f'link-helper-{variant}', argv, COMPILE_DEADLINE)
            status, text = (status, '') if status != 0 else runner.run(
                f'helper-{variant}', [str(exe)], env=SAN_ENV if variant == 'san' else {})
            summary = (re.findall(r'HELPER_TESTS [^\n]*', text) or [None])[-1]
            old = re.findall(r'OLD_GUARD [^\n]*', text)
            record['cells'].append({'cell': f'helper-{variant}', 'exit': status, 'summary': summary,
                                    'ubsan': ub_reports(text), 'oldGuardWraps': old[:6], 'oldGuardLines': len(old)})
            if status != 0 or ub_reports(text):
                problems.append(f'helper-{variant}: exit {status}, ubsan {ub_reports(text)}')
    elif args.mode == 'green':
        problems.append('helper header missing')
    else:
        record['cells'].append({'cell': 'helper', 'skipped': 'vc_int8_accum.h does not exist in this tree (base)'})
    # x86_64: cross-compile and disassemble only (supporting evidence; never executed here).
    x86_obj = out / 'x86_64-vc_x86.o'
    status, _ = runner.run('x86-cross-compile', ['xcrun', 'clang', '-target', 'x86_64-apple-macos14.0', '-O2', '-c',
                                                 '-std=c11', '-I', str(include_dir),
                                                 str(tree / 'Sources/VectorCoreC/x86_64/vc_x86.c'), '-o',
                                                 str(x86_obj)], COMPILE_DEADLINE)
    record['x86'] = {'compiled': status == 0, 'executed': False,
                     'note': 'cross-compiled and disassembled only; no x86_64 execution (no Rosetta, no emulator)'}
    if status == 0:
        _, text = runner.run('x86-disassembly', ['otool', '-tV', str(x86_obj)], COMPILE_DEADLINE)
        body = text.split('_vc_x86_dot_int8:')[-1] if '_vc_x86_dot_int8:' in text else ''
        record['x86']['dotInt8Instructions'] = len([ln for ln in body.split('\n') if '\t' in ln])
        record['x86']['objectSHA256'] = sha(x86_obj)
    else:
        problems.append('x86 cross-compile failed')
    # LLVM IR at -O2: a signed add/sub carrying `nsw` on i32 values inside an INT8 dot function would let the
    # optimizer assume no overflow (the defect class). Integer products of two int8 values cannot overflow, so
    # `mul nsw` is not counted.
    record['irNsw'] = {}
    ir_targets = [('arm64-sdot', CONFIGS['sdot'], ['common/vc_scalar.c', 'arm64/vc_arm64.c']),
                  ('arm64-neon', CONFIGS['neon'], ['arm64/vc_arm64.c']),
                  ('x86_64', ['-target', 'x86_64-apple-macos14.0'], ['common/vc_scalar.c', 'x86_64/vc_x86.c'])]
    for label, target, rels in ir_targets:
        for rel in rels:
            ir = out / f'ir-{label}-{Path(rel).stem}.ll'
            status, _ = runner.run(f'ir-{label}-{Path(rel).stem}',
                                   ['xcrun', 'clang', *target, '-O2', '-std=c11', '-S', '-emit-llvm',
                                    '-DACCELERATE_NEW_LAPACK=1', '-I', str(include_dir),
                                    str(tree / 'Sources/VectorCoreC' / rel), '-o', str(ir)], COMPILE_DEADLINE)
            if status != 0:
                problems.append(f'ir-{label}-{rel}: compile failed')
                continue
            hits, current = [], None
            for line in ir.read_text().split('\n'):
                if line.startswith('define '):
                    current = re.search(r'@(\w+)\(', line).group(1)
                elif line.startswith('}'):
                    current = None
                elif current and 'dot_int8' in current and IR_SIGNED_ADD.search(line):
                    hits.append(f'{current}: {line.strip()}')
            record['irNsw'][f'{label}:{rel}'] = hits
            if args.mode == 'green' and hits:
                problems.append(f'ir {label} {rel}: signed no-wrap additions in INT8 code: {hits[:3]}')
    lex = lexical_checks(tree)
    record['lexical'] = lex
    hashes_after = source_hashes(tree)
    record['sourcesUnchangedDuringRun'] = hashes_after == hashes_before
    if hashes_after != hashes_before:
        problems.append('a source under test changed during the run')
    if args.mode == 'green':
        for key in ('oldBlockGuards', 'int32CastsInInt8Sources', 'hardwareOrPerformanceComments',
                    'internalNamesInPublicHeaders', 'signedOverflowFlagsInPackage'):
            if lex[key]:
                problems.append(f'lexical {key}: {lex[key]}')
        unsigned_operands = [op for op in lex['helperInt32Casts'] if op not in ('a', 'b')]
        if sorted(unsigned_operands) != ['(UINT32_MAX - bits)', 'bits']:
            problems.append(f'helper (int32_t) casts: expected only the checked mapping and int8 promotions, '
                            f'got {lex["helperInt32Casts"]}')
        if record.get('symbols', {}).get('declaredUndefined'):
            problems.append(f'declared but undefined: {record["symbols"]["declaredUndefined"]}')
        if args.compare_to:
            before = json.loads((Path(args.compare_to) / 'provenance.json').read_text())['symbols']['exported']
            added = sorted(set(record['symbols']['exported']) - set(before))
            record['symbols']['addedVersusCompare'] = added
            record['symbols']['removedVersusCompare'] = sorted(set(before) - set(record['symbols']['exported']))
            if added:
                problems.append(f'new exported symbols: {added}')
    else:
        want = {'int8-sdot-san-scalar': 'vc_scalar.c', 'int8-sdot-san-arch': 'vc_arm64.c',
                'int8-neon-san-arch': 'vc_arm64.c'}
        record['redWitnesses'] = {}
        for cell, source in want.items():
            hits = [r for r in ub_by_cell.get(cell, []) if r.startswith(source) and 'signed integer overflow' in r]
            record['redWitnesses'][cell] = hits
            if not hits:
                problems.append(f'RED not observed: {cell} has no signed-overflow report in {source}')
    record['runs'] = runner.records
    (out / 'provenance.json').write_text(json.dumps(record, indent=1) + '\n')
    for problem in problems:
        print('PROBLEM', problem)
    print(f'D02_C_CONTRACT_{args.mode.upper()}_{"COMPLETE" if not problems else "FAILED"} problems={len(problems)}')
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main())
