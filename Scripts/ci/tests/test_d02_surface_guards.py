"""D02 surface guards: lexical class closures for the D02 surface decisions (stdlib only, filesystem reads only,
so they also run in a git-less export).

WHY. D02 removed the StaticDimension.Storage requirement and every `D.Storage` gate, removed ComputeDevice's
hardware-discovery-looking members, removed the proposed-but-unimplemented C declarations, and replaced the INT8
accumulation guards and casts that could overflow. Each guard below fails if one of those comes back:
  C1  the public C headers declare exactly the 13 retained, implemented functions;
  C6  no internal vc_scalar_/vc_arm64_/vc_x86_ name is declared in a public header;
  C21 no `i + N <= lanes` block guard (it wraps near SIZE_MAX) in Sources/VectorCoreC;
  C15 no `(int32_t)` cast in the INT8 sources; the private helper casts only `bits` (behind `bits <= INT32_MAX`),
      `(UINT32_MAX - bits)` and the int8 operands `a`/`b`;
  C4  no cycle-count, chip-generation, ILP or Apple-silicon claims in the touched C files (RD-C4; the claim
      ledger's rules do not cover C);
  S5  StaticDimension requires only `static var value: Int { get }`;
  S6  no `D.Storage` constraint in Sources; S2 no built-in `DimN.Storage` alias;
  S9  no `typealias Storage` in a StaticDimension conformer under Tests/;
  D8  ComputeDevice.swift contains no hardware probe (canImport(Metal), ProcessInfo, #available,
      targetEnvironment) and none of the removed member names; DeviceCapabilities is declared nowhere in Sources;
  W2  the member set of VectorProtocol/VectorFactory/IndexableVector derived from the CURRENT sources (requirement
      blocks plus public extension members, comments and strings masked) equals both
      Scripts/ci/tests/D02/protocol-members-base.tsv and the D02WitnessDispatchMatrixTests rows (requirements must be
      dynamic rows); a new member in any spelling the scan reads fails it (tree-copy control; macro-added members are
      the exception, see LIMITS);
  R1  Scripts/ci/tests/D02/Reachability.swift calls every symbol of Scripts/ci/tests/D02/gated-public-base.tsv in
      code (comments and strings masked); the SILGen binding check of D02RunConsumers.py has unit controls here;
  E1  D02DegenerateWidthTests has exactly one row per gated symbol (widths 0 and 1);
  W4  (phase 5) every read of computeProvider/simdProvider/linearAlgebraProvider in Sources/VectorCore sits in a
      function that HOOK_SITES maps to the public entries reaching it, each entry has a row naming that hook in the
      `## Provider hooks` table of the Operations documentation, and every row is reached by a read;
  W6  (phase 5 fix 1) the rows of that table, as (entry, hook, exact call list), equal the `expectRow(entry, hook,
      calls)` calls of D02ProviderHookRouteTests row for row: no missing, extra or differing row;
  H3  (phase 5) no code in Sources uses the removed bufferProvider hook, the VectorCore.Configuration struct, the
      four Q14 flag helpers, SoA.memoryEfficiency, estimatePerformanceGain, the PERFORMANCE_METRICS blocks or
      SIMDMemoryUtilities, no global alignment setting replaces memoryAlignment, and no `VectorCore` declaration
      or extension anywhere in Sources/VectorCore declares `configuration` or `Configuration` (fix 1, A-M4);
  H5  (phase 5) the CHANGELOG [Unreleased] Removed section names each phase 5 removal, and the hook entry points to
      BatchOperations.updateConfiguration and the explicit buffer APIs.
LIMITS (lexical, not a parser). The C checks strip comments where they say so; the Swift W2 and R1 checks mask
comments and string literals (swift_lexing._mask). W2 reads `[@attr ...] [public] extension [VectorCore.]Proto` blocks
(attributes on the same line included) and func/var/let/subscript/init/typealias/associatedtype members; it does NOT
see members a macro adds. That remaining gap is covered by the phase 5 symbol-graph comparison
(D02/phase2-4/patches/phase5-staged-docs.md section 4), not by this file. R1 does not see a call spelled through a
variable or a typealias of a removed name. W4 and H3 mask with `keep_interpolations=True` (the P02/T17 opt-in), so
code inside a string interpolation is seen. W4 finds the enclosing function by its declaration line (func, init, or a
computed `var name: Type {`); a hook read through a stored closure, a local alias of the hook (`let p =
Operations.simdProvider` in one function, used in another) or a macro is attributed to where the token appears, and a
public entry that reaches a mapped function through a path HOOK_SITES does not list is not seen. HOOK_SITES lists
by hand the public entries reached through another public entry (lInfinityNorm -> chebyshevDistance(to:);
PCAModel.fit -> Operations.pca and Operations.umap), so a NEW public caller of a hooked public entry is not seen by W4
(W6 sees it only once someone adds its row or its expectRow). W6 compares text: it checks that the table and the
tests agree, and the tests check the calls at run time. H3 matches names, so a removed member re-added under another
name is not seen. Each detector has must-flag and must-pass
controls below.
"""
import importlib.util
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Scripts/ci'))
from swift_lexing import _mask  # noqa: E402  (comments and string literals blanked)
C_ROOT = ROOT / 'Sources/VectorCoreC'
RETAINED_C = {'vc_has_avx2', 'vc_has_avx512f', 'vc_has_neon', 'vc_has_dotprod', 'vc_dot_fp32_512', 'vc_l2sq_fp32_512',
              'vc_range_l2sq_fp32_512', 'vc_dot_int8', 'vc_lapack_available', 'vc_lapack_sgeqrf', 'vc_lapack_sorgqr',
              'vc_lapack_sgesdd', 'vc_lapack_ssyevd'}
OLD_GUARD = re.compile(r'\bi\s*\+\s*\d+\s*<=\s*lanes\b')
PROTO = re.compile(r'^\s*(?:VC_EXPORT\s+)?(?:extern\s+)?[A-Za-z_][\w\s\*]*?\b(vc_\w+)\s*\(', re.M)
INTERNAL = re.compile(r'\bvc_(?:arm64|x86|scalar)_\w+\s*\(')
HW_COMMENT = re.compile(r'cycles|\bA1[0-9]\b|\bILP\b|Apple Silicon|M-series', re.IGNORECASE)
INT8_SOURCES = ('common/vc_scalar.c', 'arm64/vc_arm64.c', 'x86_64/vc_x86.c', 'common/vc_dispatch.c')
HELPER_CAST = re.compile(r'\(int32_t\)\s*(\([^()]*\)|\w+)')
DEVICE_PROBES = ('canImport(Metal)', 'ProcessInfo', '#available', 'targetEnvironment')
REMOVED_DEVICE = ('isAvailable', 'isAccelerated', 'requiredAlignment', 'minimumEfficientSize', 'queryCapabilities',
                  'availableDevices', 'mostCapable', 'DeviceCapabilities')
STORAGE_GATE = re.compile(r'\b\w+\.Storage\s*:\s*VectorStorageOperations\b|\bD\.Storage\b')
BUILTIN_ALIAS = re.compile(r'typealias\s+Storage\s*=\s*DimensionStorage<Dim\d+')
CONFORMER = re.compile(r'\b(?:struct|enum|class)\s+\w+\s*:\s*[^{]*\bStaticDimension\b[^{]*\{')


def strip_c_comments(text):
    text = re.sub(r'/\*.*?\*/', lambda m: '\n' * m.group().count('\n'), text, flags=re.S)
    return re.sub(r'//[^\n]*', '', text)


def declared_c_functions(text):
    return set(PROTO.findall(strip_c_comments(text)))


def storage_alias_in_conformers(text):
    """(line, conformer header) for each StaticDimension conformer whose body declares `typealias Storage`."""
    hits = []
    for match in CONFORMER.finditer(text):
        depth, i = 1, match.end()
        while i < len(text) and depth:
            depth += {'{': 1, '}': -1}.get(text[i], 0)
            i += 1
        if re.search(r'\btypealias\s+Storage\b', text[match.end():i]):
            hits.append((text.count('\n', 0, match.start()) + 1, match.group().strip()))
    return hits


def static_dimension_requirements(text):
    body = re.search(r'public protocol StaticDimension\s*:\s*Sendable\s*\{(.*?)\n\}', text, re.S)
    if not body:
        return None
    code = [line.strip() for line in body.group(1).split('\n') if line.strip() and not line.strip().startswith('//')]
    return code


# --- W2 (B-I1): the protocol member set derived from the current sources --------------------------------------
PROTOCOLS = ('VectorProtocol', 'VectorFactory', 'IndexableVector')
DECL = re.compile(r'(?<![.\w])(?P<mods>(?:(?:@\w+(?:\([^)]*\))?|public|open|package|internal|private|fileprivate|static|class|mutating|'
                  r'nonmutating|final|override|prefix|postfix|infix|nonisolated|dynamic)\s+)*)'
                  r'(?P<kind>func|var|let|subscript|init|associatedtype|typealias)\b')


def block_end(text, open_index):
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    raise ValueError('unbalanced braces')


def flatten(body):
    """The body with the contents of nested braces blanked, so only depth-0 declarations remain."""
    out, depth = [], 0
    for ch in body:
        if ch == '{':
            depth += 1
            out.append(' ')
        elif ch == '}':
            depth -= 1
            out.append(' ')
        else:
            out.append(ch if depth == 0 or ch == '\n' else ' ')
    return ''.join(out)


def labels(params, operator, subscript):
    if not params.strip():
        return []
    parts, depth, cur = [], 0, ''
    for ch in params:
        if ch in '([<':
            depth += 1
        elif ch in ')]>':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(cur)
            cur = ''
        else:
            cur += ch
    parts.append(cur)
    out = []
    for part in parts:
        names = part.split(':', 1)[0].split()
        if operator:
            out.append('_')
        elif subscript:
            out.append(names[0] if len(names) == 2 else '_')
        else:
            out.append(names[0])
    return out


def members(body, require_public):
    found = set()
    flat = flatten(body)
    for m in DECL.finditer(flat):
        mods, kind = m.group('mods'), m.group('kind')
        if re.search(r'\b(?:private|fileprivate|internal|package)\b', mods):
            continue
        if require_public and not re.search(r'\b(?:public|open)\b', mods):
            continue
        rest = flat[m.end():]
        if kind in ('var', 'let', 'associatedtype', 'typealias'):
            found.add(re.match(r'\s*(\w+)', rest).group(1))
            continue
        name = ''
        if kind == 'func':
            nm = re.match(r'\s*(\w+|[^\s\w(<]+)\s*', rest)
            name, rest = nm.group(1), rest[nm.end():]
        else:
            name = kind
            rest = rest.lstrip()
        if rest.startswith('<'):
            depth, i = 0, 0
            while i < len(rest):
                depth += {'<': 1, '>': -1}.get(rest[i], 0)
                i += 1
                if depth == 0:
                    break
            rest = rest[i:].lstrip()
        if not rest.startswith('('):
            continue
        depth, j = 0, 0
        while j < len(rest):
            depth += {'(': 1, ')': -1}.get(rest[j], 0)
            if depth == 0:
                break
            j += 1
        operator = kind == 'func' and not re.match(r'\w', name)
        found.add(f'{name}({"".join(l + ":" for l in labels(rest[1:j], operator, kind == "subscript"))})')
    return found


def protocol_members(root):
    """Member paths (symbol-graph spelling, `Proto.name(label:)`) of the three protocols, from the CURRENT sources:
    the requirement block of each protocol plus the public members of every `extension Proto` (B-I1)."""
    paths = set()
    for path in sorted((root / 'Sources/VectorCore').rglob('*.swift')):
        masked = _mask(path.read_text())
        for proto in PROTOCOLS:
            for m in re.finditer(rf'\bprotocol\s+{proto}\b[^{{]*\{{', masked):
                end = block_end(masked, m.end() - 1)
                paths |= {f'{proto}.{x}' for x in members(masked[m.end():end], require_public=False)}
            for m in re.finditer(rf'^[ \t]*(?:@\w+(?:\([^)]*\))?\s+)*(?P<pub>public\s+)?extension\s+(?:VectorCore\.)?{proto}\b'
                                 rf'[^{{]*\{{', masked, re.M):
                end = block_end(masked, m.end() - 1)
                paths |= {f'{proto}.{x}' for x in members(masked[m.end():end], require_public=not m.group('pub'))}
    return paths


def tsv_paths(name):
    rows = [line.split('\t') for line in (ROOT / 'Scripts/ci/tests/D02' / name).read_text().splitlines()
            if line and not line.startswith('#') and not line.startswith(('path\t', 'access\t'))]
    return rows


def dispatch_rows(source):
    block = source[source.index('let d02DispatchRows'):source.index('\n]\n', source.index('let d02DispatchRows'))]
    return re.findall(r'\("((?:VectorProtocol|VectorFactory|IndexableVector)\.[^"]+)",\s*(\.dynamic|\.static|nil)', block)


def edge_rows(source):
    block = source[source.index('// BEGIN D02 EDGE ROWS'):source.index('// END D02 EDGE ROWS')]
    return re.findall(r'^\s*"((?:Vector|VectorMath)\.[^"]+)":', block, re.M)


# --- W4 (phase 5): every provider-hook read sits in a function the Operations hook table covers -------------------
HOOK_READ = re.compile(r'(?<![\w$])(?:Operations\.)?(computeProvider|simdProvider|linearAlgebraProvider)\b')
HOOK_DECL = re.compile(r'\bvar\s+$')
MEMBER_HEAD = re.compile(r'^[ \t]*(?:@\w+(?:\([^)]*\))?\s+)*(?:(?:public|private|internal|fileprivate|package|static|class|'
                         r'mutating|nonmutating|final|override|nonisolated)\s+)*'
                         r'(?:func\s+(?P<func>[^\s(<]+)|var\s+(?P<var>\w+)\s*:[^=\n{]*\{|(?P<init>init)\b)', re.M)
TYPE_HEAD = re.compile(r'^[ \t]*(?:@\w+(?:\([^)]*\))?\s+)*(?:(?:public|private|internal|fileprivate|package|final)\s+)*'
                       r'(?:extension|struct|class|enum|actor)\s+(?P<name>[\w.]+)', re.M)
_ROUTED = ('validatedNearest', 'streamNearest', 'topk_euclid_optimized', 'topk_cosine_fused_512', 'topk_dot_512',
           'computeDistances', 'batchKernelDistances_euclid', 'batchKernelDistances_cosineFused')
# (owner, enclosing function) of a hook read -> the public entries that reach it. Private Operations helpers are
# listed with the public entries that call them (findNearest, and findNearestBatch's default `.perQuery` route).
HOOK_SITES = {('Operations', m): ('Operations.findNearest', 'Operations.findNearestBatch') for m in _ROUTED}
HOOK_SITES.update({('Operations', m): (f'Operations.{m}',) for m in
                   ('findNearestBatch', 'distanceMatrix', 'normalize', 'centroid', 'statistics')})
HOOK_SITES.update({('Vector', m): (f'Vector {m}',) for m in
                   ('.*', './', 'mean', 'sum', 'softmax', 'clamped', 'l1Norm', 'lInfinityNorm', 'manhattanDistance', 'min',
                    'max', 'minElement', 'maxElement', 'clamp', 'absoluteValue', 'squareRoot')})
HOOK_SITES.update({('DynamicVector', m): (f'DynamicVector {m}',) for m in ('min', 'max', 'clamped')})
# Public entries reached through another public entry (fix 1, A-M1): listed by hand (LIMITS).
HOOK_SITES[('Vector', 'lInfinityNorm')] = ('Vector lInfinityNorm', 'Vector chebyshevDistance')
HOOK_SITES[('PCAModel', 'fit')] = ('PCAModel.fit', 'Operations.pca', 'Operations.umap')


def _blocks(masked, head):
    """(name, open, close) for each match of `head` whose first `{` (at parenthesis depth 0) opens a block."""
    found = []
    for m in head.finditer(masked):
        depth, i = 0, m.end() - (1 if masked[m.end() - 1] == '{' else 0)
        while i < len(masked):
            ch = masked[i]
            if ch in '([':
                depth += 1
            elif ch in ')]':
                depth -= 1
            elif ch == '{' and depth == 0:
                break
            elif ch in ';}' and depth == 0:
                i = len(masked)
            i += 1
        if i < len(masked):
            name = next(g for g in (m.groupdict().get(k) for k in ('func', 'var', 'init', 'name')) if g)
            found.append((name, i, block_end(masked, i)))
    return found


def hook_read_sites(root):
    """[(file, line, hook, owner, member)] for every read of a provider hook in Sources/VectorCore (comments and
    string text masked, interpolated code kept: a read inside `\\(...)` is a read; declarations and `$hook`
    projections are not reads). owner/member are the innermost enclosing type (or extension) and function/computed
    property, or None."""
    sites = []
    for path in sorted((root / 'Sources/VectorCore').rglob('*.swift')):
        masked = _mask(path.read_text(), keep_interpolations=True)
        members, types = _blocks(masked, MEMBER_HEAD), _blocks(masked, TYPE_HEAD)
        for m in HOOK_READ.finditer(masked):
            if HOOK_DECL.search(masked[max(0, m.start() - 40):m.start()]):
                continue

            def innermost(blocks, pos=m.start()):
                inside = [b for b in blocks if b[1] < pos < b[2]]
                return max(inside, key=lambda b: b[1])[0] if inside else None
            sites.append((str(path.relative_to(root)), masked.count('\n', 0, m.start()) + 1, m.group(1),
                          innermost(types), innermost(members)))
    return sites


def hook_table(root):
    """{entry: hook cell} from the `## Provider hooks` table of the Operations documentation; an entry is
    `Operations.x` / `PCAModel.fit`, or `Vector m` / `DynamicVector m` for each backticked member of a row."""
    text = (root / 'Sources/VectorCore/Operations/Operations.swift').read_text()
    rows = {}
    section = text[text.index('/// ## Provider hooks'):text.index('public enum Operations')]
    for line in section.split('\n'):
        cells = [c.strip() for c in line[len('///'):].strip().strip('|').split('|')] if line.startswith('/// |') else []
        if len(cells) != 3 or cells[0] in ('Public entry', '---'):
            continue
        tokens = re.findall(r'`([^`]+)`', cells[0])
        if tokens[0] in ('Vector', 'DynamicVector'):
            for token in tokens[1:]:
                key = f'{tokens[0]} {token.split("(")[0]}'
                rows[key] = rows.get(key, '') + cells[1]
        else:
            rows[tokens[0]] = rows.get(tokens[0], '') + cells[1]
    return rows


def hook_table_problems(root):
    problems, reached = [], set()
    rows = hook_table(root)
    for file, line, hook, owner, member in hook_read_sites(root):
        entries = HOOK_SITES.get((owner, member))
        if entries is None:
            problems.append(f'{file}:{line}: {hook} read in {owner}.{member}, which no hook-table entry covers')
            continue
        for entry in entries:
            reached.add(entry)
            if entry not in rows:
                problems.append(f'{file}:{line}: {hook} read reaches {entry}, which has no row in the hook table')
            elif f'`{hook}`' not in rows[entry]:
                problems.append(f'{file}:{line}: the {entry} row does not name {hook}')
    problems += [f'hook-table row {entry} has no hook read (stale row)' for entry in sorted(set(rows) - reached)]
    return problems


# --- W6 (phase 5 fix 1): the hook table's rows equal the route tests' expectRow calls ---------------------------
ROUTE_TESTS = 'Tests/ComprehensiveTests/D02ProviderHookRouteTests.swift'
SPY_HOOK = {'compute': 'computeProvider', 'batchKernel': 'computeProvider', 'simd': 'simdProvider',
            'linearAlgebra': 'linearAlgebraProvider'}
EXPECT_ROW = re.compile(r'expectRow\(\s*"([^"]*)"\s*,\s*\.(\w+)\s*,\s*\[([^\]]*)\]')


def hook_table_rows(root):
    """[(entry, hook, calls)] from the Operations hook table: the first cell without backticks, the backticked names
    of the second cell, and the backticked names of the last cell before its first `(` (the notes)."""
    text = (root / 'Sources/VectorCore/Operations/Operations.swift').read_text()
    rows = []
    for line in text[text.index('/// ## Provider hooks'):text.index('public enum Operations')].split('\n'):
        cells = [c.strip() for c in line[len('///'):].strip().strip('|').split('|')] if line.startswith('/// |') else []
        if len(cells) == 3 and cells[0] not in ('Public entry', '---'):
            rows.append((cells[0].replace('`', ''), ', '.join(re.findall(r'`([^`]+)`', cells[1])),
                         tuple(re.findall(r'`([^`]+)`', cells[2].split('(')[0]))))
    return rows


def route_test_rows(root):
    """[(entry, hook, calls)] from the expectRow calls in code of D02ProviderHookRouteTests (a call inside a comment
    is not a row; the string arguments are read from the unmasked text at the code positions)."""
    text = (root / ROUTE_TESTS).read_text()
    masked = _mask(text)
    assert len(masked) == len(text)
    rows = []
    for m in re.finditer(r'(?<!func )\bexpectRow\(', masked):   # string contents are masked; the declaration is not a row
        call = EXPECT_ROW.match(text, m.start())
        assert call, text[m.start():m.start() + 120]
        rows.append((call.group(1), SPY_HOOK.get(call.group(2), call.group(2)),
                     tuple(re.findall(r'"([^"]*)"', call.group(3)))))
    return rows


def hook_row_problems(root):
    table, tests = hook_table_rows(root), route_test_rows(root)
    problems = [f'duplicate {side} row {row[0]} / {row[1]}' for side, rows in (('table', table), ('test', tests))
                for row in sorted(set(rows)) if rows.count(row) > 1]
    table_keys, test_keys = {r[:2]: r[2] for r in table}, {r[:2]: r[2] for r in tests}
    for key in sorted(set(table_keys) | set(test_keys)):
        if key not in test_keys:
            problems.append(f'hook-table row {key[0]} / {key[1]} has no expectRow in {ROUTE_TESTS}')
        elif key not in table_keys:
            problems.append(f'expectRow {key[0]} / {key[1]} has no row in the Operations hook table')
        elif table_keys[key] != test_keys[key]:
            problems.append(f'row {key[0]} / {key[1]}: table {list(table_keys[key])} != test {list(test_keys[key])}')
    return problems


# --- H1/H2/H3, RD-R8/RD-R2 and RD-R0 (phase 5): removed members stay removed -----------------------------------
REMOVED_PUBLIC = re.compile(r'\b(?:bufferProvider|isSIMDEnabled|isReleaseMode|releaseOnly|performancePrecondition|'
                            r'memoryEfficiency|estimatePerformanceGain|reportMetrics|SIMDMemoryUtilities)\b|'
                            r'\bPERFORMANCE_METRICS\b|\bstruct\s+Configuration\b[^{]*\{[^}]*\bmemoryAlignment\b')
REMOVED_CONFIGURATION = re.compile(r'\b(?:Configuration|configuration|memoryAlignment)\b')


def vectorcore_configuration_hits(root):
    """Code (comments and strings masked) inside any `enum VectorCore` or `extension VectorCore` block of any file in
    Sources/VectorCore that names Configuration, configuration or memoryAlignment (H2; fix 1, A-M4)."""
    hits = []
    for path in sorted((root / 'Sources/VectorCore').rglob('*.swift')):
        masked = _mask(path.read_text(), keep_interpolations=True)
        for name, start, end in _blocks(masked, TYPE_HEAD):
            if name == 'VectorCore':
                hits += [f'{path.relative_to(root)}:{masked.count(chr(10), 0, m.start()) + 1}: {m.group()}'
                         for m in REMOVED_CONFIGURATION.finditer(masked, start, end)]
    return hits


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace('.py', ''), ROOT / 'Scripts/ci/tests' / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class D02CSurfaceGuards(unittest.TestCase):
    def test_public_headers_declare_exactly_the_retained_functions(self):
        declared = set()
        for header in sorted((C_ROOT / 'include').glob('*.h')):
            declared |= declared_c_functions(header.read_text())
        self.assertEqual(declared, RETAINED_C)

    def test_no_internal_names_in_public_headers(self):
        hits = [f'{h.name}: {m.group()}' for h in sorted((C_ROOT / 'include').glob('*.h'))
                for m in INTERNAL.finditer(strip_c_comments(h.read_text()))]
        self.assertEqual(hits, [])

    def test_no_wrapping_block_guards(self):
        hits = [f'{p.relative_to(ROOT)}:{n}' for p in sorted(C_ROOT.rglob('*')) if p.suffix in ('.c', '.h')
                for n, line in enumerate(p.read_text().split('\n'), 1) if OLD_GUARD.search(line)]
        self.assertEqual(hits, [])

    def test_int8_casts(self):
        casts = [f'{rel}:{n}' for rel in INT8_SOURCES for n, line in
                 enumerate(strip_c_comments((C_ROOT / rel).read_text()).split('\n'), 1) if '(int32_t)' in line]
        self.assertEqual(casts, [])
        operands = HELPER_CAST.findall(strip_c_comments((C_ROOT / 'common/vc_int8_accum.h').read_text()))
        self.assertEqual(sorted(op for op in operands if op not in ('a', 'b')), ['(UINT32_MAX - bits)', 'bits'])
        self.assertIn('if (bits <= (uint32_t)INT32_MAX)', (C_ROOT / 'common/vc_int8_accum.h').read_text())

    def test_no_hardware_or_speed_comments_in_touched_c_files(self):
        hits = [f'{rel}:{n}' for rel in ('common/vc_scalar.c', 'arm64/vc_arm64.c', 'x86_64/vc_x86.c',
                                          'include/VectorCoreC.h')
                for n, line in enumerate((C_ROOT / rel).read_text().split('\n'), 1) if HW_COMMENT.search(line)]
        self.assertEqual(hits, [])

    def test_c_detector_controls(self):
        self.assertTrue(OLD_GUARD.search('for (; i + 64 <= lanes; i += 64) {'))
        self.assertTrue(OLD_GUARD.search('for (;i+16<=lanes;i+=16)'))
        self.assertFalse(OLD_GUARD.search('for (; vc_int8_block_fits(lanes, i, 64); i += 64) {'))
        self.assertEqual(declared_c_functions('VC_EXPORT int32_t vc_l2sq_int8(const int8_t* a); // TODO\n'
                                              '// VC_EXPORT float vc_x(void);\n/* VC_EXPORT void vc_y(void); */'),
                         {'vc_l2sq_int8'})
        self.assertTrue(INTERNAL.search('extern int32_t vc_scalar_dot_int8(const int8_t* a);'))
        self.assertEqual(HELPER_CAST.findall('return (int32_t)acc; x = (int32_t)(a + b);'), ['acc', '(a + b)'])
        self.assertTrue(HW_COMMENT.search('// hides FMA latency (~4 cycles on M-series)'))
        self.assertTrue(HW_COMMENT.search('// 4 accumulators for ILP.'))
        self.assertFalse(HW_COMMENT.search('// 4 independent accumulators'))


class D02SwiftSurfaceGuards(unittest.TestCase):
    def test_static_dimension_requires_only_value(self):
        text = (ROOT / 'Sources/VectorCore/Core/Dimension.swift').read_text()
        self.assertEqual(static_dimension_requirements(text), ['static var value: Int { get }'])
        self.assertEqual(BUILTIN_ALIAS.findall(text), [])
        self.assertIn('public typealias Storage = ArrayStorage<DynamicDimension>', text)

    def test_no_storage_gate_in_sources_or_tests(self):
        # Code only: the text before a `//` comment (D02's own comments may name the removed gate).
        hits = [f'{p.relative_to(ROOT)}:{n}' for base in ('Sources', 'Tests') for p in sorted((ROOT / base).rglob('*.swift'))
                for n, line in enumerate(p.read_text().split('\n'), 1) if STORAGE_GATE.search(line.split('//')[0])]
        self.assertEqual(hits, [])

    def test_no_storage_alias_in_test_dimensions(self):
        hits = [f'{p.relative_to(ROOT)}:{line}: {header}' for p in sorted((ROOT / 'Tests').rglob('*.swift'))
                for line, header in storage_alias_in_conformers(p.read_text())]
        self.assertEqual(hits, [])

    def test_compute_device_has_no_probe_or_removed_member(self):
        text = (ROOT / 'Sources/VectorCore/Execution/ComputeDevice.swift').read_text()
        self.assertEqual([t for t in DEVICE_PROBES if t in text], [])
        self.assertEqual([t for t in REMOVED_DEVICE if re.search(rf'\b{t}\b', text)], [])
        declared = [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'Sources').rglob('*.swift'))
                    if re.search(r'\b(?:struct|enum|class)\s+DeviceCapabilities\b', p.read_text())]
        self.assertEqual(declared, [])

    def test_swift_detector_controls(self):
        self.assertTrue(STORAGE_GATE.search('extension Vector where D.Storage: VectorStorageOperations {'))
        self.assertTrue(STORAGE_GATE.search('func f<E: StaticDimension>(_: E.Type) where E.Storage : VectorStorageOperations {'))
        self.assertFalse(STORAGE_GATE.search('let s: Vector<D>.Storage'))
        self.assertEqual(len(storage_alias_in_conformers(
            'private enum P: StaticDimension { static let value = 0; typealias Storage = DimensionStorage<P, Float> }\n'
            'struct Q: StaticDimension {\n    static let value = 3\n    typealias Storage = Int\n}\n'
            'struct R: Codable { typealias Storage = Int }\nenum S: StaticDimension { static let value = 1 }\n')), 2)
        self.assertEqual(static_dimension_requirements(
            'public protocol StaticDimension: Sendable {\n    static var value: Int { get }\n'
            '    associatedtype Storage: VectorStorage\n}\n'),
            ['static var value: Int { get }', 'associatedtype Storage: VectorStorage'])

    def test_dispatch_matrix_lists_every_protocol_member(self):
        # B-I1: the member set comes from the CURRENT sources, and must equal both the base snapshot and the rows.
        live = protocol_members(ROOT)
        snapshot = {path: role for path, role in tsv_paths('protocol-members-base.tsv')}
        listed = dispatch_rows((ROOT / 'Tests/ComprehensiveTests/D02WitnessDispatchMatrixTests.swift').read_text())
        self.assertEqual(sorted(live), sorted(snapshot), 'protocol members changed: update the matrix and the TSV')
        self.assertEqual(sorted(path for path, _ in listed), sorted(live))
        self.assertEqual(len(live), 77)
        for path, dispatch in listed:
            if snapshot[path] == 'requirement':
                self.assertIn(dispatch, ('.dynamic', 'nil'), path)

    def test_member_scan_controls(self):
        # Must-fail fixture: a synthetic protocol and extension; internal and private members are not counted.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'Sources/VectorCore').mkdir(parents=True)
            (root / 'Sources/VectorCore/A.swift').write_text(
                'public protocol VectorFactory: VectorProtocol {\n    static func create(from s: [Float]) throws -> Self\n}\n'
                'public extension VectorFactory {\n    static func + (lhs: Self, rhs: Self) -> Self { lhs }\n'
                '    private func hidden() {}\n    // func commented() {}\n    var label: String { "x { y" }\n}\n'
                'extension VectorFactory {\n    func internalOnly() {}\n    public subscript(index: Int) -> Int { 0 }\n}\n')
            self.assertEqual(sorted(protocol_members(root)),
                             ['VectorFactory.+(_:_:)', 'VectorFactory.create(from:)', 'VectorFactory.label',
                              'VectorFactory.subscript(_:)'])
        # RR-M2 must-fail fixtures: a same-line attribute, the module-qualified spelling and a typealias member are each
        # found; a `package` member and a `.init()` default argument are not counted.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'Sources/VectorCore').mkdir(parents=True)
            (root / 'Sources/VectorCore/B.swift').write_text(
                '@available(macOS 14, *) public extension VectorProtocol {\n    func attributed() -> Int { 1 }\n}\n'
                'extension VectorCore.VectorProtocol {\n    public func qualified() -> Int { 2 }\n}\n'
                'public extension IndexableVector {\n    typealias Alias = Int\n'
                '    package func packaged() {}\n    func defaulted(_ x: Int = .init()) {}\n}\n')
            self.assertEqual(sorted(protocol_members(root)),
                             ['IndexableVector.Alias', 'IndexableVector.defaulted(_:)', 'VectorProtocol.attributed()',
                              'VectorProtocol.qualified()'])
        # Tree-copy control (mut-newmember): a new public extension member is found, so the comparison fails.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / 'Sources/VectorCore', root / 'Sources/VectorCore')
            target = root / 'Sources/VectorCore/Protocols/VectorProtocol.swift'
            text = target.read_text()
            anchor = 'public extension VectorProtocol {\n'
            self.assertIn(anchor, text)
            target.write_text(text.replace(anchor, anchor + '    func d02BrandNewMember() -> Int { 1 }\n', 1))
            live = protocol_members(root)
            self.assertIn('VectorProtocol.d02BrandNewMember()', live)
            self.assertNotEqual(sorted(live), sorted(path for path, _ in tsv_paths('protocol-members-base.tsv')))

    def test_reachability_consumer_calls_every_gated_symbol(self):
        module = load('D02RunConsumers.py')
        patterns = module.gated_symbol_patterns()
        source = (ROOT / 'Scripts/ci/tests/D02/Reachability.swift').read_text()
        self.assertEqual(len(patterns), 61)
        self.assertEqual(module.count_guard_missing(source, patterns), [])
        # Controls (A2-M3, mut-evasion): a removed call, a call moved into a comment, or into a string, is missing.
        self.assertEqual(module.count_guard_missing(source.replace('.squareRoot()', '.sqrtX()'), patterns),
                         ['Vector.squareRoot()'])
        evaded = (source.replace('    _ = unit.isNormalized\n', '    // _ = unit.isNormalized\n', 1)
                  .replace('    _ = a.sparsity(threshold: 0)\n', '    _ = "a.sparsity(threshold: 0)"\n', 1))
        self.assertNotEqual(evaded, source)
        self.assertEqual(module.count_guard_missing(evaded, patterns), ['Vector.isNormalized', 'Vector.sparsity(threshold:)'])

    def test_binding_check_controls(self):
        # A2-I2: the SILGen binding check accepts only Vector's own unconstrained members.
        module = load('D02RunConsumers.py')
        bound = []
        for path in module.gated_symbol_patterns():
            owner, name, labels = module._tsv_member(path)
            prefix = ('static ' if owner == 'VectorMath' else '') + f'VectorCore.{owner}.'
            if labels is None:
                bound.append(f'{prefix}{name}.getter : Swift.Float')
            elif name in ('.<', '.>'):
                bound.append(f'static {prefix}{name} infix(VectorCore.Vector<A>, VectorCore.Vector<A>) -> VectorCore.Vector<A>')
            else:
                args = ', '.join(('' if label == '_' else f'{label}: ') + 'Swift.Int' for label in labels)
                bound.append(f'{prefix}{name}({args}) -> Swift.Int')
        self.assertEqual(module.binding_failures(bound), ([], []))
        fallback_l1 = [d.replace('VectorCore.Vector.l1Norm.getter',
                                 '(extension in VectorCore):VectorCore.VectorProtocol.l1Norm.getter') for d in bound]
        self.assertEqual(module.binding_failures(fallback_l1)[0], ['Vector.l1Norm'])
        self.assertTrue(module.binding_failures(fallback_l1)[1])
        no_finite = [d for d in bound if 'Vector.isFinite.getter' not in d]
        self.assertEqual(module.binding_failures(no_finite, ['VectorProtocol.isFinite!getter']),
                         (['Vector.isFinite'], ['VectorProtocol.isFinite!getter']))
        constrained = [d.replace('VectorCore.Vector.chebyshevDistance', '(extension in VectorCore):VectorCore.Vector'
                                 '<A where A.Storage: VectorCore.VectorStorageOperations>.chebyshevDistance') for d in bound]
        self.assertEqual(module.binding_failures(constrained)[0], ['Vector.chebyshevDistance(to:)'])
        wrong_label = [d.replace('repeatingPattern(safe: Swift.Int)', 'repeatingPattern(Swift.Int)') for d in bound]
        self.assertIn('Vector.repeatingPattern(safe:)', module.binding_failures(wrong_label)[0])

    def test_member_backstop_controls(self):
        # RR-I2: the phase 5 symbol-graph backstop of D02RunConsumers.py fails on any difference from the graph.
        module = load('D02RunConsumers.py')
        base = sorted(row[0] for row in tsv_paths('protocol-members-base.tsv'))
        self.assertEqual(module.member_backstop_differences(base, base, base, list(base)), {})
        hidden = base + ['VectorProtocol.macroAdded()']          # e.g. a member a macro adds: the scan cannot see it
        found = module.member_backstop_differences(hidden, base, base, list(base))
        self.assertEqual(sorted(found), ['rows', 'scan', 'tsv'])
        self.assertEqual(found['scan']['onlyInGraph'], ['VectorProtocol.macroAdded()'])
        self.assertIn('rowsDuplicated', module.member_backstop_differences(base, base, base, list(base) + base[:1]))

    def test_degenerate_width_table_covers_every_gated_symbol(self):
        # A2-I1 class closure: the D02DegenerateWidthTests rows are exactly the 61 gated symbols.
        rows = edge_rows((ROOT / 'Tests/ComprehensiveTests/D02DegenerateWidthTests.swift').read_text())
        gated = [row[2] for row in tsv_paths('gated-public-base.tsv')]
        self.assertEqual(len(gated), 61)
        self.assertEqual(sorted(rows), sorted(gated))
        self.assertEqual(len(rows), len(set(rows)))

    def test_hook_reads_have_table_rows(self):
        # W4 (phase 5): every provider-hook read in Sources/VectorCore sits in a function mapped to the public entries
        # that reach it; each entry has a row in the Operations hook table naming that hook; every row is reached.
        self.assertEqual(hook_table_problems(ROOT), [])
        self.assertEqual({site[2] for site in hook_read_sites(ROOT)},
                         {'computeProvider', 'simdProvider', 'linearAlgebraProvider'})

    def test_hook_table_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / 'Sources/VectorCore', root / 'Sources/VectorCore')
            ops, math = (root / 'Sources/VectorCore/Operations/Operations.swift',
                         root / 'Sources/VectorCore/Operations/VectorMath.swift')
            ops_text, math_text = ops.read_text(), math.read_text()
            self.assertEqual(hook_table_problems(root), [])
            # A new reader in a function no entry covers.
            math.write_text(math_text + '\nextension Vector {\n    public var d02NewReader: Float '
                                        '{ Operations.simdProvider.sum(toArray()) }\n}\n')
            found = hook_table_problems(root)
            self.assertEqual(len(found), 1, found)
            self.assertIn('Vector.d02NewReader, which no hook-table entry covers', found[0])
            math.write_text(math_text)
            # A row removed, a row that stops naming its hook, and a stale row.
            row = next(line for line in ops_text.split('\n') if line.startswith('/// | `Operations.centroid`'))
            ops.write_text(ops_text.replace(row + '\n', '')
                           .replace('/// | `Operations.statistics`, two vectors | `simdProvider` |',
                                    '/// | `Operations.statistics`, two vectors | `computeProvider` |')
                           .replace('/// | `PCAModel.fit`, exact path |', '/// | `Operations.d02Stale` | `computeProvider` | x |\n'
                                                                          '/// | `PCAModel.fit`, exact path |'))
            found = hook_table_problems(root)
            self.assertTrue(any('reaches Operations.centroid, which has no row' in p for p in found), found)
            self.assertTrue(any('the Operations.statistics row does not name simdProvider' in p for p in found), found)
            self.assertIn('hook-table row Operations.d02Stale has no hook read (stale row)', found)
        # Declarations, `$hook` projections, comments and strings are not reads; a read is attributed to its function.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'Sources/VectorCore').mkdir(parents=True)
            (root / 'Sources/VectorCore/A.swift').write_text(
                'public enum Operations {\n    @TaskLocal public static var computeProvider: any P = Q()\n}\n'
                'extension Vector {\n    // Operations.simdProvider in a comment\n'
                '    public var mean: Float {\n        let s = "Operations.simdProvider"\n'
                '        let t = "\\(Operations.simdProvider.sum(toArray()))"\n'
                '        return Operations.$simdProvider.withValue(R()) { Operations.simdProvider.mean(toArray()) }\n'
                '    }\n}\n')
            self.assertEqual(hook_read_sites(root), [('Sources/VectorCore/A.swift', 8, 'simdProvider', 'Vector', 'mean'),
                                                     ('Sources/VectorCore/A.swift', 9, 'simdProvider', 'Vector', 'mean')])

    def test_hook_table_rows_equal_the_route_tests(self):
        # W6 (phase 5 fix 1, A-I1/A-M1 class closure): the Operations hook table and the expectRow calls of
        # D02ProviderHookRouteTests are equal row for row, as (entry, hook, exact ordered calls).
        self.assertEqual(hook_row_problems(ROOT), [])
        table = hook_table_rows(ROOT)
        self.assertEqual(len(table), len(route_test_rows(ROOT)))
        self.assertIn(('Vector softmax()', 'simdProvider', ('max', 'divide')), table)
        self.assertIn(('Vector chebyshevDistance(to:)', 'simdProvider', ('abs',)), table)
        self.assertIn(('Operations.umap, default .pca initialization', 'linearAlgebraProvider', ('svdThin',)), table)

    def test_hook_row_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ops, tests = root / 'Sources/VectorCore/Operations/Operations.swift', root / ROUTE_TESTS
            for path in (ops, tests):
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(ROOT / path.relative_to(root), path)
            ops_text, tests_text = ops.read_text(), tests.read_text()
            self.assertEqual(hook_row_problems(root), [])
            # Must pass: a backticked name inside the notes is not a call; an empty list is "none".
            self.assertIn(('Operations.findNearestBatch, default .perQuery route, one query', 'computeProvider',
                           ('parallelExecute', 'parallelReduce')), hook_table_rows(root))
            self.assertIn(('Vector lInfinityNorm, an input holding a NaN', 'simdProvider', ()), hook_table_rows(root))
            # Must fail: a dropped op (the A-I1 defect: softmax listed as `divide` only).
            ops.write_text(ops_text.replace('| `simdProvider` | `max`, `divide` |', '| `simdProvider` | `divide` |'))
            self.assertEqual(hook_row_problems(root),
                             ["row Vector softmax() / simdProvider: table ['divide'] != test ['max', 'divide']"])
            # Must fail: an extra table row.
            first = '/// | `Vector` `softmax()`'
            ops.write_text(ops_text.replace(first, '/// | `Vector` `d02Extra` | `simdProvider` | `abs` |\n' + first))
            self.assertEqual(hook_row_problems(root),
                             [f'hook-table row Vector d02Extra / simdProvider has no expectRow in {ROUTE_TESTS}'])
            ops.write_text(ops_text)
            # Must fail: a missing test (the chebyshevDistance expectRow commented out).
            call = 'try await expectRow("Vector chebyshevDistance(to:)"'
            tests.write_text(tests_text.replace(call, '// ' + call))
            self.assertEqual(hook_row_problems(root),
                             [f'hook-table row Vector chebyshevDistance(to:) / simdProvider has no expectRow in {ROUTE_TESTS}'])
            # Must fail: an expectRow under the wrong hook is a missing row on both sides.
            tests.write_text(tests_text.replace('expectRow("Vector sum", .simd,', 'expectRow("Vector sum", .compute,'))
            self.assertEqual(sorted(hook_row_problems(root)),
                             ['expectRow Vector sum / computeProvider has no row in the Operations hook table',
                              f'hook-table row Vector sum / simdProvider has no expectRow in {ROUTE_TESTS}'])

    def test_removed_members_stay_removed(self):
        # H1/H2/H3 and RD-R8/RD-R2 (phase 5): no code in Sources (comments and strings masked) declares or uses a
        # removed name; SIMDMemoryUtilities.swift (RD-R0, phase 1 item 16) stays deleted; no global alignment setting
        # replaces memoryAlignment; no VectorCore block in Sources/VectorCore names Configuration (fix 1, A-M4).
        hits = [f'{p.relative_to(ROOT)}:{masked.count(chr(10), 0, m.start()) + 1}: {m.group()[:40]}'
                for p in sorted((ROOT / 'Sources').rglob('*.swift'))
                for masked in [_mask(p.read_text(), keep_interpolations=True)]
                for m in REMOVED_PUBLIC.finditer(masked)]
        self.assertEqual(hits, [])
        self.assertFalse((ROOT / 'Sources/VectorCore/Utilities/SIMDMemoryUtilities.swift').exists())
        core = _mask((ROOT / 'Sources/VectorCore/VectorCore.swift').read_text())
        self.assertIsNone(REMOVED_CONFIGURATION.search(core))
        self.assertEqual(vectorcore_configuration_hits(ROOT), [])
        globals_ = [f'{p.relative_to(ROOT)}' for p in sorted((ROOT / 'Sources').rglob('*.swift'))
                    if re.search(r'(?m)^public\s+(?:var|let)\s+\w*[aA]lignment\b', _mask(p.read_text()))]
        self.assertEqual(globals_, [])
        # Controls: each removed spelling is matched; the kept BufferProvider API and memoryFootprint are not.
        for text in ('@TaskLocal public static var bufferProvider: any BufferProvider = SwiftBufferPool.shared',
                     'public var isSIMDEnabled: Bool {', 'releaseOnly { }', 'performancePrecondition(x)',
                     'public var memoryEfficiency: String', 'estimatePerformanceGain(candidateCount: 1, dimension: 2)',
                     '#if PERFORMANCE_METRICS', 'internal enum SIMDMemoryUtilities {',
                     'public struct Configuration: Sendable {\n var memoryAlignment: Int = 64\n}'):
            self.assertIsNotNone(REMOVED_PUBLIC.search(text), text)
        for text in ('public protocol BufferProvider: Sendable {', 'public var memoryFootprint: Int',
                     'public struct Configuration: Sendable {\n public var minimumChunkSize: Int = 256\n}'):
            self.assertIsNone(REMOVED_PUBLIC.search(text), text)

    def test_vectorcore_configuration_controls(self):
        # A-M4 (fix 1): re-adding the removed global configuration in ANOTHER file of Sources/VectorCore fails (the
        # reviewer's H3-gap mutation); another type's Configuration and comment text do not.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'Sources/VectorCore').mkdir(parents=True)
            gap = root / 'Sources/VectorCore/BatchOperations.swift'
            gap.write_text('public enum BatchOperations {\n    public struct Configuration: Sendable {}\n}\n'
                           '// extension VectorCore { static let configuration }\n')
            self.assertEqual(vectorcore_configuration_hits(root), [])
            gap.write_text(gap.read_text() + 'extension VectorCore {\n'
                           '    public static let configuration = BatchOperations.Configuration()\n}\n')
            self.assertEqual(vectorcore_configuration_hits(root),
                             ['Sources/VectorCore/BatchOperations.swift:6: configuration',
                              'Sources/VectorCore/BatchOperations.swift:6: Configuration'])

    def test_changelog_names_the_phase5_removals_and_migration(self):
        # H5 and D9 (phase 5): the [Unreleased] Removed section names every member removed in phase 5, and the hook
        # entry points to BatchOperations' configuration actor and the explicit buffer APIs.
        text = (ROOT / 'CHANGELOG.md').read_text()
        start = text.index('## [Unreleased]')
        unreleased = text[start:text.index('\n## [', start + 1)]
        removed = unreleased[unreleased.index('### Removed'):]
        removed = removed[:removed.index('\n### ')]
        for name in ('`Operations.bufferProvider`', '`VectorCore.Configuration`', '`VectorCore.configuration`',
                     '`isSIMDEnabled`', '`isReleaseMode`', '`releaseOnly`', '`performancePrecondition`',
                     '`SoA.memoryEfficiency`', '`BatchKernels_SoA.estimatePerformanceGain`'):
            self.assertIn(name, removed)
        entry = next(block for block in re.split(r'\n(?=- )', removed) if '`Operations.bufferProvider`' in block)
        for name in ('`BatchOperations.updateConfiguration`', '`BufferProvider`', '`SwiftBufferPool`',
                     '`BufferConfiguration.alignment`'):
            self.assertIn(name, entry)

    def test_ir_signed_add_regex(self):
        # A1-M4: the IR scan also matches `add nuw nsw`; the module asserts its own controls at import.
        module = load('D02CContract.py')
        self.assertTrue(module.IR_SIGNED_ADD.search('%9 = add nuw nsw i32 %7, %8'))
        self.assertFalse(module.IR_SIGNED_ADD.search('%9 = add nuw i32 %7, %8'))

if __name__ == '__main__':
    unittest.main()
