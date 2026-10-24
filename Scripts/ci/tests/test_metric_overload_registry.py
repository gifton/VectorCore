"""Policy: every concrete `distance`/`batchDistance` declaration on a DistanceMetric type has a registry key.

WHY: T14 phase 1 found 30 public concrete overloads in Sources/VectorCore/Operations/OptimizedDistanceMetrics.swift
(`distance(_:_:)` and `batchDistance(query:candidates:)` for Vector512Optimized, Vector768Optimized and
Vector1536Optimized on five metrics). A call with concrete static types binds such an overload, while generic and
existential code binds the DistanceMetric witness, and 18 of the 30 returned different bits from the witness. T14 phase 2
routed the Euclidean and dot-product witnesses to the kernels the search APIs use and removed the 24 non-cosine
overloads; T14 phase 3 routed the cosine witness to CosineKernels' fused kernel, the arithmetic of the cosine search
routes, and removed the last 6, so the registry is empty. Any concrete overload is a second dispatch path for one
metric and type, so this test makes each one visible: it needs a key in
`MetricWitnessEquivalenceTests.concreteOverloadRegistry` (Tests/ComprehensiveTests/MetricWitnessEquivalenceTests.swift),
and that file's `registryKeysMapToCases` maps every key to its bit-equality cases or to the named pending list.

RULE: in Sources/, Benchmarks/ and Tests/ *.swift, a DistanceMetric type is DistanceMetric itself, a protocol whose
inheritance clause names a DistanceMetric type, or a struct/class/enum/actor whose declaration or extension inheritance
clause names one (`VectorCore.`-qualified names accepted; resolved to a fixed point across the whole tree, so refining
protocols and subclasses count). A header may be separated from its `{` by blank lines; a header that contains a
declaration keyword followed by a name (for example `import struct A.B` before the next declaration) or a lone `=` is
not a header. A member's `where` clause is read up to its body, the enclosing `}` or the next declaration, so blank
lines may precede it. A `func distance` or `func batchDistance` (also written with a backticked name) declared directly in
the body of such a type, of an extension of it (any `where` clause), or of such a protocol, at any access level, is
CONCRETE when it has no generic parameter clause, or when its `where` clause makes a generic parameter itself equal
to a type (`V == X` or `X == V`; a constraint on a member such as `V.Scalar == Float` does not count). Its key is
`<extended type>.<name>(<labels>) <parameter types>`, for example
`CosineDistance.distance(_:_:) Vector512Optimized, Vector512Optimized`. The live tree's concrete declarations must equal
the registry keys exactly: a declaration without a key fails, and a key without a declaration fails (staleness).

LIMITS (lexical heuristic, not a type checker): these are not seen: a conformer named through a typealias (for
example `typealias E = EuclideanDistance` then `extension E { func distance(...) }`); a conformance added by a macro or
declared outside the scanned roots; and members injected by an extension of a protocol that is not itself a
DistanceMetric type, including a protocol that DistanceMetric inherits (for example
`extension Sendable where Self == EuclideanDistance { func distance(...) }`). `#if` branches are not evaluated (a
declaration in any branch counts). Strings and comments are blanked with the shared `swift_lexing._mask` (its
documented limits apply). A same-type constraint written through a typealias of the generic parameter is not
recognized. The header stop words are a heuristic: a real header that contains one of them followed by a name, or a
lone `=`, is skipped. Parameter types are compared as written (whitespace collapsed).

CONTROLS (this file): must-FLAG fixtures (an extension overload, a type-body overload, a multi-line signature,
`extension DistanceMetric where Self == X`, an overload inside `#if`, a same-type-constrained generic, a refining
protocol's conformer, a qualified conformance, a subclass of a conformer class, a backticked name, a reversed
same-type constraint, blank lines before a type's `{` and before a member's `where`, and two that must yield exactly
their keys: an `import struct` before a declaration, and a generic requirement before a constrained one, which must
not borrow its `where`); must-PASS fixtures (the generic
witnesses, comments and strings, a non-metric type's `distance`, a nested type inside a metric, a local function);
a MUTATION control per rule; a tree-copy mutation control (an added unregistered overload fails, a removed registered one
is stale); the live tree must match the registry and the scanner must SEE the seven built-in conformers. No git history
is used, so the test passes out of repository.
"""

import importlib.util
import re
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location('swift_lexing', ROOT / 'Scripts/ci/swift_lexing.py')
lexing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lexing)

SCANNED_ROOTS = ('Sources', 'Benchmarks', 'Tests')
REGISTRY_FILE = 'Tests/ComprehensiveTests/MetricWitnessEquivalenceTests.swift'
REGISTRY_NAME = 'concreteOverloadRegistry'
STATUSES = {'covered'}
BUILTIN_CONFORMERS = {'EuclideanDistance', 'CosineDistance', 'DotProductDistance', 'ManhattanDistance',
                      'ChebyshevDistance', 'HammingDistance', 'MinkowskiDistance'}
ROOT_PROTOCOL = 'DistanceMetric'
NAMES = ('distance', 'batchDistance')
RULES = ('mask', 'extensions', 'type_bodies', 'protocol_bodies', 'transitive', 'same_type', 'qualified',
         'backtick_names', 'same_type_reversed', 'blank_line_headers', 'header_stop_words', 'member_tails')

DECLARATION = re.compile(r'(?<![\w.])(struct|class|enum|actor|protocol|extension)\s+((?:\w+\.)*\w+)')
MEMBER = re.compile(r'(?<![\w.])func\s+(\w+)')
MEMBER_BACKTICKED = re.compile(r'(?<![\w.])func\s+(?:`(\w+)`|(\w+))')
# A declaration keyword followed by a name, or a lone `=`: the text between a name and the next `{` runs into
# another declaration or statement, so it is not a type header (`: class, P` and `==` do not match).
HEADER_STOP = re.compile(r'(?<![\w.])(?:struct|class|enum|actor|protocol|extension|func|var|let|import|typealias|case)'
                         r'\s+[`A-Za-z_]|(?<![=!<>])=(?!=)')


def _matching(text, open_at, opening, closing):
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == opening:
            depth += 1
        elif text[i] == closing:
            depth -= 1
            if depth == 0:
                return i
    return len(text)


def _skip_angles(text, i):
    """Offset after a balanced `<...>` at i (`->` inside does not close)."""
    depth = 0
    while i < len(text):
        c = text[i]
        if c == '<':
            depth += 1
        elif c == '>' and text[i - 1] != '-':
            depth -= 1
            if depth == 0:
                return i + 1
        elif c in '{;':
            return i
        i += 1
    return i


def _split_top_level(text):
    parts, depth, current = [], 0, []
    for k, c in enumerate(text):
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
        elif c == '<' and k > 0 and (text[k - 1].isalnum() or text[k - 1] in '_?'):
            depth += 1
        elif c == '>' and depth > 0 and text[k - 1] != '-':
            depth -= 1
        if c == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(c)
    if ''.join(current).strip():
        parts.append(''.join(current))
    return parts


def _base_name(name, rules):
    """`VectorCore.DistanceMetric` -> `DistanceMetric` (qualified rule); generic arguments dropped."""
    name = name.split('<')[0].strip()
    if '.' in name:
        return name.rsplit('.', 1)[1] if 'qualified' in rules else None
    return name


def declarations(text, rules=RULES):
    """[(kind, name, inherited names, body start, body end)] for every type/extension/protocol declaration (masked)."""
    found = []
    for m in DECLARATION.finditer(text):
        kind, name = m.group(1), m.group(2)
        j = m.end()
        if j < len(text) and text[j] == '<':
            j = _skip_angles(text, j)
        brace = text.find('{', j)
        if brace < 0:
            continue
        header = text[j:brace]
        if ';' in header:
            continue
        if 'blank_line_headers' not in rules and '\n\n' in header:
            continue
        if 'header_stop_words' in rules and HEADER_STOP.search(header):
            continue
        inherited = []
        clause = header.split(' where ')[0].split('\nwhere ')[0]
        if ':' in clause:
            for part in _split_top_level(clause.split(':', 1)[1]):
                base = _base_name(part.strip(), rules)
                if base:
                    inherited.append(base)
        extended = _base_name(name, rules) if kind == 'extension' else name
        if extended is None:
            continue
        found.append((kind, extended, inherited, brace + 1, _matching(text, brace, '{', '}')))
    return found


def metric_types(sources, rules=RULES):
    """(protocol names, conformer names) of DistanceMetric types across all masked sources, at a fixed point."""
    decls = [d for text in sources for d in declarations(text, rules)]
    protocols, conformers = {ROOT_PROTOCOL}, set()
    changed = True
    while changed:
        changed = False
        for kind, name, inherited, _, _ in decls:
            known = protocols | conformers if 'transitive' in rules else {ROOT_PROTOCOL}
            if not any(i in known for i in inherited):
                continue
            target = protocols if kind == 'protocol' else conformers
            if name not in target:
                target.add(name)
                changed = True
    return protocols, conformers


def _members(text, start, end, rules=RULES):
    """`func distance`/`func batchDistance` declared at depth 0 of text[start:end]: (name, offset, generic, params, where)."""
    member = MEMBER_BACKTICKED if 'backtick_names' in rules else MEMBER
    depth, i, found = 0, start, []
    while i < end:
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        elif depth == 0:
            m = member.match(text, i)
            name = m and next(g for g in m.groups() if g)
            if m and name in NAMES:
                j = m.end()
                while j < end and text[j] in ' \t\n':
                    j += 1
                generic = ''
                if j < end and text[j] == '<':
                    k = _skip_angles(text, j)
                    generic, j = text[j + 1:k - 1], k
                while j < end and text[j] in ' \t\n':
                    j += 1
                if j < end and text[j] == '(':
                    close = _matching(text, j, '(', ')')
                    params = _split_top_level(text[j + 1:close])
                    if 'member_tails' in rules:
                        # The signature's tail (`-> T where ...`) ends at a body, the enclosing `}` or the next
                        # declaration, so blank lines may precede `where` and a requirement does not borrow the
                        # `where` clause of the requirement after it.
                        stop = HEADER_STOP.search(text, close + 1, end)
                        tail_end = min([p for p in (text.find('{', close, end), text.find('}', close, end)) if p >= 0]
                                       + ([stop.start()] if stop else []) + [end])
                    else:
                        tail_end = min([p for p in (text.find('{', close), text.find('\n\n', close)) if p >= 0] or [end])
                    tail = text[close + 1:tail_end]
                    where = tail.split('where', 1)[1] if re.search(r'\bwhere\b', tail) else ''
                    found.append((name, i, generic, params, where))
                    i = close + 1
                    continue
        i += 1
    return found


def _is_concrete(generic, where, rules):
    if not generic.strip():
        return True
    if 'same_type' not in rules:
        return False
    names = [p.split(':')[0].strip() for p in _split_top_level(generic)]
    if any(re.search(r'(?<![\w.])' + re.escape(n) + r'\s*==', where) for n in names if n):
        return True
    return 'same_type_reversed' in rules and any(
        re.search(r'==\s*' + re.escape(n) + r'(?![\w.])', where) for n in names if n)


def _key(type_name, name, params):
    labels, types = [], []
    for p in params:
        head, _, rest = p.partition(':')
        words = [w for w in head.split() if not w.startswith('@')]
        labels.append(words[0].strip('`') if words else '_')
        types.append(' '.join(rest.split('=')[0].split()))
    return f'{type_name}.{name}({"".join(label + ":" for label in labels)}) {", ".join(types)}'


def scan(sources, rules=RULES):
    """{key: (source index, line)} for every concrete distance/batchDistance on a DistanceMetric type in sources."""
    masked = [lexing._mask(s) if 'mask' in rules else s for s in sources]
    protocols, conformers = metric_types(masked, rules)
    found = {}
    for index, text in enumerate(masked):
        for kind, name, _, start, end in declarations(text, rules):
            if kind == 'protocol':
                if name not in protocols or 'protocol_bodies' not in rules:
                    continue
            elif kind == 'extension':
                if name not in protocols | conformers or 'extensions' not in rules:
                    continue
            elif name not in conformers or 'type_bodies' not in rules:
                continue
            for member, offset, generic, params, where in _members(text, start, end, rules):
                if _is_concrete(generic, where, rules):
                    found[_key(name, member, params)] = (index, text.count('\n', 0, offset) + 1)
    return found


def tree_sources(root):
    paths = [p for top in SCANNED_ROOTS if (Path(root) / top).is_dir() for p in sorted((Path(root) / top).rglob('*.swift'))]
    return paths, [p.read_text(encoding='utf-8') for p in paths]


def scan_tree(root):
    """({key: 'path:line'} of concrete declarations, protocols, conformers) for a tree."""
    paths, sources = tree_sources(root)
    found = scan(sources)
    masked = [lexing._mask(s) for s in sources]
    protocols, conformers = metric_types(masked)
    return ({k: f'{paths[i].relative_to(root).as_posix()}:{line}' for k, (i, line) in found.items()}, protocols, conformers)


def registry(root):
    """{key: status} from the registry literal in REGISTRY_FILE (read as text)."""
    text = (Path(root) / REGISTRY_FILE).read_text(encoding='utf-8')
    start = text.index(f'static let {REGISTRY_NAME}: [String: String] = [')
    open_at = text.index('[', text.index('=', start))
    body = text[open_at:_matching(text, open_at, '[', ']')]
    return dict(re.findall(r'"([^"\n]+)"\s*:\s*"([^"\n]+)"', body))


METRIC_HEADER = 'public protocol DistanceMetric<Scalar>: Sendable {\n    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Scalar\n}\n'

MUST_FLAG = {
    'extension_overload': ('struct M: DistanceMetric {}\nextension M {\n'
                           '    public func distance(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float { 0 }\n}\n',
                           ['M.distance(_:_:) Vector512Optimized, Vector512Optimized']),
    'type_body_overload': ('struct M: Sendable, DistanceMetric {\n'
                           '    func batchDistance(query: Vector768Optimized, candidates: [Vector768Optimized]) -> [Float] { [] }\n}\n',
                           ['M.batchDistance(query:candidates:) Vector768Optimized, [Vector768Optimized]']),
    'multiline_signature': ('struct M: DistanceMetric {}\nextension M {\n    @inlinable\n    public func batchDistance(\n'
                            '        query: Vector1536Optimized,\n        candidates: [Vector1536Optimized]\n'
                            '    ) -> [DistanceScore] {\n        []\n    }\n}\n',
                            ['M.batchDistance(query:candidates:) Vector1536Optimized, [Vector1536Optimized]']),
    'protocol_extension_where_self': ('extension DistanceMetric where Self == EuclideanDistance {\n'
                                      '    func distance(_ a: DynamicVector, _ b: DynamicVector) -> Float { 0 }\n}\n',
                                      ['DistanceMetric.distance(_:_:) DynamicVector, DynamicVector']),
    'inside_if_directive': ('#if arch(arm64)\nextension M {\n    func distance(_ a: V, _ b: V) -> Float { 0 }\n}\n#endif\n'
                            'struct M: DistanceMetric {}\n', ['M.distance(_:_:) V, V']),
    'same_type_generic': ('struct M: DistanceMetric {}\nextension M {\n    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float\n'
                          '        where V == Vector512Optimized { 0 }\n}\n', ['M.distance(_:_:) V, V']),
    'refining_protocol_conformer': ('protocol FastMetric: DistanceMetric {}\nstruct F: FastMetric {\n'
                                    '    func distance(_ a: X, _ b: X) -> Float { 0 }\n}\n', ['F.distance(_:_:) X, X']),
    'qualified_conformance': ('struct Q: VectorCore.DistanceMetric {\n    func distance(_ a: X, _ b: X) -> Float { 0 }\n}\n',
                              ['Q.distance(_:_:) X, X']),
    'subclass_of_conformer': ('class Base: DistanceMetric {}\nfinal class Sub: Base {\n'
                              '    func distance(_ a: X, _ b: X) -> Float { 0 }\n}\n', ['Sub.distance(_:_:) X, X']),
    'backticked_name': ('struct M: DistanceMetric {}\nextension M {\n'
                        '    public func `distance`(_ a: Vector768Optimized, _ b: Vector768Optimized) -> Float { 0 }\n}\n',
                        ['M.distance(_:_:) Vector768Optimized, Vector768Optimized']),
    'reversed_same_type': ('struct M: DistanceMetric {}\nextension M {\n    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float'
                           ' where Vector768Optimized == V { 0 }\n}\n', ['M.distance(_:_:) V, V']),
    'blank_line_before_brace': ('struct M: DistanceMetric\n\n{\n    func distance(_ a: X, _ b: X) -> Float { 0 }\n}\n'
                                'extension M\n\n{\n    func batchDistance(query: X, candidates: [X]) -> [Float] { [] }\n}\n',
                                ['M.batchDistance(query:candidates:) X, [X]', 'M.distance(_:_:) X, X']),
    'import_before_declaration': ('import struct Foo.Bar\n\nstruct M: DistanceMetric {\n'
                                  '    func distance(_ a: X, _ b: X) -> Float { 0 }\n}\n', ['M.distance(_:_:) X, X']),
    'blank_line_before_member_where': ('struct M: DistanceMetric {}\nextension M {\n'
                                       '    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float\n\n'
                                       '        where V == Vector768Optimized\n    { 0 }\n}\n', ['M.distance(_:_:) V, V']),
    'requirement_then_constrained': ('protocol FastMetric: DistanceMetric {\n'
                                     '    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float\n'
                                     '    func batchDistance<V: VectorProtocol>(query: V, candidates: [V]) -> [Float] where V == X\n}\n',
                                     ['FastMetric.batchDistance(query:candidates:) V, [V]']),
}

MUST_PASS = {
    'generic_witnesses': ('struct M: DistanceMetric {\n    @inlinable\n'
                          '    public func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float { 0 }\n'
                          '    public func batchDistance<Vector: VectorProtocol>(\n        query: Vector,\n'
                          '        candidates: [Vector]\n    ) -> [Float] where Vector.Scalar == Float { [] }\n}\n'
                          + METRIC_HEADER),
    'comments_and_strings': ('struct M: DistanceMetric {}\n'
                             '// extension M { func distance(_ a: X, _ b: X) -> Float { 0 } }\n'
                             '/* extension M {\n func batchDistance(query: X, candidates: [X]) -> [Float] { [] }\n } */\n'
                             'let s = "extension M { func distance(_ a: X, _ b: X) -> Float { 0 } }"\n'),
    'non_metric_type': ('extension VectorProtocol where Scalar == Float {\n'
                        '    func distance(to other: Self, metric: SupportedDistanceMetric = .euclidean) -> Float { 0 }\n}\n'
                        'struct Ruler {\n    func distance(_ a: X, _ b: X) -> Float { 0 }\n}\n'),
    'nested_type_in_metric': ('struct M: DistanceMetric {\n    struct Helper {\n'
                              '        func distance(_ a: X, _ b: X) -> Float { 0 }\n    }\n}\n'),
    'local_function': ('struct M: DistanceMetric {\n    func run() -> Float {\n'
                       '        func distance(_ a: X, _ b: X) -> Float { 0 }\n        return distance(X(), X())\n    }\n}\n'),
}

# rule -> (control name, keys found when the rule is disabled)
MUTATION_CONTROLS = {
    'mask': ('comments_and_strings', 2),
    'extensions': ('extension_overload', 0),
    'type_bodies': ('type_body_overload', 0),
    'protocol_bodies': ('protocol_body_requirement', 0),
    'transitive': ('refining_protocol_conformer', 0),
    'same_type': ('same_type_generic', 0),
    'qualified': ('qualified_conformance', 0),
    'backtick_names': ('backticked_name', 0),
    'same_type_reversed': ('reversed_same_type', 0),
    'blank_line_headers': ('blank_line_before_brace', 0),
    'header_stop_words': ('import_before_declaration', 2),
    'member_tails': ('requirement_then_constrained', 2),
}

EXTRA_FLAG = {
    'protocol_body_requirement': ('protocol FastMetric: DistanceMetric {\n'
                                  '    func distance(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float\n}\n',
                                  ['FastMetric.distance(_:_:) Vector512Optimized, Vector512Optimized']),
}


def _fixture(name):
    return ({**MUST_FLAG, **EXTRA_FLAG}.get(name) or (MUST_PASS[name], None))[0]


class MetricOverloadRegistryTests(unittest.TestCase):
    def test_must_flag_controls(self):
        for name, (source, keys) in {**MUST_FLAG, **EXTRA_FLAG}.items():
            with self.subTest(name):
                self.assertEqual(sorted(scan([source])), keys)

    def test_must_pass_controls(self):
        for name, source in MUST_PASS.items():
            with self.subTest(name):
                self.assertEqual(scan([source]), {})

    def test_conformance_resolves_across_files(self):
        found = scan(['extension EuclideanDistance {\n    func distance(_ a: X, _ b: X) -> Float { 0 }\n}\n',
                      'public struct EuclideanDistance: DistanceMetric {}\n'])
        self.assertEqual(list(found), ['EuclideanDistance.distance(_:_:) X, X'])
        self.assertEqual(found['EuclideanDistance.distance(_:_:) X, X'], (0, 2))

    def test_every_rule_has_a_mutation_control(self):
        self.assertEqual(set(MUTATION_CONTROLS), set(RULES))

    def test_each_rule_bites_by_mutation(self):
        for rule, (name, when_disabled) in MUTATION_CONTROLS.items():
            with self.subTest(rule):
                source = _fixture(name)
                enabled = len(scan([source]))
                disabled = len(scan([source], rules=tuple(r for r in RULES if r != rule)))
                self.assertNotEqual(enabled, disabled, 'mutating %s did not change %s' % (rule, name))
                self.assertEqual(disabled, when_disabled)

    def test_registry_statuses_and_phase3_contents(self):
        entries = registry(ROOT)
        self.assertTrue(set(entries.values()) <= STATUSES, entries)
        # T14 phase 3 removed the last six (cosine) overloads, so no concrete overload remains.
        self.assertEqual(entries, {})

    def test_live_tree_matches_the_registry_and_the_scanner_sees_the_conformers(self):
        found, protocols, conformers = scan_tree(ROOT)
        keys = set(registry(ROOT))
        self.assertEqual({k: v for k, v in found.items() if k not in keys}, {}, 'unregistered concrete overloads')
        self.assertEqual(sorted(keys - set(found)), [], 'stale registry keys')
        # Visibility floor: the scanner resolves the seven built-in conformers and the root protocol.
        self.assertTrue(BUILTIN_CONFORMERS <= conformers, sorted(BUILTIN_CONFORMERS - conformers))
        self.assertIn(ROOT_PROTOCOL, protocols)

    def test_tree_copy_mutations_fail(self):
        with tempfile.TemporaryDirectory() as name:
            copy = Path(name) / 'tree'
            shutil.copytree(ROOT / 'Sources', copy / 'Sources')
            (copy / REGISTRY_FILE).parent.mkdir(parents=True)
            shutil.copy(ROOT / REGISTRY_FILE, copy / REGISTRY_FILE)
            found, _, _ = scan_tree(copy)
            self.assertEqual(set(found), set(registry(copy)))
            # An added unregistered overload is reported with its location.
            added = copy / 'Sources/VectorCore/Operations/T14Mutation.swift'
            added.write_text('extension EuclideanDistance {\n    public func distance(_ a: Vector512Optimized, '
                             '_ b: Vector512Optimized) -> Float { 0 }\n}\n')
            found, _, _ = scan_tree(copy)
            self.assertEqual({k: v for k, v in found.items() if k not in registry(copy)},
                             {'EuclideanDistance.distance(_:_:) Vector512Optimized, Vector512Optimized':
                              'Sources/VectorCore/Operations/T14Mutation.swift:2'})
            added.unlink()
            # A registered overload is accepted; removing it leaves its key stale. The live registry is empty since
            # T14 phase 3, so the copy registers a fixture overload and then removes it.
            key = 'CosineDistance.distance(_:_:) Vector768Optimized, Vector768Optimized'
            registry_path = copy / REGISTRY_FILE
            text = registry_path.read_text()
            empty = f'static let {REGISTRY_NAME}: [String: String] = [:]'
            self.assertEqual(text.count(empty), 1)
            registry_path.write_text(text.replace(empty, f'static let {REGISTRY_NAME}: [String: String] = ["{key}": "covered"]'))
            added.write_text('extension CosineDistance {\n    public func distance(_ a: Vector768Optimized, '
                             '_ b: Vector768Optimized) -> Float { 0 }\n}\n')
            found, _, _ = scan_tree(copy)
            self.assertEqual(set(found), {key})
            self.assertEqual(set(registry(copy)), {key})
            added.unlink()
            found, _, _ = scan_tree(copy)
            self.assertEqual(sorted(set(registry(copy)) - set(found)), [key])


if __name__ == '__main__':
    unittest.main()
