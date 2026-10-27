"""Policy: no module extension declares a member that shadows a curated stdlib member of the extended type.

WHY: VectorCore's `Sources/VectorCore/Utilities/BinaryFormat.swift` declared
`extension UnsafeRawPointer { func loadUnaligned<T>(as:) }`, whose body allocated, copied and freed on every call. Inside
the module it won overload resolution against the stdlib's allocation-free `loadUnaligned(fromByteOffset:as:)` (whose
`fromByteOffset` defaults to 0, so a call written `loadUnaligned(as:)` matches both), so every 8-lane load in the FP16
kernels heap-allocated 16 bytes. A second extension redeclared the stdlib's `UnsafeRawBufferPointer.loadUnaligned(
fromByteOffset:as:)` signature and forwarded to the first. T12 phase 2 deleted both (its emitted-code evidence, kept outside
this repository, found no caller of the second: the serialization reads bind to the stdlib's BitwiseCopyable overload; not
verified by this test). `Sources/VectorCore/Platform/SIMDExtensions.swift`'s `sum()` (Scalar == Float and Scalar == Double)
is a second instance of the class (a module member that shadows `SIMD.sum()`), so this test closes the class, not only
pointer allocation (T12 phase 1 root ruling (a)). T14 reviewed that `sum()` and kept it (T14 phase 1 root ruling (c));
Tests/ComprehensiveTests/SIMDSumSemanticsTests.swift pins its semantics.

RULE: in Sources/, Benchmarks/ and Tests/ *.swift, a `func`, `var` or `let` declared directly in the body of an
`extension` of a covered stdlib type (FAMILIES; `Swift.`-qualified names, a generic argument clause such as
`extension SIMD8<UInt16>`, attributes, access modifiers and `where`/conformance clauses are accepted) is a finding when
its base name is curated for that type's family (CURATED) and
  * func: one of its call shapes equals one of the curated stdlib call shapes for that name. A call shape is the sequence
    of argument labels (`_` for none) with any subset of the defaulted parameters omitted: the stdlib's
    `loadUnaligned(fromByteOffset: Int = 0, as:)` has the shapes (fromByteOffset, as) and (as), so the deleted
    `loadUnaligned(as:)` collides on (as). The CURATED tables list call shapes already expanded this way.
  * var/let: always (a property that reuses a curated member name is flagged whatever its type).
Reviewed exceptions live in ALLOWLIST, pinned to exactly the two SIMDExtensions.swift `sum()` declarations with the
reason from T14's semantic review; an entry that no longer matches a finding fails (staleness).

CURATED is a hand-written table (root ruling (a) minimum: the four raw-pointer types with load, loadUnaligned,
storeBytes, copyMemory, bindMemory, withMemoryRebound, assumingMemoryBound; SIMD with sum, min, max, replacing,
wrappedSum; plus the related members listed). A family uses the UNION of its types' shapes, so an extension of the
immutable pointer types that declares a mutable-only name such as storeBytes is also flagged (conservative). The
shapes were checked against the Swift.swiftinterface of the macOS 15.2, 26.0 and 27.0 SDKs by T12 phase 2's evidence
tool (T12StdlibMembers.py, outside this repository; not re-checked by this test, which does not read any SDK), so a
member a later SDK adds is not covered until CURATED is extended.

LIMITS (lexical heuristic, not a type checker):
  * Only labels are compared, not parameter types or static-ness: a same-label member with different parameter types,
    or a static member named like an instance member, is flagged although Swift may treat it as a distinct overload
    (conservative; resolve by renaming or by a reviewed ALLOWLIST entry).
  * Not seen: a covered type named through a typealias (`extension RawPtr`); members injected through an extension of a
    protocol the stdlib type conforms to (`extension Collection { func min() }` reaches UnsafeRawBufferPointer);
    `init`, `subscript` and operator declarations; members of nested types; free functions; types outside FAMILIES;
    stdlib members outside CURATED.
  * `#if` branches are not evaluated (a member in any branch counts). Strings and comments are blanked with the shared
    `swift_lexing._mask` (its documented limits apply: raw strings, nested interpolation strings).
  * Generic clauses are balanced on `<`/`>` (a `>` after `-` is an arrow); a default value containing a bare `<` or `>`
    comparison can mis-split parameters.

CONTROLS (this file): must-FLAG fixtures (the two deleted BinaryFormat.swift extensions verbatim, a SIMD sum() outside
the allowlist, a concrete SIMD4 extension, qualified and generic-argument headers, a defaulted parameter, a property, a
multi-line signature, a member inside `#if`); must-PASS fixtures (a same-name member on a non-stdlib type, different
labels, an uncurated name, comments and strings, a local function, a nested type's member, call sites); a MUTATION
control per rule; the live tree must have no unreviewed finding and the scanner must SEE the allowlisted declarations.
No git history is used, so the test passes out of repository.
"""

import importlib.util
import itertools
import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location('swift_lexing', ROOT / 'Scripts/ci/swift_lexing.py')
lexing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lexing)

SCANNED_ROOTS = ('Sources', 'Benchmarks', 'Tests')
RULES = ('detect', 'mask', 'labels', 'defaults', 'direct_members', 'qualified', 'concrete_simd', 'generic_args',
         'properties')

FAMILIES = {
    'raw-pointer': ('UnsafeRawPointer', 'UnsafeMutableRawPointer', 'UnsafeRawBufferPointer', 'UnsafeMutableRawBufferPointer'),
    'simd': ('SIMD', 'SIMD2', 'SIMD3', 'SIMD4', 'SIMD8', 'SIMD16', 'SIMD32', 'SIMD64'),
}

# Curated stdlib members per family: name -> call shapes (label tuples, defaulted parameters expanded; '_' = no label).
CURATED = {
    'raw-pointer': {
        'load': {('fromByteOffset', 'as'), ('as',)},
        'loadUnaligned': {('fromByteOffset', 'as'), ('as',)},
        'storeBytes': {('of', 'toByteOffset', 'as'), ('of', 'as')},
        'copyMemory': {('from', 'byteCount'), ('from',)},
        'copyBytes': {('from', 'count'), ('from',)},
        'bindMemory': {('to', 'capacity'), ('to',)},
        'withMemoryRebound': {('to', 'capacity', '_'), ('to', '_')},
        'assumingMemoryBound': {('to',)},
        'initializeMemory': {('as', 'repeating', 'count'), ('as', 'from', 'count'), ('as', 'repeating'), ('as', 'from'),
                             ('as', 'fromContentsOf'), ('as', 'at', 'count', 'to'), ('as', 'at', 'to'), ('as', 'count', 'to'),
                             ('as', 'to')},
        'moveInitializeMemory': {('as', 'from', 'count'), ('as', 'fromContentsOf')},
    },
    'simd': {
        'sum': {()},
        'wrappedSum': {()},
        'min': {()},
        'max': {()},
        'replacing': {('with', 'where')},
        'replace': {('with', 'where')},
        'squareRoot': {()},
        'rounded': {('_',)},
        'round': {('_',)},
        'addProduct': {('_', '_')},
        'addingProduct': {('_', '_')},
        'clamp': {('lowerBound', 'upperBound')},
        'clamped': {('lowerBound', 'upperBound')},
    },
}

SUM_REASON = ('semantic review done in T14: kept; lanes added in index order from +0, the order of the Swift 6.4 stdlib '
              'member, which starts from -0 instead, so only an all-(-0) vector differs (+0 here); pinned by '
              'SIMDSumSemanticsTests')
# (path, extended type with its where clause as written, member signature) -> reason. Pinned verbatim by
# test_allowlist_is_pinned_to_the_ruled_entries. The reason records T14's semantic review (T14 phase 1 root ruling (c):
# keep). Its order and start follow from the adjacent declarations (`self.indices.reduce(Scalar.zero) { $0 + self[$1] }`);
# the comparison with the stdlib member is SIMDSumSemanticsTests.stdlibMemberDiffersOnlyByItsNegativeZeroStart, which this
# test does not run.
ALLOWLIST = {
    ('Sources/VectorCore/Platform/SIMDExtensions.swift', 'SIMD where Scalar == Float', 'sum()'): SUM_REASON,
    ('Sources/VectorCore/Platform/SIMDExtensions.swift', 'SIMD where Scalar == Double', 'sum()'): SUM_REASON,
}

EXTENSION = re.compile(r'(?<![\w.])extension\s+')
IDENT = re.compile(r'`?([A-Za-z_]\w*)`?')
MEMBER = re.compile(r'(?<![\w.])(func|var|let)\s+(`?[A-Za-z_]\w*`?)')


def matching_brace(text, open_at):
    """(body start, body end) for the `{` at open_at in masked text; body end is the offset of the matching `}`."""
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return open_at + 1, i
    return open_at + 1, len(text)


def _skip_generic_clause(text, i):
    """Offset after a balanced `<...>` starting at i (`->` inside does not close)."""
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


def _split_top_level(params):
    parts, depth, current = [], 0, []
    for k, c in enumerate(params):
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
        elif c == '<' and k > 0 and (params[k - 1].isalnum() or params[k - 1] in '_?'):
            depth += 1
        elif c == '>' and depth > 0 and params[k - 1] != '-':
            depth -= 1
        if c == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(c)
    if ''.join(current).strip():
        parts.append(''.join(current))
    return parts


def _parameter(text):
    """(label, defaulted) of one parameter declaration such as `fromByteOffset offset: Int = 0`."""
    head, _, rest = text.partition(':')
    words = [w for w in head.split() if not w.startswith('@')]
    label = words[0].strip('`') if words else '_'
    depth, defaulted = 0, False
    for k, c in enumerate(rest):
        if c in '([{<':
            depth += 1
        elif c in ')]}' or (c == '>' and rest[k - 1] != '-'):
            depth -= 1
        elif c == '=' and depth <= 0 and rest[k - 1] not in '=!<>' and rest[k + 1:k + 2] != '=':
            defaulted = True
            break
    return label, defaulted


def call_shapes(parameters, rules=RULES):
    """Every label sequence a call can use: each subset of defaulted parameters may be omitted."""
    parsed = [_parameter(p) for p in parameters]
    if 'defaults' not in rules:
        return {tuple(label for label, _ in parsed)}
    optional = [i for i, (_, defaulted) in enumerate(parsed) if defaulted][:10]
    shapes = set()
    for count in range(len(optional) + 1):
        for omitted in itertools.combinations(optional, count):
            shapes.add(tuple(label for i, (label, _) in enumerate(parsed) if i not in omitted))
    return shapes


def direct_members(text, start, end, rules=RULES):
    """Members declared at depth 0 of text[start:end] (masked): dicts with name, kind, shapes, signature, offset."""
    depth, i, found = 0, start, []
    while i < end:
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        elif depth == 0 or 'direct_members' not in rules:
            m = MEMBER.match(text, i)
            if m and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == '_')):
                kind, name = m.group(1), m.group(2).strip('`')
                j = m.end()
                if kind == 'func':
                    while j < end and text[j] in ' \t\n':
                        j += 1
                    if j < end and text[j] == '<':
                        j = _skip_generic_clause(text, j)
                    while j < end and text[j] in ' \t\n':
                        j += 1
                    if j < end and text[j] == '(':
                        close, level = j, 0
                        while close < end:
                            if text[close] == '(':
                                level += 1
                            elif text[close] == ')':
                                level -= 1
                                if level == 0:
                                    break
                            close += 1
                        params = _split_top_level(text[j + 1:close])
                        labels = [_parameter(p)[0] for p in params]
                        found.append({'kind': 'func', 'name': name, 'shapes': call_shapes(params, rules), 'offset': i,
                                      'signature': name + '(' + ''.join(label + ':' for label in labels) + ')'})
                        i = close + 1
                        continue
                else:
                    found.append({'kind': kind, 'name': name, 'shapes': set(), 'offset': i, 'signature': name})
                i = j
                continue
        i += 1
    return found


def extensions(source, rules=RULES):
    """(extended type, header text, member list) for every extension of a covered type in source."""
    text = lexing._mask(source) if 'mask' in rules else source
    covered = set(FAMILIES['raw-pointer']) | ({'SIMD'} if 'concrete_simd' not in rules else set(FAMILIES['simd']))
    result = []
    for m in EXTENSION.finditer(text):
        j = m.end()
        qualified = text.startswith('Swift.', j)
        if qualified:
            if 'qualified' not in rules:
                continue
            j += len('Swift.')
        name = IDENT.match(text, j)
        if not name or name.group(1) not in covered:
            continue
        after = name.end()
        if text.startswith('<', after) and 'generic_args' not in rules:
            continue
        brace = text.find('{', after)
        if brace < 0:
            continue
        header = ' '.join(text[j:brace].split())
        start, end = matching_brace(text, brace)
        result.append((name.group(1), header, direct_members(text, start, end, rules)))
    return result


def family_of(type_name):
    return next(f for f, names in FAMILIES.items() if type_name in names)


def scan(source, rules=RULES):
    """[(line, extended-type header, member signature)] for members that shadow a curated stdlib member."""
    if 'detect' not in rules:
        return []
    findings = []
    for type_name, header, members in extensions(source, rules):
        table = CURATED[family_of(type_name)]
        for member in members:
            if member['name'] not in table:
                continue
            if member['kind'] == 'func':
                hit = bool(member['shapes'] & table[member['name']]) if 'labels' in rules else True
            else:
                hit = 'properties' in rules
            if hit:
                line = source.count('\n', 0, member['offset']) + 1
                findings.append((line, header, member['signature']))
    return findings


def scan_tree(root, allowlist=ALLOWLIST):
    """({path: findings} not in the allowlist, allowlist keys matched, covered extension blocks seen)."""
    unreviewed, matched, blocks = {}, set(), 0
    for top in SCANNED_ROOTS:
        base = Path(root) / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob('*.swift')):
            relative = path.relative_to(root).as_posix()
            source = path.read_text(encoding='utf-8')
            blocks += len(extensions(source))
            for line, header, signature in scan(source):
                key = (relative, header, signature)
                if key in allowlist:
                    matched.add(key)
                else:
                    unreviewed.setdefault(relative, []).append((line, header, signature))
    return unreviewed, matched, blocks


# The two extensions deleted from Sources/VectorCore/Utilities/BinaryFormat.swift by T12 phase 2, verbatim.
DELETED_BINARYFORMAT_SHADOWS = '''// MARK: - UnsafeRawBufferPointer Extensions

extension UnsafeRawBufferPointer {
    /// Load unaligned value from buffer
    @inlinable
    func loadUnaligned<T>(fromByteOffset offset: Int, as type: T.Type) -> T {
        assert(offset >= 0 && offset + MemoryLayout<T>.size <= count)
        return self.baseAddress!.advanced(by: offset).loadUnaligned(as: type)
    }
}

extension UnsafeRawPointer {
    /// Load unaligned value from pointer
    @inlinable
    func loadUnaligned<T>(as type: T.Type) -> T {
        let buffer = UnsafeMutableRawPointer.allocate(
            byteCount: MemoryLayout<T>.size,
            alignment: MemoryLayout<T>.alignment
        )
        defer { buffer.deallocate() }

        buffer.copyMemory(from: self, byteCount: MemoryLayout<T>.size)
        return buffer.load(as: type)
    }
}
'''

MUST_FLAG = {
    'deleted_binaryformat_shadows': (DELETED_BINARYFORMAT_SHADOWS, 2),
    'simd_sum_outside_allowlist': ('extension SIMD where Scalar == Float {\n    func sum() -> Scalar { 0 }\n}\n', 1),
    'concrete_simd4': ('extension SIMD4 where Scalar == Float {\n    func sum() -> Float { self[0] + self[1] }\n}\n', 1),
    'qualified_type': ('extension Swift.UnsafeMutableRawPointer {\n'
                       '    func storeBytes<T>(of value: T, as type: T.Type) {}\n}\n', 1),
    'generic_args_header': ('extension SIMD8<UInt16> {\n    func wrappedSum() -> UInt16 { 0 }\n}\n', 1),
    'attributes_and_modifiers': ('@available(macOS 14, *)\npublic extension UnsafeRawBufferPointer {\n'
                                 '    @inlinable public func load<T>(as type: T.Type) -> T { fatalError() }\n}\n', 1),
    'defaulted_parameter': ('extension UnsafeRawPointer {\n'
                            '    func load<T>(fromByteOffset offset: Int = 0, as type: T.Type, checked: Bool = false) -> T {'
                            ' fatalError() }\n}\n', 1),
    'property': ('extension SIMD where Scalar: FloatingPoint {\n    var sum: Scalar { 0 }\n}\n', 1),
    'multiline_signature': ('extension UnsafeMutableRawBufferPointer {\n    func withMemoryRebound<T, Result>(\n'
                            '        to type: T.Type,\n        _ body: (UnsafeMutableBufferPointer<T>) throws -> Result\n'
                            '    ) rethrows -> Result { fatalError() }\n}\n', 1),
    'replacing_with_mask': ('extension SIMD where Scalar: FloatingPoint {\n'
                            '    func replacing(with other: Scalar, where mask: SIMDMask<MaskStorage>) -> Self { self }\n}\n', 1),
    'inside_if_directive': ('#if arch(arm64)\nextension SIMD where Scalar == Double {\n    func max() -> Scalar { 0 }\n}\n'
                            '#endif\n', 1),
    'two_members_one_extension': ('extension UnsafeRawPointer {\n    func bindMemory<T>(to type: T.Type, capacity count: Int)'
                                  ' -> UnsafePointer<T> { fatalError() }\n    func assumingMemoryBound<T>(to: T.Type)'
                                  ' -> UnsafePointer<T> { fatalError() }\n}\n', 2),
}

MUST_PASS = {
    'non_stdlib_type_same_name': ('struct ByteReader {}\nextension ByteReader {\n'
                                  '    func loadUnaligned<T>(as type: T.Type) -> T { fatalError() }\n'
                                  '    func sum() -> Float { 0 }\n}\n', 0),
    'different_labels': ('extension UnsafeRawPointer {\n'
                         '    func loadUnaligned<T>(at offset: Int, as type: T.Type) -> T { fatalError() }\n'
                         '    func load(fromOffset offset: Int) -> UInt32 { 0 }\n}\n', 0),
    'uncurated_name': ('extension UnsafeRawPointer {\n    func readLittleEndian32(at offset: Int) -> UInt32 { 0 }\n}\n'
                       'extension SIMD where Scalar: FloatingPoint {\n'
                       '    static func load(from pointer: UnsafePointer<Scalar>, count: Int, offset: Int) -> Self { Self() }\n}\n', 0),
    'comments_and_strings': ('// extension UnsafeRawPointer { func loadUnaligned<T>(as type: T.Type) -> T }\n'
                             '/* extension SIMD4 where Scalar == Float {\n func sum() -> Float { 0 }\n } */\n'
                             'let s = "extension SIMD { func sum() -> Scalar { 0 } }"\n', 0),
    'local_function': ('extension UnsafeRawPointer {\n    func helper() -> Int {\n'
                       '        func load(as type: Int.Type) -> Int { 0 }\n        return load(as: Int.self)\n    }\n}\n', 0),
    'nested_type_member': ('extension UnsafeRawPointer {\n    struct Box {\n'
                           '        func load<T>(as type: T.Type) -> T { fatalError() }\n    }\n}\n', 0),
    'call_sites': ('func f(p: UnsafeRawPointer, v: SIMD4<Float>) -> Float {\n'
                   '    let x = p.loadUnaligned(as: UInt32.self)\n    return Float(x) + v.sum()\n}\n', 0),
    'module_protocol_named_like_stdlib': ('extension SIMDStorage where Element == SIMD4<Float> {\n'
                                          '    func sum() -> Float { 0 }\n}\n', 0),
}

# rule -> (control name, findings when the rule is disabled)
MUTATION_CONTROLS = {
    'detect': ('deleted_binaryformat_shadows', 0),
    'mask': ('comments_and_strings', 3),
    'labels': ('different_labels', 2),
    'defaults': ('defaulted_parameter', 0),
    'direct_members': ('nested_type_member', 1),
    'qualified': ('qualified_type', 0),
    'concrete_simd': ('concrete_simd4', 0),
    'generic_args': ('generic_args_header', 0),
    'properties': ('property', 0),
}


def _fixture(name):
    return (MUST_FLAG.get(name) or MUST_PASS[name])[0]


class NoStdlibMemberShadowTests(unittest.TestCase):
    def test_must_flag_controls(self):
        for name, (source, count) in MUST_FLAG.items():
            with self.subTest(name):
                self.assertEqual(len(scan(source)), count, scan(source))

    def test_must_pass_controls(self):
        for name, (source, _) in MUST_PASS.items():
            with self.subTest(name):
                self.assertEqual(scan(source), [])

    def test_deleted_binaryformat_shadows_are_named_exactly(self):
        found = scan(DELETED_BINARYFORMAT_SHADOWS)
        self.assertEqual([(header, signature) for _, header, signature in found],
                         [('UnsafeRawBufferPointer', 'loadUnaligned(fromByteOffset:as:)'),
                          ('UnsafeRawPointer', 'loadUnaligned(as:)')])

    def test_every_rule_has_a_mutation_control(self):
        self.assertEqual(set(MUTATION_CONTROLS), set(RULES))

    def test_each_rule_bites_by_mutation(self):
        for rule, (name, when_disabled) in MUTATION_CONTROLS.items():
            with self.subTest(rule):
                source = _fixture(name)
                enabled = len(scan(source))
                disabled = len(scan(source, rules=tuple(r for r in RULES if r != rule)))
                self.assertNotEqual(enabled, disabled, 'mutating %s did not change %s' % (rule, name))
                self.assertEqual(disabled, when_disabled)

    def test_call_shapes_expand_defaulted_parameters(self):
        self.assertEqual(call_shapes(['fromByteOffset offset: Int = 0', 'as type: T.Type']),
                         {('fromByteOffset', 'as'), ('as',)})
        self.assertEqual(call_shapes(['_ x: Int', 'y: [Int: Int] = [:]', 'z: (Int, Int) -> Bool = { $0 < $1 }']),
                         {('_', 'y', 'z'), ('_', 'y'), ('_', 'z'), ('_',)})
        self.assertEqual(call_shapes(['a: Bool = 1 == 2']), {('a',), ()})
        self.assertEqual(call_shapes([]), {()})

    def test_allowlist_is_pinned_to_the_ruled_entries(self):
        self.assertEqual(set(ALLOWLIST), {
            ('Sources/VectorCore/Platform/SIMDExtensions.swift', 'SIMD where Scalar == Float', 'sum()'),
            ('Sources/VectorCore/Platform/SIMDExtensions.swift', 'SIMD where Scalar == Double', 'sum()')})
        for reason in ALLOWLIST.values():
            self.assertEqual(reason, SUM_REASON)
        # Quoted verbatim, so a reworded reason fails here.
        self.assertEqual(SUM_REASON, 'semantic review done in T14: kept; lanes added in index order from +0, the order of '
                                     'the Swift 6.4 stdlib member, which starts from -0 instead, so only an all-(-0) '
                                     'vector differs (+0 here); pinned by SIMDSumSemanticsTests')

    def test_allowlist_is_honoured_only_for_its_own_path(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for relative in ('Sources/VectorCore/Platform/SIMDExtensions.swift', 'Sources/VectorCore/Other.swift'):
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                (root / relative).write_text('extension SIMD where Scalar == Float {\n    func sum() -> Scalar { 0 }\n}\n')
            unreviewed, matched, blocks = scan_tree(root)
            self.assertEqual(unreviewed, {'Sources/VectorCore/Other.swift': [(2, 'SIMD where Scalar == Float', 'sum()')]})
            self.assertEqual(len(matched), 1)
            self.assertEqual(blocks, 2)

    def test_live_tree_has_no_unreviewed_shadow_and_the_scanner_sees_the_allowlisted_ones(self):
        unreviewed, matched, blocks = scan_tree(ROOT)
        self.assertEqual(unreviewed, {}, unreviewed)
        # Staleness: every allowlist entry must still match a live declaration (this also shows the scanner reads the
        # SIMD extensions of SIMDExtensions.swift).
        self.assertEqual(matched, set(ALLOWLIST))
        # Visibility floor: SIMDExtensions.swift has three covered extension blocks.
        self.assertGreaterEqual(blocks, 3)


if __name__ == '__main__':
    unittest.main()
