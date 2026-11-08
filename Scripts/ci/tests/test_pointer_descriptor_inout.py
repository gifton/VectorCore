"""Review production inout mutable-pointer descriptors, independently of borrow/readback lint.

Scans Sources/**/*.swift, including inactive #if branches, callback/function types and protocol
requirements. Benchmarks/ is outside this scope even when it is a built target; Tests/ is also excluded.
The four POINTERS below are prohibited with zero production exceptions.
Source-local nominal aliases, simple generic aliases and same-type constraints are followed across
files. Names are conservatively joined by basename, not resolved with Swift lexical/module lookup:
ambiguous or cyclic relevant aliases require review. Composite types containing such a descriptor
also require review; this does not assert that rebinding every composite is a compiler trigger.

Scope: this is a declaration review gate, NOT a type checker, call-site analysis or proof of compiler
safety. Unaliased nominal/generic types are treated as scalar/state; imported aliases, unconstrained
generic specializations, protocol witness inference without explicit aliases/equalities, macros and
generated source outside Sources are not resolved. Unsupported relevant alias/type syntax fails
review rather than being accepted. Closure/function aliases containing explicit inout are scanned
at their declaration. Existing scalar/state inout and by-value pointer callbacks remain permitted.

Uses swift_lexing._mask with interpolation code retained and escaped identifiers normalized. Its
documented regex-literal/nested-block-comment limitations apply. Tests pin multiline, qualified,
escaped, comment/string and interpolation spellings. The retired SIMDStorage declaration is a
rejection fixture, including at its original path and scope. Intentional Tests canaries and
the existing lost-store scanner, compiler gates and runtime assertions are unchanged.
"""
import importlib.util
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location('pointer_inout_lexing', ROOT / 'Scripts/ci/swift_lexing.py')
lexing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lexing)
POINTERS = {'UnsafeMutablePointer', 'UnsafeMutableBufferPointer',
            'UnsafeMutableRawPointer', 'UnsafeMutableRawBufferPointer'}
LEGACY_PATH = 'Sources/VectorCore/Protocols/SIMDStorage.swift'
LEGACY_DECLARATION = ('mutating func withUnsafeMutableBufferPointer<R>(_ body: '
                         '(inout UnsafeMutableBufferPointer<Element>) throws -> R) rethrows -> R')
# Retained rejection fixtures only: no production declaration or path receives an exception.
LEGACY_PROTOCOL = ('public protocol SIMDStorage { associatedtype Element\n'
                      'var count: Int { get }\n'
                      'func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Element>) throws -> R) rethrows -> R\n'
                      + LEGACY_DECLARATION + '\nsubscript(index: Int) -> Element { get set }\n}')

TOKEN = re.compile(r'->|==|[A-Za-z_]\w*|[^\s]')
IDENTIFIER = re.compile(r'[A-Za-z_]\w*\Z')


def type_at(tokens, index):
    """Small type-expression reader; unknown/malformed syntax raises for the relevant use."""
    def token(at):
        return tokens[at] if at < len(tokens) else ''

    start = token(index)
    if start in ('inout', 'borrowing', 'consuming'):
        child, index = type_at(tokens, index + 1)
        return (start, (child,)), index
    if start == '[':
        child, index = type_at(tokens, index + 1)
        children = [child]
        if token(index) == ':':
            child, index = type_at(tokens, index + 1)
            children.append(child)
        if token(index) != ']':
            raise ValueError('unclosed collection type')
        node, index = ('collection', tuple(children)), index + 1
    elif start == '(':
        children, index = [], index + 1
        while token(index) != ')':
            if not token(index):
                raise ValueError('unclosed tuple type')
            if IDENTIFIER.fullmatch(token(index)) and token(index + 1) == ':':
                index += 2
            child, index = type_at(tokens, index)
            children.append(child)
            if token(index) != ',':
                break
            index += 1
        if token(index) != ')':
            raise ValueError('unsupported tuple type')
        node, index = ('tuple', tuple(children)), index + 1
    elif IDENTIFIER.fullmatch(start) and start not in ('some', 'any'):
        name, index = start, index + 1
        while token(index) == '.' and IDENTIFIER.fullmatch(token(index + 1)):
            name += '.' + token(index + 1)
            index += 2
        children = []
        if token(index) == '<':
            index += 1
            while True:
                child, index = type_at(tokens, index)
                children.append(child)
                if token(index) != ',':
                    break
                index += 1
            if token(index) != '>':
                raise ValueError('unsupported generic type')
            index += 1
        node = (name, tuple(children))
    else:
        raise ValueError('unsupported type syntax')
    while token(index) in ('?', '!'):
        node, index = ('optional', (node,)), index + 1
    while token(index) in ('async', 'throws', 'rethrows'):
        index += 1
        if token(index) == '(':
            raise ValueError('typed throws requires review')
    if token(index) == '->':
        result, index = type_at(tokens, index + 1)
        node = ('function', (node, result))
    return node, index


def alias_table(prepared):
    aliases = {}

    def add(name, value):
        aliases.setdefault(name.split('.')[-1], set()).add(value)

    for _, _, tokens, _ in prepared:
        for index, token in enumerate(tokens):
            if token in ('typealias', 'associatedtype') and index + 1 < len(tokens):
                name, cursor, parameters = tokens[index + 1], index + 2, []
                malformed = False
                if cursor < len(tokens) and tokens[cursor] == '<':
                    cursor += 1
                    while cursor < len(tokens) and tokens[cursor] not in ('>', '=', '{', '}'):
                        if tokens[cursor] != ',':
                            if not IDENTIFIER.fullmatch(tokens[cursor]):
                                malformed = True
                            parameters.append(tokens[cursor])
                        cursor += 1
                    cursor += cursor < len(tokens) and tokens[cursor] == '>'
                if token == 'associatedtype' and cursor < len(tokens) and tokens[cursor] == ':':
                    # Constraints do not erase an explicit default. Without a default the member
                    # remains abstract (witness inference is outside the documented source scope).
                    while cursor < len(tokens) and tokens[cursor] not in (
                            '=', '{', '}', ';', 'func', 'var', 'let', 'associatedtype', 'typealias'):
                        cursor += 1
                if cursor >= len(tokens) or tokens[cursor] != '=':
                    if token == 'typealias':
                        add(name, None)
                    continue
                try:
                    node, end = type_at(tokens, cursor + 1)
                    # Composition/where clauses are not silently truncated to their first type.
                    if malformed or (end < len(tokens) and tokens[end] in ('&', 'where', '.')):
                        raise ValueError('unsupported alias declaration')
                    add(name, (tuple(parameters), node))
                except ValueError:
                    add(name, None)
            # A conservative global equivalence edge, including associated-type paths and reverse form.
            if IDENTIFIER.fullmatch(token):
                left = None
                equality = False
                try:
                    left, end = type_at(tokens, index)
                    if end < len(tokens) and tokens[end] == '==':
                        equality = True
                        right, right_end = type_at(tokens, end + 1)
                        if right_end < len(tokens) and tokens[right_end] in ('.', '&'):
                            raise ValueError('unsupported equality type continuation')
                        if not left[1] and left[0].split('.')[-1] not in POINTERS:
                            add(left[0], ((), right))
                        elif not right[1] and right[0].split('.')[-1] not in POINTERS:
                            add(right[0], ((), left))
                except ValueError:
                    if equality and left is not None and not left[1]:
                        add(left[0], None)
    return aliases


def descriptor_risk(node, aliases, bindings=None, visiting=frozenset()):
    bindings = {} if bindings is None else bindings
    name, children = node
    short = name.split('.')[-1]
    if short in POINTERS:
        return 'mutable pointer descriptor'
    if short in bindings:
        return descriptor_risk(bindings[short], aliases, {}, visiting)
    if short in aliases:
        choices = aliases[short]
        if len(choices) != 1 or None in choices:
            return 'cyclic, ambiguous or unsupported relevant alias: ' + short
        parameters, target = next(iter(choices))
        if len(parameters) != len(children):
            return 'unresolved generic alias: ' + short
        # Substitute before recursing so nested identity aliases cannot capture each other's parameters.
        def substitute(value):
            n, args = value
            if n in bindings:
                return bindings[n]
            return n, tuple(substitute(arg) for arg in args)
        supplied = tuple(substitute(child) for child in children)
        expansion = (short, supplied)
        if expansion in visiting or len(visiting) >= 64:
            return 'cyclic or over-depth relevant alias: ' + short
        return descriptor_risk(target, aliases, dict(zip(parameters, supplied)), visiting | {expansion})
    for child in children:
        risk = descriptor_risk(child, aliases, bindings, visiting)
        if risk:
            return risk
    return None


def scan(files):
    prepared = []
    for path, source in sorted(files.items()):
        code = lexing._unbacktick(lexing._mask(source, keep_interpolations=True))
        matches = list(TOKEN.finditer(code))
        prepared.append((path, code, [m.group() for m in matches], matches))
    aliases, hits = alias_table(prepared), []
    for path, code, tokens, matches in prepared:
        for index, token in enumerate(tokens):
            if token != 'inout':
                continue
            try:
                node, end = type_at(tokens, index + 1)
                if end < len(tokens) and tokens[end] in ('.', '&'):
                    raise ValueError('unsupported inout type continuation')
                risk = descriptor_risk(node, aliases)
            except ValueError as error:
                end, risk = index + 2, str(error)
            if not risk:
                continue
            type_name = ''.join(tokens[index + 1:end])
            line = code.count('\n', 0, matches[index].start()) + 1
            hits.append(((path, type_name), line, risk))
    return hits


def problems(files):
    """Report every finding; there is no waiver/exception argument or production allowlist."""
    return [f'{key[0]}:{line}: {key[1]}: {reason}' for key, line, reason in scan(files)]


class PointerDescriptorInoutTests(unittest.TestCase):
    def test_direct_declarations_callbacks_and_spellings(self):
        for pointer in sorted(POINTERS):
            type_name = pointer + ('<Float>' if pointer in {'UnsafeMutablePointer', 'UnsafeMutableBufferPointer'} else '')
            for spelling in (type_name, 'Swift.' + type_name, '`' + pointer + '`' + type_name[len(pointer):],
                             '(' + type_name + ')', type_name.replace('<', '\n<')):
                for source in ('func write(_ p: inout TYPE) {}',
                               'protocol P { func borrow(_ body: (inout TYPE) throws -> Void) }',
                               'typealias Callback = (inout TYPE) -> Void',
                               'let callback = { (p: inout TYPE) in p }'):
                    with self.subTest(pointer=pointer, spelling=spelling, source=source):
                        self.assertTrue(scan({'Sources/X.swift': source.replace('TYPE', spelling)}))

    def test_alias_chains_generic_aliases_and_constraints_across_files(self):
        cases = [
            ('typealias P = UnsafeMutablePointer<Float>\ntypealias Q = P', 'Q'),
            ('typealias P<T> = UnsafeMutableBufferPointer<T>\ntypealias Q<U> = P<U>', 'Q<Float>'),
            ('typealias Identity<T> = T', 'Identity<UnsafeMutableRawPointer>'),
            ('typealias Identity<T> = T', 'Identity<Identity<UnsafeMutableRawPointer>>'),
            ('struct Scope { typealias P = Swift.UnsafeMutableRawBufferPointer }', 'Scope.P'),
            ('protocol P { associatedtype Buffer = UnsafeMutableRawPointer }', 'T.Buffer'),
            ('func other<T>(_: T) where T.Buffer == UnsafeMutablePointer<Float> {}', 'T.Buffer'),
            ('func other<T>(_: T) where UnsafeMutableRawPointer == T.Buffer {}', 'T.Buffer'),
        ]
        for declarations, use in cases:
            with self.subTest(declarations=declarations, use=use):
                files = {'Sources/A.swift': declarations, 'Sources/B.swift': f'func f(_ p: inout {use}) {{}}'}
                self.assertTrue(scan(files))

    def test_ambiguous_cycles_and_unsupported_relevant_aliases_require_review(self):
        for aliases in ('typealias P = Int\ntypealias P = UnsafeMutableRawPointer',
                        'typealias P = Q\ntypealias Q = P',
                        'typealias P<T> = P<Array<T>>',
                        'typealias P<T: Equatable> = UnsafeMutablePointer<T>',
                        'typealias P = @Unknown Unsupported'):
            with self.subTest(aliases=aliases):
                self.assertTrue(scan({'Sources/X.swift': aliases + '\nfunc f(_ p: inout P) {}'}))
        for source in ('typealias P<T> = P<Array<T>>\nfunc f(_ p: inout P<Int>) {}',
                       'func f(_ p: inout Container<Int>.Pointer) {}'):
            self.assertTrue(scan({'Sources/X.swift': source}))

    def test_scalar_state_by_value_and_documentation_controls(self):
        controls = [
            'func f(_ p: UnsafeMutablePointer<Float>) {}',
            'func f(_ body: (UnsafeMutableBufferPointer<Float>) -> Void) {}',
            'func f(_ x: inout Int, _ state: inout Configuration, _ a: inout [Float]) {}',
            'func f<T>(_ x: inout T, _ h: inout Hasher, _ s: inout Self) {}',
            'typealias Count = Int\nfunc f(_ n: inout Count) {}',
            'typealias List<T> = Array<T>\nfunc f(_ n: inout List<Float>) {}',
            'typealias Identity<T> = T\nfunc f(_ n: inout Identity<Identity<Int>>) {}',
            '// inout UnsafeMutableRawPointer\nlet text = "inout UnsafeMutableRawPointer"',
            '/* inout UnsafeMutableRawPointer */ let text = #"inout UnsafeMutableRawPointer"#',
            'let text = """\ninout UnsafeMutableRawPointer\n"""',
        ]
        for source in controls:
            with self.subTest(source=source):
                self.assertEqual(scan({'Sources/X.swift': source}), [])
        self.assertTrue(scan({'Sources/X.swift': 'let text = "\\({ (p: inout UnsafeMutableRawPointer) in 1 })"'}))

    def test_retired_requirement_is_rejected_without_path_or_scope_exceptions(self):
        variants = [LEGACY_PROTOCOL, LEGACY_PROTOCOL * 2,
                    'struct Decoy {\n' + LEGACY_PROTOCOL + '\n}',
                    LEGACY_PROTOCOL.replace('body:', 'other:'),
                    LEGACY_PROTOCOL.replace('Element', 'Float'),
                    LEGACY_PROTOCOL.replace('mutating func', 'public mutating func')]
        for suffix in (' where R: Equatable', '\n    where R: Equatable',
                       ' { fatalError() }', '\n    { fatalError() }'):
            variants.append(LEGACY_PROTOCOL.replace(LEGACY_DECLARATION, LEGACY_DECLARATION + suffix))
        variants.append(LEGACY_PROTOCOL.replace(LEGACY_DECLARATION,
                                               '@available(*, deprecated)\n    ' + LEGACY_DECLARATION))
        for path in (LEGACY_PATH, 'Sources/Other.swift'):
            for source in variants:
                with self.subTest(path=path, source=source):
                    self.assertTrue(problems({path: source}))
        self.assertEqual(problems({}), [])
        self.assertTrue(problems({LEGACY_PATH: 'typealias Buffer<T> = UnsafeMutableBufferPointer<T>\n'
                                 'protocol SIMDStorage { associatedtype Element; '
                                 'mutating func withUnsafeMutableElements<R>(_ body: '
                                 '(inout Buffer<Element>) throws -> R) rethrows -> R }'}))

    def test_by_value_distinct_name_requirement_is_allowed_and_inout_mutation_rejected(self):
        good = LEGACY_PROTOCOL.replace('withUnsafeMutableBufferPointer<R>', 'withUnsafeMutableElements<R>')
        good = good.replace('(inout UnsafeMutableBufferPointer<Element>)', '(UnsafeMutableBufferPointer<Element>)')
        self.assertEqual(problems({LEGACY_PATH: good}), [])
        bad = good.replace('(UnsafeMutableBufferPointer<Element>)', '(inout UnsafeMutableBufferPointer<Element>)')
        self.assertNotEqual(bad, good)
        self.assertTrue(problems({LEGACY_PATH: bad}))

    def test_detector_mutation_controls(self):
        from unittest import mock
        direct = {'Sources/X.swift': 'func f(_ p: inout UnsafeMutableRawPointer) {}'}
        aliased = {'Sources/A.swift': 'typealias P<T> = UnsafeMutableBufferPointer<T>',
                   'Sources/B.swift': 'func f(_ p: inout P<Float>) {}'}
        self.assertTrue(scan(direct))
        self.assertTrue(scan(aliased))
        with mock.patch.dict(globals(), POINTERS=set()):
            self.assertEqual(scan(direct), [])
        with mock.patch.dict(globals(), alias_table=lambda _: {}):
            self.assertEqual(scan(aliased), [])
        self.assertTrue(problems({LEGACY_PATH: LEGACY_DECLARATION}))

    def test_review_unsupported_relevant_alias_syntax_fails_closed(self):
        for source in ('enum E: Error { case e }; typealias F = () throws(E) -> UnsafeMutableRawPointer; '
                       'func f(_ x: inout F) {}',
                       'protocol P { associatedtype Buffer: Equatable = UnsafeMutableRawPointer }; '
                       'func f<T: P>(_ x: inout T.Buffer) {}',
                       'protocol P { associatedtype F }; func f<T:P>(_ x:inout T.F) '
                       'where T.F == @Sendable () -> UnsafeMutableRawPointer {}',
                       'protocol P { associatedtype F }; func f<T:P>(_ x:inout T.F) '
                       'where T.F == Container<Int>.Pointer {}'):
            with self.subTest(source=source):
                self.assertTrue(scan({'Sources/X.swift': source}))

    def test_real_production_tree_has_zero_pointer_descriptor_inout_findings(self):
        files = {str(p.relative_to(ROOT)): p.read_text() for p in (ROOT / 'Sources').rglob('*.swift')}
        self.assertGreater(len(files), 100)
        self.assertIn(LEGACY_PATH, files)
        self.assertEqual(problems(files), [])
        self.assertEqual(scan(files), [])


if __name__ == '__main__':
    unittest.main()
