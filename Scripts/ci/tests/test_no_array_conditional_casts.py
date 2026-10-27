"""Policy (P02 phase E, root rulings RD4-RD6; phase D-F review B-I1, m1, m2, n1, n2; final review FA-3, FA-4, C-M2,
FN-2): no conditional array-target cast in Sources/VectorCore.

WHY: in unspecialized generic code a conditional cast of an array to an array type (`values as? [T]`, `values is [T]`,
`values as! [T]`, or the pattern `case let c as [T]`) converts the array element by element. P02 phase 1 probe B (one
host) measured one request of 16 * count + 32 bytes per `[Vector512Optimized]` cast, on success AND on failure; that
figure was taken with the former 16-byte vector handle (the handle stride is the storage reference's since P02 phase
F, `P02HandleLayoutTests.handleStrideIsTheStorageStride`). P02 replaced every such expression in Sources/VectorCore (39
after phase C deleted six with the implicit GEMM route; 45 in the phase 1 inventory) with the type-identity helper
`exactArray(_:as:)` (Sources/VectorCore/Operations/ArrayIdentity.swift), which shares the caller's buffer
(`P02ArrayIdentityTests`). This guard fails on any new cast in the FORMS below; LIMITS lists the spellings it does not
see, so it closes the class only for the listed forms.

FORMS, in every Sources/VectorCore/**/*.swift file, on the shared lexer's text (Scripts/ci/swift_lexing): comments
and literal string text blanked, the code inside string interpolations (`\\(...)`, and `\\#(...)` in a raw string)
KEPT, raw strings ending only at their own delimiter (so `#"a"b"#` hides nothing after it), and backtick escapes
turned into spaces (`_unbacktick`), so `` `Array`<T> `` and `` Swift.`Array`<T> `` read as `Array<T>` (`lex` below:
`_mask(text, keep_interpolations=True)`). Until the P02 final fix this guard carried its own interpolation-keeping
masker; the shared lexer now provides that mode:
  * the operators `as?`, `as!` and `is`, each starting a word (so `areas?[0]` and `deltas![0]` are not casts);
  * the pattern cast `as` (with no `?` or `!`) when a `case` or `catch` keyword opens the same pattern: `case let c as
    [T]:`, `if case let c as [T] = ...`, `guard case`, `for case let`, `case let .some(c as [T])`, `catch let e as
    [T]`. The pattern is searched backwards from `as` to the nearest `case`/`catch`, stopping at `{`, `}`, `;`, at
    `=` or `:` outside brackets, and at `where` or `in` outside brackets;
each followed (optionally after whitespace, including a newline, and any number of opening parentheses, as in
`as? ([T])`) by `[`, `Array<`, `ContiguousArray<` or their `Swift.`-qualified forms. Every branch of `#if` is scanned
(no preprocessing). The ALLOWLIST is EMPTY; an entry would map `path:line` to a reviewed reason, and a stale entry
would fail.

Fixtures below: MUST_FAIL (each form, a multi-line cast, a cast inside `#if`, `Array<`/`ContiguousArray<`, the
`Swift.` spellings, a dictionary type, the pattern casts, parenthesized targets, casts inside interpolations) and
MUST_PASS (comments, strings, `exactArray(`, an element cast `query as? Vector512Optimized`, `is` inside an identifier,
`areas?[0]`, `deltas![0]`, coercions `[] as [Float]` outside a pattern, element pattern casts). Out-of-scope paths are
a tree fixture (`test_out_of_scope_paths_pass`). Controls: a mutation control (disabling one form, one target, the
parenthesis prefix, the interpolation scan or the backtick normalization lets exactly the fixtures that need it
through), a tree-copy control that copies Sources/VectorCore to a temporary directory, re-adds one live site and must
find exactly that site, and LIMIT fixtures that assert each limit below is still unseen (so this text cannot silently
go stale). The lexer's own forms are pinned by Scripts/ci/tests/test_swift_lexing.py. Filesystem reads only (works
git-less).

LIMITS (lexical, not a type checker; each is a LIMIT fixture):
  * a target spelled through a typealias (`typealias Rows = [Vector512Optimized]; values as? Rows`) or a generic
    parameter (`values as? T` with `T == [V]`);
  * `ArraySlice<`, `Set<`, `Dictionary<` and `Optional<[T]>` targets;
  * an unconditional coercion `x as [T]` outside a `case`/`catch` pattern (a static upcast or a literal's type, not a
    conditional cast), and a pattern cast separated from its `case`/`catch` by one of the stop tokens above;
  * the lexer does not lex regex literals (`/…/`): their text is read as code, and a quote inside one opens a string
    that hides the rest of its line (`let r = /"/; let c = values as? [T]`);
  * Sources/VectorCoreBenchmarking (its JSON dictionary casts), Benchmarks/ and Tests/ are out of scope by path.
`exactArray` is a type-identity test, not an element-wise conversion. Unlike `as?` and `is` it returns nil for `[T?]`,
for existential or `Any` element arrays, for class upcasts, and for an EMPTY array of another element type (`as?` and
`is` succeed there vacuously; `P02ArrayIdentityTests.nilForAnotherElementType` pins nil, and
`nilForOptionalElements` pins the Optional case). At the converted sites none of those inputs reaches a test with a
different outcome: the source is `[V]` for a generic `V: VectorProtocol` that is a concrete struct (P02 phase A+B
review F9), and each kind of site has its own guard (P02 final review C-M2): the pairwise identity chain runs only after
`pairwiseDistances`' `guard n > 0`, which excludes the empty array; the `.gram` admission (`Operations.gramAdmission`)
runs only after `findNearestBatch`'s nonempty checks on both arrays; `SoAFP16.init(vectors:)` converts only behind
`VectorType.self == T.self`, an identity; every other site tests `query as? T` on the same `V` first. A site with another
kind of source or guard needs these differences re-derived.
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


SCOPE = 'Sources/VectorCore'
FORMS = {
    'as?': r'(?<![\w.])as\?',
    'as!': r'(?<![\w.])as!',
    'is': r'(?<![\w.])is',
    'pattern': r'(?<![\w.])as(?![?!\w])',   # only where `in_pattern` holds
}
TARGETS = {
    'bracket': r'\[',
    'Array<': r'(?:Swift\s*\.\s*)?Array\s*<',
    'ContiguousArray<': r'(?:Swift\s*\.\s*)?ContiguousArray\s*<',
}
PARENS = r'(?:\(\s*)*'
ALLOWLIST = {}   # empty after P02 phase E (RD6): `path:line` -> reviewed reason


def pattern(forms=FORMS, targets=TARGETS, parens=True):
    """One compiled alternative per form: {form: regex of the form followed by a target}."""
    tail = r'\s*' + (PARENS if parens else '') + r'(?:' + '|'.join(targets.values()) + r')'
    return {form: re.compile(regex + tail) for form, regex in forms.items()}


CAST = pattern()


def _depths(masked):
    """Bracket depth ((), []) at each offset of `masked`."""
    depths, depth = [], 0
    for ch in masked:
        depths.append(depth)
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth = max(0, depth - 1)
    return depths


def in_pattern(masked, depths, at):
    """True when a `case` or `catch` keyword opens the pattern that the `as` at offset `at` is part of."""
    i = at - 1
    while i >= 0:
        ch = masked[i]
        if ch in '{};':
            return False
        if ch in '=:' and depths[i] == 0:
            return False
        if ch.isalnum() or ch == '_':
            j = i
            while j >= 0 and (masked[j].isalnum() or masked[j] == '_'):
                j -= 1
            word = masked[j + 1:i + 1]
            if word in ('case', 'catch') and (j < 0 or masked[j] != '.'):
                return True
            if word in ('where', 'in') and depths[j + 1] == 0:
                return False
            i = j
            continue
        i -= 1
    return False


def lex(text):
    """The text the guard reads (see FORMS): the shared lexer's mask, with backtick escapes turned into spaces."""
    return lexing._unbacktick(lexing._mask(text, keep_interpolations=True))


def find(text, cast=None, mask=lex):
    """Sorted line numbers of array-target casts in Swift source `text`."""
    cast = CAST if cast is None else cast
    masked = mask(text)
    depths = _depths(masked)
    starts = set()
    for form, regex in cast.items():
        for match in regex.finditer(masked):
            if form != 'pattern' or in_pattern(masked, depths, match.start()):
                starts.add(match.start())
    return sorted({masked.count('\n', 0, start) + 1 for start in starts})


def scan(root, cast=None):
    """['path:line', ...] for every array-target cast in root/Sources/VectorCore/**/*.swift."""
    found = []
    for path in sorted((root / SCOPE).rglob('*.swift')):
        rel = path.relative_to(root).as_posix()
        found += [f'{rel}:{line}' for line in find(path.read_text(encoding='utf-8'), cast)]
    return found


def problems(found, allowlist=ALLOWLIST):
    out = [f'{site}: conditional array-target cast (use exactArray(_:as:))' for site in found if site not in allowlist]
    out += [f'{site}: stale allowlist entry' for site in allowlist if site not in found]
    return out


# name: (text, form, target, feature). `feature` is the guard feature a fixture needs beyond its form and target:
# 'paren' (the parenthesis prefix) or 'interpolation' (the interpolation scan); None otherwise.
MUST_FAIL = {
    'as? bracket': ('if let c = vectors as? [Vector512Optimized] { }\n', 'as?', 'bracket', None),
    'as! bracket': ('let c = vectors as! [Vector768Optimized]\n', 'as!', 'bracket', None),
    'is bracket': ('guard vectors is [Vector1536Optimized] else { return nil }\n', 'is', 'bracket', None),
    'multi-line': ('let c = vectors\n    as?\n    [Vector512Optimized]\n', 'as?', 'bracket', None),
    'inside #if': ('#if arch(arm64)\nlet c = vectors as? [Vector512Optimized]\n#else\nlet c: [Int]? = nil\n#endif\n',
                   'as?', 'bracket', None),
    'Array<': ('let c = values as? Array<Vector512Optimized>\n', 'as?', 'Array<', None),
    'ContiguousArray<': ('let c = storage as! ContiguousArray<SIMD4<Float>>\n', 'as!', 'ContiguousArray<', None),
    'Swift.Array< is': ('if values is Swift.Array<Float> { }\n', 'is', 'Array<', None),
    'Swift.Array< as?': ('let c = values as? Swift.Array<Vector512Optimized>\n', 'as?', 'Array<', None),
    'dictionary': ('let d = object as? [String: Any]\n', 'as?', 'bracket', None),
    'no space': ('let ok = values is[Float]\n', 'is', 'bracket', None),
    'case is': ('switch values {\ncase is [Float]:\n    return 1\ndefault:\n    return 0\n}\n', 'is', 'bracket', None),
    # Phase D-F review B-I1: pattern casts.
    'switch case pattern': ('switch values {\ncase let c as [Vector512Optimized]:\n    return c.count\ndefault:\n'
                            '    return 0\n}\n', 'pattern', 'bracket', None),
    'if case pattern': ('if case let c as [Vector512Optimized] = values { use(c) }\n', 'pattern', 'bracket', None),
    'guard case pattern': ('guard case let c as Array<Float> = values else { return }\n', 'pattern', 'Array<', None),
    'for case pattern': ('for case let row as [Float] in rows { use(row) }\n', 'pattern', 'bracket', None),
    'payload pattern': ('if case let .some(c as [Vector768Optimized]) = maybe { use(c) }\n', 'pattern', 'bracket',
                        None),
    'labelled payload pattern': ('switch pair {\ncase let .pair(first: c as Swift.Array<Float>, second: _):\n'
                                 '    use(c)\ndefault:\n    break\n}\n', 'pattern', 'Array<', None),
    'multi-line pattern': ('switch values {\ncase let c\n    as ContiguousArray<Float>:\n    use(c)\ndefault:\n'
                           '    break\n}\n', 'pattern', 'ContiguousArray<', None),
    'catch pattern': ('do { try run() } catch let e as [Float] { use(e) }\n', 'pattern', 'bracket', None),
    # Phase D-F review B-I1: parenthesized targets.
    'parenthesized as?': ('let c = values as? ([Vector512Optimized])\n', 'as?', 'bracket', 'paren'),
    'parenthesized is': ('if values is ( (Swift.Array<Float>) ) { }\n', 'is', 'Array<', 'paren'),
    'parenthesized pattern': ('if case let c as ([Float]) = values { use(c) }\n', 'pattern', 'bracket', 'paren'),
    # Phase D-F review B-I1: casts inside string interpolations.
    'interpolation': ('let s = "\\(String(describing: values as? [Vector512Optimized]))"\n', 'as?', 'bracket',
                      'interpolation'),
    'multi-line string interpolation': ('let s = """\n    count \\(values is [Float] ? 1 : 0)\n    """\n', 'is',
                                        'bracket', 'interpolation'),
    'nested interpolation': ('let s = "\\(label ?? "none") \\((values as! ContiguousArray<Float>).count)"\n', 'as!',
                             'ContiguousArray<', 'interpolation'),
    # Final review FN-2: a raw string with an odd quote count hid the rest of its line; and a raw-string interpolation.
    'after a raw string with an odd quote count': ('let s = #"a"b"#; let c = values as? [Vector512Optimized]\n', 'as?',
                                                   'bracket', None),
    'raw string interpolation': ('let s = #"n \\#(values as? [Float])"#\n', 'as?', 'bracket', 'interpolation'),
    # Final review FA-3: backtick-escaped target names (each compiles and is a live conditional array cast).
    'escaped Array< as?': ('let ok = (values as? `Array`<Vector512Optimized>) != nil\n', 'as?', 'Array<', 'backtick'),
    'escaped Swift.Array< is': ('if values is Swift.`Array`<Vector512Optimized> { }\n', 'is', 'Array<', 'backtick'),
    'escaped pattern Array<': ('if case let c as `Array`<Vector512Optimized> = values { use(c) }\n', 'pattern', 'Array<',
                               'backtick'),
    'escaped ContiguousArray<': ('let c = storage as! `Swift`.`ContiguousArray`<Float>\n', 'as!', 'ContiguousArray<',
                                 'backtick'),
}
MUST_PASS = {
    'line comment': '// if let c = vectors as? [Vector512Optimized] { }\n',
    'doc comment': '/// `values as? [Target]` converts element by element; `is [T]` and `as! [T]` too.\n',
    'block comment': '/* vectors is [Vector512Optimized] */ let x = 1\n',
    'string': 'let message = "vectors as? [Vector512Optimized] copies"\n',
    'multi-line string': 'let s = """\nvalues as! [Float]\n"""\n',
    'string beside an interpolation': 'let s = "\\(values.count) values; values as? [T] is text"\n',
    'comment inside an interpolation': 'let s = "\\(values.count /* as? [Float] */)"\n',
    'exactArray': 'if let c = exactArray(vectors, as: Vector512Optimized.self) { }\n',
    'exactArray is-test': 'guard exactArray(vectors, as: Vector512Optimized.self) != nil else { return nil }\n',
    'exactArray in an interpolation': 'let s = "\\(exactArray(values, as: Float.self) != nil)"\n',
    'element cast': 'if let q = query as? Vector512Optimized, metric is EuclideanDistance { }\n',
    'identifier': 'let thisIs = [1]; let isX = thisIs[0]; let axis = [Float](repeating: 0, count: 2)\n',
    'subscript after as': 'let y = (x as? Foo)?[0]\n',
    # Phase D-F review B-m1: names ending in "as" before an optional or forced subscript.
    'optional-chained subscript': 'let first = areas?[0]\n',
    'force-unwrapped subscript': 'let last = deltas![0]\n',
    'coercion': 'let empty = [] as [Float]\n',
    'coercion in a case body': 'switch kind {\ncase .empty: return [] as [Float]\ndefault: return values\n}\n',
    'coercion after if-case': 'if case .some = maybe, let rows = [] as [Float] { use(rows) }\n',
    'coercion after where': 'switch v {\ncase let x where ([] as [Float]).isEmpty: use(x)\ndefault: break\n}\n',
    'element pattern cast': 'switch value {\ncase let v as Vector512Optimized: use(v)\ndefault: break\n}\n',
    'catch element cast': 'do { try run() } catch let error as VectorError { log(error) }\n',
    'member named case before a coercion': 'use(Kind.case, [] as [Float])\n',
}
# Each LIMITS item: asserted NOT found, so the docstring's LIMITS stays accurate.
LIMIT = {
    'typealias': 'typealias Rows = [Vector512Optimized]\nlet c = values as? Rows\n',
    'generic parameter': 'func f<T>(_ v: Any, _: T.Type) -> T? { v as? T }\n',
    'ArraySlice<': 'let c = values as? ArraySlice<Float>\n',
    'Set<': 'let c = values as? Set<Float>\n',
    'Dictionary<': 'let c = values as? Dictionary<String, Float>\n',
    'Optional<[T]>': 'let c = values as? Optional<[Float]>\n',
    'coercion outside a pattern': 'let c = values as [Any]\n',
    'pattern behind a stop token': 'if case .some = maybe, let c = values as [Any] { use(c) }\n',
    'a quote inside a regex literal': 'let r = /"/; let c = values as? [Float]\n',
}


class Fixtures(unittest.TestCase):
    def test_must_fail(self):
        for name, (text, _, _, _) in MUST_FAIL.items():
            with self.subTest(name=name):
                self.assertTrue(find(text), name)

    def test_must_pass(self):
        for name, text in MUST_PASS.items():
            with self.subTest(name=name):
                self.assertEqual(find(text), [], name)

    def test_limits_are_still_unseen(self):
        for name, text in LIMIT.items():
            with self.subTest(name=name):
                self.assertEqual(find(text), [], f'{name}: now caught; update LIMITS')

    def test_mutation_each_alternative_bites(self):
        # Disabling one form, one target, the parenthesis prefix, the interpolation scan (the lexer's former mode) or
        # the backtick normalization lets exactly the fixtures that need it through.
        def missed(cast=None, mask=lex):
            return {name for name, (text, _, _, _) in MUST_FAIL.items() if not find(text, cast, mask)}

        for form in FORMS:
            weakened = pattern({k: v for k, v in FORMS.items() if k != form}, TARGETS)
            self.assertEqual(missed(weakened), {n for n, (_, f, _, _) in MUST_FAIL.items() if f == form}, form)
        for target in TARGETS:
            weakened = pattern(FORMS, {k: v for k, v in TARGETS.items() if k != target})
            self.assertEqual(missed(weakened), {n for n, (_, _, t, _) in MUST_FAIL.items() if t == target}, target)
        self.assertEqual(missed(pattern(parens=False)),
                         {n for n, (_, _, _, feature) in MUST_FAIL.items() if feature == 'paren'})
        former = lambda text: lexing._unbacktick(lexing._mask(text, keep_interpolations=False))  # noqa: E731
        self.assertEqual(missed(mask=former),
                         {n for n, (_, _, _, feature) in MUST_FAIL.items() if feature == 'interpolation'})
        self.assertEqual(missed(mask=lambda text: lexing._mask(text, keep_interpolations=True)),
                         {n for n, (_, _, _, feature) in MUST_FAIL.items() if feature == 'backtick'})

    def test_pattern_form_needs_its_case_or_catch(self):
        # Without the `in_pattern` condition the bare `as` alternative would flag every coercion fixture.
        bare = re.compile(FORMS['pattern'] + r'\s*' + PARENS + r'(?:' + '|'.join(TARGETS.values()) + r')')
        coercions = [MUST_PASS[name] for name in ('coercion', 'coercion in a case body', 'coercion after if-case',
                                                  'coercion after where', 'member named case before a coercion')]
        for text in coercions:
            with self.subTest(text=text):
                self.assertTrue(bare.search(lex(text)), text)
                self.assertEqual(find(text), [], text)

    def test_allowlist_semantics(self):
        self.assertEqual(problems(['a.swift:3'], {'a.swift:3': 'reviewed'}), [])
        self.assertEqual(len(problems(['a.swift:3'], {})), 1)
        self.assertEqual(problems([], {'a.swift:3': 'reviewed'}), ['a.swift:3: stale allowlist entry'])


class RealTree(unittest.TestCase):
    def test_scope_exists_and_is_scanned(self):
        files = list((ROOT / SCOPE).rglob('*.swift'))
        self.assertGreater(len(files), 50)
        # Non-vacuity: the converted sites are in the scanned text.
        text = (ROOT / SCOPE / 'Operations/BatchOperations.swift').read_text(encoding='utf-8')
        self.assertIn('exactArray(vectors, as: Vector512Optimized.self)', text)

    def test_allowlist_is_empty(self):
        self.assertEqual(ALLOWLIST, {})

    def test_no_array_conditional_casts(self):
        self.assertEqual(problems(scan(ROOT)), [])

    def test_tree_copy_with_one_live_site_is_flagged(self):
        # A git-less copy of Sources/VectorCore with ONE converted site put back to its old cast: exactly that
        # site is reported.
        with tempfile.TemporaryDirectory() as scratch:
            copy = Path(scratch)
            shutil.copytree(ROOT / SCOPE, copy / SCOPE)
            target = copy / SCOPE / 'Operations/BatchOperations.swift'
            text = target.read_text(encoding='utf-8')
            old = 'exactArray(vectors, as: Vector768Optimized.self)'
            self.assertIn(old, text)
            at = text.index(old)
            target.write_text(text[:at] + 'vectors as? [Vector768Optimized]' + text[at + len(old):], encoding='utf-8')
            line = text.count('\n', 0, at) + 1
            self.assertEqual(scan(copy), [f'{SCOPE}/Operations/BatchOperations.swift:{line}'])
            self.assertEqual(len(problems(scan(copy))), 1)

    def test_out_of_scope_paths_pass(self):
        # Phase D-F review B-n2: the same cast under Sources/VectorCoreBenchmarking, Tests/ and Benchmarks/ is not
        # scanned; under Sources/VectorCore it is (control).
        cast = 'let d = object as? [String: Any]\n'
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            for rel in ('Sources/VectorCoreBenchmarking/Report.swift', 'Tests/ComprehensiveTests/Probe.swift',
                        'Benchmarks/Probe.swift'):
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text(cast, encoding='utf-8')
            (root / SCOPE).mkdir(parents=True)
            (root / SCOPE / 'Clean.swift').write_text('let c = exactArray(values, as: Float.self)\n', encoding='utf-8')
            self.assertEqual(scan(root), [])
            (root / SCOPE / 'Live.swift').write_text(cast, encoding='utf-8')
            self.assertEqual(scan(root), [f'{SCOPE}/Live.swift:1'])


if __name__ == '__main__':
    unittest.main()
