"""Policy (T14 phase 3, class closure): every public range helper is in the position-independence test.

WHY: a range helper scores candidates in pairs from `range.lowerBound` and an odd last candidate with a single-candidate
kernel. The eight cosine helpers of `BatchKernels` used other accumulators in their pair loops than in their tail
kernels, so a search score depended on the number of candidates and on the chunking. T14 phase 3 gave the pair loops
their tail kernels' arithmetic, and Tests/ComprehensiveTests/BatchKernelsRangeClosureTests.swift
(`pairSlotsEqualTheOddTail`) checks every listed helper: pair slot 0, pair slot 1 and the odd tail, also in a range that
starts above 0, give the single-candidate bits. This guard keeps that list complete; it does not evaluate any helper.

DISCOVERY, on Sources/**/*.swift with comments and strings blanked by Scripts/ci/swift_lexing._mask: a declaration
`public static func range_<name>(` (other attributes and modifiers may precede it) whose nearest enclosing type or
extension header is not declared `internal`, `fileprivate` or `private` (a public member of an internal type is not
public API). The helper is `<TypeName>.range_<name>`, with the type name of that header (an extension's extended type).

RULES:
  R1 the set of discovered helpers equals the set of string literals `"Type.range_name"` between the
     `// BEGIN PUBLIC RANGE HELPERS` and `// END PUBLIC RANGE HELPERS` markers of the test file; a discovered helper that
     is not listed (unregistered) and a listed name that is not discovered (stale) both fail;
  R2 the list has no duplicates.
The Swift test `everyListedHelperHasAnEvaluator` ties every listed name to an evaluator. The real-tree check and the
fixture checks read only the filesystem, so they run in a git-less export.

LIMITS (lexical, not a parser): a helper whose name does not start with `range_`, one declared without `public static
func` on one line, or one reached only through a typealias or a protocol requirement is not discovered. An extension's
visibility is read from its own header only, so a public member in an extension of an internal type declared elsewhere
is treated as public (and must be listed). A type header whose last line before `{` does not hold the type keyword
(for example a conformance list continued on its own line) is not seen as a type scope, so its helpers are not
discovered. `#if` branches are not evaluated.
"""
import importlib.util
import re
import sys
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location('swift_lexing', ROOT / 'Scripts/ci/swift_lexing.py')
lexing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lexing)

TEST_FILE = 'Tests/ComprehensiveTests/BatchKernelsRangeClosureTests.swift'
BEGIN, END = '// BEGIN PUBLIC RANGE HELPERS', '// END PUBLIC RANGE HELPERS'
_DECL = re.compile(r'\bpublic\s+static\s+func\s+(range_\w+)\s*[(<]')
_TYPE = re.compile(r'\b(class|struct|enum|extension|actor)\s+([A-Za-z_][\w.]*)')
_HIDDEN = re.compile(r'\b(?:internal|fileprivate|private)\b')
_LISTED = re.compile(r'"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\.range_\w+)"')


def discover(source):
    """`Type.range_name` for each public range helper declared in one file."""
    mask = lexing._mask(source)
    scopes = lexing._Scopes(source)
    found = []
    for match in _DECL.finditer(mask):
        scope = scopes.at[match.start()]
        while scope and not scopes.transparent[scope]:
            scope = scopes.parent[scope]
        if not scope:
            continue  # a free function is not a range helper of a type
        open_index = scopes.opens[scope]
        previous = max(mask.rfind(ch, 0, open_index) for ch in '{};')
        header = lexing._header_chunk(mask[previous + 1:open_index])
        kind = _TYPE.search(header)
        if not kind or _HIDDEN.search(header[:kind.start()]):
            continue
        found.append(f'{kind.group(2)}.{match.group(1)}')
    return found


def discover_tree(files):
    helpers = []
    for _, source in sorted(files.items()):
        helpers += discover(source)
    return helpers


def listed(text):
    start, end = text.index(BEGIN), text.index(END)
    return _LISTED.findall(text[start:end])


def check(files, test_text):
    found, names = set(discover_tree(files)), listed(test_text)
    problems = [f'R1 unregistered public range helper: {name}' for name in sorted(found - set(names))]
    problems += [f'R1 stale list entry: {name}' for name in sorted(set(names) - found)]
    problems += [f'R2 duplicate list entry: {name}' for name in sorted({n for n in names if names.count(n) > 1})]
    return problems


def real_files():
    return {path.relative_to(ROOT).as_posix(): path.read_text(encoding='utf-8')
            for path in sorted((ROOT / 'Sources').rglob('*.swift'))}


def _list(*names):
    entries = ''.join(f'        "{name}",\n' for name in names)
    return f'struct T {{\n    static let helpers = [\n        {BEGIN}\n{entries}        {END}\n    ]\n}}\n'


PUBLIC_ENUM = ('public enum K {\n'
               '    public static func range_a_512(query: Q, candidates: [Q], range: Range<Int>, out: O) {\n'
               '        var i = range.lowerBound\n    }\n}\n')


class FixtureControls(unittest.TestCase):
    """Each must-flag fixture is reported; hidden, commented and string occurrences are not."""

    def test_unlisted_public_helper_is_flagged(self):
        self.assertEqual(check({'a.swift': PUBLIC_ENUM}, _list()), ['R1 unregistered public range helper: K.range_a_512'])

    def test_listed_public_helper_passes(self):
        self.assertEqual(check({'a.swift': PUBLIC_ENUM}, _list('K.range_a_512')), [])

    def test_stale_and_duplicate_entries_are_flagged(self):
        problems = check({'a.swift': PUBLIC_ENUM}, _list('K.range_a_512', 'K.range_a_512', 'K.range_gone'))
        self.assertEqual(problems, ['R1 stale list entry: K.range_gone', 'R2 duplicate list entry: K.range_a_512'])

    def test_member_of_an_internal_type_is_not_public(self):
        for keyword in ('internal enum', '@usableFromInline internal enum', 'fileprivate struct', 'private extension'):
            with self.subTest(keyword):
                source = PUBLIC_ENUM.replace('public enum', keyword)
                self.assertEqual(discover(source), [])

    def test_extension_attributes_generics_and_multiline_headers(self):
        source = ('extension K {\n    @inlinable\n    public static func range_b<V>(query: V) {}\n}\n'
                  'public struct S<T>: Sendable\n{\n    @inline(__always) public static func range_c(x: Int) {}\n}\n')
        self.assertEqual(discover(source), ['K.range_b', 'S.range_c'])

    def test_comments_strings_free_functions_and_non_public_members_are_ignored(self):
        source = ('public enum K {\n    // public static func range_x(\n    let s = "public static func range_y("\n'
                  '    static func range_z(a: Int) {}\n    internal static func range_w(a: Int) {}\n}\n'
                  'public static func range_free(a: Int) {}\n')
        self.assertEqual(discover(source), [])

    def test_discovery_bites_by_mutation(self):
        # Each rule input is replaced in turn: the visibility rule `_HIDDEN` (without it the internal type's member is
        # reported) and the declaration rule `_DECL` (without it a public helper is not discovered).
        hidden = PUBLIC_ENUM.replace('public enum', 'internal enum')
        self.assertEqual(discover(hidden), [])
        self.assertEqual(discover(PUBLIC_ENUM), ['K.range_a_512'])
        with mock.patch.object(sys.modules[__name__], '_HIDDEN', re.compile(r'(?!)')):
            self.assertEqual(discover(hidden), ['K.range_a_512'])
        with mock.patch.object(sys.modules[__name__], '_DECL', re.compile(r'(?!)')):
            self.assertEqual(discover(PUBLIC_ENUM), [])
        self.assertEqual(discover(hidden), [])  # restored after the mutations


class RealTree(unittest.TestCase):

    def test_real_tree_list_is_complete_and_current(self):
        text = (ROOT / TEST_FILE).read_text(encoding='utf-8')
        self.assertEqual(check(real_files(), text), [])

    def test_real_tree_visibility_floor(self):
        found = set(discover_tree(real_files()))
        # Public types' helpers are seen; the internal TopKSelectionKernels' public members are not public API.
        self.assertIn('BatchKernels.range_cosine_fused_512', found)
        self.assertIn('MixedPrecisionKernels.range_cosine_mixed_512', found)
        self.assertFalse(any(name.startswith('TopKSelectionKernels.') for name in found), sorted(found))
        self.assertEqual(len(found), 21)


if __name__ == '__main__':
    unittest.main()
