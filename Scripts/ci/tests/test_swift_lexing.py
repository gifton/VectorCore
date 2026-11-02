r"""Policy (P02 final fix, review A FA-2, FA-3, FN-2): the shared Swift lexer's masking forms.

`Scripts/ci/swift_lexing._mask` blanks comments and string literals and keeps everything else, with the same length
and newlines. Since the P02 final fix a raw string ends only at a quote followed by as many `#` (every mode), and
`_mask(source, keep_interpolations=True)` keeps the code inside a string interpolation (`\( … )`, and `\#( … )` in a raw
string with one `#`) and blanks only literal text. The default mode still blanks interpolated code (see the module
docstring for why). `_unbacktick` turns escaped identifiers into plain names.

Each form has MUST_MASK (text that must be blanked) and MUST_KEEP (code that must survive) fixtures, checked in the
keep-interpolations mode. Controls: the default mode must blank the interpolated fixtures' code (so they need the
opt-in) and must equal the opt-in mode on every other fixture (raw strings included); on every Sources/VectorCore file
both masks have the source's length and newlines, and the opt-in mask differs from the default one only where the
default is blank, at least somewhere (non-vacuous). Filesystem reads only.
"""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location('swift_lexing', ROOT / 'Scripts/ci/swift_lexing.py')
lexing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lexing)

# name: (source, [substrings that must survive in the mask], [substrings that must not], interpolated?)
FORMS = {
    'line comment': ('let a = 1 // gramBatch(x)\n', ['let a = 1'], ['gramBatch'], False),
    'block comment': ('/* as? [T] */ let b = 2\n', ['let b = 2'], ['as?'], False),
    'string text': ('let s = "values as? [T]"; let c = 3\n', ['let c = 3'], ['values', 'as?'], False),
    'escaped quote': ('let s = "a\\"b as? [T]"; let d = 4\n', ['let d = 4'], ['as?'], False),
    'multi-line string text': ('let s = """\n  as? [T]\n  """\nlet e = 5\n', ['let e = 5'], ['as?'], False),
    'unterminated string ends at its line': ('let s = "open\nlet f = 6\n', ['let f = 6'], ['open'], False),
    'interpolation': ('let s = "a \\(f(x)) b"\n', ['f(x)'], ['a ', ' b'], True),
    'interpolation with a nested string': ('let s = "\\(label ?? "none")"\n', ['label ??'], ['none'], True),
    'interpolation with a comment': ('let s = "\\(v /* as? [T] */ .count)"\n', ['v', '.count'], ['as?'], True),
    'nested interpolation': ('let s = "x \\("in \\(deep(1)) y") z"\n', ['deep(1)'], ['x ', ' y', ' z', 'in '], True),
    'interpolation in a multi-line string': ('let s = """\n  \\(g as? [T]) d\n  """\n', ['g as? [T]'], [' d'], True),
    'raw string text': ('let s = #"a as? [T] b"#; let g = 7\n', ['let g = 7'], ['as?'], False),
    'raw string with an odd quote count': ('let s = #"a"b"#; let c = values as? [T]\n',
                                           ['let c = values as? [T]'], ['a"b'], False),
    'two-hash raw string': ('let s = ##"a "# b"##; let h = 8\n', ['let h = 8'], ['"#'], False),
    'raw string: a plain backslash paren is text': ('let s = #"\\(notCode)"#\n', [], ['notCode'], False),
    'raw string interpolation': ('let s = #"x \\#(values as? [Float]) y"#\n', ['values as? [Float]'], ['x ', ' y'], True),
    'raw string escaped quote is text': ('let s = #"a\\"b"#; let i = 9\n', ['let i = 9'], ['a\\"b'], False),
}


class Forms(unittest.TestCase):
    def test_must_keep_and_must_mask(self):
        for name, (source, keep, drop, _) in FORMS.items():
            with self.subTest(name=name):
                masked = lexing._mask(source, keep_interpolations=True)
                self.assertEqual(len(masked), len(source))
                self.assertEqual([i for i, c in enumerate(masked) if c == '\n'],
                                 [i for i, c in enumerate(source) if c == '\n'])
                for text in keep:
                    self.assertIn(text, masked, (name, masked))
                for text in drop:
                    self.assertNotIn(text, masked, (name, masked))

    def test_default_mode_blanks_interpolated_code(self):
        # Control: the default mode loses the interpolated fixtures' code (they need the opt-in) and equals the opt-in
        # mode on every other fixture, raw strings included.
        for name, (source, keep, _, interpolated) in FORMS.items():
            with self.subTest(name=name):
                masked = lexing._mask(source)
                self.assertEqual(masked, lexing._mask(source, keep_interpolations=False), name)
                if interpolated:
                    self.assertTrue(any(text not in masked for text in keep), (name, masked))
                else:
                    self.assertEqual(masked, lexing._mask(source, keep_interpolations=True), name)

    def test_unbacktick(self):
        source = 'let c = values as? `Array`<T>; if x is Swift.`Array`<T> { `gramBatch`(q) }\n'
        plain = lexing._unbacktick(lexing._mask(source))
        self.assertEqual(len(plain), len(source))
        self.assertNotIn('`', plain)
        for text in ('as?  Array <T>', 'Swift. Array <T>', ' gramBatch (q)'):
            self.assertIn(text, plain)

    def test_statement_parens_stay_balanced_with_interpolations(self):
        source = 'let s = "\\(f(a, (b)))"\nlet t = 1\n'
        self.assertEqual([n for n, _ in lexing._statements(source)], [1, 2])


class RealTree(unittest.TestCase):
    def test_masks_keep_length_and_newlines_and_differ_only_in_interpolations(self):
        differing = 0
        files = sorted((ROOT / 'Sources/VectorCore').rglob('*.swift'))
        self.assertGreater(len(files), 50)
        for path in files:
            text = path.read_text(encoding='utf-8')
            new, old = lexing._mask(text, keep_interpolations=True), lexing._mask(text)
            self.assertEqual(len(new), len(text), path)
            self.assertEqual(new.count('\n'), text.count('\n'), path)
            for a, b in zip(new, old):
                if a != b:
                    self.assertEqual(b, ' ', path)
                    differing += 1
        self.assertGreater(differing, 0)


if __name__ == '__main__':
    unittest.main()
