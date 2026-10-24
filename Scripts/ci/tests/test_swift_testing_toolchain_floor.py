"""Policy: Swift tests must not use the VALUE of `#expect(throws:)` or `#require(throws:)`.

WHY: those macros return the thrown error only from Swift 6.1's swift-testing on; on Swift 6.0 they return `()`.
CI compiles the tests with Xcode 16.2 (Swift 6.0) as well as 26.3, while the local toolchain is Swift 6.4, so
`let error = #expect(throws: VectorError.self) { ... }` followed by `error?.kind` builds and passes locally and
fails both Xcode 16.2 jobs with "cannot use optional chaining on non-optional value of type '()'" (PR #56,
d0eb404, StorageAlignmentTests.swift:29-32). Local gates cannot catch this class; this test does, lexically.

RULE: in Sources/, Tests/ and Benchmarks/ *.swift, outside comments and string-literal text, a finding is
  * a binding or assignment whose right-hand side is `(try)? #expect(throws:` or `(try)? #require(throws:`
    (`let x =`, `var x: T? =`, `x =`, `if let x =`, `guard let x =`), except the discard `_ =`;
  * `return (try)? #expect(throws:` / `#require(throws:`.
The statement form `#expect(throws: E.self) { ... }` is the Swift 6.0 form and is allowed; to inspect the error,
use do/catch with `catch let error as VectorError` and `Issue.record`, as the tree already does.

LIMITS: lexical. A member access on the macro's result after its closing brace (`#expect(throws: ...) { }.kind`)
and a use as a call argument are not detected. `#if compiler(>=6.1)` is not evaluated: the floor is the oldest
toolchain CI compiles with, so no exemption exists here.

CONTROLS: must-FLAG inline fixtures (the d0eb404 shape, a typed `var` binding, a reassignment, a `return`, a
`try #require` binding); must-PASS fixtures (statement form, `_ =` discard, do/catch, the shape inside a comment
and inside a string); the live tree has zero findings and the checker SEES at least 50 statement-form expectations,
so the scan is not vacuous. No git history is used, so the test passes out of repo.
"""
import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location('swift_lexing', ROOT / 'Scripts/ci/swift_lexing.py')
lexing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lexing)

MACRO = r'(?:try\s+)?#(?:expect|require)\(\s*throws\s*:'
VALUE_USE = re.compile(r'(?:(?P<target>[A-Za-z_]\w*)\s*(?::\s*[^=\n]+?)?\s*=\s*|\breturn\s+)' + MACRO)
ANY = re.compile(r'#(?:expect|require)\(\s*throws\s*:')


def findings(source):
    """([(line, text)] per value use of a throws expectation, count of every throws expectation seen)."""
    code = lexing._mask(source)
    lines = source.splitlines()
    found = []
    for match in VALUE_USE.finditer(code):
        if match.group('target') == '_':
            continue
        line = code.count('\n', 0, match.start()) + 1
        found.append((line, lines[line - 1].strip()))
    return found, len(ANY.findall(code))


def tree_files():
    return [p for folder in ('Sources', 'Tests', 'Benchmarks') for p in sorted((ROOT / folder).rglob('*.swift'))]


MUST_FLAG = {
    'd0eb404 shape': ('let error = #expect(throws: VectorError.self, "count \\(count)") {\n'
                      '    _ = try AlignedMemory.allocateAligned(type: Float.self, count: count)\n}\n'
                      '#expect(error?.kind == .invalidDimension, "count \\(count): \\(String(describing: error))")\n'),
    'typed var': 'var thrown: VectorError? = #expect(throws: VectorError.self) { try f() }\n',
    'reassignment': 'thrown = #expect(throws: VectorError.self) { try f() }\n',
    'return': 'func g() -> VectorError? {\n    return #expect(throws: VectorError.self) { try f() }\n}\n',
    'require binding': 'let e = try #require(throws: VectorError.self) { try f() }\n',
    'if let': 'if let e = #expect(throws: VectorError.self) { try f() } { use(e) }\n',
}
MUST_PASS = {
    'statement form': '#expect(throws: VectorError.self) { _ = try f() }\n',
    'discard': '_ = #expect(throws: VectorError.self) { try f() }\n',
    'do/catch': ('do {\n    _ = try f()\n    Issue.record("no error")\n} catch let error as VectorError {\n'
                 '    #expect(error.kind == .invalidDimension)\n}\n'),
    'comment': '// let error = #expect(throws: VectorError.self) { try f() }\n#expect(throws: Never.self) { try f() }\n',
    'string': 'let text = "let error = #expect(throws: VectorError.self) { try f() }"\n',
}


class Fixtures(unittest.TestCase):
    def test_value_uses_are_flagged_once_each(self):
        for name, source in MUST_FLAG.items():
            found, _ = findings(source)
            self.assertEqual(len(found), 1, (name, found))

    def test_statement_form_discard_do_catch_comment_and_string_pass(self):
        for name, source in MUST_PASS.items():
            found, seen = findings(source)
            self.assertEqual(found, [], (name, found))
            self.assertEqual(seen, 0 if name in ('do/catch', 'string') else 1, (name, seen))


class RealTree(unittest.TestCase):
    def test_tree_has_no_value_uses_and_the_scan_is_not_vacuous(self):
        found, seen = [], 0
        for path in tree_files():
            file_found, file_seen = findings(path.read_text(encoding='utf-8'))
            found += [(str(path.relative_to(ROOT)), line, text) for line, text in file_found]
            seen += file_seen
        self.assertEqual(found, [], found)
        self.assertGreaterEqual(seen, 50, seen)


if __name__ == '__main__':
    unittest.main()
