"""Policy (HEURISTIC): Swift tests must not derive write locations from the checkout or package tree.

A test that branches on `lastPathComponent == "workspaces"` writes outside the
repository (or fails) depending on where the checkout lives; a test that writes
to `<package>/.build/...` (found through `#filePath`) litters the package tree
and breaks on read-only checkouts. Tests use FileManager.default.temporaryDirectory,
or an explicit opt-in environment variable. Reading a fixture through a
`#filePath`-derived root is allowed; only write operations are checked.

This is a lexical heuristic, not a proof. Scope and known limits:

* Statements are joined across lines (balanced () and []) before matching, so a
  write split over several lines is still one statement. Line numbers are the
  statement's first line.
* Rules: checkout-name heuristics; `.build` path literals; writes that use a name
  tainted by `#filePath`/`#file`/`#fileID` or by `currentDirectoryPath`; writes with
  a relative string-literal path (they resolve against the cwd, the package root
  under `swift test`).
* Taint comes from `let`/`var` declarations, and from functions and computed
  properties whose BODY mentions a source location or a tainted name (default
  arguments such as `file: StaticString = #filePath` do not taint, because the
  signature is not part of the body).
* Scope rules. The source is segmented into brace scopes (`_scopes`): string
  literals, `//` and `/* */` comments are blanked first, then `{`/`}` are matched
  by depth, consistent with the paren-depth statement joining above. A scope whose
  header is `class|struct|enum|extension|actor|protocol` is TRANSPARENT (its
  members are file-wide); every other scope (function, initializer, closure,
  computed property, `if`/`for`/`guard` body, ...) is LOCAL.
    - A `let`/`var` taints its name only inside the nearest LOCAL scope that
      encloses the declaration, nested bodies included; at file level or directly
      in a type body it taints the name file-wide.
    - Function and computed-property NAMES tainted by their bodies taint file-wide
      (a call to such a name anywhere is a real use), including nested functions.
    - A mention is evaluated at its own position, so a tainted name is visible only
      from scopes inside the declaring scope. A same-named untainted local in a
      sibling function is a different variable and is not flagged.
* Two taint kinds. LOCAL names (function-local `let`/`var`, file-level globals) keep
  the member exemption below. MEMBER-LIKE names (declared directly in a type body,
  function names, computed-property names, and `self.x = …` targets) are reachable as
  instance members from any base, so they count as mentions after a `.` from ANY base
  (`h.out`, `H().out`, `H.shared.out`, `f().out`, `type(of: self).out`). Name-based:
  a member-like taint also fires for an unrelated type's same-named member.
* A `let`/`var` (or an assignment `x = …`) whose expression is a closure uses the whole
  closure body as its expression: `= {` ... `}()` (also inside `if let x = {…}()`), the
  Allman form (`=` ending the line, `{` on the next line), and a statement ending in a
  closure opener (`{`, `{ _ in`, `{ a, b in`, `{ (a, b) in`, `{ [weak self] x in`) unless
  introduced by if/guard/while/for/switch/case.
* Assignments `x = …`, `self.x = …` and base-qualified `h.x = …`, `T.x = …`, `Self.x = …`
  (e.g. in `setUp`) taint `x`. The qualified forms and unresolved names are member-like and
  file-wide; an unqualified `x` that resolves to a visible local declaration is in-scope.
* What counts as a mention of a tainted name: an identifier that is not (a) an
  argument label (preceded by `(` or `,` and followed by `:`, so `write(cases: x)`
  does not mention `cases`, while a ternary `c ? root : y` still mentions `root`),
  nor (b) a member name after a single `.`, unless that name is member-like tainted
  (`foo.path` mentions `foo`, not a local `path`).
  A range operator `a...n` is not a member access.
* Remaining limits (lexical heuristic, all unverified beyond the named tests):
    - Taint is per file and never crosses files.
    - A tainted OUTER name shadowed by a same-named local in a nested scope is still
      treated as tainted (conservative); function parameters are not declarations and
      do not shadow.
    - Taint does not flow through parameters (`func w(dir:)` called with a tainted
      value), `switch`/`if case let` bindings, or tuple-destructuring declarations.
    - Only the first assignment per statement is read; compound assignments (`+=`) and
      assignments that are not at the start of a statement (or after `{`/`;`) are not.
    - Brace matching can be unbalanced by raw strings (`#"..."#`) and by nested string
      literals inside interpolation.
    - Closure openers are recognised only by the CLOSURE_OPENER shapes above; a closure
      with an explicit return type on a separate line, or an opener inside an unbalanced
      multi-line argument list, is not extended. A multi-statement expression other than
      a closure is read only to the end of its statement.
    - A dictionary-literal or tuple key after `(` or `,` (`(a, root: v)`) is read as a
      label; member-like taint is by name, so an unrelated same-named member also fires.
* Write operations recognised: see WRITE_OPERATION. Anything else (for example a
  write through a third-party wrapper) is not seen.
"""

import importlib.util
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_LEXING_SPEC = importlib.util.spec_from_file_location('swift_lexing', ROOT / 'Scripts/ci/swift_lexing.py')
swift_lexing = importlib.util.module_from_spec(_LEXING_SPEC)
_LEXING_SPEC.loader.exec_module(swift_lexing)
_Scopes, _classify, _mask, _statements = (swift_lexing._Scopes, swift_lexing._classify,
                                          swift_lexing._mask, swift_lexing._statements)
IDENTIFIER = swift_lexing.IDENTIFIER
LAST_COMPONENT_COMPARISON = re.compile(r'lastPathComponent\s*[=!]=')
WORKSPACES_LITERAL = re.compile(r'"workspaces"')
BUILD_LITERAL = re.compile(r'"[^"\n]*\.build(?:/[^"\n]*)?"')
SOURCE_LOCATION = re.compile(r'#filePath|#fileID|#file\b|currentDirectoryPath|currentDirectoryURL')
WRITE_OPERATION = re.compile(
    r'createDirectory|createFile|forWritingTo|forWritingAtPath|forUpdating|forUpdatingAtPath'
    r'|OutputStream\s*\(\s*(?:toFileAtPath|url)'
    r'|\.write\(|\bwrite\(\s*to(?:File)?'
    r'|moveItem|copyItem|removeItem|replaceItem|createSymbolicLink|createSymbolicLinkAtPath'
    r'|\bfopen\s*\([^)]*,\s*\"[^\"]*[wa+][^\"]*\"|\bopen\s*\([^)]*\bO_(?:WRONLY|RDWR|CREAT)|\bcreat\s*\(')
# A write whose path argument is a literal that is not absolute: resolved against the cwd.
RELATIVE_LITERAL = re.compile(r'(?:atPath|toFile|toFileAtPath|fileURLWithPath|forWritingAtPath|forUpdatingAtPath|path)\s*:\s*"([^"/\\][^"\\]*)"'
                              r'|\bfopen\s*\(\s*"([^"/][^"]*)"|\bopen\s*\(\s*"([^"/][^"]*)"')
DECLARATION = re.compile(r'(?=\b(?:let|var)\s+(\w+)\s*(?::[^={]+)?=(.*))', re.S)  # overlapping: every `let`/`var`
DECLARATION_NAME = re.compile(r'\b(?:let|var)\s+(\w+)')
# `x = …` / `self.x = …` at the start of a statement or after `{`/`;` (not `let x =`, not `==`, `+=`).
ASSIGNMENT = re.compile(r'(?:^|[;{])\s*((?:\w[\w.]*\.)?)(\w+)\s*=(?!=)(.*)', re.S)
# A statement ending in a closure opener: `{`, `{ _ in`, `{ a, b in`, `{ (a, b) in`, `{ [weak self] x in`.
CLOSURE_OPENER = re.compile(r'\{\s*(?:(?:\[[^\]]*\]\s*)?(?:\([^)]*\)\s*(?:->\s*[^\s{]+\s*)?|[\w,\s:_]*?\s)?in)?\s*$')
CONTROL_FLOW = re.compile(r'\b(?:if|guard|while|for|switch|case)\b')


def violations(source):
    """Return (line, rule) pairs for forbidden path heuristics in Swift source."""
    found = []
    source = source.replace('\r\n', '\n').replace('\r', '\n')
    statements = _statements(source)
    for number, text in statements:
        if LAST_COMPONENT_COMPARISON.search(text):
            found.append((number, 'lastPathComponent comparison'))
        if WORKSPACES_LITERAL.search(text):
            found.append((number, '"workspaces" literal'))
        if BUILD_LITERAL.search(text):
            found.append((number, '.build path literal'))
    line_start = [0]
    for line in source.split('\n'):
        line_start.append(line_start[-1] + len(line) + 1)
    scopes = _Scopes(source)
    tainted = {}  # name -> set of declaring scopes in which the name is tainted
    members = set()  # names tainted as type members / functions / computed properties: reachable after any `.`

    def mentions(start, end):
        for match in IDENTIFIER.finditer(source, start, end):
            name = match.group()
            kind = _classify(source, match.start(), match.end())
            if kind == 'member':
                if name in members:
                    return True
            elif kind == 'use' and tainted.get(name, set()) & scopes.ancestors[scopes.at[match.start()]]:
                return True
        return False

    def is_member_site(offset):
        scope = scopes.at[offset]
        return bool(scope) and scopes.transparent[scope]

    opens_at = {opened: scope for scope, opened in enumerate(scopes.opens) if scope}

    def closure_end(text, offset, expression, keyword, end):
        """End of the expression that starts at text[expression:], extended over a closure body.

        Covers `= {` (also after if/guard: an `=` directly followed by `{` is a closure), the
        Allman form (`=` ending the statement, `{` on the next line), and a statement ending in
        a closure opener (`.map { _ in`) unless it is introduced by if/guard/while/for/switch/case.
        """
        rest = text[expression:]
        bare = rest.lstrip()
        scope = None
        if bare.startswith('{'):
            scope = opens_at.get(offset + len(text) - len(bare))
        elif not bare and not CONTROL_FLOW.search(text[:keyword]):
            following = len(source) - len(source[offset + len(text):].lstrip())
            scope = opens_at.get(following)
        elif not CONTROL_FLOW.search(text[:keyword]):
            opener = CLOSURE_OPENER.search(text, expression)
            scope = opens_at.get(offset + opener.start()) if opener else None
        return max(end, scopes.closes[scope] + 1) if scope else end

    declared = []  # (name, declaring scope, is type member) for every let/var, initialised or not
    candidates = []  # (name, scope, is member, expression start, expression end)
    for number, text in statements:
        offset = line_start[number - 1]
        for declaration in DECLARATION_NAME.finditer(text):
            at = offset + declaration.start()
            declared.append((declaration.group(1), scopes.declaring(at), is_member_site(at)))
        for declaration in DECLARATION.finditer(text):
            at = offset + declaration.start()
            start, end = offset + declaration.start(2), offset + len(text)
            end = closure_end(text, offset, declaration.start(2), declaration.start(), end)
            candidates.append((declaration.group(1), scopes.declaring(at), is_member_site(at), start, end))
        for assignment in ASSIGNMENT.finditer(text):
            name, at = assignment.group(2), offset + assignment.start(2)
            visible = scopes.ancestors[scopes.at[at]]
            homes = [d for d in declared if d[0] == name and d[1] in visible]
            if assignment.group(1) or not homes:  # `self.x`, `h.x`, `T.x`, or no visible declaration
                scope, member = 0, True
            else:
                scope, member = max(homes, key=lambda d: d[1])[1:]
            end = closure_end(text, offset, assignment.start(3), assignment.start(), offset + len(text))
            candidates.append((name, scope, member, offset + assignment.start(3), end))
    candidates += [(name, 0, True, start, end) for name, start, end in scopes.bodies()]
    changed = True
    while changed:  # transitive closure over declarations, assignments, functions and computed properties
        changed = False
        for name, scope, member, start, end in candidates:
            if (scope not in tainted.get(name, ()) or (member and name not in members)) \
                    and (SOURCE_LOCATION.search(source, start, end) or mentions(start, end)):
                tainted.setdefault(name, set()).add(scope)
                if member:
                    members.add(name)
                changed = True
    for number, text in statements:
        if not WRITE_OPERATION.search(text):
            continue
        offset = line_start[number - 1]
        if SOURCE_LOCATION.search(text) or mentions(offset, offset + len(text)):
            found.append((number, 'write location derived from #filePath'))
        if RELATIVE_LITERAL.search(text):
            found.append((number, 'write to a relative literal path'))
    return sorted(set(found))


def scan(tests_dir):
    result = {}
    for path in sorted(Path(tests_dir).rglob('*.swift')):
        hits = violations(path.read_text(encoding='utf-8'))
        if hits:
            result[str(path)] = hits
    return result


class SwiftTestOutputPathTests(unittest.TestCase):
    def test_repository_tests_do_not_infer_paths_from_directory_names(self):
        self.assertEqual(scan(ROOT / 'Tests'), {})

    def test_negative_control_flags_synthetic_heuristics(self):
        equal = 'let b = url.deletingLastPathComponent().lastPathComponent == "x" ? a : b'
        unequal = 'if root.lastPathComponent  != name { }'
        literal = 'let s = "workspaces"'
        self.assertEqual([r for _, r in violations(equal)], ['lastPathComponent comparison'])
        self.assertEqual([r for _, r in violations(unequal)], ['lastPathComponent comparison'])
        self.assertEqual([r for _, r in violations(literal)], ['"workspaces" literal'])
        both = 'x.lastPathComponent == "workspaces"'
        self.assertEqual(len(violations(both)), 2)

    def test_negative_control_accepts_compliant_sources(self):
        ok = ('let d = FileManager.default.temporaryDirectory.appendingPathComponent("R05-children")\n'
              'let n = url.lastPathComponent\n')
        self.assertEqual(violations(ok), [])

    ORIGINAL_R04 = (
        '        let fallback=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent()'
        '.deletingLastPathComponent().appendingPathComponent(".build/R04-child-captures").path\n'
        '        let directory=env["R04_CAPTURE_DIRECTORY"] ?? fallback\n'
        '        try FileManager.default.createDirectory(atPath:directory,withIntermediateDirectories:true)\n'
        '        let path=URL(fileURLWithPath:directory).appendingPathComponent("R04-\\(scenario).log")\n'
        '        FileManager.default.createFile(atPath:path.path,contents:nil)\n')

    def test_negative_control_flags_the_original_r04_capture_path(self):
        rules = {rule for _, rule in violations(self.ORIGINAL_R04)}
        self.assertEqual(rules, {'.build path literal', 'write location derived from #filePath'})

    def test_taint_is_transitive_without_a_build_literal(self):
        source = ('let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()\n'
                  'let out = root.appendingPathComponent("captures")\n'
                  'let dir = env["X"] ?? out.path\n'
                  'try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)\n')
        self.assertEqual(violations(source), [(4, 'write location derived from #filePath')])

    def test_reading_a_filepath_derived_fixture_is_allowed(self):
        source = ('let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()\n'
                  'return try Data(contentsOf: root.appendingPathComponent("Scripts/ci/fixtures/S03-x.json"))\n'
                  'let tmp = FileManager.default.temporaryDirectory\n'
                  'FileManager.default.createFile(atPath: tmp.path, contents: nil)\n')
        self.assertEqual(violations(source), [])

    def test_build_literal_is_flagged_but_names_containing_build_are_not(self):
        self.assertEqual([r for _, r in violations('let p = ".build/x"')], ['.build path literal'])
        self.assertEqual([r for _, r in violations('let p = "Tests.build"')], ['.build path literal'])
        self.assertEqual(violations('let rebuild = "rebuild-needed"\nlet b = "a build"'), [])

    # ---- Negative controls: one per pattern added in fix round 1 ----
    TAINTED = 'let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()\n'

    def rules(self, source):
        return {rule for _, rule in violations(source)}

    def test_multiline_write_statement_is_joined(self):
        source = (self.TAINTED + 'try FileManager.default.createDirectory(\n'
                  '    at: root,\n    withIntermediateDirectories: true)\n')
        self.assertEqual(violations(source), [(2, 'write location derived from #filePath')])

    def test_multiline_initializer_taints_its_name(self):
        source = ('let out = URL(\n    fileURLWithPath: #filePath)\n'
                  'FileManager.default.createFile(atPath: out.path, contents: nil)\n')
        self.assertEqual(violations(source), [(3, 'write location derived from #filePath')])

    def test_function_body_taints_the_function_name(self):
        source = ('func captureRoot() -> URL {\n    URL(fileURLWithPath: #filePath).deletingLastPathComponent()\n}\n'
                  'try FileManager.default.createDirectory(at: captureRoot(), withIntermediateDirectories: true)\n')
        self.assertEqual(violations(source), [(4, 'write location derived from #filePath')])

    def test_computed_property_body_taints_the_property_name(self):
        source = ('var base: URL {\n    URL(fileURLWithPath: #file)\n}\n'
                  'try data.write(to: base.appendingPathComponent("x"))\n')
        self.assertIn('write location derived from #filePath', self.rules(source))

    def test_default_argument_source_location_does_not_taint(self):
        source = ('func record(file: StaticString = #filePath) { print(file) }\n'
                  'FileManager.default.createFile(atPath: record(), contents: nil)\n')
        self.assertEqual(violations(source), [])

    def test_each_new_write_operation_is_recognised(self):
        operations = [
            'FileHandle(forWritingAtPath: root.path)',
            'FileHandle(forWritingTo: root)',
            'FileHandle(forUpdating: root)',
            'FileHandle(forUpdatingAtPath: root.path)',
            'creat(root.path, 0o644)',
            'OutputStream(toFileAtPath: root.path, append: false)',
            'OutputStream(url: root, append: false)',
            'open(root.path, O_WRONLY | O_CREAT, 0o644)',
            'open(root.path, O_RDWR)',
            'fopen(root.path, "w")',
            'try FileManager.default.createSymbolicLink(at: root, withDestinationURL: root)',
            'try "x".write(toFile: root.path, atomically: true, encoding: .utf8)',
            'try data.write(to: root)',
        ]
        for operation in operations:
            with self.subTest(operation=operation):
                self.assertEqual(self.rules(self.TAINTED + operation + '\n'), {'write location derived from #filePath'})

    def test_read_only_open_and_fopen_are_allowed(self):
        self.assertEqual(violations(self.TAINTED + 'open(root.path, O_RDONLY)\nfopen(root.path, "r")\n'), [])

    def test_relative_literal_writes_are_flagged(self):
        sources = [
            'FileManager.default.createDirectory(atPath: "captures", withIntermediateDirectories: true)',
            'FileManager.default.createFile(atPath: "out/log.txt", contents: nil)',
            'try data.write(to: URL(fileURLWithPath: "out.json"))',
            'try "x".write(toFile: "out.txt", atomically: true, encoding: .utf8)',
            'fopen("log.txt", "w")',
            'open("log.txt", O_WRONLY)',
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assertEqual(self.rules(source), {'write to a relative literal path'})

    def test_absolute_literal_writes_are_allowed(self):
        self.assertEqual(violations('FileManager.default.createFile(atPath: "/private/tmp/x.log", contents: nil)'), [])

    def test_current_directory_writes_are_flagged(self):
        source = ('let cwd = FileManager.default.currentDirectoryPath\n'
                  'FileManager.default.createFile(atPath: cwd + "/x", contents: nil)\n')
        self.assertEqual(violations(source), [(2, 'write location derived from #filePath')])
        self.assertEqual(self.rules('try data.write(to: URL(fileURLWithPath: FileManager.default.currentDirectoryPath))'),
                         {'write location derived from #filePath'})

    # ---- T05: label-aware and scope-aware taint ----
    V04_SHAPE = (
        'final class Suite {\n'
        '    func testFingerprint() throws {\n'
        '        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()\n'
        '        let source = try String(contentsOf: root.appendingPathComponent("Suites/A.swift"))\n'
        '        let name = source.split(separator: "\\n").count\n'
        '        let cases = [name]\n'
        '        XCTAssertFalse(cases.isEmpty)\n'
        '    }\n'
        '    func testWritesCSV() throws {\n'
        '        let csvOutputPath = FileManager.default.temporaryDirectory.appendingPathComponent("out.csv").path\n'
        '        let cases = [1]\n'
        '        let name = "x"\n'
        '        try CSVWriter.write(cases: cases,\n'
        '                            name: name,\n'
        '                            metadata: [:],\n'
        '                            unavailable: [],\n'
        '                            to: csvOutputPath)\n'
        '    }\n'
        '}\n')

    def test_v04_false_positive_shape_passes(self):  # control (a)
        self.assertEqual(violations(self.V04_SHAPE), [])

    def test_argument_labels_are_not_mentions(self):
        source = (self.TAINTED + 'let name = root\n'
                  'try CSVWriter.write(name: 1, to: FileManager.default.temporaryDirectory)\n'
                  'try CSVWriter.write(1, name: 2, to: FileManager.default.temporaryDirectory)\n')
        self.assertEqual(violations(source), [])

    def test_ternary_branch_is_still_a_mention(self):
        source = (self.TAINTED + 'let flag = true\n'
                  'try data.write(to: flag ? root : FileManager.default.temporaryDirectory)\n')
        self.assertEqual(violations(source), [(3, 'write location derived from #filePath')])

    def test_member_names_are_not_mentions_but_tainted_bases_are(self):
        untainted_base = (self.TAINTED + 'let tmp = FileManager.default.temporaryDirectory\n'
                          'let url = URL(fileURLWithPath: #filePath)\n'
                          'try data.write(to: tmp.root)\n')
        self.assertEqual(violations(untainted_base), [])
        tainted_base = self.TAINTED + 'try data.write(to: root.deletingLastPathComponent())\n'  # control (e)
        self.assertEqual(violations(tainted_base), [(2, 'write location derived from #filePath')])
        member_of_tainted = self.TAINTED + 'let x = root\ntry data.write(to: x.path)\n'
        self.assertEqual(violations(member_of_tainted), [(3, 'write location derived from #filePath')])

    def test_same_named_untainted_local_in_another_function_passes(self):  # control (b)
        source = ('func reader() { let root = URL(fileURLWithPath: #filePath); print(root) }\n'
                  'func writer() {\n'
                  '    let root = FileManager.default.temporaryDirectory\n'
                  '    try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)\n'
                  '}\n')
        self.assertEqual(violations(source), [])

    def test_tainted_local_used_in_a_write_in_the_same_function_fails(self):  # control (c)
        source = ('func writer() {\n'
                  '    let root = URL(fileURLWithPath: #filePath)\n'
                  '    if true {\n'
                  '        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)\n'
                  '    }\n'
                  '}\n')
        self.assertEqual(violations(source), [(4, 'write location derived from #filePath')])

    def test_tainted_function_and_property_names_taint_file_wide(self):  # control (d)
        function = ('func benchRoot() -> URL {\n    let r = URL(fileURLWithPath: #filePath)\n    return r\n}\n'
                    'func writer() {\n    try? FileManager.default.createDirectory(at: benchRoot(), withIntermediateDirectories: true)\n}\n')
        self.assertEqual(violations(function), [(6, 'write location derived from #filePath')])
        prop = ('final class T {\n    var base: URL {\n        URL(fileURLWithPath: #file)\n    }\n'
                '    func writer() {\n        try? data.write(to: base.appendingPathComponent("x"))\n    }\n}\n')
        self.assertEqual(violations(prop), [(6, 'write location derived from #filePath')])

    def test_type_member_declarations_taint_file_wide(self):
        source = ('final class T {\n    static let root = URL(fileURLWithPath: #filePath)\n'
                  '    func writer() { try? data.write(to: T.root) }\n}\n')
        self.assertEqual(violations(source), [(3, 'write location derived from #filePath')])

    def test_taint_flows_through_nested_bodies_of_the_declaring_scope(self):
        source = ('func writer() {\n    let root = URL(fileURLWithPath: #filePath)\n'
                  '    let run = {\n        try? data.write(to: root)\n    }\n    run()\n}\n')
        self.assertEqual(violations(source), [(4, 'write location derived from #filePath')])

    def test_local_taint_does_not_leak_through_a_sibling_type_member(self):
        source = ('func reader() { let root = URL(fileURLWithPath: #filePath); print(root) }\n'
                  'final class T {\n    func writer(root: URL) { try? data.write(to: root) }\n}\n')
        self.assertEqual(violations(source), [])

    def test_braces_in_strings_and_comments_do_not_shift_scopes(self):
        source = ('func reader() {\n    let s = "}} {"  // } }\n'
                  '    let root = URL(fileURLWithPath: #filePath)\n}\n'
                  'func writer() {\n    let root = FileManager.default.temporaryDirectory\n'
                  '    try? data.write(to: root)\n}\n')
        self.assertEqual(violations(source), [])

    def test_nested_shadow_of_an_outer_tainted_name_is_still_flagged(self):
        # Documented conservative limit: shadowing in a nested scope is not modelled.
        source = (self.TAINTED + 'func other() { let root = FileManager.default.temporaryDirectory\n'
                  'try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true) }\n')
        self.assertTrue(violations(source))

    # ---- T05 round 1: member-like taint, closure initialisers, assignments ----
    MUST_FAIL = (
        'final class H { var out: URL { URL(fileURLWithPath: #filePath) } }\n'
        'func w() { let h = H(); try? d.write(to: h.out) }\n',
        'final class H { let out = URL(fileURLWithPath: #filePath) }\n'
        'func w() { try? d.write(to: H().out) }\n',
        'final class H { static let shared = H(); let out = URL(fileURLWithPath: #filePath) }\n'
        'func w() { try? d.write(to: H.shared.out) }\n',
        'final class T { let paths = Paths(); func w() { try? d.write(to: paths.out) } }\n'
        'struct Paths { var out: URL { URL(fileURLWithPath: #filePath) } }\n',
        'func fixtureRoot() -> URL { URL(fileURLWithPath: #filePath) }\n'
        'extension URL { func sub() -> URL { fixtureRoot() } }\n'
        'func w() { try? d.write(to: tmp.sub()) }\n',
        'final class T { static let root = URL(fileURLWithPath: #filePath)\n'
        '    func w() { try? d.write(to: type(of: self).root) } }\n',
    )

    def test_member_like_taint_is_reached_through_any_base(self):
        for source in self.MUST_FAIL:
            with self.subTest(source=source):
                self.assertEqual(self.rules(source), {'write location derived from #filePath'})

    def test_local_names_keep_the_member_exemption(self):
        source = ('func r() { let out = URL(fileURLWithPath: #filePath); print(out) }\n'
                  'func w(h: H) { try? d.write(to: h.out) }\n')
        self.assertEqual(violations(source), [])

    def test_multiline_closure_initialisers_taint_their_name(self):
        static = ('final class T {\n    static let root: URL = {\n        let p = URL(fileURLWithPath: #filePath)\n'
                  '        return p\n    }()\n    func w() { try? d.write(to: T.root) }\n}\n')
        local = ('func w() {\n    let x: URL = {\n        let p = URL(fileURLWithPath: #filePath)\n'
                 '        return p\n    }()\n    try? d.write(to: x)\n}\n')
        self.assertEqual(violations(static), [(6, 'write location derived from #filePath')])
        self.assertEqual(violations(local), [(6, 'write location derived from #filePath')])

    def test_closure_initialiser_without_a_source_location_is_clean(self):
        source = ('func w() {\n    let x: URL = {\n        let p = FileManager.default.temporaryDirectory\n'
                  '        return p\n    }()\n    try? d.write(to: x)\n}\n')
        self.assertEqual(violations(source), [])

    def test_setup_assignment_taints_the_member(self):
        source = ('final class T {\n    var dir: URL!\n'
                  '    override func setUp() {\n        dir = URL(fileURLWithPath: #filePath)\n    }\n'
                  '    func w() { try? d.write(to: dir) }\n}\n')
        self.assertEqual(violations(source), [(6, 'write location derived from #filePath')])
        explicit = source.replace('        dir = URL', '        self.dir = URL')
        self.assertEqual(violations(explicit), [(6, 'write location derived from #filePath')])

    def test_local_assignment_taints_only_its_scope(self):
        source = ('func r() { var dir: URL; dir = URL(fileURLWithPath: #filePath); print(dir) }\n'
                  'func w() { var dir = FileManager.default.temporaryDirectory; try? d.write(to: dir) }\n')
        self.assertEqual(violations(source), [])
        same = 'func w() { var dir: URL\n dir = URL(fileURLWithPath: #filePath)\n try? d.write(to: dir) }\n'
        self.assertEqual(violations(same), [(3, 'write location derived from #filePath')])

    def test_every_declaration_in_one_statement_is_seen(self):
        # Two declarations share a statement; the write is a separate statement, so only taint of `b`
        # (the SECOND declaration) can flag it. Fails if DECLARATION stops being an overlapping lookahead.
        source = 'func w() { let a = 1; let b = URL(fileURLWithPath: #filePath)\n    try? d.write(to: b) }\n'
        self.assertEqual(violations(source), [(2, 'write location derived from #filePath')])

    def test_url_scheme_in_a_string_does_not_truncate_the_line(self):
        source = ('let u = URL(string: "http://x")!; let root = URL(fileURLWithPath: #filePath)\n'
                  'try? d.write(to: root)\n')
        self.assertEqual(violations(source), [(2, 'write location derived from #filePath')])
        multi = 'let s = foo("http://x",\n   root)\n'
        self.assertEqual([n for n, _ in _statements(multi)], [1])

    # ---- T05 round 2 ----
    def test_base_qualified_assignment_taints_the_member(self):
        source = ('final class H { var out: URL! }\n'
                  'func w() { let t = URL(fileURLWithPath: #filePath); let h = H(); h.out = t; try? d.write(to: h.out) }\n')
        self.assertEqual(violations(source), [(2, 'write location derived from #filePath')])
        for target in ('T.dir', 'Self.dir', 'self.dir'):
            with self.subTest(target=target):
                source = ('final class T { static var dir: URL!\n    func w() {\n'
                          '        let t = URL(fileURLWithPath: #filePath)\n        %s = t\n'
                          '        try? d.write(to: type(of: self).dir)\n    } }\n' % target)
                self.assertEqual(violations(source), [(5, 'write location derived from #filePath')])

    def test_closure_parameter_lists_and_allman_openers(self):
        for opener in ('{ _ in', '{ arg in', '{ (a, b) in', '{ [weak self] x in', '{ a, b in'):
            with self.subTest(opener=opener):
                source = ('let x = [1].map %s\n    URL(fileURLWithPath: #filePath)\n}; try? d.write(to: x[0])\n' % opener)
                self.assertEqual(violations(source), [(3, 'write location derived from #filePath')])
        allman = 'let x: URL =\n{\n    URL(fileURLWithPath: #filePath)\n}()\ntry? d.write(to: x)\n'
        self.assertEqual(violations(allman), [(5, 'write location derived from #filePath')])
        clean = 'let x = [1].map { _ in\n    FileManager.default.temporaryDirectory\n}\ntry? d.write(to: x[0])\n'
        self.assertEqual(violations(clean), [])

    def test_closure_assigned_after_declaration_uses_the_body(self):
        source = ('func w() {\n    var x: URL\n    x = {\n        URL(fileURLWithPath: #filePath)\n    }()\n'
                  '    try? d.write(to: x)\n}\n')
        self.assertEqual(violations(source), [(6, 'write location derived from #filePath')])
        member = ('final class T {\n    var x: URL!\n    override func setUp() {\n        self.x = {\n'
                  '            URL(fileURLWithPath: #filePath)\n        }()\n    }\n    func w() { try? d.write(to: x) }\n}\n')
        self.assertEqual(violations(member), [(8, 'write location derived from #filePath')])

    def test_if_let_closure_initialiser_uses_the_body(self):
        source = ('func w() {\n    if let x: URL = {\n        URL(fileURLWithPath: #filePath)\n    }() {\n'
                  '        try? d.write(to: x)\n    }\n}\n')
        self.assertEqual(violations(source), [(5, 'write location derived from #filePath')])

    def test_paren_counting_uses_masked_text(self):
        string_paren = 'let s = foo(\n    "a)",\n    root)\nlet t = 1\n'
        self.assertEqual([n for n, _ in _statements(string_paren)], [1, 4])
        comment_paren = 'let t = 1 // (\nlet u = 2\n'
        self.assertEqual([n for n, _ in _statements(comment_paren)], [1, 2])
        # The test is only meaningful if raw counting would differ.
        raw = swift_lexing._mask  # _statements resolves _mask in swift_lexing's namespace
        try:
            swift_lexing._mask = lambda text: text
            self.assertNotEqual([n for n, _ in _statements(string_paren)], [1, 4])
            self.assertNotEqual([n for n, _ in _statements(comment_paren)], [1, 2])
        finally:
            swift_lexing._mask = raw


if __name__ == '__main__':
    unittest.main()
