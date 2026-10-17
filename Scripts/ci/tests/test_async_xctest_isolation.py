"""Policy (HEURISTIC): an async XCTest test method must be main-actor isolated.

Why (T08, V04/phaseC-fix5-rulings.md): on Swift 6.4 / Xcode 27 a NONISOLATED `async` XCTest test
method that does real awaited work between assertions recorded a varying subset of its failures
(33-48 of 48+, 56-58 of 58, identical deterministic computations); sync and `@MainActor` methods
recorded all of them in every run. A method whose failures are ALL dropped reports PASSED. The
mechanism is unproven (a minimal standalone package does not reproduce it), so this policy is
empirical: it forbids the shape that was observed to lose failures.

RULES, applied to every Tests/**/*.swift file (all files are analysed together, so a class
hierarchy or an extension may span files):

* R1 (test methods): in a type that derives from `XCTestCase`, a member `func test…()` with NO
  parameters and an `async` effect (`async`, `async throws`, `throws(E) async`, also split across
  lines) must be main-actor isolated. It is isolated if (a) the method has `@MainActor`, or
  (b) its enclosing type is: the class (or a superclass declared in the scanned files) has
  `@MainActor`, or the enclosing `extension` has `@MainActor`, or the extension's class does.
  A `nonisolated` modifier or a `@concurrent` attribute on the method is never isolated, even in a
  `@MainActor` class or extension (`@concurrent`: T08/review/probes/concurrent_probe.swift).
  `extension XCTestCase` members are checked too (they reach every test class).
* R2 (assertion helpers): a member `async` function that is NOT a test (setUp/tearDown overrides
  included), whose body mentions `XCT[A-Za-z]+(` other than `XCTSkip(`, `XCTestExpectation(`,
  `XCTestCase(` and `XCTWaiter(` (they are not assertions), and which is not isolated by the same rule, is flagged: its assertions would be
  recorded from a nonisolated async context, the shape the finding is about. A helper may instead
  return its values to a main-actor test that asserts them.

DERIVATION: a `class` is XCTestCase-derived if an entry of its inheritance list is `XCTestCase`
(also `XCTest.XCTestCase`) or the name of another scanned class that is, to a fixpoint across files.
An `extension T` member is checked when T is derived or is `XCTestCase`. Classes are keyed by simple
name. Headers are joined up to the `{` (multi-line inheritance lists), and generic arguments are
removed before the list is split (`Base<Int, String>` -> `Base`). Parameter lists are scanned with
balanced parentheses; `async` is read from the function's OWN effects clause after them.

LIMITS (lexical heuristic on the shared lexer Scripts/ci/swift_lexing.py; not a parser; nothing is
proven beyond the named controls in this file):
* Not seen: a test class whose XCTestCase ancestry is reached through a TYPEALIAS, a class declared
  outside Tests/ (a base class from another module or from Sources), or a macro-generated class.
  (A subclass declared inside a function body IS seen and flagged: control 'subclass declared inside
  a function body'.) Isolation inferred by Swift from a `@MainActor` PROTOCOL conformance is not
  credited, so such a class is flagged (fail-safe).
* Duplicate simple names (nested types in different files) are merged: derived if ANY declaration is,
  main-actor-isolated only if EVERY declaration is.
* Helpers: R2 sees only members of a derived type, and only by the textual `XCT…(` in the body. NOT
  seen: free functions and helpers of non-XCTestCase types that assert (e.g. `func assertFoo() async`
  at file scope or in a helper struct), assertions reached through a closure passed to such a helper,
  assertions made through `#expect`/Swift Testing, and helpers that merely call another assertion
  helper. A @MainActor method that calls a nonisolated async helper hands the assertions to it.
* Extensions: an `@MainActor` extension isolates its own members only; a plain extension of a
  `@MainActor` class inherits the class isolation (Swift semantics). `@MainActor(unsafe)` counts as
  `@MainActor`. Any other global actor does NOT count (only `@MainActor` is the observed-safe shape).
* `nonisolated` overrides: a `nonisolated` method/helper is flagged wherever it is; a method with
  an `override` of a nonisolated base-class `func test…() async` is judged by its own spelling only.
* Only zero-parameter `test…` methods are XCTest tests; `static`/`class` funcs are skipped.
  Swift Testing `@Test` functions are out of scope (they were not observed to lose failures).
* `#if` branches are all scanned without preprocessing.
* The attribute/modifier scan of a declaration looks at the same line before `func` plus the
  immediately preceding lines that contain only attributes or modifiers. An attribute separated by
  a blank line or other code is not attributed to the method (a fail-safe false positive).

CONTROLS (all in this file): NEGATIVE: inline fixtures reproducing the pre-conversion shapes
(KernelAutoTuner-style tests, the ExecutionBoundary `checkHints` helper, the BenchmarkContractTests
`assertSweepProperty` sweep) are RED with exact counts and green once `@MainActor` is added, so RED
needs no git history. When the history is available, the real pre-conversion files are also checked
(git show 657a100: the 24 methods and 1 helper in KernelAutoTunerTests/ExecutionBoundaryTests;
git show tree 1c84131: the 16 methods and the `assertSweepProperty` helper of BenchmarkContractTests).
Those history checks raise unittest.SkipTest when the revision is unavailable (CI's shallow checkout,
an export outside any repository, a loose object that was never pushed); they never error.
Inline must-flag fixtures cover every spelling. POSITIVE: inline must-pass fixtures, and the live
tree must have zero findings while the guard demonstrably SEES at least the converted methods (lower
bounds, so new async tests need no guard edit; every async test it sees must be isolated).
MUTATION: disabling each rule makes its must-flag controls pass unflagged.
"""

import importlib.util
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location('swift_lexing', ROOT / 'Scripts/ci/swift_lexing.py')
lexing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lexing)
_mask, _Scopes = lexing._mask, lexing._Scopes

INTEGRATION_COMMIT = '657a100'  # pre-conversion root tree (24 methods + 1 helper); may be unavailable (shallow clone)
V04_BASE_TREE = '1c84131'       # pre-conversion BenchmarkContractTests (16 methods); an unreachable tree, may be unavailable
RULES = ('derivation_fixpoint', 'method_attr', 'class_attr', 'extension_attr', 'nonisolated', 'concurrent',
         'effect_across_lines', 'helper_rule')

TYPE_KEYWORD = re.compile(r'(?<![\w.])(class|struct|enum|extension|actor|protocol)\s+(?!func\b|var\b|let\b)([A-Za-z_][\w.]*)')
ATTR_OR_MODIFIER_ONLY = re.compile(r'^\s*(?:(?:@[\w.]+(?:\([^)]*\))?|[a-z]+(?:\([^)]*\))?)\s*)*$')
FUNC_NAME = re.compile(r'(?<![\w.])func\s+(\w+)')
EFFECTS = re.compile(r'(?:async|throws|rethrows|\([^)]*\)|\s)*')
# XCTSkip / XCTestExpectation / XCTestCase / XCTWaiter are not assertions (M3).
XCT_CALL = re.compile(r'\bXCT(?!Skip\b|estExpectation\b|estCase\b|Waiter\b)[A-Za-z]+\s*\(')
MAINACTOR = re.compile(r'@MainActor\b')
NONISOLATED = re.compile(r'\bnonisolated\b')
# `@concurrent` is NOT main-actor isolated even inside a @MainActor class or extension: on Swift 6.4
# T08/review/probes/concurrent_probe.swift reports "main actor-isolated ... cannot be called from
# outside of the actor" for a call from a `@concurrent func test...() async` in a @MainActor class.
CONCURRENT = re.compile(r'@concurrent\b')


def _line_decl_prefix(mask, keyword_start):
    """(attributes+modifiers text) of the declaration whose keyword begins at `keyword_start`."""
    line_start = mask.rfind('\n', 0, keyword_start) + 1
    parts = [mask[line_start:keyword_start]]
    cursor = line_start
    while cursor > 0:
        prev_end = cursor - 1
        prev_start = mask.rfind('\n', 0, prev_end) + 1
        line = mask[prev_start:prev_end]
        if line.strip() and ATTR_OR_MODIFIER_ONLY.match(line) and not any(c in line for c in '{};'):
            parts.append(line)
            cursor = prev_start
        else:
            break
    return ' '.join(parts)


def _boundary_before(mask, index):
    return max(mask.rfind(c, 0, index) for c in '{};') + 1


def _balanced(mask, start, open_ch, close_ch):
    """Index just past the delimiter matching mask[start] == open_ch, or -1. `->` never closes `>`."""
    depth = 0
    for i in range(start, len(mask)):
        if mask[i] == open_ch:
            depth += 1
        elif mask[i] == close_ch and not (close_ch == '>' and mask[i - 1] == '-'):
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def _strip_generics(text):
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r'<[^<>]*>', '', text)
    return text


def _type_info(mask, scopes, scope):
    """(kind, simple name, attribute text, inherited names) for a type scope, or None.

    The header is everything from the previous `{`, `}` or `;` up to this scope's `{`, so a multi-line
    inheritance list is joined (the shared lexer's transparent flag keeps only the last line and is not
    used here). Generic arguments are removed before the inheritance list is split, so `Base<Int, String>`
    contributes the base name `Base`.
    """
    open_at = scopes.opens[scope]
    header = mask[_boundary_before(mask, open_at):open_at]
    base = open_at - len(header)
    for match in TYPE_KEYWORD.finditer(header):
        line_start = header.rfind('\n', 0, match.start()) + 1
        if not ATTR_OR_MODIFIER_ONLY.match(header[line_start:match.start()]):
            continue
        attrs = _line_decl_prefix(mask, base + match.start())
        rest = _strip_generics(header[match.end():])
        rest = re.split(r'\bwhere\b', rest)[0]
        inherited = []
        if ':' in rest:
            for entry in rest.split(':', 1)[1].split(','):
                names = re.findall(r'[A-Za-z_][\w.]*', entry)
                if names:
                    inherited.append(names[-1].split('.')[-1])
        return match.group(1), _strip_generics(match.group(2)).split('.')[-1], attrs, inherited
    return None


def _function_headers(mask):
    """Yield (name, start, params text, effects text, end of effects) per `func`, with balanced parens."""
    for match in FUNC_NAME.finditer(mask):
        i = match.end()
        while i < len(mask) and mask[i].isspace():
            i += 1
        if i < len(mask) and mask[i] == '<':
            i = _balanced(mask, i, '<', '>')
            while 0 < i < len(mask) and mask[i].isspace():
                i += 1
        if i < 0 or i >= len(mask) or mask[i] != '(':
            continue
        end = _balanced(mask, i, '(', ')')
        if end < 0:
            continue
        effects = EFFECTS.match(mask, end)
        yield match.group(1), match.start(), mask[i + 1:end - 1], effects.group(0), effects.end(), end


def analyze(files, disabled=frozenset()):
    """Scan [(path, source)]; return (findings, seen) where `seen` lists every guarded async test.

    findings: sorted [(path, line, type, member, rule)] with rule 'R1' or 'R2'.
    seen: sorted [(path, line, type, member)] of every R1-eligible async test method (isolated or not).
    `disabled` removes a rule from RULES (mutation controls only).
    """
    parsed = []
    classes = {}  # simple name -> list of (inherited, mainactor)
    for path, source in files:
        mask = _mask(source)
        scopes = _Scopes(source)
        types = {}
        for scope in range(1, len(scopes.parent)):
            info = _type_info(mask, scopes, scope)
            if info:
                types[scope] = info
                kind, name, attrs, inherited = info
                if kind == 'class':
                    classes.setdefault(name, []).append((inherited, bool(MAINACTOR.search(attrs))))
        parsed.append((path, source, mask, scopes, types))

    # `extension XCTestCase` members reach every test class, so XCTestCase itself is a checked type.
    derived = {'XCTestCase'} | {name for name, decls in classes.items()
                                if any('XCTestCase' in inherited for inherited, _ in decls)}
    if 'derivation_fixpoint' not in disabled:
        changed = True
        while changed:
            changed = False
            for name, decls in classes.items():
                if name not in derived and any(set(inherited) & derived for inherited, _ in decls):
                    derived.add(name)
                    changed = True

    def class_isolated(name, seen=()):
        if 'class_attr' in disabled or name in seen or name not in classes:
            return False
        decls = classes[name]
        if all(attr for _, attr in decls):
            return True
        return any(class_isolated(parent, seen + (name,))
                   for inherited, _ in decls for parent in inherited)

    findings, observed = [], []
    for path, source, mask, scopes, types in parsed:
        for name, start, params, effects, effects_end, params_end in _function_headers(mask):
            owner = scopes.at[start]
            if owner not in types:
                continue
            kind, type_name, type_attrs, _ = types[owner]
            if kind not in ('class', 'extension') or type_name not in derived:
                continue
            decl = _line_decl_prefix(mask, start)
            if re.search(r'\b(?:static|class)\b', decl):
                continue
            # The effects clause is read from after the BALANCED parameter list, so a closure-typed
            # parameter's `async` is never taken for the function's own effect.
            effect_text = effects if 'effect_across_lines' not in disabled else effects.split('\n')[0]
            if not re.search(r'\basync\b', effect_text):
                continue
            is_test = name.startswith('test') and not params.strip()
            line = source.count('\n', 0, start) + 1
            if is_test:
                observed.append((path, line, type_name, name))
            else:
                if 'helper_rule' in disabled:
                    continue
                body_scope = min((s for s in range(1, len(scopes.parent))
                                  if scopes.parent[s] == owner and scopes.opens[s] >= effects_end - 1),
                                 key=lambda s: scopes.opens[s], default=None)
                if body_scope is None or not XCT_CALL.search(mask[scopes.opens[body_scope]:scopes.closes[body_scope]]):
                    continue
            isolated = False
            if 'nonisolated' not in disabled and NONISOLATED.search(decl):
                pass
            elif 'concurrent' not in disabled and CONCURRENT.search(decl):
                pass
            elif 'method_attr' not in disabled and MAINACTOR.search(decl):
                isolated = True
            elif kind == 'extension' and 'extension_attr' not in disabled and MAINACTOR.search(type_attrs):
                isolated = True
            elif class_isolated(type_name):
                isolated = True
            if not isolated:
                findings.append((path, line, type_name, name, 'R1' if is_test else 'R2'))
    return sorted(findings), sorted(observed)


def _tests_files():
    return [(str(p.relative_to(ROOT)), p.read_text(encoding='utf-8'))
            for p in sorted((ROOT / 'Tests').rglob('*.swift'))]


def _git_show(rev, path):
    """File text at `rev`, or SkipTest when git, the repository, the object or the path is unavailable
    (CI's default shallow checkout, an export outside any repository, a loose object that was never pushed)."""
    try:
        done = subprocess.run(['git', 'show', f'{rev}:{path}'], cwd=ROOT, capture_output=True, text=True,
                              check=True, timeout=30)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, OSError) as error:
        raise unittest.SkipTest(f'{rev}:{path} is not available in this checkout ({type(error).__name__})')
    return done.stdout


def _flagged(source, *extra, disabled=frozenset()):
    files = [('fixture.swift', source)] + [(f'extra{i}.swift', s) for i, s in enumerate(extra)]
    return analyze(files, disabled)[0]


CONVERTED_FILES = {
    'Tests/ComprehensiveTests/KernelAutoTunerTests.swift': 22,
    'Tests/ComprehensiveTests/ExecutionBoundaryTests.swift': 2,
    'Tests/BenchmarkContractTests/BenchmarkSelectionContractTests.swift': 15,
    'Tests/BenchmarkContractTests/RunnerContractTests.swift': 1,
}

# (name, source, extra sources, rule that must be removed to make it pass unflagged)
MUST_FLAG = [
    ('direct', 'final class A: XCTestCase {\n    func testX() async {}\n}\n', (), None),
    ('async throws', 'final class A: XCTestCase {\n    func testX() async throws {}\n}\n', (), None),
    ('throws(E) async', 'final class A: XCTestCase {\n    func testX() throws(MyErr) async {}\n}\n', (), None),
    ('effect on next line', 'final class A: XCTestCase {\n    func testX()\n        async throws {}\n}\n', (), 'effect_across_lines'),
    ('unchecked sendable', 'final class A: XCTestCase, @unchecked Sendable {\n    func testX() async {}\n}\n', (), None),
    ('indirect subclass', 'class B: XCTestCase {}\nfinal class A: B {\n    func testX() async {}\n}\n', (), 'derivation_fixpoint'),
    ('cross-file subclass', 'final class A: B {\n    func testX() async {}\n}\n', ('class B: XCTestCase {}\n',), 'derivation_fixpoint'),
    ('plain extension', 'final class A: XCTestCase {}\nextension A {\n    func testX() async {}\n}\n', (), None),
    ('cross-file extension', 'extension A {\n    func testX() async {}\n}\n', ('final class A: XCTestCase {}\n',), None),
    ('nonisolated in MainActor class', '@MainActor\nfinal class A: XCTestCase {\n    nonisolated func testX() async {}\n}\n', (), 'nonisolated'),
    ('nonisolated same line + attr', 'final class A: XCTestCase {\n    @MainActor nonisolated func testX() async {}\n}\n', (), 'nonisolated'),
    ('MainActor on previous method only', 'final class A: XCTestCase {\n    @MainActor\n    func testA() async {}\n\n    func testB() async {}\n}\n', (), None),
    ('MainActor on property above', 'final class A: XCTestCase {\n    @MainActor var x = 1\n    func testB() async {}\n}\n', (), None),
    ('MainActor only in a comment', 'final class A: XCTestCase {\n    // @MainActor\n    func testB() async {}\n}\n', (), None),
    ('MainActor only in a string', 'final class A: XCTestCase {\n    let s = "@MainActor"\n    func testB() async {}\n}\n', (), None),
    ('MainActor on nested type only', '@MainActor final class N {}\nfinal class A: XCTestCase {\n    @MainActor struct S {}\n    func testB() async {}\n}\n', (), None),
    ('other global actor', '@MyActor\nfinal class A: XCTestCase {\n    func testB() async {}\n}\n', (), None),
    ('MainActor on extension of another type', 'final class A: XCTestCase {}\n@MainActor extension Z {}\nextension A {\n    func testB() async {}\n}\n', (), None),
    ('class attr after blank line', '@MainActor\n\nfinal class A: XCTestCase {\n    func testB() async {}\n}\n', (), None),
    ('helper with XCTAssert', 'final class A: XCTestCase {\n    func check() async throws {\n        XCTAssertEqual(1, 1)\n    }\n}\n', (), 'helper_rule'),
    ('private helper XCTFail', 'final class A: XCTestCase {\n    private func check() async {\n        XCTFail("x")\n    }\n}\n', (), 'helper_rule'),
    ('nonisolated helper in MainActor class', '@MainActor final class A: XCTestCase {\n    nonisolated func check() async {\n        XCTAssertTrue(true)\n    }\n}\n', (), 'nonisolated'),
    ('extension XCTestCase async helper', 'extension XCTestCase {\n    func assertFoo() async {\n        XCTAssertTrue(true)\n    }\n}\n', (), None),
    ('extension XCTestCase async test', 'extension XCTestCase {\n    func testShared() async {}\n}\n', (), None),
    ('extension XCTest.XCTestCase helper', 'extension XCTest.XCTestCase {\n    func assertFoo() async {\n        XCTFail("x")\n    }\n}\n', (), None),
    ('multi-line inheritance list', 'final class A: XCTestCase,\n    @unchecked Sendable {\n    func testX() async {}\n}\n', (), None),
    ('multi-line inheritance, base on 2nd line', 'final class A:\n    XCTestCase {\n    func testX() async {}\n}\n', (), None),
    ('generic base, one argument', 'class Base<T>: XCTestCase {}\nfinal class A: Base<Int> {\n    func testX() async {}\n}\n', (), 'derivation_fixpoint'),
    ('generic base, two arguments', 'class Base<T, U>: XCTestCase {}\nfinal class A: Base<Int, String> {\n    func testX() async {}\n}\n', (), 'derivation_fixpoint'),
    ('test in a generic XCTestCase subclass', 'class Base<T>: XCTestCase where T: Equatable {\n    func testX() async {}\n}\n', (), None),
    ('@concurrent test in MainActor class', '@MainActor final class A: XCTestCase {\n    @concurrent func testX() async {}\n}\n', (), 'concurrent'),
    ('@concurrent test in MainActor extension', 'final class A: XCTestCase {}\n@MainActor extension A {\n    @concurrent func testX() async {}\n}\n', (), 'concurrent'),
    ('helper with closure-typed parameter', 'final class A: XCTestCase {\n    func check(_ f: (Int) -> Bool) async {\n        XCTAssertTrue(f(1))\n    }\n}\n', (), None),
    ('helper with call in default argument', 'final class A: XCTestCase {\n    func check(_ n: Int = Int(3)) async {\n        XCTAssertEqual(n, 3)\n    }\n}\n', (), None),
    ('helper with tuple parameter', 'final class A: XCTestCase {\n    func check(_ p: (Int, Int)) async {\n        XCTAssertEqual(p.0, p.1)\n    }\n}\n', (), None),
    ('helper with async closure parameter and own async', 'final class A: XCTestCase {\n    func check(_ f: () async -> Int) async {\n        XCTAssertEqual(await f(), 1)\n    }\n}\n', (), None),
    ('async setUp override asserting', 'final class A: XCTestCase {\n    override func setUp() async throws {\n        XCTAssertTrue(true)\n    }\n}\n', (), None),
    ('subclass declared inside a function body', 'func make() {\n    final class A: XCTestCase {\n        func testX() async {}\n    }\n}\n', (), None),
    ('duplicate name, only one MainActor', '@MainActor final class A: XCTestCase {\n    func testX() async {}\n}\n', ('enum NS {\n    final class A: XCTestCase {\n        func testY() async {}\n    }\n}\n',), None),
    ('async test inside #if', 'final class A: XCTestCase {\n#if os(macOS)\n    func testX() async {}\n#endif\n}\n', (), None),
]

MUST_PASS = [
    ('method attr same line', 'final class A: XCTestCase {\n    @MainActor func testX() async {}\n}\n'),
    ('method attr previous line', 'final class A: XCTestCase {\n    @MainActor\n    func testX() async throws {}\n}\n'),
    ('method attr then other attr', 'final class A: XCTestCase {\n    @MainActor\n    @available(macOS 13, *)\n    func testX() async {}\n}\n'),
    ('method attr with modifier', 'final class A: XCTestCase {\n    @MainActor\n    private func testX() async {}\n}\n'),
    ('class attr previous line', '@MainActor\nfinal class A: XCTestCase {\n    func testX() async {}\n}\n'),
    ('class attr same line', '@MainActor final class A: XCTestCase {\n    func testX() async {}\n}\n'),
    ('extension attr', 'final class A: XCTestCase {}\n@MainActor\nextension A {\n    func testX() async {}\n}\n'),
    ('plain extension of MainActor class', '@MainActor final class A: XCTestCase {}\nextension A {\n    func testX() async {}\n}\n'),
    ('inherits MainActor from base', '@MainActor class B: XCTestCase {}\nfinal class A: B {\n    func testX() async {}\n}\n'),
    ('sync test', 'final class A: XCTestCase {\n    func testX() {}\n    func testY() throws {}\n}\n'),
    ('test with parameter is not a test', 'final class A: XCTestCase {\n    func testX(_ n: Int) async {}\n}\n'),
    ('static func', 'final class A: XCTestCase {\n    static func testX() async {}\n}\n'),
    ('not an XCTestCase', 'final class A {\n    func testX() async {}\n}\nstruct S {\n    func testY() async {}\n}\n'),
    ('commented-out test', 'final class A: XCTestCase {\n    // func testX() async {}\n    /* func testY() async {} */\n}\n'),
    ('test in string', 'final class A: XCTestCase {\n    let s = "func testX() async {}"\n}\n'),
    ('local function inside a method', 'final class A: XCTestCase {\n    func testX() {\n        func testInner() async {}\n    }\n}\n'),
    ('MainActor helper', 'final class A: XCTestCase {\n    @MainActor\n    func check() async {\n        XCTAssertTrue(true)\n    }\n}\n'),
    ('async helper without XCT', 'final class A: XCTestCase {\n    func make() async -> Int { 1 }\n}\n'),
    ('sync helper with XCT', 'final class A: XCTestCase {\n    func check() {\n        XCTAssertTrue(true)\n    }\n}\n'),
    ('helper in a non-XCTestCase', 'struct H {\n    func check() async {\n        XCTAssertTrue(true)\n    }\n}\n'),
    ('@MainActor(unsafe)', 'final class A: XCTestCase {\n    @MainActor(unsafe) func testX() async {}\n}\n'),
    ('XCTSkip in an async helper', 'final class A: XCTestCase {\n    func skipIfNeeded() async throws {\n        throw XCTSkip("no gpu")\n    }\n}\n'),
    ('XCTestExpectation in an async helper', 'final class A: XCTestCase {\n    func make() async -> XCTestExpectation {\n        XCTestExpectation(description: "x")\n    }\n}\n'),
    ('MainActor extension of XCTestCase helper', '@MainActor\nextension XCTestCase {\n    func assertFoo() async {\n        XCTAssertTrue(true)\n    }\n}\n'),
    ('MainActor method in extension of XCTestCase', 'extension XCTestCase {\n    @MainActor\n    public func assertFoo() async {\n        XCTAssertTrue(true)\n    }\n}\n'),
    ('multi-line inheritance, MainActor', '@MainActor\nfinal class A: XCTestCase,\n    @unchecked Sendable {\n    func testX() async {}\n}\n'),
    ('class brace on next line, MainActor', '@MainActor\nfinal class A: XCTestCase\n{\n    func testX() async {}\n}\n'),
    ('generic MainActor base', '@MainActor class Base<T>: XCTestCase {}\nfinal class A: Base<Int> {\n    func testX() async {}\n}\n'),
    ('sync helper taking an async closure', 'final class A: XCTestCase {\n    func check(_ f: () async -> Int) {\n        XCTAssertTrue(true)\n    }\n}\n'),
    ('async helper with parenthesised parameter, no XCT', 'final class A: XCTestCase {\n    func make(_ n: Int = Int(3)) async -> Int { n }\n}\n'),
    ('helper mentioning XCT only in a string', 'final class A: XCTestCase {\n    func b() async { print("XCTAssert(") }\n}\n'),
]


class FixtureControls(unittest.TestCase):
    def test_must_flag_controls(self):
        for name, source, extra, _ in MUST_FLAG:
            with self.subTest(name):
                self.assertTrue(_flagged(source, *extra), name)

    def test_must_pass_controls(self):
        for name, source in MUST_PASS:
            with self.subTest(name):
                self.assertEqual(_flagged(source), [], name)

    def test_cross_file_main_actor_class_isolates_plain_extension(self):
        self.assertEqual(_flagged('extension A {\n    func testX() async {}\n}\n',
                                  '@MainActor final class A: XCTestCase {}\n'), [])

    def test_finding_rules_are_distinguished(self):
        found = _flagged('final class A: XCTestCase {\n    func testX() async {}\n'
                         '    func check() async { XCTFail("x") }\n}\n')
        self.assertEqual([f[4] for f in found], ['R1', 'R2'])

    def test_mutation_each_rule_bites(self):
        for name, source, extra, rule in MUST_FLAG:
            if rule is None:
                continue
            with self.subTest(f'{name} / {rule}'):
                self.assertTrue(_flagged(source, *extra), name)
                self.assertEqual(_flagged(source, *extra, disabled=frozenset({rule})), [],
                                 f'disabling {rule} must make "{name}" pass unflagged')

    def test_every_rule_has_a_biting_mutation_control(self):
        covered = {rule for _, _, _, rule in MUST_FLAG if rule}
        # method_attr/class_attr/extension_attr are exercised from the must-pass side below.
        self.assertEqual(covered, {'derivation_fixpoint', 'nonisolated', 'concurrent', 'effect_across_lines', 'helper_rule'})

    def test_mutation_isolation_rules_bite(self):
        for rule, source in (
            ('method_attr', 'final class A: XCTestCase {\n    @MainActor\n    func testX() async {}\n}\n'),
            ('class_attr', '@MainActor final class A: XCTestCase {\n    func testX() async {}\n}\n'),
            ('extension_attr', 'final class A: XCTestCase {}\n@MainActor extension A {\n    func testX() async {}\n}\n'),
        ):
            with self.subTest(rule):
                self.assertEqual(_flagged(source), [])
                self.assertTrue(_flagged(source, disabled=frozenset({rule})),
                                f'disabling {rule} must flag a currently passing control')

    def test_rules_constant_matches_mutation_switches(self):
        used = set(re.findall(r"'(\w+)' (?:not )?in disabled", Path(__file__).read_text()))
        self.assertEqual(used, set(RULES))


INLINE_ROOT_ORIGINALS = {
    # Shapes of the pre-conversion originals (657a100), reproduced inline so RED is proven without git history.
    'KernelAutoTuner-style': (
        'final class KernelAutoTunerTests: XCTestCase {\n'
        '    func testSmallBatchSelectionFP32() async throws {\n'
        '        let strategy = try await tuner.selectStrategy(for: workload)\n'
        '        XCTAssertEqual(strategy, .cpuAoS, "Small FP32 batches should use cpuAoS")\n'
        '    }\n\n'
        '    func testScoreCalculation() {\n        XCTAssertTrue(true)\n    }\n\n'
        '    func testZeroBatchSize() async throws {\n        XCTAssertTrue(true)\n    }\n}\n', {'R1': 2}),
    'ExecutionBoundary-style': (
        'final class ExecutionBoundaryTests: XCTestCase, @unchecked Sendable {\n'
        '    func testNegativeBatchRejectedBeforeDroppedWork() async throws {\n'
        '        do { _ = try await run(); XCTFail("must throw") } catch let e as VectorError { XCTAssertEqual(e.kind, .invalidConfiguration) }\n'
        '    }\n\n'
        '    func testHintValidationBeforeEmpty() throws {\n        try runChild("hints", expectsTrap: false)\n    }\n\n'
        '    private func checkHints() async throws {\n'
        '        for p in providers {\n            do { _ = try await invoke(p) ; XCTFail("x") }\n'
        '            catch let e as VectorError { XCTAssertEqual(e.kind, .invalidConfiguration) }\n        }\n    }\n\n'
        '    func testChildProbe() async throws {\n        try await checkHints()\n    }\n}\n', {'R1': 2, 'R2': 1}),
    'BenchmarkContractTests-sweep-style': (
        'final class BenchmarkCalibrationContractTests: XCTestCase {\n'
        '    private func assertSweepProperty(_ run: (Int, VirtualClock, Double) async throws -> BenchResult) async throws {\n'
        '        var topUps = 0\n        for step in 0..<80 {\n'
        '            let result = try await run(step, clock, 1.0)\n'
        '            XCTAssertEqual(result.sampleNsPerInvocation[0], 5.0, accuracy: 1.0, "step \\(step)")\n'
        '        }\n        XCTAssertGreaterThan(topUps, 20)\n    }\n\n'
        '    func testExtensionLoopNeverMeasuresASubTickInterval() async throws {\n'
        '        try await assertSweepProperty { step, clock, minTime in try await harness(step) }\n    }\n}\n', {'R1': 1, 'R2': 1}),
}


def _count_by_rule(findings):
    counts = {}
    for finding in findings:
        counts[finding[4]] = counts.get(finding[4], 0) + 1
    return counts


class OriginalInstances(unittest.TestCase):
    """RED on the pre-conversion shapes: inline (always runs) and from history (skipped when unavailable)."""

    def test_inline_pre_conversion_shapes_are_flagged(self):
        for name, (source, expected) in INLINE_ROOT_ORIGINALS.items():
            with self.subTest(name):
                self.assertEqual(_count_by_rule(_flagged(source)), expected)

    def test_inline_shapes_pass_once_main_actor_is_added(self):
        for name, (source, _) in INLINE_ROOT_ORIGINALS.items():
            with self.subTest(name):
                converted = re.sub(r'^(    )((?:private )?func (?:test\w*\(\)|assertSweepProperty|checkHints)[^\n]*async)',
                                   r'\1@MainActor \2', source, flags=re.M)
                self.assertNotEqual(converted, source)
                self.assertEqual(_flagged(converted), [], name)

    def test_negative_control_root_originals(self):
        files = [(p, _git_show(INTEGRATION_COMMIT, p)) for p in (
            'Tests/ComprehensiveTests/KernelAutoTunerTests.swift',
            'Tests/ComprehensiveTests/ExecutionBoundaryTests.swift')]
        findings, seen = analyze(files)
        by_file = {}
        for path, _, _, _, rule in findings:
            by_file.setdefault((path, rule), 0)
            by_file[(path, rule)] += 1
        self.assertEqual(by_file, {
            ('Tests/ComprehensiveTests/KernelAutoTunerTests.swift', 'R1'): 22,
            ('Tests/ComprehensiveTests/ExecutionBoundaryTests.swift', 'R1'): 2,
            ('Tests/ComprehensiveTests/ExecutionBoundaryTests.swift', 'R2'): 1,
        })
        self.assertEqual(len(seen), 24)

    def test_negative_control_benchmark_originals(self):
        files = [(p, _git_show(V04_BASE_TREE, p)) for p in (
            'Tests/BenchmarkContractTests/BenchmarkSelectionContractTests.swift',
            'Tests/BenchmarkContractTests/RunnerContractTests.swift')]
        findings, _ = analyze(files)
        self.assertEqual(_count_by_rule(findings), {'R1': 16, 'R2': 1})
        self.assertEqual([f[3] for f in findings if f[4] == 'R2'], ['assertSweepProperty'])

    def test_unavailable_revision_skips_instead_of_erroring(self):
        with self.assertRaises(unittest.SkipTest):
            _git_show('0000000000000000000000000000000000000000', 'Tests/none.swift')

    def test_originals_flag_with_main_actor_removed(self):
        # Mechanical closure: the live converted files with every `@MainActor` blanked are RED again.
        for path, count in CONVERTED_FILES.items():
            with self.subTest(path):
                stripped = (ROOT / path).read_text().replace('@MainActor', '')
                findings, _ = analyze([(path, stripped)])
                self.assertGreaterEqual(len([f for f in findings if f[4] == 'R1']), count)


class LiveTree(unittest.TestCase):
    def test_no_nonisolated_async_xctest_methods(self):
        findings, _ = analyze(_tests_files())
        self.assertEqual(findings, [],
                         'nonisolated async XCTest method or assertion helper (XCTest dropped failures there): '
                         'add @MainActor (or return values to a @MainActor test)')

    def test_guard_sees_the_converted_methods(self):
        # Lower bounds only (M4): new async tests are allowed; test_no_nonisolated_async_xctest_methods
        # asserts that every async test the guard sees is isolated.
        _, seen = analyze(_tests_files())
        counts = {}
        for path, _, _, _ in seen:
            counts[path] = counts.get(path, 0) + 1
        for path, count in CONVERTED_FILES.items():
            self.assertGreaterEqual(counts.get(path, 0), count, path)
        benchmark = sum(c for p, c in counts.items() if p.startswith('Tests/BenchmarkContractTests/'))
        self.assertGreaterEqual(benchmark, 16)

    def test_extension_helpers_are_checked_on_the_live_tree(self):
        # TestHelpers/MockErrorTestExtensions.swift declares an async XCTestCase-extension helper with
        # assertions; it must be seen (and isolated). Removing its @MainActor must flag it.
        path = 'Tests/ComprehensiveTests/TestHelpers/MockErrorTestExtensions.swift'
        source = (ROOT / path).read_text()
        self.assertEqual(analyze([(path, source)])[0], [])
        self.assertEqual([f[3:] for f in analyze([(path, source.replace('@MainActor', ''))])[0]],
                         [('assertThrowsMockError', 'R2')])


if __name__ == '__main__':
    unittest.main()
