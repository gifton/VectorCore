"""Policy (T15, cosine class closure): every cosine implementation is registered in the T2 exceptional table.

WHY: cosine distance clamped a NaN similarity to distance 0 (`min(1, NaN) == 1` in Swift), so NaN and infinite rows
ranked nearest; 20 of 40 cosine bodies had the defect, several with their own copy of the clamp. T15 applies one
exceptional-value policy to every body and pins it with a registry-driven table
(Tests/ComprehensiveTests/T15CosineExceptionalTableTests.swift). This guard keeps the registry complete for the bodies
its lexical discovery finds (below): such a body must be registered, so it is evaluated against the policy, and a
removed one must be unregistered. It does not evaluate the policy itself; the T2 table does.

DISCOVERY, on Sources/**/*.swift and Benchmarks/**/*.swift with comments and strings blanked by
Scripts/ci/swift_lexing._mask, one site per function or computed-property body (swift_lexing._Scopes.bodies):
  M1 the body's name contains "cosine" (any case);
  M2 the body contains a cosine clamp idiom: `max(-1, min(1,`, `min(1, max(-1,`, `max(0, min(2,` (integer or `.0`
     literals, optionally `Swift.`-qualified) or `simd_clamp(`; the site is the innermost named body around it;
  M3 the body calls `calculateCosineDistance(` (other than its own declaration);
  M4 the body is declared in a type or extension whose name contains "Cosine" (nearest enclosing type scope).
A site is `path::name`; overloads and nested functions with one name count once per body.

RULES:
  R1 the multiset of discovered sites equals the multiset of string literals `"path::name"` between the
     `// BEGIN T15 COSINE SITES` and `// END T15 COSINE SITES` markers of Tests/ComprehensiveTests/T15CosineRegistry.swift
     (unregistered and stale sites are both failures); `//` and `/* */` comments inside that block are removed first
     (strings kept), so a commented-out entry is not a registration;
  R2 every body that contains a clamp idiom (M2) also contains an `isFinite` or `isNaN` token somewhere in the same
     body. This is a body-level TOKEN check: it does not show that the token guards the clamp (a body whose zero
     branch keeps an `isFinite` token passes with its clamp unguarded). The semantic check is the T2 table,
     T15CosineExceptionalTableTests.everyImplementationFollowsThePolicy, which evaluates every registered body on NaN
     and infinite fixtures;
  R3 every `.exempt("...")` entry carries a non-empty reason, and every `.external("path::Suite.test")` entry names an
     existing file that declares `func test`.
The real-tree tests and the fixture tests always run, in a git-less export too (filesystem reads only).

LIMITS (lexical heuristic, not a parser): a cosine computation in a body that has no "cosine" in its name, no clamp
idiom, no helper call and no Cosine-named enclosing type is not discovered (for example a route that calls a
registered kernel, such as `MixedPrecisionProvider.distance` or `VectorProtocol.distance(to:metric:)`; those are rows
of the T2 table, not sites). A clamp written another way (comparisons such as `d < 0 ? 0 : d > 2 ? 2 : d`, or
`clamp(...)` helpers) is not an M2 idiom. R2 is satisfied by any `isFinite`/`isNaN` token in the same body, wherever
it is; whether the guard is correct is the T2 table's job. The registry block's comment removal does not nest
`/* */` comments (Swift does) and does not lex string interpolation. `#if` branches are not evaluated; C declarations
(VectorCoreC.h) and Tests are not scanned. Type scopes are found with swift_lexing's TYPE_HEADER; a type whose
declaration spans several lines before its `{` is still found because the header chunk is the statement before `{`.
"""
import collections
import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location('swift_lexing', ROOT / 'Scripts/ci/swift_lexing.py')
lexing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lexing)

REGISTRY = 'Tests/ComprehensiveTests/T15CosineRegistry.swift'
SCANNED = ('Sources', 'Benchmarks')
BEGIN, END = '// BEGIN T15 COSINE SITES', '// END T15 COSINE SITES'

_ONE = r'1(?:\.0)?'
_CLAMPS = (
    re.compile(r'\bmax\(\s*-\s*' + _ONE + r'\s*,\s*(?:Swift\.)?min\(\s*' + _ONE + r'\s*,'),
    re.compile(r'\bmin\(\s*' + _ONE + r'\s*,\s*(?:Swift\.)?max\(\s*-\s*' + _ONE + r'\s*,'),
    re.compile(r'\bmax\(\s*0(?:\.0)?\s*,\s*(?:Swift\.)?min\(\s*2(?:\.0)?\s*,'),
    re.compile(r'\bsimd_clamp\('),
)
_HELPER_CALL = re.compile(r'\bcalculateCosineDistance\s*\(')
_HELPER_DECL = re.compile(r'\bfunc\s+calculateCosineDistance\s*\(')
_TYPE_NAME = re.compile(r'\b(?:class|struct|enum|extension|actor|protocol)\s+([A-Za-z_][\w.]*)')
_GUARD = re.compile(r'\.isFinite\b|\.isNaN\b')
_SITE_LITERAL = re.compile(r'"((?:Sources|Benchmarks)/[^"\s]+\.swift::\w+)"')
_EXEMPT = re.compile(r'\.exempt\(\s*"([^"]*)"\s*\)')
_EXTERNAL = re.compile(r'\.external\(\s*"([^"]*)"\s*\)')


def _type_names(source, mask, scopes):
    """Name of each transparent (type) scope, or None."""
    names = [None] * len(scopes.parent)
    for scope in range(1, len(scopes.parent)):
        if not scopes.transparent[scope]:
            continue
        open_index = scopes.opens[scope]
        previous = max(mask.rfind(ch, 0, open_index) for ch in '{};')
        chunk = lexing._header_chunk(mask[previous + 1:open_index])
        found = _TYPE_NAME.search(chunk)
        names[scope] = found.group(1) if found else None
    return names


def _named_owner(scopes, offset):
    scope = scopes.at[offset]
    while scope and not scopes.name[scope]:
        scope = scopes.parent[scope]
    return scope or None


def discover(path, source):
    """(sites, unguarded) for one file: sites is a list of `path::name`, one per discovered body; unguarded lists
    `path::name@line` of bodies with a clamp idiom and no isFinite/isNaN token (R2)."""
    mask = lexing._mask(source)
    scopes = lexing._Scopes(source)
    types = _type_names(source, mask, scopes)
    hits = {}

    def mark(scope, rule):
        hits.setdefault(scope, set()).add(rule)

    for scope, name in enumerate(scopes.name):
        if not name:
            continue
        if 'cosine' in name.lower():
            mark(scope, 'M1')
        walk = scopes.parent[scope]
        while walk is not None and walk != 0 and not scopes.transparent[walk]:
            walk = scopes.parent[walk]
        if walk and types[walk] and 'Cosine' in types[walk]:
            mark(scope, 'M4')
    clamp_owners = set()
    for pattern in _CLAMPS:
        for match in pattern.finditer(mask):
            owner = _named_owner(scopes, match.start())
            if owner is None:
                hits.setdefault(('file', source.count('\n', 0, match.start()) + 1), set()).add('M2')
                continue
            mark(owner, 'M2')
            clamp_owners.add(owner)
    for match in _HELPER_CALL.finditer(mask):
        line_start = mask.rfind('\n', 0, match.start()) + 1
        if _HELPER_DECL.search(mask[line_start:match.end()]):
            continue
        owner = _named_owner(scopes, match.start())
        if owner is not None:
            mark(owner, 'M3')
    sites, unguarded = [], []
    for scope in hits:
        if isinstance(scope, tuple):
            sites.append(f'{path}::<file-level clamp at line {scope[1]}>')
        else:
            sites.append(f'{path}::{scopes.name[scope]}')
    for owner in clamp_owners:
        body = mask[scopes.opens[owner] + 1:scopes.closes[owner]]
        if not _GUARD.search(body):
            unguarded.append(f'{path}::{scopes.name[owner]}@{source.count(chr(10), 0, scopes.opens[owner]) + 1}')
    return sites, unguarded


def discover_tree(files):
    """files: {relative path: source}. Returns (Counter of sites, sorted unguarded list)."""
    sites, unguarded = collections.Counter(), []
    for path in sorted(files):
        found, bad = discover(path, files[path])
        sites.update(found)
        unguarded.extend(bad)
    return sites, sorted(unguarded)


def strip_comments(text):
    """Same-length copy of `text` with `//` and `/* */` comments blanked (newlines kept); string literals are kept,
    so a `//` inside a string is not a comment. Block comments do not nest; interpolation is not lexed."""
    out, n, i = list(text), len(text), 0

    def blank(a, b):
        for k in range(a, min(b, n)):
            if out[k] != '\n':
                out[k] = ' '

    while i < n:
        if text.startswith('"""', i):
            j = text.find('"""', i + 3)
            i = n if j < 0 else j + 3
        elif text[i] == '"':
            k = i + 1
            while k < n and text[k] not in '"\n':
                k += 2 if text[k] == '\\' else 1
            i = k + 1
        elif text.startswith('//', i):
            j = text.find('\n', i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2)
            j = n if j < 0 else j + 2
            blank(i, j)
            i = j
        else:
            i += 1
    return ''.join(out)


def registry_block(text):
    """The text between the markers with its comments removed (R1, R3 read only live entries)."""
    start, end = text.find(BEGIN), text.find(END)
    if start < 0 or end < start:
        raise ValueError('registry markers missing or out of order')
    return strip_comments(text[start:end])


def registered(text):
    return collections.Counter(_SITE_LITERAL.findall(registry_block(text)))


def registry_problems(text, exists):
    """R3 problems. `exists(path, test)` reports whether `path` exists and declares `func test`."""
    block = registry_block(text)
    problems = [f'exemption without a reason: {m.start()}' for m in _EXEMPT.finditer(block) if not m.group(1).strip()]
    for match in _EXTERNAL.finditer(block):
        reference = match.group(1)
        if '::' not in reference:
            problems.append(f'external reference without ::Suite.test: {reference}')
            continue
        path, symbol = reference.split('::', 1)
        if not exists(path, symbol.rsplit('.', 1)[-1]):
            problems.append(f'external reference does not resolve: {reference}')
    return problems


def check(files, registry_text, exists):
    """All rule violations as strings (empty when the tree and registry agree)."""
    sites, unguarded = discover_tree(files)
    listed = registered(registry_text)
    problems = [f'R1 unregistered cosine site: {site} (x{count})' for site, count in sorted((sites - listed).items())]
    problems += [f'R1 stale registry entry: {site} (x{count})' for site, count in sorted((listed - sites).items())]
    problems += [f'R2 clamp without an isFinite/isNaN check in the same body: {site}' for site in unguarded]
    problems += ['R3 ' + p for p in registry_problems(registry_text, exists)]
    return problems


def real_files():
    files = {}
    for base in SCANNED:
        for path in sorted((ROOT / base).rglob('*.swift')):
            files[path.relative_to(ROOT).as_posix()] = path.read_text(encoding='utf-8')
    return files


def real_exists(path, test):
    target = ROOT / path
    return target.is_file() and re.search(r'\bfunc\s+' + re.escape(test) + r'\s*\(', target.read_text(encoding='utf-8')) is not None


def registry_text():
    return (ROOT / REGISTRY).read_text(encoding='utf-8')


def _registry(*sites, extra=''):
    entries = ''.join(f'        ("{site}", .table(["x"])),\n' for site in sites)
    return f'enum R {{\n    static let sites = [\n        {BEGIN}\n{entries}{extra}        {END}\n    ]\n}}\n'


GUARDED = ('enum CosineThing {\n'
           '    static func helper(_ s: Float) -> Float {\n'
           '        guard s.isFinite else { return .nan }\n'
           '        return 1 - max(-1, min(1, s))\n'
           '    }\n'
           '}\n')


class FixtureControls(unittest.TestCase):
    """The guard bites: each must-fail fixture is flagged; registered, guarded sites pass."""

    def never(self, path, test):
        return False

    def test_registered_guarded_site_passes(self):
        files = {'Sources/A.swift': GUARDED}
        self.assertEqual(check(files, _registry('Sources/A.swift::helper'), self.never), [])

    def test_unregistered_cosine_function_is_flagged(self):
        files = {'Sources/A.swift': 'func cosineFoo(_ a: [Float]) -> Float { a.reduce(0, +) }\n'}
        problems = check(files, _registry(), self.never)
        self.assertIn('R1 unregistered cosine site: Sources/A.swift::cosineFoo (x1)', problems)

    def test_stale_entry_is_flagged(self):
        files = {'Sources/A.swift': GUARDED}
        problems = check(files, _registry('Sources/A.swift::helper', 'Sources/A.swift::removedKernel'), self.never)
        self.assertEqual(problems, ['R1 stale registry entry: Sources/A.swift::removedKernel (x1)'])

    def test_unregistered_clamp_site_is_flagged(self):
        source = 'func score(_ s: Float) -> Float {\n    guard s.isFinite else { return .nan }\n    return 1.0 - max(-1.0, min(1.0, s))\n}\n'
        problems = check({'Benchmarks/B.swift': source}, _registry(), self.never)
        self.assertEqual(problems, ['R1 unregistered cosine site: Benchmarks/B.swift::score (x1)'])

    def test_every_clamp_idiom_is_discovered(self):
        for idiom in ('1 - max(-1, min(1, s))', '1 - min(1.0, Swift.max(-1.0, s))', 'max(0.0, min(2.0, 1 - s))',
                      'simd_clamp(d, .zero, SIMD4<Float>(repeating: 2))'):
            source = 'func f(_ s: Float, _ d: SIMD4<Float>) -> Any {\n    return %s\n}\n' % idiom
            sites, unguarded = discover_tree({'Sources/C.swift': source})
            self.assertEqual(sites, collections.Counter({'Sources/C.swift::f': 1}), idiom)
            self.assertEqual(len(unguarded), 1, idiom)

    def test_restored_nan_clamp_is_flagged(self):
        """Mutation: dropping the finite guard around a registered clamp fails R2."""
        unguarded = GUARDED.replace('        guard s.isFinite else { return .nan }\n', '')
        problems = check({'Sources/A.swift': unguarded}, _registry('Sources/A.swift::helper'), self.never)
        self.assertEqual(len(problems), 1)
        self.assertTrue(problems[0].startswith('R2 '), problems)

    def test_helper_caller_is_flagged(self):
        source = 'func route(_ d: Float) -> Float { CosineKernels.calculateCosineDistance(dot: d, sumAA: 1, sumBB: 1) }\n'
        problems = check({'Sources/D.swift': source}, _registry(), self.never)
        self.assertEqual(problems, ['R1 unregistered cosine site: Sources/D.swift::route (x1)'])

    def test_body_in_a_cosine_type_is_flagged(self):
        source = 'extension CosineDistance {\n    func distance(_ a: V2048, _ b: V2048) -> Float { kernel(a, b) }\n}\n'
        problems = check({'Sources/E.swift': source}, _registry(), self.never)
        self.assertEqual(problems, ['R1 unregistered cosine site: Sources/E.swift::distance (x1)'])

    def test_overloads_count_once_per_body(self):
        source = 'enum K {\n    static func cosine512(_ a: A) -> Float { 0 }\n    static func cosine512(_ a: B) -> Float { 0 }\n}\n'
        once = check({'Sources/F.swift': source}, _registry('Sources/F.swift::cosine512'), self.never)
        self.assertEqual(once, ['R1 unregistered cosine site: Sources/F.swift::cosine512 (x1)'])
        twice = check({'Sources/F.swift': source},
                      _registry('Sources/F.swift::cosine512', 'Sources/F.swift::cosine512'), self.never)
        self.assertEqual(twice, [])

    def test_comments_and_strings_are_not_sites(self):
        source = ('// func cosineOld() { 1 - max(-1, min(1, s)) }\n'
                  'let note = "1 - max(-1, min(1, s)) calculateCosineDistance("\n'
                  'func euclid(_ a: Float) -> Float { a }\n')
        self.assertEqual(check({'Sources/G.swift': source}, _registry(), self.never), [])

    def test_protocol_requirement_without_body_is_not_a_site(self):
        source = 'protocol P {\n    func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float\n}\n'
        self.assertEqual(check({'Sources/H.swift': source}, _registry(), self.never), [])

    def test_exemption_needs_a_reason_and_external_must_resolve(self):
        extra = ('        ("Sources/A.swift::helper", .exempt("")),\n'
                 '        ("Sources/A.swift::helper", .external("Tests/X.swift::S.missing")),\n')
        text = _registry(extra=extra)
        problems = check({'Sources/A.swift': GUARDED + GUARDED.replace('CosineThing', 'CosineOther')}, text, self.never)
        self.assertTrue(any(p.startswith('R3 exemption without a reason') for p in problems), problems)
        self.assertIn('R3 external reference does not resolve: Tests/X.swift::S.missing', problems)

    def test_commented_registry_entry_is_not_a_registration(self):
        """Phase 2 review bypass A: an entry inside a `//` or `/* */` comment does not register the site."""
        for commented in ('        // ("Sources/A.swift::helper", .table(["x"])),\n',
                          '        /* ("Sources/A.swift::helper", .table(["x"])), */\n',
                          '        /*\n        ("Sources/A.swift::helper", .table(["x"])),\n        */\n'):
            problems = check({'Sources/A.swift': GUARDED}, _registry(extra=commented), self.never)
            self.assertEqual(problems, ['R1 unregistered cosine site: Sources/A.swift::helper (x1)'], commented)

    def test_comment_markers_inside_strings_are_kept(self):
        """Control: `//` inside a registry string literal is not a comment, so the entry still registers."""
        extra = '        ("Sources/A.swift::helper", .exempt("see the // note and /* this */ too")),\n'
        self.assertEqual(check({'Sources/A.swift': GUARDED}, _registry(extra=extra), self.never), [])

    def test_register_skip_of_a_real_site_is_flagged(self):
        """Mutation: removing one real registry entry (here the first) is reported as unregistered."""
        text = registry_text()
        block = registry_block(text)
        first = _SITE_LITERAL.search(block)
        line_start = text.rfind('\n', 0, text.find(first.group(0))) + 1
        line_end = text.find('\n', line_start) + 1
        mutated = text[:line_start] + text[line_end:]
        problems = check(real_files(), mutated, real_exists)
        self.assertIn(f'R1 unregistered cosine site: {first.group(1)} (x1)', problems)


class RealTree(unittest.TestCase):
    maxDiff = None

    def test_registry_matches_every_discovered_cosine_site(self):
        self.assertEqual(check(real_files(), registry_text(), real_exists), [])

    def test_discovery_sees_the_known_implementations(self):
        """Sanity: discovery is not vacuous (the T15 phase 1 inventory's anchor bodies are found)."""
        sites, _ = discover_tree(real_files())
        for anchor in ('Sources/VectorCore/Operations/Kernels/CosineKernels.swift::calculateCosineDistance',
                       'Sources/VectorCore/Operations/DistanceMetrics.swift::distance',
                       'Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift::computeDistance',
                       'Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift::cosineFP16Core',
                       'Benchmarks/VectorCoreBench/Utils/Correctness.swift::cosineDist'):
            self.assertIn(anchor, sites)
        self.assertGreater(sum(sites.values()), 80)


if __name__ == '__main__':
    unittest.main()
