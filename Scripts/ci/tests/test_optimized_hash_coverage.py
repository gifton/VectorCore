"""Policy (P01, OWN-07): each optimized vector's hash(into:) feeds every SIMD storage block, then the dimension.

WHY: Vector384/512/768/1536Optimized hashed only SIMD blocks 0, 1, L-2 and L-1 plus the dimension, so vectors
differing only in lanes [8, D-8) always collided (OWN-07). Equality scans every block. The ruled form (P01 root
ruling Q13) is

    for <block> in _storage { hasher.combine(<block>) }
    hasher.combine(<D>)

which hashes each SIMD4<Float> lane with Float's Hashable semantics (so +0 and -0 hash equally, as Float
equality requires). This check pins COORDINATE COVERAGE only, not the mathematical quality of the hash.

RULES, on the `extension Vector<D>Optimized: Hashable` scope of each file, with comments and strings blanked by
Scripts/ci/swift_lexing._mask:
  H1 exactly one `func hash(into hasher: inout Hasher)` in the scope;
  H2 its body contains `for X in _storage {` whose body is exactly `hasher.combine(X)`, and that loop is at
     brace depth 0 of the hash body (not under an `if`, `switch`, closure or other block that could skip it);
  H3 `hasher.combine(<D>)` with the file's literal dimension comes after that loop, also at depth 0;
  H4 no `_storage[` subscript (the sampled form indexes four literal blocks);
  H5 no `bitPattern` (raw-bit hashing would break +0/-0 equality) and no other `hasher.combine(` call;
  H6 no `let`/`var` declaration named `_storage` in the body (a shadow would make the loop iterate something else);
  H7 no `return`, `throw`, `fatalError` or `preconditionFailure` anywhere before the block loop (an early exit,
     for example `guard cond else { return }`, would skip it even though the loop sits at depth 0).

The real-source test and the fixture and mutation tests always run, in a git-less export too. The fixtures prove
that the check bites: the base body (the omission) is flagged, the ruled loop passes, an `if`-wrapped loop and a
shadowed `_storage` are flagged, and restoring the sampled body into each real file is flagged.

LIMITS (lexical heuristic, not a parser): a loop over a renamed alias of `_storage`, a helper function that
hashes the blocks, or a `while` loop is rejected even if equivalent; `#if` branches are not evaluated.
Known gaps of the shadow rule (H6), none present in the real files: it matches only `let|var _storage`, so these
forms still pass: tuple destructuring (`let (_storage, other) = ...`), a backticked name (`` let `_storage` ``),
a closure parameter named `_storage` (`{ (_storage: [Int]) in ... }`), and `for _storage in ...`.
FixtureControls.test_documented_shadow_rule_gaps_still_pass pins that each of these forms passes and that
`case let _storage` is caught (it contains `let _storage`); strengthening H6 must update that test and this text.
H7 is lexical too: an early exit spelled any other way (for example a call to a helper that never returns, or
`exit(0)`) is not recognised.
"""
import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location('swift_lexing', ROOT / 'Scripts/ci/swift_lexing.py')
lexing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lexing)

WIDTHS = (384, 512, 768, 1536)


def source_path(width):
    return ROOT / f'Sources/VectorCore/Vectors/Vector{width}Optimized.swift'


def _matching_brace(masked, open_index):
    depth = 0
    for index in range(open_index, len(masked)):
        if masked[index] == '{':
            depth += 1
        elif masked[index] == '}':
            depth -= 1
            if depth == 0:
                return index
    raise ValueError('unbalanced braces')


def _hashable_scope(masked, width):
    match = re.search(rf'\bextension\s+Vector{width}Optimized\s*:\s*Hashable\s*\{{', masked)
    if not match:
        return None
    start = match.end() - 1
    return start, _matching_brace(masked, start)


def hash_body(source, width):
    """(masked body text) of hash(into:) in the Hashable extension, or None with a reason."""
    masked = lexing._mask(source)
    scope = _hashable_scope(masked, width)
    if scope is None:
        return None, 'H1 no `extension Vector%dOptimized: Hashable` scope' % width
    scope_text = masked[scope[0]:scope[1] + 1]
    functions = list(re.finditer(r'\bfunc\s+hash\s*\(\s*into\s+hasher\s*:\s*inout\s+Hasher\s*\)\s*\{', scope_text))
    if len(functions) != 1:
        return None, 'H1 expected exactly one hash(into:) in the Hashable scope, found %d' % len(functions)
    open_index = functions[0].end() - 1
    close_index = _matching_brace(scope_text, open_index)
    return scope_text[open_index + 1:close_index], None


def _depth_at(text, index):
    """Brace depth of `index` within `text` (0 means directly in the function body)."""
    return text.count('{', 0, index) - text.count('}', 0, index)


def violations(source, width):
    """List of rule violations for one file's text (empty when the ruled full-coverage form is present)."""
    body, reason = hash_body(source, width)
    if body is None:
        return [reason]
    found = []
    loop = re.search(r'\bfor\s+(\w+)\s+in\s+_storage\s*\{', body)
    if not loop:
        found.append('H2 no `for X in _storage {` loop')
    else:
        if _depth_at(body, loop.start()) != 0:
            found.append('H2 the block loop is nested inside another block (brace depth %d), not at depth 0'
                         % _depth_at(body, loop.start()))
        loop_close = _matching_brace(body, loop.end() - 1)
        loop_body = ' '.join(body[loop.end():loop_close].split())
        if loop_body != f'hasher.combine({loop.group(1)})':
            found.append('H2 loop body is %r, expected `hasher.combine(%s)`' % (loop_body, loop.group(1)))
        after = body[loop_close + 1:]
        dimension = re.search(rf'\bhasher\.combine\(\s*{width}\s*\)', after)
        if not dimension:
            found.append(f'H3 no `hasher.combine({width})` after the block loop')
        elif _depth_at(after, dimension.start()) != 0:
            found.append(f'H3 `hasher.combine({width})` is nested inside another block, not at depth 0')
    if '_storage[' in body.replace(' ', ''):
        found.append('H4 `_storage[` subscript (sampled block) in hash(into:)')
    if 'bitPattern' in body:
        found.append('H5 raw `bitPattern` hashing in hash(into:)')
    if re.search(r'\b(?:let|var)\s+_storage\b', body):
        found.append('H6 `let`/`var` declaration named `_storage` shadows the stored blocks in hash(into:)')
    if loop:
        early = re.search(r'\b(return|throw|fatalError|preconditionFailure)\b', body[:loop.start()])
        if early:
            found.append('H7 early exit `%s` before the block loop in hash(into:)' % early.group(1))
    combines = len(re.findall(r'\bhasher\.combine\(', body))
    if combines != 2:
        found.append('H5 expected exactly two hasher.combine calls (block loop, dimension), found %d' % combines)
    return found


def sampled_body(width):
    """The pre-P01 body: the omission this policy exists to catch (also the mutation that restores it)."""
    last = width // 4 - 1
    return ('\n        #if DEBUG\n        assert(_storage.count >= %d, "Invalid _storage size for hashing")\n        #endif\n\n'
            '        // Hash first and last few SIMD4 vectors for efficiency\n'
            '        hasher.combine(_storage[0])\n        hasher.combine(_storage[1])\n'
            '        hasher.combine(_storage[%d])\n        hasher.combine(_storage[%d])\n'
            '        hasher.combine(%d) // Include dimension\n    ' % (width // 4, last - 1, last, width))


def ruled_body(width):
    return '\n        for block in _storage {\n            hasher.combine(block)\n        }\n        hasher.combine(%d)\n    ' % width


def fixture(width, body):
    return ('extension Vector%dOptimized: Equatable {\n    public static func == (lhs: Vector%dOptimized, rhs: Vector%dOptimized) -> Bool { true }\n}\n\n'
            'extension Vector%dOptimized: Hashable {\n    public func hash(into hasher: inout Hasher) {%s}\n}\n'
            % (width, width, width, width, body))


def replace_body(source, width, body):
    """Replace the real hash(into:) body (between its braces) with `body`."""
    masked = lexing._mask(source)
    scope = _hashable_scope(masked, width)
    if scope is None:
        raise ValueError('no Hashable scope')
    function = re.compile(r'\bfunc\s+hash\s*\(\s*into\s+hasher\s*:\s*inout\s+Hasher\s*\)\s*\{').search(masked, scope[0], scope[1])
    open_index = function.end() - 1
    close_index = _matching_brace(masked, open_index)
    return source[:open_index + 1] + body + source[close_index:]


def restore_sampled(source, width):
    """Mutation: restore the sampled body (the omission) into a real file's text."""
    return replace_body(source, width, sampled_body(width))


class FixtureControls(unittest.TestCase):
    def test_sampled_body_is_flagged(self):
        for width in WIDTHS:
            found = violations(fixture(width, sampled_body(width)), width)
            self.assertTrue(any(v.startswith('H2') for v in found), found)
            self.assertTrue(any(v.startswith('H4') for v in found), found)

    def test_ruled_loop_passes(self):
        for width in WIDTHS:
            self.assertEqual(violations(fixture(width, ruled_body(width)), width), [])

    def test_loop_plus_sampled_block_is_flagged(self):
        body = ruled_body(512).replace('hasher.combine(512)', 'hasher.combine(_storage[127])\n        hasher.combine(512)')
        found = violations(fixture(512, body), 512)
        self.assertTrue(any(v.startswith('H4') for v in found), found)

    def test_missing_or_wrong_dimension_is_flagged(self):
        self.assertTrue(any(v.startswith('H3') for v in violations(fixture(768, ruled_body(768).replace('combine(768)', '')), 768)))
        self.assertTrue(any(v.startswith('H3') for v in violations(fixture(768, ruled_body(768).replace('768', '512')), 768)))

    def test_raw_bits_and_partial_loops_are_flagged(self):
        raw = ruled_body(384).replace('hasher.combine(block)', 'hasher.combine(block.x.bitPattern)')
        self.assertTrue(any(v.startswith('H5') or v.startswith('H2') for v in violations(fixture(384, raw), 384)))
        partial = ruled_body(384).replace('for block in _storage', 'for block in _storage.prefix(4)')
        self.assertTrue(any(v.startswith('H2') for v in violations(fixture(384, partial), 384)))

    def test_loop_under_a_condition_is_flagged(self):
        wrapped = ('\n        if _storage.count > 0 {\n            for block in _storage {\n'
                   '                hasher.combine(block)\n            }\n        }\n        hasher.combine(%d)\n    ')
        for width in WIDTHS:
            found = violations(fixture(width, wrapped % width), width)
            self.assertTrue(any(v.startswith('H2') and 'depth' in v for v in found), (width, found))
        always_false = '\n        if false {\n            for block in _storage { hasher.combine(block) }\n        }\n        hasher.combine(512)\n    '
        self.assertTrue(any('depth' in v for v in violations(fixture(512, always_false), 512)))

    def test_conditional_dimension_is_flagged(self):
        body = ruled_body(768).replace('hasher.combine(768)', 'if true { hasher.combine(768) }')
        self.assertTrue(any(v.startswith('H3') for v in violations(fixture(768, body), 768)))

    def test_shadowed_storage_is_flagged(self):
        for decl in ('let _storage = ContiguousArray<SIMD4<Float>>()', 'var _storage = ContiguousArray<SIMD4<Float>>()'):
            body = '\n        %s\n' % decl + ruled_body(384).lstrip('\n')
            found = violations(fixture(384, body), 384)
            self.assertTrue(any(v.startswith('H6') for v in found), (decl, found))

    def test_documented_shadow_rule_gaps_still_pass(self):
        """LIMITS pin: the H6 gaps named in the module docstring pass; `case let _storage` is caught."""
        gaps = ('let (_storage, other) = (ContiguousArray<SIMD4<Float>>(), 0)',
                'let `_storage` = ContiguousArray<SIMD4<Float>>()',
                'let count = { (_storage: [Int]) in _storage.count }([])',
                'for _storage in [1, 2] { _ = _storage }')
        for gap in gaps:
            body = '\n        %s\n' % gap + ruled_body(512).lstrip('\n')
            self.assertEqual(violations(fixture(512, body), 512), [], gap)
        caught = '\n        if case let _storage = 1 { _ = _storage }\n' + ruled_body(512).lstrip('\n')
        self.assertTrue(any(v.startswith('H6') for v in violations(fixture(512, caught), 512)))

    def test_early_exit_before_the_loop_is_flagged(self):
        exits = ('guard _storage.count > 1000 else { return }', 'return', 'if true { return }',
                 'fatalError("never")', 'preconditionFailure("never")', 'if true { throw NSError() }')
        for width in WIDTHS:
            for early in exits:
                body = '\n        %s\n' % early + ruled_body(width).lstrip('\n')
                found = violations(fixture(width, body), width)
                self.assertTrue(any(v.startswith('H7') for v in found), (width, early, found))

    def test_exit_words_in_comments_or_after_the_loop_do_not_trip_h7(self):
        commented = '\n        // return early? no\n' + ruled_body(512).lstrip('\n')
        self.assertEqual(violations(fixture(512, commented), 512), [])

    def test_commented_loop_does_not_count(self):
        body = '\n        // for block in _storage { hasher.combine(block) }\n' + sampled_body(1536)
        self.assertTrue(any(v.startswith('H2') for v in violations(fixture(1536, body), 1536)))

    def test_restoring_the_omission_in_each_real_file_is_flagged(self):
        """Mutation: whatever the real body is today, restoring the sampled body must be caught."""
        for width in WIDTHS:
            mutated = restore_sampled(source_path(width).read_text(encoding='utf-8'), width)
            found = violations(mutated, width)
            self.assertTrue(any(v.startswith('H4') for v in found), (width, found))

    def test_ruled_body_in_each_real_file_passes(self):
        """The ruled loop placed in each real file's context passes: the check is satisfiable after phase B."""
        for width in WIDTHS:
            ruled = replace_body(source_path(width).read_text(encoding='utf-8'), width, ruled_body(width))
            self.assertEqual(violations(ruled, width), [], width)


class RealSources(unittest.TestCase):
    maxDiff = None

    def test_every_optimized_hash_feeds_every_block_and_the_dimension(self):
        problems = {width: violations(source_path(width).read_text(encoding='utf-8'), width) for width in WIDTHS}
        self.assertEqual({w: p for w, p in problems.items() if p}, {},
                         'optimized hash(into:) bodies that do not feed every SIMD block (OWN-07)')


if __name__ == '__main__':
    unittest.main()
