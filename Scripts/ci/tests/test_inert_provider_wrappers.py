"""Policy (P01 final fix I2; Q17 class): no inert `Operations.$computeProvider` install around BatchOperations-only code.

WHY: `Operations.computeProvider` is a task-local read only by `Operations`; `BatchOperations` does not read it
(premise P1-P3 below, pinned on the real sources). So a test that installs a provider (for example
`CPUComputeProvider(mode: .parallel)`) and then calls only `BatchOperations` APIs runs on whatever route
BatchOperations' own configuration picks: the install is inert, and the test states a mechanism that does not
exist (P01 phase F Q17B showed the vacuity; root's Q17 ruling removed two such wrappers, and the final fix round
removed the last three). This check closes that class mechanically.

PREMISE (the reason an install is inert), pinned on the real sources with comments and strings blanked by
Scripts/ci/swift_lexing._mask:
  P1 Sources/VectorCore/Operations/BatchOperations.swift contains no `computeProvider` token;
  P2 it contains no `Operations.` member reference other than through `BatchOperations` (so it cannot reach the
     provider through an Operations API);
  P3 among Sources/**/*.swift, only Sources/VectorCore/Operations/Operations.swift contains `computeProvider`.
If any of these fails, an install may no longer be inert around BatchOperations: re-derive the rule before
changing this file.

RULE W1, on every Tests/**/*.swift file (masked the same way): an install
`Operations.$computeProvider.withValue(<provider>, ...)`, optionally module-qualified and with any whitespace
around the dots, has a SCOPE made of its trailing closure and of every closure passed at the top level of its
argument list after the provider argument (the `operation:` form). The install is flagged when its scope contains
a `BatchOperations.` reference and no other `Operations.` reference (an `Operations.$computeProvider` install
inside the scope does not count as a use). A scope that also names an Operations API passes, because such a call
can read the provider (see L3).

The real-source tests and the fixture and mutation tests always run, in a git-less export too. The fixtures prove
that the check bites: the three removed sites (verbatim) are flagged, a legitimate Operations call inside the scope
passes, and re-inserting a wrapper into each fixed real file is flagged.

INTERPOLATION (T17): every scan uses `_mask(..., keep_interpolations=True)`, so a call written inside a string
interpolation is code (it was limit L5, now closed: FixtureControls.test_interpolated_code_is_seen flags a
BatchOperations-only scope whose call sits in an interpolation, and passes one whose Operations read sits in an
interpolation); string text outside `\\(...)` is still not code.

LIMITS (lexical heuristic, not a parser; FixtureControls.test_documented_limits_still_pass pins a code form for
each of L1-L4 and the `#if` part of L5, L6 is a premise limit with no fixture, and strengthening the rule must update that test and this
text):
  L1 an install through an alias (`let install = Operations.$computeProvider; install.withValue(...)`) is not seen;
  L2 only direct references count: a scope that reaches BatchOperations through a helper function is not flagged,
     and a scope whose helper reaches Operations still counts as BatchOperations-only when it also names
     `BatchOperations.` directly (move such a call out of the scope);
  L3 any other `Operations.` member reference in the scope passes it, even one that never reads the provider
     (for example `Operations.self`);
  L4 a scope that names neither `BatchOperations.` nor another `Operations.` member is not judged, although it
     may be inert too (for example one that calls only `TopKSelection`);
  L5 `#if` branches are not evaluated (a BatchOperations-only scope under `#if false` is still flagged); Swift files outside
     Tests/ are not scanned;
  L6 P1-P3 do not trace calls through a third type (BatchOperations -> X -> Operations with X outside
     Operations.swift and BatchOperations.swift).
"""
import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location('swift_lexing', ROOT / 'Scripts/ci/swift_lexing.py')
lexing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lexing)

BATCH_OPERATIONS = 'Sources/VectorCore/Operations/BatchOperations.swift'
OPERATIONS = 'Sources/VectorCore/Operations/Operations.swift'

INSTALL = re.compile(r'(?<![\w$])Operations\s*\.\s*\$computeProvider\s*\.\s*withValue\s*\(')
BATCH_USE = re.compile(r'\bBatchOperations\s*\.')
OPERATIONS_USE = re.compile(r'(?<![\w$])Operations\s*\.(?!\s*\$computeProvider\b)')
PROVIDER_TOKEN = re.compile(r'\bcomputeProvider\b')

# The three sites removed by the final fix round, verbatim from tree 307441c (must be flagged).
REMOVED_WRAPPER_SITE = '''
    @Test func batchSelectionPreservesInputEdgeBehavior() async {
        let query = Vector<Dim2>(x: 0, y: 0)
        let candidates = [Float.nan, 2, 1, .nan].map { Vector<Dim2>(x: $0, y: 0) }
        await Operations.$computeProvider.withValue(CPUComputeProvider(mode: .sequential)) {
            let zeroK = await BatchOperations.findNearest(to: query, in: candidates, k: 0)
            let negativeK = await BatchOperations.findNearest(to: query, in: candidates, k: -1)
            let empty = await BatchOperations.findNearest(to: query, in: [Vector<Dim2>](), k: 3)
            #expect(zeroK.isEmpty)
            #expect(negativeK.isEmpty)
            #expect(empty.isEmpty)
            for k in [4, 99] {
                let result = await BatchOperations.findNearest(to: query, in: candidates, k: k)
                #expect(result.map(\\.index) == [2, 1, 0, 3])
                #expect(result.count == candidates.count)
                #expect(result[0].distance == 1)
                #expect(result[1].distance == 2)
                #expect(result[2].distance.isNaN)
                #expect(result[3].distance.isNaN)
            }
        }
    }
'''
REMOVED_SERIAL_TWIN = '''
        for k in [3, 10] {
            let observed = await p01ObservedNearest(cancelled: false) {
                await Operations.$computeProvider.withValue(CPUComputeProvider(mode: .sequential)) {
                    await BatchOperations.findNearest(to: query, in: rows, k: k)
                }
            }
'''
REMOVED_PARALLEL_TWIN = '''
        for k in [3, 103] {
            let observed = await p01ObservedNearest(cancelled: false) {
                await Operations.$computeProvider.withValue(CPUComputeProvider(mode: .parallel)) {
                    await BatchOperations.findNearest(to: zero, in: rows, k: k)
                }
            }
'''


def _close(masked, open_index, opening, closing):
    depth = 0
    for index in range(open_index, len(masked)):
        if masked[index] == opening:
            depth += 1
        elif masked[index] == closing:
            depth -= 1
            if depth == 0:
                return index
    raise ValueError('unbalanced %s%s' % (opening, closing))


def install_scopes(masked):
    """(offset of the install, scope text) for each `Operations.$computeProvider.withValue(` in masked text."""
    for match in INSTALL.finditer(masked):
        open_paren = match.end() - 1
        close_paren = _close(masked, open_paren, '(', ')')
        parts, argument, depth, index = [], 0, 0, open_paren + 1
        while index < close_paren:
            char = masked[index]
            if depth == 0 and char == ',':
                argument += 1
            elif depth == 0 and char == '{' and argument >= 1:
                end = _close(masked, index, '{', '}')
                parts.append(masked[index:end + 1])
                index = end
            elif char in '([{':
                depth += 1
            elif char in ')]}':
                depth -= 1
            index += 1
        after = close_paren + 1
        while after < len(masked) and masked[after] in ' \t\r\n':
            after += 1
        if after < len(masked) and masked[after] == '{':
            parts.append(masked[after:_close(masked, after, '{', '}') + 1])
        yield match.start(), '\n'.join(parts)


def violations(source, label='<fixture>'):
    """W1 violations in one Swift file's text: `label:line` of each inert install."""
    masked = lexing._mask(source, keep_interpolations=True)
    found = []
    for offset, scope in install_scopes(masked):
        if BATCH_USE.search(scope) and not OPERATIONS_USE.search(scope):
            found.append('%s:%d W1 Operations.$computeProvider install whose scope calls only BatchOperations '
                         '(BatchOperations does not read the provider)' % (label, masked.count('\n', 0, offset) + 1))
    return found


def premise_violations(batch_source, provider_files):
    """P1-P3 on BatchOperations.swift's text and the set of Sources paths whose masked text names computeProvider."""
    masked = lexing._mask(batch_source, keep_interpolations=True)
    found = []
    if PROVIDER_TOKEN.search(masked):
        found.append('P1 BatchOperations.swift names computeProvider')
    if OPERATIONS_USE.search(masked) or INSTALL.search(masked):
        found.append('P2 BatchOperations.swift references an Operations member')
    if set(provider_files) != {OPERATIONS}:
        found.append('P3 Sources files naming computeProvider are %s, expected only %s' % (sorted(provider_files), OPERATIONS))
    return found


def real_provider_files():
    return {str(path.relative_to(ROOT)) for path in sorted((ROOT / 'Sources').rglob('*.swift'))
            if PROVIDER_TOKEN.search(lexing._mask(path.read_text(encoding='utf-8'), keep_interpolations=True))}


def test_files():
    return sorted((ROOT / 'Tests').rglob('*.swift'))


def wrap_first_batch_call(source):
    """Mutation: re-insert an inert wrapper around the first `await BatchOperations.` expression of a line."""
    pattern = re.compile(r'^([ \t]*(?:let \w+ = )?)await (BatchOperations\.[^\n]*\))$', re.M)
    match = pattern.search(source)
    if match is None:
        raise ValueError('no `await BatchOperations.` line to wrap')
    wrapped = '%sawait Operations.$computeProvider.withValue(CPUComputeProvider(mode: .parallel)) { await %s }' % (
        match.group(1), match.group(2))
    return source[:match.start()] + wrapped + source[match.end():]


def wrap(body, install='Operations.$computeProvider.withValue(CPUComputeProvider(mode: .parallel))'):
    return '    func probe() async throws {\n        try await %s {\n%s\n        }\n    }\n' % (install, body)


BATCH_CALL = '            _ = await BatchOperations.findNearest(to: query, in: rows, k: 3)'
OPERATIONS_CALL = '            _ = try await Operations.findNearest(to: query, in: rows, k: 3)'


class FixtureControls(unittest.TestCase):
    def test_removed_sites_are_flagged(self):
        for site in (REMOVED_WRAPPER_SITE, REMOVED_SERIAL_TWIN + '        }\n', REMOVED_PARALLEL_TWIN + '        }\n'):
            found = violations(site)
            self.assertEqual(len(found), 1, (site, found))
            self.assertIn('W1', found[0])

    def test_batch_only_forms_are_flagged(self):
        forms = (
            wrap(BATCH_CALL),
            wrap(BATCH_CALL + '\n' + BATCH_CALL),
            wrap(BATCH_CALL, install='VectorCore.Operations\n            .$computeProvider\n            .withValue(p)'),
            '    _ = await Operations.$computeProvider.withValue(p, operation: {\n%s\n    })\n' % BATCH_CALL,
            '    _ = await Operations.$computeProvider.withValue(p) { await VectorCore.BatchOperations.findNearest(to: q, in: r, k: 1) }\n',
            # A nested install is not a use of the provider.
            wrap(BATCH_CALL + '\n            await Operations.$computeProvider.withValue(q) { _ = 1 }'),
        )
        for form in forms:
            self.assertEqual(len(violations(form)), 1, form)

    def test_operations_use_in_scope_passes(self):
        forms = (
            wrap(OPERATIONS_CALL),
            wrap(OPERATIONS_CALL + '\n' + BATCH_CALL),
            wrap(OPERATIONS_CALL + '\n            // BatchOperations.findNearest(to: query, in: rows, k: 3)\n'
                 '            print("BatchOperations.findNearest")'),
            wrap('            #expect(!(Operations.computeProvider is any BatchKernelProvider))\n' + BATCH_CALL),
            '    _ = try await Operations.$computeProvider.withValue(p, operation: {\n%s\n%s\n    })\n'
            % (OPERATIONS_CALL, BATCH_CALL),
        )
        for form in forms:
            self.assertEqual(violations(form), [], form)

    def test_batch_operations_outside_the_scope_passes(self):
        source = wrap(OPERATIONS_CALL) + BATCH_CALL + '\n'
        self.assertEqual(violations(source), [])
        # A closure inside the provider argument is not part of the scope.
        provider_only = '    await Operations.$computeProvider.withValue(P(plan: { BatchOperations.plan($0) })) { _ = 1 }\n'
        self.assertEqual(violations(provider_only), [])

    def test_commented_or_quoted_wrapper_passes(self):
        commented = '    // await Operations.$computeProvider.withValue(p) { await BatchOperations.findNearest(to: q, in: r) }\n'
        quoted = '    let text = "Operations.$computeProvider.withValue(p) { BatchOperations.findNearest() }"\n'
        block = '    /* await Operations.$computeProvider.withValue(p) {\n        await BatchOperations.findNearest(to: q, in: r)\n    } */\n'
        for form in (commented, quoted, block):
            self.assertEqual(violations(form), [], form)

    def test_documented_limits_still_pass(self):
        """LIMITS pin: L1-L4 forms pass today; strengthening W1 must update this test and the docstring."""
        limits = (
            # L1 alias install.
            '    let install = Operations.$computeProvider\n    await install.withValue(p) {\n%s\n    }\n' % BATCH_CALL,
            # L2 BatchOperations reached through a helper.
            wrap('            await runBatchOnly(query, rows)'),
            # L3 a non-reading Operations reference.
            wrap('            _ = Operations.self\n' + BATCH_CALL),
            # L4 neither BatchOperations nor Operations named.
            wrap('            _ = TopKSelection.select(k: 1, from: [1, 2])'),
        )
        for form in limits:
            self.assertEqual(violations(form), [], form)
        # L2 false positive: a helper that reaches Operations does not count, so a direct BatchOperations call is
        # still flagged. L5: `#if` is not evaluated.
        flagged = (wrap('            await helperThatCallsOperations()\n' + BATCH_CALL),
                   wrap('#if false\n' + BATCH_CALL + '\n#endif'))
        for form in flagged:
            self.assertEqual(len(violations(form)), 1, form)

    def test_interpolated_batch_call_is_flagged(self):
        """T17: L5 is closed. A BatchOperations call inside an interpolation is code, so it is flagged."""
        call = 'print("\\(await BatchOperations.findNearest(to: query, in: rows, k: 3))")'
        self.assertEqual(len(violations(wrap('            ' + call))), 1)

    def test_interpolated_provider_read_counts_as_a_read(self):
        """T17: an Operations read inside an interpolation is code, so the scope reads the provider."""
        read = 'print("\\(Operations.computeProvider)")'
        self.assertEqual(violations(wrap('            ' + read + '\n' + BATCH_CALL)), [])

    def test_names_in_plain_string_text_are_not_code(self):
        """T17: the same names in plain string text (no interpolation) are not code."""
        text = 'print("await BatchOperations.findNearest(to: query)")'
        self.assertEqual(violations(wrap('            ' + text)), [])

    def test_premise_rules_bite(self):
        source = (ROOT / BATCH_OPERATIONS).read_text(encoding='utf-8')
        files = real_provider_files()
        head = 'public enum BatchOperations {'
        reads = source.replace(head, head + '\n    static func probe() { _ = Operations.computeProvider }', 1)
        self.assertNotEqual(reads, source)
        found = premise_violations(reads, files)
        self.assertTrue(any(v.startswith('P1') for v in found) and any(v.startswith('P2') for v in found), found)
        calls = source.replace(head, head + '\n    static func probe() async throws {\n'
                               '        _ = try await Operations.findNearest(to: DynamicVector([1]), in: [DynamicVector([1])])\n    }', 1)
        self.assertEqual([v[:2] for v in premise_violations(calls, files)], ['P2'])
        self.assertEqual([v[:2] for v in premise_violations(source, files | {'Sources/VectorCore/Other.swift'})], ['P3'])
        commented = source.replace(head, head + '\n    // Operations.computeProvider is not read here.', 1)
        self.assertEqual(premise_violations(commented, files), [])

    def test_reinserting_a_wrapper_in_each_fixed_file_is_flagged(self):
        """Mutation: whatever the fixed files hold today, wrapping one of their BatchOperations calls is caught."""
        for relative in ('Tests/ComprehensiveTests/TopKNaNWrapperTests.swift', 'Tests/ComprehensiveTests/P01BatchRouteTests.swift'):
            source = (ROOT / relative).read_text(encoding='utf-8')
            self.assertEqual(violations(source, relative), [], relative)
            mutated = wrap_first_batch_call(source)
            self.assertNotEqual(mutated, source)
            self.assertEqual(len(violations(mutated, relative)), 1, relative)


class RealSources(unittest.TestCase):
    maxDiff = None

    def test_premise_batch_operations_does_not_read_the_provider(self):
        source = (ROOT / BATCH_OPERATIONS).read_text(encoding='utf-8')
        self.assertEqual(premise_violations(source, real_provider_files()), [],
                         'the inert-wrapper premise changed: re-derive W1 before editing this policy')

    def test_no_inert_provider_install_around_batch_operations(self):
        found = []
        for path in test_files():
            found += violations(path.read_text(encoding='utf-8'), str(path.relative_to(ROOT)))
        self.assertEqual(found, [], 'inert Operations.$computeProvider installs around BatchOperations-only code')

    def test_scan_sees_the_installs(self):
        """Non-vacuity: the Tests scan finds installs with an Operations use (so the scope extraction runs)."""
        scopes = [scope for path in test_files()
                  for _, scope in install_scopes(lexing._mask(path.read_text(encoding='utf-8'), keep_interpolations=True))]
        self.assertGreater(len(scopes), 0)
        self.assertTrue(any(OPERATIONS_USE.search(scope) for scope in scopes))


if __name__ == '__main__':
    unittest.main()
