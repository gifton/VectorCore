"""Claim ledger guard (V04): live check, extractor pins, mutation fixtures M1-M7, controls, the fix-round-1 cases
(every forbidden-phrase rule, the memory-saving class rule, body pins and flag tests, the narrowed R5 sentence rule,
superseded text, the Q14 defining-file count and numeric-unit literals) and the fix-round-2 cases (escape-leading
literals, the rule-5 fixture table, rule fixtures and screens, empty paths, forbidden-phrase literals that are never
excluded, and drift-guard citations compared by resolved test identity), fix round 3 (rule 5's hedge must qualify
the claim, line continuations, empty screens and malformed fixtures), and the final fix round (Markdown regions and
paths, rule 6 numeric performance claims, rule 7 source line-number citations, and the tombstones and corrected texts
of that round).

The check (Scripts/ci/check_claim_ledger.py) is an inventory and lexical guard backed by a manual review;
see its module docstring for the rules and limits. These tests show (a) the checked-in ledger matches the
sources, (b) each rule fails on the mutation it exists for, run against temporary copies of the
repository files the check reads, and (c) the controls (unchanged tree, rewrap-only edits, exact `== 0`
test code) do not fail. Nothing beyond these fixtures is proven.
"""

import copy
import hashlib
import importlib.util
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location('check_claim_ledger', ROOT / 'Scripts/ci/check_claim_ledger.py')
chk = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(chk)

LEDGER = 'Scripts/ci/claim_ledger.json'
SLA = 'Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift'
SOA = 'Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift'
MPK = 'Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift'
BATCH = 'Benchmarks/VectorCoreBench/Suites/BatchBench.swift'
COW = 'Sources/VectorCore/Storage/COWDynamicStorage.swift'
FOLD = 'Benchmarks/VectorCoreBench/Utils/OutputFold.swift'
PERF = 'Tests/ComprehensiveTests/MixedPrecisionPerformanceBenchmark.swift'
BENCH = 'Tests/MinimalTests/ComprehensiveBenchmarks.swift'
ATTR = 'Sources/VectorCore/Optimization/OptimizationAttributes.swift'
BOUNDARIES = 'Benchmarks/VectorCoreBench/Core/Boundaries.swift'
MODEL = 'Sources/VectorCoreBenchmarking/MeasurementBoundary.swift'
GUIDE = 'Docs/Performance_Guide.md'
BOUNDARY_DOC = 'Docs/Benchmark_Measurement_Boundaries.md'
README = 'README.md'
PLAIN_TEST = 'Tests/ComprehensiveTests/OptimizedVector512Tests.swift'   # not a region
PLAIN_SOURCE = 'Sources/VectorCore/Utilities/TriangularMatrix.swift'    # not a region
DRIFT = 'Tests/BenchmarkContractTests/BenchmarkContractTests.swift::testBodyFingerprintsMatchRecordedHashes'
LEDGER_EXCLUDE = re.compile(json.loads((ROOT / LEDGER).read_text(encoding='utf-8'))['vocabulary']['excludeLiterals'])


def rules(problems):
    return {problem.split(' ', 1)[0] for problem in problems}


class LiveLedger(unittest.TestCase):
    def test_checked_in_ledger_matches_sources(self):
        problems = chk.check(chk.Repo(ROOT))
        self.assertEqual(problems, [], '\n'.join(problems))

    def test_every_tier_a_statement_has_exactly_one_entry(self):
        """Independent recount: tier-A (symbol, fingerprint) groups equal live entries' coverage, region by region."""
        repo = chk.Repo(ROOT)
        ledger, vocabulary, exclude = chk.load(repo)
        live = [e for e in ledger['entries'] if e['disposition'] != 'removed']
        for region in ledger['regions']:
            path = region['path']
            groups = {}
            for st in chk.parse(repo, path, exclude).statements:
                if chk.tier_a(st.match, vocabulary):
                    groups[(st.symbol, st.fp)] = groups.get((st.symbol, st.fp), 0) + 1
            covered = {}
            for entry in live:
                if entry['path'] == path and (entry['symbol'], entry['fingerprint']) in groups:
                    key = (entry['symbol'], entry['fingerprint'])
                    covered[key] = covered.get(key, 0) + entry.get('occurrences', 1)
            self.assertEqual(covered, groups, path)

    def test_ledger_shape(self):
        ledger = json.loads((ROOT / LEDGER).read_text(encoding='utf-8'))
        self.assertEqual(ledger['schemaVersion'], 1)
        ids = [entry['id'] for entry in ledger['entries']]
        self.assertEqual(len(ids), len(set(ids)))
        # Tombstones exist and every one names an absent historical statement.
        tombstones = [e for e in ledger['entries'] if e['disposition'] == 'removed']
        self.assertGreater(len(tombstones), 0)
        self.assertTrue(all(e['history']['originalFingerprint'] == e['fingerprint'] for e in tombstones))
        # Q14 and Q17 records are present.
        self.assertTrue(any(item['symbol'] == 'performanceCritical' for item in ledger['callerInventory']))
        self.assertEqual(sum(site['count'] for site in ledger['recordedSites']), 39)
        # Q14 pins uses inside the defining file too.
        self.assertTrue(all(isinstance(item.get('definedInOccurrences'), int) for item in ledger['callerInventory']))

    def test_drift_guard_is_never_note_evidence(self):
        """Live ledger: a named-test reference that resolves to the drift guard (any spelling) belongs to an entry listed
        in bodyPinDescribers, which is exactly V04-C-0602 (the statement describing the guard). R8 enforces the same rule
        for every body-pin test; FixRound2 shows it failing on alternate spellings."""
        repo = chk.Repo(ROOT)
        ledger, _, exclude = chk.load(repo)
        self.assertEqual(ledger['bodyPinDescribers'], ['V04-C-0602'])
        drift = chk.test_identity(repo, DRIFT, exclude)
        self.assertIsNotNone(drift)
        for entry in ledger['entries']:
            evidence = entry['evidence']
            if evidence['kind'] != 'named-test':
                continue
            references = evidence['reference'] if isinstance(evidence['reference'], list) else [evidence['reference']]
            if any(chk.test_identity(repo, reference, exclude) == drift for reference in references):
                self.assertIn(entry['id'], ledger['bodyPinDescribers'])


class Extractor(unittest.TestCase):
    EXCLUDE = LEDGER_EXCLUDE

    def parse(self, source):
        return chk.SourceFile('X.swift', source, self.EXCLUDE)

    def test_paragraphs_bullets_and_fences(self):
        source = ('/// First line\n/// wrapped.\n///\n/// - Bullet one\n///   continues\n/// - Bullet two\n'
                  '/// ```swift\n/// let a = b // Cheap copy\n/// ```\nfunc f() {}\n')
        texts = [st.text for st in self.parse(source).statements]
        self.assertEqual(texts, ['First line wrapped.', '- Bullet one continues', '- Bullet two', 'let a = b // Cheap copy'])

    def test_rewrap_keeps_the_fingerprint(self):
        one = self.parse('/// Alpha beta gamma delta.\nfunc f() {}\n').statements[0]
        two = self.parse('/// Alpha beta\n///    gamma   delta.\nfunc f() {}\n').statements[0]
        self.assertEqual(one.fp, two.fp)
        self.assertEqual(one.fp, chk.fingerprint('alpha beta gamma delta.'))

    def test_symbols(self):
        source = ('enum K {\n  /// Doc of f.\n  @inlinable\n  static func f() {\n    // Inside f.\n    let x = 1 // trailing\n  }\n'
                  '  // MARK: - Section\n\n  struct S { let n = 2 }\n}\n')
        found = {st.text: st.symbol for st in self.parse(source).statements}
        self.assertEqual(found['Doc of f.'], 'enum K::func f')
        self.assertEqual(found['Inside f.'], 'enum K::func f')
        self.assertEqual(found['trailing'], 'enum K::func f')
        self.assertEqual(found['MARK: - Section'], 'enum K::struct S')

    def test_literals(self):
        source = ('struct T {\n  @Test("Latency: fast path")\n  func g() {\n'
                  '    print("part one " + "part two \\(x) done")\n    let id = "batch.euclid2.512"\n  }\n}\n')
        found = {st.text: st.symbol for st in self.parse(source).statements if st.kind == 'string'}
        self.assertEqual(found, {'Latency: fast path': 'struct T::func g', 'part one part two \\() done': 'struct T::func g'})

    def test_numeric_unit_literals_are_statements_and_pure_identifiers_are_not(self):
        digest = '081006cf564ee7550dbbdb6b81311da51907744bb4fcad18336aeffc80a3fe5a'
        source = ('func t(n: Int) {\n  let a = "50%"\n  let b = "10-20%"\n  let c = "3x-faster"\n  let d = "\\(n)x"\n'
                  '  let e = "batch.euclid2.512"\n  let f = "Vector\\(n)Optimized"\n  let g = "' + digest + '"\n'
                  '  let h = "u64-bitfold-sink-v1"\n  let i = "1"\n}\n')
        texts = [st.text for st in self.parse(source).statements]
        self.assertEqual(texts, ['50%', '10-20%', '3x-faster', '\\()x'])

    def test_code_tokens_are_never_statements(self):
        source = 'func t() {\n  #expect(value == 0)\n  #expect(bits.bitPattern == 0x7FC0_0000)\n}\n'
        self.assertEqual(self.parse(source).statements, [])


class TemporaryRepository(unittest.TestCase):
    """A temporary copy of every file the check reads; no tests of its own."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        for directory in ('Sources', 'Tests', 'Benchmarks'):
            shutil.copytree(ROOT / directory, cls.root / directory,
                            ignore=shutil.ignore_patterns('.build', '*.o', '*.swiftmodule'))
        (cls.root / 'Scripts/ci').mkdir(parents=True)
        shutil.copy2(ROOT / LEDGER, cls.root / LEDGER)
        # Literal paths outside those trees that the ledger names (Markdown regions and forbidden-phrase paths).
        ledger = json.loads((ROOT / LEDGER).read_text(encoding='utf-8'))
        literal = {region['path'] for region in ledger['regions']}
        literal |= {path for rule in ledger['forbiddenPhrases'] for path in rule['paths'] if not chk.GLOB_META.search(path)}
        for path in sorted(literal):
            if not (cls.root / path).exists():
                (cls.root / path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / path, cls.root / path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_with(self, path, transform=None, ledger_transform=None):
        target = self.root / path if path else None
        original = target.read_text(encoding='utf-8') if target else None
        ledger_file = self.root / LEDGER
        ledger_text = ledger_file.read_text(encoding='utf-8')
        try:
            if target:
                changed = transform(original)
                self.assertTrue(changed != original, f'mutation did not apply to {path}')
                target.write_text(changed, encoding='utf-8')
            if ledger_transform:
                ledger = json.loads(ledger_text)
                ledger_transform(ledger)
                ledger_file.write_text(json.dumps(ledger, ensure_ascii=False), encoding='utf-8')
            return chk.check(chk.Repo(self.root))
        finally:
            if target:
                target.write_text(original, encoding='utf-8')
            ledger_file.write_text(ledger_text, encoding='utf-8')

    @staticmethod
    def entry(ledger, path, needle):
        found = [e for e in ledger['entries'] if e['path'] == path and needle in e['statement']]
        assert len(found) == 1, (path, needle, len(found))
        return found[0]


class Mutations(TemporaryRepository):
    """M1-M7 and the controls on a temporary copy of every file the check reads."""

    def test_control_unchanged_copy_passes(self):
        self.assertEqual(self.run_with(None), [])

    def test_control_rewrap_and_whitespace_only(self):
        def rewrap(text):
            old = ('/// Single-position sensitivity (derivable here in one step): $p$ is odd, hence invertible mod $2^{64}$, so\n'
                   '/// $h \\mapsto (h \\oplus w)\\cdot p$ is a bijection')
            new = ('/// Single-position   sensitivity (derivable here in one step): $p$ is odd,\n'
                   '///    hence invertible mod $2^{64}$, so $h \\mapsto (h \\oplus w)\\cdot p$ is a bijection')
            assert old in text
            return text.replace(old, new) + '\n\n'
        self.assertEqual(self.run_with(FOLD, rewrap), [])

    def test_control_exact_zero_test_code_not_flagged(self):
        def add_code(text):
            anchor = '    func validateHeuristic() throws {\n'
            assert anchor in text
            return text.replace(anchor, anchor + '        #expect(Float(0).bitPattern == 0)\n        #expect(0.0 == 0)\n')
        self.assertEqual(self.run_with(PERF, add_code), [])

    def test_m1a_deleting_the_source_reference_fails_r3(self):
        problems = self.run_with(SLA, lambda t: t.replace('LinearAlgebraRangeTests.sharedExponentSweep exercises', 'This exercises'))
        self.assertIn('R3', rules(problems), problems)

    def test_m1b_deleting_the_ledger_reference_fails(self):
        def drop(ledger):
            self.entry(ledger, SLA, 'householder qr')['evidence']['reference'] = ''
        problems = self.run_with(None, ledger_transform=drop)
        self.assertTrue({'R8', 'R3'} & rules(problems), problems)

    def test_m1c_nonexistent_test_fails_r3(self):
        def point(ledger):
            evidence = self.entry(ledger, SLA, 'householder qr')['evidence']
            evidence['reference'] = evidence['reference'].replace('sharedExponentSweep', 'sharedExponentSweepThatDoesNotExist')
        problems = self.run_with(None, ledger_transform=point)
        self.assertIn('R3', rules(problems), problems)
        self.assertTrue(any('no test func sharedExponentSweepThatDoesNotExist' in p for p in problems), problems)

    def test_m2_unsupported_quantitative_assertion_fails_r6(self):
        anchor = '    /// Euclidean squared distance for 512-dimensional vectors using SoA layout\n'
        problems = self.run_with(SOA, lambda t: t.replace(anchor, '    /// Achieves ~3x speedup on Apple Silicon.\n    ///\n' + anchor))
        self.assertIn('R6', rules(problems), problems)
        self.assertTrue(any('achieves ~3x speedup' in p.lower() for p in problems), problems)

    def test_m3_changed_statement_without_fingerprint_update_fails_r1(self):
        def edit(text):
            old = 'so results are **bitwise-identical** to the 4-way path (`tiledEuclid512MatchesFourWayBitwise`'
            assert text.count(old) == 1
            return text.replace(old, 'so results are **bit-identical** to the 4-way path (`tiledEuclid512MatchesFourWayBitwise`')
        problems = self.run_with(SOA, edit)
        self.assertIn('R1', rules(problems), problems)

    def test_m4_reintroduced_tombstoned_text_fails_r2_and_r7(self):
        def revert(text):
            old = ('        // Build SoA once, before warmup and timing. This preparation is outside the timed body\n'
                   '        // (see the case\'s measurement boundary: inputPreparation = excluded).\n')
            assert old in text
            return text.replace(old, '        // Build SoA once (this cost is part of the benchmark)\n', 1)
        problems = self.run_with(BATCH, revert)
        self.assertTrue({'R2', 'R7'} <= rules(problems), problems)

    def test_m4_reintroduced_removed_statement_fails_r2(self):
        anchor = '/// - O(n) copy on first mutation after sharing\n'
        problems = self.run_with(COW, lambda t: t.replace(anchor, anchor + '/// - No overhead for unique instances\n'))
        self.assertIn('R2', rules(problems), problems)

    def test_m5_forbidden_variant_fails_r7(self):
        anchor = '    public func acquire() -> MixedPrecisionKernels.Vector512FP16? {\n'
        problems = self.run_with(MPK, lambda t: t.replace(anchor, anchor + '        // lock-free fast path\n'))
        self.assertIn('R7', rules(problems), problems)

    def test_m6_statement_moved_to_another_symbol_fails_r1(self):
        def move(text):
            line = '    /// - Complexity: O(n) where n is count\n'
            target = '    /// Access the underlying memory as an unsafe buffer pointer for reading.\n'
            assert text.count(line) == 1 and text.count(target) == 1
            return text.replace(line, '').replace(target, target + '    ///\n' + line)
        problems = self.run_with(COW, move)
        self.assertIn('R1', rules(problems), problems)
        self.assertIn('R6', rules(problems), problems)

    def test_m7_duplicate_id_fails_r8(self):
        def duplicate(ledger):
            ledger['entries'][1]['id'] = ledger['entries'][0]['id']
        problems = self.run_with(None, ledger_transform=duplicate)
        self.assertTrue(any(p.startswith('R8 duplicate id') for p in problems), problems)

    def test_m7_double_covered_fingerprint_fails_r8(self):
        def cover_twice(ledger):
            extra = copy.deepcopy(next(e for e in ledger['entries'] if e['disposition'] == 'retained'))
            extra['id'] = 'V04-C-9999'
            ledger['entries'].append(extra)
        problems = self.run_with(None, ledger_transform=cover_twice)
        self.assertTrue(any(p.startswith('R8 fingerprint double-covered') for p in problems), problems)

    def test_r5_unverified_contract_needs_wording(self):
        def strip(ledger):
            entry = next(e for e in ledger['entries'] if e['disposition'] == 'unverified-contract' and e['kind'] == 'comment')
            entry['statement'] = 'a contract without the required words'
            entry['fingerprint'] = chk.fingerprint(entry['statement'])
        problems = self.run_with(None, ledger_transform=strip)
        self.assertIn('R5', rules(problems), problems)

    def test_r4_dated_artifact_must_exist_and_match(self):
        artifact = 'Tests/ComprehensiveTests/OptimizedVector512Tests.swift'
        digest = hashlib.sha256((self.root / artifact).read_bytes()).hexdigest()

        def cite(sha):
            def apply(ledger):
                entry = next(e for e in ledger['entries'] if e['disposition'] == 'retained')
                entry['evidence'] = {'kind': 'dated-artifact', 'reference': {'path': artifact, 'sha256': sha, 'date': '2026-10-04'},
                                     'domain': 'fixture', 'limitations': 'fixture'}
            return apply
        self.assertNotIn('R4', rules(self.run_with(None, ledger_transform=cite(digest))))
        self.assertIn('R4', rules(self.run_with(None, ledger_transform=cite('0' * 64))))

    def test_q14_new_caller_of_a_gated_helper_fails(self):
        anchor = '    public init(capacity: Int, dimension: Int = 512) {\n'
        problems = self.run_with(MPK, lambda t: t.replace(anchor, anchor + '        _ = performanceCritical { 0 }\n'))
        self.assertIn('Q14', rules(problems), problems)

    def test_q17_new_discarded_result_in_a_timed_body_fails(self):
        anchor = '                _ = v1.dotProduct(v2)\n'
        problems = self.run_with(BENCH, lambda t: t.replace(anchor, anchor + anchor, 1))
        self.assertIn('Q17', rules(problems), problems)

    def test_no_blanket_suppression_for_new_non_claims(self):
        anchor = '    // MARK: - Dimension-Specific Public APIs\n'
        problems = self.run_with(SOA, lambda t: t.replace(anchor, anchor + '    // MARK: - Cache helpers\n'))
        self.assertIn('R6', rules(problems), problems)


class FixRound1(TemporaryRepository):
    """Fix round 1 (V04 phase F review): forbidden-phrase rules, memory-saving class, body pins, R5 sentence rule,
    superseded text, Q14 defining file, numeric-unit literals. Same temporary copy as Mutations."""

    # One sample per forbidden-phrase rule, keyed by a substring of the rule's pattern: (file, appended source line).
    SAMPLES = {
        'this cost is part of the benchmark': (BATCH, '// Build once (this cost is part of the benchmark)'),
        'lock[- ]free': (MPK, '// lock-free fast path'),
        'garbage[- ]collect': (MPK, '// released by the garbage collector'),
        'prefetch hint': (SOA, '/// Prefetch hint for the next block'),
        'memory|ram|bandwidth|storage|footprint': (PLAIN_SOURCE, '// Provides 4x memory reduction'),
        'nanoseconds?|microseconds?': (GUIDE, '**Impact**: 5-10x speedup for large batches'),
        '(?:swift|md|py):': (BOUNDARIES, '// see MixedPrecisionKernels.swift:3375-3388'),
        'battle[- ]tested': (PLAIN_SOURCE, '/// High-performance helper.'),                     # rule 8 (T13 RD1)
        'slowdown|improvements?|throughput|performance|gains?': (PLAIN_SOURCE, '/// ~75ns per call'),   # rule 9 (RD4)
        'zero[- ]?copy': (PLAIN_SOURCE, '/// zero-copy view'),                                 # rule 10 (T13 RD5)
    }
    MEMORY_FAIL = ['// 4x memory reduction', 'let label = "Memory savings: 50%"', '// Should save ~50% memory',
                   '// - Savings: ~50% for large n', 'let line = "    Reduction: \\(ratio)x"',
                   '// 4x more documents fit in memory', '// reducing memory usage by approximately 50%']
    MEMORY_PASS = ['// it is not a measured memory saving', '// Logical scalar payload model (not process memory)',
                   '// no memory saving is measured', 'let label = "Error reduction: \\(e)%"', '// Horizontal reduction and store']

    def append(self, lines):
        return lambda text: text + '\n' + '\n\n'.join(lines) + '\n'

    def covering_entries(self, path, transform):
        """Ledger transform adding a reviewed entry for every tier-A statement the source transform introduces in a
        region file, so that R6 is satisfied and any remaining failure is the rule under test."""
        before = (self.root / path).read_text(encoding='utf-8')
        repo = chk.Repo(self.root)
        ledger, vocabulary, exclude = chk.load(repo)
        old = {(st.symbol, st.fp) for st in chk.source_file(path, before, exclude).statements}
        new = [st for st in chk.source_file(path, transform(before), exclude).statements
               if (st.symbol, st.fp) not in old and chk.tier_a(st.match, vocabulary)]
        regions = {region['path'] for region in ledger['regions']}

        def apply(ledger):
            if path not in regions:
                return
            for index, st in enumerate(new):
                ledger['entries'].append({
                    'id': f'V04-C-98{index:02d}', 'path': path, 'symbol': st.symbol, 'fingerprint': st.fp, 'statement': st.norm,
                    'kind': st.kind, 'category': 'non-claim', 'disposition': 'not-a-claim', 'reason': 'fixture',
                    'evidence': {'kind': 'none', 'reference': '', 'domain': '', 'limitations': ''}, 'acceptedContract': None,
                    'history': {'originalStatement': None, 'originalFingerprint': None, 'origin': 'test'}})
        return apply

    def test_every_forbidden_phrase_rule_fails_its_mutation_independently_of_r6(self):
        ledger = json.loads((self.root / LEDGER).read_text(encoding='utf-8'))
        self.assertEqual(len(ledger['forbiddenPhrases']), 10)
        for rule in ledger['forbiddenPhrases']:
            keys = [key for key in self.SAMPLES if key in rule['pattern']]
            self.assertEqual(len(keys), 1, f'forbidden-phrase rule without exactly one mutation sample: {rule["pattern"][:60]}')
            path, line = self.SAMPLES[keys[0]]
            self.assertTrue(re.search(rule['pattern'], chk.normalize(re.sub(r'^\s*/+\s*', '', line))), line)
            transform = self.append([line])
            with self.subTest(rule=keys[0]):
                problems = self.run_with(path, transform, ledger_transform=self.covering_entries(path, transform))
                self.assertNotIn('R6', rules(problems), problems)
                self.assertTrue(any(p.startswith(f'R7 {path}:') for p in problems), problems)

    def test_memory_saving_claims_fail_and_disclaimers_pass(self):
        problems = self.run_with(PLAIN_TEST, self.append(self.MEMORY_FAIL))
        hits = [p for p in problems if p.startswith(f'R7 {PLAIN_TEST}:')]
        self.assertEqual(len(hits), len(self.MEMORY_FAIL), problems)
        self.assertEqual(self.run_with(PLAIN_TEST, self.append(self.MEMORY_PASS)), [])

    def test_memory_rule_has_no_exemption_and_fails_the_old_runtime_text(self):
        problems = self.run_with(PLAIN_TEST, self.append(['let s = "• 50% memory savings with minimal accuracy loss"']))
        self.assertTrue(any(p.startswith(f'R7 {PLAIN_TEST}:') for p in problems), problems)
        # D02 (RD-R2) reworded the `.fp16` runtime text, so the rule carries no exemption: restoring the old text in
        # MixedPrecisionKernels.swift fails it there too.
        memory = next(r for r in json.loads((self.root / LEDGER).read_text(encoding='utf-8'))['forbiddenPhrases']
                      if 'memory|ram|bandwidth' in r['pattern'])
        self.assertNotIn('exempt', memory)
        new = '\\n• FP16: 2 bytes per scalar, FP32: 4 (scalar payload)"'
        old = '\\n• 50% memory savings with minimal accuracy loss"'

        def restore(text):
            assert text.count(new) == 1
            return text.replace(new, old)
        problems = self.run_with(MPK, restore)
        self.assertTrue(any(p.startswith(f'R7 {MPK}:') for p in problems), problems)

    def test_synthetic_exemption_suppresses_its_statement_and_a_stale_one_is_reported(self):
        # D02 phase 5 fix 1 (A-M3): the live ledger carries no exemption since D02 RD-R2, so this synthetic one keeps
        # the checker's exemption code (shape check, suppression, stale report) under test.
        transform = self.append(['let s = "• 50% memory savings with minimal accuracy loss"'])
        before = (self.root / PLAIN_TEST).read_text(encoding='utf-8')
        _, _, exclude = chk.load(chk.Repo(self.root))
        old = {st.fp for st in chk.source_file(PLAIN_TEST, before, exclude).statements}
        added = [st.fp for st in chk.source_file(PLAIN_TEST, transform(before), exclude).statements if st.fp not in old]
        self.assertEqual(len(added), 1, added)

        def exemption(item):
            def apply(ledger):
                next(r for r in ledger['forbiddenPhrases'] if 'memory|ram|bandwidth' in r['pattern'])['exempt'] = [item]
            return apply
        valid = {'path': PLAIN_TEST, 'fingerprint': added[0], 'owner': 'D02', 'reason': 'fixture'}
        self.assertEqual(self.run_with(PLAIN_TEST, transform, ledger_transform=exemption(valid)), [])
        problems = self.run_with(PLAIN_TEST, transform, ledger_transform=exemption(dict(valid, fingerprint='0' * 16)))
        self.assertTrue(any(p.startswith(f'R7 {PLAIN_TEST}:') for p in problems), problems)
        self.assertTrue(any(f'exemption {PLAIN_TEST} {"0" * 16} matches no present statement (stale exemption)' in p
                            for p in problems), problems)
        malformed = self.run_with(None, ledger_transform=exemption({'path': PLAIN_TEST}))
        self.assertTrue(any('an exemption needs path, 16-hex fingerprint, owner and reason' in p for p in malformed),
                        malformed)

    def test_literal_forbidden_path_that_matches_nothing_fails_r8(self):
        def rename(ledger):
            rule = next(r for r in ledger['forbiddenPhrases'] if 'prefetch hint' in r['pattern'])
            rule['paths'][0] = 'Sources/VectorCore/Operations/Kernels/DoesNotExist.swift'
        problems = self.run_with(None, ledger_transform=rename)
        self.assertTrue(any(p.startswith('R8') and 'matches no file' in p for p in problems), problems)

    def test_anchored_pattern_without_screen_fails_r8(self):
        def unscreen(ledger):
            del next(r for r in ledger['forbiddenPhrases'] if 'memory|ram|bandwidth' in r['pattern'])['screen']
        problems = self.run_with(None, ledger_transform=unscreen)
        self.assertTrue(any(p.startswith('R8') and 'screen' in p for p in problems), problems)

    def test_body_pin_must_resolve_and_is_not_named_evidence(self):
        def bad_pin(ledger):
            entry = next(e for e in ledger['entries'] if 'bodyPin' in e['evidence'])
            entry['evidence']['bodyPin'] = [DRIFT.replace('testBodyFingerprints', 'testNoSuchFingerprints')]
        problems = self.run_with(None, ledger_transform=bad_pin)
        self.assertTrue(any(p.startswith('R8') and 'bodyPin does not resolve' in p for p in problems), problems)

        def double(ledger):
            entry = next(e for e in ledger['entries'] if e['evidence']['kind'] == 'named-test'
                         and DRIFT in (e['evidence']['reference'] if isinstance(e['evidence']['reference'], list)
                                       else [e['evidence']['reference']]))
            entry['evidence']['bodyPin'] = [DRIFT]
        problems = self.run_with(None, ledger_transform=double)
        self.assertTrue(any(p.startswith('R8') and 'not evidence' in p for p in problems), problems)

    def test_flag_tests_must_resolve(self):
        def bad(ledger):
            entry = next(e for e in ledger['entries'] if 'flagTests' in e['evidence'])
            entry['evidence']['flagTests'] = ['Tests/BenchmarkContractTests/BenchmarkContractTests.swift::testNoSuchFlags']
        problems = self.run_with(None, ledger_transform=bad)
        self.assertTrue(any(p.startswith('R8') and 'flagTests does not resolve' in p for p in problems), problems)

    def test_r5_inherited_sentence_must_refer_to_runtime_text(self):
        def drop_reference(text):
            old = '/// Not verified: the runtime text of the routes, representations and notes below is written by hand from the route'
            assert text.count(old) == 1
            return text.replace(old, '/// Not verified: the routes, representations and notes below are written by hand from the route')
        problems = self.run_with(BOUNDARIES, drop_reference)
        self.assertEqual(rules(problems), {'R5'}, problems)
        self.assertTrue(any(p.startswith('R5 V04-C-0617') for p in problems), problems)

    def test_superseded_text_is_a_tombstone(self):
        def revert(text):
            new = ('`@inline(__always)` requests inlining of this wrapper only; it does not\n'
                   '/// mark `operation` always-inline.')
            assert text.count(new) == 1
            return text.replace(new, '`@inline(__always)` applies to this wrapper; it does not change how the\n'
                                     '/// closure body is compiled.')
        problems = self.run_with(ATTR, revert)
        self.assertTrue(any(p.startswith('R2 V04-C-0432') and 'superseded' in p for p in problems), problems)

    def test_q14_new_use_inside_the_defining_file_fails(self):
        problems = self.run_with(ATTR, self.append(['let q14Probe = performanceCritical { 0 }']))
        self.assertTrue(any(p.startswith('Q14 V04-Q14-performanceCritical') and 'defining file' in p for p in problems), problems)

    def test_numeric_unit_literals_in_a_region_are_uncovered_claims(self):
        problems = self.run_with(SOA, self.append(['let a = "3x-faster"', 'let b = "10-20%"', 'let c = "50%"']))
        uncovered = [p for p in problems if p.startswith(f'R6 {SOA}:')]
        for literal in ('3x-faster', '10-20%', '50%'):
            self.assertTrue(any(p.endswith(': ' + literal) for p in uncovered), (literal, problems))


class FixtureHelpers(TemporaryRepository):
    '''Helpers for the fix-round cases; no tests of its own.'''

    def append(self, lines):
        return lambda text: text + '\n' + '\n\n'.join(lines) + '\n'

    @staticmethod
    def swift_literal(text):
        if '\n' in text:   # a fixture with a line break (a line continuation) needs a multi-line literal
            return 'let fixture = """\n' + text + '\n"""'
        return 'let fixture = "' + text.replace('"', '\\"') + '"'

    def rule(self, ledger, needle):
        return next(r for r in ledger['forbiddenPhrases'] if needle in r['pattern'])


class FixRound2(FixtureHelpers):
    '''Fix round 2 (V04 phase F re-review N-1 to N-6) on the same temporary copy.'''

    def test_escape_leading_literals_are_seen_by_rule5_and_r6(self):
        lines = ['print("\\nMemory savings: 50%")', 'print("\\nSaves 50% memory")', 'print("\\n  Savings: 50%")']
        problems = self.run_with(PLAIN_TEST, self.append(lines))
        self.assertEqual(len([p for p in problems if p.startswith(f'R7 {PLAIN_TEST}:')]), 3, problems)
        problems = self.run_with(SOA, self.append(['let note = "\\nFaster on Apple silicon"']))
        self.assertTrue(any(p.startswith(f'R6 {SOA}:') and 'Faster on Apple silicon' in p for p in problems), problems)

    def test_rule5_fixture_table_through_source(self):
        ledger = json.loads((self.root / LEDGER).read_text(encoding='utf-8'))
        fixtures = self.rule(ledger, 'memory|ram|bandwidth')['fixtures']
        self.assertGreaterEqual(len(fixtures['fail']), 20)
        self.assertGreaterEqual(len(fixtures['pass']), 10)
        problems = self.run_with(PLAIN_TEST, self.append([self.swift_literal(t) for t in fixtures['fail']]))
        hits = [p for p in problems if p.startswith(f'R7 {PLAIN_TEST}:')]
        self.assertEqual(len(hits), len(fixtures['fail']), problems)
        problems = self.run_with(PLAIN_TEST, self.append([self.swift_literal(t) for t in fixtures['pass']]))
        self.assertEqual(problems, [])
        comments = ['// ' + t for t in fixtures['fail'] if '\\' not in t]
        problems = self.run_with(PLAIN_TEST, self.append(comments))
        self.assertEqual(len([p for p in problems if p.startswith(f'R7 {PLAIN_TEST}:')]), len(comments), problems)

    def test_rule_fixtures_and_screens_are_validated(self):
        def unmatched(ledger):
            self.rule(ledger, 'memory|ram|bandwidth')['fixtures']['fail'][0] = 'nothing to see here'
        def blind_screen(ledger):
            self.rule(ledger, 'memory|ram|bandwidth')['screen'] = 'zzzz'
        def matching_pass(ledger):
            self.rule(ledger, 'memory|ram|bandwidth')['fixtures']['pass'].append('4x memory reduction')
        def no_fixtures(ledger):
            del self.rule(ledger, 'this cost is part')['fixtures']
        for transform, message in ((unmatched, 'does not match the pattern'), (blind_screen, 'screen rejects'),
                                   (matching_pass, 'must-pass fixture'), (no_fixtures, 'needs at least one must-fail')):
            with self.subTest(message=message):
                problems = self.run_with(None, ledger_transform=transform)
                self.assertTrue(any(p.startswith('R8') and message in p for p in problems), problems)

    def test_empty_paths_fail_r8(self):
        def empty(ledger):
            self.rule(ledger, 'this cost is part')['paths'] = []
        problems = self.run_with(None, ledger_transform=empty)
        self.assertTrue(any(p.startswith('R8') and 'paths is missing or empty' in p for p in problems), problems)

    def test_anchors_and_lookarounds_need_a_screen(self):
        patterns = ('fast path$', 'path\\Z', '\\Afast', 'fast(?= path)')

        def add(ledger):   # one run: a new pattern changes the exclusion key, so every file is lexed again
            for pattern in patterns:
                ledger['forbiddenPhrases'].append({'pattern': pattern, 'paths': ['Sources/**/*.swift'], 'reason': 'probe',
                                                   'fixtures': {'fail': ['fast path']}})
        problems = self.run_with(None, ledger_transform=add)
        for pattern in patterns:
            with self.subTest(pattern=pattern):
                self.assertTrue(any(p.startswith('R8') and repr(pattern)[:20] in p and 'explicit keyword screen' in p
                                    for p in problems), problems)

    def test_forbidden_phrase_literals_are_never_excluded(self):
        problems = self.run_with(PLAIN_SOURCE, self.append(['let mode = "lock-free"']))
        self.assertTrue(any(p.startswith(f'R7 {PLAIN_SOURCE}:') for p in problems), problems)
        problems = self.run_with(SOA, self.append(['let hint = "auto-prefetch"']))
        self.assertTrue(any(p.startswith(f'R7 {SOA}:') for p in problems), problems)
        _, _, exclude = chk.load(chk.Repo(self.root))
        texts = [st.text for st in chk.SourceFile('X.swift', 'let a = "lock-free"\\nlet b = "zero-copy"\\n', exclude).statements]
        self.assertEqual(texts, ['lock-free', 'zero-copy'])   # both match a forbidden phrase (T13 RD5), so neither is excluded

    def test_drift_guard_cited_by_any_spelling_fails_r8(self):
        qualified = DRIFT.replace('::', '::BenchmarkBoundaryDriftContractTests.')
        wrong_suite = DRIFT.replace('::', '::BenchmarkContractTests.')

        def recite(entry_id, reference, keep_pin):
            def apply(ledger):
                entry = next(e for e in ledger['entries'] if e['id'] == entry_id)
                entry['disposition'] = 'retained'
                pins = entry['evidence'].get('bodyPin')
                entry['evidence'] = {'kind': 'named-test', 'reference': [reference], 'domain': 'probe', 'limitations': 'probe'}
                if keep_pin and pins:
                    entry['evidence']['bodyPin'] = pins
            return apply
        for entry_id, reference, keep_pin in (('V04-C-0638', wrong_suite, True), ('V04-C-0638', wrong_suite, False),
                                              ('V04-C-0638', qualified, False), ('V04-C-0633', DRIFT, False),
                                              ('V04-C-0633', qualified, True)):
            with self.subTest(entry=entry_id, reference=reference.split('::')[1], keep_pin=keep_pin):
                problems = self.run_with(None, ledger_transform=recite(entry_id, reference, keep_pin))
                self.assertTrue(any(p.startswith(f'R8 {entry_id}') and 'body-pin test' in p for p in problems), problems)

    def test_stale_body_pin_describer_fails_r8(self):
        def stale(ledger):
            ledger['bodyPinDescribers'].append('V04-C-0001')
        problems = self.run_with(None, ledger_transform=stale)
        self.assertTrue(any(p.startswith('R8 bodyPinDescribers V04-C-0001') for p in problems), problems)


class FixRound3(FixtureHelpers):
    """Fix round 3 (re-review F-1 to F-3) on the same temporary copy."""

    HEDGE_FAIL = ['saves 50% memory, which is not verified by competitors', 'saves 50% memory -- never claimed by anyone',
                  'saves 50% memory and that was not measured last year, but trust me',
                  'saves 50% memory, not measured elsewhere but real here', 'saves 50% memory, not promised by others',
                  'not measured; saves 50% memory']
    HEDGE_PASS = ['saves 50% memory (not verified)', 'saves 50% memory (unverified)']

    def test_hedge_must_qualify_the_claim_and_end_its_sentence(self):
        ledger = json.loads((self.root / LEDGER).read_text(encoding='utf-8'))
        fixtures = self.rule(ledger, 'memory|ram|bandwidth')['fixtures']
        self.assertTrue(set(self.HEDGE_FAIL) <= set(fixtures['fail']))
        self.assertTrue(set(self.HEDGE_PASS) <= set(fixtures['pass']))
        problems = self.run_with(PLAIN_TEST, self.append([self.swift_literal(t) for t in self.HEDGE_FAIL]))
        self.assertEqual(len([p for p in problems if p.startswith(f'R7 {PLAIN_TEST}:')]), len(self.HEDGE_FAIL), problems)
        self.assertEqual(self.run_with(PLAIN_TEST, self.append([self.swift_literal(t) for t in self.HEDGE_PASS])), [])

    def test_line_continuation_joins_lines_before_matching(self):
        literal = 'let note = """\nSaves 50% \\\nmemory\n"""'
        problems = self.run_with(PLAIN_TEST, self.append([literal]))
        self.assertTrue(any(p.startswith(f'R7 {PLAIN_TEST}:') for p in problems), problems)

    def test_empty_screen_and_malformed_fixtures_fail_r8_without_a_traceback(self):
        def empty_screen(ledger):
            self.rule(ledger, 'memory|ram|bandwidth')['screen'] = ''
        def non_string_fixture(ledger):
            self.rule(ledger, 'memory|ram|bandwidth')['fixtures']['fail'].append(5)
            self.rule(ledger, 'lock[- ]free')['fixtures']['pass'].append(None)
        def fixtures_not_object(ledger):
            self.rule(ledger, 'lock[- ]free')['fixtures'] = ['lock-free']
        for transform, messages in ((empty_screen, ['screen must be a non-empty string', 'explicit keyword screen']),
                                    (non_string_fixture, ['fixtures.fail has non-string entries',
                                                          'fixtures.pass has non-string entries']),
                                    (fixtures_not_object, ['fixtures must be an object'])):
            with self.subTest(case=transform.__name__):
                problems = self.run_with(None, ledger_transform=transform)
                for message in messages:
                    self.assertTrue(any(p.startswith('R8') and message in p for p in problems), (message, problems))


class MarkdownExtractor(unittest.TestCase):
    """Markdown statements (normalization markdown-text-v1) and section symbols."""

    def test_paragraphs_items_rows_headings_fences_and_quotes(self):
        source = ('Intro line one\nwrapped.\n\n# Title\n\n## Part A\n\n- Item one\n  continues\n- Item two\n\n'
                  '| a | b |\n|---|---|\n| 1 | 2 |\n\n```bash\n# a shell comment\nrun --x\n```\n\n> Quoted\n> text.\n\n---\n'
                  '### Part B\n\nLast.\n')
        found = [(st.text, st.symbol) for st in chk.MarkdownFile('X.md', source).statements]
        self.assertEqual(found, [
            ('Intro line one wrapped.', '<file>'), ('# Title', 'section Title'),
            ('## Part A', 'section Title::section Part A'), ('- Item one continues', 'section Title::section Part A'),
            ('- Item two', 'section Title::section Part A'), ('| a | b |', 'section Title::section Part A'),
            ('|---|---|', 'section Title::section Part A'), ('| 1 | 2 |', 'section Title::section Part A'),
            ('# a shell comment', 'section Title::section Part A'), ('run --x', 'section Title::section Part A'),
            ('Quoted text.', 'section Title::section Part A'), ('### Part B', 'section Title::section Part A::section Part B'),
            ('Last.', 'section Title::section Part A::section Part B')])
        self.assertTrue(all(st.kind == 'markdown' for st in chk.MarkdownFile('X.md', source).statements))

    def test_sibling_heading_replaces_the_previous_one_and_globs_select_swift_only(self):
        found = {st.text: st.symbol for st in chk.MarkdownFile('X.md', '# T\n## A\na\n## B\nb\n# U\nu\n').statements}
        self.assertEqual(found['b'], 'section T::section B')
        self.assertEqual(found['u'], 'section U')
        self.assertEqual(chk.Repo(ROOT).glob('Docs/**/*.md'), [])
        self.assertIsInstance(chk.source_file('Y.md', 'x', LEDGER_EXCLUDE), chk.MarkdownFile)
        self.assertIsInstance(chk.source_file('Y.swift', 'x', LEDGER_EXCLUDE), chk.SourceFile)


class FinalFixRound(FixtureHelpers):
    """Final fix round (V04 final review I-1, I-2, I-4, M-1, M-2) on the same temporary copy."""

    def test_rule6_fixture_table_through_markdown_and_sentence_scope(self):
        ledger = json.loads((self.root / LEDGER).read_text(encoding='utf-8'))
        rule = self.rule(ledger, 'nanoseconds?|microseconds?')
        fixtures = rule['fixtures']
        self.assertGreaterEqual(len(fixtures['fail']), 15)
        own = f'({rule["reason"]})'   # README.md is also under rule 8 (T13 RD1) and is a region (T13 RD7)
        problems = self.run_with(README, self.append(fixtures['fail']))
        self.assertEqual(len([p for p in problems if p.startswith(f'R7 {README}:') and own in p]), len(fixtures['fail']),
                         problems)
        self.assertEqual(rules(self.run_with(README, self.append(fixtures['pass']))) - {'R6'}, set())
        # The hedge must be in the claim's own sentence.
        self.assertIn('Not verified. It takes ~90 ns per call.', fixtures['fail'])
        self.assertIn('It takes ~90 ns per call (not verified in this repository).', fixtures['pass'])

    def test_uncovered_markdown_claim_in_the_guide_fails_r6_and_rule6(self):
        problems = self.run_with(GUIDE, self.append(['Batch search is 3x faster on M3.']))
        self.assertTrue(any(p.startswith(f'R6 {GUIDE}:') for p in problems), problems)
        self.assertTrue(any(p.startswith(f'R7 {GUIDE}:') for p in problems), problems)
        problems = self.run_with(GUIDE, self.append(['Batch search may be faster on M3 (not verified in this repository).']))
        self.assertEqual(rules(problems), {'R6'}, problems)

    def test_removed_guide_claims_are_tombstones(self):
        for line in ['- **Sequential access**: Prefetcher-friendly memory patterns',
                     '```bash\nswift run -c release vectorcore-bench batch --profile short\n```',
                     'Run `python scripts/compare_benchmarks.py bench.json baseline.json`.']:
            with self.subTest(line=line[:40]):
                problems = self.run_with(GUIDE, self.append([line]))
                self.assertIn('R2', rules(problems), problems)

    def test_reverted_unverified_definition_fails_r2(self):
        def revert(text):
            new = ('/// `unverified` means not established for this case ID (no applicable measurement, or a measurement that '
                   'cannot be\n')
            assert text.count(new) == 1
            start = text.index(new)
            end = text.index('/// neighbours and the pair case.\n') + len('/// neighbours and the pair case.\n')
            return text[:start] + ('/// `unverified` means the benchmark author could not establish the answer from the adjacent '
                                   'source and no\n/// measurement has resolved it yet. It is a declared gap, never a claim in '
                                   'either direction.\n') + text[end:]
        problems = self.run_with(MODEL, revert)
        self.assertIn('R2', rules(problems), problems)

    def test_rule7_stale_line_citation_fails_in_swift_and_markdown(self):
        def revert(text):
            new = 'see the doc comments of batchEuclideanSoA and batchEuclideanSquaredSoA in MixedPrecisionKernels.swift). "'
            assert text.count(new) == 1
            return text.replace(new, 'MixedPrecisionKernels.swift:3375-3388). "')
        problems = self.run_with(BOUNDARIES, revert)
        self.assertTrue(any(p.startswith(f'R7 {BOUNDARIES}:') for p in problems), problems)
        self.assertIn('R2', rules(problems), problems)   # the corrected entry keeps the old reason text absent
        problems = self.run_with(BOUNDARY_DOC, self.append(['The public `batchEuclideanSoA` (`:3367-3375`) returns sqrt.']))
        self.assertTrue(any(p.startswith(f'R7 {BOUNDARY_DOC}:') for p in problems), problems)

    def test_overclaiming_pin_comment_fails_r2(self):
        def revert(text):
            new = '/// `unverified`, and, for each resolved note, the one record set it names and its observation, against a table the\n'
            assert text.count(new) == 1
            start = text.index(new)
            end = text.index('cannot detect a record that disagrees with both.\n') + len('cannot detect a record that disagrees with both.\n')
            return text[:start] + '/// `unverified`, and the value and record named by each resolved note.\n' + text[end:]
        problems = self.run_with(BOUNDARIES, revert)
        self.assertIn('R2', rules(problems), problems)


class ClosurePolishRound(FixtureHelpers):
    """Closure polish round: rule 7 checks comments, not string literals, in every test file (commentOnlyPaths)."""

    def test_rule7_flags_a_test_comment_but_not_a_test_string_literal(self):
        problems = self.run_with(PLAIN_TEST, self.append(['// documented contract at MixedPrecisionKernels.swift:303-305']))
        self.assertTrue(any(p.startswith(f'R7 {PLAIN_TEST}:') for p in problems), problems)
        ledger = json.loads((self.root / LEDGER).read_text(encoding='utf-8'))
        literals = self.rule(ledger, '(?:swift|md|py):')['fixtures']['passLiteral']
        self.assertEqual(self.run_with(PLAIN_TEST, self.append(['let location = "' + text + '"' for text in literals])), [])

    def test_comment_only_scope_is_validated(self):
        def stale(ledger):
            self.rule(ledger, '(?:swift|md|py):')['commentOnlyPaths'] = ['Tests/NoSuchDirectory/*.swift']
        def unmatched_literal(ledger):
            self.rule(ledger, '(?:swift|md|py):')['fixtures']['passLiteral'].append('no citation here')
        def literal_without_scope(ledger):
            del self.rule(ledger, '(?:swift|md|py):')['commentOnlyPaths']
        for transform, message in ((stale, 'commentOnlyPaths entry'), (unmatched_literal, 'does not test the literal scope'),
                                   (literal_without_scope, 'passLiteral needs commentOnlyPaths')):
            with self.subTest(message=message):
                problems = self.run_with(None, ledger_transform=transform)
                self.assertTrue(any(p.startswith('R8') and message in p for p in problems), problems)

    def test_without_the_scope_the_log_location_literals_would_fail(self):
        def everything(ledger):
            rule = self.rule(ledger, '(?:swift|md|py):')
            rule['paths'] = rule['paths'] + rule.pop('commentOnlyPaths')
            del rule['fixtures']['passLiteral']
        problems = self.run_with(None, ledger_transform=everything)
        logging = [p for p in problems if p.startswith('R7 Tests/ComprehensiveTests/LoggingContractTests.swift:')]
        self.assertEqual(len(logging), 4, problems)   # the four log-location literals (lines 81, 107 twice, 112)



class T13PhaseTwoRules(FixtureHelpers):
    """T13 phase 2A (rulings RD1-RD5 and RD8) on the same temporary copy: rule 8 (qualitative labels), the spelled-out
    forms added to rule 6, rule 9 (rule 6S: numeric performance claims in library sources), rule 10 (the zero-copy
    phrase), the tier-A vocabulary additions and rule 7's README, CHANGELOG and maintained-doc paths. A mutation is
    counted by the rule's own reason, so another rule matching the same text cannot satisfy it; G4 is shown directly:
    with the rule removed, the same mutation yields no R7 at all on that path."""

    MAINTAINED = ('Docs/API_Overview_Map.md', 'Docs/Benchmark_Measurement_Boundaries.md', 'Docs/HowTo_NearestNeighbor.md',
                  'Docs/Linear_Algebra_and_Projection.md', 'Docs/Maintaining.md', 'Docs/Memory_Alignment.md',
                  'Docs/Numerical_Behavior.md', 'Docs/Package_Boundaries.md', 'Docs/Performance_Guide.md', 'Docs/README.md',
                  'Docs/Serialization.md', 'Docs/SoA_Layout_Contract.md', 'Docs/executed-ci-contracts.md')
    LABELS = 'battle[- ]tested'                                       # a substring of rule 8 only
    RULE6S = 'slowdown|improvements?|throughput|performance|gains?'   # a substring of rule 9 only
    ZERO = 'zero[- ]?copy'                                            # rule 10
    RULE6 = 'nanoseconds?|microseconds?'                              # rule 6
    RULE7 = '(?:swift|md|py):'                                        # rule 7

    def ledger(self):
        return json.loads((self.root / LEDGER).read_text(encoding='utf-8'))

    def own(self, problems, path, key):
        marker = '(' + self.rule(self.ledger(), key)['reason'] + ')'
        return [p for p in problems if p.startswith(f'R7 {path}:') and marker in p]

    def test_rule_keys_identify_one_rule_each_and_fixtures_are_validated(self):
        ledger = self.ledger()
        self.assertEqual(len(ledger['forbiddenPhrases']), 10)
        for key in (self.LABELS, self.RULE6S, self.ZERO, self.RULE6, self.RULE7):
            self.assertEqual(sum(key in rule['pattern'] for rule in ledger['forbiddenPhrases']), 1, key)
        for key in (self.LABELS, self.RULE6S, self.ZERO):
            def unmatched(ledger, key=key):
                self.rule(ledger, key)['fixtures']['fail'][0] = 'nothing to see here'
            with self.subTest(rule=key):
                problems = self.run_with(None, ledger_transform=unmatched)
                self.assertTrue(any(p.startswith('R8') and 'does not match the pattern' in p for p in problems), problems)

    def test_c1_label_rule_fails_unhedged_labels_in_sources_and_readme(self):
        problems = self.run_with(PLAIN_SOURCE, self.append(['/// High-performance helper.']))
        self.assertEqual(len(self.own(problems, PLAIN_SOURCE, self.LABELS)), 1, problems)
        hedged = self.append(['/// High-performance helper (not verified in this repository).'])
        self.assertEqual(self.run_with(PLAIN_SOURCE, hedged), [])
        self.assertEqual(self.run_with(PLAIN_SOURCE, self.append(['/// The fast path skips the heap.'])), [])
        problems = self.run_with(README, self.append(['Search is efficient.']))
        self.assertEqual(len(self.own(problems, README, self.LABELS)), 1, problems)

    def test_c1_label_fixture_table_through_swift_comments(self):
        fixtures = self.rule(self.ledger(), self.LABELS)['fixtures']
        self.assertGreaterEqual(len(fixtures['fail']), 20)
        problems = self.run_with(PLAIN_SOURCE, self.append(['/// ' + text for text in fixtures['fail']]))
        self.assertEqual(len(self.own(problems, PLAIN_SOURCE, self.LABELS)), len(fixtures['fail']), problems)
        problems = self.run_with(PLAIN_SOURCE, self.append(['/// ' + text for text in fixtures['pass']]))
        self.assertEqual(self.own(problems, PLAIN_SOURCE, self.LABELS), [], problems)

    def test_c1_label_rule_has_no_exemption(self):
        literal = 'let s = "• Use FP16 for candidates, FP32 for queries\\n• Optimal balance of memory efficiency and accuracy"'
        problems = self.run_with(PLAIN_SOURCE, self.append([literal]))
        self.assertEqual(len(self.own(problems, PLAIN_SOURCE, self.LABELS)), 1, problems)
        # D02 (RD-R2) reworded the `.mixed` runtime text, so the rule carries no exemption: restoring the old text in
        # MixedPrecisionKernels.swift fails it there too.
        self.assertNotIn('exempt', self.rule(self.ledger(), self.LABELS))
        new = '"• Use FP16 for candidates, FP32 for queries\\n• Candidates: 2 bytes per scalar, queries: 4 (scalar payload)"'
        old = '"• Use FP16 for candidates, FP32 for queries\\n• Optimal balance of memory efficiency and accuracy"'

        def restore(text):
            assert text.count(new) == 1
            return text.replace(new, old)
        problems = self.run_with(MPK, restore)
        self.assertEqual(len(self.own(problems, MPK, self.LABELS)), 1, problems)

    def test_c2_vocabulary_additions_are_tier_a(self):
        ledger = self.ledger()
        vocabulary = [re.compile(each) for each in ledger['vocabulary']['tierA']]
        before = vocabulary[:-4]   # the four T13 RD2 additions are appended last
        for text in ('handles edge cases', 'edge-case handling', 'reliable', 'production-ready', 'copy-free',
                     'without copying', 'twice the size', '3-fold', 'several orders of magnitude'):
            with self.subTest(text=text):
                self.assertTrue(chk.tier_a(text, vocabulary))
                self.assertFalse(chk.tier_a(text, before))
        self.assertFalse(chk.tier_a('a 4x4 tile', vocabulary))

    def test_c3_rule6_spelled_forms_fail_through_readme(self):
        fixtures = self.rule(self.ledger(), self.RULE6)['fixtures']
        self.assertGreaterEqual(len(fixtures['fail']), 37)
        for form in ('twice as fast', 'three times faster', '3-fold speedup', 'threefold faster', 'an order of magnitude faster',
                     'faster by an order of magnitude', '50 percent faster', 'fifty per cent faster', 'halves the latency',
                     'doubles the throughput'):
            self.assertTrue(any(form in text for text in fixtures['fail']), form)
        problems = self.run_with(README, self.append(fixtures['fail']))
        self.assertEqual(len(self.own(problems, README, self.RULE6)), len(fixtures['fail']), problems)
        problems = self.run_with(README, self.append(fixtures['pass']))
        self.assertEqual([p for p in problems if p.startswith('R7')], [], problems)

    def test_c4_rule6s_in_library_sources(self):
        fixtures = self.rule(self.ledger(), self.RULE6S)['fixtures']
        self.assertGreaterEqual(len(fixtures['fail']), 25)
        self.assertGreaterEqual(len(fixtures['pass']), 30)
        problems = self.run_with(PLAIN_SOURCE, self.append(['/// ' + text for text in fixtures['fail']]))
        self.assertEqual(len(self.own(problems, PLAIN_SOURCE, self.RULE6S)), len(fixtures['fail']), problems)
        problems = self.run_with(PLAIN_SOURCE, self.append(['/// ' + text for text in fixtures['pass']]))
        self.assertEqual(self.own(problems, PLAIN_SOURCE, self.RULE6S), [], problems)
        problems = self.run_with(PLAIN_SOURCE, self.append(['/// ~75ns per call']))
        self.assertEqual(len(self.own(problems, PLAIN_SOURCE, self.RULE6S)), 1, problems)
        self.assertEqual(self.run_with(PLAIN_SOURCE, self.append(['/// Process 128 SIMD4 lanes with 2× unrolling'])), [])

    def test_c6_zero_copy_is_forbidden_in_comments_literals_and_docs(self):
        problems = self.run_with(PLAIN_SOURCE, self.append(['let m = "zero-copy"', '/// zero-copy view']))
        self.assertEqual(len(self.own(problems, PLAIN_SOURCE, self.ZERO)), 2, problems)
        for path in (README, 'Docs/Maintaining.md', 'Docs/Package_Boundaries.md'):
            with self.subTest(path=path):
                problems = self.run_with(path, self.append(['A zero-copy handoff.']))
                self.assertEqual(len(self.own(problems, path, self.ZERO)), 1, problems)

    def test_c13_rule7_covers_readme_changelog_and_maintained_docs(self):
        rule = self.rule(self.ledger(), self.RULE7)
        self.assertTrue({README, 'CHANGELOG.md', *self.MAINTAINED} <= set(rule['paths']), rule['paths'])
        problems = self.run_with('CHANGELOG.md', self.append(['- see `VectorProtocol.swift:520`']))
        self.assertEqual(len(self.own(problems, 'CHANGELOG.md', self.RULE7)), 1, problems)
        problems = self.run_with('Docs/HowTo_NearestNeighbor.md', self.append(['See Operations.swift:317 for the gate.']))
        self.assertEqual(len(self.own(problems, 'Docs/HowTo_NearestNeighbor.md', self.RULE7)), 1, problems)

    def test_g4_each_new_rule_is_what_fails_its_mutation(self):
        for key, path, line in ((self.LABELS, PLAIN_SOURCE, '/// High-performance helper.'),
                                (self.RULE6S, PLAIN_SOURCE, '/// ~75ns per call'),
                                (self.ZERO, PLAIN_SOURCE, '/// zero-copy view'),
                                (self.RULE6, README, 'Prepared candidates take 30 percent less time per query.')):
            def drop(ledger, key=key):
                ledger['forbiddenPhrases'] = [rule for rule in ledger['forbiddenPhrases'] if key not in rule['pattern']]
            with self.subTest(rule=key):
                problems = self.run_with(path, self.append([line]))
                self.assertEqual(len(self.own(problems, path, key)), 1, problems)
                problems = self.run_with(path, self.append([line]), ledger_transform=drop)
                self.assertEqual([p for p in problems if p.startswith(f'R7 {path}:')], [], problems)



class T13FinalFix(FixtureHelpers):
    """T13 final fix: Unicode hyphen folding (A-M5), the sentence boundary after closing characters (A-M1) and the
    tightened `no ... guarantee` hedge (A-M2). The per-rule fixture tables carry the rule-level cases; these tests show
    the matching text, the sentence split and the end-to-end R7 result directly."""

    HYPHENS = ('\u2010', '\u2011', '\u2012', '\u2013', '\u2212', '\u00ad')
    LABELS = 'battle[- ]tested'

    def test_matching_text_folds_unicode_hyphens_and_fingerprints_stay_raw(self):
        for char in self.HYPHENS:
            with self.subTest(char=hex(ord(char))):
                raw = chk.normalize('A zero' + char + 'cost abstraction')
                self.assertEqual(chk.neutralize(raw), 'a zero-cost abstraction')
                self.assertNotEqual(chk.fingerprint(raw), chk.fingerprint('a zero-cost abstraction'))
                vocabulary = [re.compile(each) for each in json.loads((ROOT / LEDGER).read_text(encoding='utf-8'))
                              ['vocabulary']['tierA']]
                self.assertTrue(chk.tier_a(chk.neutralize(raw), vocabulary))
                self.assertFalse(chk.tier_a(raw, vocabulary))   # unfolded, the vocabulary does not see it

    def test_sentence_split_after_closing_characters(self):
        for text in ('**Not verified.** X is fast.', '(Not verified.) X is fast.', '"Not verified." X is fast.',
                     'Not verified.\u2019 X is fast.'):
            with self.subTest(text=text):
                self.assertEqual(len(chk._sentences('// ' + text)), 2, text)
        self.assertEqual(len(chk._sentences('// It is not verified (see x.y) that X is fast.')), 1)   # a mid-sentence period

    def test_end_to_end_closing_characters_hyphens_and_guarantee(self):
        failing = ('**Not verified.** The kernel is fast.', '(Not verified.) The kernel is efficient.',
                   'Robust, with no compromises, guaranteed.', 'A thread\u2011safe pool.', 'High\u2011performance search.',
                   'A zero\u2011copy view.', 'Enables zero-copying import.', '**Not verified.** It takes ~90 ns per call.',
                   'A 3\u2011fold speedup.')
        for text in failing:
            with self.subTest(text=text):
                problems = self.run_with(PLAIN_SOURCE, self.append(['/// ' + text]))
                self.assertIn('R7', rules(problems), problems)
        for text in ('It is **not verified** that the kernel is fast.', 'There is no guarantee that the split is optimal.',
                     'There are no guarantees that the split is optimal.'):
            with self.subTest(text=text):
                problems = self.run_with(PLAIN_SOURCE, self.append(['/// ' + text]))
                self.assertNotIn('R7', rules(problems), problems)


class T13PhaseTwoRegions(FixtureHelpers):
    """T13 phase 2B (rulings RD6, RD7, RD9, RD10, RD11): the new whole-file regions, the CHANGELOG `[Unreleased]`
    section scope, the `contracts` table, the dated note on the 2026-09-05 verification record and the routed items.
    Checks on the live tree use ROOT; mutations use the same temporary copy as the other classes."""

    OPS = 'Sources/VectorCore/Operations/Operations.swift'
    V768 = 'Sources/VectorCore/Vectors/Vector768Optimized.swift'
    HOWTO = 'Docs/HowTo_NearestNeighbor.md'
    MAP = 'Docs/API_Overview_Map.md'
    CHANGELOG = 'CHANGELOG.md'
    UNRELEASED = 'section Changelog::section [Unreleased]'
    VERIFY = 'Docs/verification-topk-nan-contract-2026-09-05.md'
    VERIFY_SHA = '1de36e84b9dfbfc84152835dc179f5148dc6507c1a77ebf23d3198ba59055962'   # the record as committed (7d8fc6b)
    MAP_SWIFT_SHA = 'bc04568defb9782c8fffda3c7072aa30861e274f4731e883f0ff6beb4006e8a2'  # its Swift block at 7d8fc6b
    REGIONS = (README, OPS, HOWTO, MAP, 'Sources/VectorCore/Vectors/Vector384Optimized.swift', V768,
               'Sources/VectorCore/Vectors/Vector1536Optimized.swift')

    @staticmethod
    def live():
        return json.loads((ROOT / LEDGER).read_text(encoding='utf-8'))

    def test_c5_c9_c14_new_regions_and_the_changelog_scope_are_live(self):
        ledger = self.live()
        regions = {region['path'] for region in ledger['regions']}
        for path in self.REGIONS:
            self.assertIn(path, regions)
        self.assertEqual([(item['path'], item['sections']) for item in ledger['sectionRegions']],
                         [(self.CHANGELOG, [self.UNRELEASED])])
        self.assertIn(self.CHANGELOG, {path for rule in ledger['forbiddenPhrases'] for path in rule['paths']})  # copied

    def test_section_scope_has_exactly_one_entry_per_tier_a_statement(self):
        """Independent recount for the section region (the whole-file recount in LiveLedger covers `regions`)."""
        repo = chk.Repo(ROOT)
        ledger, vocabulary, exclude = chk.load(repo)
        groups = {}
        for st in chk.parse(repo, self.CHANGELOG, exclude).statements:
            inside = st.symbol == self.UNRELEASED or st.symbol.startswith(self.UNRELEASED + '::')
            if inside and chk.tier_a(st.match, vocabulary):
                groups[(st.symbol, st.fp)] = groups.get((st.symbol, st.fp), 0) + 1
        covered = {}
        for entry in ledger['entries']:
            if entry['path'] == self.CHANGELOG and entry['disposition'] != 'removed':
                key = (entry['symbol'], entry['fingerprint'])
                covered[key] = covered.get(key, 0) + entry.get('occurrences', 1)
        self.assertEqual(covered, groups)
        self.assertGreaterEqual(len(groups), 42)

    def test_c5_readme_claims_are_uncovered_and_labels_forbidden(self):
        problems = self.run_with(README, self.append(['Search is efficient.']))
        self.assertTrue(any(p.startswith(f'R6 {README}:') for p in problems), problems)
        self.assertTrue(any(p.startswith(f'R7 {README}:') for p in problems), problems)

    def test_c9_new_claim_in_operations_fails_r6_only(self):
        problems = self.run_with(self.OPS, self.append(['/// Avoids one copy per call.']))
        self.assertEqual(rules(problems), {'R6'}, problems)

    def test_c14_removed_vector_figures_are_tombstones(self):
        problems = self.run_with(self.V768, self.append(['/// - Dot product: ~150ns (4x unrolled with multiple accumulators)']))
        self.assertIn('R2', rules(problems), problems)
        self.assertTrue(any('tombstoned phrase' in p for p in problems), problems)


    def test_c8_reverted_doc_rows_fail_r2_and_dropped_rows_fail_r1(self):
        old = ('- Top‑K ties now resolve deterministically via `TieBreaker` in `TopKSelection` (default `.smallerIndex`; '
               'also `.insertionOrder`, `.smallerValue`), so equal‑distance results are reproducible run‑to‑run.')

        def revert(text):
            lines = text.split('\n')
            index = [i for i, line in enumerate(lines) if line.startswith('- `TopKSelection` takes a `TieBreaker`')]
            assert len(index) == 1
            lines[index[0]] = old
            return '\n'.join(lines)
        problems = self.run_with(self.HOWTO, revert)
        self.assertTrue({'R1', 'R2'} <= rules(problems), problems)

        def drop_row(text):
            lines = [line for line in text.split('\n') if not line.startswith('| Search with a closure metric |')]
            return '\n'.join(lines)
        problems = self.run_with(self.MAP, drop_row)
        self.assertEqual(rules(problems), {'R1'}, problems)

    def test_c8b_api_map_swift_block_is_unchanged(self):
        blocks = re.findall(r'```swift\n(.*?)```', (ROOT / self.MAP).read_text(encoding='utf-8'), re.S)
        self.assertEqual([hashlib.sha256(block.encode('utf-8')).hexdigest() for block in blocks], [self.MAP_SWIFT_SHA])

    def test_c9_changelog_scope_is_the_unreleased_section(self):
        anchor = '### Documentation\n\n- Refresh the README and core references'
        bullet = '- Search is faster on every host.\n'

        def unreleased(text):
            assert text.count(anchor) == 1
            return text.replace(anchor, anchor.replace('\n\n- Refresh', '\n\n' + bullet + '- Refresh'))
        problems = self.run_with(self.CHANGELOG, unreleased)
        self.assertEqual(rules(problems), {'R6'}, problems)
        self.assertEqual(self.run_with(self.CHANGELOG, self.append([bullet.strip()])), [])   # a released section

        def misspelled(ledger):
            ledger['sectionRegions'][0]['sections'] = ['section Changelog::section [Unrelesed]']
        problems = self.run_with(None, ledger_transform=misspelled)
        self.assertTrue(any(p.startswith('R8 sectionRegions') and 'stale scope' in p for p in problems), problems)

        def outside(ledger):
            entry = next(e for e in ledger['entries'] if e['path'] == self.CHANGELOG)
            entry['symbol'] = 'section Changelog::section [0.3.3] - 2026-09-06::section Fixed'
        problems = self.run_with(None, ledger_transform=outside)
        self.assertTrue(any(p.startswith('R8') and 'outside the sections' in p for p in problems), problems)

    def test_c10_superseded_note_leaves_the_record_byte_identical(self):
        lines = (ROOT / self.VERIFY).read_text(encoding='utf-8').split('\n')
        self.assertEqual(lines[:2], ['# Top-K NaN contract verification — 2026-09-05', ''])
        end = 2
        while lines[end].startswith('>'):
            end += 1
        note = ' '.join(lines[2:end])
        self.assertTrue(note.startswith('> **Superseded in part — 2026-10-06') and 'a5b2189' in note and 'P01' in note, note)
        self.assertEqual(lines[end], '')
        record = '\n'.join(lines[:2] + lines[end + 1:])
        self.assertEqual(hashlib.sha256(record.encode('utf-8')).hexdigest(), self.VERIFY_SHA)

    def test_c11_accepted_contracts_resolve(self):
        live = self.live()
        used = {part for e in live['entries'] if e['acceptedContract'] for part in e['acceptedContract'].split('/')}
        self.assertEqual(used, set(live['contracts']))

        def unknown(ledger):
            next(e for e in ledger['entries'] if e['id'] == 'V04-C-0500')['acceptedContract'] = 'P01-Q99'
        def stale(ledger):
            ledger['contracts']['X99'] = {'description': 'probe', 'source': 'probe'}
        def sourceless(ledger):
            ledger['contracts']['Q11']['source'] = ''
        for transform, message in ((unknown, "acceptedContract 'P01-Q99' is not in contracts"),
                                   (stale, 'contracts X99: no entry names it'), (sourceless, 'contracts Q11: needs')):
            with self.subTest(message=message):
                problems = self.run_with(None, ledger_transform=transform)
                self.assertTrue(any(p.startswith('R8') and message in p for p in problems), problems)

    def test_r8_malformed_whole_file_and_empty_contract_branches(self):
        def malformed(ledger):
            ledger['sectionRegions'].append({'path': self.CHANGELOG, 'sections': []})
        def whole_file(ledger):
            ledger['sectionRegions'].append({'path': README, 'sections': ['section README'], 'reason': 'probe'})
        def empty_contract(ledger):
            next(e for e in ledger['entries'] if e['id'] == 'V04-C-0500')['acceptedContract'] = ''
        for transform, message in ((malformed, 'needs a path, a non-empty list of `section ...` symbols and a reason'),
                                   (whole_file, f'sectionRegions {README}: already a region'),
                                   (empty_contract, "acceptedContract '' is not a contract ID")):
            with self.subTest(message=message):
                problems = self.run_with(None, ledger_transform=transform)
                self.assertTrue(any(p.startswith('R8') and message in p for p in problems), problems)

    def test_c12_routed_follow_ups_are_recorded(self):
        routed = self.live()['routed']
        cosine = [r for r in routed if r['path'] == 'Sources/VectorCore/Protocols/VectorProtocol.swift']
        self.assertTrue(any(r['owner'] == 'D02' and 'cosineDistance(to:)' in r['item'] for r in cosine), cosine)
        memory = [r for r in routed if r['path'] == 'Sources/VectorCore/Utilities/SIMDMemoryUtilities.swift']
        self.assertTrue(any(r['owner'] == 'D02' for r in memory), memory)


    def test_release_step_in_maintaining_retires_the_unreleased_entries(self):
        def release(text):
            assert text.count('## [Unreleased]\n') == 1
            return text.replace('## [Unreleased]\n', '## [Unreleased]\n\n## [0.3.4] - 2026-10-07\n')

        def retire(ledger):
            ledger['entries'] = [e for e in ledger['entries'] if e['path'] != self.CHANGELOG]
        problems = self.run_with(self.CHANGELOG, release)
        self.assertIn('R1', rules(problems), problems)   # the step is required: the old anchors go stale
        self.assertEqual(self.run_with(self.CHANGELOG, release, ledger_transform=retire), [])


if __name__ == '__main__':
    unittest.main()
