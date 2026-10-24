"""Mutate explicit maintained policy inputs; do not infer arbitrary Swift dataflow."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('performance_policy', ROOT / 'Scripts/ci/check_performance_policy.py')
POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)


class PerformancePolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='performance-policy-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inventory = json.loads((ROOT / 'Scripts/ci/performance_sites.json').read_text())
        paths = {row['file'] for key in ['assertions', 'measurements'] for row in self.inventory[key]}
        paths |= {'Scripts/ci/performance_sites.json', 'Tests/ComprehensiveTests/PerfGate.swift',
                  'Tests/MinimalTests/PerformanceObservation.swift'}
        for relative in paths:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)

    def test_complete_repository_inventory(self):
        result = POLICY.validate(ROOT)
        self.assertEqual(result['assertionSites'], 75)  # 71 existing + two deliberate gate probes + T06 NEON and quantization-overhead.
        self.assertEqual(result['observationSites'], 30)  # 29 existing + one observation probe.

    def test_every_inventoried_family_rejects_removed_or_direct_assertion(self):
        for row in self.inventory['assertions']:
            path = self.root / row['file']
            original = path.read_text()
            helper = 'expectXCTestPerformance' if row['framework'] == 'XCTest' else 'expectPerformance'
            token = helper + '("' + row['id'] + '", '
            self.assertIn(token, original)
            for replacement in ['removedPerformanceCall(', '#expect(']:
                with self.subTest(site=row['id'], replacement=replacement):
                    path.write_text(original.replace(token, replacement))
                    with self.assertRaises(ValueError):
                        POLICY.validate(self.root)
            path.write_text(original)

    def test_unknown_duplicate_and_pre_evaluated_sites_rejected(self):
        row = self.inventory['assertions'][0]
        path = self.root / row['file']
        original = path.read_text()
        for changed in [original.replace(row['id'], 'unknown.site', 1),
                        original + '\nexpectPerformance("' + row['id'] + '", ' + row['expression'] + ')\n',
                        original.replace(row['expression'], 'precomputedBoolean', 1)]:
            path.write_text(changed)
            with self.assertRaises(ValueError):
                POLICY.validate(self.root)
        path.write_text(original)

    def test_gate_and_lazy_enforcement_mutations(self):
        path = self.root / 'Tests/ComprehensiveTests/PerfGate.swift'
        original = path.read_text()
        mutations = [('guard strictPerfGatesEnabled else { return }', ''),
                     ('@autoclosure () -> Bool', 'Bool'),
                     ('== "1"', '!= "0"'),
                     ('guard strictPerfGatesEnabled', '_ = condition()\n    guard strictPerfGatesEnabled'),
                     ('Issue.record(', 'print('), ('XCTFail(', 'print(')]
        for before, after in mutations:
            with self.subTest(before=before):
                path.write_text(original.replace(before, after))
                with self.assertRaises(ValueError):
                    POLICY.validate(self.root)
        path.write_text(original)

    def test_measurement_enumeration_and_wrapper_mutations(self):
        for row in self.inventory['measurements']:
            path = self.root / row['file']
            original = path.read_text()
            path.write_text(original.replace('observePerformance("' + row['id'] + '")', 'measure'))
            with self.subTest(site=row['id']), self.assertRaises(ValueError):
                POLICY.validate(self.root)
            path.write_text(original)
        new = self.root / 'Tests/MinimalTests/IntroducedTests.swift'
        new.write_text('func testNew() {\n    measure { print("new") }\n}\n')
        with self.assertRaises(ValueError):
            POLICY.validate(self.root)
        new.unlink()
        path = self.root / 'Tests/MinimalTests/PerformanceObservation.swift'
        original = path.read_text()
        for changed in [original.replace('== "1"', '!= "0"'), original.replace('block()', 'block(); block()')]:
            path.write_text(changed)
            with self.assertRaises(ValueError):
                POLICY.validate(self.root)

    def test_indirect_boolean_inventory_cannot_bless_eager_reduction(self):
        inventory_path = self.root / 'Scripts/ci/performance_sites.json'
        original_inventory = inventory_path.read_text()
        for site in ['BatchKernels_SoATests.testBlockingEfficiency.4', 'BatchKernels_SoATests.testScalingWithBatchSize.13']:
            data = json.loads(original_inventory)
            row = next(row for row in data['assertions'] if row['id'] == site)
            path = self.root / row['file']
            original = path.read_text()
            self.assertIn(row['expression'], original)
            path.write_text(original.replace(row['expression'], 'eagerTimingBoolean', 1))
            # Even changing the maintained row must not permit a pre-evaluated Boolean.
            row['expression'] = 'eagerTimingBoolean'
            inventory_path.write_text(json.dumps(data))
            with self.subTest(site=site), self.assertRaisesRegex(ValueError, 'precomputed timing Boolean'):
                POLICY.validate(self.root)
            path.write_text(original)
        inventory_path.write_text(original_inventory)

    def test_clock_derived_entry_requires_leading_strict_gate(self):
        row = next(row for row in self.inventory['assertions']
                   if row['id'] == 'MixedPrecisionKernelTests.testThroughputScaling.5')
        path = self.root / row['file']
        original = path.read_text()
        condition = row['entryCondition']
        self.assertIn('if ' + condition + ' {', original)
        for replacement in ['previousThroughput > 0 && batchSize <= 100',
                            'previousThroughput > 0 && strictPerfGatesEnabled && batchSize <= 100']:
            with self.subTest(condition=replacement):
                path.write_text(original.replace(condition, replacement, 1))
                with self.assertRaisesRegex(ValueError, 'gate strict mode first'):
                    POLICY.validate(self.root)
        path.write_text(original)
