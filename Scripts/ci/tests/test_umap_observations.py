"""Numerical budgets are chosen before CI measurements, never widened by them."""
from copy import deepcopy
from pathlib import Path
import struct
import sys
import unittest

CI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CI))


def word(value):
    return struct.unpack('>I', struct.pack('>f', value))[0]


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((CI / 'observe_umap_reference.py').exists(), 'tracked observation comparator is missing')
        self.module = __import__('observe_umap_reference')
        self.profile = {'cases': {'case': {'reference': 'gold', 'tolerance': 1e-4}},
                        'words': {'gold': [word(1), word(-2)]}, 'rowOffsets': [0, 1, 2],
                        'neighborIndices': [1, 0]}
        self.capture = {'cases': {'case': [word(1), word(-2)]},
                        'rowOffsets': [0, 1, 2], 'neighborIndices': [1, 0]}

    def compare(self, candidate=None, reference=None):
        return self.module.compare(candidate or self.capture, reference or self.capture, self.profile)

    def test_zero_deviation_reports_all_cases(self):
        report = self.compare()
        self.assertTrue(report['passed'])
        self.assertEqual(report['cases']['case']['historical']['maxAbsolute'], 0)
        self.assertEqual(report['cases']['case']['sameRunner']['maxAbsolute'], 0)

    def test_sub_budget_difference_reports_actual_maximum(self):
        changed = deepcopy(self.capture)
        changed['cases']['case'][1] = word(-2 + 0.00005)
        report = self.compare(changed)
        self.assertTrue(report['passed'])
        deviation = report['cases']['case']['historical']
        self.assertGreater(deviation['maxAbsolute'], 0.000049)
        self.assertLess(deviation['maxAbsolute'], 0.000051)
        self.assertEqual(deviation['worstIndex'], 1)
        self.assertGreater(deviation['maxULP'], 0)

    def test_route_miss_fails_and_retains_measured_difference(self):
        changed = deepcopy(self.capture)
        changed['cases']['case'][0] = word(1.001)
        report = self.compare(changed)
        self.assertFalse(report['passed'])
        self.assertGreater(report['cases']['case']['historical']['maxAbsolute'], 1e-4)

    def test_missing_or_extra_cases_are_not_silently_omitted(self):
        for candidate in ({'cases': {}, 'rowOffsets': [0, 1, 2], 'neighborIndices': [1, 0]},
                          dict(self.capture, cases=dict(self.capture['cases'], extra=[word(0)]))):
            with self.assertRaisesRegex(ValueError, 'inventory'):
                self.compare(candidate)

    def test_shape_change_is_failure(self):
        changed = deepcopy(self.capture)
        changed['cases']['case'].pop()
        with self.assertRaisesRegex(ValueError, 'shape'):
            self.compare(changed)

    def test_nonfinite_or_invalid_words_are_rejected(self):
        for value in [0x7f800000, 0x7fc00000, -1, 2**32, True]:
            changed = deepcopy(self.capture)
            changed['cases']['case'][0] = value
            with self.assertRaises(ValueError):
                self.compare(changed)

    def test_discrete_identity_is_required(self):
        for field in ['rowOffsets', 'neighborIndices']:
            changed = deepcopy(self.capture)
            changed[field][0] += 1
            self.assertFalse(self.compare(changed)['passed'])

    def test_same_runner_control_distinguishes_environment_from_drift(self):
        changed = deepcopy(self.capture)
        changed['cases']['case'][0] = word(1.001)
        report = self.compare(changed, changed)
        self.assertFalse(report['passed'])  # historical ceiling still binds
        self.assertEqual(report['cases']['case']['sameRunner']['maxAbsolute'], 0)

    def test_same_runner_regression_cannot_hide_inside_historical_budget(self):
        control = deepcopy(self.capture)
        control['cases']['case'][0] = word(1.001)
        self.assertFalse(self.compare(reference=control)['passed'])


if __name__ == '__main__':
    unittest.main()
