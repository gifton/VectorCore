"""The external consumer judge must reject stale values and incomplete executions."""
import importlib.util
from pathlib import Path
import sys
import unittest

CI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CI))
SPEC = importlib.util.spec_from_file_location('buffer_value_consumer', CI / 'consumer_buffer_value.py')
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class ConsumerObservationTests(unittest.TestCase):
    expected = {'singleton': '3f800000', 'empty': '0,1,731'}
    good = ('BUFFER_VALUE singleton observed=3f800000\n'
            'BUFFER_VALUE empty observed=0,1,731\n'
            'BUFFER_VALUE_COMPLETE count=2 failures=0\n')
    stale = good.replace('observed=3f800000', 'observed=40000000').replace('failures=0', 'failures=1')

    def judge(self, code, output, **kwargs):
        return DRIVER.judge(code, output, self.expected, **kwargs)

    def test_complete_correct_observations_pass(self):
        self.assertTrue(self.judge(0, self.good)['passed'])

    def test_public_fixture_dotted_case_identifiers(self):
        text = self.good.replace('singleton', 'concrete.float.singleton')
        expected = {'concrete.float.singleton': '3f800000', 'empty': '0,1,731'}
        self.assertTrue(DRIVER.judge(0, text, expected)['passed'])

    def test_zero_exit_does_not_hide_a_stale_value(self):
        self.assertFalse(self.judge(0, self.stale)['passed'])
        # A falsely optimistic completion marker must also be rejected.
        self.assertFalse(self.judge(0, self.stale.replace('failures=1', 'failures=0'))['passed'])

    def test_missing_duplicate_extra_or_malformed_rows_fail(self):
        first = self.good.splitlines(keepends=True)[0]
        for text in [self.good.replace(first, ''), first + self.good,
                     self.good.replace('empty observed=', 'unregistered observed='),
                     self.good.replace('observed=', 'got=', 1),
                     self.good.replace('count=2', 'count=1'), self.good + 'extra\n', '',
                     self.good.replace('BUFFER_VALUE_COMPLETE count=2 failures=0\n', '')]:
            with self.subTest(text=text):
                self.assertFalse(self.judge(0, text)['passed'])

    def test_process_failure_cannot_pass_positive(self):
        self.assertFalse(self.judge(1, self.good)['passed'])

    def test_omitted_write_negative_requires_the_exact_stale_witness(self):
        witness = {'singleton': '40000000', 'empty': '0,1,731'}
        self.assertTrue(self.judge(1, self.stale, negative_witness=witness)['passed'])
        for code, text in [(0, self.stale), (-6, self.stale), (1, ''), (1, self.good),
                           (1, self.stale.replace('40000000', '0')),
                           (1, self.stale.replace('0,1,731', 'garbage').replace('failures=1', 'failures=2')),
                           (1, self.stale.replace('failures=1', 'failures=2'))]:
            with self.subTest(code=code, text=text):
                self.assertFalse(self.judge(code, text, negative_witness=witness)['passed'])

    def test_partial_negative_oracle_is_rejected(self):
        self.assertFalse(self.judge(1, self.stale, negative_witness={'singleton': '40000000'})['passed'])


if __name__ == '__main__':
    unittest.main()
