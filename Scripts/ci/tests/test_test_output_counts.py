"""Test-run summaries parse identically across toolchains (T20 round 3, PR #56 CI run 3).

Sample lines follow the renderings in real logs: Swift 6.0 from the Xcode 16.2 CI gate artifacts,
Swift 6.4 from local Xcode 27 runs; some counts and durations are illustrative. The old
representative check is kept below as a control.
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_performance_gates import representative_executed
from swiftpm_products import swift_testing_runs, xctest_executed

ST60_PASS_MANY = '✔ Test run with 1703 tests passed after 698.441 seconds.'
ST60_PASS_ONE = '✔ Test run with 1 test passed after 7.797 seconds.'
ST60_ZERO = '✔ Test run with 0 tests passed after 0.006 seconds.'
ST60_FAIL = '✘ Test run with 1 test failed after 0.013 seconds with 1 issue.'
ST64_PASS_MANY = 'Test run with 1701 tests in 252 suites passed after 253.030 seconds.'
ST64_PASS_ONE = 'Test run with 1 test in 1 suite passed after 0.1 seconds.'
ST64_ZERO = 'Test run with 0 tests in 0 suites passed after 0.001 seconds.'
ST64_FAIL = 'Test run with 1701 tests in 252 suites failed after 394.082 seconds with 3 issues.'
XC_ONE = '\t Executed 1 test, with 0 failures (0 unexpected) in 0.103 (0.104) seconds'
XC_ONE_FAILED = '\t Executed 1 test, with 1 failure (0 unexpected) in 0.033 (0.033) seconds'
XC_ZERO = '\t Executed 0 tests, with 0 failures (0 unexpected) in 0.000 (0.000) seconds'
XC_CASE = "Test Case '-[ComprehensiveTests.KernelAutoTunerTests testProfileWorkload512]' passed (0.103 seconds)."


def joined(*lines):
    return '\n'.join(lines) + '\n'


class SwiftTestingRuns(unittest.TestCase):
    def test_both_renderings(self):
        for line, expected in [(ST60_PASS_MANY, (1703, 'passed')), (ST60_PASS_ONE, (1, 'passed')),
                               (ST60_ZERO, (0, 'passed')), (ST60_FAIL, (1, 'failed')),
                               (ST64_PASS_MANY, (1701, 'passed')), (ST64_PASS_ONE, (1, 'passed')),
                               (ST64_ZERO, (0, 'passed')), (ST64_FAIL, (1701, 'failed'))]:
            with self.subTest(line=line):
                self.assertEqual(swift_testing_runs(line), [expected])

    def test_counts_are_whole_numbers_not_prefixes(self):
        self.assertEqual(swift_testing_runs('✔ Test run with 10 tests passed after 1 seconds.'), [(10, 'passed')])
        self.assertEqual(swift_testing_runs('no summary here'), [])


class XCTestExecuted(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(xctest_executed(joined(XC_ONE, XC_ONE, XC_ONE)), 1)
        self.assertEqual(xctest_executed(XC_ONE_FAILED), 1)
        self.assertEqual(xctest_executed(XC_ZERO), 0)
        self.assertEqual(xctest_executed('\t Executed 185 tests, with 27 tests skipped and 0 failures'), 185)
        self.assertEqual(xctest_executed(''), 0)


class RepresentativeExecuted(unittest.TestCase):
    # The `measured-profile` log shape that failed on Xcode 16.2: XCTest ran the test, then the
    # Swift Testing runner reported its own empty selection.
    XCTEST_REPRESENTATIVE_60 = joined(XC_CASE, XC_ONE, XC_ONE, XC_ONE, ST60_ZERO)

    def test_xctest_representative_on_both_toolchains(self):
        self.assertTrue(representative_executed(self.XCTEST_REPRESENTATIVE_60))
        self.assertTrue(representative_executed(joined(XC_CASE, XC_ONE, ST64_ZERO)))

    def test_swift_testing_representative_on_both_toolchains(self):
        self.assertTrue(representative_executed(joined(XC_ZERO, ST60_PASS_ONE)))
        self.assertTrue(representative_executed(joined(XC_ZERO, ST64_PASS_ONE)))

    def test_nothing_executed_is_rejected_on_both_toolchains(self):
        for log in [joined(XC_ZERO, XC_ZERO, ST60_ZERO), joined(XC_ZERO, ST64_ZERO), '']:
            with self.subTest(log=log):
                self.assertFalse(representative_executed(log))

    def test_control_the_former_check_misread_the_swift_6_0_log(self):
        def former(log):
            return ('Test Case ' in log or 'Test run with 1 test' in log) and '0 tests passed' not in log
        self.assertFalse(former(self.XCTEST_REPRESENTATIVE_60))
        self.assertTrue(former(joined(XC_CASE, XC_ONE, ST64_ZERO)))


if __name__ == '__main__':
    unittest.main()
