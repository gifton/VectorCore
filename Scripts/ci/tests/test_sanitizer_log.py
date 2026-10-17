"""verify_body accepts both Swift Testing log renderings and rejects mismatches."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_sanitizers import verify_body

TAIL = '\nTest run with 1 test in 1 suite passed after 0.1 seconds.\n'


class VerifyBodyTests(unittest.TestCase):
    function = {'id': 'S/testF', 'displayName': 'testF()'}
    custom = {'id': 'S/testF', 'displayName': 'custom name'}

    def test_function_name_rendering(self):
        verify_body(self.function, 'Test testF() started.\nTest testF() passed after 1s' + TAIL)

    def test_custom_name_old_and_aka_renderings(self):
        verify_body(self.custom, 'Test "custom name" started.\nTest "custom name" passed after 1s' + TAIL)
        verify_body(self.custom, "Test \"custom name\" (aka 'testF()') started.\n"
                    "Test \"custom name\" (aka 'testF()') passed after 1s" + TAIL)

    def test_wrong_aka_missing_pass_and_skips_are_rejected(self):
        for log in ["Test \"custom name\" (aka 'other()') started.\nTest \"custom name\" (aka 'other()') passed" + TAIL,
                    'Test "custom name" started.' + TAIL,
                    'Test "custom name" started.\nTest "custom name" passed' + TAIL + ' skipped',
                    'Test "custom name" started.\nTest "custom name" passed\nTest run with 2 tests in 1 suite passed']:
            with self.subTest(log=log), self.assertRaises(ValueError):
                verify_body(self.custom, log)

    def test_swift_6_0_run_summary(self):
        # Swift 6.0 (Xcode 16.2) omits "in N suites"; run summaries in the PR #56 CI run 3 rendering.
        body = 'Test testF() started.\nTest testF() passed after 1s\n'
        verify_body(self.function, body + '✔ Test run with 1 test passed after 7.797 seconds.\n')
        for tail in ['✔ Test run with 0 tests passed after 0.006 seconds.\n',
                     '✔ Test run with 2 tests passed after 0.1 seconds.\n',
                     '✘ Test run with 1 test failed after 0.013 seconds with 1 issue.\n']:
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                verify_body(self.function, body + tail)

    def test_cancelled_body_is_rejected_even_when_started(self):
        # Swift Testing 6.4 reports a self-cancelling test as "was cancelled"; that is never a pass.
        cancelled = ('Test "custom name" (aka \'testF()\') started.\n'
                     'Test "custom name" (aka \'testF()\') was cancelled after 0.001 seconds.\n' + TAIL)
        plain = 'Test testF() started.\nTest testF() was cancelled after 0.001 seconds.' + TAIL
        for row, log in [(self.custom, cancelled), (self.function, plain)]:
            with self.subTest(row=row['displayName']), self.assertRaises(ValueError):
                verify_body(row, log)


if __name__ == '__main__':
    unittest.main()
