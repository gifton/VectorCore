import importlib.util
import json
from pathlib import Path
import unittest

MODULE = Path(__file__).with_name('scope_check.py')
spec = importlib.util.spec_from_file_location('scope_check', MODULE)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def item(record_kind, **payload):
    return json.dumps({'version': 0, 'kind': record_kind, 'payload': payload})


def stream(*events):
    return '\n'.join([item('test', id='M.S/f()/F.swift:1:1', kind='function'),
                      item('event', kind='runStarted'), *events])


class ScopeControls(unittest.TestCase):
    def test_swift_testing_requires_actual_start_end_and_complete_run(self):
        body = [item('event', kind='testStarted', testID='M.S/f()/F.swift:1:1'),
                item('event', kind='testEnded', testID='M.S/f()/F.swift:1:1',
                     messages=[{'symbol': 'pass', 'text': 'Test passed.'}]),
                item('event', kind='runEnded')]
        self.assertTrue(checker.swift_testing(stream(*body))['accepted'])
        for bad in ['', stream(), stream(*body[:-1]), stream(*body[1:]), stream(*body)+'\n{']:
            with self.subTest(bad=bad):
                self.assertFalse(checker.swift_testing(bad)['accepted'])

    def test_swift_testing_failed_skipped_and_unknown_body_never_pass(self):
        for event, expected in [('testSkipped', 'skipped'), ('issueRecorded', 'failed')]:
            text = stream(item('event', kind='testStarted', testID='M.S/f()/F.swift:1:1'),
                          item('event', kind=event, testID='M.S/f()/F.swift:1:1'),
                          item('event', kind='testEnded', testID='M.S/f()/F.swift:1:1', messages=[{'symbol':'pass'}]),
                          item('event', kind='runEnded'))
            result = checker.swift_testing(text)
            self.assertFalse(result['accepted'])
            self.assertEqual(len(result[expected]), 1)
        result=checker.swift_testing(stream(item('event',kind='testStarted',testID='unknown'),item('event',kind='runEnded')))
        self.assertFalse(result['accepted'])
        self.assertIn('unknown',result['unknown'])

    def test_xctest_rejects_missing_truncated_failed_skipped_and_count_mismatch(self):
        begin="Test Case '-[M.S f]' started.\n"
        end="Test Case '-[M.S f]' passed (0.01 seconds).\n"
        summary="Test Suite 'All tests' passed at date.\n Executed 1 test, with 0 failures (0 unexpected) in 0.01 seconds\n"
        self.assertTrue(checker.xctest(begin+end+summary,['M.S/f'])['accepted'])
        for bad in ['',begin, end+summary, begin+end, begin+end+summary.replace('1 test','2 tests'),
                    begin+end.replace('passed','failed')+summary,
                    begin+end.replace('passed','skipped')+summary.replace('0 failures','1 test skipped and 0 failures')]:
            with self.subTest(bad=bad):self.assertFalse(checker.xctest(bad,['M.S/f'])['accepted'])

    def test_independent_discovery_catches_missing_function_inventory(self):
        text = stream(item('event', kind='testStarted', testID='M.S/f()/F.swift:1:1'),
                      item('event', kind='testEnded', testID='M.S/f()/F.swift:1:1', messages=[{'symbol':'pass'}]),
                      item('event', kind='runEnded'))
        self.assertFalse(checker.swift_testing(text, ['M.S/f()', 'M.S/missing()'])['accepted'])
        self.assertTrue(checker.swift_testing(text, ['M.S/f()'])['accepted'])

    def test_interrupted_failing_function_stays_pending_until_end(self):
        text = stream(item('event', kind='testStarted', testID='M.S/f()/F.swift:1:1'),
                      item('event', kind='issueRecorded', testID='M.S/f()/F.swift:1:1'))
        result = checker.swift_testing(text)
        self.assertEqual(result['failed'], [])
        self.assertEqual(result['pending'], ['M.S/f()/F.swift:1:1'])
        self.assertEqual(result['interrupted'], ['M.S/f()/F.swift:1:1'])
        self.assertFalse(result['accepted'])

    def test_suite_skip_retains_all_descendant_functions(self):
        text = '\n'.join([item('test', id='M.S', kind='suite'),
                          item('test', id='M.S/f()/F.swift:1:1', kind='function'),
                          item('event', kind='runStarted'),
                          item('event', kind='testSkipped', testID='M.S'),
                          item('event', kind='runEnded')])
        result = checker.swift_testing(text)
        self.assertEqual(result['skipped'], ['M.S/f()/F.swift:1:1'])
        self.assertEqual(result['pending'], [])
        self.assertFalse(result['accepted'])

    def test_selected_xctest_terminal_label_still_requires_all_discovery(self):
        text = "Test Case '-[M.S f]' started.\nTest Case '-[M.S f]' passed (0.01 seconds).\n" + \
               "Test Suite 'Selected tests' passed at date.\n Executed 1 test, with 0 failures (0 unexpected) in 0.01 seconds\n"
        self.assertTrue(checker.xctest(text, ['M.S/f'])['accepted'])
        self.assertFalse(checker.xctest(text, ['M.S/f', 'M.S/missing'])['accepted'])
        self.assertFalse(checker.xctest(text, [])['accepted'])

    def test_parent_images_reject_child_only_and_printed_path_controls(self):
        paths = ['/host', '/runtime', '/bundle']
        lines = '\n'.join('dyld[42]: <uuid> ' + p for p in paths)
        self.assertTrue(checker.images_verified(lines, 42, paths))
        for bad in [lines.replace('[42]', '[43]'), '\n'.join(paths), lines.splitlines()[0]]:
            self.assertFalse(checker.images_verified(bad, 42, paths))
        self.assertFalse(checker.images_verified(lines, 42, []))

    def test_empty_discovery_and_nonzero_exit_never_qualify(self):
        self.assertFalse(checker.xctest('',[])['accepted'])
        self.assertFalse(checker.qualifies({'accepted':True},{'exit':1,'timeout':False}))
        self.assertFalse(checker.qualifies({'accepted':True},{'exit':0,'timeout':True}))
        self.assertFalse(checker.qualifies({'accepted':True},{}))
        self.assertTrue(checker.qualifies({'accepted':True},{'exit':0,'timeout':False,'status':'completed'}))

if __name__ == '__main__': unittest.main()
