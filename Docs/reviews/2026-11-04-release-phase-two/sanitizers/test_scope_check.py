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

    def test_independent_discovery_preserves_exact_empty_and_partial_scope(self):
        universe = ['M.S/f()', 'M.S/missing()']
        text = stream(item('event', kind='testStarted', testID='M.S/f()/F.swift:1:1'),
                      item('event', kind='testEnded', testID='M.S/f()/F.swift:1:1', messages=[{'symbol':'pass'}]),
                      item('event', kind='runEnded'))
        for output, passed, pending in [('', [], universe), (text, ['M.S/f()'], ['M.S/missing()'])]:
            with self.subTest(output=bool(output)):
                result = checker.swift_testing(output, universe)
                self.assertEqual(result['discovered'], universe)
                self.assertEqual(result['passed'], passed)
                self.assertEqual(result['pending'], pending)
                self.assertEqual(len(result['discovered']), 2)
                self.assertEqual(len(result['pending']), len(pending))
                self.assertFalse(result['accepted'])
        full = checker.swift_testing(text, ['M.S/f()'])
        self.assertTrue(full['accepted'])
        self.assertEqual(full['eventInventory'], ['M.S/f()/F.swift:1:1'])
        self.assertEqual(full['started'], ['M.S/f()'])

    def test_selector_collision_preserves_unfinished_event_body_as_pending(self):
        first, second = 'M.S/f()/F.swift:1:1', 'M.S/f()/F.swift:2:1'
        for state in ['not-started', 'started', 'passed']:
            with self.subTest(second=state):
                rows = [item('test', id=first, kind='function'), item('test', id=second, kind='function'),
                        item('event', kind='runStarted'), item('event', kind='testStarted', testID=first),
                        item('event', kind='testEnded', testID=first, messages=[{'symbol':'pass'}])]
                if state != 'not-started': rows.append(item('event', kind='testStarted', testID=second))
                if state == 'passed': rows.append(item('event', kind='testEnded', testID=second, messages=[{'symbol':'pass'}]))
                rows.append(item('event', kind='runEnded'))
                result = checker.swift_testing('\n'.join(rows), ['M.S/f()'])
                self.assertFalse(result['accepted'])
                self.assertEqual(result['discovered'], ['M.S/f()'])
                self.assertEqual(result['eventInventory'], [first, second])
                self.assertEqual(result['selectorCollisions'], {'M.S/f()':[first, second]})
                self.assertEqual(result['eventPending'], [] if state == 'passed' else [second])
                self.assertEqual(result['pending'], [] if state == 'passed' else ['M.S/f()'])
                self.assertEqual(result['passed'], ['M.S/f()'] if state == 'passed' else [])
                self.assertEqual(result['failed'], [])
                self.assertEqual(result['skipped'], [])

    def test_selector_collision_retains_completed_failure_and_skip_events(self):
        first, second = 'M.S/f()/A.swift:1:1', 'M.S/f()/B.swift:1:1'
        for state in ['failed', 'skipped']:
            with self.subTest(second=state):
                rows = [item('test', id=first, kind='function'), item('test', id=second, kind='function'),
                        item('event', kind='runStarted'), item('event', kind='testStarted', testID=first),
                        item('event', kind='testEnded', testID=first, messages=[{'symbol':'pass'}])]
                if state == 'failed':
                    rows += [item('event', kind='testStarted', testID=second),
                             item('event', kind='issueRecorded', testID=second),
                             item('event', kind='testEnded', testID=second, messages=[{'symbol':'fail'}])]
                else: rows.append(item('event', kind='testSkipped', testID=second))
                rows.append(item('event', kind='runEnded'))
                result = checker.swift_testing('\n'.join(rows), ['M.S/f()'])
                self.assertFalse(result['accepted'])
                self.assertEqual(result['eventPassed'], [first])
                self.assertEqual(result['event' + state.title()], [second])
                self.assertEqual(result['selectorCollisions'], {'M.S/f()':[first, second]})
                self.assertEqual(result['passed'], [])
                self.assertEqual(result[state], ['M.S/f()'])
                self.assertEqual(result['pending'], [])
                self.assertEqual(result['eventPending'], [])

    def test_unexpected_event_inventory_cannot_expand_independent_universe(self):
        result = checker.swift_testing(stream(), ['M.S/other()'])
        self.assertEqual(result['discovered'], ['M.S/other()'])
        self.assertEqual(result['pending'], ['M.S/other()'])
        self.assertEqual(result['discoveryUnexpected'], ['M.S/f()'])
        self.assertFalse(result['accepted'])

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
