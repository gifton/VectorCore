"""Mutations close unowned test-environment gates, including aliases."""
from pathlib import Path
import sys
import tempfile
import unittest

CI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CI))


class EnvironmentPolicyTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((CI / 'check_test_environment.py').exists(), 'environment policy is missing')
        self.policy = __import__('check_test_environment')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'Tests').mkdir()
        self.source = self.root / 'Tests/Fixture.swift'
        self.source.write_text('let enabled = ProcessInfo.processInfo.environment["FEATURE"] == "1"\n')
        (self.root / 'activate.py').write_text('env["FEATURE"] = "1"\n')
        (self.root / 'scope.md').write_text('FEATURE: diagnostic fixture only.\n')
        self.controls = {'FEATURE': {'kind': 'gate', 'scope': 'diagnostic fixture only',
                                     'document': 'scope.md', 'activation': 'activate.py', 'value': '1'}}

    def check(self):
        return self.policy.check(self.root, self.controls, {})

    def test_named_tracked_gate_is_accepted(self):
        self.assertEqual(self.check(), [])

    def test_unknown_gate_is_rejected(self):
        self.source.write_text(self.source.read_text().replace('FEATURE', 'NEW_GATE'))
        self.assertTrue(any('NEW_GATE' in issue for issue in self.check()))

    def test_removed_activation_is_rejected(self):
        (self.root / 'activate.py').write_text('env = {}\n')
        self.assertTrue(any('activation' in issue for issue in self.check()))

    def test_comment_or_bare_mention_is_not_activation(self):
        (self.root / 'activate.py').write_text('# env["FEATURE"] = "1"\nprint("FEATURE")\n')
        self.assertTrue(any('activation' in issue for issue in self.check()))

    def test_renaming_only_setter_is_rejected(self):
        (self.root / 'activate.py').write_text('env["RENAMED"] = "1"\n')
        self.assertTrue(any('activation' in issue for issue in self.check()))

    def test_dictionary_alias_and_constant_key_are_discovered(self):
        self.source.write_text('let key = "FEATURE"\nlet snapshot = ProcessInfo.processInfo.environment\n'
                               'let other = snapshot\nlet enabled = other[key] == "1"\n')
        self.assertEqual(self.check(), [])
        self.source.write_text(self.source.read_text().replace('FEATURE', 'NEW_GATE'))
        self.assertTrue(any('NEW_GATE' in issue for issue in self.check()))

    def test_process_info_alias_is_discovered(self):
        self.source.write_text('let info = ProcessInfo.processInfo\nlet env = info.environment\n'
                               'let enabled = env["NEW_GATE"] == "1"\n')
        self.assertTrue(any('NEW_GATE' in issue for issue in self.check()))

    def test_unsupported_dynamic_key_fails_closed(self):
        self.source.write_text('let value = ProcessInfo.processInfo.environment[makeKey()]\n')
        self.assertTrue(any('unresolved' in issue for issue in self.check()))

    def test_getenv_is_not_a_bypass(self):
        self.source.write_text('let enabled = getenv("NEW_GATE") != nil\n')
        self.assertTrue(any('NEW_GATE' in issue for issue in self.check()))

    def test_whitespace_in_process_info_access_is_not_a_bypass(self):
        self.source.write_text('let flag = ProcessInfo . processInfo . environment["NEW_GATE"] == "1"\n')
        self.assertTrue(any('NEW_GATE' in issue for issue in self.check()))

    def test_new_literal_loop_key_is_discovered(self):
        self.source.write_text('for key in ["FEATURE", "NEW_GATE"] {\n'
                               '    let value = ProcessInfo.processInfo.environment[key]\n}\n')
        self.assertTrue(any('NEW_GATE' in issue for issue in self.check()))

    def test_unrecognized_environment_receiver_fails_closed(self):
        self.source.write_text('let flag = ProcessInfo().environment["NEW_GATE"] == "1"\n')
        self.assertTrue(any('unresolved' in issue for issue in self.check()))

    def test_comments_and_literals_are_not_accesses(self):
        self.source.write_text('// ProcessInfo.processInfo.environment["NEW_GATE"]\n'
                               'let text = #"ProcessInfo.processInfo.environment["NEW_GATE"]"#\n')
        self.assertEqual(self.check(), [])

    def test_internal_selector_requires_a_real_setter(self):
        self.controls['FEATURE'] = {'kind': 'internal', 'scope': 'child selection',
                                    'owner': 'activate.py'}
        self.assertEqual(self.check(), [])
        (self.root / 'activate.py').write_text('# FEATURE is set here\n')
        self.assertTrue(any('setter' in issue for issue in self.check()))

    def test_repository_controls_have_tracked_owners(self):
        controls, special = self.policy.load_controls()
        self.assertEqual(self.policy.check(CI.parents[1], controls, special), [])


if __name__ == '__main__':
    unittest.main()
