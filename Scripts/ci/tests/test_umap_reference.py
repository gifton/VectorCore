"""Reference-runner failure controls; no Swift build in the policy suite."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

CI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CI))


class ReferenceRunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((CI / 'umap_reference.py').is_file(), 'tracked reference validator is missing')
        self.module = __import__('umap_reference')
        self.profile = self.module.load_profile()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.context = {'runID': 'test-current', 'configuration': 'debug',
                        'profile': self.profile['id'], 'fixtureSHA256': 'a' * 64}
        for case_id, case in self.profile['cases'].items():
            words = self.profile['words'][case['reference']]
            record = dict(self.context, provider=self.profile['provider'], mode='exact', caseID=case_id, wordCount=len(words), words=words)
            (self.directory / (case_id + '.json')).write_text(json.dumps(record))

    def validate(self, exit_code=0):
        return self.module.validate_receipts(self.directory, self.profile, self.context, exit_code)

    def change(self, field, value):
        path = next(self.directory.glob('*.json'))
        record = json.loads(path.read_text())
        record[field] = value
        path.write_text(json.dumps(record))

    def test_complete_inventory_is_35_cases_834_words(self):
        self.assertEqual(self.validate(), {'cases': 35, 'words': 834})

    def test_lost_flag_does_not_pass_with_green_child(self):
        self.change('mode', 'tolerance')
        with self.assertRaisesRegex(ValueError, 'mode'):
            self.validate()

    def test_missing_comparison_fails(self):
        next(self.directory.glob('*.json')).unlink()
        with self.assertRaisesRegex(ValueError, 'inventory'):
            self.validate()

    def test_empty_selection_fails(self):
        for path in self.directory.iterdir():
            path.unlink()
        with self.assertRaisesRegex(ValueError, 'inventory'):
            self.validate()

    def test_duplicate_case_fails(self):
        (self.directory / 'duplicate.json').write_text(next(self.directory.glob('*.json')).read_text())
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.validate()

    def test_unknown_case_fails(self):
        self.change('caseID', 'unknown')
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            self.validate()

    def test_one_word_change_fails_even_with_claimed_success(self):
        path = next(self.directory.glob('*.json'))
        record = json.loads(path.read_text())
        record['words'][0] ^= 1
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'words'):
            self.validate()

    def test_stale_or_wrong_context_fails(self):
        for field in self.context:
            with self.subTest(field=field):
                path = next(self.directory.glob('*.json'))
                original = path.read_text()
                self.change(field, 'wrong')
                with self.assertRaisesRegex(ValueError, field):
                    self.validate()
                path.write_text(original)

    def test_malformed_and_wrong_count_fail(self):
        self.change('wordCount', 0)
        with self.assertRaisesRegex(ValueError, 'wordCount'):
            self.validate()
        next(self.directory.glob('*.json')).write_text('not JSON')
        with self.assertRaises(ValueError):
            self.validate()

    def test_failed_child_cannot_be_overridden_by_receipts(self):
        with self.assertRaisesRegex(ValueError, 'process'):
            self.validate(exit_code=1)

    def test_every_host_identity_component_is_required(self):
        host = dict(self.profile['host'])
        self.module.require_host(host, self.profile)
        for key in host:
            with self.subTest(key=key):
                changed = dict(host, **{key: 'different'})
                with self.assertRaisesRegex(ValueError, key):
                    self.module.require_host(changed, self.profile)

    def test_controlled_environment_sets_flag_after_sanitization(self):
        child = self.module.test_environment({'PATH': '/usr/bin', 'VECTORCORE_UMAP_CHECK_BASELINE_BITS': '0',
                                              'VECTORCORE_TEST_EXTENDED': '1', 'DYLD_INSERT_LIBRARIES': 'bad',
                                              'VC_FORCE_SWIFT_LINALG': '1'}, self.context, self.directory)
        self.assertEqual(child['VECTORCORE_UMAP_CHECK_BASELINE_BITS'], '1')
        self.assertEqual(child['VECTORCORE_TEST_EXTENDED'], '0')
        self.assertEqual(child['VECTORCORE_STRICT_PERF'], '0')
        self.assertNotIn('DYLD_INSERT_LIBRARIES', child)
        self.assertNotIn('VC_FORCE_SWIFT_LINALG', child)
        self.assertEqual(json.loads(child['VECTORCORE_UMAP_REFERENCE_CONTEXT'])['directory'], str(self.directory))


if __name__ == '__main__':
    unittest.main()
