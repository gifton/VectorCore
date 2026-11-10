"""A scoped acceptance must never hide supported-route failures or admit a release."""
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qualification_fixture import CaptureFixture, write_json


class CompilerQualificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = CaptureFixture(self.tmp.name)
        self.assertIsNotNone(importlib.util.find_spec('compiler_qualification'),
                             'scoped qualification assessment is not implemented')
        self.qualifier = importlib.import_module('compiler_qualification')

    def check(self):
        f = self.fixture
        return self.qualifier.check(f.profile_path, f.evidence, f.source)

    def assertRejected(self):
        result = self.check()
        self.assertEqual(result['vectorCoreQualification'], 'rejected', result)
        self.assertIs(result['releaseAdmitted'], False)
        self.assertTrue(result['problems'])
        return result

    def test_known_failing_compiler_can_have_scoped_acceptance_without_release(self):
        result = self.check()
        self.assertEqual(result['vectorCoreQualification'], 'accepted', result)
        self.assertEqual(result['compilerProfile'], 'known-failing')
        self.assertIs(result['releaseAdmitted'], False)
        self.assertIs(result['supportPolicyAdopted'], True)
        self.assertEqual(result['diagnosticFailures'], {'compiler': ['first']})

    def test_supported_failure_is_not_a_diagnostic_exception(self):
        f = self.fixture
        f.alter_stdout('supported', 'BUFFER_VALUE first observed=2\nBUFFER_VALUE_COMPLETE count=1 failures=1\n', code=1)
        f.commands['supported']['exit'] = 1
        f.case_specs[0]['observed'] = {'first': '2'}
        # Even an internally consistent newly pinned failure cannot be allowed in a supported group.
        f.save(repin=True)
        self.assertRejected()

    def test_unexpected_diagnostic_pass_requires_new_profile(self):
        f = self.fixture
        f.alter_stdout('compiler', 'BUFFER_VALUE first observed=1\nBUFFER_VALUE_COMPLETE count=1 failures=0\n', code=0)
        f.commands['compiler']['exit'] = 0
        f.save(repin=True)
        self.assertRejected()

    def test_missing_duplicate_or_unknown_case_rejects(self):
        f = self.fixture
        original = list(f.capture['cases'])
        for cases in [original[:-1], original + original[:1], original + [dict(original[0], id='surprise')]]:
            f.capture['cases'] = cases
            f.save(repin=True)
            with self.subTest(cases=cases):
                self.assertRejected()

    def test_numerical_row_inventory_and_completion_are_strict(self):
        f = self.fixture
        bad = ['BUFFER_VALUE_COMPLETE count=0 failures=0\n',
               'BUFFER_VALUE first observed=1\nBUFFER_VALUE first observed=1\nBUFFER_VALUE_COMPLETE count=2 failures=0\n',
               'BUFFER_VALUE first observed=1\n',
               'BUFFER_VALUE first observed=1\nBUFFER_VALUE_COMPLETE count=1 failures=0\nextra\n']
        for text in bad:
            f.alter_stdout('supported', text)
            f.save(repin=True)
            with self.subTest(text=text):
                self.assertRejected()

    def test_missing_negative_control_rejects(self):
        f = self.fixture
        f.profile['cases'] = [v for v in f.profile['cases'] if v['id'] != 'omitted']
        f.capture['cases'] = [v for v in f.capture['cases'] if v['id'] != 'omitted']
        f.save(repin=True)
        self.assertRejected()

    def test_malformed_profile_and_no_profile_never_qualify(self):
        for text in ['null', '[]', '{}', '{"schemaVersion":1,"schemaVersion":1}']:
            self.fixture.profile_path.write_text(text)
            with self.subTest(text=text):
                self.assertRejected()
        self.fixture.profile_path.unlink()
        self.assertRejected()

    def test_unknown_format_or_policy_never_qualifies(self):
        f = self.fixture
        f.profile['policyId'] = 'automatically-accept-new-compilers'
        f.save()
        self.assertRejected()
        f.profile['policyId'] = 'vc-0.4.0-xcode27.2-bounded-v1'
        f.case_specs[0]['format'] = 'trust-summary'
        f.capture['cases'][0]['format'] = 'trust-summary'
        f.save(repin=True)
        self.assertRejected()

    def test_original_lost_store_format_preserves_wrong_bits_and_process_exit(self):
        case = {'format': 'lost-store', 'correct': {'value': '3f800000'},
                'observed': {'value': '40000000'}, 'role': 'diagnostic'}
        command = {'record': {'exit': 0}, 'text': 'LOST_STORE observed=2.0 bits=40000000 expected=1.0\n'}
        result = self.qualifier.observations(case, command)
        self.assertFalse(result['numericalCorrect'])
        self.assertEqual(result['failures'], ['value'])
        for text in ['LOST_STORE observed=1.0 bits=40000000 expected=1.0\n',
                     command['text'] * 2, command['text'].rstrip(), '']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.qualifier.observations(case, dict(command, text=text))
        with self.assertRaises(ValueError):
            self.qualifier.observations(case, dict(command, record={'exit': 1}))

    def test_required_rows_in_diagnostic_group_cannot_fail(self):
        f = self.fixture
        f.case_specs[2]['requiredCorrectRows'] = ['first']
        f.save()
        self.assertRejected()

    def test_diagnostic_required_rows_must_exist_and_be_unique(self):
        f = self.fixture
        for names in [['unknown'], ['first', 'first']]:
            f.case_specs[2]['requiredCorrectRows'] = names
            f.save()
            with self.subTest(names=names):
                self.assertRejected()

    def test_auxiliary_checks_cannot_be_ignored(self):
        f = self.fixture
        f.capture['checks'] = [{'id': 'unassessed-suite', 'format': 'trust-green-summary', 'command': 'supported'}]
        f.save(repin=True)
        self.assertRejected()
        f.profile['checks'] = f.capture['checks']
        f.save()
        self.assertRejected()

    def test_standalone_diagnostic_can_have_its_own_omitted_write_control(self):
        f = self.fixture
        f.add_case('diagnostic-omitted', 'negative-control', '1', '2', negative_for='compiler')
        f.capture['commands']['diagnostic-omitted'] = f.commands['diagnostic-omitted']['receipt']
        f.save(repin=True)
        result = self.check()
        self.assertEqual(result['vectorCoreQualification'], 'accepted', result)
        self.assertFalse(result['observations']['diagnostic-omitted']['numericalCorrect'])

    def test_cli_output_is_exclusive_and_incomplete_input_returns_nonzero(self):
        f = self.fixture
        output = f.root / 'result.json'
        command = [sys.executable, '-B', str(Path(self.qualifier.__file__)),
                   '--profile', str(f.profile_path), '--evidence-root', str(f.evidence),
                   '--source-root', str(f.source), '--output', str(output)]
        p = subprocess.run(command, capture_output=True, text=True, timeout=15)
        self.assertEqual(p.returncode, 0, p.stderr)
        before = output.read_bytes()
        p = subprocess.run(command, capture_output=True, text=True, timeout=15)
        self.assertNotEqual(p.returncode, 0)
        self.assertEqual(output.read_bytes(), before)
        output.unlink()
        f.capture['collectionComplete'] = False
        f.save(repin=True)
        p = subprocess.run(command, capture_output=True, text=True, timeout=15)
        self.assertNotEqual(p.returncode, 0)
        self.assertIs(json.loads(output.read_text())['releaseAdmitted'], False)


if __name__ == '__main__':
    unittest.main()
