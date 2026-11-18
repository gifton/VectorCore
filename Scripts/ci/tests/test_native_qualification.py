"""Individual native outcomes and C-route rows remain strict under the policy."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TEXT = """Test Case '-[MinimalTests.BaselineTests testBasicVectorCreation]' started.
Test Case '-[MinimalTests.BaselineTests testBasicVectorCreation]' passed (0.001 seconds).
Executed 1 test, with 0 failures (0 unexpected) in 0.001 seconds
✔ Test run with 2 tests in 1 suite passed after 0.001 seconds.
"""
XML = '''<testsuites><testsuite name="VectorCore">
<testcase classname="ComprehensiveTests.Example" name="first()" time="0.01"/>
<testcase classname="ComprehensiveTests.Example" name="second()" time="0.02"/>
</testsuite></testsuites>'''


class NativeAssessmentTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('native_qualification'), 'native outcome assessment missing')
        self.mod = importlib.import_module('native_qualification')

    def test_individual_suite_outcomes_are_reconciled(self):
        result = self.mod.suite_observations(TEXT, XML)
        self.assertEqual(result['xctest'], {'-[MinimalTests.BaselineTests testBasicVectorCreation]': 'passed'})
        self.assertEqual(result['swiftTesting'], {'ComprehensiveTests.Example/first()': {'status': 'passed', 'issues': []},
                                                  'ComprehensiveTests.Example/second()': {'status': 'passed', 'issues': []}})
        self.mod.assess_suite(result, result, 0)

    def test_missing_duplicate_or_extra_suite_outcome_rejects(self):
        expected = self.mod.suite_observations(TEXT, XML)
        changes = [XML.replace('first()', 'second()'), XML.replace('</testsuites>', ''),
                   XML.replace('<testcase classname="ComprehensiveTests.Example" name="first()" time="0.01"/>', '')]
        for xml in changes:
            with self.subTest(xml=xml), self.assertRaises(ValueError):
                result = self.mod.suite_observations(TEXT, xml)
                self.mod.assess_suite(result, expected, 0)

    def test_xctest_markers_can_follow_buffered_test_output(self):
        expected = self.mod.suite_observations(TEXT, XML)
        # XCTest and test stdout share the pipe. The retained Debug run has
        # several complete XCTest markers after a partial diagnostic line.
        interleaved = TEXT.replace('Test Case ', 'partial diagnosticTest Case ')
        self.assertEqual(self.mod.suite_observations(interleaved, XML), expected)

    def test_unknown_failure_is_not_admitted_even_in_new_expected_table(self):
        xml = XML.replace('time="0.01"/>', 'time="0.01"><failure message="wrong sum"/></testcase>')
        text = TEXT.replace('2 tests in 1 suite passed', '2 tests in 1 suite failed')
        result = self.mod.suite_observations(text, xml)
        with self.assertRaises(ValueError):
            self.mod.assess_suite(result, result, 1)

    def test_log_issue_missing_from_xml_or_missing_completion_rejects(self):
        expected = self.mod.suite_observations(TEXT, XML)
        bad = [TEXT + '✘ Test first() recorded an issue at Example.swift:1:1: wrong sum\n',
               TEXT.split('✔')[0], TEXT.replace("' passed (", "' failed (")]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.mod.assess_suite(self.mod.suite_observations(text, XML), expected, 0)

    def test_original_canary_failure_requires_exact_reviewed_message(self):
        identity = self.mod.CANARY
        classname, name = identity.rsplit('/', 1)
        xml = f'<testsuites><testsuite><testcase classname="{classname}" name="{name}"><failure message="Expectation failed: (observed → 2.0) == 1"/></testcase></testsuite></testsuites>'
        text = TEXT.replace('2 tests in 1 suite passed', '1 test in 1 suite failed')
        text += '✘ Test literalArrayStoreIsObservedAfterOutOfLineClosure() recorded an issue at LostStoreCanaryTests.swift:47:9: Expectation failed: (observed → 2.0) == 1\n'
        expected = self.mod.suite_observations(text, xml)
        self.mod.assess_suite(expected, expected, 1)
        changed = self.mod.suite_observations(text.replace('2.0', '3.0'), xml.replace('2.0', '3.0'))
        with self.assertRaises(ValueError):
            self.mod.assess_suite(changed, expected, 1)

    def test_c_route_inventory_values_and_forwarding_counts(self):
        refs = self.mod.route_references()
        self.assertEqual(len(refs), 36)
        # Unit-scale dot is the exact sum of the integer fixture products.
        manual = sum((i % 15 - 7) * ((i * 5 + 3) % 19 - 9) for i in range(512))
        self.assertEqual(refs[(512, 'equal', 'dot')], manual)
        for enabled in (False, True):
            rows = []
            for (dimension, mode, metric), reference in refs.items():
                calls = 3 if enabled and (mode == 'equal' or (mode == 'unequal' and metric != 'euclidean')) else 0
                rows.append({'dimension': dimension, 'mode': mode, 'metric': metric, 'actual': reference,
                             'reference': reference, 'calls': calls, 'expectedCalls': calls})
            def output(values):
                return '\n'.join(['V02_ROUTE_ENTER', *['V02_ROUTE ' + json.dumps(r) for r in values],
                    *[f'V02_C_TAIL {n} calls=1' for n in (0, 1, 3, 4, 15, 16, 17, 31, 32, 33)],
                    'V02_ROUTE_COMPLETE widths=3 routes=36 tails=10']) + '\n'
            self.mod.assess_routes(output(rows), enabled)
            for mutation in ('missing', 'duplicate', 'wrong-number', 'nan', 'wrong-reference', 'wrong-count'):
                changed = copy.deepcopy(rows)
                if mutation == 'missing': changed.pop()
                if mutation == 'duplicate': changed[-1] = changed[0]
                if mutation == 'wrong-number': changed[0]['actual'] += 1
                if mutation == 'nan': changed[0]['actual'] = float('nan')
                if mutation == 'wrong-reference': changed[0]['reference'] += 1
                if mutation == 'wrong-count': changed[0]['calls'] += 1
                with self.subTest(enabled=enabled, mutation=mutation), self.assertRaises(ValueError):
                    self.mod.assess_routes(output(changed), enabled)

    def test_c_route_link_binding_rejects_other_objects_binary_or_feature(self):
        self.assertTrue(hasattr(self.mod, 'route_link_binding'), 'C producer binding missing')
        provenance = {'linkedObjects': [{'path': '/capture/library.o'}, {'path': '/capture/shim.o'}],
                      'consumer': {'path': '/capture/consumer'}}
        link = {'argv': ['swiftc', '-O', '/capture/library.o', '/capture/shim.o', '-o', '/capture/consumer']}
        executed = {'argv': ['/capture/consumer', '1']}
        self.mod.route_link_binding(link, executed, provenance, 'release', True)
        for argv in [['swiftc', '-O', '/capture/library.o', '-o', '/capture/consumer'],
                     ['swiftc', '-O', '/capture/library.o', '/capture/shim.o', '-o', '/capture/other'],
                     ['swiftc', '-Onone', '/capture/library.o', '/capture/shim.o', '-o', '/capture/consumer']]:
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                self.mod.route_link_binding({'argv': argv}, executed, provenance, 'release', True)
        with self.assertRaises(ValueError):
            self.mod.route_link_binding(link, {'argv': ['/capture/consumer', '0']}, provenance, 'release', True)


if __name__ == '__main__':
    unittest.main()
