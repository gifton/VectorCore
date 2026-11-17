"""Fail-closed typed discovery controls; no native process is executed."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from discovery import parse_discovery, partition_discovery, xctest_command, known_warning_records, DiscoveryError

XCT = ['M.C/testAlpha', 'M.C/testBeta']
SWIFT = ['M.entry()', 'M.S/f()', 'M.S/Nested/parameterized(n:_:)']
WARN = ['==123==WARN: Invalid dyld module map detected. This is most likely a bug in the sanitizer.',
        '==123==WARN: Backtraces may be unreliable.']


def lines(values):
    return '\n'.join(values) + '\n'


class DiscoveryControls(unittest.TestCase):
    def test_clean_top_level_nested_and_xctest_records_are_typed_exactly(self):
        result = parse_discovery(lines(XCT + SWIFT), source='full.stdout')
        self.assertEqual(result['xctest'], XCT)
        self.assertEqual(result['swiftTesting'], sorted(SWIFT))
        self.assertEqual(result['diagnostics'], [])

    def test_observed_warning_pair_is_retained_in_either_stdout_not_selectors(self):
        for stream in ['full', 'helper']:
            with self.subTest(stream=stream):
                full = lines(XCT + SWIFT + (WARN if stream == 'full' else []))
                helper = lines(SWIFT + (WARN if stream == 'helper' else []))
                result = partition_discovery(full, helper, full_source='full.stdout', helper_source='helper.stdout')
                self.assertEqual(result['xctest'], XCT)
                self.assertEqual(result['swiftTesting'], sorted(SWIFT))
                offset = len(XCT + SWIFT) if stream == 'full' else len(SWIFT)
                self.assertEqual(result['diagnostics'], [
                    {'source':stream + '.stdout','line':offset + i + 1,'text':text,'pid':123,
                     'kind':['invalid-dyld-module-map','unreliable-backtraces'][i]}
                    for i,text in enumerate(WARN)])
                self.assertEqual(xctest_command('/host','/bundle',result['xctest']),
                                 ['/host','-XCTest','M.C/testAlpha,M.C/testBeta','/bundle'])

    def test_unknown_warning_prose_and_near_matches_are_rejected(self):
        for unknown in ['==123==WARN: some other warning.', 'warning: build note',
                        WARN[0].replace('==123==','==0=='), WARN[1][:-1],
                        'Test run with 1 test passed.', 'arbitrary diagnostic']:
            with self.subTest(unknown=unknown):
                with self.assertRaises(DiscoveryError):parse_discovery(lines(XCT + SWIFT + [unknown]),source='full.stdout')

    def test_empty_truncated_or_malformed_records_are_rejected(self):
        for text in ['', '\n', ' ', 'M.entry()', 'M.entry(\n', 'M.S/f(n:bad)\n',
                     'M.S//f()\n', 'M.C/testAlpha trailing\n', 'M.C/\n', 'M.foo\n']:
            with self.subTest(text=text):
                with self.assertRaises(DiscoveryError):parse_discovery(text,source='full.stdout')

    def test_duplicate_and_dual_framework_ambiguous_records_are_rejected(self):
        for values in [XCT + [XCT[0]], SWIFT + [SWIFT[0]], ['M.C/testAlpha', 'M.C/testAlpha()']]:
            with self.subTest(values=values):
                with self.assertRaises(DiscoveryError):parse_discovery(lines(values),source='full.stdout')

    def test_empty_or_inconsistent_framework_partitions_are_rejected(self):
        for full,helper in [(lines(SWIFT),lines(SWIFT)), (lines(XCT+SWIFT),''),
                            (lines(XCT+SWIFT),lines(XCT+SWIFT)),
                            (lines(XCT+SWIFT),lines(SWIFT[:-1])),
                            (lines(XCT+SWIFT),lines(SWIFT+['M.other()']))]:
            with self.subTest(full=full,helper=helper):
                with self.assertRaises(DiscoveryError):partition_discovery(full,helper)

    def test_xctest_argv_refuses_empty_duplicate_wrong_kind_or_warning(self):
        for selectors in [[], [XCT[0],XCT[0]], [SWIFT[0]], WARN, [XCT[0],WARN[0]]]:
            with self.subTest(selectors=selectors):
                with self.assertRaises(DiscoveryError):xctest_command('/host','/bundle',selectors)

    def test_stderr_warning_index_preserves_stream_line_and_exact_text(self):
        text='dyld[123]: <uuid> /host\n\n' + lines(WARN)
        result=known_warning_records(text,source='helper.stderr')
        self.assertEqual(result,[{'source':'helper.stderr','line':i+3,'text':value,'pid':123,
                                 'kind':['invalid-dyld-module-map','unreliable-backtraces'][i]}
                                for i,value in enumerate(WARN)])
        with self.assertRaises(DiscoveryError):parse_discovery(text,source='helper.stdout')


if __name__=='__main__':unittest.main()
