"""Scoped lexical controls; these do not prove arbitrary test semantics."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("b01_guard", ROOT / "Scripts/ci/fixtures/B01_inventory_guard.py")
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)
SUITES = [ROOT / "Tests/ComprehensiveTests" / name for name in
          ["SerializationEdgeCaseTests.swift", "VectorSerializationTests.swift"]]
GOLDEN = ROOT / "Tests/ComprehensiveTests/B01Golden.swift"

class B01SerializationInventoryTests(unittest.TestCase):
    def test_actual_registered_bodies_are_nonempty(self):
        for path in SUITES:
            self.assertTrue(GUARD.registered_bodies(path.read_text()))
            self.assertEqual(GUARD.empty_tests(path.read_text()), [], path.name)

    def test_all_sixteen_real_serialization_tests_remain(self):
        names = {row[0] for row in GUARD.registered_bodies(SUITES[1].read_text())}
        self.assertEqual(names, {
            "testBinaryRoundTrip_VectorFixedDim", "testBinaryRoundTrip_DynamicVector",
            "testBinaryDecode_InvalidMagicThrows", "testBinaryDecode_UnsupportedVersionThrows",
            "testBinaryDecode_InvalidTotalSizeThrows", "testBinaryDecode_FixedDimensionMismatchThrows",
            "testBinaryDecode_ChecksumMismatchThrows_DataTamper", "testBinaryDecode_ChecksumMismatchThrows_ChecksumTamper",
            "testBase64RoundTrip_VectorFixedDim", "testBase64Decode_InvalidStringThrows",
            "testJSONRoundTrip_VectorFixedDim", "testJSONRoundTrip_DynamicVector",
            "testJSONDecode_NonJSONTextThrows", "testDynamicJSONDecode_MismatchedElementsThrows",
            "testBinaryDecode_ZeroLengthThrows", "testBinaryRoundTrip_DeterministicWordsSmall"})

    def test_comment_only_body_mutations_are_rejected(self):
        for path in SUITES:
            source = path.read_text()
            name, start, end, _ = GUARD.registered_bodies(source)[0]
            mutant = source[:start] + '\n // removed assertion\n /* nested /* comment */ */\n' + source[end:]
            self.assertIn(name, GUARD.empty_tests(mutant))

    def test_empty_body_mutation_is_rejected(self):
        self.assertEqual(GUARD.empty_tests('@Test func missing() throws { }'), ['missing'])

    def test_comments_and_string_braces_do_not_end_body(self):
        source = '@Test("}") func real() { let text = #"{ /* } */"#; #expect(text.count > 0) }'
        self.assertEqual(GUARD.empty_tests(source), [])
        self.assertEqual(len(GUARD.registered_bodies(source)), 1)

    def test_required_golden_frames_are_present(self):
        self.assertEqual(GUARD.golden_errors(GOLDEN.read_text()), [])

    def test_empty_golden_mutations_are_rejected(self):
        import re
        source = GOLDEN.read_text()
        for name in ['emptyFrame', 'ordinaryFrame', 'specialFrame']:
            for replacement in ['[]', 'Data()']:
                mutant = re.sub(r'(static let ' + name + r': \[UInt8\] = )\[[^\]]*\]',
                                lambda m: m[1] + replacement, source)
                self.assertIn(name, GUARD.golden_errors(mutant))
        # Empty coordinate words are valid: the required empty frame is 16 bytes.
        self.assertIn('emptyWords: [UInt32] = []', source)

if __name__ == '__main__':
    unittest.main()
