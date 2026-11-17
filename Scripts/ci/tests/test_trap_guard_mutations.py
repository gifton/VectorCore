"""Mutation editor safety checks; runtime detection is exercised by the Swift runner."""
import importlib.util
from pathlib import Path
import unittest
import tempfile
import sys

CI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CI))
spec = importlib.util.spec_from_file_location('trap_guard_mutations', CI / 'trap_guard_mutations.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TrapMutationEditorTests(unittest.TestCase):
    def test_output_cannot_recursively_copy_itself(self):
        for output in [Path('/tmp/repo'), Path('/tmp/repo/evidence')]:
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, 'outside the repository'):
                module.validate_output_path(output, Path('/tmp/repo'))
        module.validate_output_path(Path('/tmp/repo-evidence'), Path('/tmp/repo'))
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / 'repo'
            repository.mkdir()
            link = Path(temporary) / 'output-link'
            link.symlink_to(repository, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'outside the repository'):
                module.validate_output_path(link / 'evidence', repository)

    def test_reuse_refuses_unrestored_mutant(self):
        original = {'Sources/file.swift': 'original-hash'}
        module.validate_reused_source(original, {'sourceHashes': original})
        with self.assertRaisesRegex(ValueError, 'original production hashes'):
            module.validate_reused_source({'Sources/file.swift': 'mutant-hash'}, {'sourceHashes': original})

    def test_zero_selected_tests_is_not_a_baseline(self):
        self.assertEqual(module.executed_test_count('Test run with 0 tests passed.'), 0)
        self.assertEqual(module.executed_test_count('Test run with 3 tests in 1 suite passed.'), 3)
        self.assertEqual(module.executed_test_count('Executed 1 test, with 0 failures'), 1)
        self.assertEqual(module.executed_test_count('Build complete!'), 0)
        self.assertEqual(module.executed_test_count('Executed 1 test, with 0 failures\nTest run with 0 tests passed.'), 1)

    def test_selected_occurrence_only_is_removed(self):
        source = b'first\nguard\nmiddle\nguard\nlast\n'
        target = {'id': 'vectorized-count', 'remove': 'guard\n', 'occurrence': 1}
        self.assertEqual(module.delete_guard(source, target), b'first\nguard\nmiddle\nlast\n')
        self.assertEqual(source, b'first\nguard\nmiddle\nguard\nlast\n')

    def test_ambiguous_and_missing_guards_refused(self):
        target = {'id': 'basis', 'remove': 'guard\n'}
        for source in [b'unchanged\n', b'guard\nguard\n']:
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, 'expected 1 exact guards'):
                module.delete_guard(source, target)

    def test_ordinary_guard_preserves_other_bytes(self):
        target = {'id': 'basis', 'remove': 'guard\n'}
        self.assertEqual(module.delete_guard(b'prefix\nguard\nsuffix\n', target), b'prefix\nsuffix\n')


if __name__ == '__main__':
    unittest.main()
