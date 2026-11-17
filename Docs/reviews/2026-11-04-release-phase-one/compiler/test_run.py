"""Acceptance controls: process success alone must never accept a stale store."""
import importlib.util
from pathlib import Path
import unittest
import shutil
import tempfile

spec = importlib.util.spec_from_file_location('compiler_probe_driver', Path(__file__).with_name('run.py'))
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)

class AcceptanceTests(unittest.TestCase):
    def test_exact_value_and_bits_and_exit_are_required(self):
        good = 'LOST_STORE observed=1.0 bits=3f800000 expected=1.0\n'
        self.assertTrue(driver.accepts(0, good))
        for code, output in [(1, good), (0, ''),
                             (0, good.replace('1.0 bits', '2.0 bits')),
                             (0, good.replace('3f800000', '40000000')),
                             (0, good + good),
                             (0, 'LOST_STORE observed=2.0 bits=40000000 expected=1.0\n')]:
            with self.subTest(code=code, output=output):
                self.assertFalse(driver.accepts(code, output))

    def test_missing_probe_cannot_silently_shrink_matrix(self):
        new = Path(__file__).with_name('probes')
        retained = driver.ROOT / 'Docs/reviews/2026-11-01-compiler-qualification/probes'
        for original in driver.probe_sources(new, retained):
            with self.subTest(omitted=original.name), tempfile.TemporaryDirectory() as temp:
                copy = Path(temp)
                shutil.copytree(new, copy / 'new')
                shutil.copytree(retained, copy / 'retained')
                for path in copy.rglob(original.name):
                    path.unlink()
                with self.assertRaises(FileNotFoundError):
                    driver.probe_sources(copy / 'new', copy / 'retained')

if __name__ == '__main__':
    unittest.main()
