"""Exercise compiler-reproducer CLI deadline validation without a compiler."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "Docs/reviews/2026-11-02-readiness-batch/compiler/run.py"


class CompilerReproducerDeadlineTests(unittest.TestCase):
    def invoke(self, root, deadline):
        return subprocess.run(
            [sys.executable, str(RUNNER), "--swiftc", str(root / "missing-swiftc"),
             "--sdk", str(root / "missing-sdk"),
             "--importer", str(root / "missing-importer.swift"),
             "--output", str(root / "output"), f"--deadline={deadline}"],
            capture_output=True, text=True, timeout=10,
        )

    def test_invalid_deadlines_fail_before_creating_output(self):
        # Removing either the finite or positive check must fail this test.
        for deadline in ("nan", "inf", "-inf", "0", "-1"):
            with self.subTest(deadline=deadline), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                result = self.invoke(root, deadline)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("--deadline must be finite and positive", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse((root / "output").exists())

    def test_finite_positive_deadline_reaches_input_snapshot_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = self.invoke(root, "0.5")
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("FileNotFoundError", result.stderr)
            self.assertIn(str(root / "missing-importer.swift"), result.stderr)
            self.assertNotIn("--deadline must", result.stderr)
            self.assertTrue((root / "output/sources").is_dir())
            self.assertFalse((root / "output/compiler.json").exists())


if __name__ == "__main__":
    unittest.main()
