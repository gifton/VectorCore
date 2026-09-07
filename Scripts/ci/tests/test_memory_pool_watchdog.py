"""The external watchdog must stop compiler children as well as their driver."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

SPEC = importlib.util.spec_from_file_location(
    "memory_pool_check", Path(__file__).resolve().parents[1] / "check_memory_pool.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class MemoryPoolWatchdogTests(unittest.TestCase):
    def test_timeout_stops_descendants_before_cleanup(self):
        with tempfile.TemporaryDirectory(prefix="memory-pool-watchdog-") as directory:
            root = Path(directory)
            ready, release, escaped = [root / name for name in ("ready", "release", "escaped")]
            child = (
                "from pathlib import Path; import time; "
                f"Path({str(ready)!r}).write_text('ready')\n"
                f"while not Path({str(release)!r}).exists(): time.sleep(0.01)\n"
                f"Path({str(escaped)!r}).write_text('child survived')\n"
            )
            driver = (
                "import subprocess, sys, time; "
                f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(30)"
            )
            with self.assertRaises(subprocess.TimeoutExpired):
                CHECK.bounded_run([sys.executable, "-c", driver], timeout=2)
            self.assertTrue(ready.exists(), "child must have started before the timeout")
            release.touch()
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline and not escaped.exists():
                time.sleep(0.01)
            self.assertFalse(escaped.exists(), "timed-out child kept writing after driver termination")

    def test_nonzero_exit_is_reported(self):
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            CHECK.bounded_run([sys.executable, "-c", "raise SystemExit(7)"], timeout=5)
        self.assertEqual(failure.exception.returncode, 7)
