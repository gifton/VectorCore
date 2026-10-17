"""The external watchdog must stop compiler children as well as their driver."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location(
    "memory_pool_check", Path(__file__).resolve().parents[1] / "check_memory_pool.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class MemoryPoolWatchdogTests(unittest.TestCase):
    def test_cleanup_has_a_deadline_even_if_killed_process_cannot_be_reaped(self):
        command = ["unreapable-test-process"]
        execution_timeout = subprocess.TimeoutExpired(command, 0.1)
        cleanup_timeout = subprocess.TimeoutExpired(command, 2)
        process = Mock(pid=12345)

        def wait(*, timeout=None):
            self.assertIsNotNone(timeout, "cleanup must not wait indefinitely")
            self.assertGreater(timeout, 0)
            if process.wait.call_count == 1:
                raise execution_timeout
            self.assertLessEqual(timeout, 2)
            raise cleanup_timeout

        process.wait.side_effect = wait
        with patch.object(CHECK.subprocess, "Popen", return_value=process), \
                patch.object(CHECK.os, "killpg") as kill_group:
            with self.assertRaises(subprocess.TimeoutExpired) as failure:
                CHECK.bounded_run(command, timeout=0.1)
        self.assertIs(failure.exception, execution_timeout)
        self.assertIs(failure.exception.__cause__, cleanup_timeout)
        self.assertEqual(process.wait.call_count, 2)
        kill_group.assert_called_once_with(process.pid, CHECK.signal.SIGKILL)

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
