"""Exercise the merge gate through its command-line boundary."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "check_required.py"
JOBS = ["test", "lint", "repository-checks", "consumer", "platforms"]


class RequiredChecksTests(unittest.TestCase):
    def run_gate(self, payload, jobs=JOBS):
        env = dict(os.environ)
        env.pop("NEEDS_JSON", None)
        if payload is not None:
            env["NEEDS_JSON"] = payload
        return subprocess.run([sys.executable, str(SCRIPT), *jobs], env=env,
                              capture_output=True, text=True)

    def success(self):
        return {job: {"result": "success", "outputs": {}} for job in JOBS}

    def test_complete_success_passes(self):
        result = self.run_gate(json.dumps(self.success()))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_every_unsuccessful_result_blocks_merge(self):
        for job in JOBS:
            for state in ["failure", "skipped", "cancelled", "pending", None]:
                with self.subTest(job=job, state=state):
                    payload = self.success()
                    payload[job]["result"] = state
                    result = self.run_gate(json.dumps(payload))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(job, result.stderr)

    def test_missing_and_extra_jobs_block_merge(self):
        for job in JOBS:
            payload = self.success()
            del payload[job]
            with self.subTest(missing=job):
                self.assertNotEqual(self.run_gate(json.dumps(payload)).returncode, 0)
        payload = self.success()
        payload["surprise"] = {"result": "success"}
        self.assertNotEqual(self.run_gate(json.dumps(payload)).returncode, 0)

    def test_malformed_payloads_block_merge(self):
        malformed = [None, "", "{", "null", "[]", "{}", '"success"']
        for value in [None, [], "success", {}, {"result": True}]:
            payload = self.success()
            payload["test"] = value
            malformed.append(json.dumps(payload))
        for payload in malformed:
            with self.subTest(payload=payload):
                self.assertNotEqual(self.run_gate(payload).returncode, 0)

    def test_duplicate_job_result_cannot_overwrite_failure(self):
        payload = '''{
            "test": {"result": "failure"},
            "test": {"result": "success"},
            "lint": {"result": "success"},
            "repository-checks": {"result": "success"},
            "consumer": {"result": "success"},
            "platforms": {"result": "success"}
        }'''
        self.assertNotEqual(self.run_gate(payload).returncode, 0)

    def test_expected_ids_must_be_nonempty_unique_valid_set(self):
        cases = [
            ([], {}),
            ([""], {"": {"result": "success"}}),
            (["test", "test"], {"test": {"result": "success"}}),
            (["bad id"], {"bad id": {"result": "success"}}),
            (["${{ needs }}"], {"${{ needs }}": {"result": "success"}}),
        ]
        for jobs, payload in cases:
            with self.subTest(jobs=jobs):
                self.assertNotEqual(self.run_gate(json.dumps(payload), jobs).returncode, 0)
