"""Exercise the merge gate through its command-line boundary."""
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v03_process import run

SCRIPT = Path(__file__).resolve().parents[1] / "check_required.py"
ROOT = SCRIPT.parents[2]
WORKFLOW = ROOT / ".github/workflows/ci.yml"
JOBS = ["test", "lint", "repository-checks", "consumer", "platforms",
        "benchmark-contracts", "sanitizers"]


def workflow_required_contract():
    """Return (needs, argv) of the `required` job in ci.yml (stdlib text parse)."""
    text = WORKFLOW.read_text()
    block = re.search(r"^  required:\n(.*?)(?=^  \S|\Z)", text, re.M | re.S)
    if block is None:
        raise AssertionError("ci.yml has no `required` job")
    body = block.group(1)
    needs = re.search(r"^    needs: \[([^\]]*)\]\s*$", body, re.M)
    invocation = re.search(r"^\s+run: python3 Scripts/ci/check_required\.py (.+)$", body, re.M)
    if needs is None or invocation is None:
        raise AssertionError("required job lacks inline needs list or check_required.py run line")
    return ([n.strip() for n in needs.group(1).split(",")], invocation.group(1).split())


class RequiredChecksTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._calls = 0

    def test_jobs_match_workflow_needs_and_gate_argv(self):
        needs, argv = workflow_required_contract()
        self.assertEqual(JOBS, needs)
        self.assertEqual(JOBS, argv)

    def run_gate(self, payload, jobs=JOBS):
        env = dict(os.environ)
        env.pop("NEEDS_JSON", None)
        if payload is not None:
            env["NEEDS_JSON"] = payload
        self._calls += 1
        name = "gate-%03d" % self._calls
        record, _ = run([sys.executable, SCRIPT, *jobs], self._tmp.name, name,
                        cwd=ROOT, environment=env, timeout=15, expected=None)
        stderr = Path(record["stderr"]["path"]).read_text(errors="replace")
        return SimpleNamespace(returncode=record["exit"], stderr=stderr)

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
            "platforms": {"result": "success"},
            "benchmark-contracts": {"result": "success"},
            "sanitizers": {"result": "success"}
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
