"""Policy regressions use repository fixtures, never source-text assertions."""
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "validate_github.rb"
PIN = "a" * 40
WORKFLOW = """name: CI
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@PIN
        with:
          persist-credentials: false
  lint:
    runs-on: ubuntu-latest
    steps: [{run: 'true'}]
  repository-checks:
    runs-on: ubuntu-latest
    steps: [{run: 'true'}]
  consumer:
    runs-on: ubuntu-latest
    steps: [{run: 'true'}]
  platforms:
    runs-on: ubuntu-latest
    steps: [{run: 'true'}]
  required:
    name: CI Required
    if: ${{ always() }}
    needs: [test, lint, repository-checks, consumer, platforms]
    runs-on: ubuntu-latest
    steps: [{run: 'python3 Scripts/ci/check_required.py test lint repository-checks consumer platforms'}]
""".replace("PIN", PIN)


class GithubPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="github policy ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write("workflows/ci.yml", WORKFLOW)
        self.write("CODEOWNERS", "* @gifton\n")
        self.write("dependabot.yml", "version: 2\nupdates: []\n")
        self.write("ISSUE_TEMPLATE/bug.md", "---\nname: Bug\nabout: Report a bug\n---\nDetails\n")

    def write(self, relative, text):
        path = self.root / ".github" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def validate(self):
        return subprocess.run(["ruby", str(SCRIPT), str(self.root)],
                              capture_output=True, text=True)

    def test_valid_repository_passes(self):
        result = self.validate()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_parses_every_yaml_file_and_template_frontmatter(self):
        for path, content in [("other/nested.yaml", "invalid: [\n"),
                              ("ISSUE_TEMPLATE/extra.md", "---\nlabels: 'performance', 'regression'\n---\n"),
                              ("ISSUE_TEMPLATE/extra.md", "---\nname: Missing terminator\n")]:
            with self.subTest(path=path):
                self.write(path, content)
                result = self.validate()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(path, result.stderr)
                (self.root / ".github" / path).unlink()

    def test_rejects_owner_and_reviewer_placeholders(self):
        for path, content in [("CODEOWNERS", "* @yourusername\n"),
                              ("dependabot.yml", "version: 2\nreviewers: [yourusername]\n")]:
            with self.subTest(path=path):
                original = (self.root / ".github" / path).read_text()
                self.write(path, content)
                self.assertNotEqual(self.validate().returncode, 0)
                self.write(path, original)

    def test_rejects_workflow_security_and_gate_regressions(self):
        changes = [
            ("@" + PIN, "@v4"),
            ("permissions:\n  contents: read\n", ""),
            ("contents: read", "contents: write"),
            ("contents: read", "contents: read\n  security-events: write"),
            ("          persist-credentials: false\n", ""),
            ("persist-credentials: false", "persist-credentials: true"),
            ("branches: [main]", "branches: [main]\n    paths: ['Sources/**']"),
            ("${{ always() }}", "${{ success() }}"),
            ("${{ always() }}", "${{ always() && false }}"),
            ("needs: [test, lint, repository-checks, consumer, platforms]", "needs: [test, lint]"),
            ("    name: CI Required\n", ""),
        ]
        for before, after in changes:
            with self.subTest(change=after):
                self.write("workflows/ci.yml", WORKFLOW.replace(before, after))
                self.assertNotEqual(self.validate().returncode, 0)

    def test_checks_all_workflows_and_job_permissions(self):
        self.write("workflows/secondary.yaml", "name: Other\non: workflow_dispatch\njobs:\n  bad:\n    steps: [{uses: 'actions/upload-artifact@v4'}]\n")
        self.assertNotEqual(self.validate().returncode, 0)

    def test_allows_scoped_codeql_write_permission(self):
        self.write("workflows/codeql.yml", """name: CodeQL
on: workflow_dispatch
permissions:
  contents: read
jobs:
  analyze:
    permissions:
      contents: read
      security-events: write
    runs-on: ubuntu-latest
    steps:
      - uses: github/codeql-action/analyze@""" + PIN + "\n")
        result = self.validate()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_unrelated_job_write_permissions(self):
        self.write("workflows/ci.yml", WORKFLOW.replace("  test:\n", "  test:\n    permissions:\n      contents: read\n      security-events: write\n"))
        self.assertNotEqual(self.validate().returncode, 0)
