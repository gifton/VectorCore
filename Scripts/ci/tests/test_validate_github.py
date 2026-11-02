"""Policy regressions use repository fixtures, never source-text assertions."""
from pathlib import Path
import shutil
import json
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v03_process import run
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "validate_github.rb"
ROOT = SCRIPT.parents[2]
PIN = "d23441a48e516b6c34aea4fa41551a30e30af803"
WORKFLOW = (ROOT / ".github/workflows/ci.yml").read_text()



class GithubPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="github policy ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(ROOT / ".github", self.root / ".github")
        inputs = ["Scripts/ci/sanitizer_bodies.json", "Scripts/ci/sanitizer_expected_bodies.txt",
                  "Scripts/ci/check_benchmark_build_modes.py"]
        manifest = json.loads((ROOT / inputs[0]).read_text())
        inputs += list({row['source'] for rows in manifest['sanitizers'].values() for row in rows})
        for relative in inputs:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        self.command_number = 0
        self.write("workflows/ci.yml", WORKFLOW)
        self.write("CODEOWNERS", "* @gifton\n")
        self.write("dependabot.yml", "version: 2\nupdates: []\n")
        self.write("ISSUE_TEMPLATE/bug.md", "---\nname: Bug\nabout: Report a bug\n---\nDetails\n")

    def write(self, relative, text):
        path = self.root / ".github" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def validate(self):
        self.command_number += 1
        record, _ = run(["ruby", str(SCRIPT), str(self.root)], self.root / "captures",
                        str(self.command_number), cwd=ROOT, expected=None)
        return SimpleNamespace(returncode=record['exit'],
                               stdout=Path(record['stdout']['path']).read_text(),
                               stderr=Path(record['stderr']['path']).read_text())

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
            ("if: always()", "if: success()"),
            ("if: always()", "if: always() && false"),
            ("needs: [test, lint, repository-checks, consumer, platforms, benchmark-contracts, sanitizers]", "needs: [test, lint]"),
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

    def test_rejects_missing_or_disabled_semantic_commands(self):
        mutations = [
            ('swift test -c "$CONFIGURATION"', 'echo skipped-test -c "$CONFIGURATION"'),
            ('python3 Scripts/ci/check_memory_pool.py', 'echo memory-pool-not-run'),
            ('ruby Scripts/ci/validate_github.rb', 'echo policy-not-run'),
            ('python3 Scripts/ci/consumer_smoke.py', 'echo consumers-not-run'),
            ('lint --config .swiftlint.yml', 'version --config .swiftlint.yml'),
            ('xcodebuild -scheme VectorCore', 'echo xcodebuild-not-run -scheme VectorCore'),
            ('swift build --build-tests --triple', 'echo intel-not-built --triple'),
            ('swift build -c release --product vectorcore-bench', 'echo product-not-built'),
            ('python3 Scripts/ci/tests/test_benchmark_contracts.py "$BENCH_BIN"',
             'python3 -B -m unittest discover -s Scripts/ci/tests -v'),
            ('python3 Scripts/ci/check_benchmark_build_modes.py', 'echo build-modes-not-run'),
            ('python3 Scripts/ci/check_sdk_name_collisions.py', 'echo sdk-names-not-checked'),
            ('  sdk-names:\n', '  sdk-names:\n    if: false\n'),
            ('  sdk-names:\n', '  sdk-names:\n    continue-on-error: true\n'),
            ('python3 Scripts/ci/check_sanitizers.py', 'echo sanitizers-not-run'),
            ('check_required.py test lint repository-checks consumer platforms benchmark-contracts sanitizers',
             'check_required.py test lint repository-checks consumer platforms'),
            ('  benchmark-contracts:\n', '  benchmark-contracts:\n    if: false\n'),
            ('  sanitizers:\n', '  sanitizers:\n    continue-on-error: true\n'),
            ('configuration: [debug, release]', 'configuration: [debug]'),
            ('sanitizer: [address, thread]', 'sanitizer: [address]'),
        ]
        for before, after in mutations:
            with self.subTest(before=before):
                self.assertIn(before, WORKFLOW)
                self.write('workflows/ci.yml', WORKFLOW.replace(before, after))
                self.assertNotEqual(self.validate().returncode, 0)

    def test_artifact_only_job_and_standalone_drift_are_rejected(self):
        # Keep job name, matrix and artifact step while removing actual execution.
        self.write('workflows/ci.yml', WORKFLOW.replace(
            'run: python3 Scripts/ci/check_sanitizers.py', 'run: echo artifact-only'))
        self.assertNotEqual(self.validate().returncode, 0)
        self.write('workflows/ci.yml', WORKFLOW)
        path = self.root / '.github/workflows/sanitizers.yml'
        path.write_text(path.read_text().replace('check_sanitizers.py', 'not_executed.py'))
        self.assertNotEqual(self.validate().returncode, 0)

    def test_required_instrumented_body_and_filter_mutations(self):
        path = self.root / 'Scripts/ci/sanitizer_bodies.json'
        original = path.read_text()
        for mutation in ['delete', 'empty-filter', 'wide-filter']:
            data = json.loads(original)
            if mutation == 'delete':
                data['sanitizers']['address'].pop()
            else:
                data['sanitizers']['address'][0]['filter'] = '' if mutation == 'empty-filter' else '.*'
            path.write_text(json.dumps(data))
            with self.subTest(mutation=mutation):
                self.assertNotEqual(self.validate().returncode, 0)
        path.write_text(original)

    def test_aggregate_and_required_job_step_bypasses_are_rejected(self):
        for job in ['required', 'test', 'lint', 'repository-checks', 'consumer', 'platforms', 'benchmark-contracts', 'sanitizers']:
            for level in ['job', 'step']:
                marker = '  ' + job + ':\n'
                self.assertIn(marker, WORKFLOW)
                if level == 'job':
                    changed = WORKFLOW.replace(marker, marker + '    continue-on-error: true\n', 1)
                else:
                    start = WORKFLOW.index(marker)
                    position = WORKFLOW.index('      - uses:', start)
                    changed = WORKFLOW[:position] + WORKFLOW[position:].replace(
                        '      - uses:', '      - continue-on-error: true\n        uses:', 1)
                self.write('workflows/ci.yml', changed)
                with self.subTest(job=job, level=level):
                    self.assertNotEqual(self.validate().returncode, 0)
