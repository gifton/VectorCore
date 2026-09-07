"""Install and execute hooks only in throwaway repositories."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HOOKS = Path(__file__).resolve().parents[3] / ".github" / "hooks"


class HookTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vectorcore hook tests ")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "repo with spaces"
        self.repo.mkdir()
        self.env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
        for key in ["GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"]:
            self.env.pop(key, None)
        self.git("init", "-q")
        self.git("config", "user.name", "Hook Test")
        self.git("config", "user.email", "hook@example.invalid")
        shutil.copytree(HOOKS, self.repo / ".github" / "hooks")
        self.git("add", ".github")
        self.git("-c", "core.hooksPath=/dev/null", "commit", "-qm", "fixture")

    def git(self, *args, cwd=None):
        return subprocess.run(["git", *args], cwd=cwd or self.repo, env=self.env,
                              check=True, capture_output=True, text=True).stdout.strip()

    def install(self, *args, repo=None):
        repo = repo or self.repo
        return subprocess.run(["bash", str(repo / ".github/hooks/install-hooks.sh"), *args],
                              cwd=self.base, env=self.env, capture_output=True, text=True)

    def hooks_path(self, repo=None):
        repo = repo or self.repo
        path = Path(self.git("rev-parse", "--git-path", "hooks", cwd=repo))
        return path if path.is_absolute() else repo / path

    def test_installs_into_default_custom_and_worktree_hooks_paths(self):
        for custom in [None, "custom hooks", str(self.base / "absolute hooks")]:
            with self.subTest(custom=custom):
                if custom:
                    self.git("config", "core.hooksPath", custom)
                result = self.install()
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(os.access(self.hooks_path() / "pre-commit", os.X_OK))
        self.git("config", "--unset", "core.hooksPath")
        worktree = self.base / "linked worktree"
        self.git("worktree", "add", "-q", "-b", "linked", str(worktree))
        hook = self.hooks_path(worktree) / "pre-commit"
        hook.unlink()
        result = self.install(repo=worktree)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(os.access(hook, os.X_OK))

    def test_preserves_existing_hook_unless_force_requested(self):
        hook = self.hooks_path() / "pre-commit"
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_text("#!/bin/sh\nexit 42\n")
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(hook.read_text(), "#!/bin/sh\nexit 42\n")
        result = self.install("--force")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(hook.read_bytes(), (HOOKS / "pre-commit").read_bytes())
        self.assertEqual(self.install().returncode, 0)

    def test_identical_hook_becomes_executable(self):
        hook = self.hooks_path() / "pre-commit"
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_bytes((HOOKS / "pre-commit").read_bytes())
        hook.chmod(0o644)
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(os.access(hook, os.X_OK))

    def fake_swiftlint(self):
        binary = self.base / "fake tools" / "swiftlint"
        binary.parent.mkdir()
        # The external CLI boundary validates SwiftLint's documented script-input API.
        binary.write_text("#!" + sys.executable + "\n" + '''import json, os, sys
if sys.argv[1:] == ["--version"]:
    print("fixture")
    sys.exit(0)
if sys.argv[1:] != ["lint", "--use-script-input-files"]:
    sys.exit(64)
files = [os.environ["SCRIPT_INPUT_FILE_" + str(i)]
         for i in range(int(os.environ["SCRIPT_INPUT_FILE_COUNT"]))]
with open(os.environ["LINT_RECORD"], "w") as output:
    json.dump(files, output)
print("linter fixture output")
sys.exit(int(os.environ.get("LINT_EXIT", "0")))
''')
        binary.chmod(0o755)
        self.env["PATH"] = str(binary.parent) + os.pathsep + self.env["PATH"]
        self.env["LINT_RECORD"] = str(self.base / "record.json")

    def run_hook(self):
        return subprocess.run(["bash", str(self.repo / ".github/hooks/pre-commit")],
                              cwd=self.repo, env=self.env, capture_output=True, text=True)

    def test_passes_space_and_newline_filenames_and_propagates_linter_failure(self):
        self.fake_swiftlint()
        filenames = ["file with spaces.swift", "file\nwith newline.swift"]
        for name in filenames + ["notes.txt"]:
            (self.repo / name).write_text("// fixture\n")
        self.git("add", ".")
        result = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(Path(self.env["LINT_RECORD"]).exists(), "SwiftLint did not receive the supported input API")
        self.assertEqual(set(json.loads(Path(self.env["LINT_RECORD"]).read_text())), set(filenames))
        self.env["LINT_EXIT"] = "7"
        result = self.run_hook()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("linter fixture output", result.stdout + result.stderr)

    def test_does_not_run_linter_for_non_swift_changes(self):
        self.fake_swiftlint()
        (self.repo / "notes.txt").write_text("fixture\n")
        self.git("add", "notes.txt")
        result = self.run_hook()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(Path(self.env["LINT_RECORD"]).exists())
