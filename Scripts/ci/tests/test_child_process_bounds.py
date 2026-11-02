"""Reject blocking child-process APIs in test and CI drivers.

This mechanical gate covers the repeated unbounded-wait defect. Runtime tests
still establish each driver's exit handling, diagnostics and process cleanup.
"""
import ast
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[3]


SUBPROCESS_BLOCKING = ("run", "call", "check_call", "check_output")


def unbounded_subprocess_calls(source, filename="<source>"):
    """Return (line, name) for subprocess.<run|call|check_call|check_output>
    calls lacking a timeout= keyword or passing timeout=None.

    Attribute form on the `subprocess` module only; a `**kwargs` expansion is
    not treated as a timeout (conservative).
    """
    found = []
    for call in ast.walk(ast.parse(source, filename=filename)):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "subprocess"
                and call.func.attr in SUBPROCESS_BLOCKING):
            continue
        timeout = next((k.value for k in call.keywords if k.arg == "timeout"), None)
        if timeout is None or isinstance(timeout, ast.Constant) and timeout.value is None:
            found.append((call.lineno, call.func.attr))
    return sorted(found)


class ChildProcessBoundsTests(unittest.TestCase):
    def test_python_subprocess_calls_require_explicit_timeouts(self):
        violations = []
        for path in sorted((ROOT / "Scripts" / "ci").rglob("*.py")):
            for line, name in unbounded_subprocess_calls(path.read_text(), str(path)):
                violations.append(f"{path.relative_to(ROOT)}:{line}: subprocess.{name}")
        self.assertEqual(violations, [], "\n".join(violations))

    def test_subprocess_detector_negative_controls(self):
        for name in SUBPROCESS_BLOCKING:
            with self.subTest(name=name, case="missing"):
                self.assertEqual(unbounded_subprocess_calls(f"import subprocess\nsubprocess.{name}(['x'])\n"),
                                 [(2, name)])
            with self.subTest(name=name, case="none"):
                self.assertEqual(unbounded_subprocess_calls(f"import subprocess\nsubprocess.{name}(['x'], timeout=None)\n"),
                                 [(2, name)])
            with self.subTest(name=name, case="constant"):
                self.assertEqual(unbounded_subprocess_calls(f"import subprocess\nsubprocess.{name}(['x'], timeout=15)\n"), [])
            with self.subTest(name=name, case="variable"):
                self.assertEqual(unbounded_subprocess_calls(f"import subprocess\nsubprocess.{name}(['x'], timeout=limit)\n"), [])

    def test_swift_drivers_do_not_use_unbounded_waits_or_pipe_reads(self):
        violations, scanned = [], []
        for directory in (ROOT / "Tests", ROOT / "Scripts" / "ci"):
            for path in sorted(directory.rglob("*.swift")):
                source = path.read_text()
                # T10 moved every test-binary launch into ChildProbeLauncher.swift (posix_spawn), so
                # the launcher and its callers are scanned too, not only files that construct Process().
                if not any(token in source for token in ("Process()", "posix_spawn", "launchChildProbe(")):
                    continue
                scanned.append(path)
                for pattern in (r"\.waitUntilExit\s*\(", r"\.wait\s*\(\s*\)",
                                r"\.readDataToEndOfFile\s*\("):
                    for match in re.finditer(pattern, source):
                        line = source.count("\n", 0, match.start()) + 1
                        violations.append(f"{path.relative_to(ROOT)}:{line}: {match.group()}")
        self.assertEqual(violations, [], "\n".join(violations))
        # Visibility floor: the scan must still reach the launcher and its callers.
        names = {path.name for path in scanned}
        self.assertIn("ChildProbeLauncher.swift", names)
        self.assertGreaterEqual(len(scanned), 20)

    def test_python_ci_waits_require_explicit_deadlines(self):
        violations = []
        for path in sorted((ROOT / "Scripts" / "ci").rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for call in ast.walk(tree):
                if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                    continue
                if call.func.attr not in ("wait", "communicate"):
                    continue
                timeout = next((value.value for value in call.keywords
                                if value.arg == "timeout"), None)
                if timeout is None or isinstance(timeout, ast.Constant) and timeout.value is None:
                    violations.append(f"{path.relative_to(ROOT)}:{call.lineno}: {call.func.attr}")
        self.assertEqual(violations, [], "\n".join(violations))


# Timeout flags the detector follows. T10 moved every test-binary child into the launcher, whose
# result exposes `run.timedOut`; before T10 the sites declared `let flag = sem.wait(timeout:) == .timedOut`.
SEMAPHORE_TIMEOUT_DECLARATION = re.compile(
    r"\b(?:let|var)\s+(\w+)\s*=\s*[\w.]+\.wait\s*\(\s*timeout\s*:[\s\S]*?\)\s*==\s*\.timedOut")
# `let flag = run.timedOut`, also as a later binding of one `let` (`let a = run.x, flag = run.timedOut`).
LAUNCHER_TIMEOUT_DECLARATION = re.compile(r"(?:\b(?:let|var)\s+|,\s*)(\w+)\s*=\s*[\w.]+\.timedOut\b")
# A non-throwing assertion directly on the launcher property: `#expect(!run.timedOut` / `XCTAssertFalse(run.timedOut`.
LAUNCHER_TIMEOUT_ASSERTION = re.compile(r"(?:#expect\s*\(\s*!\s*|XCTAssertFalse\s*\(\s*)([\w.]+\.timedOut)\b")
LAUNCHER_TIMEOUT_READ = re.compile(r"\b[A-Za-z_]\w*\.timedOut\b")
STATUS_READ = re.compile(r"\.termination(?:Status|Reason)\b")


def timeout_checks(source):
    """Number of timeout checks the detector can see (its visibility metric): semaphore-form
    declarations plus every `<name>.timedOut` read of a launcher result."""
    return len(SEMAPHORE_TIMEOUT_DECLARATION.findall(source)) + len(LAUNCHER_TIMEOUT_READ.findall(source))


def nonthrowing_timeout_status_patterns(source):
    """Return forbidden lexical timeout-assertion/status patterns.

    A non-throwing timeout assertion (`#expect(!flag` / `XCTAssertFalse(flag`) lets the test continue,
    and a later termination-status read then runs on a child that may not have exited (Foundation
    `Process` traps; the T10 launcher returns nil). This conservative gate is not Swift control-flow
    analysis: even a later guard is not accepted as a substitute for an explicit throwing timeout
    branch (`if flag { throw … }` or `try #require(!flag)`).
    """
    violations = []
    flags = [(m.group(1), m.end()) for m in SEMAPHORE_TIMEOUT_DECLARATION.finditer(source)]
    flags += [(m.group(1), m.end()) for m in LAUNCHER_TIMEOUT_DECLARATION.finditer(source)]
    for flag, declared_at in flags:
        assertion = re.search(
            rf"(?:#expect\s*\(\s*!\s*{re.escape(flag)}\b|"
            rf"XCTAssertFalse\s*\(\s*{re.escape(flag)}\b)",
            source[declared_at:])
        if assertion is None:
            continue
        assertion_start = declared_at + assertion.start()
        status = STATUS_READ.search(source[declared_at + assertion.end():])
        if status is not None:
            line = source.count("\n", 0, assertion_start) + 1
            violations.append((line, flag, status.group()))
    for assertion in LAUNCHER_TIMEOUT_ASSERTION.finditer(source):
        status = STATUS_READ.search(source[assertion.end():])
        if status is not None:
            line = source.count("\n", 0, assertion.start()) + 1
            violations.append((line, assertion.group(1), status.group()))
    return sorted(violations)


class TimeoutAssertionPatternTests(unittest.TestCase):
    def test_both_assertion_forms_and_status_properties_are_rejected(self):
        for assertion in ('#expect(!expired, "deadline")',
                          'XCTAssertFalse(expired, "deadline")'):
            for property_name in ("terminationStatus", "terminationReason"):
                with self.subTest(assertion=assertion, property=property_name):
                    source = ("let expired = exited.wait(timeout: .now() + 15) == .timedOut\n"
                              + assertion + "\nlet status = process." + property_name)
                    self.assertEqual(nonthrowing_timeout_status_patterns(source),
                                     [(2, "expired", "." + property_name)])

    def test_multiline_assignment_and_compound_assertion_are_rejected(self):
        source = '''var timeout = self.exited.wait(
            timeout: .now() + 15
        ) == .timedOut
        #expect(!timeout && !process.isRunning, "deadline")
        if !process.isRunning { print(process.terminationStatus) }
        '''
        self.assertEqual(nonthrowing_timeout_status_patterns(source),
                         [(4, "timeout", ".terminationStatus")])

    def test_canonical_throw_and_unrelated_assertions_are_accepted(self):
        for assertion in ('#expect(!unrelated, "other condition")',
                          'XCTAssertFalse(unrelated, "other condition")', ""):
            with self.subTest(assertion=assertion):
                source = '''let timeout = exited.wait(timeout: .now() + 15) == .timedOut
                if timeout { throw ChildError.timeout }
                ''' + assertion + '''
                print(process.terminationStatus, process.terminationReason)
                '''
                self.assertEqual(nonthrowing_timeout_status_patterns(source), [])

    def test_timeout_assertion_without_status_access_is_outside_pattern(self):
        source = '''let timeout = exited.wait(timeout: .now() + 15) == .timedOut
        #expect(!timeout, "deadline")
        print("capture retained")
        '''
        self.assertEqual(nonthrowing_timeout_status_patterns(source), [])

    def test_launcher_forms_are_rejected(self):
        # T10 launcher results: a declaration (also as a later binding of one `let`) and a direct assertion.
        cases = {
            'let timedOut = process.timedOut\n#expect(!timedOut, "deadline")\nprint(process.terminationStatus)\n':
                [(2, "timedOut", ".terminationStatus")],
            'let arguments = child.arguments, timedOut = child.timedOut, cleaned = child.cleaned\n'
            'XCTAssertFalse(timedOut)\nlet reason = child.terminationReason\n':
                [(2, "timedOut", ".terminationReason")],
            '#expect(!run.timedOut && run.terminationStatus == 0, "log")\n':
                [(1, "run.timedOut", ".terminationStatus")],
            'XCTAssertFalse(run.timedOut)\nXCTAssertEqual(run.terminationReason, .exit)\n':
                [(1, "run.timedOut", ".terminationReason")],
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(nonthrowing_timeout_status_patterns(source), expected)

    def test_launcher_throwing_forms_are_accepted(self):
        for source in ('if run.timedOut { throw ChildError.timeout }\n#expect(run.terminationStatus == 0)\n',
                       'try #require(!run.timedOut, "deadline")\nprint(run.terminationStatus)\n',
                       'let timedOut = run.timedOut\nif timedOut { throw E.t }\n#expect(run.terminationReason == .exit)\n',
                       '#expect(!run.timedOut)\nprint("no status read")\n'):
            with self.subTest(source=source):
                self.assertEqual(nonthrowing_timeout_status_patterns(source), [])

    def test_visibility_metric_counts_both_forms(self):
        source = ('let a = exited.wait(timeout: .now() + 1) == .timedOut\n'
                  'if run.timedOut { throw E.t }\nlet t = other.timedOut\n// == .timedOut is not a read\n')
        self.assertEqual(timeout_checks(source), 3)

    def test_swift_repository_has_no_forbidden_timeout_assertion_pattern(self):
        violations, checks = [], 0
        for directory in (ROOT / "Tests", ROOT / "Scripts" / "ci"):
            for path in sorted(directory.rglob("*.swift")):
                source = path.read_text()
                checks += timeout_checks(source)
                for line, flag, property_name in nonthrowing_timeout_status_patterns(source):
                    violations.append(f"{path.relative_to(ROOT)}:{line}: {flag} followed by {property_name}")
        self.assertEqual(violations, [], "\n".join(violations))
        # Visibility floor: the detector went vacuous once (15 semaphore sites to 0 after the T10
        # migration); it must keep seeing the timeout checks it guards.
        self.assertGreaterEqual(checks, 20)
