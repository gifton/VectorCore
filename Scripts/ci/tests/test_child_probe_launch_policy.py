"""Policy (T10): tests re-run the test binary only through ChildProbeLauncher.swift.

WHY: 22 call sites (V04 phase F review, I-5) copied `CommandLine.arguments`, replaced only the FIRST
`--filter` value and launched the copy with Foundation `Process`. With several `--filter` arguments
the child also ran the other selected tests, and every probe-spawning test among them spawned again,
without bound; a deadline killed only the direct child and orphaned the rest. The launcher removes
every selector and gives the child exactly one, marks the child so it cannot launch a probe, and
spawns the child as the leader of its own process group, which it kills as a group (its header
states the contract; ChildProbeLauncherTests pins the behaviour). This test closes the class.

RULES (lexical, on swift_lexing._mask: comments and string literals are blanked, so they never count,
except where a rule reads string literals on purpose):
  R1 spawn     In Tests/**/*.swift outside the launcher, no process creation: a `Process` token (the nested
               type `Process.TerminationReason` excepted), `NSTask`, `NSWorkspace`, and the C spawn family
               as a TOKEN, so a call, a qualified call (`Darwin.posix_spawn(`), a backticked name and a
               function reference (`let f = posix_spawn`) all count: `posix_spawn`, `posix_spawnp`, `fork`,
               `vfork`, `execv`, `execve`, `execvp`, `execvP`, `execl`, `execle`, `execlp`, `popen`; `system`
               only as a call (`system(`; it is unavailable in Swift on Darwin, and the bare word is too
               common). SPAWN_ALLOWANCES pins the sole exception by exact count and shape:
                 - ChildProbeLauncherTests.swift: one posix_spawn of "/bin/sleep" (a group member that the
                   launcher's group cleanup must remove; never the test binary).
               Benchmark fixtures use the same package-scoped VectorCoreTestSupport launcher.
  R2 selector  In Tests/**/*.swift outside the launcher, no test-runner selector string literal: "--filter",
               "--skip", "--specifier", "-XCTest" and their "=" forms (literal text read from the source; a
               literal inside a comment does not count). SELECTOR_ALLOWANCES pins the exact count per
               allowed file (a ratchet: any new literal fails).
  R3 launcher  The launcher keeps its load-bearing tokens: the process-group spawn flag and pgroup call,
               POSIX_SPAWN_CLOEXEC_DEFAULT, WNOWAIT, a negative-PID kill, the selection flags in the dropped
               set (`--filter`, `--skip`, `--specifier`, `-s`, `--xunit-output`) and the refused set
               (`--configuration-path`), the selector shape check; three ORDERS, checked inside the
               function bodies: in `launchChildProbe` the nested-launch refusal and the child-marker
               assignment both come before the spawn call; in the reaper's `run` the group sweep
               (`sendToGroup(SIGKILL)`) comes before the reap (`waitpid`); and ONE caller: the spawn
               helper `spawnChildProbeGroupLeader(` is called exactly once in the file, inside
               `launchChildProbe`, so no second entry point can spawn without the refusal and marker.
  R4 import    Every launcher caller explicitly imports VectorCoreTestSupport.
  R5 budget    Ordinary call sites use ChildProbeBudget.standard.deadline/.cleanup. Named exceptions
               preserve deliberate self-test deadlines, Q03 sanitizer timing and T16 profiler calibration;
               each nonstandard call is pinned by budget shape, enclosing function and count.
               RunnerContractTests pins its wrapper's ordinary defaults separately.
  R6 floor     The scan must see the launcher in use: at least MIN_CALL_FILES files with a
               `launchChildProbe(` call, and the launcher itself.

LIMITS (lexical heuristic, not a parser), each NOT detected:
  * a process created INSIDE a string interpolation (swift_lexing._mask blanks interpolations);
  * a spawn reached through `dlsym("posix_spawn")` or another string-named symbol lookup, or through a
    re-declared symbol such as `@_silgen_name("posix_spawn") func launch(...)` (the name is a string);
  * swift-testing exit tests (`#expect(processExitsWith:)`): the framework re-runs the binary for that
    one body itself, which is outside this class;
  * a selector assembled from pieces ("--fil" + "ter"), read from a variable, or inside a multi-line
    string literal; "-s" (SwiftPM's deprecated alias, which the launcher still drops) is too generic to
    forbid lexically; raw strings (#"..."#) follow swift_lexing's documented limits;
  * `system` used as a function reference.
R3 checks presence and order, not behaviour (ChildProbeLauncherTests does that).

CONTROLS: must-FLAG inline fixtures (the pre-fix site shapes and each spawn spelling); must-PASS fixtures
(comments, strings, look-alike identifiers, the launcher call); MUTATION: disabling each rule flips its
control. No git history is used, so the test passes in a git-less export.
"""

import importlib.util
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location('swift_lexing', ROOT / 'Scripts/ci/swift_lexing.py')
lexing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lexing)
_mask = lexing._mask

LAUNCHER = 'Tests/VectorCoreTestSupport/ChildProbeLauncher.swift'
SPAWN = re.compile(r'\bProcess\b(?!\s*\.\s*TerminationReason\b)|\bNSTask\b|\bNSWorkspace\b'
                   r'|\b(?:posix_spawnp?|v?fork|exec(?:v|ve|vp|vP|l|le|lp)|popen)\b|\bsystem\s*\(')
SELECTOR_LITERAL = re.compile(r'"(?:--filter|--skip|--specifier|-XCTest)(?:=[^"\\\n]*)?"')
SELECTOR_SENTINEL = '__T10_SELECTOR_LITERAL__'
# path -> (token, exact count, literal every allowed call must contain or None,
#          regex the file's raw source must match or None, extra check(source, masked) -> [problem], reason)
SPAWN_ALLOWANCES = {
    'Tests/ComprehensiveTests/ChildProbeLauncherTests.swift': (
        'posix_spawn', 1, '"/bin/sleep"', None, None,
        'the launcher self-test spawns /bin/sleep as a member of the child\'s group, never the test binary'),
}
# path -> (exact selector-literal count, reason)
SELECTOR_ALLOWANCES = {
    'Tests/BenchmarkContractTests/BenchmarkSelectionContractTests.swift': (
        9, 'the vectorcore-bench CLI\'s own --filter option, parsed in-process by CLIOptions; no process launch'),
    'Tests/BenchmarkContractTests/BenchmarkModelDomainTests.swift': (
        8, 'the vectorcore-bench CLI\'s own --filter option; child scenarios use RunnerContractTests\' shared group launcher'),
    'Tests/ComprehensiveTests/ChildProbeLauncherTests.swift': (
        36, 'argument-filter fixtures passed to childProbeArguments; its only launches go through launchChildProbe'),
}
LAUNCHER_INVARIANTS = (
    ('setpgroup_flag', r'\bPOSIX_SPAWN_SETPGROUP\b', 'code'),
    ('setpgroup_call', r'\bposix_spawnattr_setpgroup\s*\(', 'code'),
    ('cloexec_default', r'\bPOSIX_SPAWN_CLOEXEC_DEFAULT\b', 'code'),
    ('wnowait', r'\bWNOWAIT\b', 'code'),
    ('group_kill', r'\bkill\s*\(\s*-\s*pid\b', 'code'),
    ('refusal_before_spawn', ('launchChildProbe', r'\brefuseNestedChildProbeLaunch\s*\(',
                              r'\bspawnChildProbeGroupLeader\s*\('), 'order'),
    ('marker_before_spawn', ('launchChildProbe', r'\bchildEnvironment\s*\[\s*childProbeMarkerKey\s*\]\s*=(?!=)',
                             r'\bspawnChildProbeGroupLeader\s*\('), 'order'),
    ('sweep_before_reap', ('run', r'\bsendToGroup\s*\(\s*SIGKILL\s*\)', r'\bwaitpid\s*\('), 'order'),
    ('single_test_selector', r'\bfunc\s+childProbeArguments\b[^{]*\{\s*try\s+validateSingleTestSelection\s*\(', 'code'),
    ('single_spawn_caller', ('launchChildProbe', r'\bspawnChildProbeGroupLeader\s*\('), 'sole_caller'),
    ('drops_selection', r'childProbeDroppedFlags[^=]*=\s*\[(?=[^\]]*"--filter")(?=[^\]]*"--skip")'
                        r'(?=[^\]]*"--specifier")(?=[^\]]*"-s")(?=[^\]]*"--xunit-output")', 'raw'),
    ('refuses_configuration', r'childProbeRefusedFlags[^=]*=\s*\[[^\]]*"--configuration-path"', 'raw'),
)
RULES = ('spawn', 'selector', 'launcher', 'import', 'budget', 'floor')
MIN_CALL_FILES = 24


def _line(source, offset):
    return source.count('\n', 0, offset) + 1


def _call_text(source, masked, start):
    """Raw text from `start` to the parenthesis that closes the first `(` after it (masked counting)."""
    open_at = masked.find('(', start)
    if open_at < 0:
        return source[start:]
    depth = 0
    for index in range(open_at, len(masked)):
        depth += (masked[index] == '(') - (masked[index] == ')')
        if depth == 0:
            return source[start:index + 1]
    return source[start:]


def spawn_findings(source, path=''):
    """[(line, text)] for process creation, after the per-file allowance (exact count) is applied."""
    masked = _mask(source)
    hits = [(m.start(), m.group()) for m in SPAWN.finditer(masked)]
    allowance = SPAWN_ALLOWANCES.get(path)
    if allowance:
        token, count, literal, file_shape, extra_check, _ = allowance
        allowed = [h for h in hits if h[1].startswith(token)
                   and (literal is None or literal in _call_text(source, masked, h[0]))]
        problems = []
        if len(allowed) != count:
            problems.append(f'expected exactly {count} allowed {token} use(s), found {len(allowed)}')
        if file_shape is not None and not re.search(file_shape, source):
            problems.append('allowed launch lost its pinned shape')
        if extra_check is not None:
            problems += extra_check(source, masked)
        if not problems:
            hits = [h for h in hits if h not in allowed]
        else:
            return [(0, f'{path}: ' + '; '.join(problems))] + [(_line(source, start), text) for start, text in hits]
    return [(_line(source, start), text) for start, text in hits]


def selector_literals(source):
    """[line] of each selector string literal outside comments."""
    # Each literal becomes an identifier BEFORE masking, so one inside a comment is still blanked while
    # one in code survives as a countable token (no newline is touched, so line numbers are kept).
    masked = _mask(SELECTOR_LITERAL.sub(SELECTOR_SENTINEL, source))
    return [_line(masked, m.start()) for m in re.finditer(SELECTOR_SENTINEL, masked)]


def selector_findings(source, path=''):
    lines = selector_literals(source)
    allowance = SELECTOR_ALLOWANCES.get(path)
    if allowance:
        count, _ = allowance
        return [] if len(lines) == count else [(0, f'{path}: expected exactly {count} selector literals, found {len(lines)}')]
    return [(line, 'test-runner selector literal') for line in lines]


def _function_body(masked, name):
    """(start, end) offsets of the body of the first `func <name>` in masked source, or None."""
    header = re.search(r'\bfunc\s+' + re.escape(name) + r'\b', masked)
    if header is None:
        return None
    open_at = masked.find('{', header.end())
    depth = 0
    for index in range(open_at, len(masked)):
        depth += (masked[index] == '{') - (masked[index] == '}')
        if depth == 0:
            return open_at, index
    return None


def _in_order(masked, function, first, second):
    """True when `first` occurs inside `function`'s body before the first `second`, and both occur."""
    body = _function_body(masked, function)
    if body is None:
        return False
    text = masked[body[0]:body[1]]
    a, b = re.search(first, text), re.search(second, text)
    return a is not None and b is not None and a.start() < b.start()


def _sole_caller(masked, function, call):
    """True when `call` (a regex) occurs exactly once outside `func` declarations, inside `function`'s body."""
    body = _function_body(masked, function)
    calls = [m.start() for m in re.finditer(call, masked)
             if not re.search(r'\bfunc\s+$', masked[max(0, m.start() - 40):m.start()])]
    return body is not None and len(calls) == 1 and body[0] < calls[0] < body[1]


def launcher_findings(source):
    masked = _mask(source)
    missing = []
    for name, pattern, view in LAUNCHER_INVARIANTS:
        if view == 'order':
            ok = _in_order(masked, *pattern)
        elif view == 'sole_caller':
            ok = _sole_caller(masked, *pattern)
        else:
            ok = re.search(pattern, masked if view == 'code' else source) is not None
        if not ok:
            missing.append(name)
    return missing


# Exact nonstandard call shapes, each allowed once inside the named function. Other calls in
# these files use the ordinary rule; the counters reject duplicate or missing exception calls.
BUDGET_EXCEPTIONS = {
    'Tests/ComprehensiveTests/ChildProbeLauncherTests.swift': (
        ('launcherSelfTestChild', r'deadline:\s*1,\s*cleanup:\s*\.kill\(seconds:\s*1\)\s*\)$'),
        ('launchSelfTest', r'deadline:\s*deadline,\s*cleanup:\s*cleanup\s*\)$')),
    'Tests/ComprehensiveTests/FP16DiagnosticsBoundaryTests.swift': (
        ('runChild', r'deadline:\s*Double\(deadline\),\s*cleanup:\s*\.kill\(seconds:\s*2\)\s*\)$'),),
    'Tests/ComprehensiveTests/T16ReductionExceptionalTableTests.swift': (
        ('profilerRowsRunInAnIsolatedChild', r'deadline:\s*120,\s*cleanup:\s*\.terminateThenKill\(grace:\s*1,\s*seconds:\s*1\)\s*\)$'),),
    'Tests/BenchmarkContractTests/RunnerContractTests.swift': (
        ('runFixture', r'deadline:\s*deadline,\s*cleanup:\s*cleanup\s*\)$'),),
}
STANDARD_BUDGET = r'deadline:\s*ChildProbeBudget\.standard\.deadline\s*,\s*cleanup:\s*ChildProbeBudget\.standard\.cleanup\s*\)$'


def budget_findings(source, path=''):
    masked = _mask(source)
    findings = []
    exceptions = BUDGET_EXCEPTIONS.get(path, ())
    counts = [0] * len(exceptions)
    bodies = [_function_body(masked, function) for function, _ in exceptions]
    for match in re.finditer(r'\blaunchChildProbe\s*\(', masked):
        # Match the masked slice too: raw comment/string text must never supply real budget arguments.
        call = _call_text(masked, masked, match.start())
        if re.search(STANDARD_BUDGET, call):
            continue
        for index, ((_, pattern), body) in enumerate(zip(exceptions, bodies)):
            if body and body[0] < match.start() < body[1] and re.search(pattern, call):
                counts[index] += 1
                break
        else:
            findings.append((_line(source, match.start()), 'ordinary child must use ChildProbeBudget.standard'))
    for (function, _), count in zip(exceptions, counts):
        if count != 1:
            findings.append((0, f'expected exactly 1 budget exception in {function}, found {count}'))
    return findings


def scan_tree(root, rules=RULES):
    """Findings for the tree under `root` (a repository checkout or export)."""
    findings, call_files, launcher_seen = [], set(), False
    for path in sorted((root / 'Tests').rglob('*.swift')):
        relative = path.relative_to(root).as_posix()
        source = path.read_text(encoding='utf-8')
        if relative == LAUNCHER:
            launcher_seen = True
            if 'launcher' in rules:
                findings += [f'{relative}: launcher invariant missing: {name}' for name in launcher_findings(source)]
            continue
        if re.search(r'\blaunchChildProbe\s*\(', _mask(source)):
            call_files.add(relative)
            if 'import' in rules and not re.search(r'(?m)^import\s+VectorCoreTestSupport\s*$', _mask(source)):
                findings.append(f'{relative}: child launcher consumer must import VectorCoreTestSupport')
        if 'budget' in rules:
            findings += [f'{relative}:{line}: {problem}' for line, problem in budget_findings(source, relative)]
        if 'spawn' in rules:
            findings += [f'{relative}:{line}: process creation outside the launcher: {text}'
                         for line, text in spawn_findings(source, relative)]
        if 'selector' in rules:
            findings += [f'{relative}:{line}: {text} outside the launcher'
                         for line, text in selector_findings(source, relative)]
    if 'floor' in rules:
        if not launcher_seen:
            findings.append(f'launcher missing: {LAUNCHER}')
        if len(call_files) < MIN_CALL_FILES:
            findings.append(f'only {len(call_files)} files call launchChildProbe (floor {MIN_CALL_FILES})')
    return findings, call_files


PRE_FIX_SITE = '''
    private func runChild(_ scenario: String) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: CommandLine.arguments[0])
        var arguments = Array(CommandLine.arguments.dropFirst())
        if let index = arguments.firstIndex(of: "--filter") { arguments[index + 1] = "FP16DiagnosticsBoundaryTests/childProbe" }
        else { arguments += ["--filter", "FP16DiagnosticsBoundaryTests/childProbe"] }
        process.arguments = arguments
        try process.run()
    }
'''
PRE_FIX_XCTEST = '''
        let process = Process()
        process.arguments = ["-XCTest", "ComprehensiveTests.ExecutionBoundaryTests/testChildProbe", Bundle(for: Self.self).bundlePath]
'''
POST_FIX_SITE = '''
import VectorCoreTestSupport
    private func runChild(_ scenario: String) throws {
        let child = try launchChildProbe(.swiftTesting(filter: "FP16DiagnosticsBoundaryTests/childProbe"),
                                         environment: environment, output: file, deadline: ChildProbeBudget.standard.deadline, cleanup: ChildProbeBudget.standard.cleanup)
        print(child.terminationReason == Process.TerminationReason.uncaughtSignal, ProcessInfo.processInfo.processIdentifier)
    }
'''

SPAWN_FLAG = {
    'pre_fix_site': (PRE_FIX_SITE, 1),
    'pre_fix_xctest': (PRE_FIX_XCTEST, 1),
    'process_run_class_method': ('try Process.run(url, arguments: a) { _ in }\n', 1),
    'process_type_init': ('let p: Process = .init()\n', 1),
    'foundation_qualified': ('let p = Foundation.Process()\n', 1),
    'nstask': ('let t = NSTask()\n', 1),
    'posix_spawn': ('posix_spawn(&pid, path, nil, nil, argv, envp)\n', 1),
    'posix_spawnp': ('posix_spawnp(&pid, "swift", nil, nil, argv, envp)\n', 1),
    'fork': ('let pid = fork()\n', 1),
    'vfork': ('let pid = vfork()\n', 1),
    'execv': ('execv(path, argv)\n', 1),
    'execvp': ('execvp("xctest", argv)\n', 1),
    'popen': ('let f = popen("swift test", "r")\n', 1),
    'system': ('_ = system("swift test")\n', 1),
    'nsworkspace': ('NSWorkspace.shared.openApplication(at: url, configuration: configuration)\n', 1),
    'function_reference': ('let spawn = posix_spawn\n', 1),
    'fork_reference': ('let makeChild: () -> pid_t = fork\n', 1),
    'backticked_call': ('_ = `posix_spawn`(&pid, path, nil, nil, argv, envp)\n', 1),
    'qualified_call': ('_ = Darwin.posix_spawn(&pid, path, nil, nil, argv, envp)\n', 1),
}
SPAWN_PASS = {
    'post_fix_site': POST_FIX_SITE,
    'comments_and_strings': '// let p = Process()\n/* fork() */\nlet s = "Process() and posix_spawn( and fork()"\n',
    'look_alikes': 'let a = ProcessInfo.processInfo\nforkJoin(x)\nsystemCheck()\nlet b = subprocess\nlet c = processIdentifier\n',
    'termination_reason': 'func f() -> (Process.TerminationReason, Int32) { (.exit, 0) }\n',
    'spawn_attribute_helpers': 'posix_spawnattr_init(&a)\nposix_spawn_file_actions_init(&b)\n',
    'more_look_alikes': 'let execution = 1\nlet popenFlag = true\nlet forked = false\nlet ecosystem = 2\n',
}
SELECTOR_FLAG = {
    'pre_fix_site': (PRE_FIX_SITE, 2),
    'pre_fix_xctest': (PRE_FIX_XCTEST, 1),
    'skip_removal': ('args.removeAll { $0 == "--skip" }\n', 1),
    'equals_form': ('args += ["--filter=Probe/child"]\n', 1),
    'specifier': ('args.append("--specifier")\n', 1),
}
SELECTOR_PASS = {
    'post_fix_site': POST_FIX_SITE,
    'comments': '// arguments.firstIndex(of: "--filter")\n/* "--skip" */\n',
    'look_alikes': 'let a = ["--filtering", "--skip-build", "filter", "-XCTestScope"]\n',
    'inside_longer_string': 'let s = "use --filter X" + "a \\"--skip\\" b"\n',
}
# rule -> (control kind, control name)
MUTATION_CONTROLS = {
    'spawn': ('spawn', 'pre_fix_site'),
    'selector': ('selector', 'pre_fix_site'),
    'launcher': ('tree', 'launcher_without_group'),
    'floor': ('tree', 'no_callers'),
    'budget': ('tree', 'ordinary_budget'),
    'import': ('tree', 'missing_import'),
}


def _write_tree(base, launcher_text, callers=MIN_CALL_FILES):
    path = base / LAUNCHER
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(launcher_text)
    for index in range(callers):
        path = base / 'Tests' / 'ComprehensiveTests' / f'Caller{index}.swift'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(POST_FIX_SITE)


def host_operations_outside_macos(source):
    """Conservative source gate: native test-process operations require an enclosing macOS branch.

    This follows nested #if/#else/#endif structure, not arbitrary Swift conditions.
    Compiler typechecking and simulator execution establish the actual SDK boundary.
    """
    stack, violations = [], []
    for line, text in enumerate(_mask(source).splitlines(), 1):
        stripped = text.strip()
        if stripped.startswith('#if '):
            stack.append(stripped == '#if os(macOS)')
        elif stripped.startswith(('#else', '#elseif')):
            if stack:
                stack[-1] = False
        elif stripped == '#endif':
            if stack:
                stack.pop()
        elif re.search(r'\b(?:posix_spawn|posix_spawnp|execve|fork|vfork)\s*\(', text):
            if not any(stack):
                violations.append(line)
    return violations


class ChildProbeLaunchPolicyTests(unittest.TestCase):
    def test_ordinary_budget_controls(self):
        self.assertEqual(budget_findings(POST_FIX_SITE), [])
        for replacement in ['5', '10', '15', '30', '120']:
            source = POST_FIX_SITE.replace('ChildProbeBudget.standard.deadline', replacement)
            self.assertNotEqual(budget_findings(source), [])
        for cleanup in ['.kill(seconds: 2)', '.terminateThenKill(grace: 1, seconds: 1)']:
            self.assertNotEqual(budget_findings(POST_FIX_SITE.replace('ChildProbeBudget.standard.cleanup', cleanup)), [])
        runner = (ROOT / 'Tests/BenchmarkContractTests/RunnerContractTests.swift').read_text()
        self.assertRegex(_mask(runner), r'deadline: Double = ChildProbeBudget\.standard\.deadline,\s*cleanup: ChildProbeCleanup = ChildProbeBudget\.standard\.cleanup')
        self.assertEqual(set(BUDGET_EXCEPTIONS), {
            'Tests/ComprehensiveTests/ChildProbeLauncherTests.swift',
            'Tests/ComprehensiveTests/FP16DiagnosticsBoundaryTests.swift',
            'Tests/ComprehensiveTests/T16ReductionExceptionalTableTests.swift',
            'Tests/BenchmarkContractTests/RunnerContractTests.swift'})

    def test_budget_comment_and_string_decoys_do_not_allow_inflation(self):
        pair = 'deadline: ChildProbeBudget.standard.deadline, cleanup: ChildProbeBudget.standard.cleanup'
        nonstandard = POST_FIX_SITE.replace(pair, 'deadline: 120, cleanup: .kill(seconds: 2)')
        self.assertNotEqual(nonstandard, POST_FIX_SITE)
        decoys = {
            'block_comment': nonstandard.replace('deadline: 120', '/* ' + pair + ' */ deadline: 120'),
            'line_comment': nonstandard.replace('deadline: 120', '// ' + pair + '\n deadline: 120'),
            'string': nonstandard.replace('environment: environment', 'environment: ["note": "' + pair + '"]'),
            'raw_string': nonstandard.replace('environment: environment', 'environment: ["note": #"' + pair + ')"#]'),
            'multiline_string': nonstandard.replace('environment: environment', 'environment: ["note": """\n' + pair + ')\n"""]'),
        }
        for name, source in decoys.items():
            with self.subTest(decoy=name):
                self.assertNotEqual(budget_findings(source), [])
        # Comments between real arguments remain whitespace, rather than hiding a valid call.
        standard = POST_FIX_SITE.replace(', cleanup: ChildProbeBudget.standard.cleanup',
                                         ', /* diagnostic ) */ cleanup: ChildProbeBudget.standard.cleanup')
        self.assertEqual(budget_findings(standard), [])

    def test_exception_files_reject_additional_nonstandard_calls(self):
        nonstandard = POST_FIX_SITE.replace('ChildProbeBudget.standard.deadline', '120')
        for path in BUDGET_EXCEPTIONS:
            source = (ROOT / path).read_text()
            with self.subTest(path=path):
                self.assertEqual(budget_findings(source, path), [])
                self.assertEqual(budget_findings(source + POST_FIX_SITE, path), [])
                self.assertNotEqual(budget_findings(source + nonstandard, path), [])

    def test_exception_calls_have_pinned_budget_shapes_and_counts(self):
        mutations = {
            'Tests/ComprehensiveTests/ChildProbeLauncherTests.swift': ('deadline: 1, cleanup: .kill(seconds: 1)', 'deadline: 120, cleanup: .kill(seconds: 1)'),
            'Tests/ComprehensiveTests/FP16DiagnosticsBoundaryTests.swift': ('deadline: Double(deadline)', 'deadline: 120'),
            'Tests/ComprehensiveTests/T16ReductionExceptionalTableTests.swift': ('deadline: 120', 'deadline: 121'),
            'Tests/BenchmarkContractTests/RunnerContractTests.swift': ('deadline: deadline, cleanup: cleanup', 'deadline: 120, cleanup: cleanup'),
        }
        for path, (old, new) in mutations.items():
            source = (ROOT / path).read_text()
            with self.subTest(path=path):
                self.assertEqual(source.count(old), 1)
                self.assertNotEqual(budget_findings(source.replace(old, new), path), [])
                # A duplicate file is rejected too; the scoped copies below exercise exact counters.
                self.assertNotEqual(budget_findings(source + source, path), [])
                masked = _mask(source)
                for function, _ in BUDGET_EXCEPTIONS[path]:
                    body = _function_body(masked, function)
                    self.assertIsNotNone(body)
                    match = next(m for m in re.finditer(r'\blaunchChildProbe\s*\(', masked)
                                 if body[0] < m.start() < body[1])
                    call = _call_text(source, masked, match.start())
                    duplicated = source[:body[1]] + call + '\n' + source[body[1]:]
                    self.assertIn((0, f'expected exactly 1 budget exception in {function}, found 2'),
                                  budget_findings(duplicated, path))

    def test_capability_inventory_matches_individual_swift_testing_gates(self):
        inventory = json.loads((ROOT / 'Docs/reviews/2026-11-04-release-phase-two/ios/capability-inventory.json').read_text())
        expected = {(row['path'], row['method']) for row in inventory
                    if row['contract'] == 'Swift Testing subprocess isolation'}
        observed = set()
        for path in (ROOT / 'Tests/ComprehensiveTests').glob('*.swift'):
            source = _mask(path.read_text())
            # Test attributes precede their function declaration. Reject a suite-level gate:
            # adding one would silently disable otherwise portable numerical tests.
            self.assertNotRegex(source, r'@Suite[^{}]*childProbeProcessesSupported')
            for match in re.finditer(r'@Test(?P<attributes>[^{}]*?)\bfunc\s+(?P<name>\w+)\s*\(', source):
                if 'childProbeProcessesSupported' in match['attributes']:
                    observed.add((str(path.relative_to(ROOT)), match['name']))
        self.assertEqual(observed, expected)
        self.assertNotIn(('Tests/ComprehensiveTests/ChildProbeLauncherTests.swift',
                          'portableTerminationPreservesExitAndSignalDiagnostics'), observed)
        self.assertNotIn(('Tests/ComprehensiveTests/ChildProbeLauncherTests.swift',
                          'unreapedChildReadsFailWithoutTrapping'), observed)

    def test_native_process_operations_require_macos_compile_boundary(self):
        for path in (ROOT / 'Tests').rglob('*.swift'):
            self.assertEqual(host_operations_outside_macos(path.read_text()), [], str(path))
        operation = 'posix_spawn(&pid, path, nil, nil, argv, env)'
        self.assertEqual(host_operations_outside_macos('#if os(macOS)\n' + operation + '\n#endif'), [])
        for source in [operation, '#if os(iOS)\n' + operation + '\n#endif',
                       '#if os(macOS)\n#else\n' + operation + '\n#endif']:
            self.assertNotEqual(host_operations_outside_macos(source), [])
        self.assertEqual(host_operations_outside_macos('// ' + operation), [])
        self.assertEqual(host_operations_outside_macos('#if os(macOS)\n#if DEBUG\n' + operation + '\n#endif\n#endif'), [])

    def test_portable_result_types_do_not_reference_foundation_process(self):
        # The iOS SDK has no Process type, including its nested TerminationReason.
        paths = [ROOT / LAUNCHER, *sorted((ROOT / 'Tests/ComprehensiveTests').glob('*.swift')),
                 *sorted((ROOT / 'Tests/BenchmarkContractTests').glob('*.swift'))]
        violations = [str(path.relative_to(ROOT)) for path in paths
                      if re.search(r'\bProcess\s*\.', _mask(path.read_text()))]
        self.assertEqual(violations, [])

    def test_shared_launcher_is_in_test_support(self):
        self.assertEqual(LAUNCHER, 'Tests/VectorCoreTestSupport/ChildProbeLauncher.swift')
        self.assertTrue((ROOT / LAUNCHER).is_file())
        self.assertFalse((ROOT / 'Tests/ComprehensiveTests/ChildProbeLauncher.swift').exists())

    def test_benchmark_leaf_cannot_bypass_shared_launcher(self):
        runner = 'Tests/BenchmarkContractTests/RunnerContractTests.swift'
        old = ('let process = Process()\n'
               'process.arguments = ["-XCTest", "BenchmarkContractTests.BenchmarkRunnerContractTests/testProcessFixture", '
               'Bundle(for: Self.self).bundlePath]\n')
        self.assertNotEqual(spawn_findings(old, runner), [])
        self.assertNotEqual(selector_findings(old, runner), [])
        text = (ROOT / runner).read_text()
        self.assertIn('import VectorCoreTestSupport', text)
        self.assertIn('try launchChildProbe(', text)


    def _launcher_text(self):
        return (ROOT / LAUNCHER).read_text(encoding='utf-8')

    def test_spawn_controls(self):
        for name, (source, count) in SPAWN_FLAG.items():
            with self.subTest(name):
                self.assertEqual(len(spawn_findings(source)), count, spawn_findings(source))
        for name, source in SPAWN_PASS.items():
            with self.subTest(name):
                self.assertEqual(spawn_findings(source), [])

    def test_selector_controls(self):
        for name, (source, count) in SELECTOR_FLAG.items():
            with self.subTest(name):
                self.assertEqual(len(selector_findings(source)), count, selector_findings(source))
        for name, source in SELECTOR_PASS.items():
            with self.subTest(name):
                self.assertEqual(selector_findings(source), [])

    def test_allowances_are_exact(self):
        sleeper = 'posix_spawn(&pid, "/bin/sleep", nil, nil, argv, env)\n'
        path = 'Tests/ComprehensiveTests/ChildProbeLauncherTests.swift'
        self.assertEqual(spawn_findings(sleeper, path), [])
        # A second spawn, or the allowed spawn aimed at another binary, is a finding.
        self.assertNotEqual(spawn_findings(sleeper * 2, path), [])
        self.assertNotEqual(spawn_findings('posix_spawn(&pid, CommandLine.arguments[0], nil, nil, argv, env)\n', path), [])
        self.assertNotEqual(spawn_findings(sleeper + 'let p = Process()\n', path), [])
        runner = 'Tests/BenchmarkContractTests/RunnerContractTests.swift'
        pinned = ('let process = Process()\n'
                  'process.arguments = ["-XCTest", "BenchmarkContractTests.BenchmarkRunnerContractTests/testProcessFixture", '
                  'Bundle(for: Self.self).bundlePath]\n')
        self.assertNotEqual(spawn_findings(pinned, runner), [])
        self.assertNotEqual(selector_findings(pinned, runner), [])
        self.assertNotEqual(spawn_findings(pinned + 'let other = Process()\n', runner), [])
        for name, count in [('BenchmarkSelectionContractTests.swift', 9), ('BenchmarkModelDomainTests.swift', 8)]:
            bench = 'Tests/BenchmarkContractTests/' + name
            self.assertEqual(selector_findings('options(["--filter", "a"])\n' * count, bench), [])
            self.assertNotEqual(selector_findings('options(["--filter", "a"])\n' * (count + 1), bench), [])

    def test_launcher_invariants_and_mutations(self):
        text = self._launcher_text()
        self.assertEqual(launcher_findings(text), [])
        refusal = '    try refuseNestedChildProbeLaunch(selector, environment: ProcessInfo.processInfo.environment)\n'
        marker = '    childEnvironment[childProbeMarkerKey] = selector.description\n'
        spawn = ('    let pid = try spawnChildProbeGroupLeader(executable: executable, arguments: arguments,\n'
                 '                                             environment: childEnvironment, output: output.fileDescriptor)\n')
        sweep = '            sendToGroup(SIGKILL)\n'
        reap = '            repeat { waited = waitpid(pid, &status, 0) } while waited == -1 && errno == EINTR\n'
        for piece in (refusal, marker, spawn, sweep, reap):
            self.assertEqual(text.count(piece), 1, piece)
        mutations = {
            'setpgroup_flag': [lambda s: s.replace('POSIX_SPAWN_SETPGROUP | ', '')],
            'setpgroup_call': [lambda s: s.replace('try check(posix_spawnattr_setpgroup(&attributes, 0), "setpgroup")\n', '')],
            'cloexec_default': [lambda s: s.replace(' | POSIX_SPAWN_CLOEXEC_DEFAULT', '')],
            'wnowait': [lambda s: s.replace('WEXITED | WNOWAIT', 'WEXITED')],
            'group_kill': [lambda s: s.replace('kill(-pid, signal)', 'kill(pid, signal)')],
            # Removed, and moved after the spawn (a child would spawn before it refuses).
            'refusal_before_spawn': [lambda s: s.replace(refusal, ''),
                                     lambda s: s.replace(refusal, '').replace(spawn, spawn + refusal)],
            'marker_before_spawn': [lambda s: s.replace(marker, ''),
                                    lambda s: s.replace(marker, '').replace(spawn, spawn + marker)],
            # Removed, and moved after the reap (the group ID could then be recycled before the sweep).
            'sweep_before_reap': [lambda s: s.replace(sweep, ''),
                                  lambda s: s.replace(sweep, '').replace(reap, reap + sweep)],
            'single_test_selector': [lambda s: s.replace('    try validateSingleTestSelection(selector.description)\n', '')],
            # A second entry point that spawns without the refusal and the marker.
            'single_spawn_caller': [lambda s: s + ('\nfunc launchUnchecked(_ e: String) throws -> pid_t {\n'
                                                   '    try spawnChildProbeGroupLeader(executable: e, arguments: [], '
                                                   'environment: [:], output: 1)\n}\n'),
                                    lambda s: s.replace(spawn, '    let pid: pid_t = 0\n')],
            'drops_selection': [lambda s: s.replace('"--filter", "--skip", "--specifier", "-s",', '"--filter", "--specifier", "-s",')],
            'refuses_configuration': [lambda s: s.replace('["--configuration-path", ', '[')],
        }
        self.assertEqual(set(mutations), {name for name, _, _ in LAUNCHER_INVARIANTS})
        for name, variants in mutations.items():
            for index, mutate in enumerate(variants):
                with self.subTest(name, variant=index):
                    mutated = mutate(text)
                    self.assertNotEqual(mutated, text, f'mutation {name}/{index} did not apply')
                    self.assertIn(name, launcher_findings(mutated))

    def test_tree_controls_and_rule_mutations(self):
        import tempfile
        text = self._launcher_text()
        cases = {
            'clean': dict(launcher_text=text),
            'launcher_without_group': dict(launcher_text=text.replace('kill(-pid, signal)', 'kill(pid, signal)')),
            'no_callers': dict(launcher_text=text, callers=0),
            'ordinary_budget': dict(launcher_text=text),
            'missing_import': dict(launcher_text=text),
        }
        results = {}
        for name, kwargs in cases.items():
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                _write_tree(base, **kwargs)
                if name in ('ordinary_budget', 'missing_import'):
                    path = base / 'Tests/ComprehensiveTests/Caller0.swift'
                    change = (('ChildProbeBudget.standard.deadline', '5') if name == 'ordinary_budget'
                              else ('import VectorCoreTestSupport', 'import Foundation'))
                    path.write_text(path.read_text().replace(*change))
                results[name] = {rules: scan_tree(base, rules)[0]
                                 for rules in [RULES] + [tuple(r for r in RULES if r != rule) for rule in RULES]}
        self.assertEqual(results['clean'][RULES], [])
        for name in ('launcher_without_group', 'no_callers', 'ordinary_budget', 'missing_import'):
            self.assertNotEqual(results[name][RULES], [], name)
        self.assertEqual(set(MUTATION_CONTROLS), set(RULES))
        for rule, (kind, control) in MUTATION_CONTROLS.items():
            with self.subTest(rule):
                without = tuple(r for r in RULES if r != rule)
                if kind == 'spawn':
                    self.assertNotEqual(spawn_findings(SPAWN_FLAG[control][0]), [])
                    with tempfile.TemporaryDirectory() as directory:
                        base = Path(directory)
                        _write_tree(base, text)
                        (base / 'Tests/ComprehensiveTests/Bad.swift').write_text('let p = Process()\n')
                        self.assertNotEqual(scan_tree(base, RULES)[0], [])
                        self.assertEqual(scan_tree(base, without)[0], [])
                elif kind == 'selector':
                    with tempfile.TemporaryDirectory() as directory:
                        base = Path(directory)
                        _write_tree(base, text)
                        (base / 'Tests/ComprehensiveTests/Bad.swift').write_text('args += ["--skip", "x"]\n')
                        self.assertNotEqual(scan_tree(base, RULES)[0], [])
                        self.assertEqual(scan_tree(base, without)[0], [])
                else:
                    self.assertNotEqual(results[control][RULES], [])
                    self.assertEqual(results[control][without], [], f'disabling {rule} must clear {control}')

    def test_live_tree_is_clean(self):
        findings, call_files = scan_tree(ROOT)
        self.assertEqual(findings, [], '\n'.join(findings))
        self.assertGreaterEqual(len(call_files), MIN_CALL_FILES)
        for path, (count, _) in SELECTOR_ALLOWANCES.items():
            self.assertTrue((ROOT / path).is_file(), path)
        for path in SPAWN_ALLOWANCES:
            self.assertTrue((ROOT / path).is_file(), path)


if __name__ == '__main__':
    unittest.main()
