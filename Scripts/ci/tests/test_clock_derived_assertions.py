"""Policy (HEURISTIC): a direct assertion must not depend on a wall-clock value.

Performance predicates evaluate only under VECTORCORE_STRICT_PERF=1, through
`expectPerformance` / `expectXCTestPerformance` (Tests/ComprehensiveTests/PerfGate.swift) and the
inventory Scripts/ci/performance_sites.json. The inventory policy (check_performance_policy.py)
only sees inventoried sites; this scan finds the ones nobody inventoried. A direct `#expect(...)`,
`#require(...)` or `XCTAssert...(...)` whose argument mentions a clock-derived name fails unless
the (file, test, normalized expression) is in DEADLINE_ALLOWLIST, which holds reviewed deadline /
liveness bounds that are correctness checks rather than performance claims.

Scope and limits (lexical heuristic on the shared lexer Scripts/ci/swift_lexing.py; this is not
Swift dataflow analysis, and nothing here is proven beyond the named controls below):

* CLOCK SOURCES (CLOCK_SOURCE, matched on source with strings and comments blanked):
  `Date()`, `NSDate()`, `Date.now`, `<identifier containing "clock">[()].now` (clock.now,
  ContinuousClock().now, myClock.now), `ContinuousClock`, `SuspendingClock`, `DispatchTime`
  (so `DispatchTime.now()`), `DispatchWallTime`, `.timeIntervalSince(`, `timeIntervalSinceNow`,
  `timeIntervalSinceReferenceDate` (also on a stored Date: conservative), `.uptimeNanoseconds`,
  `<*clock*>.measure(`/`{`, `CFAbsoluteTimeGetCurrent`, `mach_absolute_time`, `clock_gettime*`
  (incl. `clock_gettime_nsec_np`), `CACurrentMediaTime`, `systemUptime`. A bare `.now` (for example
  `Mode.now`, `foo.now()` or a `timeout: .now() + 15` deadline argument) is NOT a source.
  `Duration` arithmetic is covered transitively: a Duration is a name derived from a clock result.
  A bare `timeIntervalSince1970` on a stored Date is NOT a source (`Date().timeIntervalSince1970` is,
  through `Date()`); `x.measure(` is a source only when the base name contains `clock` (XCTest
  `measure` is policed by check_performance_policy.py).
* DERIVATION: a name is clock-derived if a `let`/`var` initializer, a tuple-destructuring
  initializer (every destructured name), an assignment (`=`, a compound `+=` `-=` `*=` `/=`, or a
  subscript target `x[i] =`), a mutator call on it (`x.append(…)`, `insert`, `merge`, `formUnion`,
  `add*`, `record*`, `push*`, `enqueue*`, `store*`, `update*`, `observe*`, `accumulate*`) whose
  argument mentions a clock source or clock-derived name, or a function / computed property
  whose RETURN expressions (or whose single-statement body) do, is clock-derived; transitive to a
  fixpoint. A function that only mentions a clock in a deadline loop (no clock in any `return`)
  does not taint its callers' results.
  Scope rules are those of the output-path policy: a local declaration taints its name only inside
  its nearest local scope; type members, functions and `self.x =` targets are file-wide and also
  match after any `.` (name-based, so an unrelated same-named member also fires). Closure bodies
  that initialise or are assigned to a name count as its expression.
* ASSERTION HELPERS: a function whose `#expect`/`#require`/`XCTAssert*` mentions one of its
  parameters is a helper; each call `helper(…)` (any `base.`, by name only, no overload
  resolution) maps arguments to parameters by position (unlabelled) or external label, and is
  flagged AT THE CALL SITE if a clock-derived argument feeds an asserted parameter. Not followed:
  helpers that call other helpers, default arguments, closures passed as parameters.
* `return` is recognised at a statement start or after `;`/`{`, so one-line bodies work. A
  single-statement body (closer-only lines ignored, also when the body sits inside its header's
  statement) is an implicit return.
  A trailing closure (`f { … }`, `f() { … }`, `f(1) { … }`) is the argument of the helper's asserted
  closure-typed parameter (a parameter whose type contains `->`); it is flagged when the closure body
  mentions a clock source or clock-derived name. `init` parameters are followed as a helper named
  after the enclosing type (`S(d)`); `Self(d)` and `.init(d)` are not.
* Trailing-closure limits: with several trailing closures only the FIRST is inspected
  (`f { 1.0 } g: { Date()… }` is not seen); a call inside an if/for/while/guard/switch/else-if head
  never takes the following `{` as a trailing closure. `S.init(d)` (like `Self(d)` and `.init(d)`)
  is not followed. Optional-typed `let v: Date? = .now` and a spaced `. now` are not recognised.
* `get { … }` accessors: a single-statement `get` body is the property's implicit return.
* TYPE-INFERRED `.now`: `let v: T = .now` is a clock read when T is Date, DispatchTime,
  DispatchWallTime, ContinuousClock.Instant or SuspendingClock.Instant. LIMIT: `timer.now` on a
  receiver whose name does not contain "clock", and `.now` in assignments or returns whose type is
  only implied by context, are not sources.
* RULE: the full masked call text of `#expect(`, `#require(` and `XCTAssert*(` is checked, so
  message strings never count (they are blanked) but a non-literal message argument would.
* INTERPOLATION (T17 decision, deliberately out of scope): this policy keeps the default `_mask`, which blanks code
  inside string interpolation, because its rule is that assertion message strings never count. Opting in to
  `keep_interpolations=True` would make `#expect(ok, "took \\(Date().timeIntervalSince(t0))")` a false positive (a
  message that merely prints a clock). The cost: a clock read written only inside an interpolation of an
  asserted string argument is not seen. Pinned by test_elapsed_time_printed_but_not_asserted_is_accepted.
* Values measured INSIDE another module or helper that this file only receives (for example
  `profile.throughput` from a production profiler) contain no clock token here and are invisible;
  KNOWN_INVISIBLE_INVENTORIED pins the exact inventoried sites that this implies.
* Name-based over-approximation: a clock value passed to a constructor (`timestamp: Date()`) taints
  the constructed value, so unrelated assertions on it also hit and need a reviewed allowlist entry.
* Not seen: taint through parameters, `switch`/`for` bindings, `XCTFail`/`Issue.record` under a
  clock `if`, `#expect(throws:)`, taint across files, raw strings, `&&` of a clock with a
  helper-hidden Boolean. A helper function that merely MENTIONS a clock in its body (for example a
  polling loop that returns a count) taints every call to it: that is conservative and reviewed
  through the allowlist, not suppressed.
"""

import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location('swift_lexing', ROOT / 'Scripts/ci/swift_lexing.py')
lexing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lexing)
_Scopes, _classify, _mask, _statements = lexing._Scopes, lexing._classify, lexing._mask, lexing._statements
IDENTIFIER, FUNCTION_HEADER = lexing.IDENTIFIER, lexing.FUNCTION_HEADER

CLOCK_SOURCE = re.compile(
    r'\bDate\s*\(\s*\)|\bNSDate\s*\(\s*\)|\bDate\s*\.\s*now\b'
    r'|\b\w*[Cc]lock\w*\s*(?:\(\s*\))?\s*\.\s*now\b'
    r'|\.timeIntervalSince\s*\(|timeIntervalSinceNow|timeIntervalSinceReferenceDate'
    r'|\bDispatchTime\b|\bDispatchWallTime\b|\.uptimeNanoseconds'
    r'|\bContinuousClock\b|\bSuspendingClock\b|\b\w*[Cc]lock\w*\.measure\s*[({]'
    r'|\bCFAbsoluteTimeGetCurrent\b|\bmach_absolute_time\b|\bclock_gettime\w*|\bCACurrentMediaTime\b'
    r'|\bsystemUptime\b')
INIT_HEADER = re.compile(r'\binit\s*[?!]?\s*\(')
TYPE_NAME = re.compile(r'\b(?:struct|class|enum|actor|extension)\s+(\w+)')
GET_HEADER = re.compile(r'\s*get(?:\s+(?:async|throws))*\s*$')
CLOCK_TYPE_NOW = re.compile(r'\b(?:let|var)\s+(\w+)\s*:\s*(?:Date|DispatchTime|DispatchWallTime|ContinuousClock\.Instant'
                            r'|SuspendingClock\.Instant)\s*=\s*\.now\b')
CONTROL_HEAD = re.compile(r'\s*(?:else\s+)?(?:if|for|while|guard|switch|repeat)\b')
RETURN = re.compile(r'(?:^|[;{])\s*(return)\b')
ASSERTION = re.compile(r'(?<![\w#])#(?:expect|require)\s*\(|\bXCTAssert\w*\s*\(')
DECLARATION = re.compile(r'(?=\b(?:let|var)\s+(\w+)\s*(?::[^={]+)?=(.*))', re.S)
TUPLE_DECLARATION = re.compile(r'(?=\b(?:let|var)\s*\(([\w\s,_]+)\)\s*(?::[^={]+)?=(.*))', re.S)
DECLARATION_NAME = re.compile(r'\b(?:let|var)\s+(\w+)')
ASSIGNMENT = re.compile(r'(?:^|[;{])\s*((?:\w[\w.]*\.)?)(\w+)(?:\s*\[[^\]]*\])*\s*(?:[-+*/])?=(?!=)(.*)', re.S)
# `name.append(x)`, `name.insert(x)`, ...: a call that plausibly stores its argument in the receiver.
MUTATOR = re.compile(r'(?:^|[;{])\s*((?:\w[\w.]*\.)?)(\w+)(?:\s*\[[^\]]*\])*\s*\.\s*'
                     r'(?:append|insert|merge|formUnion|add|record|push|enqueue|store|update|observe|accumulate)\w*\s*\((.*)', re.S)
CLOSURE_OPENER = re.compile(r'\{\s*(?:(?:\[[^\]]*\]\s*)?(?:\([^)]*\)\s*(?:->\s*[^\s{]+\s*)?|[\w,\s:_]*?\s)?in)?\s*$')
CONTROL_FLOW = re.compile(r'\b(?:if|guard|while|for|switch|case)\b')


def _argument_end(masked, start):
    """Offset of the `)` closing the call whose arguments start at `start` (masked text)."""
    depth = 0
    for i in range(start, len(masked)):
        if masked[i] in '([{':
            depth += 1
        elif masked[i] in ')]}':
            if depth == 0:
                return i
            depth -= 1
    return len(masked)


def _first_argument_end(masked, start, end):
    depth = 0
    for i in range(start, end):
        if masked[i] in '([{':
            depth += 1
        elif masked[i] in ')]}':
            depth -= 1
        elif masked[i] == ',' and depth == 0:
            return i
    return end


def _expression_end(masked, start, end):
    """End of the expression starting at `start`: the first top-level `;` (or unmatched closer) before `end`."""
    depth = 0
    for i in range(start, end):
        if masked[i] in '([{':
            depth += 1
        elif masked[i] in ')]}':
            depth -= 1
            if depth < 0:
                return i
        elif masked[i] == ';' and depth == 0:
            return i
    return end


def _split_arguments(masked, start, end):
    """[(start, end)] of the top-level comma-separated arguments in masked[start:end]."""
    parts, depth, begin = [], 0, start
    for i in range(start, end):
        if masked[i] in '([{':
            depth += 1
        elif masked[i] in ')]}':
            depth -= 1
        elif masked[i] == ',' and depth == 0:
            parts.append((begin, i))
            begin = i + 1
    if masked[begin:end].strip():
        parts.append((begin, end))
    return parts


def _compact(text):
    return re.sub(r'\s+', '', text)


def _enclosing_test(source, masked, scopes, offset):
    """Name of the outermost enclosing `func` (a test method, normally), else '<file>'."""
    scope, found = scopes.at[offset], '<file>'
    while scope:
        if scopes.name[scope]:
            lead = max(masked.rfind(c, 0, scopes.opens[scope]) for c in '{};')
            header = FUNCTION_HEADER.search(masked, lead + 1, scopes.opens[scope])
            if header:
                found = header.group(1)
        scope = scopes.parent[scope]
    return found


def clock_hits(source, why=None):
    """[(line, test, normalized expression)] for direct assertions that mention a clock-derived value."""
    source = source.replace('\r\n', '\n').replace('\r', '\n')
    masked = _mask(source)
    statements = _statements(masked)
    line_start = [0]
    for line in source.split('\n'):
        line_start.append(line_start[-1] + len(line) + 1)
    scopes = _Scopes(source)
    tainted, members = {}, set()

    def mentions(start, end):
        if CLOCK_SOURCE.search(masked, start, end):
            return True
        for match in IDENTIFIER.finditer(masked, start, end):
            name = match.group()
            kind = _classify(masked, match.start(), match.end())
            if kind == 'member':
                if name in members:
                    return True
            elif kind == 'use' and tainted.get(name, set()) & scopes.ancestors[scopes.at[match.start()]]:
                return True
        return False

    def is_member_site(offset):
        scope = scopes.at[offset]
        return bool(scope) and scopes.transparent[scope]

    opens_at = {opened: scope for scope, opened in enumerate(scopes.opens) if scope}

    def closure_end(text, offset, expression, keyword, end):
        end = _expression_end(masked, offset + expression, end)
        rest = text[expression:end - offset]
        bare = rest.lstrip()
        scope = None
        if bare.startswith('{'):
            scope = opens_at.get(end - len(bare))
        elif not bare and end < offset + len(text):
            scope = None  # a `;` ended the expression: nothing follows it that belongs to it
        elif not bare and not CONTROL_FLOW.search(text[:keyword]):
            following = len(masked) - len(masked[offset + len(text):].lstrip())
            scope = opens_at.get(following)
        elif not CONTROL_FLOW.search(text[:keyword]):
            opener = CLOSURE_OPENER.search(text, expression, end - offset)
            scope = opens_at.get(offset + opener.start()) if opener else None
        return max(end, scopes.closes[scope] + 1) if scope else end

    declared, candidates = [], []
    for number, text in statements:
        offset = line_start[number - 1]
        for declaration in DECLARATION_NAME.finditer(text):
            at = offset + declaration.start()
            declared.append((declaration.group(1), scopes.declaring(at), is_member_site(at)))
        for declaration in DECLARATION.finditer(text):
            at = offset + declaration.start()
            end = closure_end(text, offset, declaration.start(2), declaration.start(), offset + len(text))
            candidates.append((declaration.group(1), scopes.declaring(at), is_member_site(at),
                               offset + declaration.start(2), end))
        for declaration in TUPLE_DECLARATION.finditer(text):
            at = offset + declaration.start()
            for name in re.findall(r'\w+', declaration.group(1)):
                if name != '_':
                    declared.append((name, scopes.declaring(at), is_member_site(at)))
                    candidates.append((name, scopes.declaring(at), is_member_site(at),
                                       offset + declaration.start(2), offset + len(text)))
        for assignment in list(ASSIGNMENT.finditer(text)) + list(MUTATOR.finditer(text)):
            name, at = assignment.group(2), offset + assignment.start(2)
            if name == '_':
                continue
            visible = scopes.ancestors[scopes.at[at]]
            homes = [d for d in declared if d[0] == name and d[1] in visible]
            if assignment.group(1) or not homes:
                scope, member = 0, True
            else:
                scope, member = max(homes, key=lambda d: d[1])[1:]
            end = closure_end(text, offset, assignment.start(3), assignment.start(), offset + len(text))
            candidates.append((name, scope, member, offset + assignment.start(3), end))
    def body_spans(start, end):
        spans = [(offset + m.end(), offset + len(text)) for number, text in statements
                 for offset in [line_start[number - 1]] for m in RETURN.finditer(text)
                 if start <= offset + m.start(1) < end]
        # Statements that START inside the body, ignoring closer-only lines (`}` of this body or a nested one).
        inside = [number for number, text in statements
                  if start <= line_start[number - 1] < end and text.strip(' \t;}')]
        if not spans and len(inside) <= 1:  # expression-bodied (also a one-line body inside its header statement)
            spans = [(start, end)]
        return spans

    for name, start, end in scopes.bodies():
        candidates += [(name, 0, True, a, _expression_end(masked, a, b)) for a, b in body_spans(start, end)]
    for scope in range(1, len(scopes.parent)):  # `var x: T { get { … } }`: the accessor body is the value
        owner = scopes.name[scopes.parent[scope]] if scopes.parent[scope] else None
        lead = max(masked.rfind(c, 0, scopes.opens[scope]) for c in '{};')
        if owner and GET_HEADER.match(masked[lead + 1:scopes.opens[scope]]):
            candidates += [(owner, 0, True, a, _expression_end(masked, a, b))
                           for a, b in body_spans(scopes.opens[scope] + 1, scopes.closes[scope])]
    # `let v: Date = .now`: a leading `.now` is a clock read when the annotation names a clock type.
    for number, text in statements:
        for typed in CLOCK_TYPE_NOW.finditer(text):
            at = line_start[number - 1] + typed.start()
            tainted.setdefault(typed.group(1), set()).add(scopes.declaring(at))
            if is_member_site(at):
                members.add(typed.group(1))
    changed = True
    while changed:
        changed = False
        for name, scope, member, start, end in candidates:
            if (scope not in tainted.get(name, ()) or (member and name not in members)) and mentions(start, end):
                tainted.setdefault(name, set()).add(scope)
                if why is not None:
                    why.setdefault(name, source[start:end][:100].replace('\n', ' '))
                if member:
                    members.add(name)
                changed = True

    found = []
    assertions = list(ASSERTION.finditer(masked))
    for match in assertions:
        end = _argument_end(masked, match.end())
        if mentions(match.end(), end):
            first = _first_argument_end(masked, match.end(), end)
            found.append((source.count('\n', 0, match.start()) + 1,
                          _enclosing_test(source, masked, scopes, match.start()),
                          _compact(source[match.end():first])))

    # Assertion helpers: functions and initializers whose assertion mentions one of their parameters. A
    # call whose argument for such a parameter is clock-derived is flagged at the CALL SITE.
    helpers = {}  # call name (function name or type name for init) -> [(index, label, internal, closure-typed)]
    for scope in range(1, len(scopes.parent)):
        lead = max(masked.rfind(c, 0, scopes.opens[scope]) for c in '{};')
        name = opening = None
        header = FUNCTION_HEADER.search(masked, lead + 1, scopes.opens[scope]) if scopes.name[scope] else None
        if header and header.group(1) == scopes.name[scope]:
            name, opening = header.group(1), masked.find('(', header.end(), scopes.opens[scope])
        else:
            init = INIT_HEADER.search(masked, lead + 1, scopes.opens[scope])
            owner = scopes.parent[scope]
            if init and owner:
                before = max(masked.rfind(c, 0, scopes.opens[owner]) for c in '{};')
                type_name = TYPE_NAME.search(masked, before + 1, scopes.opens[owner])
                if type_name:
                    name, opening = type_name.group(1), init.end() - 1
        if not name or opening is None or opening < 0:
            continue
        parameters = _split_arguments(masked, opening + 1, _argument_end(masked, opening + 1))
        asserted = set()
        for match in assertions:
            if scopes.opens[scope] < match.start() < scopes.closes[scope]:
                inner = match.end(), _argument_end(masked, match.end())
                asserted |= {m.group() for m in IDENTIFIER.finditer(masked, *inner)
                             if _classify(masked, m.start(), m.end()) == 'use'}
        for index, (a, b) in enumerate(parameters):
            names = re.match(r'\s*(\w+)(?:\s+(\w+))?\s*:', masked[a:b])
            if names:
                label, internal = names.group(1), names.group(2) or names.group(1)
                if internal in asserted:
                    helpers.setdefault(name, []).append((index, None if label == '_' else label, internal,
                                                         '->' in masked[a:b]))
    for name, parameters in helpers.items():
        call = re.compile(r'(?<!\w)(?:\w+\s*\.\s*)?' + re.escape(name) + r'\s*[({]')
        for match in call.finditer(masked):
            if re.search(r'\bfunc\s+$', masked[:match.start()]) or re.search(r'\bfunc\s+$', masked[:match.end() - len(name) - 1]):
                continue
            site = (source.count('\n', 0, match.start()) + 1, _enclosing_test(source, masked, scopes, match.start()))
            flagged, last = False, match.end() - 1
            if masked[match.end() - 1] == '(':
                last = _argument_end(masked, match.end())
                for position, (a, b) in enumerate(_split_arguments(masked, match.end(), last)):
                    label = re.match(r'\s*(\w+)\s*:(?!:)', masked[a:b])
                    value = a + label.end() if label else a
                    hit = any((label and label.group(1) == external) or (not label and external is None and position == index)
                              for index, external, _, _ in parameters)
                    flagged = flagged or (hit and mentions(value, b))
            # A trailing closure `f(…) { … }` / `f { … }` feeds the LAST asserted closure-typed parameter.
            tail = len(masked) - len(masked[last + 1:].lstrip()) if masked[match.end() - 1] == '(' else match.end() - 1
            head = masked[max(masked.rfind(c, 0, match.start()) for c in '\n;{}') + 1:match.start()]
            body = None if CONTROL_HEAD.match(head) else opens_at.get(tail)  # `if f({…}) { … }`: body, not closure
            if body and any(closure for *_, closure in parameters):
                flagged = flagged or mentions(scopes.opens[body] + 1, scopes.closes[body])
                last = scopes.closes[body]
            if flagged:
                found.append((*site, _compact(source[match.start():last + 1])))
    return sorted(set(found))


def scan(tests_dir, root):
    """{relative file: [(line, test, expression)]} for every Swift file under tests_dir."""
    result = {}
    for path in sorted(Path(tests_dir).rglob('*.swift')):
        hits = clock_hits(path.read_text(encoding='utf-8'))
        if hits:
            result[str(path.relative_to(root))] = hits
    return result


# Reviewed non-performance hits: (file, test, normalized expression) -> reason. Every Tests/ hit
# was classified before it was listed. Performance claims are NOT allowlisted: they are gated with
# expectPerformance and inventoried (the two found by milestone 12 were migrated by T06).
# Category prefixes: DEADLINE (a bound whose failure means a hang or leak, a correctness check; none
# remain after `.now` was narrowed, because bare `.now()` timeout arguments are not clock reads),
# INJECTED-CLOCK (a deterministic fake clock, no wall time), TIMESTAMP (a stored Date compared as
# an ordered value, not a duration), FIXTURE (name-based over-approximation: a literal-valued
# fixture that merely carries `timestamp: Date()` metadata).
_INJECTED = 'INJECTED-CLOCK: counts reads of a deterministic SequenceClock; no wall-clock value is compared'
_TIMESTAMP = ('TIMESTAMP: asserts on a value constructed with a stored Date(); orders timestamps or checks '
              'unrelated fields, never a duration')
_FIXTURE = ('FIXTURE: score computed from literal throughput/latency inputs; Date() is only the '
            'timestamp: metadata argument, so the hit is name-based over-approximation')
_V04_CALIBRATION = ('INJECTED-CLOCK: asserts harness calibration against a deterministic VirtualClock or '
                    'ScriptedClock passed as now:; no wall-clock value is compared (V04 phase C)')
_OP = 'Tests/ComprehensiveTests/OperationProgressTests.swift'
_ENTRIES = {
    **{('Tests/ComprehensiveTests/LoggingContractTests.swift', test, 'clock.readCount==' + count): _INJECTED
       for test, count in [('timerBoundaryReadings', '5'), ('timerReportsCumulativeNotLapElapsed', '4'),
                           ('timerInvalidThresholdSuppressesButStillReadsClock', '3'),
                           ('timerNegativeInjectedElapsedSuppresses', '2'),
                           ('timerSuppressedWhenDebugLevelIsDisabled', '2')]},
    ('Tests/ComprehensiveTests/ErrorHandlingTests.swift', 'testErrorContext',
     'context.timestamp>=before&&context.timestamp<=after'): _TIMESTAMP,
    **{(_OP, test, expression): _TIMESTAMP for test, expression in [
        ('testOperationProgress_DefaultTimestamp', 'progress.timestamp>=before'),
        ('testOperationProgress_DefaultTimestamp', 'progress.timestamp<=after'),
        ('testOperationProgress_Equatable', 'p1==p2'),
        ('testOperationProgress_FullInit', 'progress.current==3'),
        ('testOperationProgress_FullInit', 'progress.total==7'),
        ('testOperationProgress_FullInit', 'progress.phase=="Processing"'),
        ('testOperationProgress_FullInit', 'progress.message=="Processingitems"'),
        ('testOperationProgress_FullInit', 'progress.estimatedTimeRemaining==10.0'),
        ('testOperationProgress_FullInit', 'progress.timestamp==timestamp'),
        ('testOperationProgress_NotEquatable', 'base!=differentCurrent'),
        ('testOperationProgress_NotEquatable', 'base!=differentTotal'),
        ('testOperationProgress_NotEquatable', 'base!=differentPhase')]},
    **{('Tests/ComprehensiveTests/KernelAutoTunerTests.swift', test, expression): _FIXTURE for test, expression in [
        ('testScoreCalculation', 'score.isInfinite'), ('testScoreCalculation', 'score.isNaN'),
        ('testScoreCalculation', 'score'), ('testScoreHandlesZeroThroughput', 'badProfile.score')]},
    ('Tests/ComprehensiveTests/MixedPrecisionAutoTunerTests.swift', 'testMetricsQualityScore',
     'goodMetrics.qualityScore>poorMetrics.qualityScore'): _FIXTURE,
    **{('Tests/BenchmarkContractTests/BenchmarkSelectionContractTests.swift', test, expression): _V04_CALIBRATION
       for test, expression in [
        ('testSubTickBodyAutoScalesToAnAccurateEstimate', 'result.sampleNsPerInvocation.count'),
        ('testSlowBodyCalibratesOnOneInvocationExactlyAsBefore', 'result.iterations'),
        ('testRecalibrationPerSampleUsesTheSameRule', 'result.sampleIterations.count'),
        ('testRecalibrationPerSampleUsesTheSameRule', 'result.sampleNsPerInvocation.allSatisfy{abs($0-5.0)<0.1}'),
        ('testClockThatNeverAdvancesHitsTheCapAndFailsExplicitly',
         'tryHarness.measure(name:"frozen",minTimeSeconds:0,repeats:nil,calibrationTickNanoseconds:42,now:clock.now){clock.body()}'),
        ('testExplicitRepeatsOneIsExactlyOneInvocationWithAnActionableMessage',
         'tryHarness.measure(name:"fixed",minTimeSeconds:0,repeats:1,calibrationTickNanoseconds:42,now:clock.now){clock.body()}'),
        ('testAsyncHarnessUsesTheSameCalibration', 'result.sampleNsPerInvocation.allSatisfy{abs($0-5.0)<0.1}'),
        ('testImplausiblyLargeIntervalThrowsInsteadOfTrapping',
         'tryHarness.measure(name:"jump",minTimeSeconds:0,repeats:nil,calibrationTickNanoseconds:42,now:clock.now){clock.body()}')]},
}

# Pinned hit counts: every entry is exactly one hit unless listed here.
PINNED_COUNTS = {}
DEADLINE_ALLOWLIST = {key: (reason, PINNED_COUNTS.get(key, 1)) for key, reason in _ENTRIES.items()}

# Inventoried performance sites that this lexical scan cannot see when un-gated, because the measured
# value comes from production code (no clock token in the test file). Pinned exactly: a change in
# either direction must be reviewed (test_recall_on_inventoried_sites).
KNOWN_INVISIBLE_INVENTORIED = {
    'KernelAutoTunerTests.testProfileWorkload512.1', 'KernelAutoTunerTests.testProfileWorkload512.2',
    'MixedPrecisionComprehensiveTests.testBenchmarkDotProduct.1',
    'MixedPrecisionComprehensiveTests.testBenchmarkDotProduct.2',
    'MixedPrecisionComprehensiveTests.testBenchmarkBatchOperations.3',
    'MixedPrecisionComprehensiveTests.testBenchmarkBatchOperations.4',
    'MixedPrecisionKernelTests.testPerformanceMetricsCollection.1',
}


def unreviewed(hits_by_file, allowlist):
    """(hits not allowlisted, allowlist keys that matched nothing, keys whose hit count differs).

    allowlist values are (reason, pinned hit count); a second identical expression in the same test
    therefore cannot hide behind one entry.
    """
    counts, bad = {}, []
    for file, hits in hits_by_file.items():
        for line, test, expression in hits:
            key = (file, test, expression)
            counts[key] = counts.get(key, 0) + 1
            if key not in allowlist:
                bad.append((file, line, test, expression))
    miscounted = sorted((key, counts[key], allowlist[key][1]) for key in allowlist
                        if key in counts and counts[key] != allowlist[key][1])
    return bad, sorted(set(allowlist) - set(counts)), miscounted


class ClockDerivedAssertionTests(unittest.TestCase):
    def hits(self, source):
        return [(test, expression) for _, test, expression in clock_hits(source)]

    # ---- repository scan ----
    def test_repository_has_no_unreviewed_clock_derived_assertion(self):
        bad, stale, miscounted = unreviewed(scan(ROOT / 'Tests', ROOT), DEADLINE_ALLOWLIST)
        self.assertEqual(miscounted, [], 'hit count differs from the pinned count')
        self.assertEqual(bad, [], 'gate with expectPerformance + performance_sites.json, or review into the allowlist')
        self.assertEqual(stale, [], 'allowlist entry no longer matches a hit; remove it')

    def test_allowlist_entries_are_reasoned_and_never_inventoried_performance_claims(self):
        import json
        inventory = json.loads((ROOT / 'Scripts/ci/performance_sites.json').read_text())['assertions']
        claimed = {(row['file'], row['test'], _compact(row['expression'])) for row in inventory}
        for key, (reason, count) in DEADLINE_ALLOWLIST.items():
            self.assertGreaterEqual(count, 1)
            self.assertTrue(reason.split(':')[0] in ('DEADLINE', 'INJECTED-CLOCK', 'TIMESTAMP', 'FIXTURE'), key)
            self.assertNotIn(key, claimed)
            self.assertTrue((ROOT / key[0]).is_file(), key)

    def test_migrated_sites_are_gated_calls_and_no_longer_direct_hits(self):
        hits = scan(ROOT / 'Tests', ROOT)
        for relative, site, call in [
                ('Tests/ComprehensiveTests/MixedPrecisionKernelTests.swift',
                 'MixedPrecisionKernelTests.testAppleSiliconNEON.17', 'gflops>0.1,'),
                ('Tests/ComprehensiveTests/QuantizedKernelsTests.swift',
                 'QuantizedKernelsTests.testQuantizationOverhead.11', 'ratio>1.0,')]:
            text = _compact((ROOT / relative).read_text())
            self.assertEqual(text.count('expectPerformance("' + site + '",' + call), 1)
            self.assertNotIn(relative, hits)

    def test_recall_on_inventoried_sites(self):
        """Un-gate every inventoried site (helper call -> direct assertion): the scan must see it."""
        import json
        inventory = json.loads((ROOT / 'Scripts/ci/performance_sites.json').read_text())['assertions']
        missed = set()
        for row in inventory:
            if 'deterministic' in row['category']:
                continue
            text = (ROOT / row['file']).read_text()
            helper = 'expectXCTestPerformance' if row['framework'] == 'XCTest' else 'expectPerformance'
            token = helper + '("' + row['id'] + '", '
            self.assertEqual(text.count(token), 1, row['id'])
            line = text[:text.index(token)].count('\n') + 1
            direct = text.replace(token, 'XCTAssertTrue(' if row['framework'] == 'XCTest' else '#expect(')
            if line not in {number for number, _, _ in clock_hits(direct)}:
                missed.add(row['id'])
        self.assertEqual(missed, KNOWN_INVISIBLE_INVENTORIED)

    # ---- negative controls: each must be flagged ----
    def test_time_ratio_is_flagged(self):
        source = ('func testRatio() {\n let a0 = Date()\n work()\n let a = Date().timeIntervalSince(a0)\n'
                  ' let b0 = Date()\n work2()\n let b = Date().timeIntervalSince(b0)\n let ratio = a / b\n'
                  ' #expect(ratio > 1.0, "faster")\n}\n')
        self.assertEqual(self.hits(source), [('testRatio', 'ratio>1.0')])

    def test_gflops_from_seconds_is_flagged(self):
        source = ('func testNeon() {\n let s = CFAbsoluteTimeGetCurrent()\n run()\n'
                  ' let seconds = CFAbsoluteTimeGetCurrent() - s\n let gflops = flops / seconds / 1e9\n'
                  ' if extended { #expect(gflops > 0.1) }\n}\n')
        self.assertEqual(self.hits(source), [('testNeon', 'gflops>0.1')])

    def test_dispatch_time_difference_is_flagged(self):
        source = ('func testDispatch() {\n let s = DispatchTime.now()\n run()\n let e = DispatchTime.now()\n'
                  ' let nanos = e.uptimeNanoseconds - s.uptimeNanoseconds\n #expect(nanos < 1_000_000)\n}\n')
        self.assertEqual(self.hits(source), [('testDispatch', 'nanos<1_000_000')])

    def test_continuous_clock_duration_comparison_is_flagged(self):
        measured = 'func t() {\n let clock = ContinuousClock()\n let d = clock.measure { run() }\n #expect(d < .seconds(1))\n}\n'
        self.assertEqual(self.hits(measured), [('t', 'd<.seconds(1)')])
        instants = ('func t() {\n let c = ContinuousClock()\n let s = c.now\n run()\n let d = c.now - s\n'
                    ' XCTAssertLessThan(d, .seconds(1))\n}\n')
        self.assertEqual(self.hits(instants), [('t', 'd')])

    def test_transitive_derivation_through_two_lets(self):
        source = ('func testChain() {\n let t = Date().timeIntervalSince(start)\n let rate = 100 / t\n'
                  ' let ok = rate > 5\n #expect(ok)\n}\n')
        self.assertEqual(self.hits(source), [('testChain', 'ok')])

    def test_collection_function_tuple_and_closure_derivation_is_flagged(self):
        appended = ('func a() {\n var xs: [Double] = []\n let s = Date()\n xs.append(Date().timeIntervalSince(s))\n'
                    ' #expect(xs.max()! < 1)\n}\n')
        self.assertEqual(self.hits(appended), [('a', 'xs.max()!<1')])
        returned = ('func timed() -> Double {\n let s = Date()\n run()\n return Date().timeIntervalSince(s)\n}\n'
                    'func b() {\n let t = timed()\n #expect(t < 1)\n}\n')
        self.assertEqual(self.hits(returned), [('b', 't<1')])
        destructured = 'func c() {\n let (v, t) = (compute(), Date().timeIntervalSince(s))\n #expect(t < 1)\n}\n'
        self.assertEqual(self.hits(destructured), [('c', 't<1')])
        closure = 'func d() {\n let t: Double = {\n  Date().timeIntervalSince(s)\n }()\n #require(t < 1)\n}\n'
        self.assertEqual(self.hits(closure), [('d', 't<1')])

    def test_xctest_assertion_forms_are_flagged(self):
        source = 'func testX() {\n let t = Date().timeIntervalSince(s)\n XCTAssertLessThan(t, 1.0, "slow")\n}\n'
        self.assertEqual(self.hits(source), [('testX', 't')])

    def test_scope_is_respected(self):
        source = ('func one() {\n let t = Date().timeIntervalSince(s)\n print(t)\n}\n'
                  'func two() {\n let t = 5\n #expect(t < 10)\n}\n')
        self.assertEqual(self.hits(source), [])

    # ---- positive controls: each must be accepted ----
    def test_elapsed_time_printed_but_not_asserted_is_accepted(self):
        source = ('func testPrint() {\n let s = Date()\n run()\n let e = Date().timeIntervalSince(s)\n'
                  ' print("took \\(e)")\n #expect(result == 3, "elapsed \\(e)")\n}\n')
        self.assertEqual(self.hits(source), [])
        # T17 LIMITS example: a clock read written directly inside the message interpolation is also accepted.
        direct = 'func testDirect() {\n let t0 = Date()\n run()\n #expect(ok, "took \\(Date().timeIntervalSince(t0))")\n}\n'
        self.assertEqual(self.hits(direct), [])

    def test_assertion_on_non_clock_values_is_accepted(self):
        self.assertEqual(self.hits('func t() {\n let n = compute()\n #expect(n == 3)\n XCTAssertEqual(n, 3)\n}\n'), [])

    def test_expect_performance_usage_is_accepted(self):
        source = ('func testGate() {\n let t = Date().timeIntervalSince(s)\n'
                  ' expectPerformance("X.testGate.1", t < 1.0, "slow")\n'
                  ' expectXCTestPerformance("X.testGate.2", t < 1.0)\n}\n')
        self.assertEqual(self.hits(source), [])

    def test_allowlisted_deadline_bound_is_accepted_and_stale_entries_fail(self):
        source = ('func child() {\n let timedOut = sem.wait(timeout: DispatchTime.now() + 15) == .timedOut\n'
                  ' #expect(!timedOut, "deadline")\n}\n')
        found = {'Tests/X.swift': clock_hits(source)}
        self.assertEqual(unreviewed(found, {})[0], [('Tests/X.swift', 3, 'child', '!timedOut')])
        allowed = {('Tests/X.swift', 'child', '!timedOut'): ('DEADLINE: test', 1)}
        self.assertEqual(unreviewed(found, allowed), ([], [], []))
        self.assertEqual(unreviewed({}, allowed), ([], [('Tests/X.swift', 'child', '!timedOut')], []))

    def test_a_function_that_only_mentions_a_clock_in_a_deadline_loop_does_not_taint_callers(self):
        source = ('func launch() -> Int {\n let d = Date()\n while Date().timeIntervalSince(d) < 5 { poll() }\n return 0\n}\n'
                  'func t() {\n let status = launch()\n #expect(status == 0)\n}\n')
        self.assertEqual(self.hits(source), [])

    # ---- fix round 1 controls ----
    def test_implicit_return_bodies_inside_types_are_tainted(self):
        function = ('struct S { func speed() -> Double {\n Date().timeIntervalSince(start)\n }\n'
                    ' @Test func t() { #expect(speed() < 1) } }\n')
        self.assertEqual(self.hits(function), [('t', 'speed()<1')])
        computed = ('struct S { var elapsed: Double { Date().timeIntervalSince(start) }\n'
                    ' @Test func t() { #expect(elapsed < 1) } }\n')
        self.assertEqual(self.hits(computed), [('t', 'elapsed<1')])
        multiline = ('struct S {\n var elapsed: Double {\n  Date().timeIntervalSince(start)\n }\n'
                     ' func t() {\n  #expect(elapsed < 1)\n }\n}\n')
        self.assertEqual(self.hits(multiline), [('t', 'elapsed<1')])
        untimed = 'struct S { func n() -> Int {\n 3\n }\n func t() { #expect(n() < 4) } }\n'
        self.assertEqual(self.hits(untimed), [])

    def test_semicolon_return_is_a_return(self):
        source = 'func speed() -> Double { let s = 1; return Date().timeIntervalSince(start) }\nfunc t() { #expect(speed() < 1) }\n'
        self.assertEqual(self.hits(source), [('t', 'speed()<1')])

    def test_assertion_helper_call_sites_are_flagged_by_argument_flow(self):
        helper = 'func assertFast(_ e: Double) { #expect(e < 1) }\n'
        flagged = helper + 'func t() {\n let s = Date()\n assertFast(Date().timeIntervalSince(s))\n}\n'
        self.assertEqual(self.hits(flagged), [('t', 'assertFast(Date().timeIntervalSince(s))')])
        indirect = helper + 'func t() {\n let d = Date().timeIntervalSince(s)\n self.assertFast(d)\n}\n'
        self.assertEqual(self.hits(indirect), [('t', 'self.assertFast(d)')])
        labelled = ('func check(label: String, within seconds: Double) { #expect(seconds < 1, "\\(label)") }\n'
                    'func t() {\n let d = Date().timeIntervalSince(s)\n check(label: "x", within: d)\n}\n')
        self.assertEqual(self.hits(labelled), [('t', 'check(label:"x",within:d)')])

    def test_assertion_helper_with_non_clock_flow_is_accepted(self):
        helper = 'func assertSmall(_ n: Int, note: Double) { #expect(n < 4) }\n'
        self.assertEqual(self.hits(helper + 'func t() {\n assertSmall(3, note: 0)\n}\n'), [])
        # a clock value passed only to a parameter the helper does not assert
        passed = helper + 'func t() {\n let d = Date().timeIntervalSince(s)\n assertSmall(3, note: d)\n}\n'
        self.assertEqual(self.hits(passed), [])

    def test_additional_clock_sources_are_flagged(self):
        for expression in ['clock_gettime_nsec_np(CLOCK_UPTIME_RAW)', 'Date.timeIntervalSinceReferenceDate',
                           'someDate.timeIntervalSinceReferenceDate', 'NSDate().timeIntervalSince1970']:
            with self.subTest(expression=expression):
                source = 'func t() {\n let v = %s\n #expect(v > 0)\n}\n' % expression
                self.assertEqual(self.hits(source), [('t', 'v>0')])

    def test_now_is_narrowed_to_clock_receivers(self):
        for expression in ['Date.now', 'DispatchTime.now()', 'ContinuousClock.now', 'ContinuousClock().now',
                           'SuspendingClock().now', 'clock.now', 'myClock.now', 'wallclock.now']:
            with self.subTest(expression=expression):
                source = 'func t() {\n let v = %s\n #expect(v > x)\n}\n' % expression
                self.assertEqual(self.hits(source), [('t', 'v>x')])
        for expression in ['Mode.now', 'foo.now()', 'sem.wait(timeout: .now() + 1)']:
            with self.subTest(expression=expression):
                source = 'func t() {\n let v = %s\n #expect(v > x)\n}\n' % expression
                self.assertEqual(self.hits(source), [])

    def test_allowlist_count_is_pinned(self):
        found = {'Tests/X.swift': [(3, 't', 'a<1'), (4, 't', 'a<1')]}
        key = ('Tests/X.swift', 't', 'a<1')
        self.assertEqual(unreviewed(found, {key: ('TIMESTAMP: x', 1)}), ([], [], [(key, 2, 1)]))
        self.assertEqual(unreviewed(found, {key: ('TIMESTAMP: x', 2)}), ([], [], []))

    # ---- fix round 2 controls ----
    TRAILING = 'func assertFast(_ body: () -> Double) { #expect(body() < 1) }\n'

    def test_trailing_closure_into_an_asserted_closure_parameter_is_flagged(self):
        cases = {
            'assertFast { Date().timeIntervalSince(s) }': 'assertFast{Date().timeIntervalSince(s)}',
            'assertFast() { Date().timeIntervalSince(s) }': 'assertFast(){Date().timeIntervalSince(s)}',
            'assertFast { d }': 'assertFast{d}',
        }
        for call, expression in cases.items():
            with self.subTest(call=call):
                source = self.TRAILING + 'func t() {\n let d = Date().timeIntervalSince(s)\n ' + call + '\n}\n'
                self.assertEqual(self.hits(source), [('t', expression)])
        positional = ('func af(_ n: Int, _ body: () -> Double) { #expect(body() < Double(n)) }\n'
                      'func t() {\n af(1) { Date().timeIntervalSince(s) }\n}\n')
        self.assertEqual(self.hits(positional), [('t', 'af(1){Date().timeIntervalSince(s)}')])

    def test_trailing_closure_returning_a_non_clock_value_is_accepted(self):
        source = self.TRAILING + 'func t() {\n let d = 0.5\n assertFast { d }\n assertFast() { 0.25 }\n}\n'
        self.assertEqual(self.hits(source), [])
        unasserted = ('func run(_ body: () -> Double) { _ = body() }\n'
                      'func t() {\n run { Date().timeIntervalSince(s) }\n}\n')
        self.assertEqual(self.hits(unasserted), [])

    def test_initializer_is_followed_as_an_assertion_helper(self):
        source = ('struct S {\n init(_ e: Double) { #expect(e < 1) }\n}\n'
                  'func t() {\n let d = Date().timeIntervalSince(s)\n _ = S(d)\n}\n')
        self.assertEqual(self.hits(source), [('t', 'S(d)')])
        clean = ('struct S {\n init(_ e: Double) { #expect(e < 1) }\n}\n'
                 'func t() {\n _ = S(0.5)\n}\n')
        self.assertEqual(self.hits(clean), [])

    def test_type_inferred_now_with_a_clock_annotation_is_flagged(self):
        for annotation in ['Date', 'DispatchTime', 'DispatchWallTime', 'ContinuousClock.Instant',
                           'SuspendingClock.Instant']:
            with self.subTest(annotation=annotation):
                source = 'func t() {\n let v: %s = .now\n #expect(v > x)\n}\n' % annotation
                self.assertEqual(self.hits(source), [('t', 'v>x')])
        # Documented limit: a non-clock-named receiver or an unrelated annotation is not a source.
        self.assertEqual(self.hits('func t() {\n let v = timer.now\n #expect(v > x)\n}\n'), [])
        self.assertEqual(self.hits('func t() {\n let v: Mode = .now\n #expect(v == x)\n}\n'), [])

    def test_implicit_return_get_accessor_is_tainted(self):
        source = ('struct S {\n var elapsed: Double {\n  get {\n   Date().timeIntervalSince(start)\n  }\n }\n'
                  ' func t() {\n  #expect(elapsed < 1)\n }\n}\n')
        self.assertEqual(self.hits(source), [('t', 'elapsed<1')])
        clean = ('struct S {\n var n: Int {\n  get {\n   3\n  }\n }\n func t() {\n  #expect(n < 4)\n }\n}\n')
        self.assertEqual(self.hits(clean), [])

    def test_helper_call_in_a_control_flow_head_does_not_take_the_body_as_a_closure(self):
        for head in ['if assertFast({ 0.5 }) {', 'for x in xs where assertFast({ 0.5 }) {', 'while assertFast({ 0.5 }) {']:
            with self.subTest(head=head):
                source = (self.TRAILING + 'func t() {\n let d = Date().timeIntervalSince(s)\n ' + head
                          + '\n  print(d)\n }\n}\n')
                self.assertEqual(self.hits(source), [])
        # the same shape as a statement IS a trailing closure
        source = self.TRAILING + 'func t() {\n let d = Date().timeIntervalSince(s)\n assertFast { d }\n}\n'
        self.assertEqual(self.hits(source), [('t', 'assertFast{d}')])


if __name__ == '__main__':
    unittest.main()
