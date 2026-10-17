"""Policy (T15, lost-store class closure): no direct read of a known-value local after a mutable-pointer closure.

WHY: Swift 6.4.0 (swiftlang-6.4.0.34.1, Xcode 27.0) miscompiles this shape at -O: SIL RedundantLoadElimination
forwards the value a local held before `withUnsafeMutableBufferPointer` / `withUnsafeMutablePointer(to:)` across
the closure call when the closure stays out of line and hands the pointer to a callee through an `inout` argument,
so a read after the closure returns the old value although memory holds the new one. Upstream issue
swiftlang/swift#92562 (rdar://188227399); PR #92580 and its 6.4.x backports (#92630, #92631, #92632) are believed to fix
this mechanism (unverified for this form; see T15/lost-store-investigation). Found by
`MatrixMetricContractTests.zeroQuerySoASuffixBehavioralRegression` in Release (T15 lost-store investigation); the
pure-Swift reproducer is the Release canary Tests/ComprehensiveTests/LostStoreCanaryTests.swift.
Whether the trigger fires depends on inlining, so unrelated edits can arm or disarm it; this guard flags the
lexical shape wherever it is, armed or not.

FLAGGED (on Sources/**/*.swift and Tests/**/*.swift, comments and strings blanked by
Scripts/ci/swift_lexing._mask), in one function or closure body:
  1. a KNOWN-VALUE LOCAL `x`: `var x = [ ... ]` or `var x: T = [ ... ]` whose whole initializer is an array literal
     (an Array or ContiguousArray literal, empty or not), or, for the pointer form, any `var x = <expr>`;
     a member path `withUnsafeMutablePointer(to: &a.b)` always counts (a stored property's initial value is a
     known value the guard cannot see; investigation variant w06);
  2. then `x.withUnsafeMutableBufferPointer` (trailing closure, or a function passed in parentheses) or
     `withUnsafeMutablePointer(to: &x)`;
  3. then, after that call's closing `)` or `}` and inside the scope that declares `x` (up to a new `var`/`let x`),
     a direct read: `x[...]` (not a plain `x[...] = ...` store), `x.first`, `x.last`, `x.withUnsafeBufferPointer`,
     `x.withUnsafeBytes`, a whole-array comparison `x == ...` / `... != x` (an inlined `==` reads every element),
     or, for the pointer form, any use of `x` (or `a.b`) that is not a plain assignment target, an `&x` argument or
     an argument label `x:`.

CAPTURED-VAR FORM (T16 phase 1 concern C5; hits are prefixed `captured:`): a local `var x = ...` declared before a
`withUnsafe[Mutable][Buffer]Pointer` / `withUnsafe[Mutable]Bytes` call whose closure literal assigns `x` (`x = `,
`x += `, `x[i] = `), followed by any use of `x` after the call in `x`'s scope. The fix is the W2 shape: declare the
accumulator inside the closure and return it. Status on Swift 6.4.0: not reproduced. The shapes K1-K5, K7 (captured
array-element write) and K8 (mutable-buffer closure, captured counter) are pinned in
Tests/ComprehensiveTests/LostStoreCanaryTests.swift (LostStoreCanaryCapturedVarTests), which kept the write in a
Release run on Swift 6.4.0 where the d_pure control lost its store in the same run (T15/final-fix/01.log, one host;
the same file asserts the correct value on every toolchain). The earlier out-of-repository fix-round-1 probe saw the
same for K1-K5. The form is flagged because the T15 class rule covers it, not because a miscompile of it was
observed. Allowlisted bodies carry an owner tag.

ACCEPTED (must-pass controls below): reading the values through the buffer inside the closure and returning them
(`let observed = x.withUnsafeMutableBufferPointer { all in ...; return Array(all) }`, investigation W2, which has
no call between the write and the read); `[T](repeating:count:)` and other non-literal array initializers
(investigation W1 and v02; this only holds because that initialization is not forwardable today, so such a site
should cite swiftlang/swift#92562 in a comment, as the control does); re-entering
`withUnsafeMutableBufferPointer` to read (v06).

ALLOWLIST: ALLOWED maps `path::function::name` to a reason, for reviewed sites that cannot be rewritten; each entry
must still match a hit (a stale entry fails).

LIMITS (lexical heuristic, not dataflow): stores between the declaration and the call (`x[0] = 5`) are not tracked
as known values; an initializer that is more than one array literal (`[a] + rest + [b]`, `[T](...)`, `[1, 2].map`)
is not a literal initializer; passing the whole array to a function after the closure is not a read (a non-inlined
callee reads memory, investigation `show(values)`; an inlined one is not modelled); arrays that are stored
parameters or globals are not tracked for the buffer form, and a stored property is tracked only when it has a
literal initializer (`var a: [Float] = [2]` in an enclosing type is flagged: conservative, because its initial value
is a known value); other pointer APIs (`withUnsafeMutableBytes`,
`withContiguousMutableStorageIfAvailable`, `withUnsafeMutablePointer(to:)` without a literal `&` argument) are not
matched; aliases (`let y = x`) and reads through a helper are not followed;
`#if` branches are all scanned; whether the closure is inlined is not modelled, so shapes that compile correctly today
(investigation v03, v09, v11) are still flagged. Captured-var form: only a write by name in the closure literal is seen
(not a write through a helper or a nested function), a closure passed as a function reference is skipped, and every
local `var` declared earlier in an enclosing scope is a candidate (its initializer is not inspected). Not seen, and
not reviewed or allowlisted: an `&acc` inout write inside the closure (8 sites in
QuantizedKernels.swift, `acc1`..`acc4` declared at line 596 and passed `&accN` at lines 611-638; the probe's K3
shape, which the lint cannot flag), a
mutating-member call on a captured var (`result.append(...)`, `data.removeAll()` and similar; 23 sites in T15 re-review A's widened scan, not re-counted here; for
example the `toArray` bodies of the Vector{384,512,768,1536}Optimized types, SIMDStorage.flattenToArray,
MixedPrecisionKernels.convertToFP32 and ErrorHandlingTests), iteration or member reads (`for v in x`, `x.max()`, `x.reduce`). Benchmarks/ is
not scanned. Interpolation: the scan keeps code inside string
interpolations as code (`_mask(..., keep_interpolations=True)`), so a read inside a single-line or a multi-line (triple-quoted) string interpolation counts
(fixtures: single-line and multi-line must-fail in both the buffer and captured forms, one must-pass in each); text outside the `\\(...)` is still not code. Filesystem reads only (works git-less).
"""
import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location('swift_lexing', ROOT / 'Scripts/ci/swift_lexing.py')
lexing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lexing)

SCANNED = ('Sources', 'Tests')
ISSUE = 'swiftlang/swift#92562'

# Reviewed sites that keep the shape: `path::function::name` (`captured:` prefix for the captured-var form) -> reason.
# Tags: [T14] / [T16] code owned by those tasks; [T15] T15-owned bodies; [T15-touched file] pre-existing bodies in a
# file T15 changed; [pre-existing] files T15 did not touch.
_CAPTURED = ('; captured var written in the closure and read after it. This form was not reproduced on Swift 6.4.0: in '
             'LostStoreCanaryTests.LostStoreCanaryCapturedVarTests (shapes K1-K5, K7, K8: a concrete and a '
             'protocol-requirement closure, nested closures, an inout write and a pointer write inside the closure, a '
             'captured array-element write, a mutable-buffer closure with a captured counter) kept the write in a Release '
             'run (T15/final-fix/01.log, one host) while the d_pure control lost its store in the same run. Not proof for '
             'every body; the trigger follows inlining')
ALLOWED = {
    'Tests/ComprehensiveTests/LostStoreCanaryTests.swift::lostStoreCanaryObservedValue::values':
        'the deliberate pure-Swift reproducer of the Release canary test (LostStoreCanaryTests)',
    'Tests/ComprehensiveTests/LostStoreCanaryTests.swift::lostStoreCanaryCapturedPointer::m':
        'the K4 pin of LostStoreCanaryCapturedVarTests (pointer form nested in a closure, kept on purpose)',
    'captured:Tests/ComprehensiveTests/LostStoreCanaryTests.swift::lostStoreCanaryCapturedLInf::hasNaN':
        'the K2 pin of LostStoreCanaryCapturedVarTests (the pre-W2 lInfinityNorm shape, kept on purpose)',
    'captured:Tests/ComprehensiveTests/LostStoreCanaryTests.swift::lostStoreCanaryCapturedLInf::maxValue':
        'the K2 pin of LostStoreCanaryCapturedVarTests (the pre-W2 lInfinityNorm shape, kept on purpose)',
    'captured:Tests/ComprehensiveTests/LostStoreCanaryTests.swift::lostStoreCanaryK1::m':
        'the K1 pin of LostStoreCanaryCapturedVarTests (kept on purpose)',
    'captured:Tests/ComprehensiveTests/LostStoreCanaryTests.swift::lostStoreCanaryK5::dot':
        'the K5 pin of LostStoreCanaryCapturedVarTests (kept on purpose)',
    'captured:Tests/ComprehensiveTests/LostStoreCanaryTests.swift::lostStoreCanaryK7::norms':
        'the K7 pin of LostStoreCanaryCapturedVarTests (KNNGraph.bruteForce shape, kept on purpose)',
    'captured:Tests/ComprehensiveTests/LostStoreCanaryTests.swift::lostStoreCanaryK8::calls':
        'the K8 pin of LostStoreCanaryCapturedVarTests (checkBorrow shape, kept on purpose)',
    'captured:Sources/VectorCore/ManifoldLearning/KNNGraph.swift::bruteForce::norms':
        '[pre-existing] file not touched by T15' + _CAPTURED,
    'captured:Sources/VectorCore/Operations/DistanceMetrics.swift::batchDistance::results':
        '[T14] EuclideanDistance.batchDistance witness (T14-owned)' + _CAPTURED,
    'captured:Sources/VectorCore/Operations/DistanceMetrics.swift::distance::aMagnitudeSq':
        '[T15] CosineDistance witness' + _CAPTURED,
    'captured:Sources/VectorCore/Operations/DistanceMetrics.swift::distance::bMagnitudeSq':
        '[T15] CosineDistance witness' + _CAPTURED,
    'captured:Sources/VectorCore/Operations/DistanceMetrics.swift::distance::count':
        '[T15] HammingDistance witness' + _CAPTURED,
    'captured:Sources/VectorCore/Operations/DistanceMetrics.swift::distance::dotProduct':
        '[T15] CosineDistance witness' + _CAPTURED,
    'captured:Sources/VectorCore/Operations/DistanceMetrics.swift::distance::result':
        '[T14/T15] ManhattanDistance and DotProductDistance (T14-owned) and ChebyshevDistance (T15) witnesses' + _CAPTURED,
    'captured:Sources/VectorCore/Operations/DistanceMetrics.swift::distance::sum':
        '[T15-touched file] MinkowskiDistance witness' + _CAPTURED,
    'captured:Sources/VectorCore/Operations/ExecutionOperations.swift::map::result':
        '[pre-existing] file not touched by T15' + _CAPTURED,
    'captured:Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift::calibrate::hasValues':
        '[T16] calibration min/max capture (re-tagged from T15, T15 addendum concern 3); finite-guarded: fatal on a '
        'non-finite value; T16 reduction registry .exempt entry' + _CAPTURED,
    'captured:Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift::calibrate::maximum':
        '[T16] calibration min/max capture (re-tagged from T15, T15 addendum concern 3); finite-guarded: fatal on a '
        'non-finite value; T16 reduction registry .exempt entry' + _CAPTURED,
    'captured:Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift::calibrate::minimum':
        '[T16] calibration min/max capture (re-tagged from T15, T15 addendum concern 3); finite-guarded: fatal on a '
        'non-finite value; T16 reduction registry .exempt entry' + _CAPTURED,
    'captured:Sources/VectorCore/Operations/VectorEntropy.swift::entropyTruncatedContract::allFinite':
        '[pre-existing] file not touched by T15' + _CAPTURED,
    'captured:Sources/VectorCore/Operations/VectorEntropy.swift::entropyTruncatedContract::compensation':
        '[pre-existing] file not touched by T15' + _CAPTURED,
    'captured:Sources/VectorCore/Operations/VectorEntropy.swift::entropyTruncatedContract::entropy':
        '[pre-existing] file not touched by T15' + _CAPTURED,
    'captured:Sources/VectorCore/Operations/VectorEntropy.swift::entropyTruncatedContract::sum':
        '[pre-existing] file not touched by T15' + _CAPTURED,
    'captured:Sources/VectorCore/Operations/VectorMath.swift::l1Norm::result':
        '[T15-touched file] pre-existing body' + _CAPTURED,
    'captured:Sources/VectorCore/Operations/VectorMath.swift::manhattanDistance::result':
        '[T15-touched file] pre-existing body' + _CAPTURED,
    'captured:Sources/VectorCore/Operations/VectorMath.swift::safeDivide::hasZero':
        '[T15-touched file] pre-existing body' + _CAPTURED,
    'captured:Sources/VectorCore/Protocols/SIMDStorage.swift::minMax::globalMax':
        '[T16-touched file] the SIMD4<Int8> minMax, an Int fold outside T16 (P02 ruling RD14); the SIMD4<Float> '
        'minMax, which shares this key, returns from inside its closure since T16' + _CAPTURED,
    'captured:Sources/VectorCore/Protocols/SIMDStorage.swift::minMax::globalMin':
        '[T16-touched file] the SIMD4<Int8> minMax, an Int fold outside T16 (P02 ruling RD14); the SIMD4<Float> '
        'minMax, which shares this key, returns from inside its closure since T16' + _CAPTURED,
    'captured:Sources/VectorCore/Protocols/VectorFactory.swift::createByCombining::result':
        '[pre-existing] file not touched by T15' + _CAPTURED,
    'captured:Sources/VectorCore/Protocols/VectorFactory.swift::createByTransforming::result':
        '[pre-existing] file not touched by T15' + _CAPTURED,
    'captured:Sources/VectorCore/Protocols/VectorProtocol.swift::dotProduct::result':
        '[T15-touched file] pre-existing body' + _CAPTURED,
    'captured:Sources/VectorCore/Protocols/VectorProtocol.swift::l1Norm::sum':
        '[T15-touched file] pre-existing body' + _CAPTURED,
    'captured:Sources/VectorCore/Protocols/VectorProtocol.swift::magnitude::hasNaN':
        '[T15-touched file] pre-existing body' + _CAPTURED,
    'captured:Sources/VectorCore/Protocols/VectorProtocol.swift::magnitude::maxAbs':
        '[T15-touched file] pre-existing body' + _CAPTURED,
    'captured:Sources/VectorCore/Protocols/VectorProtocol.swift::magnitude::sumSquares':
        '[T15-touched file] pre-existing body' + _CAPTURED,
    'captured:Sources/VectorCore/Protocols/VectorProtocol.swift::sum::total':
        '[T15-touched file] pre-existing body' + _CAPTURED,
    'captured:Sources/VectorCore/Storage/AlignedDynamicArrayStorage.swift::dotProduct::result':
        '[pre-existing] file not touched by T15' + _CAPTURED,
    'captured:Sources/VectorCore/Storage/AlignedDynamicArrayStorage.swift::dotProductWithDynamicStorage::result':
        '[pre-existing] file not touched by T15' + _CAPTURED,
    'captured:Sources/VectorCore/Storage/AlignedValueStorage.swift::dotProduct::result':
        '[pre-existing] file not touched by T15' + _CAPTURED,
    'captured:Sources/VectorCore/Storage/ArrayStorage.swift::dotProduct::result':
        '[pre-existing] file not touched by T15' + _CAPTURED,
    'captured:Sources/VectorCore/Storage/DimensionStorage.swift::dotProduct::result':
        '[pre-existing] file not touched by T15' + _CAPTURED,
    'captured:Sources/VectorCore/Storage/VectorStorage.swift::dotProduct::result':
        '[pre-existing] file not touched by T15' + _CAPTURED,
    'captured:Tests/ComprehensiveTests/NumericalStabilityTests.swift::computeExpectedMagnitude::maxAbs':
        '[pre-existing] file not touched by T15' + _CAPTURED,
    'captured:Tests/ComprehensiveTests/NumericalStabilityTests.swift::computeExpectedMagnitude::sumSquares':
        '[pre-existing] file not touched by T15' + _CAPTURED,
    'captured:Tests/ComprehensiveTests/NumericalStabilityTests.swift::isEffectivelyZero::allZero':
        '[pre-existing] file not touched by T15' + _CAPTURED,
    'captured:Tests/ComprehensiveTests/NumericalStabilityTests.swift::willMagnitudeOverflow::willOverflow':
        '[pre-existing] file not touched by T15' + _CAPTURED,
    'captured:Tests/ComprehensiveTests/QuantizedKernelsTests.swift::testQuantizedStorageLayout::baseAddress':
        '[pre-existing] file not touched by T15' + _CAPTURED,
    'captured:Tests/ComprehensiveTests/QuantizedKernelsTests.swift::testQuantizedStorageLayout::bufferCount':
        '[pre-existing] file not touched by T15' + _CAPTURED,
    'captured:Tests/ComprehensiveTests/SIMDProviderBoundsTests.swift::testDoubleReductionsBounds::sink':
        '[pre-existing] file not touched by T15' + _CAPTURED,
    'captured:Tests/ComprehensiveTests/SIMDProviderBoundsTests.swift::testFloatReductionsBounds::sink':
        '[pre-existing] file not touched by T15' + _CAPTURED,
    'captured:Tests/ComprehensiveTests/StorageAllocationContractTests.swift::checkBorrow::calls':
        '[pre-existing] file not touched by T15' + _CAPTURED,
}

_BUFFER_CALL = re.compile(r'(?<![\w.$])([A-Za-z_]\w*)\s*\.\s*withUnsafeMutableBufferPointer\b')
_POINTER_CALL = re.compile(r'\bwithUnsafeMutablePointer\s*\(\s*to\s*:\s*&\s*([A-Za-z_]\w*(?:\s*\.\s*[A-Za-z_]\w*)*)')
_OPEN = {'(': ')', '[': ']', '{': '}'}


def _match_close(mask, index):
    """Offset just past the bracket that closes mask[index] (or len(mask))."""
    opener, closer, depth = mask[index], _OPEN[mask[index]], 0
    for k in range(index, len(mask)):
        if mask[k] == opener:
            depth += 1
        elif mask[k] == closer:
            depth -= 1
            if depth == 0:
                return k + 1
    return len(mask)


def _skip_space(mask, index):
    while index < len(mask) and mask[index] in ' \t\r\n':
        index += 1
    return index


def _call_end(mask, index):
    """End of a call whose callee name ends at `index`: its `(...)` and/or trailing `{...}`."""
    k = _skip_space(mask, index)
    if k < len(mask) and mask[k] == '(':
        index = _match_close(mask, k)
        k = _skip_space(mask, index)
    if k < len(mask) and mask[k] == '{':
        index = _match_close(mask, k)
    return index


def _literal_initializer(mask, start):
    """True when the initializer at `start` is exactly an array literal `[...]` ending its statement."""
    k = _skip_space(mask, start)
    if k >= len(mask) or mask[k] != '[':
        return False
    end = _match_close(mask, k)
    tail = mask[end:end + 200]
    stripped = tail.lstrip(' \t')
    return stripped == '' or stripped[0] in '\n;}'


def _declaration(mask, scopes, name, call):
    """(offset, initializer start) of the nearest `var name ... =` before `call` whose scope encloses `call`."""
    pattern = re.compile(r'\bvar\s+' + re.escape(name) + r'\b\s*(?::[^=\n{]+)?=(?!=)')
    found = None
    for match in pattern.finditer(mask, 0, call):
        if scopes.at[match.start()] in scopes.ancestors[scopes.at[call]]:
            found = (match.start(), match.end())
    return found


def _owner_name(scopes, offset):
    scope = scopes.at[offset]
    while scope and not scopes.name[scope]:
        scope = scopes.parent[scope]
    return scopes.name[scope] if scope else '<top level>'


def _reads(mask, name, start, end, pointer_form):
    """Offsets of direct reads of `name` in mask[start:end]."""
    path = r'\s*\.\s*'.join(re.escape(part) for part in re.split(r'\s*\.\s*', name))
    redeclared = re.compile(r'\b(?:var|let)\s+' + path + r'(?![\w$])').search(mask, start, end)
    if redeclared:
        end = redeclared.start()                           # a new binding of the name ends the region
    hits = []
    for match in re.finditer(r'(?<![\w.$&])' + path + r'(?![\w$])', mask[:end]):
        if match.start() < start:
            continue
        k = _skip_space(mask, match.end())
        before = mask[:match.start()].rstrip()
        if re.match(r'[=!]=(?!=)', mask[k:]) or re.search(r'(?<![=!<>])[=!]=$', before):
            hits.append(match.start())                     # whole-array comparison reads every element
            continue
        if k < len(mask) and mask[k] == '[':
            close = _match_close(mask, k)
            after = _skip_space(mask, close)
            if mask.startswith('=', after) and not mask.startswith('==', after):
                continue                                   # x[i] = v is a store
            hits.append(match.start())
            continue
        member = re.match(r'\.\s*(first|last|withUnsafeBufferPointer|withUnsafeBytes)\b', mask[k:])
        if member:
            hits.append(match.start())
            continue
        if pointer_form:
            if mask.startswith('=', k) and not mask.startswith('==', k):
                continue                                   # x = v is a store
            if mask.startswith(':', match.end()) and not mask.startswith('::', match.end()):
                continue                                   # an argument label `x:`
            if re.match(r'\.\s*(withUnsafeMutableBufferPointer|withUnsafeMutableBytes)\b', mask[k:]):
                continue
            hits.append(match.start())
    return hits


def scan_source(path, source):
    """Hits in one file: `path::function::name@line` per flagged call."""
    mask = lexing._mask(source, keep_interpolations=True)
    scopes = lexing._Scopes(source)
    hits = []
    calls = [(m, False) for m in _BUFFER_CALL.finditer(mask)] + [(m, True) for m in _POINTER_CALL.finditer(mask)]
    for match, pointer_form in sorted(calls, key=lambda item: item[0].start()):
        name = re.sub(r'\s+', '', match.group(1))
        call = match.start()
        if pointer_form and '.' in name:
            scope_end = scopes.closes[scopes.at[call]]       # member path: its initial value is not visible here
        else:
            declared = _declaration(mask, scopes, name, call)
            if declared is None:
                continue
            if not pointer_form and not _literal_initializer(mask, declared[1]):
                continue
            scope_end = scopes.closes[scopes.at[declared[0]]]
        callee = 'withUnsafeMutablePointer' if pointer_form else 'withUnsafeMutableBufferPointer'
        end = _call_end(mask, mask.find(callee, call) + len(callee))
        if _reads(mask, name, end, scope_end, pointer_form):
            line = source.count('\n', 0, call) + 1
            hits.append(f'{path}::{_owner_name(scopes, call)}::{name}@{line}')
    return hits


_CAPTURED_CALL = re.compile(r'\b(withUnsafe(?:Mutable)?(?:Buffer)?Pointer|withUnsafe(?:Mutable)?Bytes)\b')
_ANY_VAR = re.compile(r'\bvar\s+([A-Za-z_]\w*)\b\s*(?::[^=\n{]+)?=(?!=)')
_ASSIGN = r'\s*(?:\[[^\]]*\]\s*)?(?:[-+*/&|^]|<<|>>)?=(?!=)'


def scan_captured_source(path, source):
    """Captured-var form: `captured:path::function::name@line` per (call, variable)."""
    mask = lexing._mask(source, keep_interpolations=True)
    scopes = lexing._Scopes(source)
    hits = []
    for match in _CAPTURED_CALL.finditer(mask):
        end = _call_end(mask, match.end())
        body = mask[match.end():end]
        if '{' not in body:
            continue                                       # a function reference, not a closure literal
        call = match.start()
        for declared in _ANY_VAR.finditer(mask, 0, call):
            name = declared.group(1)
            if scopes.at[declared.start()] not in scopes.ancestors[scopes.at[call]]:
                continue
            if re.search(r'\b(?:var|let)\s+' + re.escape(name) + r'\b', body):
                continue                                   # shadowed inside the closure
            if not re.search(r'(?<![\w.$])' + re.escape(name) + _ASSIGN, body):
                continue                                   # not written in the closure
            scope_end = scopes.closes[scopes.at[declared.start()]]
            rebound = re.compile(r'\b(?:var|let)\s+' + re.escape(name) + r'\b').search(mask, end, scope_end)
            stop = rebound.start() if rebound else scope_end
            for use in re.finditer(r'(?<![\w.$&])' + re.escape(name) + r'(?![\w$])', mask[:stop]):
                if use.start() < end:
                    continue
                k = _skip_space(mask, use.end())
                if mask.startswith('=', k) and not mask.startswith('==', k):
                    continue                               # a plain store
                if mask.startswith(':', use.end()) and not mask.startswith('::', use.end()):
                    continue                               # an argument label
                line = source.count('\n', 0, call) + 1
                hits.append(f'captured:{path}::{_owner_name(scopes, call)}::{name}@{line}')
                break
    return hits


def scan(files):
    """files: {relative path: source}. Returns sorted hits of both forms."""
    hits = []
    for path in sorted(files):
        hits.extend(scan_source(path, files[path]))
        hits.extend(scan_captured_source(path, files[path]))
    return sorted(hits)


def problems(hits, allowed):
    keys = {hit.rsplit('@', 1)[0] for hit in hits}
    result = [f'lost-store read shape (see {ISSUE}): {hit}' for hit in hits if hit.rsplit('@', 1)[0] not in allowed]
    result += [f'stale allowlist entry: {key}' for key in sorted(set(allowed) - keys)]
    result += [f'allowlist entry without a reason: {key}' for key, reason in sorted(allowed.items()) if not reason.strip()]
    return result


def real_files():
    files = {}
    for base in SCANNED:
        for path in sorted((ROOT / base).rglob('*.swift')):
            files[path.relative_to(ROOT).as_posix()] = path.read_text(encoding='utf-8')
    return files


# Must-fail shapes: the T15 lost-store investigation's reproducers (T15/lost-store-investigation, outside the repo),
# each reduced to its function body. Every one printed a lost store at -O on Swift 6.4.0 except the three marked
# "flagged by design", which compiled correctly there only because of inlining or a by-value pointer argument.
_W1 = '@inline(never) func writeOne(_ out: inout UnsafeMutableBufferPointer<Float>) { out[0] = 1 }\n'
_W2 = '@inline(never) func writeOneUMP(_ out: inout UnsafeMutablePointer<Float>) { out.pointee = 1 }\n'
MUST_FAIL = {
    'repro/lost_store (pointer form, Int)':
        '@inline(never)\nfunc write(_ p: inout UnsafeMutablePointer<Int>) {\n    p.pointee = 1\n}\n\nfunc test() {\n'
        '    var x = 0\n    withUnsafeMutablePointer(to: &x) { p in\n        var q = p\n        write(&q)\n    }\n'
        '    print(x)  // expected 1; swiftc -O prints 0\n}\ntest()\n',
    'ladder/d_pure = v01 (function reference)':
        _W1 + 'func main() {\n    var a: [Float] = [2]\n    a.withUnsafeMutableBufferPointer(writeOne)\n    report(a[0])\n}\n',
    'v05 (array escapes later)':
        _W1 + 'func main() {\n var a: [Float] = [2]\n a.withUnsafeMutableBufferPointer(writeOne)\n report(a[0])\n _ = keep(a)\n}\n',
    'v07 (re-read through withUnsafeBufferPointer)':
        _W1 + 'func main() {\n var a: [Float] = [2]\n a.withUnsafeMutableBufferPointer(writeOne)\n'
        ' a.withUnsafeBufferPointer { report($0[0]) }\n}\n',
    'v08 (rebased suffix)':
        _W1 + 'func main() {\n var a: [Float] = [2, 2, 2, 2]\n a.withUnsafeMutableBufferPointer { all in '
        'var out = UnsafeMutableBufferPointer(rebasing: all[1..<3]); writeOne(&out) }\n report(a[1])\n}\n',
    'v10 (ContiguousArray literal)':
        _W1 + 'func main() {\n var a: ContiguousArray<Float> = [2]\n a.withUnsafeMutableBufferPointer(writeOne)\n report(a[0])\n}\n',
    'w01 (local var, inout buffer)':
        _W1 + 'func main() {\n var x: Float = 2\n withUnsafeMutablePointer(to: &x) { p in '
        'var b = UnsafeMutableBufferPointer(start: p, count: 1); writeOne(&b) }\n report(x)\n}\n',
    'w03 (local var, inout pointer)':
        _W2 + 'func main() {\n var x: Float = 2\n withUnsafeMutablePointer(to: &x) { p in var q = p; writeOneUMP(&q) }\n report(x)\n}\n',
    'w04 (local var, inout holder)':
        'func main() {\n var x: Float = 2\n withUnsafeMutablePointer(to: &x) { p in var h = Holder(p: p); writeOneHolder(&h) }\n'
        ' report(x)\n}\n',
    'w06 (class field)':
        'final class Box { var x: Float = 2 }\nfunc main() {\n let bx = Box()\n'
        ' withUnsafeMutablePointer(to: &bx.x) { p in var q = p; writeOneUMP(&q) }\n report(bx.x)\n}\n',
    'w07 (array, inout pointer)':
        _W2 + 'func main() {\n var a: [Float] = [2]\n a.withUnsafeMutableBufferPointer { b in var q = b.baseAddress!; '
        'writeOneUMP(&q) }\n report(a[0])\n}\n',
    'w08 (array, inout holder)':
        'func main() {\n var a: [Float] = [2]\n a.withUnsafeMutableBufferPointer { b in var h = Holder(p: b.baseAddress!); '
        'writeOneHolder(&h) }\n report(a[0])\n}\n',
    'W0 (the MatrixMetricContractTests shape before fix round 1)':
        'func w0(_ zero: Vector512Optimized, _ soa: SoA512, _ route: Int) {\n'
        '    let sentinel = Float(bitPattern: 0x42F68001)\n    var values = [sentinel, sentinel, sentinel, sentinel]\n'
        '    values.withUnsafeMutableBufferPointer { all in\n'
        '        let out = UnsafeMutableBufferPointer(rebasing: all[1..<3])\n'
        '        BatchKernels_SoA.cosine_fused_blocked_2way(query: zero, queryNorm: 0, soa: soa, out: out)\n    }\n'
        '    #expect(values[1] == 1)\n}\n',
    'adaptiveEuclideanDistance before fix round 1':
        'func adaptive() -> Float {\n        var results: [Float] = [0]\n        results.withUnsafeMutableBufferPointer {\n'
        '            batchEuclidean512(query: queryFP16, candidates: candidatesSoA, results: $0)\n        }\n'
        '        let reduced = results[0]\n        return reduced\n}\n',
    'whole-array comparison after the closure':
        'func f() {\n    var values: [Float] = [123]\n    values.withUnsafeMutableBufferPointer { kernel($0) }\n'
        '    #expect(values == [123])\n}\n',
    'v03 (by-value pointer; flagged by design)':
        'func main() {\n var a: [Float] = [2]\n a.withUnsafeMutableBufferPointer { writeOnePtr($0.baseAddress!) }\n report(a[0])\n}\n',
    'v09 (closure inlined; flagged by design)':
        'func main() {\n var a: [Float] = [2]\n a.withUnsafeMutableBufferPointer { $0[0] = 1 }\n report(a[0])\n}\n',
    'read inside a string interpolation (keep_interpolations)':
        'func f() {\n    var a: [Float] = [2]\n    a.withUnsafeMutableBufferPointer { writeOne(&$0) }\n    print("a0=\\(a[0])")\n}\n',
    'read inside a multi-line string interpolation (keep_interpolations)':
        'func f() {\n    var a: [Float] = [2]\n    a.withUnsafeMutableBufferPointer { writeOne(&$0) }\n'
        '    print(\"\"\"\n    first:\n    \\(a[0])\n    \"\"\")\n}\n',
    'v11 (by-value pointer, .first; flagged by design)':
        'func main() {\n var a: [Float] = [2]\n a.withUnsafeMutableBufferPointer { b in writeOnePtr(b.baseAddress!) }\n'
        ' report(a.first!)\n}\n',
}

MUST_PASS = {
    'W2: read through the buffer inside the closure and return the values':
        'func w2(_ zero: Vector512Optimized, _ soa: SoA512, _ route: Int) {\n'
        '    let sentinel = Float(bitPattern: 0x42F68001)\n    var values = [sentinel, sentinel, sentinel, sentinel]\n'
        '    let observed: [Float] = values.withUnsafeMutableBufferPointer { all in\n'
        '        let out = UnsafeMutableBufferPointer(rebasing: all[1..<3])\n'
        '        BatchKernels_SoA.cosine_fused_blocked_2way(query: zero, queryNorm: 0, soa: soa, out: out)\n'
        '        return Array(all)\n    }\n    #expect(observed[1] == 1)\n}\n',
    'W1: repeating initializer, with its comment':
        'func w1() {\n    // [Float](repeating:count:) has no forwardable literal store today; reading after the closure relies on\n'
        '    // that (swiftlang/swift#92562, PR #92580).\n    var values = [Float](repeating: 2, count: 4)\n'
        '    values.withUnsafeMutableBufferPointer(writeOne)\n    report(values[0])\n}\n',
    'v02: repeating initializer, function reference':
        'func main() {\n var a = [Float](repeating: 2, count: 1)\n a.withUnsafeMutableBufferPointer(writeOne)\n report(a[0])\n}\n',
    'v06: read by re-entering withUnsafeMutableBufferPointer':
        'func main() {\n var a: [Float] = [2]\n a.withUnsafeMutableBufferPointer(writeOne)\n'
        ' a.withUnsafeMutableBufferPointer { report($0[0]) }\n}\n',
    'adaptiveEuclideanDistance after fix round 1 (value returned from the closure)':
        'func adaptive() -> Float {\n        var results: [Float] = [0]\n'
        '        let reduced: Float = results.withUnsafeMutableBufferPointer {\n'
        '            batchEuclidean512(query: queryFP16, candidates: candidatesSoA, results: $0)\n            return $0[0]\n'
        '        }\n        return reduced\n}\n',
    'store after the closure, then no read':
        'func f() {\n    var a: [Float] = [2]\n    a.withUnsafeMutableBufferPointer(writeOne)\n    a[0] = 3\n}\n',
    'array literal concatenation is not a literal initializer (LIMITS)':
        'func f() {\n    var a = [Float(1)] + input + [2]\n    a.withUnsafeMutableBufferPointer(writeOne)\n    report(a[0])\n}\n',
    'a new binding of the name ends the region (switch cases)':
        'func f(_ s: String) {\n    switch s {\n    case "a": var x: Float = 1; withUnsafeMutablePointer(to: &x) { g($0) }\n'
        '    case "b": var x: Double = 1; withUnsafeMutablePointer(to: &x) { h($0) }\n    default: break\n    }\n}\n',
    'interpolation text with no read of a mutated capture (keep_interpolations)':
        'func f(_ n: Int) {\n    var a: [Float] = [2]\n    a.withUnsafeMutableBufferPointer(writeOne)\n'
        '    a = []\n    print("a[0] n=\\(n)")\n    print(\"\"\"\n    a[0] \\(n)\n    \"\"\")\n}\n',
    'comments and strings are not code':
        'func f() {\n    var a: [Float] = [2]\n    a.withUnsafeMutableBufferPointer(writeOne)\n'
        '    // report(a[0])\n    print("a[0]")\n}\n',
}


# Captured-var form: the T15 fix-round-1 probe shapes (none lost its store on Swift 6.4.0; flagged as the class).
_P = ('protocol BufferBacked { func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) '
      'rethrows -> R }\n')
MUST_FAIL_CAPTURED = {
    'K1 captured, Array.withUnsafeBufferPointer': (
        'func k1(_ a: [Float]) -> Float {\n    var m: Float = 0\n'
        '    a.withUnsafeBufferPointer { b in for e in b { m = Swift.max(m, Swift.abs(e)) } }\n    return m\n}\n',
        ['captured:Sources/X.swift::k1::m@3']),
    'K2 old lInfinityNorm (two captured vars)': (
        _P + 'extension BufferBacked {\n    var lInf: Float {\n        var maxValue: Float = 0\n        var hasNaN = false\n'
        '        withUnsafeBufferPointer { buffer in\n            for element in buffer {\n'
        '                if element.isNaN { hasNaN = true; break }\n'
        '                maxValue = Swift.max(maxValue, Swift.abs(element))\n            }\n        }\n'
        '        return hasNaN ? .nan : maxValue\n    }\n}\n',
        ['captured:Sources/X.swift::lInf::hasNaN@6', 'captured:Sources/X.swift::lInf::maxValue@6']),
    'compound assignment in a closure (fixture only; not the probe K3, an inout write the lint cannot flag)': (
        'func k3(_ a: [Float]) -> Float {\n    var s: Float = 0\n    a.withUnsafeBufferPointer { b in s += b[0] }\n'
        '    return s * 2\n}\n', ['captured:Sources/X.swift::k3::s@3']),
    'K5 nested closures (the metric-witness shape), one hit per enclosing call': (
        'func k5(_ a: [Float], _ b: [Float]) -> Float {\n    var dot: Float = 0\n'
        '    a.withUnsafeBufferPointer { ab in\n        b.withUnsafeBufferPointer { bb in dot = kernel(ab, bb) }\n    }\n'
        '    return dot\n}\n', ['captured:Sources/X.swift::k5::dot@3', 'captured:Sources/X.swift::k5::dot@4']),
    'captured var read only inside a string interpolation (keep_interpolations; NumericalStabilityTests shape)': (
        'func d(_ a: [Float]) -> String {\n    var m: Float = 0\n    a.withUnsafeBufferPointer { b in m = b[0] }\n'
        '    return "max \\(m)"\n}\n', ['captured:Sources/X.swift::d::m@3']),
    'captured var read inside a multi-line string interpolation (keep_interpolations)': (
        'func d(_ a: [Float]) -> String {\n    var m: Float = 0\n    a.withUnsafeBufferPointer { b in m = b[0] }\n'
        '    return \"\"\"\n    max:\n      \\(m)\n    \"\"\"\n}\n', ['captured:Sources/X.swift::d::m@3']),
}
MUST_PASS_CAPTURED = {
    'K6: W2, accumulator declared inside the closure and returned':
        _P + 'extension BufferBacked {\n    var lInf: Float {\n        withUnsafeBufferPointer { buffer in\n'
        '            var maxValue: Float = 0\n            for element in buffer {\n'
        '                if element.isNaN { return Float.nan }\n'
        '                maxValue = Swift.max(maxValue, Swift.abs(element))\n            }\n'
        '            return maxValue\n        }\n    }\n}\n',
    'captured var only read in the closure':
        'func f(_ a: [Float]) -> Float {\n    var scale: Float = 2\n    scale += 1\n'
        '    let r = a.withUnsafeBufferPointer { b in b[0] * scale }\n    return r + scale\n}\n',
    'written in the closure, never read after':
        'func f(_ a: [Float]) {\n    var m: Float = 0\n    a.withUnsafeBufferPointer { b in m = b[0] }\n}\n',
    'interpolation text with no read of the mutated capture (keep_interpolations)':
        'func d(_ a: [Float]) -> String {\n    var m: Float = 0\n    a.withUnsafeBufferPointer { b in m = b[0] }\n'
        '    let k = 3\n    return "m \\(k)"\n}\n',
    'function reference, no closure literal':
        'func f(_ a: [Float]) -> Float {\n    var m: Float = 0\n    a.withUnsafeBufferPointer(g)\n    return m\n}\n',
}


class FixtureControls(unittest.TestCase):
    """The guard bites on every investigation shape and accepts the workarounds."""

    def test_captured_var_shapes_are_flagged(self):
        for label, (source, expected) in MUST_FAIL_CAPTURED.items():
            with self.subTest(label):
                self.assertEqual(scan({'Sources/X.swift': source}), expected, label)

    def test_captured_var_controls_pass(self):
        for label, source in MUST_PASS_CAPTURED.items():
            with self.subTest(label):
                self.assertEqual(scan({'Tests/Y.swift': source}), [], label)

    def test_every_investigation_shape_is_flagged(self):
        for label, source in MUST_FAIL.items():
            with self.subTest(label):
                self.assertEqual(len(scan({'Sources/X.swift': source})), 1, label)

    def test_workarounds_and_controls_pass(self):
        for label, source in MUST_PASS.items():
            with self.subTest(label):
                self.assertEqual(scan({'Tests/Y.swift': source}), [], label)

    def test_hit_names_file_function_and_variable(self):
        hits = scan({'Tests/Z.swift': MUST_FAIL['ladder/d_pure = v01 (function reference)']})
        self.assertEqual(hits, ['Tests/Z.swift::main::a@4'])   # the line of the withUnsafeMutableBufferPointer call

    def test_allowlist_entries_must_match_and_carry_reasons(self):
        hits = ['Tests/Z.swift::main::a@3']
        self.assertEqual(problems(hits, {'Tests/Z.swift::main::a': 'reviewed'}), [])
        self.assertEqual(problems(hits, {}), [f'lost-store read shape (see {ISSUE}): Tests/Z.swift::main::a@3'])
        self.assertEqual(problems([], {'Tests/Z.swift::main::a': 'reviewed'}),
                         ['stale allowlist entry: Tests/Z.swift::main::a'])
        self.assertEqual(problems(hits, {'Tests/Z.swift::main::a': ' '}),
                         ['allowlist entry without a reason: Tests/Z.swift::main::a'])


class RealTree(unittest.TestCase):
    maxDiff = None

    def test_no_unreviewed_lost_store_read(self):
        self.assertEqual(problems(scan(real_files()), ALLOWED), [])

    def test_scan_is_not_vacuous(self):
        """Sanity: the deliberate canary reproducer is found in the real tree."""
        hits = scan(real_files())
        self.assertIn('Tests/ComprehensiveTests/LostStoreCanaryTests.swift::lostStoreCanaryObservedValue::values',
                      {hit.rsplit('@', 1)[0] for hit in hits})


if __name__ == '__main__':
    unittest.main()
