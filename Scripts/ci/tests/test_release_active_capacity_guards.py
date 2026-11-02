"""Policy (PR #56 review AP2, remediation of 2026-10-23): every output-capacity check in Sources/VectorCore is active in
Release builds.

WHY: fifteen public FP16 SoA batch kernels, the six public `batchDotFP16_*`/`batchDotMixed*` kernels (through two shared
internal kernels) and nine internal SoA kernels checked `results.count >= vectorCount` (or `out.count >= N`) only under
`#if DEBUG`, then wrote through an unchecked buffer subscript at -O, so a Release caller passing a short output buffer
was written past its end with no diagnostic (Tests/ComprehensiveTests/ReleaseSizeGuardTests.swift reproduces this on
the Release row with a sentinel mapping). The sibling INT8 kernel and the FP16 range wrappers already used Release-
active named guards. Every site now uses `guard ... else { fatalError(...) }` before any write; this guard fails on any
new Debug-only capacity assert in the FORMS below, so the class cannot reopen one kernel at a time.

FORMS, in every Sources/VectorCore/**/*.swift file, on the shared lexer's text (Scripts/ci/swift_lexing._mask:
comments and string-literal text blanked, code length preserved): inside the DEBUG branch of an `#if DEBUG` region
(any nesting; `#else`/`#elseif` leaves it), a call to `assert(` or `precondition(` whose condition (the argument text
up to the first top-level comma, or the closing parenthesis) contains `<identifier>.count >= ...`, `.count > ...`,
`.vectorCount >= ...` or `.vectorCount > ...`. A multi-line call is read until its parentheses balance. The ALLOWLIST is
EMPTY; an entry would map `path:line` to a reviewed reason, and a stale entry would fail.

LIMITS (lexical, not a type checker; each is a LIMIT fixture): a check spelled with the capacity on the right
(`N <= out.count`), through a helper (`assert(hasCapacity(out, N))`), as an equality shape check
(`assert(x.storage.count == 128)`), or as a Debug-only `guard`/`fatalError` wrapped in `#if DEBUG`, is not seen;
`#if DEBUG || X` and `#elseif DEBUG` spellings are not recognised as DEBUG regions.
"""
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swift_lexing import _mask  # noqa: E402

ALLOWLIST: dict[str, str] = {}

CALL = re.compile(r'\b(?:assert|precondition)\s*\(')
CAPACITY = re.compile(r'\w+\.(?:count|vectorCount)\s*>=?\s*\S')


def _condition(text, start):
    """The first argument of the call whose opening parenthesis is at `start`, or None when unbalanced."""
    depth, i, comma = 0, start, None
    while i < len(text):
        c = text[i]
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                return text[start + 1:comma if comma is not None else i]
        elif c == ',' and depth == 1 and comma is None:
            comma = i
        i += 1
    return None


def debug_only_capacity_asserts(source):
    """[(line, condition)] for each assert/precondition on a `.count >=`/`>` condition inside an `#if DEBUG` branch."""
    masked = _mask(source)
    findings, stack = [], []          # stack entry: this `#if` level is a DEBUG level and we are in its true branch
    offsets = [0]
    for line in masked.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    lines = masked.splitlines()
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('#if'):
            stack.append(stripped[3:].strip() == 'DEBUG')
            continue
        if stripped.startswith('#else') or stripped.startswith('#elseif'):
            if stack:
                stack[-1] = False
            continue
        if stripped.startswith('#endif'):
            if stack:
                stack.pop()
            continue
        if not any(stack):
            continue
        for match in CALL.finditer(line):
            condition = _condition(masked, offsets[number - 1] + match.end() - 1)
            if condition is not None and CAPACITY.search(condition):
                findings.append((number, ' '.join(condition.split())))
    return findings


MUST_FAIL = {
    'former SoA spelling': '#if DEBUG\nassert(out.count >= N, "Output buffer too small")\n#endif\n',
    'former mixed-precision spelling': '#if DEBUG\nassert(results.count >= vectorCount, "Results buffer too small")\nassert(candidates.dimension == 512, "x")\n#endif\n',
    'precondition under DEBUG': '#if DEBUG\nprecondition(out.count > N)\n#endif\n',
    'multi-line call': '#if DEBUG\nassert(\n    results.count >= candidates.vectorCount,\n    "short")\n#endif\n',
    'nested #if inside DEBUG': '#if DEBUG\n#if os(macOS)\nassert(out.count >= N)\n#endif\n#endif\n',
    'vectorCount form': '#if DEBUG\nassert(out.vectorCount >= N)\n#endif\n',
}
MUST_PASS = {
    'release guard': 'guard out.count >= N else { fatalError("short") }\n',
    'precondition outside DEBUG': 'precondition(out.count >= N, "short")\n',
    'assert after #endif': '#if DEBUG\nlet x = 1\n#endif\nassert(out.count >= N)\n',
    'assert in the #else branch': '#if DEBUG\nlet x = 1\n#else\nassert(out.count >= N)\n#endif\n',
    'non-capacity assert under DEBUG': '#if DEBUG\nassert(candidates.dimension == 512, "x")\n#endif\n',
    'comment and string text': '#if DEBUG\n// assert(out.count >= N)\nlet s = "assert(out.count >= N)"\n#endif\n',
}
LIMITS = {
    'capacity on the right': '#if DEBUG\nassert(N <= out.count)\n#endif\n',
    'helper': '#if DEBUG\nassert(hasCapacity(out, N))\n#endif\n',
    'equality shape check': '#if DEBUG\nassert(x.storage.count == 128)\n#endif\n',
    'Debug-only guard': '#if DEBUG\nguard out.count >= N else { fatalError("short") }\n#endif\n',
    'compound condition': '#if DEBUG || TESTING\nassert(out.count >= N)\n#endif\n',
}


class ReleaseActiveCapacityGuardTests(unittest.TestCase):
    def test_fixtures_that_must_fail(self):
        for name, source in MUST_FAIL.items():
            with self.subTest(name=name):
                self.assertEqual(len(debug_only_capacity_asserts(source)), 1, name)

    def test_fixtures_that_must_pass(self):
        for name, source in MUST_PASS.items():
            with self.subTest(name=name):
                self.assertEqual(debug_only_capacity_asserts(source), [], name)

    def test_limits_are_still_unseen(self):
        for name, source in LIMITS.items():
            with self.subTest(name=name):
                self.assertEqual(debug_only_capacity_asserts(source), [], name)

    def test_sources_have_no_debug_only_capacity_asserts(self):
        found = {}
        for path in sorted((ROOT / 'Sources' / 'VectorCore').rglob('*.swift')):
            for line, condition in debug_only_capacity_asserts(path.read_text()):
                found[f'{path.relative_to(ROOT)}:{line}'] = condition
        self.assertEqual(set(ALLOWLIST) - set(found), set(), 'stale allowlist entry')
        self.assertEqual({k: v for k, v in found.items() if k not in ALLOWLIST}, {},
                         'Debug-only capacity assert; use a Release-active guard before any write')


if __name__ == '__main__':
    unittest.main()
