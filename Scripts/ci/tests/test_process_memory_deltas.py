"""Reject unsigned subtraction of whole-process memory counters in Swift sources.

HEURISTIC, line-local scan (not a Swift parser). Scope: `*.swift` under Tests/,
Sources/ and Benchmarks/. It flags a binary `-` or `-=` on one line when an
operand
  * names resident_size, resident_size_max, virtual_size or phys_footprint, or
  * is an identifier that a `let` in the same file bound directly from one of
    those names without a signed conversion,
unless every such operand is wrapped in a signed/floating conversion such as
Int64(clamping: ...), Int64(...), Int(...) or Double(...). Taint is tracked per
file by name, not by scope. Subtractions split across lines, values reached
through function returns or properties, and counters read via other names are
NOT detected. Also not detected: parenthesised operands such as
`(a.resident_size) - (b.resident_size)`.

Defect class: mach_task_basic_info counters are UInt64; whole-process RSS can
fall while parallel tests free memory, so `after - before` traps on overflow
(T04: QuantizedKernelsTests.swift, message-less SIGTRAP of the whole process).
"""
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCAN_DIRS = ("Tests", "Sources", "Benchmarks")

COUNTERS = r"(?:resident_size_max|resident_size|virtual_size|phys_footprint)"
COUNTER_RE = re.compile(COUNTERS)
SIGNED = r"(?:Int64|Int32|Int|Double|Float)"
WRAPPED = rf"{SIGNED}\((?:clamping:\s*)?[^()]*\)"
OPERAND = rf"(?:{WRAPPED}|[\w.]+)"
SUBTRACTION_RE = re.compile(rf"({OPERAND})\s*-=?\s*({OPERAND})")
WRAPPED_RE = re.compile(WRAPPED)
LET_RE = re.compile(r"\b(?:let|var)\s+(\w+)\s*(?::[^=]+)?=\s*(.*)$")


def _strip_comment(line):
    index = line.find("//")
    return line if index < 0 else line[:index]


def _unsigned_names(text):
    """Counter-derived names: a counter occurrence outside a signed wrapper."""
    return COUNTER_RE.search(WRAPPED_RE.sub("", text)) is not None


def unsigned_counter_subtractions(source):
    """Return sorted (line, text) for flagged subtractions (see module docs)."""
    lines = [_strip_comment(line) for line in source.splitlines()]
    tainted = set()
    for line in lines:
        match = LET_RE.search(line)
        if match and _unsigned_names(match.group(2)):
            tainted.add(match.group(1))

    def operand_unsigned(operand):
        if WRAPPED_RE.fullmatch(operand):
            return False
        return COUNTER_RE.search(operand) is not None or operand in tainted

    found = []
    for number, line in enumerate(lines, 1):
        for match in SUBTRACTION_RE.finditer(line):
            end = match.end()
            if line[match.end(1):match.start(2)].count("-") > 1 or line[match.end(1):end].lstrip().startswith("->"):
                continue
            if operand_unsigned(match.group(1)) or operand_unsigned(match.group(2)):
                found.append((number, line.strip()))
                break
    return found


def scan_tree(root, counts=None):
    """Scan SCAN_DIRS under root; if `counts` is a dict, record files scanned per directory."""
    violations = []
    for name in SCAN_DIRS:
        files = sorted((root / name).rglob("*.swift"))
        if counts is not None:
            counts[name] = len(files)
        for path in files:
            for line, text in unsigned_counter_subtractions(path.read_text(errors="replace")):
                violations.append(f"{path.relative_to(root)}:{line}: {text}")
    return violations


class ProcessMemoryDeltaTests(unittest.TestCase):
    def test_no_unsigned_process_memory_subtraction_in_swift_sources(self):
        counts = {}
        violations = scan_tree(ROOT, counts)
        for name in SCAN_DIRS:
            self.assertGreater(counts[name], 0, f"{name}/ has no scanned *.swift files (vacuous scan)")
        self.assertEqual(violations, [], "\n".join(violations))

    def test_negative_control_direct_subtraction(self):
        source = "let fp32Memory = info.resident_size - baselineMemory\n"
        self.assertEqual([n for n, _ in unsigned_counter_subtractions(source)], [1])

    def test_negative_control_quantized_kernels_shape(self):
        source = (
            "let baselineMemory = info.resident_size\n"
            "let largeFP32 = (0..<10000).map { _ in x }\n"
            "let fp32Memory = info.resident_size - baselineMemory\n"
        )
        self.assertEqual([n for n, _ in unsigned_counter_subtractions(source)], [3])

    def test_negative_control_bound_lets_subtracted_later(self):
        source = (
            "let before = info.phys_footprint\n"
            "work()\n"
            "let after = info.phys_footprint\n"
            "let growth = after - before\n"
        )
        self.assertEqual([n for n, _ in unsigned_counter_subtractions(source)], [4])

    def test_negative_control_mixed_precision_shape(self):
        source = (
            "let baselineMemory = result == KERN_SUCCESS ? info.resident_size : 0\n"
            "let afterMemory = result == KERN_SUCCESS ? info.resident_size : baselineMemory\n"
            "return Int(afterMemory - baselineMemory)\n"
        )
        self.assertEqual([n for n, _ in unsigned_counter_subtractions(source)], [3])

    def test_negative_control_every_counter_name(self):
        for counter in ("resident_size", "resident_size_max", "virtual_size", "phys_footprint"):
            with self.subTest(counter=counter):
                self.assertEqual(len(unsigned_counter_subtractions(f"let d = a.{counter} - b.{counter}\n")), 1)

    def test_negative_control_subtract_assign(self):
        source = "var total = base\ntotal -= info.resident_size\n"
        self.assertEqual([n for n, _ in unsigned_counter_subtractions(source)], [2])
        source = "let b = info.resident_size\nvar x = y\nx -= b\n"
        self.assertEqual([n for n, _ in unsigned_counter_subtractions(source)], [3])

    def test_negative_control_half_converted(self):
        source = "let d = Int64(clamping: a.resident_size) - b.resident_size\n"
        self.assertEqual(len(unsigned_counter_subtractions(source)), 1)

    def test_positive_control_signed_clamping_form(self):
        source = (
            "let d = Int64(clamping: after.resident_size) - Int64(clamping: before.resident_size)\n"
            "let b = Int64(clamping: info.resident_size)\n"
            "let a = Int64(clamping: info.resident_size)\n"
            "let growth = a - b\n"
            "let g2 = Int(info.resident_size) - Int(other.resident_size)\n"
        )
        self.assertEqual(unsigned_counter_subtractions(source), [])

    def test_positive_control_unrelated_subtraction_comments_and_arrows(self):
        source = (
            "let n = count - 1\n"
            "func f() -> Int { 0 }\n"
            "// let bad = info.resident_size - baseline\n"
            "let total = Int(info.resident_size)\n"
        )
        self.assertEqual(unsigned_counter_subtractions(source), [])


if __name__ == "__main__":
    unittest.main()
