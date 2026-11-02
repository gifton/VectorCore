"""Check documentation example selection and the compile-contract table without invoking the Swift compiler."""
from pathlib import Path
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "consumer_smoke.py"
ROOT = Path(__file__).resolve().parents[3]


def load_script():
    spec = importlib.util.spec_from_file_location("consumer_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
REFERENCES = (
    "Docs/API_Overview_Map.md",
    "Docs/Memory_Alignment.md",
    "Docs/SoA_Layout_Contract.md",
    "Docs/Linear_Algebra_and_Projection.md",
)


class ConsumerExampleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vectorcore-doc-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.script = self.root / "Scripts/ci/consumer_smoke.py"
        self.script.parent.mkdir(parents=True)
        shutil.copyfile(SCRIPT, self.script)
        self.write("README.md", """# Library
## Installation
```swift
not a standalone program
```
## Quick Start
```swift
print("quick")
```
## After
```swift
also not a standalone program
```
""")
        for path in REFERENCES:
            self.write(path, '# Reference\n\n```swift\nprint("reference")\n```\n')

    def write(self, path, contents):
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents)

    def list_examples(self):
        return subprocess.run(
            [sys.executable, str(self.script), "--list-examples"],
            capture_output=True, text=True, timeout=60)

    def test_includes_each_reference_block_but_not_installation_or_deferred_guides(self):
        # A README-only collector or a recursive scan would select the wrong programs.
        self.write(REFERENCES[0], """# API
```swift
print("first")
```
```swift
print("second")
```
""")
        self.write("Guides/Old.md", "```swift\ninvalid guide API\n```\n")
        self.write("Docs/Performance_Guide.md", "```swift\ninvalid tutorial API\n```\n")
        result = self.list_examples()
        self.assertEqual(result.returncode, 0, result.stderr)
        examples = json.loads(result.stdout)
        self.assertEqual([example["source"] for example in examples], [
            'print("quick")\n',
            'print("first")\n',
            'print("second")\n',
            'print("reference")\n',
            'print("reference")\n',
            'print("reference")\n',
        ])
        self.assertEqual(examples[2]["document"], REFERENCES[0])
        self.assertEqual(examples[2]["block"], 2)

    def test_missing_quick_start_fails_with_document_context(self):
        self.write("README.md", "# Library\n")
        result = self.list_examples()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("README.md", result.stderr)
        self.assertIn("Quick Start", result.stderr)

    def test_multiple_quick_start_programs_are_rejected(self):
        self.write("README.md", """## Quick Start
```swift
print("one")
```
```swift
print("two")
```
""")
        result = self.list_examples()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly one", result.stderr)

    def test_reference_without_swift_blocks_is_not_silently_skipped(self):
        self.write(REFERENCES[1], "# Memory\n")
        result = self.list_examples()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(REFERENCES[1], result.stderr)

    def test_unclosed_swift_block_is_not_silently_skipped(self):
        self.write(REFERENCES[0], """# API
```swift
print("closed")
```
```swift
print("unclosed")
""")
        result = self.list_examples()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(REFERENCES[0], result.stderr)
        self.assertIn("Unclosed", result.stderr)


class CompileContractTests(unittest.TestCase):
    """The must-not-compile contracts (PR #56 review SELF1): nine negative fixtures, each with a positive control where
    the fixture has one, compiled by the consumer job. The table is checked here without a compiler; the compiler run
    is the job itself."""

    def test_table_names_the_nine_negatives_and_their_controls(self):
        module = load_script()
        rows = module.CONTRACTS
        negatives = {row["path"] for row in rows if row["expect"] == "fail"}
        self.assertEqual(sorted(Path(p).name for p in negatives), [
            "B01BareConformanceNegative.swift", "DeviceNegative.swift", "HooksNegative.swift", "LoggerNegative.swift",
            "R01MetricScalarContract.swift", "R02CollectionUpdateConsumer.swift", "R05OldPropertyNegative.swift",
            "RoutedNegative.swift", "StorageNegative.swift"])
        self.assertEqual({row["expect"] for row in rows}, {"fail", "compile", "run"})
        for row in rows:
            with self.subTest(row=row["id"]):
                self.assertTrue((ROOT / row["path"]).is_file(), row["path"])
                if row["expect"] == "run":
                    self.assertTrue(row["marker"])
                if row["expect"] == "fail":
                    self.assertTrue(row.get("errorLines") or row.get("diagnostics"), "a negative names what must fail")
        self.assertEqual(len({row["id"] for row in rows}), len(rows), "ids are unique")

    def test_list_contracts_fails_on_a_missing_fixture_and_lists_them_otherwise(self):
        module = load_script()
        result = subprocess.run([sys.executable, str(self.script), "--list-contracts"], capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(module.CONTRACTS[0]["path"], result.stderr)
        for row in module.CONTRACTS:
            self.write(row["path"], "import VectorCore\n")
        result = subprocess.run([sys.executable, str(self.script), "--list-contracts"], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        listed = json.loads(result.stdout)
        self.assertEqual([row["id"] for row in listed], [row["id"] for row in module.CONTRACTS])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vectorcore-contract-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.script = self.root / "Scripts/ci/consumer_smoke.py"
        self.script.parent.mkdir(parents=True)
        shutil.copyfile(SCRIPT, self.script)

    def write(self, path, contents):
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents)


class ContractJudgementTests(unittest.TestCase):
    """`judge_contract(row, status, output, source)`: (passed, detail) for one compiler result."""

    def setUp(self):
        self.module = load_script()

    def judge(self, row, status, output, source="import VectorCore\n\nfunc a() {}\nfunc b() {}\n"):
        return self.module.judge_contract(dict({"id": "x", "path": "Fixture.swift"}, **row), status, output, source)

    def test_negative_with_every_code_line_erroring(self):
        row = {"expect": "fail", "errorLines": "code"}
        good = "Fixture.swift:3:6: error: a\nFixture.swift:4:6: error: b\n"
        self.assertTrue(self.judge(row, 1, good)[0])
        self.assertFalse(self.judge(row, 0, good)[0], "a successful build is not a failure")
        self.assertFalse(self.judge(row, 1, "Fixture.swift:3:6: error: a\n")[0], "line 4 did not error")
        self.assertFalse(self.judge(row, 1, good + "Fixture.swift:3:9: error: c\n")[0], "two errors on one line")
        self.assertFalse(self.judge(row, 1, "Other.swift:3:6: error: a\nOther.swift:4:6: error: b\n")[0], "errors elsewhere")

    def test_negative_with_pinned_lines_and_diagnostics(self):
        row = {"expect": "fail", "errorLines": [5, 8], "diagnostics": ["requires that"], "diagnosticsAny": ["Encodable", "Decodable"]}
        output = "Fixture.swift:5:1: error: x requires that T: Encodable\nFixture.swift:8:1: error: y requires that U: Decodable\n"
        self.assertTrue(self.judge(row, 1, output)[0])
        self.assertFalse(self.judge(row, 1, output.replace("requires that", "needs"))[0], "required diagnostic absent")
        self.assertFalse(self.judge(row, 1, output.replace("Encodable", "Hashable").replace("Decodable", "Hashable"))[0], "none of the alternatives")
        self.assertFalse(self.judge(row, 1, output + "Fixture.swift:9:1: error: z requires that\n")[0], "an extra line errored")
        # A line listed twice expects exactly two errors there (BinaryCodable reports one per protocol).
        twice = {"expect": "fail", "errorLines": [5, 5]}
        both = "Fixture.swift:5:1: error: needs BinaryEncodable\nFixture.swift:5:1: error: needs BinaryDecodable\n"
        self.assertTrue(self.judge(twice, 1, both)[0])
        self.assertFalse(self.judge(twice, 1, "Fixture.swift:5:1: error: needs BinaryEncodable\n")[0], "only one error")

    def test_optional_error_lines_absorb_compiler_dependent_diagnostics(self):
        # D02 LoggerNegative line 7 (a bare `Logger` with both modules imported) is ambiguous under Swift 6.4 and accepted
        # by Swift 6.0: the row requires line 8 and tolerates line 7, and still rejects any other line or a success.
        row = {"expect": "fail", "errorLines": [8], "optionalErrorLines": [7]}
        self.assertTrue(self.judge(row, 1, "Fixture.swift:8:1: error: a\n")[0])
        self.assertTrue(self.judge(row, 1, "Fixture.swift:7:1: error: b\nFixture.swift:8:1: error: a\n")[0])
        self.assertFalse(self.judge(row, 1, "Fixture.swift:7:1: error: b\n")[0], "the required line did not error")
        self.assertFalse(self.judge(row, 1, "Fixture.swift:8:1: error: a\nFixture.swift:9:1: error: c\n")[0], "an unlisted line errored")
        self.assertFalse(self.judge(row, 1, "Fixture.swift:8:1: error: a\nFixture.swift:8:9: error: d\n")[0], "two errors on a line listed once")
        self.assertFalse(self.judge(row, 0, "")[0])

    def test_colour_coded_diagnostics_are_read(self):
        # swift build emits `path:line:col: ESC[1;31merror: ESC[1;39mmessage`; the judge strips the escapes first.
        row = {"expect": "fail", "errorLines": [3], "diagnostics": ["cannot find"]}
        coloured = "/tmp/x/Sources/C/Fixture.swift:3:6: \x1b[1;31merror: \x1b[1;39mcannot find 'a'\x1b[0;0m\n"
        self.assertTrue(self.judge(row, 1, coloured)[0])

    def test_compile_and_run_expectations(self):
        self.assertTrue(self.judge({"expect": "compile"}, 0, "")[0])
        self.assertFalse(self.judge({"expect": "compile"}, 1, "Fixture.swift:3:1: error: a")[0])
        self.assertTrue(self.judge({"expect": "run", "marker": "OK_MARK"}, 0, "...\nOK_MARK\n")[0])
        self.assertFalse(self.judge({"expect": "run", "marker": "OK_MARK"}, 0, "nothing")[0], "marker absent")
        self.assertFalse(self.judge({"expect": "run", "marker": "OK_MARK"}, 2, "OK_MARK")[0], "nonzero exit")


if __name__ == "__main__":
    unittest.main()
