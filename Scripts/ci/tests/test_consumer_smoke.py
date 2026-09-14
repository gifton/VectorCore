"""Check documentation example selection without invoking the Swift compiler."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "consumer_smoke.py"
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
            capture_output=True, text=True)

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


if __name__ == "__main__":
    unittest.main()
