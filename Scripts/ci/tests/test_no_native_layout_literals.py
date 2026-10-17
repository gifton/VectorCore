"""Policy: native SwiftPM object-layout literals live only in swiftpm_products.py.

Drivers must discover products, objects, module directories and module maps through
Scripts/ci/swiftpm_products.py, which accepts exactly the native and swiftbuild layouts.
A literal such as `.build/release`, `VectorCoreC.build/module.modulemap` or a
`VectorCore.build/*.o` glob silently assumes the native layout and breaks under
swiftbuild (the class closed in T03 fix round 1; first instances: L03/R05 drivers).

Scope: Scripts/ and .github/ text files (py, swift, rb, sh, yml, json). This is a lexical
heuristic. Allowlist entries need a justification; the allowlist is checked for staleness.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCANNED_SUFFIXES = {'.py', '.swift', '.rb', '.sh', '.yml', '.yaml', '.json'}
SCANNED_ROOTS = ('Scripts', '.github')

PATTERNS = [
    ('.build/<config> literal', re.compile(r'\.build/(?:debug|release)\b')),
    ('<arch>-apple-macosx/<config> literal', re.compile(r'apple-macosx/(?:debug|release)\b')),
    ('<Target>.build/module.modulemap literal', re.compile(r'\.build/module\.modulemap')),
    ('<Target>.build object glob', re.compile(r'\w\.build/(?:\*\*/)?\*\.o\b')),
    ('<Target>.build directory join', re.compile(r"""(?:/\s*|\(\s*)['"]\w+\.build['"]\s*\)?\s*\)?\s*\.(?:r?glob|iterdir)""")),
]

# Paths (relative to the repository root, '/' separated) allowed to contain such literals, with the reason.
ALLOWLIST = {
    'Scripts/ci/swiftpm_products.py': 'the single place that knows both layouts',
    'Scripts/ci/tests/test_no_native_layout_literals.py': 'defines the patterns and their negative controls',
    'Scripts/ci/claim_ledger.json': 'claim-ledger data that quotes Docs/Performance_Guide.md command lines verbatim (live '
                                    'and tombstoned statements); never resolved as a path',
}


def hits(text):
    found = []
    for number, line in enumerate(text.splitlines(), 1):
        for label, pattern in PATTERNS:
            if pattern.search(line):
                found.append((number, label))
    return found


def scan(root, allowlist=ALLOWLIST):
    result = {}
    for top in SCANNED_ROOTS:
        base = Path(root) / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob('*')):
            relative = path.relative_to(root).as_posix()
            if not path.is_file() or path.suffix not in SCANNED_SUFFIXES or relative in allowlist:
                continue
            found = hits(path.read_text(encoding='utf-8', errors='replace'))
            if found:
                result[relative] = found
    return result


class NativeLayoutLiteralTests(unittest.TestCase):
    def test_repository_has_no_native_layout_literals_outside_the_allowlist(self):
        self.assertEqual(scan(ROOT), {})

    def test_allowlist_entries_exist_and_are_justified(self):
        for relative, reason in ALLOWLIST.items():
            with self.subTest(path=relative):
                self.assertTrue((ROOT / relative).is_file())
                self.assertTrue(reason.strip())

    def test_allowlist_entries_are_not_stale(self):
        # An entry that no longer contains any literal must be removed.
        for relative in ALLOWLIST:
            if relative.endswith('test_no_native_layout_literals.py'):
                continue  # its literals sit in regexes and controls built at runtime
            with self.subTest(path=relative):
                self.assertTrue(hits((ROOT / relative).read_text(encoding='utf-8')), 'stale allowlist entry')

    def test_negative_controls_flag_every_pattern(self):
        controls = {
            '.build/<config> literal': "build = Path(root) / '.build/release'",
            '<arch>-apple-macosx/<config> literal': "bin = scratch / 'arm64-apple-macosx/debug'",
            '<Target>.build/module.modulemap literal': "m = build / 'VectorCoreC.build/module.modulemap'",
            '<Target>.build object glob': "objs = sorted(build.glob('VectorCore.build/*.o'))",
            '<Target>.build directory join': "objs = sorted((build / 'VectorCoreC.build').rglob('*.o'))",
        }
        self.assertEqual(set(controls), {label for label, _ in PATTERNS})
        for label, line in controls.items():
            with self.subTest(label=label):
                self.assertIn(label, [rule for _, rule in hits(line)])
        recursive = "sorted(build.glob('VectorCoreC.build/**/*.o'))"
        self.assertIn('<Target>.build object glob', [rule for _, rule in hits(recursive)])

    def test_compliant_sources_are_accepted(self):
        compliant = ("objects = products.link_objects(bin_path, ['VectorCore'])\n"
                     "modules = products.module_search_path(bin_path)\n"
                     "name = 'VectorCore.build'  # a bare name, no layout path\n")
        self.assertEqual(hits(compliant), [])

    def test_scan_reports_a_violating_file_and_honours_the_allowlist(self):
        import tempfile
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / 'Scripts').mkdir()
            (root / 'Scripts' / 'driver.py').write_text("p = base / '.build/release'\n")
            self.assertEqual(scan(root), {'Scripts/driver.py': [(1, '.build/<config> literal')]})
            self.assertEqual(scan(root, allowlist={'Scripts/driver.py': 'test'}), {})


if __name__ == '__main__':
    unittest.main()
