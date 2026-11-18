"""Keep personal workstation paths and captured temp roots out of current files.

Synthetic absolute-path fixtures and ordinary system/runtime paths are permitted;
the existing Swift benchmark fixture is the only home-path exception. Findings
contain filenames and line numbers, never the matched private path.
"""
from pathlib import Path
import os
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
PATTERNS = (
    re.compile(r'/(?:Users|home)/[^/\s"\'<>`)]+', re.I),
    re.compile(r'[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s"\'<>`]+', re.I),
    re.compile(r'/(?:private/)?var/folders/[^\s"\'<>`]+'),
    re.compile(r'/(?:private/)?tmp/[^/\s"\'<>`]*20\d{2}[-_]?\d{2}[-_]?\d{2}[^/\s"\'<>`]*'),
)
SYNTHETIC_FILE = 'Tests/BenchmarkContractTests/BenchmarkSelectionContractTests.swift'
SYNTHETIC_HOME = '/' + 'Users' + '/me'


def findings(relative, text):
    found = set()
    patterns = PATTERNS
    if relative.startswith('Docs/'):
        patterns += (re.compile(r'/(?:private/)?tmp/'),)
    for number, line in enumerate(text.replace('\\/', '/').splitlines(), 1):
        for pattern in patterns:
            for match in pattern.finditer(line):
                if relative == SYNTHETIC_FILE and match.group() == SYNTHETIC_HOME:
                    continue
                found.add(number)
    return sorted(found)


def scan(root):
    root = Path(root)
    files = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z'], timeout=30)
    result = {}
    for name in files.decode().split('\0'):
        if not name:
            continue
        path = root / name
        if path.is_symlink():
            text = os.readlink(path)
        elif path.is_file():
            text = path.read_bytes().decode('utf-8', errors='replace')
        else:
            continue
        hits = findings(name, text)
        if hits:
            result[name] = hits
    return result


class PublicPathTests(unittest.TestCase):
    def test_current_repository_contains_no_personal_or_captured_paths(self):
        result = scan(ROOT)
        summary = '\n'.join(f'{name}: lines {lines[:5]} ({len(lines)} total)' for name, lines in result.items())
        self.assertEqual(len(result), 0, summary)

    def test_home_variants_links_json_keys_and_capture_roots_are_detected(self):
        home = '/' + 'Users' + '/fictional-developer'
        examples = [home, home + '/project/file.swift', '[record](' + home + '/evidence.json)',
                    '{"' + home + '/capture": "digest"}',
                    ('/' + 'home' + '/fictional-developer/project'),
                    ('C:' + '\\' + 'Users' + '\\fictional-developer\\project'),
                    ('/' + 'private/var/folders' + '/example/cache/result.json'),
                    ('/' + 'private/tmp' + '/capture-2026-01-01-unique/result.json'),
                    home.replace('/', '\\/')]
        for value in examples:
            with self.subTest(value=value):
                self.assertEqual(findings('Docs/record.md', 'public\n' + value + '\n'), [2])

    def test_synthetic_exception_does_not_hide_other_users_or_other_files(self):
        fixture = SYNTHETIC_HOME + '/project/out'
        self.assertEqual(findings(SYNTHETIC_FILE, fixture), [])
        self.assertEqual(findings('Docs/record.md', fixture), [1])
        self.assertEqual(findings(SYNTHETIC_FILE, fixture.replace('/me/', '/someone-else/')), [1])

    def test_portable_roots_and_standard_system_paths_are_allowed(self):
        for text in ('${SOURCE_ROOT}/Sources', '${EVIDENCE_ROOT}/receipt.json',
                     '/Applications/Xcode.app/Contents/Developer', '/usr/bin/swift',
                     'Scripts/ci/compiler_qualification.py', '/tmp/runtime-test-file'):
            with self.subTest(text=text):
                self.assertEqual(findings('Scripts/fixture.py', text), [])

    def test_scans_tracked_working_contents_and_symlink_text_without_following_links(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            subprocess.run(['git', 'init', '-q', str(root)], check=True, capture_output=True, timeout=20)
            path = root / 'record.json'
            path.write_text('{}')
            link = root / 'link'
            link.symlink_to('/' + 'Users' + '/fictional-developer/missing')
            subprocess.run(['git', '-C', str(root), 'add', '.'], check=True, capture_output=True, timeout=20)
            path.write_text('{"root":"' + '/' + 'home' + '/fictional-developer/project"}')
            (root / 'untracked.md').write_text('/' + 'Users' + '/fictional-developer/private')
            self.assertEqual(scan(root), {'link': [1], 'record.json': [1]})


if __name__ == '__main__':
    unittest.main()
