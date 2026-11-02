"""Policy (PR #56 review SW4-1): Appendices A and B of the downstream upgrade list are regenerated from CHANGELOG.md.

WHY: the index of `[Unreleased]` entries (CL keys) was compiled by hand on 2026-10-16 and keyed by position; T18/T19
then added about thirty entries, so the keys pointed at the wrong bullets and 19 Breaking entries had no row. The
generator (Scripts/ci/downstream_appendices.py) derives Appendix A (one row per top-level bullet, title as the stable
anchor, Task and Rows carried over by title or mapped by the task ids in the title) and Appendix B (every Breaking
entry with its rows, unmapped ones listed as `none`) from the CHANGELOG at the current commit, and recomputes the
header's row-count table. `--check` fails when the committed guide differs from the regenerated text, so the index
cannot drift again; `--write` regenerates it. The parser and the mapping are pinned on fixtures here.

LIMITS: Task and Rows for an entry whose title changed are carried over only when the first 48 normalized characters
still match; otherwise they come from the task ids in the title, or are `none`. A row's Source must name the task id
as a word (`S01`), as every existing row does. Bodies are not parsed for ids.
"""
from pathlib import Path
import importlib.util
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / 'Scripts/ci/downstream_appendices.py'


def load():
    spec = importlib.util.spec_from_file_location('downstream_appendices', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHANGELOG = """# Changelog
## [Unreleased]

### Added

- **`Widget` (S01).** Body.

### Changed

- **Breaking: the long title spans
  two lines (R01, R05).** Body line one.
  Body line two.

- **Breaking: no task here.** Body.

- Plain bullet without a bold title (E02).

## [0.3.3] - 2026-09-06

- **Released entry (X99).** Not unreleased.
"""

GUIDE = """# Guide

| Section | Rows |
|---|---|
| 1. Compile-breaking | 0 |
| 3. Behaviour changes | 0 |

## 1. Compile-breaking: removed or renamed public API

| ID | Change | Consumer action | Source | Tag |
|---|---|---|---|---|
| CB-01 | Removed thing. | Do. | S01 (abc1234); CL-01 | stated |

## 3. Behaviour changes with no compile error (the dangerous list)

| ID | Change | Consumer action | Source | Tag |
|---|---|---|---|---|
| BH-01 | Behaviour. | Do. | R05 (def5678) | stated |
| BH-02 | Other. | Do. | E02 (0123456) | observed |

## Appendix A. CHANGELOG `[Unreleased]` entry index (CL keys)

old intro

| Key | Section | Entry (title as written) | Task (commit) | Rows |
|---|---|---|---|---|
| CL-01 | Changed | Breaking: no task here | T07 (fedcba9) | BH-02, CB-01 |

## Appendix B. CHANGELOG Breaking lines mapped to rows

old

## Appendix C. Completeness

kept as is
"""


class ParserTests(unittest.TestCase):
    def test_bullets_titles_and_tasks(self):
        m = load()
        bullets = m.unreleased_bullets(CHANGELOG)
        self.assertEqual([(b.section, b.title) for b in bullets], [
            ('Added', '`Widget` (S01).'),
            ('Changed', 'Breaking: the long title spans two lines (R01, R05).'),
            ('Changed', 'Breaking: no task here.'),
            ('Changed', 'Plain bullet without a bold title (E02).'),
        ])
        self.assertEqual([m.task_ids(b.title) for b in bullets], [['S01'], ['R01', 'R05'], [], ['E02']])
        self.assertEqual([b.breaking for b in bullets], [False, True, True, False])


class RenderTests(unittest.TestCase):
    def test_carry_over_mapping_and_unmapped_breaking(self):
        m = load()
        text = m.regenerate(GUIDE, CHANGELOG)
        a = text[text.index('## Appendix A.'):text.index('## Appendix B.')]
        b = text[text.index('## Appendix B.'):text.index('## Appendix C.')]
        self.assertIn('| CL-01 | Added | `Widget` (S01). | S01 | CB-01 |', a)                       # mapped by task id
        self.assertIn('| CL-02 | Changed | Breaking: the long title spans two lines (R01, R05). | R01, R05 | BH-01 |', a)
        self.assertIn('| CL-03 | Changed | Breaking: no task here. | T07 (fedcba9) | BH-02, CB-01 |', a)  # carried over
        self.assertIn('| CL-04 | Changed | Plain bullet without a bold title (E02). | E02 | BH-02 |', a)
        self.assertIn('(2 entries', b)
        self.assertIn('| CL-02 | Breaking: the long title spans two lines (R01, R05). | BH-01 |', b)
        self.assertIn('| CL-03 | Breaking: no task here. | BH-02, CB-01 |', b)
        self.assertIn('| 1. Compile-breaking | 1 |', text)
        self.assertIn('| 3. Behaviour changes | 2 |', text)
        self.assertIn('kept as is', text)
        self.assertEqual(m.regenerate(text, CHANGELOG), text, 'regeneration is idempotent')

    def test_unmapped_breaking_entry_is_listed_as_none(self):
        m = load()
        guide = GUIDE.replace('| CL-01 | Changed | Breaking: no task here | T07 (fedcba9) | BH-02, CB-01 |\n', '')
        text = m.regenerate(guide, CHANGELOG)
        b = text[text.index('## Appendix B.'):text.index('## Appendix C.')]
        self.assertIn('| CL-03 | Breaking: no task here. | none |', b)
        self.assertIn('1 entry has no row', b)


class CommittedGuideTests(unittest.TestCase):
    def test_committed_guide_matches_the_changelog(self):
        result = subprocess.run([sys.executable, str(SCRIPT), '--check'], capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-2000:])


if __name__ == '__main__':
    unittest.main()
