#!/usr/bin/env python3
"""Regenerate Appendices A and B of the downstream upgrade list from CHANGELOG.md `[Unreleased]` (PR #56 review SW4-1).

Appendix A has one row per top-level `[Unreleased]` bullet, keyed CL-nn in order. The entry title (the bullet's bold
text, on one line) is the stable anchor: Task and Rows carry over from the committed appendix when the title is
unchanged (or when its first 48 normalized characters still match), otherwise Task is the task ids in the title's
trailing parenthetical and Rows are the rows of this document whose Source names one of those ids. Appendix B lists
every entry whose title starts with "Breaking" with its rows; an entry with no row is listed as `none`, never hidden.
The header's row-count table is recomputed from the sections. `--check` exits 1 when the committed guide differs from
the regenerated text (Scripts/ci/tests/test_downstream_appendices.py runs it); `--write` regenerates the guide.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / 'Docs/superpowers/execution/2026-09-18-vectorcore-stabilization/DOWNSTREAM_UPGRADES.md'
CHANGELOG = ROOT / 'CHANGELOG.md'
TASK = r'(?:[A-Z]\d{2}|#\d+)'
TRAILING_TASKS = re.compile(r'\((' + TASK + r'(?:,\s*' + TASK + r')*)\)\.?\s*$')
ROW = re.compile(r'^\| ([A-Z]+-\d+) \| (.*?) \| (.*?) \| (.*?) \| (.*?) \|$', re.M)
APPENDIX_ROW = re.compile(r'^\| (CL-\d+) \| (\w+) \| (.+?) \| (.*?) \| (.*?) \|$', re.M)
SECTION_TABLE = re.compile(r'^\| Section \| Rows \|\n\|---\|---\|\n(?:\|.*\|\n)+', re.M)


@dataclass
class Bullet:
    section: str
    title: str
    body: str

    @property
    def breaking(self):
        return self.title.startswith('Breaking')


def unreleased_bullets(changelog):
    start = changelog.index('## [Unreleased]')
    end = changelog.find('\n## [', start + 1)
    text = changelog[start:end if end != -1 else len(changelog)]
    bullets, section, current = [], None, None

    def close():
        nonlocal current
        if current is not None:
            bullets.append(Bullet(current[0], _title(current[1]), current[1]))
            current = None

    for line in text.splitlines():
        if line.startswith('### '):
            close()
            section = line[4:].strip()
        elif line.startswith('- '):
            close()
            current = [section, line[2:]]
        elif current is not None and line.startswith('  '):
            current[1] += '\n' + line
        elif current is not None and not line.strip():
            close()
    close()
    return bullets


def _title(body):
    match = re.match(r'\*\*(.+?)\*\*', body, re.S)
    return ' '.join((match.group(1) if match else body.split('\n')[0]).split())


def task_ids(title):
    match = TRAILING_TASKS.search(title)
    return [part.strip() for part in match.group(1).split(',')] if match else []


def normalize(title):
    return re.sub(r'[`*]', '', title).strip().rstrip('.').lower()


def _row_sources(guide):
    body = guide[:guide.index('## Appendix A.')]
    return {key: source for key, _, _, source, _ in ROW.findall(body) if not key.startswith('CL-')}


def rows_for_tasks(tasks, sources):
    found = {key for key, source in sources.items() if any(re.search(r'\b' + re.escape(task) + r'\b', source) for task in tasks)}
    return sorted(found, key=lambda key: (key.split('-')[0], int(key.split('-')[1])))


def _carry_over(guide):
    previous = {}
    for key, section, title, task, rows in APPENDIX_ROW.findall(guide):
        previous[normalize(title)] = (task, rows)
    return previous


def _lookup(previous, title):
    norm = normalize(title)
    if norm in previous:
        return previous[norm]
    prefix = norm[:48]
    matches = [value for key, value in previous.items() if key[:48] == prefix]
    return matches[0] if len(matches) == 1 else None


def _section_counts(guide):
    counts = []
    for heading, body in zip(re.findall(r'^## (.+)$', guide, re.M), re.split(r'^## .+$', guide, flags=re.M)[1:]):
        rows = len([key for key, *_ in ROW.findall(body) if not key.startswith('CL-')])
        label = heading.split(':')[0].split('.')[0] + '.' if heading[0].isdigit() else heading.split('.')[0] + '.'
        if rows and (heading[0].isdigit() or heading.startswith('Appendix E')):
            counts.append((heading, rows))
    return counts


def _section_label(heading):
    names = {'1': '1. Compile-breaking', '2': '2. Signature changes', '3': '3. Behaviour changes', '4': '4. ABI/layout',
             '5': '5. Performance', '6': '6. New APIs', '7': '7. Benchmarking and C'}
    if heading[0].isdigit():
        return names.get(heading[0], heading.split(':')[0])
    return 'Appendix E. Limitations'


def regenerate(guide, changelog):
    bullets = unreleased_bullets(changelog)
    previous, sources = _carry_over(guide), _row_sources(guide)
    entries = []
    for index, bullet in enumerate(bullets, 1):
        carried = _lookup(previous, bullet.title)
        if carried is not None:
            task, rows = carried
        else:
            tasks = task_ids(bullet.title)
            task = ', '.join(tasks) if tasks else 'none'
            mapped = rows_for_tasks(tasks, sources) if tasks else []
            rows = ', '.join(mapped) if mapped else 'none'
        entries.append((f'CL-{index:02d}', bullet, task, rows))
    appendix_a = ['## Appendix A. CHANGELOG `[Unreleased]` entry index (CL keys)', '',
                  'CL keys number the top-level bullets of `CHANGELOG.md` `[Unreleased]` at the current commit, in order; the entry '
                  'title is the stable anchor (a key moves when an entry is inserted before it). Task and Rows carry over from the '
                  'previous index when the title is unchanged; otherwise Task is the task id in the title and Rows are the rows '
                  'of this document whose Source names it. `none` marks an entry no row carries. Generated by '
                  '`Scripts/ci/downstream_appendices.py`; `Scripts/ci/tests/test_downstream_appendices.py` fails when this text is '
                  'stale.', '',
                  '| Key | Section | Entry (title as written) | Task (commit) | Rows |', '|---|---|---|---|---|']
    for key, bullet, task, rows in entries:
        appendix_a.append(f'| {key} | {bullet.section} | {bullet.title} | {task} | {rows} |')
    breaking = [(key, bullet, rows) for key, bullet, task, rows in entries if bullet.breaking]
    unmapped = sum(1 for _, _, rows in breaking if rows == 'none')
    appendix_b = ['## Appendix B. CHANGELOG Breaking lines mapped to rows', '',
                  f'Every top-level `[Unreleased]` bullet whose title starts with "Breaking" ({len(breaking)} entries); '
                  + (f'{unmapped} ' + ('entry has' if unmapped == 1 else 'entries have') + ' no row and ' + ('is' if unmapped == 1 else 'are') + ' listed as `none`.'
                     if unmapped else 'every one has at least one row.'), '',
                  '| CHANGELOG entry | Title | Rows |', '|---|---|---|']
    for key, bullet, rows in breaking:
        appendix_b.append(f'| {key} | {bullet.title} | {rows} |')
    head = guide[:guide.index('## Appendix A.')]
    tail = guide[guide.index('## Appendix C.'):]
    text = head + '\n'.join(appendix_a) + '\n\n' + '\n'.join(appendix_b) + '\n\n' + tail
    counts = _section_counts(text)
    table = '| Section | Rows |\n|---|---|\n' + ''.join(f'| {_section_label(h)} | {n} |\n' for h, n in counts)
    return SECTION_TABLE.sub(lambda _: table, text, count=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true', help='exit 1 when the committed guide is stale')
    group.add_argument('--write', action='store_true', help='regenerate the committed guide')
    args = parser.parse_args(argv)
    current = GUIDE.read_text()
    regenerated = regenerate(current, CHANGELOG.read_text())
    if args.write:
        GUIDE.write_text(regenerated)
        print('regenerated', GUIDE.relative_to(ROOT))
        return 0
    if regenerated != current:
        for number, (old, new) in enumerate(zip(current.splitlines(), regenerated.splitlines()), 1):
            if old != new:
                print(f'first difference at line {number}:\n  committed:   {old[:160]}\n  regenerated: {new[:160]}')
                break
        print('DOWNSTREAM_UPGRADES.md appendices are stale; run: python3 Scripts/ci/downstream_appendices.py --write')
        return 1
    print('DOWNSTREAM_UPGRADES.md appendices are current')
    return 0


if __name__ == '__main__':
    sys.exit(main())
