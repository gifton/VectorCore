"""F009 review ratchet for relative test oracles, not general Swift dataflow analysis.

Scan Tests/**/*.swift after the shared lexer masks comments and string literals.
Inventory <= comparisons whose RHS multiplies abs/max/scale/relTol/relativeTolerance,
and the absError > ... && relError > ... rejection form. Strict < comparisons and
fixed absolute tolerances do not admit inf <= inf and are outside this rule.

Each occurrence needs a reviewed (path, function, normalized statement) entry.
Finite-gated entries retain named code fragments in that function; independent
bounded-reference entries retain the reviewed function body digest. Dormant helper
entries must remain declaration-only across Tests and Sources. New, missing or
duplicated entries require review. Controls remove every registered finite gate,
alter bounded bodies, activate dormant helpers and add unreviewed comparison forms.

Limits: lexical syntax only; no aliases, arbitrary variable names, interprocedural
reasoning, macro expansion or control-flow dominance proof. A retained fragment
alone is not proof that it executes; behavioral Swift tests exercise the repaired
helper, and the inventory records the source review. Bounded entries' callers and
fixture generators are review evidence, not transitively hashed dependencies.
"""
import hashlib
import importlib.util
import json
import re
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('oracle_swift_lexing', ROOT / 'Scripts/ci/swift_lexing.py')
lexing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lexing)
INVENTORY = ROOT / 'Scripts/ci/relative_oracle_inventory.json'


def normalized(text):
    return re.sub(r'\s+', '', text)


def digest(text):
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def sites(source):
    masked = lexing._mask(source)
    bodies = list(lexing._Scopes(masked).bodies())
    offsets = [0]
    for line in masked.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    for line, statement in lexing._statements(masked):
        rhs = statement.split('<=', 1)[-1]
        relative = '<=' in statement and '*' in rhs and re.search(r'\b(?:abs|max|scale|relTol|relativeTolerance)\b', rhs)
        negative = re.search(r'\babsError\s*>.*&&.*\brelError\s*>', statement)
        if not (relative or negative):
            continue
        enclosing = [body for body in bodies if body[1] <= offsets[line - 1] < body[2]]
        name, start, end = min(enclosing, key=lambda body: body[2] - body[1]) if enclosing else ('<file>', 0, len(masked))
        yield {'function': name, 'expression': normalized(statement), 'line': line, 'body': masked[start:end]}


def problems(sources, reviewed):
    key = lambda item: (item['path'], item['function'], item['expression'])
    entries = {key(item): item for item in reviewed}
    findings = []
    if len(entries) != len(reviewed):
        findings.append('duplicate inventory key')
    found = Counter()
    combined = '\n'.join(lexing._mask(source) for source in sources.values())
    for path, source in sources.items():
        if not path.startswith('Tests/'):
            continue
        for site in sites(source):
            identity = (path, site['function'], site['expression'])
            found[identity] += 1
            entry = entries.get(identity)
            location = f'{path}:{site["line"]} ({site["function"]})'
            if entry is None:
                findings.append(f'{location}: unreviewed relative oracle')
                continue
            if not entry.get('reason'):
                findings.append(f'{location}: missing review reason')
            category = entry['category']
            if category == 'finite-gated':
                if not entry['required'] or any(fragment not in normalized(site['body']) for fragment in entry['required']):
                    findings.append(f'{location}: missing finite gate')
            elif category == 'bounded-reference':
                if digest(site['body']) != entry['body_sha256']:
                    findings.append(f'{location}: bounded-reference body changed; review required')
            elif category == 'dormant':
                if len(re.findall(r'\b' + re.escape(site['function']) + r'\b', combined)) != 1:
                    findings.append(f'{location}: dormant oracle activated; repair before use')
                if digest(site['body']) != entry['body_sha256']:
                    findings.append(f'{location}: dormant body changed; review required')
            else:
                findings.append(f'{location}: unknown inventory category {category}')
    for identity in entries:
        if found[identity] != 1:
            findings.append(f'{identity}: expected one occurrence, found {found[identity]}')
    return findings


class RelativeOracleGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reviewed = json.loads(INVENTORY.read_text())['sites']
        cls.sources = {str(path.relative_to(ROOT)): path.read_text()
                       for directory in ['Tests', 'Sources'] for path in (ROOT / directory).rglob('*.swift')}

    def test_live_inventory(self):
        self.assertTrue(self.reviewed)
        self.assertEqual(problems(self.sources, self.reviewed), [])

    def test_unreviewed_relative_forms_are_rejected(self):
        for expression in ['return diff <= absTol + relTol * scale',
                           '#expect(abs(actual-ref)<=1e-5*max(1,abs(ref)))',
                           '#expect(abs(actual-ref)<=1e-2+1e-3*abs(ref))',
                           'if absError > absoluteTolerance && relError > relativeTolerance { return false }']:
            with self.subTest(expression=expression):
                source = 'func unreviewed() {\n' + expression + '\n}'
                self.assertTrue(any('unreviewed relative oracle' in item for item in problems({'Tests/New.swift': source}, [])))

    def test_comments_strings_and_absolute_tolerances_are_not_sites(self):
        source = '''func control() {
            // return diff <= absTol + relTol * scale
            let text = "#expect(abs(a-b) <= tolerance * abs(b))"
            #expect(abs(a-b) <= 1e-5)
            #expect(abs(a-b) < tolerance * abs(b))
        }'''
        self.assertEqual(list(sites(source)), [])

    def test_removing_each_finite_gate_is_rejected(self):
        controls = 0
        for entry in self.reviewed:
            if entry['category'] != 'finite-gated':
                continue
            for gate in entry['required']:
                with self.subTest(path=entry['path'], function=entry['function'], gate=gate):
                    source = self.sources[entry['path']]
                    start, end = next((start, end) for name, start, end in lexing._Scopes(source).bodies()
                                      if name == entry['function'])
                    pattern = r'\s*'.join(re.escape(character) for character in gate)
                    match = re.search(pattern, lexing._mask(source)[start:end])
                    self.assertIsNotNone(match, 'registered gate is absent')
                    mutated = source[:start + match.start()] + 'true' + source[start + match.end():]
                    reviewed = [item for item in self.reviewed if item['path'] == entry['path']]
                    self.assertTrue(any(f'({entry["function"]})' in item and
                                        ('missing finite gate' in item or 'unreviewed relative oracle' in item)
                                        for item in problems({entry['path']: mutated}, reviewed)))
                    controls += 1
        self.assertGreater(controls, 0)

    def test_bounded_bodies_and_dormant_activation_require_review(self):
        for entry in self.reviewed:
            if entry['category'] not in ['bounded-reference', 'dormant']:
                continue
            with self.subTest(path=entry['path'], function=entry['function']):
                source = self.sources[entry['path']]
                start = next(start for name, start, _ in lexing._Scopes(source).bodies() if name == entry['function'])
                mutated = source[:start] + '\nlet changedReference = Float.infinity\n' + source[start:]
                self.assertTrue(any('body changed' in item for item in problems({entry['path']: mutated}, [entry])))
                if entry['category'] == 'dormant':
                    activated = dict(self.sources, **{'Tests/New.swift': 'func activate() { _ = ' + entry['function'] + ' }'})
                    self.assertTrue(any('dormant oracle activated' in item for item in problems(activated, self.reviewed)))

    def test_missing_and_duplicate_sites_require_review(self):
        entry = self.reviewed[0]
        self.assertTrue(any('expected one occurrence' in item for item in problems({}, [entry])))
        self.assertTrue(any('duplicate inventory key' in item for item in problems({}, [entry, entry])))
