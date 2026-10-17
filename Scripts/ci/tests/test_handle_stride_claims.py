"""Policy (HEURISTIC): the in-memory size of an owning handle must not appear in tests unreviewed.

`MemoryLayout<Vector512Optimized>.stride` is the inline size of an owning handle. It is unrelated to
the size of the payload the handle owns, yet tests used it as "50% memory saved" (the TOOL-08 class,
V04 phase 1 inventory 6.7). This policy flags, in every Tests/**/*.swift file, each lexically visible:

* `MemoryLayout<T>.size` / `.stride` (also written `Swift.MemoryLayout`) where some identifier of T names
  a HANDLE TYPE, and
* `MemoryLayout.size(ofValue:)` / `.stride(ofValue:)` / `MemoryLayout<T>.size(ofValue:)` of any value
  (the value's type is not lexically known, so every such use needs review),

wherever it appears (an assertion, `print`, `Issue.record`, `NSLog`, a logger, a custom assertion helper
or a bare `let`): the expression itself is the trigger, not the sink it flows into. A hit passes only if
the reviewed ALLOWLIST holds its (file, enclosing function, whitespace-free expression) with the exact
occurrence count and a reason starting with a category prefix. An entry whose count differs, or that
matches nothing, FAILS (stale check). A payload or memory-saving claim is never allowlisted: fix it.

HANDLE TYPES are derived from Sources/**/*.swift (not a name list), by the shared lexer's masking:
a type declared `class`/`final class`/`actor`, plus every `struct` with at least one STORED property
(a `let`/`var` at type-body depth, not `static`/`class`, with no accessor body) whose declared type is
`Array`, `ContiguousArray`, `[...]`, `Dictionary`, `Set`, `Data`, `String`, `ManagedBuffer`, an
`Unsafe*Pointer` / `Unsafe*BufferPointer`, a handle type, or a `typealias` whose right-hand side is one
of these (a fixpoint). An unannotated stored property counts only if its initializer starts with `[`,
`Array(`, `ContiguousArray(` and so on. Generic uses match by base name (`Vector<Dim512>` -> `Vector`).
A `typealias Name = <expression mentioning a handle type>` in the scanned Tests files adds `Name` to the
handle set for the scan (a fixpoint, independent of declaration order, with no scoping), so
`private typealias H = Vector512Optimized` does not hide `MemoryLayout<H>`. Scope: the enforcing test
expands aliases across ALL scanned Tests files together (an alias declared in one file applies to every
other); `--report` expands them per file only. A computed property (accessor body, no initializer) does not make its
struct a handle.

Limits (nothing beyond the named controls is proven):
* Every `#if` branch is seen: the source is scanned without preprocessing, so a stored property in any
  branch counts and a layout in any branch is flagged.
* A stored property with `didSet`/`willSet` observers AND an initializer is treated as stored; one with
  observers and no initializer is treated as computed. Macro-generated storage and multi-line property
  declarations (a type or initializer continued on the next line) are not seen.
* NOT flagged: stdlib owning containers (`MemoryLayout<[Float]>`, `MemoryLayout<ContiguousArray<Float>>`,
  `<Array<Float>>`, `<String>`): the set holds only types declared in Sources/. Handle types declared in
  Tests/ are not flagged either (only their typealiases of Sources handles are).
* An `@available`-annotated stored property IS seen.
* A typealias name declared with several right-hand sides is in the set if ANY is.
* Alias over-approximation fails safe (extra findings, never missed ones): any alias whose right-hand
  side merely mentions a handle name is a handle, e.g. `typealias Sc = Vector512Optimized.Scalar`
  (really `Float`) and a generic alias `typealias Dup<Key> = ...` whose right-hand side names a handle.
* Base-name matching over-approximates: a generic parameter or local type that shadows a handle name is
  flagged (report-only scan: `Sources/VectorCore/Utilities/MemoryUtilities.swift:67`,
  `MemoryLayout<Vector>.size` with `<Vector: SIMD>`; `VectorData<Vector: ...>` in
  `Sources/VectorCore/Types/VectorData.swift:38` would be flagged the same way). Nested types are keyed by
  simple name.
* Not seen: `MemoryLayout<Self>`, `MemoryLayout` reached through a typealias of the layout type itself,
  layouts inside raw-string interpolations, and types reached only through protocol existentials or
  generic parameters of a Tests-side generic.
* `_code_view` restores the code of every `\\(` interpolation. It does not tell a string from a comment
  or an escaped backslash (`"\\\\("`), so it can restore MORE text than strictly code; that fail-safe
  can only add findings, never hide one (pinned by test_code_view_is_failsafe).
Benchmarks/ and Sources/ are scanned report-only (`--report`), not enforced here.

Controls (all in this file): NEGATIVE (the original instances at a788372 via `git show`, plus inline
fixtures for every must-flag form) must yield findings; POSITIVE must not; each rule is shown to bite by
mutation (disabling it makes its must-flag controls pass unflagged); the live scan of Tests/ must match the
allowlist exactly.
"""

import importlib.util
import re
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location('swift_lexing', ROOT / 'Scripts/ci/swift_lexing.py')
lexing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lexing)
_mask, _Scopes = lexing._mask, lexing._Scopes

BASELINE_COMMIT = 'a788372'
CATEGORIES = ('LAYOUT-PIN:', 'HANDLE-MODEL:')
STORAGE_BASES = {'Array', 'ContiguousArray', 'Dictionary', 'Set', 'Data', 'String', 'ManagedBuffer'}
TYPE_DECL = re.compile(r'(?<![\w.])(final\s+class|class|actor|struct)\s+(?!func\b|var\b|let\b)(\w+)')
TYPEALIAS = re.compile(r'\btypealias\s+(\w+)\s*(?:<[^=]*>)?\s*=\s*([^\n;{]+)')
PROPERTY = re.compile(r'\b(let|var)\s+(\w+)\s*(?::\s*([^={]+?))?\s*(=.*)?$', re.S)
INFERRED_STORAGE = re.compile(r'=\s*(?:\[|Array\s*[<(]|ContiguousArray\s*[<(]|Dictionary\s*[<(]|Set\s*[<(]|Data\s*\()')
LAYOUT = re.compile(r'(?<![\w.])(?:Swift\s*\.\s*)?MemoryLayout\b')
IDENT = re.compile(r'[A-Za-z_]\w*')


def _matching(text, start, open_ch, close_ch):
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch and not (close_ch == '>' and text[i - 1] == '-'):
            depth -= 1
            if depth == 0:
                return i
    return -1


def _code_view(source):
    """`_mask(source)` with the code inside string interpolations `\\( ... )` restored.

    Offsets equal source offsets. Plain string text and plain comment text stay blank, so a layout
    expression merely written in them is not seen. The restoration is driven only by the two characters
    `\\(` whose position `_mask` blanked, so it also restores an interpolation-looking `\\(...)` inside a
    comment and after an escaped backslash (`"\\\\("`). That over-restoration is fail-safe (it can only
    add findings, never hide one) and is pinned by test_code_view_is_failsafe.
    """
    masked = list(_mask(source))
    i = 0
    while True:
        i = source.find('\\(', i)
        if i < 0:
            return ''.join(masked)
        if masked[i] == ' ':
            end = _matching(source, i + 1, '(', ')')
            if end > 0:
                masked[i + 2:end] = list(source[i + 2:end])
                i = end
                continue
        i += 2


def _base_name(type_text):
    """Base identifier of a type expression: `Swift.ContiguousArray<X>?` -> ContiguousArray."""
    text = type_text.strip().rstrip('?!').strip()
    head = re.match(r'[\w.]+', text)
    return head.group(0).split('.')[-1] if head else ''


def _is_storage_type(type_text, handles):
    text = type_text.strip().rstrip('?!').strip()
    if text.startswith('['):
        return True
    base = _base_name(text)
    if base == 'Optional':
        inner = text[text.index('<') + 1:text.rindex('>')] if '<' in text else ''
        return _is_storage_type(inner, handles)
    return (base in STORAGE_BASES or (base.startswith('Unsafe') and base.endswith('Pointer'))
            or base in handles)


def _type_bodies(masked):
    """(kind, name, body) for every class/actor/struct declaration; nested blocks flattened to `{}`."""
    for match in TYPE_DECL.finditer(masked):
        open_at = masked.find('{', match.end())
        if open_at < 0:
            continue
        close_at = _matching(masked, open_at, '{', '}')
        if close_at < 0:
            continue
        flat, depth = [], 0
        for ch in masked[open_at + 1:close_at]:
            if ch == '{':
                if depth == 0:
                    flat.append('{}')
                depth += 1
            elif ch == '}':
                depth -= 1
            elif depth == 0:
                flat.append(ch)
        yield match.group(1).split()[-1], match.group(2), ''.join(flat)


def _stored_property_types(body, include_computed=False):
    """(name, declared type text or None, initializer text) for stored properties at body depth.

    `include_computed` exists only for the mutation control (a computed `var x: T {}` then counts).
    """
    for statement in re.split(r'[\n;]', body):
        found = re.search(r'\b(let|var)\b', statement)
        if not found or re.search(r'\b(?:static|class)\b', statement[:found.start()]):
            continue
        text = statement[found.start():].strip()
        if include_computed:
            text = re.sub(r'\s*\{\}\s*$', '', text)
        match = PROPERTY.match(text)
        if match:
            yield match.group(2), match.group(3), match.group(4) or ''


def derive_handle_types(sources, _include_computed=False, _fixpoint=True):
    """Handle-type names derived from {path: source}; see the module docstring for the rule.

    The underscore parameters exist only for the mutation controls.
    """
    classes, structs, aliases = set(), {}, {}
    for source in sources.values():
        masked = _mask(source)
        for kind, name, body in _type_bodies(masked):
            if kind in ('class', 'actor'):
                classes.add(name)
            else:
                structs.setdefault(name, []).append(body)
        for alias, target in TYPEALIAS.findall(masked):
            aliases.setdefault(alias, []).append(target)
    handles = set(classes)
    changed, passes = True, 0
    while changed and (_fixpoint or passes == 0):
        changed = False
        passes += 1
        for alias, targets in aliases.items():
            if alias not in handles and any(_is_storage_type(t, handles) for t in targets):
                handles.add(alias)
                changed = True
        for name, bodies in structs.items():
            if name in handles:
                continue
            for body in bodies:
                if any((declared is not None and _is_storage_type(declared, handles))
                       or (declared is None and INFERRED_STORAGE.match(init.strip()))
                       for _, declared, init in _stored_property_types(body, _include_computed)):
                    handles.add(name)
                    changed = True
                    break
    return handles


def expand_aliases(sources, handles, _fixpoint=True):
    """`handles` plus every typealias in `sources` whose right-hand side mentions a handle (fixpoint).

    `_fixpoint=False` (one pass in declaration order) exists only for the mutation control.
    """
    aliases = []
    for source in sources:
        aliases.extend(TYPEALIAS.findall(_mask(source)))
    result, changed, passes = set(handles), True, 0
    while changed and (_fixpoint or passes == 0):
        changed = False
        passes += 1
        for alias, target in aliases:
            if alias not in result and any(name in result for name in IDENT.findall(target)):
                result.add(alias)
                changed = True
    return result


def _load_tree(directory):
    return {str(p.relative_to(ROOT)): p.read_text(encoding='utf-8', errors='replace')
            for p in sorted((ROOT / directory).rglob('*.swift'))}


_SOURCES_HANDLES = None


def sources_handle_types():
    global _SOURCES_HANDLES
    if _SOURCES_HANDLES is None:
        _SOURCES_HANDLES = derive_handle_types(_load_tree('Sources'))
    return _SOURCES_HANDLES


def find_layout_uses(source, handles, rules=('type-layout', 'ofValue'), sinks_only=False, aliases=True, alias_fixpoint=True):
    """[(line, enclosing function, whitespace-free expression)] of handle layouts in `source`.

    `rules` and `sinks_only` exist only so the mutation controls can disable a rule; production
    use leaves them at their defaults (both rules on, every position).
    """
    if aliases:
        handles = expand_aliases([source], handles, alias_fixpoint)
    masked = _code_view(source)
    scopes = _Scopes(source)
    found = []
    for match in LAYOUT.finditer(masked):
        i = match.end()
        while i < len(masked) and masked[i].isspace():
            i += 1
        generic = None
        if i < len(masked) and masked[i] == '<':
            close = _matching(masked, i, '<', '>')
            if close < 0:
                continue
            generic, i = masked[i + 1:close], close + 1
        member = re.compile(r'\s*\.\s*(size|stride)\b').match(masked, i)
        if not member:
            continue
        end = member.end()
        of_value = re.compile(r'\s*\(\s*ofValue\b').match(masked, end)
        if of_value:
            if 'ofValue' not in rules:
                continue
            paren = masked.find('(', end)
            end = _matching(masked, paren, '(', ')') + 1
        else:
            if 'type-layout' not in rules or generic is None:
                continue
            if not any(name in handles for name in IDENT.findall(generic)):
                continue
        line = masked.count('\n', 0, match.start()) + 1
        scope = scopes.at[match.start()]
        while scope and not scopes.name[scope]:
            scope = scopes.parent[scope]
        function = scopes.name[scope] if scope else '<file>'
        expression = re.sub(r'\s+', '', source[match.start():end])
        if sinks_only and not re.search(r'(?<![\w#])#(?:expect|require)\s*\(|\bXCTAssert\w*\s*\(|(?<![\w.])print\s*\(',
                                        masked[source.rfind('\n', 0, match.start()) + 1:match.start()]):
            continue
        found.append((line, function, expression))
    return found


def check_files(files, handles, allowlist):
    """Problems for {path: source} against the reviewed allowlist ({(file, function, expr): (reason, count)})."""
    problems, counts = [], Counter()
    handles = expand_aliases(files.values(), handles)
    for path, source in files.items():
        for line, function, expression in find_layout_uses(source, handles):
            key = (path, function, expression)
            counts[key] += 1
            if key not in allowlist:
                problems.append(f'{path}:{line}: {function}: {expression}: handle layout not allowlisted')
    for key, (reason, count) in allowlist.items():
        if not reason.startswith(CATEGORIES):
            problems.append(f'{key}: allowlist reason lacks a category prefix {CATEGORIES}')
        if counts[key] != count:
            problems.append(f'{key}: allowlist count {count} != {counts[key]} actual (stale entry)')
    return problems


# Reviewed allowlist. (file, enclosing function, expression) -> (reason, count).
# Every entry is an explicitly labelled handle-layout display, never a payload or memory-saving claim.
_BENCH = 'Tests/ComprehensiveTests/MixedPrecisionPerformanceBenchmark.swift'
_MODEL = 'HANDLE-MODEL: printed in the handle-stride columns/handle-array model of analyzeRepresentationMemoryModel, labelled as handle size, never as payload or savings'
_LAYOUT = 'Tests/ComprehensiveTests/P02HandleLayoutTests.swift'
_PIN = ('LAYOUT-PIN: P02 RD7 pins the inline handle size (stride == ContiguousArray<SIMD4<Float>> stride after scalarCount '
        'became computed); an equality of handle layouts, never a payload or memory claim')
ALLOWLIST = {
    (_BENCH, 'analyzeRepresentationMemoryModel', 'MemoryLayout<Vector512Optimized>.stride'): (_MODEL, 2),
    (_BENCH, 'analyzeRepresentationMemoryModel', 'MemoryLayout<MixedPrecisionKernels.Vector512FP16>.stride'): (_MODEL, 2),
    (_BENCH, 'analyzeRepresentationMemoryModel', 'MemoryLayout<Vector768Optimized>.stride'): (_MODEL, 1),
    (_BENCH, 'analyzeRepresentationMemoryModel', 'MemoryLayout<MixedPrecisionKernels.Vector768FP16>.stride'): (_MODEL, 1),
    (_BENCH, 'analyzeRepresentationMemoryModel', 'MemoryLayout<Vector1536Optimized>.stride'): (_MODEL, 1),
    (_BENCH, 'analyzeRepresentationMemoryModel', 'MemoryLayout<MixedPrecisionKernels.Vector1536FP16>.stride'): (_MODEL, 1),
    (_LAYOUT, 'handleStrideIsTheStorageStride', 'MemoryLayout<Vector384Optimized>.stride'): (_PIN, 1),
    (_LAYOUT, 'handleStrideIsTheStorageStride', 'MemoryLayout<Vector512Optimized>.stride'): (_PIN, 1),
    (_LAYOUT, 'handleStrideIsTheStorageStride', 'MemoryLayout<Vector768Optimized>.stride'): (_PIN, 1),
    (_LAYOUT, 'handleStrideIsTheStorageStride', 'MemoryLayout<Vector1536Optimized>.stride'): (_PIN, 1),
}


def _baseline(path):
    done = subprocess.run(['git', 'show', f'{BASELINE_COMMIT}:{path}'], cwd=ROOT, capture_output=True,
                          text=True, timeout=30)
    return done.stdout if done.returncode == 0 else None


# Lines that MUST be found at BASELINE_COMMIT (inventory 6.7 instances and the benchmark table).
# Hits there are by direct layout expression; the original files contained these expressions.
ORIGINAL_INSTANCES = {
    'Tests/ComprehensiveTests/MixedPrecisionPart1ValidationTests.swift': {153, 154},
    'Tests/ComprehensiveTests/MixedPrecisionKernelsTests.swift': {328, 329, 370, 371, 807, 808},
    'Tests/ComprehensiveTests/MixedPrecisionPerformanceBenchmark.swift': {302, 303, 305, 306, 308, 309, 322, 323},
}

MUST_FLAG = {
    'size(ofValue:)': ('let n = MemoryLayout.size(ofValue: v)\n', 'ofValue'),
    'stride(ofValue:)': ('let n = MemoryLayout.stride(ofValue: v)\n', 'ofValue'),
    'generic ofValue': ('let n = MemoryLayout<Float>.size(ofValue: v)\n', 'ofValue'),
    'INT8 handle': ('let n = MemoryLayout<Vector512INT8>.stride\n', 'type-layout'),
    'generic Vector': ('let n = MemoryLayout<Vector<Dim512>>.size\n', 'type-layout'),
    'Issue.record': ('Issue.record("\\(MemoryLayout<Vector512FP16>.stride)")\n', 'sink'),
    'NSLog': ('NSLog("%d", MemoryLayout<Vector512Optimized>.stride)\n', 'sink'),
    'logger': ('logger.info("\\(MemoryLayout<SoAFP16<Vector512Optimized>>.stride)")\n', 'sink'),
    'custom helper': ('assertSmall(MemoryLayout<Vector512FP16>.size)\n', 'sink'),
    'Swift.MemoryLayout': ('let n = Swift.MemoryLayout<Vector512Optimized>.stride\n', 'type-layout'),
    'function-type argument': ('let n = MemoryLayout<() -> Vector512FP16>.size\n', 'type-layout'),
    'private alias': ('private typealias H = Vector512Optimized\nlet n = MemoryLayout<H>.stride\n', 'alias'),
    'type-level alias': ('struct T {\n  typealias H = Vector512FP16\n  func f() { _ = MemoryLayout<H>.size }\n}\n', 'alias'),
    'alias of alias': ('typealias A = Vector768Optimized\nfileprivate typealias B = A\nlet n = MemoryLayout<B>.stride\n', 'alias'),
    'reverse alias chain': ('typealias B = A\ntypealias A = Vector768Optimized\nlet n = MemoryLayout<B>.stride\n', 'alias-fixpoint'),
    'bare let': ('let x = MemoryLayout<SoAFP16<Vector512Optimized>>.stride\n', 'sink'),
}
MUST_PASS = {
    'Float stride': 'let a = MemoryLayout<Float>.stride\n',
    'SIMD4<Float>': 'let a = MemoryLayout<SIMD4<Float>>.stride\n',
    'UInt16 size': 'let a = MemoryLayout<UInt16>.size\n',
    'in string': 'print("MemoryLayout<Vector512Optimized>.stride is a handle")\n',
    'in comment': '// let a = MemoryLayout<Vector512Optimized>.stride\n',
    'scalar alias': 'typealias S = Float\nlet a = MemoryLayout<S>.stride\n',
    'alignment': 'let a = MemoryLayout<Vector512Optimized>.alignment\n',
}


class HandleTypeDerivation(unittest.TestCase):
    def test_pinned_types_are_derived_from_sources(self):
        derived = sources_handle_types()
        required = {'Vector512Optimized', 'Vector512FP16', 'Vector768FP16', 'Vector1536FP16', 'SoAFP16',
                    'Vector512INT8', 'Vector768INT8', 'Vector1536INT8', 'SoAINT8', 'Vector'}
        self.assertEqual(required - derived, set())

    def test_scalar_only_struct_is_not_a_handle(self):
        self.assertNotIn('LinearQuantizationParams', sources_handle_types())

    def test_new_handle_struct_is_picked_up_without_a_list_edit(self):
        source = ('struct Fresh { var items: [Int] }\nstruct Plain { let a: Int; var b: Float }\n'
                  'final class Boxed { var x = 1 }\nstruct Wraps { var inner: Fresh }\n'
                  'struct Computed { var a: Int { 1 }\n static var shared: [Int] = [] }\n'
                  'struct Aliased { typealias Store = ContiguousArray<Float>\n var s: Store }\n')
        derived = derive_handle_types({'x.swift': source})
        self.assertEqual(derived & {'Fresh', 'Plain', 'Boxed', 'Wraps', 'Computed', 'Aliased', 'Store'},
                         {'Fresh', 'Boxed', 'Wraps', 'Aliased', 'Store'})
        flagged = find_layout_uses('let n = MemoryLayout<Fresh>.stride\nlet m = MemoryLayout<Plain>.stride\n', derived)
        self.assertEqual([expr for _, _, expr in flagged], ['MemoryLayout<Fresh>.stride'])


class DerivationRulePins(unittest.TestCase):
    """Each derivation rule is pinned by a control that fails when the rule is removed."""

    COMPUTED = 'struct OnlyComputed {\n  var x: [Float] { [] }\n  let n: Int\n}\n'
    CHAIN = ('struct A { var b: B }\nstruct B { var c: C }\nstruct C { var data: ContiguousArray<Float> }\n'
             'struct Z { var n: Int }\n')

    def test_computed_only_struct_is_not_a_handle(self):
        self.assertNotIn('OnlyComputed', derive_handle_types({'x.swift': self.COMPUTED}))

    def test_mutation_computed_exclusion_bites(self):
        self.assertIn('OnlyComputed', derive_handle_types({'x.swift': self.COMPUTED}, _include_computed=True))

    def test_transitive_fixpoint(self):
        derived = derive_handle_types({'x.swift': self.CHAIN})
        self.assertEqual(derived & {'A', 'B', 'C', 'Z'}, {'A', 'B', 'C'})

    def test_mutation_fixpoint_bites(self):
        derived = derive_handle_types({'x.swift': self.CHAIN}, _fixpoint=False)
        self.assertEqual(derived & {'A', 'B', 'C', 'Z'}, {'C'})

    def test_observed_stored_property_with_initializer_is_stored(self):
        source = 'struct Obs { var items: [Int] = [] { didSet {} } }\n'
        self.assertIn('Obs', derive_handle_types({'x.swift': source}))

    def test_code_view_is_failsafe(self):
        """Interpolation-looking text in a comment or after an escaped backslash is also scanned."""
        handles = {'Vector512Optimized'}
        for source in ('// \\(MemoryLayout<Vector512Optimized>.stride)\n',
                       'let s = "\\\\(MemoryLayout<Vector512Optimized>.stride)"\n'):
            with self.subTest(source):
                self.assertTrue(find_layout_uses(source, handles))


class HandleLayoutPolicy(unittest.TestCase):
    def test_live_tests_match_the_allowlist_exactly(self):
        problems = check_files(_load_tree('Tests'), sources_handle_types(), ALLOWLIST)
        self.assertEqual(problems, [], '\n'.join(problems))

    def test_negative_control_original_instances(self):
        """RED on the pre-fix files: the inventory 6.7 layout expressions are found at their original lines."""
        for path, required in ORIGINAL_INSTANCES.items():
            source = _baseline(path)
            if source is None:
                self.skipTest(f'{BASELINE_COMMIT} not available in this checkout')
            found = {line for line, _, _ in find_layout_uses(source, sources_handle_types())}
            self.assertTrue(required <= found, f'{path}: required {sorted(required)}, found {sorted(found)}')

    def test_must_flag_controls(self):
        for label, (source, _) in MUST_FLAG.items():
            with self.subTest(label):
                self.assertTrue(find_layout_uses(source, sources_handle_types()), label)

    def test_must_pass_controls(self):
        for label, source in MUST_PASS.items():
            with self.subTest(label):
                self.assertEqual(find_layout_uses(source, sources_handle_types()), [], label)

    def test_mutation_each_rule_bites(self):
        """Disabling a rule lets its must-flag controls through, so the controls depend on the rule."""
        handles = sources_handle_types()
        for rule, kwargs in (('ofValue', {'rules': ('type-layout',)}),
                             ('type-layout', {'rules': ('ofValue',)}),
                             ('alias', {'aliases': False}),
                             ('alias-fixpoint', {'alias_fixpoint': False}),
                             ('sink', {'sinks_only': True})):
            for label, (source, needs) in MUST_FLAG.items():
                if needs != rule:
                    continue
                with self.subTest(f'{rule}: {label}'):
                    self.assertTrue(find_layout_uses(source, handles))
                    self.assertEqual(find_layout_uses(source, handles, **kwargs), [])

    def test_mutation_name_list_misses_what_derivation_finds(self):
        """The retired `Optimized|FP16` name list would have missed INT8 and generic Vector."""
        legacy = {n for n in sources_handle_types() if 'Optimized' in n or 'FP16' in n}
        for label in ('INT8 handle', 'generic Vector'):
            source = MUST_FLAG[label][0]
            self.assertTrue(find_layout_uses(source, sources_handle_types()), label)
            self.assertEqual(find_layout_uses(source, legacy), [], label)

    def test_allowlist_semantics(self):
        handles = {'Handle'}
        path, source = 'T.swift', 'func pin() {\n  XCTAssertEqual(MemoryLayout<Handle>.size, 16)\n}\n'
        key = (path, 'pin', 'MemoryLayout<Handle>.size')
        self.assertEqual(len(check_files({path: source}, handles, {})), 1)
        self.assertEqual(check_files({path: source}, handles, {key: ('LAYOUT-PIN: inline size', 1)}), [])
        self.assertTrue(check_files({path: source}, handles, {key: ('LAYOUT-PIN: inline size', 2)}))
        self.assertTrue(check_files({}, handles, {key: ('LAYOUT-PIN: inline size', 1)}))
        self.assertTrue(check_files({path: source}, handles, {key: ('no category', 1)}))

    def test_allowlisted_pin_does_not_hide_a_second_use(self):
        handles = {'Handle'}
        source = 'func pin() {\n  _ = MemoryLayout<Handle>.size\n  _ = MemoryLayout<Handle>.size\n}\n'
        key = ('T.swift', 'pin', 'MemoryLayout<Handle>.size')
        self.assertTrue(check_files({'T.swift': source}, handles, {key: ('LAYOUT-PIN: x', 1)}))


def report(directory):
    handles = sources_handle_types()
    for path, source in _load_tree(directory).items():
        for line, function, expression in find_layout_uses(source, handles):
            print(f'{path}:{line}: {function}: {expression}')


if __name__ == '__main__':
    if '--report' in sys.argv:
        for each in sys.argv[sys.argv.index('--report') + 1:]:
            report(each)
    else:
        unittest.main()
