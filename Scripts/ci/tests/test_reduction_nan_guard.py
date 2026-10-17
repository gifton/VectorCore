"""Policy (T16, reduction NaN class closure): every Float/Double min/max fold in Sources has a reviewed NaN decision.

WHY: Swift's two-argument `max`/`min` keep a NaN only in the FIRST argument (`max(1, .nan) == 1`, `max(.nan, 1)` is
NaN), `Sequence.max()`/`.min()` keep one only at index 0, `SIMD.max()`/`.min()` and `Float.maximum` ignore NaN, and
vDSP `maxv`/`minv` drop it in their scalar tail. So a fold written with them reports a finite extremum for data that
contains NaN, depending on where the NaN sits. T16 phase 1 found 47 such sites in 23 functions; the Swift and vDSP
providers disagreed on about half of 411 fixtures. T16 rules one contract (propagate NaN; first-NaN index; S1/Z1/E1/I1)
and pins it with the registry-driven table Tests/ComprehensiveTests/T16ReductionExceptionalTableTests.swift. This guard
keeps the site registry complete: a fold it discovers must be registered with a reviewed decision, and a registered
fold that no longer exists is stale. It does not evaluate the contract itself; the table does.

DISCOVERY, on Sources/**/*.swift with comments and strings blanked by Scripts/ci/swift_lexing._mask. Each hit is
attributed to its innermost function, computed property, `init`, `subscript` or `deinit` body and to the nearest
enclosing type or extension. Shapes:
  F1 assignment fold    ACC = [Swift.]max|min(ACC, e)  or  ACC = [Swift.]max|min(e, ACC)  (balanced parentheses, so
                        multi-line arguments count; ACC may be a property path or a subscript such as `mins[i]`);
  F2 horizontal         recv.max() / recv.min() / recv.max(by: / recv.min(by: / recv.max { / recv.min {   (Sequence and
                        SIMD; a numeric type receiver such as `Int.max {` is a property, not a call);
  F3 reduce             reduce(x, [Swift.]max|min);
  F5 vDSP               vDSP_maxv / minv / maxmgv / minmgv (any `i`/`D` suffix), vDSP.maximum/minimum[Magnitude](;
  F6 stdlib static      Float|Double|Float16|CGFloat|Scalar|Self|T.maximum|minimum|maximumMagnitude|minimumMagnitude(
                        (NaN-IGNORING statics; a `func` declaration of that name is not a call);
  F7 comparison fold    if <clause with ACC on EITHER side of <|<=|>|>=> { ... ACC = ... }  (argmin/argmax style; the
                        assignment may come anywhere in the block, so `if x < best { at = i; best = x }` and
                        `if best > x { best = x }` both count; multi-line; `&&`/`||`/`,` clauses are split; a clause may be parenthesised or negated:
                        `if (x < best) { best = x }`, `if !(x >= best) { best = x }`). A clause whose
                        other side is a numeric literal (`if x < 0 { x = 0 }`) is a clamp by a constant, not a fold;
  F8 C / SIMD pairwise  fmax / fmin / fmaxf / fminf (NaN-ignoring maxNum/minNum) and pointwiseMax / pointwiseMin
                        (SIMD; a NaN in the second operand is dropped); a `func` declaration of that name is not a call.
A site key is `path::Type.body::shape:operand` (`path::body::...` outside a type); the operand is the accumulator
(F1, F7), the receiver (F2), `reduce` (F3) or the called name (F5, F6, F8). Overloads and repeated shapes count once per
occurrence (a multiset).

TYPE FILTER (P02 ruling RD14: Int clamps and folds are out of scope), F1 and F7 only: the accumulator's base
identifier is looked up as a `var`/`let` declaration in the enclosing body, then in the whole file. An Int-family
annotation, or (without an annotation) an integer-literal / `.count` / `Int(` initializer, is `int` and is NOT a site.
A Float-family annotation or a float initializer is `float`; anything else is `unknown` and counts as a site
(conservative). An unrecognized annotation never falls through to the initializer (`var scale: S = 0` is unknown).

RULES:
  R1 the multiset of discovered site keys equals the multiset of site literals `"Sources/...::..."` between the
     `// BEGIN T16 REDUCTION SITES` and `// END T16 REDUCTION SITES` markers of
     Tests/ComprehensiveTests/T16ReductionRegistry.swift (unregistered and stale keys both fail);
  R2 every entry is `("<key>", .tabled("<row>"))`, `.external("<path>::<Suite>.<test>")` or
     `.exempt("<category>: <reason>")`:
       .tabled    the row id is declared (`row("<row>"` or `asyncRow("<row>"`) in the registry file outside the block;
       .external  the path exists and declares `func <test>(` (T15-owned folds point at T15's tests);
       .exempt    the category is one of finite-guarded, by-construction, int, routed:<task>, and the reason is
                  non-empty;
     an entry that does not parse as one of these is malformed;
  R3 the block is read with `//` and `/* */` comments removed (strings kept), so a commented-out entry is not a
     registration.
There is NO automatic acceptance: an `isNaN`/`isFinite` token in the body does not count as handling (phase 1 showed a
result-only check such as `guard let s = deltas.max(), s.isFinite` passing a token rule while the fold itself drops a
NaN). Every site needs a reviewed entry. `ReductionNaN.propagatingMax(`/`propagatingMin(` calls are not F1 shapes.
The real-tree, fixture and mutation tests use filesystem reads only, so they run in a git-less export too.

LIMITS (lexical heuristic, not a parser or dataflow analysis):
  * A `.tabled` entry asserts that the table exercises the site; whether the handling is correct is the table's job.
  * Type inference is a declaration lookup only (no inference through expressions, generics or closures).
  * Not seen: folds through `reduce(into:)` or other closures (`reduce(0) { max($0, $1) }`), folds through helper
    functions, ternary folds (`m = x > m ? x : m`), element-wise maps (`zip(a, b).map { max($0, $1) }`),
    `while` comparison folds, `let`/`case` pattern conditions, comparison folds whose accumulator update is not a plain `ACC = ...` assignment in the
    `if` block (`ACC += `, a ternary, a `switch`), folds that route the accumulator through a helper or `inout`
    (`update(&best, x)`), `guard`-style folds, `fmax`-like calls spelled through a typealias or a function reference
    (`reduce(0, fmax)` is seen by F3 only for `max`/`min`), `.max { }` on a receiver that is a numeric type name,
    and clamps of a value by constants (`t = max(0, min(1, t))` has no accumulator operand;
    the T15 cosine clamp idioms are T15's guard, test_cosine_registry.py, which this guard does not duplicate).
  * Element-wise `min`/`max`/`clip` bodies are not folds and are not discovered; they are rows of the table (C2).
  * Not scanned: Tests, Benchmarks, C sources (VectorCoreC has no min/max today), macros.
  * `#if` branches are not evaluated (every branch counts).
  * `v.max()` on a Collection is the stdlib method even where `v` is a VectorCore vector with a `max` property.
  * Block comments in the registry do not nest and string interpolation there is not lexed.
"""
import collections
import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location('swift_lexing', ROOT / 'Scripts/ci/swift_lexing.py')
lexing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lexing)

REGISTRY = 'Tests/ComprehensiveTests/T16ReductionRegistry.swift'
BEGIN, END = '// BEGIN T16 REDUCTION SITES', '// END T16 REDUCTION SITES'
EXEMPT_CATEGORIES = re.compile(r'^(?:finite-guarded|by-construction|int|routed:[A-Za-z0-9_-]+)$')

_ACC = r'([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?:\[[^\]\n]*\])?)'
_CALL = re.compile(r'(?<![\w.])(?:Swift\.)?(max|min)\(')
_ASSIGN = re.compile(_ACC + r'\s*=\s*$')
_F2 = re.compile(r'\.(max|min)\s*(?:\(\s*(?:\)|by\s*:)|\{)')
_F8 = re.compile(r'(?<!\w)((?:[A-Za-z_]\w*\.)?(?:fmaxf?|fminf?|pointwiseMax|pointwiseMin))\s*\(')
_IF = re.compile(r'\bif\b')
_LITERAL = re.compile(r'^-?\d[\w.]*$')
_NUMERIC_TYPE = re.compile(r'^(?:U?Int\d*|Float\d*|Double|CGFloat)$')
_CLAUSE = re.compile(r'^(.+?)\s*([<>]=?)\s*(.+)$')
_F3 = re.compile(r'\breduce\s*\([^()]*,\s*(?:Swift\.)?(max|min)\s*\)')
_F5 = re.compile(r'\b(vDSP_(?:max|min)(?:mg)?v(?:i)?D?)\b|\b(vDSP\.(?:maximum|minimum)(?:Magnitude)?)\s*\(')
_F6 = re.compile(r'\b((?:Float|Double|Float16|CGFloat|Scalar|Self|T)\.(?:maximum|minimum|maximumMagnitude|minimumMagnitude))\s*\(')
_INIT_HEADER = re.compile(r'(?<![\w.])(init[?!]?\s*(?:<[^>]*>)?\s*\(|subscript\s*\(|deinit\b)')
_TYPE_NAME = re.compile(r'\b(?:class|struct|enum|extension|actor|protocol)\s+([A-Za-z_][\w.]*)')
_INT_TYPES = re.compile(r'^(?:U?Int(?:8|16|32|64)?|Swift\.U?Int\w*)$')
_FLOAT_TYPES = re.compile(r'^(?:Float|Double|Float16|CGFloat|Scalar|TimeInterval|Float32|Float64)$')
_FLOAT_INIT = re.compile(r'\d\.\d|\.infinity|\.nan\b|greatestFiniteMagnitude|leastNormalMagnitude|\bFloat\(|\bDouble\('
                         r'|\[Float\]|\[Double\]')
_INT_INIT = re.compile(r'^-?\d+$|\.count\b|\bInt\(|\[Int\]')
_SITE_LITERAL = re.compile(r'"(Sources/[^"\s]+\.swift::[^"\s]+)"')
_ENTRY = re.compile(r'\(\s*"(Sources/[^"\s]+\.swift::[^"\s]+)"\s*,\s*\.(tabled|external|exempt)\(\s*"([^"]*)"\s*\)\s*\)')
_ROW_DECL = re.compile(r'\b(?:row|asyncRow)\(\s*"([^"]+)"')


def _matching(text, open_at):
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] in '([':
            depth += 1
        elif text[i] in ')]':
            depth -= 1
            if depth == 0:
                return i
    return len(text)


def _split_top(text):
    parts, depth, current = [], 0, []
    for ch in text:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    parts.append(''.join(current))
    return [p.strip() for p in parts]


def _base_name(acc):
    return re.split(r'[.\[]', acc)[0]


def _decl_type(name, *texts):
    """'int', 'float' or 'unknown' for the first `var`/`let` declaration of `name` in `texts` (in order)."""
    pattern = re.compile(r'\b(?:var|let)\s+(?:[^=\n]*?,\s*)?' + re.escape(name)
                         + r'\b\s*(?::\s*([^=\n{,]+?))?\s*(?:=\s*([^\n]+))?$', re.M)
    for text in texts:
        for match in pattern.finditer(text):
            annotation, initializer = (match.group(1) or '').strip(), (match.group(2) or '').strip()
            if annotation:
                head = re.sub(r'[?!]$', '', annotation).strip()
                if _INT_TYPES.match(head):
                    return 'int'
                if _FLOAT_TYPES.match(head) or head in ('[Float]', '[Double]', 'ContiguousArray<Float>'):
                    return 'float'
                return 'unknown'
            first = re.split(r',\s*[A-Za-z_]\w*\s*=', initializer)[0]
            if _FLOAT_INIT.search(first):
                return 'float'
            if _INT_INIT.search(first.strip()):
                return 'int'
            if initializer:
                return 'unknown'
    return 'unknown'


class _File:
    """Lexed view of one source file: body and type attribution by offset."""

    def __init__(self, source):
        self.source = source
        self.mask = lexing._mask(source)
        self.scopes = lexing._Scopes(source)
        self.line_starts = [0] + [i + 1 for i, ch in enumerate(source) if ch == '\n']

    def line_of(self, offset):
        lo, hi = 0, len(self.line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.line_starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    def _header(self, scope):
        open_index = self.scopes.opens[scope]
        previous = max(self.mask.rfind(ch, 0, open_index) for ch in '{};')
        return lexing._header_chunk(self.mask[previous + 1:open_index])

    def _special_name(self, scope):
        if self.scopes.transparent[scope]:
            return None
        found = _INIT_HEADER.search(self._header(scope))
        if not found:
            return None
        return 'init' if found.group(1).startswith('init') else found.group(1).split('(')[0].strip()

    def body(self, offset):
        """(name, start, end, owning scope) of the innermost named body around `offset`."""
        scope = self.scopes.at[offset]
        while scope and not (self.scopes.name[scope] or self._special_name(scope)):
            scope = self.scopes.parent[scope]
        if not scope:
            return '<file>', 0, len(self.source), 0
        name = self.scopes.name[scope] or self._special_name(scope)
        return name, self.scopes.opens[scope] + 1, self.scopes.closes[scope], scope

    def type_name(self, scope):
        walk = self.scopes.parent[scope] if scope else None
        while walk:
            if self.scopes.transparent[walk]:
                found = _TYPE_NAME.search(self._header(walk))
                if found:
                    return found.group(1)
            walk = self.scopes.parent[walk]
        return None


def _unwrap(clause):
    """Clause text without leading `!` and balanced outer parentheses (`(x < best)`, `!(x >= best)`)."""
    text = clause.strip()
    while True:
        if text.startswith('!'):
            text = text[1:].strip()
        elif text.startswith('(') and _matching(text, 0) == len(text) - 1:
            text = text[1:-1].strip()
        else:
            return text


def _comparison_fold(mask, start):
    """Accumulator (compact) of the `if` whose condition starts at `start`, or None. The condition has a top-level
    `<`/`<=`/`>`/`>=` clause with the accumulator on EITHER side, and the `{ }` block assigns it anywhere
    (`if x < best { at = i; best = x }`, `if best > x { best = x }`, multi-line)."""
    depth, brace = 0, None
    for i in range(start, len(mask)):
        ch = mask[i]
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth -= 1
        elif ch == '{' and depth == 0:
            brace = i
            break
        elif ch == ';' or ch == '}':
            return None
    if brace is None:
        return None
    close = brace
    level = 0
    for i in range(brace, len(mask)):
        if mask[i] == '{':
            level += 1
        elif mask[i] == '}':
            level -= 1
            if level == 0:
                close = i
                break
    else:
        return None
    block = mask[brace + 1:close]
    condition = re.sub(r'\s+', ' ', mask[start:brace]).strip()
    for clause in re.split(r'&&|\|\||,', condition):
        found = _CLAUSE.match(_unwrap(clause))
        if not found:
            continue
        for side, other in ((found.group(3), found.group(1)), (found.group(1), found.group(3))):
            operand = side.strip()
            if not re.fullmatch(_ACC, operand) or _LITERAL.match(other.strip()):
                continue  # a comparison with a numeric literal is a clamp by a constant, not a fold over data
            compact = operand.replace(' ', '')
            if re.search(r'(?<![\w.\]])' + re.escape(compact).replace(r'\ ', r'\s*') + r'\s*=(?!=)', block):
                return compact
    return None


def discover(path, source):
    """List of (key, line, type) for one file; `type` is int/float/unknown/any. Int rows are returned too (the caller
    filters them) so the RD14 filter is observable."""
    view = _File(source)
    mask = view.mask
    found = []

    def record(offset, shape, operand, typ='any'):
        name, _, _, scope = view.body(offset)
        owner = view.type_name(scope)
        qualified = f'{owner}.{name}' if owner else name
        found.append((f'{path}::{qualified}::{shape}:{operand}', view.line_of(offset), typ))

    lines = mask.split('\n')
    for number, line in enumerate(lines, 1):
        offset = view.line_starts[number - 1]
        for call in _CALL.finditer(line):
            assigned = _ASSIGN.search(line[:call.start()])
            if not assigned:
                continue
            accumulator = assigned.group(1)
            open_at = offset + call.end() - 1
            args = _split_top(mask[open_at + 1:_matching(mask, open_at)])
            if len(args) != 2:
                continue
            compact = accumulator.replace(' ', '')
            if compact not in (args[0].replace(' ', ''), args[1].replace(' ', '')):
                continue
            _, start, end, _ = view.body(offset + call.start())
            typ = _decl_type(_base_name(accumulator), mask[start:end], mask)
            record(offset + call.start(), 'F1', compact, typ)
        for match in _F2.finditer(line):
            before = line[:match.start()]
            if before.strip() and not re.search(r'[\w\])]$', before):
                continue  # `x == .max {`: an implicit member, not a method call (a leading-dot chain line is a call)
            receiver = re.search(r'([A-Za-z_][\w.]*(?:\[[^\]]*\])?)\s*$', before)
            if receiver and _NUMERIC_TYPE.match(receiver.group(1)):
                continue  # `Int.max { ... }` is a type property followed by a block, not a Sequence method
            record(offset + match.start(), 'F2', receiver.group(1) if receiver else '?')
        for match in _F3.finditer(line):
            record(offset + match.start(), 'F3', 'reduce')
        for match in _F5.finditer(line):
            record(offset + match.start(), 'F5', match.group(1) or match.group(2))
        for match in _F8.finditer(line):
            if re.search(r'\bfunc\s*$', line[:match.start()]):
                continue
            record(offset + match.start(), 'F8', match.group(1).split('.')[-1])
        for match in _F6.finditer(line):
            if re.search(r'\bfunc\s*$', line[:match.start()]):
                continue
            record(offset + match.start(), 'F6', match.group(1))
    for match in _IF.finditer(mask):
        accumulator = _comparison_fold(mask, match.end())
        if accumulator:
            _, start, end, _ = view.body(match.start())
            record(match.start(), 'F7', accumulator, _decl_type(_base_name(accumulator), mask[start:end], mask))
    return found


def discover_tree(files):
    """files: {relative path: source}. Returns (Counter of site keys, list of excluded int keys)."""
    sites, excluded = collections.Counter(), []
    for path in sorted(files):
        for key, _line, typ in discover(path, files[path]):
            if typ == 'int':
                excluded.append(key)
            else:
                sites[key] += 1
    return sites, excluded


def strip_comments(text):
    """Same-length copy of `text` with `//` and `/* */` comments blanked (newlines kept); string literals are kept."""
    out, n, i = list(text), len(text), 0

    def blank(a, b):
        for k in range(a, min(b, n)):
            if out[k] != '\n':
                out[k] = ' '

    while i < n:
        if text.startswith('"""', i):
            j = text.find('"""', i + 3)
            i = n if j < 0 else j + 3
        elif text[i] == '"':
            k = i + 1
            while k < n and text[k] not in '"\n':
                k += 2 if text[k] == '\\' else 1
            i = k + 1
        elif text.startswith('//', i):
            j = text.find('\n', i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2)
            j = n if j < 0 else j + 2
            blank(i, j)
            i = j
        else:
            i += 1
    return ''.join(out)


def split_registry(text):
    """(block, outside): the BEGIN/END block and the rest of the file, both with comments removed."""
    start, end = text.find(BEGIN), text.find(END)
    if start < 0 or end < start:
        raise ValueError('registry markers missing or out of order')
    clean = strip_comments(text)
    return clean[start:end], clean[:start] + clean[end:]


def registered(text):
    block, _ = split_registry(text)
    return collections.Counter(_SITE_LITERAL.findall(block))


def registry_problems(text, exists):
    """R2 problems. `exists(path, test)` reports whether `path` exists and declares `func test(`."""
    block, outside = split_registry(text)
    rows = set(_ROW_DECL.findall(outside))
    problems = []
    entries = _ENTRY.findall(block)
    literals = _SITE_LITERAL.findall(block)
    parsed = collections.Counter(site for site, _, _ in entries)
    for site, count in sorted((collections.Counter(literals) - parsed).items()):
        problems.append(f'malformed entry (expected .tabled/.external/.exempt with one string): {site} (x{count})')
    for site, kind, value in entries:
        if kind == 'tabled':
            if value not in rows:
                problems.append(f'tabled row is not declared in the registry: {site} -> {value}')
        elif kind == 'external':
            if '::' not in value or '.' not in value.split('::', 1)[1]:
                problems.append(f'external reference is not path::Suite.test: {site} -> {value}')
                continue
            path, symbol = value.split('::', 1)
            if not exists(path, symbol.rsplit('.', 1)[-1]):
                problems.append(f'external reference does not resolve: {site} -> {value}')
        else:
            category, _, reason = value.partition(':')
            if category == 'routed':
                task, _, reason = reason.partition(':')
                category = f'routed:{task.strip()}'
            if not EXEMPT_CATEGORIES.match(category.strip()) or not reason.strip():
                problems.append(f'exemption needs "<category>: <reason>" with a known category: {site} -> {value!r}')
    return problems


def check(files, registry_text, exists):
    """All rule violations as strings (empty when the tree and the registry agree)."""
    sites, _ = discover_tree(files)
    listed = registered(registry_text)
    problems = [f'R1 unregistered reduction site: {site} (x{count})' for site, count in sorted((sites - listed).items())]
    problems += [f'R1 stale registry entry: {site} (x{count})' for site, count in sorted((listed - sites).items())]
    problems += ['R2 ' + p for p in registry_problems(registry_text, exists)]
    return problems


def real_files():
    return {path.relative_to(ROOT).as_posix(): path.read_text(encoding='utf-8')
            for path in sorted((ROOT / 'Sources').rglob('*.swift'))}


def real_exists(path, test):
    target = ROOT / path
    return target.is_file() and re.search(r'\bfunc\s+' + re.escape(test) + r'\s*\(', target.read_text(encoding='utf-8')) is not None


def registry_text():
    return (ROOT / REGISTRY).read_text(encoding='utf-8')


def _registry(*entries, rows=('row.x',)):
    """A minimal registry file: `entries` are (key, coverage source) pairs; `rows` are declared row ids."""
    body = ''.join(f'        ("{key}", {coverage}),\n' for key, coverage in entries)
    declared = ''.join(f'    static let r{i} = row("{row}", .reduction)\n' for i, row in enumerate(rows))
    return (f'enum R {{\n{declared}    static let sites = [\n        {BEGIN}\n{body}        {END}\n    ]\n}}\n')


def _sites(source, path='Sources/A.swift'):
    return discover_tree({path: source})[0]


# Must-FLAG fixtures: (label, source, expected keys).
MUST_FLAG = [
    ('F1 accumulator first', 'enum P {\n static func f(_ a: [Float]) -> Float {\n  var m: Float = 0\n'
     '  for x in a { m = max(m, x) }\n  return m\n }\n}\n', ['Sources/A.swift::P.f::F1:m']),
    ('F1 accumulator second, Swift-qualified', 'func f(_ a: [Float]) -> Float {\n var m = -Float.infinity\n'
     ' for x in a { m = Swift.min(x, m) }\n return m\n}\n', ['Sources/A.swift::f::F1:m']),
    ('F1 property accumulator', 'actor P {\n var s = S()\n func f(_ x: Float) {\n  s.loss = max(s.loss, x)\n }\n}\n',
     ['Sources/A.swift::P.f::F1:s.loss']),
    ('F1 subscript accumulator', 'func f(_ r: [[Float]]) {\n var m = [Float](repeating: 0, count: 2)\n'
     ' for v in r { for i in 0..<2 { m[i] = min(m[i], v[i]) } }\n}\n', ['Sources/A.swift::f::F1:m[i]']),
    ('F1 multi-line arguments', 'func f(_ a: [Float]) -> Float {\n var m: Float = 0\n for x in a {\n'
     '  m = max(\n   m,\n   abs(x)\n  )\n }\n return m\n}\n', ['Sources/A.swift::f::F1:m']),
    ('F2 Sequence', 'func f(_ errors: [Float]) -> Float { errors.max() ?? 0 }\n', ['Sources/A.swift::f::F2:errors']),
    ('F2 SIMD and by:', 'func f(_ v: SIMD8<Float>, _ a: [Float]) -> Float {\n let x = v.min()\n'
     ' return a.max(by: <)! + x\n}\n', ['Sources/A.swift::f::F2:v', 'Sources/A.swift::f::F2:a']),
    ('F3 reduce', 'func f(_ a: [Float]) -> Float { a.reduce(-.infinity, max) }\n', ['Sources/A.swift::f::F3:reduce']),
    ('F5 vDSP', 'func f(_ p: UnsafePointer<Float>, _ n: Int) -> Float {\n var r: Float = 0\n'
     ' vDSP_maxv(p, 1, &r, vDSP_Length(n))\n return r\n}\n', ['Sources/A.swift::f::F5:vDSP_maxv']),
    ('F5 vDSP Double magnitude', 'func f(_ p: UnsafePointer<Double>, _ n: Int) -> Double {\n var r = 0.0\n'
     ' vDSP_maxmgvD(p, 1, &r, vDSP_Length(n))\n return r\n}\n', ['Sources/A.swift::f::F5:vDSP_maxmgvD']),
    ('F6 NaN-ignoring static', 'func f(_ a: Float, _ b: Float) -> Float { Float.maximum(a, b) }\n',
     ['Sources/A.swift::f::F6:Float.maximum']),
    ('F7 multi-line comparison fold', 'func f(_ a: [Float]) -> Int {\n var best = a[0], at = 0\n'
     ' for i in a.indices {\n  if a[i] <\n     best { best = a[i]; at = i }\n }\n return at\n}\n',
     ['Sources/A.swift::f::F7:best']),
    ('fold inside an init', 'struct Q {\n let hi: Float\n init(_ a: [Float]) {\n  var m: Float = 0\n'
     '  for x in a { m = max(m, x) }\n  hi = m\n }\n}\n', ['Sources/A.swift::Q.init::F1:m']),
    ('F7 accumulator on the left', 'func f(_ a: [Float]) -> Float {\n var best = a[0]\n for x in a {\n'
     '  if best > x { best = x }\n }\n return best\n}\n', ['Sources/A.swift::f::F7:best']),
    ('F7 index assigned before the accumulator', 'func f(_ a: [Float]) -> Int {\n var best = a[0]\n var at = 0\n'
     ' for i in a.indices {\n  if a[i] < best {\n   at = i\n   best = a[i]\n  }\n }\n return at\n}\n',
     ['Sources/A.swift::f::F7:best']),
    ('F7 one-line index first, accumulator on the left', 'func f(_ a: [Float]) -> Int {\n var best = a[0], at = 0\n'
     ' for i in a.indices { if best >= a[i] { at = i; best = a[i] } }\n return at\n}\n',
     ['Sources/A.swift::f::F7:best']),
    ('F2 trailing-closure max', 'func f(_ a: [Float]) -> Float { a.max { $0 < $1 } ?? 0 }\n',
     ['Sources/A.swift::f::F2:a']),
    ('F2 trailing-closure min on a subscript receiver', 'func f(_ r: [[Float]]) -> Float? {\n return r[0].min { $0 < $1 }\n}\n',
     ['Sources/A.swift::f::F2:r[0]']),
    ('F2 leading-dot chain line .max()', 'func f(_ a: [Float]) -> Float {\n return a.map { $0 * 2 }\n   .max() ?? 0\n}\n',
     ['Sources/A.swift::f::F2:?']),
    ('F2 leading-dot chain line .max { }', 'func f(_ a: [Float]) -> Float {\n return a.map { $0 * 2 }\n   .max { $0 < $1 } ?? 0\n}\n',
     ['Sources/A.swift::f::F2:?']),
    ('F2 leading-dot chain line .min(by:)', 'func f(_ a: [Float]) -> Float {\n return a.map { $0 * 2 }\n   .min(by: <) ?? 0\n}\n',
     ['Sources/A.swift::f::F2:?']),
    ('F7 parenthesised condition', 'func f(_ a: [Float]) -> Float {\n var best = a[0]\n for x in a {\n'
     '  if (x < best) { best = x }\n }\n return best\n}\n', ['Sources/A.swift::f::F7:best']),
    ('F7 negated condition (drops NaN)', 'func f(_ a: [Float]) -> Float {\n var best = a[0]\n for x in a {\n'
     '  if !(x >= best) { best = x }\n }\n return best\n}\n', ['Sources/A.swift::f::F7:best']),
    ('F8 fmax', 'func f(_ a: [Double]) -> Double {\n var m = 0.0\n for x in a { m = fmax(m, x) }\n return m\n}\n',
     ['Sources/A.swift::f::F8:fmax']),
    ('F8 fminf, fmaxf and fmin', 'func f(_ a: Float, _ b: Float) -> Float {\n return fminf(a, b) + fmaxf(a, b) + Float(fmin(1, 2))\n}\n',
     ['Sources/A.swift::f::F8:fminf', 'Sources/A.swift::f::F8:fmaxf', 'Sources/A.swift::f::F8:fmin']),
    ('F8 pointwiseMax and pointwiseMin', 'func f(_ rows: [SIMD4<Float>]) -> SIMD4<Float> {\n var m = rows[0]\n'
     ' for v in rows { m = pointwiseMax(m, v); m = simd.pointwiseMin(m, v) }\n return m\n}\n',
     ['Sources/A.swift::f::F8:pointwiseMax', 'Sources/A.swift::f::F8:pointwiseMin']),
    ('result-only isFinite check is not handling', 'func f(_ d: [Double]) -> Double? {\n'
     ' guard let s = d.max(), s.isFinite else { return nil }\n return s\n}\n', ['Sources/A.swift::f::F2:d']),
    ('isNaN token in the body is not handling', 'func f(_ a: [Float]) -> Float {\n var m: Float = 0\n'
     ' for x in a { if x.isNaN { continue }; m = max(m, x) }\n return m\n}\n', ['Sources/A.swift::f::F1:m']),
    ('unknown accumulator type counts (conservative)', 'func f<S: FloatingPoint>(_ a: [S]) -> S {\n var scale: S = 0\n'
     ' for x in a { scale = max(scale, x) }\n return scale\n}\n', ['Sources/A.swift::f::F1:scale']),
]

# Must-PASS fixtures: no site.
MUST_PASS = [
    ('Int fold, literal initializer', 'func f(_ a: [Int]) -> Int {\n var peak = 0\n for x in a { peak = max(peak, x) }\n'
     ' return peak\n}\n'),
    ('Int8 fold, annotation', 'func f(_ a: [Int8]) -> Int8 {\n var m: Int8 = .max\n for x in a { m = min(m, x) }\n'
     ' return m\n}\n'),
    ('Int clamps', 'func f(_ k: Int, _ count: Int, _ s: Int) -> Int {\n let a = min(k, count)\n'
     ' let end = min(s + 8, count)\n return a + end\n}\n'),
    ('T15 cosine clamp idiom', 'func f(_ s: Float) -> Float { 1 - max(-1, min(1, s)) }\n'),
    ('declarations', 'enum P {\n static func maximum(_ a: UnsafePointer<Float>, count: Int) -> Float { 0 }\n'
     ' func max(_ a: [Float]) -> Float { 0 }\n}\n'),
    ('comments and strings', '// m = max(m, x); a.max(); vDSP_maxv(p, 1, &r, n)\nlet s = "m = max(m, x) a.max()"\n'),
    ('ReductionNaN helper', 'func f(_ a: [Float]) -> Float {\n var m: Float = 0\n'
     ' for x in a { m = ReductionNaN.propagatingMax(m, x) }\n return m\n}\n'),
    ('two-operand element-wise', 'func f(_ a: [Float], _ b: [Float], _ r: inout [Float]) {\n'
     ' for i in a.indices { r[i] = Swift.min(a[i], b[i]) }\n}\n'),
    ('implicit member before a block is not a call', 'func f(_ v: Int) -> Int {\n if v == .max { return 0 }\n'
     ' if v != .min { return 1 }\n return v\n}\n'),
    ('clamp by a constant is not a fold', 'func f(_ x: Float) -> Float {\n var s = x\n if s < 0 { s = 0 }\n'
     ' if s > 1 { s = 1 }\n return s\n}\n'),
    ('numeric type property before a block', 'func f(_ v: Int) -> Int {\n if v < Int.max { return v + 1 }\n'
     ' if v == .max { return 0 }\n return v\n}\n'),
    ('comparison without an accumulator update', 'func f(_ a: [Float]) -> Int {\n var n = 0\n for x in a {\n'
     '  if x > a[0] { n += 1 }\n }\n return n\n}\n'),
    ('if block that assigns a different variable', 'func f(_ a: [Float], _ b: Float) -> Float {\n var out = b\n'
     ' for x in a { if x > b { out = x } }\n return out\n}\n'),
    ('declarations of fmax-like names', 'extension SIMD4<Float> {\n func pointwiseMax(_ a: Self) -> Self { self }\n}\n'),
    ('parameter clamp by constants', 'func f(_ t: Float) -> Float {\n let c = Swift.max(0, Swift.min(1, t))\n return c\n}\n'),
]


class FixtureControls(unittest.TestCase):
    """The guard bites on every shape and passes the controls."""

    def never(self, path, test):
        return False

    def test_must_flag_fixtures_are_discovered(self):
        for label, source, expected in MUST_FLAG:
            with self.subTest(label):
                self.assertEqual(_sites(source), collections.Counter(expected), label)

    def test_must_pass_fixtures_have_no_site(self):
        for label, source in MUST_PASS:
            with self.subTest(label):
                self.assertEqual(_sites(source), collections.Counter(), label)

    def test_int_fold_is_excluded_but_observable(self):
        source = MUST_PASS[0][1]
        sites, excluded = discover_tree({'Sources/A.swift': source})
        self.assertEqual((sites, excluded), (collections.Counter(), ['Sources/A.swift::f::F1:peak']))

    def test_enclosing_type_separates_same_named_bodies(self):
        source = ('struct A {\n static func maximum(_ p: UnsafePointer<Float>) -> Float {\n var r: Float = 0\n'
                  ' vDSP_maxv(p, 1, &r, 1)\n return r\n }\n}\n'
                  'struct B {\n static func maximum(_ p: UnsafePointer<Double>) -> Double {\n var r = 0.0\n'
                  ' vDSP_maxvD(p, 1, &r, 1)\n return r\n }\n}\n')
        self.assertEqual(_sites(source), collections.Counter({'Sources/A.swift::A.maximum::F5:vDSP_maxv': 1,
                                                              'Sources/A.swift::B.maximum::F5:vDSP_maxvD': 1}))

    def test_registered_site_passes(self):
        source = MUST_FLAG[0][1]
        text = _registry(('Sources/A.swift::P.f::F1:m', '.tabled("row.x")'))
        self.assertEqual(check({'Sources/A.swift': source}, text, self.never), [])

    def test_unregistered_and_stale_entries_fail(self):
        source = MUST_FLAG[0][1]
        problems = check({'Sources/A.swift': source}, _registry(('Sources/A.swift::P.gone::F1:m', '.tabled("row.x")')),
                         self.never)
        self.assertEqual(problems, ['R1 unregistered reduction site: Sources/A.swift::P.f::F1:m (x1)',
                                    'R1 stale registry entry: Sources/A.swift::P.gone::F1:m (x1)'])

    def test_multiplicity_counts(self):
        source = 'func f(_ a: [Float]) -> Float {\n var m: Float = 0\n m = max(m, a[0])\n m = max(m, a[1])\n return m\n}\n'
        once = check({'Sources/A.swift': source}, _registry(('Sources/A.swift::f::F1:m', '.tabled("row.x")')), self.never)
        self.assertEqual(once, ['R1 unregistered reduction site: Sources/A.swift::f::F1:m (x1)'])

    def test_commented_entry_is_not_a_registration(self):
        source = MUST_FLAG[0][1]
        for commented in ('        // ("Sources/A.swift::P.f::F1:m", .tabled("row.x")),\n',
                          '        /* ("Sources/A.swift::P.f::F1:m", .tabled("row.x")), */\n'):
            text = _registry().replace(f'        {END}', commented + f'        {END}')
            self.assertEqual(check({'Sources/A.swift': source}, text, self.never),
                             ['R1 unregistered reduction site: Sources/A.swift::P.f::F1:m (x1)'], commented)

    def test_comment_markers_inside_strings_are_kept(self):
        source = MUST_FLAG[0][1]
        text = _registry(('Sources/A.swift::P.f::F1:m', '.exempt("finite-guarded: see the // note and /* this */")'))
        self.assertEqual(check({'Sources/A.swift': source}, text, self.never), [])

    def test_exemption_needs_category_and_reason(self):
        source = MUST_FLAG[0][1]
        for bad in ('.exempt("")', '.exempt("finite-guarded:")', '.exempt("because: it is fine")',
                    '.exempt("routed:: reason")', '.exempt("no category here")'):
            problems = check({'Sources/A.swift': source}, _registry(('Sources/A.swift::P.f::F1:m', bad)), self.never)
            self.assertEqual(len(problems), 1, bad)
            self.assertTrue(problems[0].startswith('R2 exemption needs'), problems)
        for good in ('.exempt("int: Int8 fold")', '.exempt("by-construction: seeded data")',
                     '.exempt("routed:P02: Int tile clamp")'):
            self.assertEqual(check({'Sources/A.swift': source}, _registry(('Sources/A.swift::P.f::F1:m', good)),
                                   self.never), [], good)

    def test_tabled_row_must_be_declared(self):
        source = MUST_FLAG[0][1]
        problems = check({'Sources/A.swift': source}, _registry(('Sources/A.swift::P.f::F1:m', '.tabled("missing")')),
                         self.never)
        self.assertEqual(problems, ['R2 tabled row is not declared in the registry: Sources/A.swift::P.f::F1:m -> missing'])

    def test_external_must_resolve(self):
        source = MUST_FLAG[0][1]
        text = _registry(('Sources/A.swift::P.f::F1:m', '.external("Tests/X.swift::S.missing")'))
        self.assertEqual(check({'Sources/A.swift': source}, text, self.never),
                         ['R2 external reference does not resolve: Sources/A.swift::P.f::F1:m -> Tests/X.swift::S.missing'])
        self.assertEqual(check({'Sources/A.swift': source}, text, lambda p, t: (p, t) == ('Tests/X.swift', 'missing')), [])

    def test_malformed_entry_is_flagged(self):
        source = MUST_FLAG[0][1]
        problems = check({'Sources/A.swift': source}, _registry(('Sources/A.swift::P.f::F1:m', '.table(["x"])')),
                         self.never)
        self.assertEqual(problems, ['R2 malformed entry (expected .tabled/.external/.exempt with one string): '
                                    'Sources/A.swift::P.f::F1:m (x1)'])


class TreeMutations(unittest.TestCase):
    """Mutation controls on an in-memory copy of the real tree."""

    def test_added_unregistered_fold_fails(self):
        files = real_files()
        path = 'Sources/VectorCore/Operations/Operations.swift'
        files[path] += '\nfunc t16Mutant(_ a: [Float]) -> Float {\n    var worst: Float = 0\n' \
                       '    for x in a { worst = max(worst, x) }\n    return worst\n}\n'
        problems = check(files, registry_text(), real_exists)
        self.assertEqual(problems, [f'R1 unregistered reduction site: {path}::t16Mutant::F1:worst (x1)'])

    def test_each_widened_shape_added_to_the_real_tree_is_flagged(self):
        """Mutation control for F2 (closure), F7 (both operand orders, index first) and F8: one added fold each."""
        path = 'Sources/VectorCore/Operations/Operations.swift'
        added = {
            'F2:errs': 'func t16M1(_ errs: [Float]) -> Float { errs.max { $0 < $1 } ?? 0 }\n',
            'F7:best': 'func t16M2(_ a: [Float]) -> Int {\n var best = a[0]\n var at = 0\n for i in a.indices {\n'
                       '  if a[i] < best { at = i; best = a[i] }\n }\n return at\n}\n',
            'F8:fmax': 'func t16M3(_ a: [Double]) -> Double {\n var m = 0.0\n for x in a { m = fmax(m, x) }\n return m\n}\n',
            'F8:pointwiseMin': 'func t16M4(_ a: [SIMD4<Float>]) -> SIMD4<Float> {\n var m = a[0]\n'
                               ' for v in a { m = pointwiseMin(m, v) }\n return m\n}\n',
        }
        for operand, snippet in added.items():
            with self.subTest(operand):
                files = real_files()
                files[path] += '\n' + snippet
                name = re.search(r'func (t16M\d)', snippet).group(1)
                self.assertEqual(check(files, registry_text(), real_exists),
                                 [f'R1 unregistered reduction site: {path}::{name}::{operand} (x1)'])

    def test_removed_fold_makes_its_entry_stale(self):
        files = real_files()
        sites, _ = discover_tree(files)
        key = sorted(k for k in sites if '::F5:' in k)[0]
        path = key.split('::', 1)[0]
        callee = key.rsplit(':', 1)[1]
        source = files[path]
        start = source.index(callee + '(')
        files[path] = source[:start] + 't16Removed(' + source[start + len(callee) + 1:]
        problems = check(files, registry_text(), real_exists)
        self.assertIn(f'R1 stale registry entry: {key} (x1)', problems)

    def test_unregistering_a_real_entry_is_flagged(self):
        text = registry_text()
        block, _ = split_registry(text)
        first = _ENTRY.search(block)
        line_start = text.rfind('\n', 0, text.index(f'"{first.group(1)}"')) + 1
        line_end = text.index('\n', line_start) + 1
        mutated = text[:line_start] + text[line_end:]
        self.assertIn(f'R1 unregistered reduction site: {first.group(1)} (x1)', check(real_files(), mutated, real_exists))


class RealTree(unittest.TestCase):
    maxDiff = None

    def test_registry_matches_every_discovered_reduction_site(self):
        self.assertEqual(check(real_files(), registry_text(), real_exists), [])

    def test_discovery_sees_the_provider_bodies(self):
        """Sanity: discovery is not vacuous (the phase 1 anchor bodies are found)."""
        sites, excluded = discover_tree(real_files())
        for anchor in ('Sources/VectorCore/Platform/SwiftFloatSIMDProvider.swift::SwiftFloatSIMDProvider.maximum::F1:result',
                       'Sources/VectorCore/Platform/SwiftDoubleSIMDProvider.swift::SwiftDoubleSIMDProvider.minimum::F2:va',
                       'Sources/VectorCore/Platform/AccelerateSIMDProvider.swift::AccelerateFloatProvider.maximum::F5:vDSP_maxv',
                       'Sources/VectorCore/Platform/AccelerateSIMDProvider.swift::AccelerateDoubleProvider.maximumMagnitude::F5:vDSP_maxmgvD',
                       'Sources/VectorCore/Protocols/SIMDStorage.swift::SIMDStorage.minMax::F2:simd',
                       'Sources/VectorCore/Protocols/VectorProtocol.swift::VectorProtocol.lInfinityNorm::F1:maxValue'):
            self.assertIn(anchor, sites)
        self.assertGreater(sum(sites.values()), 40)
        # RD14: the Int8 minMax folds, the buffer pool's peak, the tuning `var N_star: Int` (two comparisons) and the
        # triangular iterator's `private var col = 1` are typed int and excluded.
        self.assertEqual(sorted(excluded), [
            'Sources/VectorCore/Protocols/SIMDStorage.swift::SIMDStorage.minMax::F1:globalMax',
            'Sources/VectorCore/Protocols/SIMDStorage.swift::SIMDStorage.minMax::F1:globalMin',
            'Sources/VectorCore/Providers/AutoTuning.swift::AutoTuning.calculateTuning::F7:N_star',
            'Sources/VectorCore/Providers/AutoTuning.swift::AutoTuning.calculateTuning::F7:N_star',
            'Sources/VectorCore/Providers/SwiftBufferPool.swift::SwiftBufferPool.acquire::F1:peak',
            'Sources/VectorCore/Utilities/TriangularMatrix.swift::Iterator.next::F7:col'])

if __name__ == '__main__':
    unittest.main()
