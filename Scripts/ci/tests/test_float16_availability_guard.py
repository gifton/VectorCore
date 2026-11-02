"""Policy: every `Float16` token must be lexically inside an `#if` branch that excludes Intel macOS.

WHY: the Intel-macOS and Mac Catalyst x86_64 SDKs mark `Float16` unavailable. CI's `platforms` job
compiles the library and tests for x86_64-apple-macosx14.0 and for Mac Catalyst, and unguarded
Float16 uses were introduced at least four times by local milestones that never replayed that job
(f4d5b49, 84f42d0, 1c55a85, 101b374). This test closes the class mechanically.

RULE: a `Float16` token (also `_Float16`, `CFloat16`, `simd_half*`; outside comments and string-literal text,
but INSIDE string interpolations, which are code) in Sources/, Tests/ or Benchmarks/ *.swift is a finding unless the conjunction of the conditions of
the enclosing `#if` / `#elseif` / `#else` branches is UNSATISFIABLE in both environments:
  A. x86_64 macOS     : arch(x86_64)=true, os(macOS)=os(OSX)=true,  os(iOS)=false, targetEnvironment(macCatalyst)=false
  B. x86_64 Catalyst  : arch(x86_64)=true, os(macOS)=os(OSX)=false, os(iOS)=true,  targetEnvironment(macCatalyst)=true
Every other arch(...)/os(...) is false; whitespace inside an atom is ignored (`arch( x86_64 )`); every other atom (canImport, compiler, swift, custom flags such
as NATIVE_FLOAT16_SUPPORTED, targetEnvironment(simulator), ...) is a FREE variable that the checker
may set adversarially, so a guard only counts if it excludes both environments for every value of
the free atoms. An `#else` branch is the negation of all earlier conditions of its chain; `#elseif`
likewise. Nested `#if`s AND their effective conditions.

ACCEPTED (examples; anything equivalent under the evaluation above):
  * the library form      `!(arch(x86_64) && (os(macOS) || targetEnvironment(macCatalyst)))`
  * a compiler-gated form `compiler(>=5.3) && (arch(arm64) || arch(x86_64)) && !(...library form...)`
  * a custom-flag form    `NATIVE_FLOAT16_SUPPORTED && !(...library form...)`
  * `arch(arm64)` alone   (stricter than needed: it also excludes x86_64 simulators, where Float16
    exists; accepted because it is never unsafe, only narrower. DECISION: accepted.)
  * `!arch(x86_64)`, and the `#else` of `#if arch(x86_64) && (os(macOS) || targetEnvironment(macCatalyst))`.
REJECTED: no `#if`; `#if arch(x86_64)`; `#if os(iOS)` / `#if !os(macOS)` / `#if !targetEnvironment(macCatalyst)`
  (each alone leaves one of the two environments); `#if canImport(X)`; Float16 in the `#else` of the
  guard; Float16 used under a runtime constant (`if nativeFloat16Supported { Float16(...) }`), because
  a runtime branch is still compiled on Intel.

LIMITS (lexical heuristic, not a parser): directives must start a line (after whitespace); the string/comment
scanner (`code_view`, own implementation because swift_lexing._mask blanks interpolations) handles escapes,
nested comments, multi-line and raw strings and nested interpolation, but not every exotic literal; more than
14 distinct free atoms in one branch are treated as satisfiable (conservative: flagged); a guard condition this evaluator
cannot parse is reported as a finding (conservative); `Float16` spelled through a typealias
declared inside a guard is not tracked (the typealias use is not a token); other x86_64 environments
(for example an Intel iOS simulator, where Float16 exists) are intentionally not evaluated.

T17 DECISION (does NOT opt in to `swift_lexing._mask(..., keep_interpolations=True)`): this guard already sees
interpolated code through its own `code_view`, and its `interpolation` mutation control disables that rule by falling
back to the default `_mask`, which blanks interpolations. Opting `_mask` in would make the disabled rule see the same
code, so the control (`interp_simple`) would stop biting (vacuous). Pinned by
test_default_mask_blanks_interpolation_so_the_mutation_control_bites.

CONTROLS (this file): must-FLAG inline fixtures (pre-fix shapes of the four defect files, wrong
conditions, #else); must-PASS inline fixtures (guarded code, comments, strings, SIMD4<Float16> in a
guard, nested, #else of the exclusion); the live tree must have zero findings and the checker must
SEE a minimum number of guarded tokens; MUTATION: disabling each rule flips its controls. No git
history is used, so the test passes out of repo.
"""

import importlib.util
import itertools
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location('swift_lexing', ROOT / 'Scripts/ci/swift_lexing.py')
lexing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lexing)
_mask = lexing._mask

TOKEN = re.compile(r'(?<![A-Za-z0-9])(?:_?Float16|CFloat16|simd_half\w*)(?!\w)')
TOKEN_PLAIN = re.compile(r'(?<!\w)Float16(?!\w)')
DIRECTIVE = re.compile(r'^[ \t]*#(if|elseif|else|endif)\b(.*)$')
RULES = ('detect', 'negate_prior', 'nesting', 'both_environments', 'adversarial_free_atoms', 'free_atom_cap',
         'atom_whitespace', 'elseif_accumulate', 'unbalanced_endif', 'unterminated_if', 'osx_alias',
         'extra_tokens', 'interpolation')

ENVIRONMENTS = (
    {'arch(x86_64)': True, 'os(macOS)': True, 'os(OSX)': True, 'os(iOS)': False, 'targetEnvironment(macCatalyst)': False},
    {'arch(x86_64)': True, 'os(macOS)': False, 'os(OSX)': False, 'os(iOS)': True, 'targetEnvironment(macCatalyst)': True},
)
KNOWN_FALSE = re.compile(r'^(arch|os)\(')


class ParseError(Exception):
    pass


def _tokenize(text, rules=RULES):
    tokens, i = [], 0
    while i < len(text):
        c = text[i]
        if c.isspace():
            i += 1
        elif text.startswith('&&', i) or text.startswith('||', i):
            tokens.append(text[i:i + 2]); i += 2
        elif c in '!()':
            tokens.append(c); i += 1
        elif c.isalpha() or c == '_':
            j = i
            while j < len(text) and (text[j].isalnum() or text[j] == '_'):
                j += 1
            if j < len(text) and text[j] == '(':
                depth, k = 0, j
                while k < len(text):
                    depth += (text[k] == '(') - (text[k] == ')')
                    k += 1
                    if depth == 0:
                        break
                if depth != 0:
                    raise ParseError(text)
                tokens.append(('atom', re.sub(r'\s+', '', text[i:k]) if 'atom_whitespace' in rules else text[i:k])); i = k
            else:
                tokens.append(('atom', text[i:j])); i = j
        else:
            raise ParseError(text)
    return tokens


def parse_condition(text, rules=RULES):
    """Expression tree: ('atom', name) | ('not', e) | ('and', a, b) | ('or', a, b)."""
    tokens, pos = _tokenize(text, rules), [0]

    def peek():
        return tokens[pos[0]] if pos[0] < len(tokens) else None

    def take():
        pos[0] += 1
        return tokens[pos[0] - 1]

    def primary():
        t = peek()
        if t is None:
            raise ParseError(text)
        if t == '!':
            take(); return ('not', primary())
        if t == '(':
            take(); e = expr_or()
            if peek() != ')':
                raise ParseError(text)
            take(); return e
        if isinstance(t, tuple):
            take(); return t
        raise ParseError(text)

    def expr_and():
        e = primary()
        while peek() == '&&':
            take(); e = ('and', e, primary())
        return e

    def expr_or():
        e = expr_and()
        while peek() == '||':
            take(); e = ('or', e, expr_and())
        return e

    result = expr_or()
    if peek() is not None:
        raise ParseError(text)
    return result


def _atoms(e, out):
    if e[0] == 'atom':
        out.add(e[1])
    else:
        for child in e[1:]:
            _atoms(child, out)


def _eval(e, values):
    kind = e[0]
    if kind == 'atom':
        return values[e[1]]
    if kind == 'not':
        return not _eval(e[1], values)
    if kind == 'and':
        return _eval(e[1], values) and _eval(e[2], values)
    return _eval(e[1], values) or _eval(e[2], values)


def satisfiable(exprs, rules=RULES):
    """True when some Intel-macOS/Catalyst environment and free-atom assignment makes all exprs true."""
    atoms = set()
    for e in exprs:
        _atoms(e, atoms)
    environments = ENVIRONMENTS if 'both_environments' in rules else ENVIRONMENTS[:1]
    if 'osx_alias' not in rules:
        environments = tuple({k: v for k, v in env.items() if k != 'os(OSX)'} for env in environments)
    for env in environments:
        fixed, free = {}, []
        for a in atoms:
            if a in env:
                fixed[a] = env[a]
            elif a == 'true' or a == 'false':
                fixed[a] = a == 'true'
            elif KNOWN_FALSE.match(a):
                fixed[a] = False
            else:
                free.append(a)
        if 'adversarial_free_atoms' not in rules:
            assignments = [tuple(False for _ in free)]
        elif len(free) > 14:
            if 'free_atom_cap' in rules:
                return True
            return False
        else:
            assignments = itertools.product((False, True), repeat=len(free))
        for combo in assignments:
            values = dict(fixed)
            values.update(zip(free, combo))
            if all(_eval(e, values) for e in exprs):
                return True
    return False


def code_view(source):
    """Same-length copy of `source`: comments and string-literal TEXT blanked, interpolation segments kept as code.

    Handles `//`, nested `/* */`, "..." and triple-quoted strings with escapes, raw strings `#"..."#`
    (interpolation is `\\#(`), and nested strings inside interpolations. Newlines are preserved.
    """
    out, n = list(source), len(source)

    def blank(a, b):
        for k in range(a, min(b, n)):
            if out[k] != '\n':
                out[k] = ' '

    def string(i):
        hashes = 0
        while source[i + hashes] == '#':
            hashes += 1
        quote = '"""' if source.startswith('"""', i + hashes) else '"'
        closing = quote + '#' * hashes
        interpolation = '\\' + '#' * hashes + '('
        j = i + hashes + len(quote)
        blank(i, j)
        while j < n:
            if source.startswith(closing, j):
                blank(j, j + len(closing))
                return j + len(closing)
            if source.startswith(interpolation, j):
                blank(j, j + len(interpolation))
                j = code(j + len(interpolation), True)
            elif source[j] == '\\' and hashes == 0:
                blank(j, j + 2)
                j += 2
            else:
                blank(j, j + 1)
                j += 1
        return j

    def code(i, in_interpolation):
        depth = 0
        while i < n:
            c = source[i]
            if source.startswith('//', i):
                j = source.find('\n', i)
                j = n if j < 0 else j
                blank(i, j)
                i = j
            elif source.startswith('/*', i):
                level, j = 1, i + 2
                while j < n and level:
                    if source.startswith('/*', j):
                        level += 1; j += 2
                    elif source.startswith('*/', j):
                        level -= 1; j += 2
                    else:
                        j += 1
                blank(i, j)
                i = j
            elif c == '"' or (c == '#' and re.match(r'#+"', source[i:i + 40])):
                i = string(i)
            else:
                if c == '(':
                    depth += 1
                elif c == ')':
                    if in_interpolation and depth == 0:
                        return i + 1
                    depth -= 1
                i += 1
        return i

    code(0, False)
    return ''.join(out)


def scan(source, rules=RULES):
    """[(line, message)] for each Float16 token not excluded from Intel macOS / Catalyst."""
    if 'detect' not in rules:
        return []
    masked = code_view(source) if 'interpolation' in rules else _mask(source)
    token = TOKEN if 'extra_tokens' in rules else TOKEN_PLAIN
    findings, stack = [], []   # stack entries: {'chain': [conds of this #if/#elseif chain], 'current': [exprs of the active branch]}
    for number, line in enumerate(masked.split('\n'), 1):
        m = DIRECTIVE.match(line)
        if m:
            kind, rest = m.group(1), m.group(2).strip()
            try:
                if kind == 'if':
                    cond = parse_condition(rest, rules)
                    stack.append({'chain': [cond], 'current': [cond]})
                elif kind in ('elseif', 'else'):
                    if not stack:
                        findings.append((number, 'unbalanced #' + kind)); continue
                    top = stack[-1]
                    prior = top['chain']
                    cond = parse_condition(rest, rules) if kind == 'elseif' else ('atom', 'true')
                    negated = [('not', p) for p in prior] if 'negate_prior' in rules else []
                    top['current'] = negated + [cond]
                    if kind == 'elseif' and 'elseif_accumulate' in rules:
                        top['chain'] = prior + [cond]
                else:
                    if not stack:
                        if 'unbalanced_endif' in rules:
                            findings.append((number, 'unbalanced #endif'))
                        continue
                    stack.pop()
            except ParseError:
                findings.append((number, 'unparseable #' + kind + ' condition: ' + rest))
                if kind == 'if':
                    stack.append({'chain': [('atom', 'unparseable')], 'current': [('atom', 'unparseable')]})
                elif stack:
                    stack[-1]['current'] = [('atom', 'unparseable')]
            continue
        for _ in token.finditer(line):
            if 'nesting' in rules:
                effective = [e for entry in stack for e in entry['current']]
            else:
                effective = list(stack[-1]['current']) if stack else []
            if not stack or satisfiable(effective, rules):
                findings.append((number, 'Float16 reachable on Intel macOS / Mac Catalyst: ' + line.strip()))
    if stack and 'unterminated_if' in rules:
        findings.append((0, 'unterminated #if'))
    return findings


def guarded_count(source):
    """Number of Float16 tokens that sit inside an accepted guard (for the live-tree visibility floor)."""
    total = len(TOKEN.findall(code_view(source)))
    return total - len([f for f in scan(source) if f[1].startswith('Float16')])


GUARD = '#if !(arch(x86_64) && (os(macOS) || targetEnvironment(macCatalyst)))'

MUST_FLAG = {
    'unguarded_call': ('let h = Float16(1)\n', 1),
    'unguarded_array': ('struct N01HalfVector { var storage: [Float16] }\n', 1),
    'unguarded_self': ('let isNativeHalf = S.self == Float16.self\n', 1),
    'prefix_bitpattern': ('let x = Float16(bitPattern: 3)\nlet y = Float16.infinity.bitPattern\nlet z = Float16.nan\n', 3),
    'prefix_policy_test': ('let restored = lhs.map { Float(Float16($0)) }\n', 1),
    'prefix_simd': ('let v = SIMD4<Float16>(a)\n', 1),
    'wrong_arch_only': ('#if arch(x86_64)\nlet h = Float16(1)\n#endif\n', 1),
    'wrong_os_ios_only': ('#if os(iOS)\nlet h = Float16(1)\n#endif\n', 1),
    'wrong_not_macos_only': ('#if !os(macOS)\nlet h = Float16(1)\n#endif\n', 1),
    'wrong_not_catalyst_only': ('#if !targetEnvironment(macCatalyst)\nlet h = Float16(1)\n#endif\n', 1),
    'wrong_catalyst_env_only': ('#if !(arch(x86_64) && os(macOS))\nlet h = Float16(1)\n#endif\n', 1),
    'wrong_free_atom': ('#if canImport(Foundation)\nlet h = Float16(1)\n#endif\n', 1),
    'wrong_native_flag_only': ('#if NATIVE_FLOAT16_SUPPORTED\nlet h = Float16(1)\n#endif\n', 1),
    'else_branch_of_guard': ('%s\n#else\nlet h = Float16(1)\n#endif\n' % GUARD, 1),
    'after_endif': ('%s\nlet a = 1\n#endif\nlet h = Float16(1)\n' % GUARD, 1),
    'runtime_constant_is_not_a_guard': ('if nativeFloat16Supported { _ = Float16(1) }\n', 1),
    'nested_unguarded_inner': ('%s\n#if canImport(X)\n#else\n#endif\n#endif\n#if canImport(X)\nlet h: Float16 = 1\n#endif\n' % GUARD, 1),
    'unparseable_condition': ('#if foo @@ bar\nlet h = Float16(1)\n#endif\n', 2),
    'interp_simple': ('let s = "v \\(Float16(1))"\n', 1),
    'interp_nested_string': ('let s = "a \\("b \\(Float16(1))")"\n', 1),
    'interp_multiline': ('let s = """\nx \\(Float16(1))\n"""\n', 1),
    'interp_raw': ('let s = #"Float16 \\#(Float16(1))"#\n', 1),
    'osx_alias': ('#if os(OSX)\nlet h = Float16(1)\n#endif\n', 1),
    'cfloat16': ('let a: CFloat16 = 1\n', 1),
    'simd_half': ('let v: simd_half4 = .init()\n', 1),
    'underscore_float16': ('let a: _Float16 = 1\n', 1),
    'many_free_atoms': ('#if ' + ' && '.join('FLAG%d' % i for i in range(15)) + '\nlet h = Float16(1)\n#endif\n', 1),
    'ws_atom': ('#if arch( x86_64 )\nlet h = Float16(1)\n#endif\n', 1),
    'unbalanced_endif': ('#endif\nlet a = 1\n', 1),
    'unterminated_if': ('#if canImport(X)\nlet a = 1\n', 1),
}

MUST_PASS = {
    'library_form': ('%s\nlet h = Float16(1)\n#endif\n' % GUARD, 0),
    'compiler_gated_form': ('#if compiler(>=5.3) && (arch(arm64) || arch(x86_64)) && !(arch(x86_64) && (os(macOS) || targetEnvironment(macCatalyst)))\nlet a: Float16 = 1\n#else\nlet a = 0\n#endif\n', 0),
    'custom_flag_form': ('#if NATIVE_FLOAT16_SUPPORTED && !(arch(x86_64) && (os(macOS) || targetEnvironment(macCatalyst)))\nreturn Float(Float16(bitPattern: x))\n#else\nreturn 0\n#endif\n', 0),
    'arm64_alone_accepted': ('#if arch(arm64)\nlet h = Float16(1)\n#endif\n', 0),
    'not_x86': ('#if !arch(x86_64)\nlet h = Float16(1)\n#endif\n', 0),
    'simd_in_guard': ('%s\nlet v = SIMD4<Float16>(a)\n#endif\n' % GUARD, 0),
    'else_of_exclusion': ('#if arch(x86_64) && (os(macOS) || targetEnvironment(macCatalyst))\nlet n = 0\n#else\nlet h = Float16(1)\n#endif\n', 0),
    'nested_in_guard': ('%s\n#if canImport(Foundation)\nlet h = Float16(1)\n#endif\n#endif\n' % GUARD, 0),
    'guard_around_test_func': ('%s\n@Test func half() { _ = Float16(2).bitPattern }\n#endif\n@Test func plain() {}\n' % GUARD, 0),
    'comments_and_strings': ('// Float16(1)\n/* [Float16] */\nlet s = "Float16(2) \\(x)"\n/// uses Float16\n', 0),
    'identifier_substrings': ('let nativeFloat16Supported = true\nlet MaxFloat16Value: Float = 1\n', 0),
    'elseif_chain': ('#if os(Linux)\nlet a = 0\n#elseif !(arch(x86_64) && (os(macOS) || targetEnvironment(macCatalyst)))\nlet h = Float16(1)\n#endif\n', 0),
    'raw_string_without_hash_interpolation': ('let s = #"Float16 \\(Float16(1))"#\n', 0),
    'interp_in_guard': ('%s\nlet s = "v \\(Float16(1))"\n#endif\n' % GUARD, 0),
    'osx_in_guard': ('#if !(arch(x86_64) && (os(OSX) || targetEnvironment(macCatalyst)))\nlet h = Float16(1)\n#endif\n', 0),
    'ws_guard': ('#if !( arch( x86_64 ) && ( os( macOS ) || targetEnvironment( macCatalyst ) ) )\nlet h = Float16(1)\n#endif\n', 0),
    'elseif_chain_else': ('#if canImport(X)\nlet a = 0\n#elseif arch(x86_64) && (os(macOS) || targetEnvironment(macCatalyst))\nlet b = 0\n#else\nlet h = Float16(1)\n#endif\n', 0),
    'alias_names_in_comments': ('// CFloat16 simd_half4 _Float16\nlet isFloat16Supported = true\n', 0),
    'directive_in_comment_ignored': ('// #if canImport(X)\n%s\nlet h = Float16(1)\n#endif\n' % GUARD, 0),
}

# rule -> (control name, expected-flag-count when the rule is disabled)
MUTATION_CONTROLS = {
    'detect': ('unguarded_call', 0),
    'negate_prior': ('else_of_exclusion', 1),
    'nesting': ('nested_in_guard', 1),
    'both_environments': ('wrong_catalyst_env_only', 0),
    'adversarial_free_atoms': ('wrong_free_atom', 0),
    'free_atom_cap': ('many_free_atoms', 0),
    'atom_whitespace': ('ws_atom', 0),
    'elseif_accumulate': ('elseif_chain_else', 1),
    'unbalanced_endif': ('unbalanced_endif', 0),
    'unterminated_if': ('unterminated_if', 0),
    'osx_alias': ('osx_alias', 0),
    'extra_tokens': ('cfloat16', 0),
    'interpolation': ('interp_simple', 0),
}


def live_files():
    for top in ('Sources', 'Tests', 'Benchmarks'):
        base = ROOT / top
        if base.is_dir():
            yield from sorted(base.rglob('*.swift'))


def _fixture(name):
    return (MUST_FLAG.get(name) or MUST_PASS[name])[0]


class Float16AvailabilityGuardTests(unittest.TestCase):
    def test_must_flag_controls(self):
        for name, (source, count) in MUST_FLAG.items():
            with self.subTest(name):
                self.assertEqual(len(scan(source)), count, scan(source))

    def test_must_pass_controls(self):
        for name, (source, _) in MUST_PASS.items():
            with self.subTest(name):
                self.assertEqual(scan(source), [])

    def test_pre_fix_shapes_are_red_and_guarded_shapes_green(self):
        pre = ('struct N01HalfVector { var storage: [Float16] }\n'
               'func t() { let r = N01HalfVector(Array(repeating: Float16(1), count: 4)) }\n'
               'let isNativeHalf = S.self == Float16.self\n'
               'let b = Float16(bitPattern: 1)\n')
        self.assertEqual(len(scan(pre)), 4)
        post = '%s\n%s#endif\n' % (GUARD, pre)
        self.assertEqual(scan(post), [])

    def test_every_rule_has_a_mutation_control(self):
        self.assertEqual(set(MUTATION_CONTROLS), set(RULES))

    def test_each_rule_bites_by_mutation(self):
        for rule, (name, flagged_when_disabled) in MUTATION_CONTROLS.items():
            with self.subTest(rule):
                source = _fixture(name)
                enabled = len(scan(source))
                disabled = len(scan(source, rules=tuple(r for r in RULES if r != rule)))
                self.assertNotEqual(enabled, disabled, 'mutating %s did not change %s' % (rule, name))
                self.assertEqual(disabled, flagged_when_disabled)

    def test_default_mask_blanks_interpolation_so_the_mutation_control_bites(self):
        source = _fixture('interp_simple')
        self.assertIsNone(TOKEN_PLAIN.search(_mask(source)))
        self.assertTrue(TOKEN_PLAIN.search(_mask(source, keep_interpolations=True)))   # what an opt-in would see

    def test_whitespace_normalisation_also_protects_the_accepted_form(self):
        fixture = _fixture('ws_guard')
        self.assertEqual(scan(fixture), [])
        self.assertEqual(len(scan(fixture, rules=tuple(r for r in RULES if r != 'atom_whitespace'))), 1)

    def test_live_tree_is_clean_and_checker_sees_guards(self):
        findings, guarded, files = [], 0, 0
        for path in live_files():
            text = path.read_text(encoding='utf-8')
            if not re.search(r'Float16|simd_half', text):
                continue
            files += 1
            guarded += guarded_count(text)
            findings += ['%s:%d: %s' % (path.relative_to(ROOT), n, msg) for n, msg in scan(text)]
        self.assertEqual(findings, [], '\n'.join(findings))
        # Visibility floor: the checker must actually see the known guarded sites.
        self.assertGreaterEqual(files, 8)
        self.assertGreaterEqual(guarded, 28)


if __name__ == '__main__':
    unittest.main()
