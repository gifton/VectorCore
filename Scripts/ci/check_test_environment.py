#!/usr/bin/env python3
"""Bounded lexical ownership policy for test environment reads.

Discovers direct accesses, dictionary/process-info aliases, literal constant keys,
and getenv. Dynamic keys require an exact file/expression classification. This is
not interprocedural Swift data-flow analysis; see Docs/Test_Environment_Controls.md.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import shlex

from swift_lexing import _mask, _unbacktick, _Scopes

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name('test_environment_controls.json')


def literal(text):
    try:
        value = json.loads(text.strip())
        return value if isinstance(value, str) else None
    except (ValueError, TypeError):
        return None


def string_constants(source):
    """Only complete immutable string initializers are literal keys."""
    mask = _unbacktick(_mask(source, keep_interpolations=True))
    constants = {}
    for match in re.finditer(r'\b(let|var)\s+(\w+)\s*(?::\s*String)?\s*=\s*("(?:\\.|[^"\\\n])*")', source):
        if mask[match.start():match.start() + 3] not in ('let', 'var'):
            continue
        tail = mask[match.end():]
        line_tail = tail.split('\n', 1)[0].strip()
        complete = not line_tail or line_tail[0] in ';,}'
        continued = re.match(r'\s*(?:[.+*/%&|^<>=?!-]|as\b|is\b)', tail)
        value = literal(match[3]) if match[1] == 'let' and complete and not continued else None
        name = match[2]
        constants[name] = value if name not in constants or constants[name] == value else None
    return constants


def chain_pattern(name):
    return re.escape(name).replace(r'\.', r'\s*\.\s*')


def discover(source, shared_constants=None):
    mask = _unbacktick(_mask(source, keep_interpolations=True))
    constants = dict(shared_constants or {})
    constants.update(string_constants(source))
    info = {'ProcessInfo.processInfo', 'Foundation.ProcessInfo.processInfo'}
    aliases = {'environment', 'env'}
    assignments = []
    for match in re.finditer(r'\b(?:let|var)\s+(\w+)(?:[ \t]*:[^=\n]+)?\s*=\s*', source):
        if mask[match.start():match.start() + 3] not in ('let', 'var'):
            continue
        rhs = re.match(r'\w+(?:\s*\.\s*\w+)*', mask[match.end():])
        if rhs:
            assignments.append((match[1], re.sub(r'\s+', '', rhs[0]), match.end() + rhs.end()))
    for _ in range(len(assignments) + 1):
        prior = (set(info), set(aliases))
        for name, rhs, end in assignments:
            if re.match(r'\s*\[', mask[end:]):
                continue  # A subscript value is not an alias of the dictionary.
            if rhs in info:
                info.add(name)
            if rhs in aliases or any(rhs == owner + '.environment' for owner in info):
                aliases.add(name)
        if prior == (info, aliases):
            break
    receivers = sorted(aliases | {name + '.environment' for name in info}, key=len, reverse=True)
    pattern = r'(?<![\w.])(?:' + '|'.join(chain_pattern(name) for name in receivers) + r')\s*\['
    reads, writes, unresolved = set(), set(), set()
    scopes = _Scopes(source)
    loops = []
    for loop in re.finditer(r'\bfor\s+(\w+)\s+in\s*\[', mask):
        end = mask.find(']', loop.end())
        opening = mask.find('{', end)
        try:
            values = json.loads(source[loop.end() - 1:end + 1])
        except ValueError:
            continue
        if opening in scopes.opens and isinstance(values, list) and all(isinstance(v, str) for v in values):
            loops.append((loop[1], opening, scopes.closes[scopes.opens.index(opening)], values))
    def unresolved_site(expression, offset):
        owner = next((scopes.name[i] for i in sorted(scopes.ancestors[scopes.at[offset]], reverse=True) if scopes.name[i]), '<file>')
        return owner + ':' + expression
    for match in re.finditer(pattern, mask):
        end = mask.find(']', match.end())
        expression = source[match.end():end].strip() if end >= 0 else '<unterminated>'
        key = literal(expression) or constants.get(expression)
        write = end >= 0 and bool(re.match(r'\s*=(?!=)', mask[end + 1:]))
        if key is None and not write:
            finite = [values for name, start, stop, values in loops if name == expression and start < match.start() < stop]
            if finite:
                reads.update(finite[-1])
            else:
                unresolved.add(unresolved_site(expression, match.start()))
        elif key is not None:
            (writes if write else reads).add(key)
    for match in re.finditer(r'\bgetenv\s*\(', mask):
        end = mask.find(')', match.end())
        expression = source[match.end():end].strip()
        key = literal(expression) or constants.get(expression)
        if key is None:
            unresolved.add(unresolved_site('getenv:' + expression, match.start()))
        else:
            reads.add(key)
    # A following subscript is consumed only by the normalized receiver pattern.
    # Other operations on a known environment dictionary require a named review.
    dictionary_pattern = r'(?<![\w.])(?:' + '|'.join(chain_pattern(name) for name in receivers) + r')\b'
    for match in re.finditer(dictionary_pattern, mask):
        rest = mask[match.end():]
        before = mask[max(0, match.start() - 200):match.start()]
        if re.match(r'\s*\[', rest):
            continue
        member = re.match(r'\s*\.\s*(\w+)', rest)
        if member:
            unresolved.add(unresolved_site('<environment member:' + member[1] + '>', match.start()))
        elif '.' in re.sub(r'\s+', '', match[0]):
            if re.search(r'\b(?:let|var)\s+\w+(?:\s*:[^=\n]+)?\s*=\s*$', before):
                continue
            unresolved.add(unresolved_site('<environment snapshot>', match.start()))
    for member in re.finditer(r'\.\s*environment\b', mask):
        prefix = mask[:member.start()]
        known = any(re.search(chain_pattern(name) + r'\s*$', prefix) for name in info)
        if not known and not re.match(r'\s*=(?!=)', mask[member.end():]):
            unresolved.add(unresolved_site('<environment receiver>', member.start()))
    return reads, writes, unresolved


def has_setter(path, key, value=None):
    if not path.is_file():
        return False
    source = path.read_text()
    if key not in source:
        return False
    if path.suffix == '.py':
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False
        pairs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                pairs += [(target.slice, node.value) for target in node.targets if isinstance(target, ast.Subscript)]
            elif isinstance(node, ast.Dict):
                pairs += list(zip(node.keys, node.values))
            elif isinstance(node, ast.keyword):
                pairs.append((ast.Constant(node.arg), node.value))
        return any(isinstance(k, ast.Constant) and k.value == key and
                   (value is None or isinstance(v, ast.Constant) and str(v.value) == value) for k, v in pairs)
    if path.suffix == '.swift':
        mask = _mask(source)
        # Subscript assignment or a dictionary literal passed to the child launcher.
        for match in re.finditer(r'\[\s*"' + re.escape(key) + r'"\s*\]\s*=(?!=)|"' + re.escape(key) + r'"\s*:', source):
            anchor = match.start() if source[match.start()] == '[' else match.end() - 1
            if mask[anchor] not in ('[', ':'):
                continue
            tail = source[match.end():].lstrip()
            if value is None or tail.startswith(json.dumps(value)):
                return True
        # A constant-key write such as the child-probe marker.
        if value is None:
            if key in discover(source)[1]:
                return True
            for declaration in re.finditer(r'\blet\s+(\w+)(?:\s*:\s*String)?\s*=\s*"' + re.escape(key) + '"', source):
                if mask[declaration.start():declaration.start() + 3] == 'let' and re.search(r'\[\s*' + declaration[1] + r'\s*\]\s*=(?!=)', mask):
                    return True
        return False
    if path.suffix in ('.yml', '.yaml'):
        # Bounded workflow mapping traversal; ignore scalar run text/action inputs.
        parents = []
        scalar_indent = None
        for line in source.splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            indent = len(line) - len(line.lstrip())
            if scalar_indent is not None and indent > scalar_indent:
                continue
            scalar_indent = None
            while parents and indent <= parents[-1][0]:
                parents.pop()
            match = re.match(r"\s*(?:-\s+)?(['\"]?)([\w-]+)\1:\s*(.*?)\s*$", line)
            if not match:
                continue
            name, setting = match[2], match[3].split(' #', 1)[0].strip()
            ancestry = [entry[1] for entry in parents]
            environment_scope = (ancestry == ['env'] or
                                 len(ancestry) == 3 and ancestry[0] == 'jobs' and ancestry[-1] == 'env' or
                                 len(ancestry) == 4 and ancestry[0] == 'jobs' and ancestry[-2:] == ['steps', 'env'])
            if name == key and environment_scope and (value is None or setting.strip("'\"") == value):
                return True
            if setting.startswith(('|', '>')):
                scalar_indent = indent
            elif not setting or setting.startswith('#'):
                parents.append((indent, name))
        return False
    # Supported shell forms have assignments before an actual swift/python command.
    blocks = re.findall(r'```(?:sh|bash|shell)\n(.*?)```', source, re.S) if path.suffix == '.md' else [source]
    for block in blocks:
        for line in block.splitlines():
            try:
                tokens = shlex.split(line, comments=True)
            except ValueError:
                continue
            if tokens and tokens[0] == 'env':
                tokens = tokens[1:]
            settings = {}
            while tokens and re.match(r'^[A-Za-z_][A-Za-z_0-9]*=', tokens[0]):
                name, setting = tokens.pop(0).split('=', 1)
                settings[name] = setting
            if tokens and tokens[0] in ('swift', 'python3') and key in settings:
                if value is None or settings[key] == value:
                    return True
    return False


def load_controls():
    value = json.loads(MANIFEST.read_text())
    return value['controls'], value['dynamic']


def check(root, controls, special):
    issues = []
    used = set()
    shared = {}
    for path in (root / 'Tests').rglob('*.swift'):
        source = path.read_text()
        # Only file-level declarations may supply keys in a different source file.
        values = string_constants(source)
        for match in re.finditer(r'^(?:(?:public|package|internal) )?(?:let|var)\s+(\w+)\b', source, re.M):
            name = match[1]
            value = values.get(name)
            shared[name] = value if name not in shared or shared[name] == value else None
    for path in sorted((root / 'Tests').rglob('*.swift')):
        relative = path.relative_to(root).as_posix()
        reads, _, unresolved = discover(path.read_text(), shared)
        for expression in unresolved:
            entry = special.get(relative, {}).get(expression)
            if (not entry or not entry.get('reason') or not isinstance(entry.get('keys'), list)
                    or entry.get('sourceSHA256') != hashlib.sha256(path.read_bytes()).hexdigest()):
                issues.append(f'{relative}: unresolved environment access {expression}')
            else:
                reads.update(entry['keys'])
        used.update(reads)
    for key in sorted(used):
        entry = controls.get(key)
        if not entry or not entry.get('scope'):
            issues.append(f'unclassified test environment key: {key}')
            continue
        kind = entry.get('kind')
        if kind == 'system':
            if not entry.get('owner'):
                issues.append(f'{key}: system owner missing')
        elif kind == 'internal':
            if not has_setter(root / entry['owner'], key):
                issues.append(f'{key}: internal setter missing in {entry["owner"]}')
        elif kind == 'gate':
            if not has_setter(root / entry['activation'], key, entry['value']):
                issues.append(f'{key}: tracked activation missing in {entry["activation"]}')
            document = root / entry['document']
            if not document.is_file() or key not in document.read_text():
                issues.append(f'{key}: documented scope missing')
        else:
            issues.append(f'{key}: unsupported classification {kind}')
    return issues


if __name__ == '__main__':
    problems = check(ROOT, *load_controls())
    if problems:
        raise SystemExit('\n'.join(problems))
    print('All discovered test environment reads have classified, tracked owners.')
