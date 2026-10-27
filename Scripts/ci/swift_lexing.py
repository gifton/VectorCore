r"""Shared lexical helpers for the Swift source policies (stdlib only).

Extracted unchanged from test_swift_test_output_paths.py so the output-path policy and the
clock-derived-assertion policy lex Swift identically. This is a lexical heuristic, not a parser:

* `_mask` blanks string literals and `//`, `/* */` comments, keeping length and newlines, so offsets
  in the mask equal offsets in the source. A raw string `#"…"#` ends only at a quote followed by as
  many `#` (P02 final fix, review A FN-2: before, `#"a"b"#` hid the rest of its line).
  `_mask(source, keep_interpolations=True)` keeps the code inside string interpolations as CODE
  (`\( … )`, and `\#( … )` in a raw string with one `#`, nested interpolations and multi-line strings
  included) and blanks only literal text (review A FA-2). The default still blanks interpolated code:
  the P02 final-fix blast radius showed that keeping it for every user would make the clock-assertion
  policy read interpolated message text as code, against its documented rule, and would change two
  guards' documented limits (test_inert_provider_wrappers L5, test_float16_availability_guard's own
  interpolation rule). A guard that must see interpolated code opts in (test_gram_route_lexical.py,
  test_no_array_conditional_casts.py).
* `_unbacktick` replaces every backtick of a masked text with a space (same length), so an escaped
  identifier such as `` `Array` `` or `` `gramBatch` `` reads as the plain name (review A FA-3).
* `_statements` joins lines while () or [] are unbalanced (counted on the masked text).
* `_Scopes` segments brace scopes; `class|struct|enum|extension|actor|protocol` scopes are
  transparent, every other scope is local. See test_swift_test_output_paths.py for the
  documented limits.
* `_classify` tells a use of an identifier from an argument label or a member name.
LIMITS of `_mask`: regex literals (`/…/`, `#/…/#`) are not lexed; `/* */` comments do not nest (the
first `*/` ends one). Behavior is pinned by Scripts/ci/tests/test_swift_lexing.py (must-mask and
must-keep fixtures for each form) and by the users' own discovery tests.
"""

import re

FUNCTION_HEADER = re.compile(r'\bfunc\s+(\w+)')
COMPUTED_HEADER = re.compile(r'\bvar\s+(\w+)\s*:[^={]*$')
TYPE_HEADER = re.compile(r'\s*(?:[@\w.]+(?:\([^)]*\))?\s+)*?(?:class|struct|enum|extension|actor|protocol)\s+(?!func\b|var\b|let\b)\w')
CONTINUATION = re.compile(r'\s*(?:->|where\b|throws\b|rethrows\b|async\b)')
IDENTIFIER = re.compile(r'(?<!\w)[A-Za-z_]\w*')


def _statements(source):
    """(first line number, text) per statement: lines joined while () or [] are unbalanced.

    Parens are counted on the MASKED source (strings and comments blanked first), so
    `"http://x"` or a `)` inside a string does not truncate or unbalance a line.
    Lines are joined with a single space, so offsets inside `text` equal offsets in
    `source` from the statement's first line start (source must use LF line endings only).
    """
    result, buffer, start, depth = [], [], None, 0
    lines = source.split('\n')
    masked = _mask(source).split('\n')
    if lines and lines[-1] == '':
        lines.pop()
        masked.pop()
    for number, (line, counted) in enumerate(zip(lines, masked), 1):
        if not buffer:
            start = number
        buffer.append(line)
        depth += counted.count('(') + counted.count('[') - counted.count(')') - counted.count(']')
        if depth <= 0:
            result.append((start, ' '.join(buffer)))
            buffer, depth = [], 0
    if buffer:
        result.append((start, ' '.join(buffer)))
    return result


def _mask(source, keep_interpolations=False):
    """Same-length copy of `source` with string-literal text and comments blanked (newlines kept).

    Interpolated code is blanked too unless `keep_interpolations` is True; raw strings (`#"…"#`, `##"…"##`, …) end only
    at a quote followed by as many `#`, and their escapes and interpolations need as many `#` after the backslash.
    """
    out, n = list(source), len(source)

    def blank(a, b):
        for k in range(a, min(b, n)):
            if out[k] != '\n':
                out[k] = ' '

    def code(i, interpolation):
        # Scan code from offset i; inside an interpolation, return the offset of the `)` that closes it.
        depth = 0
        while i < n:
            ch = source[i]
            if ch == '"' or ch == '#':
                j = i
                while j < n and source[j] == '#':
                    j += 1
                if j < n and source[j] == '"':
                    i = string(i, j - i)
                    continue
            if source.startswith('//', i):
                j = source.find('\n', i)
                j = n if j < 0 else j
                blank(i, j)
                i = j
                continue
            if source.startswith('/*', i):
                j = source.find('*/', i + 2)
                j = n if j < 0 else j + 2
                blank(i, j)
                i = j
                continue
            if interpolation and ch == '(':
                depth += 1
            elif interpolation and ch == ')':
                if depth == 0:
                    return i
                depth -= 1
            i += 1
        return n

    def string(i, hashes):
        # i is at the first `#` (or at the opening quote when hashes == 0); returns the offset after the literal.
        quote = i + hashes
        multi = source.startswith('"""', quote)
        k = quote + (3 if multi else 1)
        close = ('"""' if multi else '"') + '#' * hashes
        escape = '\\' + '#' * hashes
        blank(i, k)
        while k < n:
            if source.startswith(close, k):
                blank(k, k + len(close))
                return k + len(close)
            if not multi and source[k] == '\n':
                return k  # an unterminated single-line literal ends at its line
            if source.startswith(escape, k):
                after = k + len(escape)
                if after < n and source[after] == '(':
                    end = code(after + 1, True)
                    if keep_interpolations:
                        blank(k, after + 1)
                        blank(end, end + 1)
                    else:
                        blank(k, end + 1)
                    k = end + 1
                    continue
                blank(k, after + 1)  # an escape: the backslash (and its `#`s) and one character
                k = after + 1
                continue
            blank(k, k + 1)
            k += 1
        return n

    code(0, False)
    return ''.join(out)


def _unbacktick(masked):
    """`masked` with every backtick replaced by a space (same length): `` `Array` `` reads as `Array`."""
    return masked.replace('`', ' ')


def _header_chunk(header):
    """Last statement of the text preceding a `{` (multi-line signatures stay together)."""
    chunks, current, depth = [], [], 0
    for line in header.split('\n'):
        current.append(line)
        depth += line.count('(') + line.count('[') - line.count(')') - line.count(']')
        if depth <= 0:
            joined = ' '.join(current)
            if chunks and CONTINUATION.match(joined):
                chunks[-1] += ' ' + joined
            elif joined.strip():
                chunks.append(joined)
            current, depth = [], 0
    if current and ' '.join(current).strip():
        chunks.append(' '.join(current))
    return chunks[-1] if chunks else ''


class _Scopes:
    """Brace scopes of a source. Scope 0 is the file. `at[i]` is the innermost scope of offset i."""

    def __init__(self, source):
        mask = _mask(source)
        self.parent, self.opens, self.closes = [None], [-1], [len(source)]
        self.transparent, self.name = [True], [None]
        self.at = [0] * (len(source) + 1)
        stack, boundary = [0], 0
        for i, ch in enumerate(mask):
            self.at[i] = stack[-1]
            if ch == '{':
                chunk = _header_chunk(mask[boundary:i])
                function = FUNCTION_HEADER.search(chunk)
                computed = None if function else COMPUTED_HEADER.search(chunk)
                self.parent.append(stack[-1])
                self.opens.append(i)
                self.closes.append(len(source))
                self.transparent.append(bool(TYPE_HEADER.match(chunk)))
                self.name.append((function or computed).group(1) if (function or computed) else None)
                stack.append(len(self.parent) - 1)
                boundary = i + 1
            elif ch == '}':
                if len(stack) > 1:
                    self.closes[stack.pop()] = i
                boundary = i + 1
            elif ch == ';':
                boundary = i + 1
        self.at[len(source)] = 0
        self.ancestors = []
        for scope in range(len(self.parent)):
            chain, walk = set(), scope
            while walk is not None:
                chain.add(walk)
                walk = self.parent[walk]
            self.ancestors.append(chain)

    def declaring(self, offset):
        """Scope a `let`/`var` at `offset` binds in: nearest enclosing non-type scope (0 = file-wide)."""
        scope = self.at[offset]
        while scope and self.transparent[scope]:
            scope = self.parent[scope]
        return scope

    def bodies(self):
        """(name, body start, body end) for functions and computed properties."""
        for scope, name in enumerate(self.name):
            if name:
                yield name, self.opens[scope] + 1, self.closes[scope]


def _classify(text, start, end):
    """'member' (name after a single `.`), 'label' (argument label) or 'use' for the identifier text[start:end]."""
    if start > 0 and text[start - 1] == '.' and not (start > 1 and text[start - 2] == '.'):
        return 'member'
    after = end
    while after < len(text) and text[after] in ' \t\n':
        after += 1
    before = start - 1
    while before >= 0 and text[before] in ' \t\n':
        before -= 1
    if after < len(text) and text[after] == ':' and before >= 0 and text[before] in '(,':
        return 'label'
    return 'use'
