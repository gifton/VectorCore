#!/usr/bin/env python3
"""Claim ledger check (V04): every tier-A statement in the maintained claim regions has exactly one
reviewed disposition in Scripts/ci/claim_ledger.json, and the evidence each entry names exists.

WHAT THIS IS. An inventory and lexical guard backed by a full manual source review. It is not a
theorem prover and not a semantic detector for arbitrary English: it cannot tell whether a claim is
true. It enforces only that (a) every comment or string statement in a region that matches the
tier-A vocabulary has exactly one reviewed ledger entry, (b) that entry still binds to the current
text at the same symbol, (c) the evidence the entry names exists, and (d) removed statements and
forbidden obsolete phrases stay absent. A named test supports its claim only within its own fixture
domain. Stdlib only. READS: the ledger; the region files; the files named as evidence (test files of
`named-test`, `bodyPin` and `flagTests` references, dated artifacts); every Swift file matched by a
forbidden-phrase path or a caller-inventory glob (with today's ledger that is every Swift file under
Sources/, Tests/ and Benchmarks/, each read once); the Markdown files named as literal forbidden-phrase
paths (today README.md, CHANGELOG.md and the 13 maintained documents under Docs/); and the recorded-site files.

STATEMENTS (normalization comment-text-v1). Extracted from `//`, `///` and `/* */` comments and from
single-line, multi-line (`\"\"\"`) and raw string literals. Code tokens are never inspected (so exact
`== 0` or bit checks in tests are never flagged). A statement is one doc paragraph or one bullet:
consecutive own-line comments with the same marker form a block, a blank comment line ends a
paragraph, a line starting `-`/`*`/`+`/`•`/`N.`/`N)`/`#`/`MARK:` starts a new statement, wrapped lines
are joined, and each line inside a ``` fence is its own statement. A trailing comment (after code)
is its own block. A string literal is segmented the same way (a one-line literal is one statement);
each interpolation `\\(...)` becomes the placeholder `\\()`. A literal matching the ledger's
`excludeLiterals` is not a statement. That regex admits only pure identifiers: segments joined by `.`
or `-`, each segment a word that starts with a letter or `_` (interpolations allowed after the first
character), a lone interpolation, a decimal integer, or a hexadecimal digest of 16 or more digits.
So `"batch.euclid2.512"` and `"Vector\\()Optimized"` are not statements, while `"50%"`, `"10-20%"`,
`"3x-faster"` and `"\\(n)x"` are (Extractor tests). A literal whose matching text matches any forbidden
phrase is never excluded (`"lock-free"`, `"auto-prefetch"`, `"zero-copy"`); a tier-A word in a pure identifier
that matches no forbidden phrase (`"optimized"` in a case ID) is still excluded. Normalized
text: NFC, whitespace collapsed, lowercased; fingerprint = first 16 hex digits of sha256(normalized
text). MATCHING TEXT: the vocabulary, the forbidden phrases and tombstone phrases are matched on the
normalized text with Swift escapes neutralized (`\\n`, `\\t`, `\\r`, `\\0` → space; `\\"`, `\\'`,
`\\\\` → the character; `\\u{hex}` → the character; raw-string `\\#n` likewise; a backslash followed by
whitespace, a multi-line literal's line continuation, joins the next line), so `"\\nMemory savings: 50%"`
is seen as `memory savings: 50%`; fingerprints stay on the raw text.

MARKDOWN (normalization markdown-text-v1). A `.md` path (a region, or a literal forbidden-phrase path) is read as
prose, not lexed as Swift: a statement is one paragraph (wrapped lines joined), one list item (with its indented
continuation lines), one table row, one heading, or one line inside a fenced code block; a leading blockquote `>` is
stripped and a blank line or a thematic break (`---`) ends a paragraph. Inline Markdown (emphasis, code spans, links)
is kept verbatim. The symbol is the heading chain, `section <heading text>` joined by `::` (`<file>` before the first
heading). Markdown files are always parsed (no whole-file screen), and a glob path never selects them (Extractor and
MarkdownRegion tests).

SECTION REGIONS (T13 RD6). A `sectionRegions` item names a Markdown path and a list of section symbols (`section ...`
heading chains). R6 counts only the tier-A statements whose symbol is a listed section or lies below one; an entry for
that path whose symbol is outside every listed section is an R8 problem, and so is a listed section that no longer
exists in the file (a stale scope). Tombstones (R2) still apply to the whole file. Today: CHANGELOG.md, section
`[Unreleased]` (the release step that retires its entries is in Docs/Maintaining.md).

SYMBOLS. A statement attaches to `kind name` scopes joined by `::` (modifiers dropped, e.g.
`enum BatchKernels_SoA::func euclid2_tiled`). Brace scopes are named by the declaration that opens
them; control-flow blocks and closures are unnamed and transparent. A comment at file or type level
also attaches to the next declaration (blank, comment and attribute-only lines skipped); a trailing
comment at that level attaches to the declaration on its own line; a literal at that level attaches
to the nearest preceding declaration line in the same scope. Inside a body, the enclosing named
chain is the symbol.

RULES. R1 a live (retained / corrected / unverified-contract / not-a-claim) entry's path, symbol
and fingerprint are attached in current source. R2 a `removed` tombstone's fingerprint and
`absentPhrase` are absent from its path, and so are the `history.originalFingerprint` of a rewritten
live entry and every fingerprint in its `history.superseded` list. R3 every `named-test` reference
`path::Suite.testName` resolves to a `func`/`def` (XCTest `test...`, Swift Testing `@Test`, or Python
`test_...`) named by the part after the last `.` in that file, and the statement or its comment block
contains that method name; only the method part is matched, so a citation that loses its suite
qualifier still passes. R4 a `dated-artifact` reference names an existing in-repo path with a 64-hex
sha256 that matches and an ISO date. Two references are compared by their resolved identity (path, enclosing
type, method): `path::method` and `path::Suite.method` of one test are the same test. R5 an `unverified-contract` statement carries explicit wording
("not verified", "unverified", "not established", "not promised", or "no ... guarantee"). A string
literal is runtime text and cannot carry the wording without changing output, so for a literal the
wording may instead sit in ONE SENTENCE of its adjacent comment block (comment lines directly above
it, a trailing comment, and the doc comment of the nearest enclosing named declaration) that also
refers to runtime text: the sentence names `Q11` or says "runtime", "returned" or "printed" followed
by "text", "string(s)", "output" or "note(s)". R6 every tier-A statement (on its matching text) in a
region is covered by exactly one entry (occurrences counted). R7 no statement (on its matching text) of a file matched by a forbidden phrase's
paths matches its pattern, except a statement whose (path, fingerprint) is listed in the rule's
`exempt` list (each exemption names an owner and must still match a present statement). A file matched
only by the rule's `commentOnlyPaths` is checked on its comment statements only (its string literals,
such as log-location fixtures, are not); such a rule's `fixtures.passLiteral` strings must match the
pattern, showing the literal scope is what lets them pass. A path
without glob metacharacters (`*`, `?`, `[`) is a literal file path; a glob is matched with fnmatch
below its literal directory prefix. A rule's `screen` (default: its pattern) is run first on a
whole-file normalized, escape-neutralized text to skip files, and an explicit screen is also run on each
statement before the pattern; it must be a necessary condition for the pattern, so a pattern with an anchor (`^`, `$`, `\\A`, `\\Z`) or a lookaround must carry an explicit
keyword `screen` (a non-empty string; an empty screen fails R8). R8 IDs are unique, (path, symbol, fingerprint) is not double-covered, every entry
has the required fields and enumerated values, every `evidence.bodyPin` and `evidence.flagTests`
reference resolves to a test whose identity the entry's `named-test` references do not also cite, a
test named in any entry's `bodyPin` is `named-test` evidence only for an entry listed in the ledger's
`bodyPinDescribers` (each listed entry must cite one), and every forbidden-phrase rule has non-empty
`paths` (each matching at least one file), well-formed exemptions, and `fixtures.fail` (strings; each
must match the pattern and pass the screen; each `fixtures.pass` string must not match). Every non-null
`acceptedContract`, split on `/`, must be a key of the ledger's `contracts` table, whose items carry a
non-empty description and source and must each be named by at least one entry (T13 RD10). A body pin (a drift guard that hashes the source a note was
written from) and a flag test (a test that checks a record field the note relates to) are recorded
for traceability only: neither is evidence for the note. Recorded sites (`recordedSites`, Q17) must
match the lexical count of their pattern in the named symbol. Q14 (`callerInventory`): the lexical
uses of each gated symbol, per file outside its defining file, must equal `callers`, and its
occurrences inside the defining file (declaration included) must equal `definedInOccurrences`.

LIMITS (nothing beyond the named controls in Scripts/ci/tests/test_claim_ledger.py is proven).
* Lexical vocabulary: a claim phrased without any tier-A term is not seen; a tier-A term in a
  non-claim needs a reviewed `not-a-claim` entry (no blanket suppression). A forbidden phrase is a
  regex: a phrasing it does not match is not seen (each rule's scope is its fixture table).
* Sentence-scoped hedges (rules 6, 8 and 9): a sentence ends at `.`, `!` or `?` followed by up to two closing
  characters from `*_)]"'` and the curly quotes and then whitespace (so `**Not verified.** X is fast.` is two sentences),
  and a match passes when its own sentence carries a hedge ("not verified", "unverified", "not measured",
  "not established", "not promised"; rule 8 also "no ... guarantee(s)", "no speed claim" and "not a
  performance/speed claim") or an in-repository artifact path (Benchmarks/, Docs/, Tests/ or Scripts/
  ending in .md, .json, .csv or .txt; the path is not checked to exist). The check does not verify that
  the hedge qualifies the claimed words, only that it is in the same sentence.
* Rule 8 (qualitative labels, T13 RD1) matches the label list in its pattern only: "optimized" alone
  (a type-family name), "deterministic", "accurate", "scalable" or "accelerated" are not matched, and
  "fast path(s)"/"fast route(s)" are allowed as branch names (so are "faster/fastest path/route") while a hyphenated
  compound such as "fast-path" is matched. The noun forms "efficiency", "robustness" and "reliability" are not matched
  (live examples in non-region sources, routed as V04-R-0011: the CPUContext "cache efficiency" comment, Dimension
  "memory efficiency", the QuantizedKernels "Robustness Check" heading), and an inline hardware claim such as the
  AlignedMemory "Apple Silicon prefers 64-byte alignment" comment is outside the vocabulary (V04-R-0012). Rule 9 (rule 6S, T13 RD4) counts a multiplier only with an attached speed word,
  so a bare "4x" claim without one passes. Rule 10 (T13 RD5) matches the spelling `zero[- ]?copy(?:ing)?` only.
  Rules 8 and 10 cover library sources, README.md and the maintained documents; Tests, Benchmarks,
  CHANGELOG.md and dated records are not covered.
* Matching text folds the Unicode hyphens U+2010, U+2011, U+2012, U+2013, U+2212 and U+00AD to `-`; other look-alikes
  (U+FE63, U+FF0D) are not folded.
* Symbols are names, not signatures: overloads (several `init`, several `func f`) share a symbol,
  so moving a statement between two overloads of one name is not detected. `#if` branches are not
  evaluated; regex literals are not lexed; a brace inside one `#if` branch only can mis-scope.
* Q14 counts words, not resolved references: a homonym (another type's member with the same name)
  counts as a use, so `callers` records reviewed homonyms too.
* An `adjacent-invariant` or `unverified-contract` entry's reasoning is reviewed text; the check
  verifies only its presence and wording, not the reasoning.
* A named test is checked to exist and to be cited; whether it runs is shown by the suite run, not
  by this check.
* A `contracts` source is reviewed text: the check does not open or hash it. A section region's scope
  is a symbol prefix, so a statement moved under another heading of the same section stays in scope.
"""

import argparse
import bisect
import fnmatch
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = 'Scripts/ci/claim_ledger.json'
_SPEC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SPEC_DIR))
from swift_lexing import _header_chunk  # noqa: E402  (shared, pinned helper)

DISPOSITIONS = ('retained', 'corrected', 'unverified-contract', 'removed', 'not-a-claim')
LIVE = ('retained', 'corrected', 'unverified-contract', 'not-a-claim')
CATEGORIES = ('performance-quantitative', 'performance-qualitative', 'robustness', 'numerical-accuracy',
              'memory-model', 'measurement-boundary', 'concurrency', 'non-claim')
EVIDENCE_KINDS = ('adjacent-invariant', 'named-test', 'dated-artifact', 'unverified-contract', 'none')
EVIDENCE_FOR = {
    'retained': ('adjacent-invariant', 'named-test', 'dated-artifact'),
    'corrected': ('adjacent-invariant', 'named-test', 'dated-artifact', 'unverified-contract', 'none'),
    'unverified-contract': ('unverified-contract',),
    'removed': ('none',),
    'not-a-claim': ('none',),
}
REQUIRED = ('id', 'path', 'symbol', 'fingerprint', 'statement', 'category', 'disposition', 'evidence',
            'acceptedContract', 'history')
UNVERIFIED_WORDING = re.compile(r'not verified|unverified|not established|not promised|\bno\b[^.;:]{0,80}\bguarantee')
# R5 for a string literal: the inherited sentence must also refer to runtime text (or carry a Q11 marker).
REFERS_TO_RUNTIME_TEXT = re.compile(r'\bq11\b|\b(?:runtime|returned|printed)\s+(?:text|strings?|output|notes?)\b')
CLOSERS = '*_)]"\'\u2019\u201d'   # closing characters that may follow a sentence-ending period (`.**`, `.)`, `."`, `.’`)
SENTENCE_END = re.compile(rf'(?<=[.!?])\s+|(?<=[.!?][{re.escape(CLOSERS)}])\s+|(?<=[.!?][{re.escape(CLOSERS)}]{{2}})\s+')
HYPHENS = str.maketrans({c: '-' for c in '\u2010\u2011\u2012\u2013\u2212\u00ad'})   # Unicode hyphens folded to '-' for matching
GLOB_META = re.compile(r'[*?\[]')
# A pattern with an anchor or a lookaround is not a necessary condition when run on whole-file text.
SCREEN_NEEDED = re.compile(r'\^|\$|\\[AZ]|\(\?<?[=!]')
# Swift escapes inside a normalized (lowercased) literal: \0 \t \n \r \" \' \\ and \u{hex}, also in raw-string form (\#n),
# and a line continuation (a backslash that ended a line of a multi-line literal; whitespace follows it after joining).
ESCAPE = re.compile(r'\\#*(?:u\{([0-9a-f]{1,8})\}|([0ntr"\'\\])|(\s+))')
FENCE = re.compile(r'^`{3}')
STARTS_STATEMENT = re.compile(r'^(?:[-*+\u2022](?:\s|$)|\d+[.)]\s|#{1,6}\s|mark:|MARK:)')
SHA256 = re.compile(r'^[0-9a-f]{64}$')
ISO_DATE = re.compile(r'^\d{4}-\d{2}-\d{2}$')

_ATTR = r'@[\w.]+(?:\s*\((?:[^()]|\([^()]*\))*\))?'
_MOD = (r'(?:public|private|fileprivate|internal|package|open|static|class|final|override|mutating|'
        r'nonmutating|lazy|weak|unowned|convenience|required|nonisolated|indirect|dynamic|optional|prefix|'
        r'postfix|infix|consuming|borrowing|distributed|isolated|unsafe)(?:\s*\([^)]*\))?')
DECL = re.compile(rf'^\s*(?:{_ATTR}\s*)*(?:{_MOD}\s+)*(func|init|deinit|subscript|var|let|case|struct|class|'
                  rf'enum|extension|actor|protocol|typealias|associatedtype|macro)(?![\w])\s*'
                  rf'(`?[A-Za-z_][\w.]*`?|[^\s\w(<{{,;:`]+)?')
TYPE_KINDS = ('struct', 'class', 'enum', 'extension', 'actor', 'protocol')
SKIPPABLE = re.compile(rf'^\s*(?:(?:{_ATTR}\s*)+|#\w+.*)?$')


# --------------------------------------------------------------------------------------------- lexing

class Lexed:
    """Comments, string literals and a same-length mask (comments and literals blanked) of a source."""

    def __init__(self, source):
        self.source = source
        self.comments = []   # (start, end, marker)       marker: '///', '//', '/*'
        self.strings = []    # (start, end, content)      content with interpolations as \(…)
        self._mask = list(source)
        self._lex(0, len(source), top=True)
        self.mask = ''.join(self._mask)
        self.line_starts = [0] + [i + 1 for i, ch in enumerate(source) if ch == '\n']

    def _blank(self, a, b):
        for k in range(a, b):
            if self._mask[k] != '\n':
                self._mask[k] = ' '

    def line_of(self, offset):
        return bisect.bisect_right(self.line_starts, offset)

    def _lex(self, i, limit, top=False, stop_at_paren=False):
        """Lex code from i; with stop_at_paren return the offset just after the balancing ')'."""
        s, depth = self.source, 0
        while i < limit:
            if s.startswith('//', i):
                j = s.find('\n', i)
                j = limit if j < 0 else j
                marker = '///' if s.startswith('///', i) and not s.startswith('////', i) else '//'
                self.comments.append((i, j, marker))
                self._blank(i, j)
                i = j
            elif s.startswith('/*', i):
                j, nest = i + 2, 1
                while j < limit and nest:
                    if s.startswith('/*', j):
                        nest, j = nest + 1, j + 2
                    elif s.startswith('*/', j):
                        nest, j = nest - 1, j + 2
                    else:
                        j += 1
                self.comments.append((i, j, '/*'))
                self._blank(i, j)
                i = j
            elif s[i] == '"' or (s[i] == '#' and re.match(r'#+"', s[i:i + 16])):
                hashes = 0
                while s[i + hashes] == '#':
                    hashes += 1
                j = self._string(i, hashes, limit)
                self._blank(i, j)
                i = j
            elif stop_at_paren and s[i] == '(':
                depth, i = depth + 1, i + 1
            elif stop_at_paren and s[i] == ')':
                if depth == 0:
                    return i + 1
                depth, i = depth - 1, i + 1
            else:
                i += 1
        return limit

    def _string(self, i, hashes, limit):
        s = self.source
        start = i
        i += hashes
        multi = s.startswith('"""', i)
        quote = '"""' if multi else '"'
        close = quote + '#' * hashes
        escape = '\\' + '#' * hashes
        i += len(quote)
        parts = []
        while i < limit:
            if s.startswith(close, i):
                i += len(close)
                break
            if not multi and s[i] == '\n':
                break
            if s.startswith(escape, i):
                k = i + len(escape)
                if k < limit and s[k] == '(':
                    i = self._lex(k + 1, limit, stop_at_paren=True)
                    parts.append('\\()')
                    continue
                parts.append(s[i:k + 1])
                i = k + 1
                continue
            parts.append(s[i])
            i += 1
        content = ''.join(parts)
        if multi:
            content = content.split('\n', 1)[1] if '\n' in content else content
            content = content.rsplit('\n', 1)[0] if '\n' in content else content
        self.strings.append((start, i, content, multi))
        return i


# ----------------------------------------------------------------------------------------- statements

def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip().lower()


def fingerprint(text):
    return hashlib.sha256(normalize(text).encode('utf-8')).hexdigest()[:16]


def neutralize(norm):
    """Matching text of a normalized statement: Swift escapes replaced (`\\n`, `\\t`, `\\r`, `\\0` → space; `\\"`, `\\'`,
    `\\\\` → the character; `\\u{hex}` → the character; a backslash followed by whitespace, a multi-line literal's
    line continuation, joins the next line; the Unicode hyphens U+2010, U+2011, U+2012, U+2013, U+2212 and the soft hyphen
    U+00AD are folded to `-`), then normalized again. Vocabulary, forbidden phrases and
    tombstone phrases are matched on this text; fingerprints stay on the normalized raw text."""
    def replace(match):
        if match.group(1):
            try:
                return chr(int(match.group(1), 16))
            except (ValueError, OverflowError):
                return ' '
        if match.group(3):
            return ''   # line continuation: the next line joins this one
        return ' ' if match.group(2) in '0ntr' else match.group(2)
    norm = norm.translate(HYPHENS)
    return normalize(ESCAPE.sub(replace, norm)) if '\\' in norm else norm


def _segment(lines):
    """[(line index, text)] of statements from comment/literal lines (markers already stripped)."""
    out, current, first, fenced = [], [], None, False

    def flush():
        nonlocal current, first
        if current:
            out.append((first, ' '.join(current)))
        current, first = [], None

    for index, raw in enumerate(lines):
        text = raw.strip()
        if FENCE.match(text):
            flush()
            fenced = not fenced
            continue
        if fenced:
            flush()
            if text:
                out.append((index, text))
            continue
        if not text:
            flush()
            continue
        if STARTS_STATEMENT.match(text):
            flush()
        if first is None:
            first = index
        current.append(text)
    flush()
    return out


class Statement:
    __slots__ = ('path', 'line', 'symbol', 'text', 'norm', 'match', 'fp', 'kind', 'block')

    def __init__(self, path, line, symbol, text, kind, block):
        self.path, self.line, self.symbol, self.text, self.kind, self.block = path, line, symbol, text, kind, block
        self.norm = normalize(text)
        self.match = neutralize(self.norm)   # what the vocabulary and the forbidden phrases see
        self.fp = hashlib.sha256(self.norm.encode('utf-8')).hexdigest()[:16]


class SourceFile:
    """Statements and symbols of one Swift source."""

    def __init__(self, path, source, exclude_literal):
        self.path, self.source = path, source
        self.lexed = Lexed(source)
        self.lines = source.split('\n')
        self.mask_lines = self.lexed.mask.split('\n')
        self._by_line = None
        self._scopes()
        self.statements = []
        self._comment_statements()
        self._string_statements(exclude_literal)
        self.symbols = {st.symbol for st in self.statements} | set(self.scope_symbols)

    # Scopes: (parent, open offset, close offset, label or None, is_type)
    def _scopes(self):
        mask = self.lexed.mask
        self.parent, self.opens, self.closes, self.label, self.is_type = [None], [-1], [len(mask)], [None], [True]
        stack, boundary = [0], 0
        self.scope_at_line = {}
        for i, ch in enumerate(mask):
            if ch == '{':
                chunk = _header_chunk(mask[boundary:i])
                decl = DECL.match(chunk)
                label, is_type = None, False
                if decl:
                    label = _label(decl)
                    is_type = decl.group(1) in TYPE_KINDS
                self.parent.append(stack[-1])
                self.opens.append(i)
                self.closes.append(len(mask))
                self.label.append(label)
                self.is_type.append(is_type)
                stack.append(len(self.parent) - 1)
                boundary = i + 1
            elif ch == '}':
                if len(stack) > 1:
                    self.closes[stack.pop()] = i
                boundary = i + 1
            elif ch == ';':
                boundary = i + 1
        self.scope_symbols = set()
        for scope in range(1, len(self.parent)):
            chain = self.chain(scope)
            if chain:
                self.scope_symbols.add('::'.join(chain))

    def scope_at(self, offset):
        """Innermost scope whose braces strictly enclose offset."""
        best = 0
        for scope in range(1, len(self.opens)):
            if self.opens[scope] < offset < self.closes[scope] and self.opens[scope] > self.opens[best]:
                best = scope
        return best

    def chain(self, scope):
        names = []
        while scope:
            if self.label[scope]:
                names.append(self.label[scope])
            scope = self.parent[scope]
        return list(reversed(names))

    def _at_declaration_level(self, scope):
        return scope == 0 or self.is_type[scope]

    def _symbol(self, offset, line, end_line, trailing=False, literal=False):
        scope = self.scope_at(offset)
        chain = self.chain(scope)
        if self._at_declaration_level(scope):
            decl = None
            if trailing:
                decl = DECL.match(self.mask_lines[line - 1])
            elif literal:
                own = self.mask_lines[line - 1]
                decl = DECL.match(own)
                if not decl and own.lstrip().startswith('@'):
                    # An attribute argument (`@Test("...")`) belongs to the declaration it decorates.
                    for number in range(line + 1, len(self.mask_lines) + 1):
                        if SKIPPABLE.match(self.mask_lines[number - 1]):
                            continue
                        decl = DECL.match(self.mask_lines[number - 1])
                        break
                elif not decl:
                    # Nearest preceding declaration line in the same scope (a multi-line initializer).
                    floor = self.lexed.line_of(self.opens[scope]) if scope else 0
                    for number in range(line - 1, floor, -1):
                        text = self.mask_lines[number - 1]
                        start = self.lexed.line_starts[number - 1] + len(text) - len(text.lstrip())
                        if self.scope_at(start) != scope:
                            continue
                        decl = DECL.match(text)
                        if decl:
                            break
            else:
                for number in range(end_line + 1, len(self.mask_lines) + 1):
                    text = self.mask_lines[number - 1]
                    if SKIPPABLE.match(text):
                        continue
                    decl = DECL.match(text)
                    break
            if decl:
                chain = chain + [_label(decl)]
        return '::'.join(chain) if chain else '<file>'

    def _comment_statements(self):
        lexed, source = self.lexed, self.source
        blocks = []
        for start, end, marker in lexed.comments:
            line = lexed.line_of(start)
            own_line = not source[lexed.line_starts[line - 1]:start].strip()
            if marker == '/*':
                body = source[start + 2:end - 2 if source.endswith('*/', 0, end) else end]
                if body.startswith('*'):
                    body = body[1:]
                raw = [re.sub(r'^\s*\*(?!/)', '', each) for each in body.split('\n')]
                blocks.append([start, line, lexed.line_of(end), raw, own_line, marker])
                continue
            text = source[start + len(marker):end]
            previous = blocks[-1] if blocks else None
            if (own_line and previous and previous[4] and previous[5] == marker and previous[2] == line - 1):
                previous[2] = line
                previous[3].append(text)
            else:
                blocks.append([start, line, line, [text], own_line, marker])
        for start, line, end_line, raw, own_line, marker in blocks:
            symbol = self._symbol(start, line, end_line, trailing=not own_line)
            block_text = '\n'.join(raw)
            for index, text in _segment(raw):
                self.statements.append(Statement(self.path, line + index, symbol, text, 'comment',
                                                 (line, end_line, block_text)))

    def _string_statements(self, exclude_literal):
        """One statement per literal; single-line literals joined by `+` form one statement."""
        lexed, mask = self.lexed, self.lexed.mask
        groups = []
        for start, end, content, multi in sorted(lexed.strings):
            previous = groups[-1] if groups else None
            if (previous and not multi and not previous[-1][3] and previous[-1][1] <= start
                    and re.fullmatch(r'\s*\+\s*', mask[previous[-1][1]:start])):
                previous.append((start, end, content, multi))
            else:
                groups.append([(start, end, content, multi)])
        for group in groups:
            start, multi = group[0][0], group[0][3]
            content = ''.join(part[2] for part in group)
            if not content.strip() or exclude_literal.match(content):
                continue
            line = lexed.line_of(start)
            symbol = self._symbol(start, line, line, literal=True)
            block = self._adjacent_block(line, start)
            segments = _segment(content.split('\n')) if multi else [(0, content)]
            for index, text in segments:
                if text.strip():
                    self.statements.append(Statement(self.path, line + index + (1 if multi else 0), symbol,
                                                     text, 'string', block))

    def _doc_above(self, line):
        """Own-line comment lines directly above `line` (attribute-only lines skipped)."""
        lines, number = [], line - 1
        while number >= 1:
            text = self.lines[number - 1].strip()
            if text.startswith('//') or (text.startswith('*') or text.startswith('/*')) and not self.mask_lines[number - 1].strip():
                lines.insert(0, text)
            elif not (text.startswith('@') and SKIPPABLE.match(self.mask_lines[number - 1])):
                break
            number -= 1
        return number + 1, lines

    def _comments_on_line(self, line):
        """(start, end) of the comments that start on `line`, in source order (index built once per file)."""
        if self._by_line is None:
            self._by_line = defaultdict(list)
            for start, end, _ in self.lexed.comments:
                self._by_line[self.lexed.line_of(start)].append((start, end))
        return self._by_line.get(line, ())

    def _adjacent_block(self, line, offset):
        """(first, last, text): the comment lines directly above `line`, a trailing comment on it, and the
        doc comment of the nearest enclosing named declaration (a literal's adjacent comment block)."""
        first, lines = self._doc_above(line)
        for start, end in self._comments_on_line(line):
            if self.source[self.lexed.line_starts[line - 1]:start].strip():
                lines.append(self.source[start:end])
        scope = self.scope_at(offset)
        while scope and not self.label[scope]:
            scope = self.parent[scope]
        if scope:
            header = self.lexed.line_of(self.opens[scope])
            for number in range(header, max(header - 40, 0), -1):
                if DECL.match(self.mask_lines[number - 1]):
                    lines.extend(self._doc_above(number)[1])
                    break
        return (first, line, '\n'.join(lines))


MD_FENCE = re.compile(r'^(`{3,}|~{3,})')
MD_HEADING = re.compile(r'^(#{1,6})\s+(.*?)(?:\s+#+)?\s*$')
MD_BREAK = re.compile(r'^(?:-{3,}|\*{3,}|_{3,})$')
MD_ITEM = re.compile(r'^(?:[-*+](?:\s|$)|\d+[.)](?:\s|$))')


class MarkdownFile:
    """Statements and section symbols of one Markdown document (normalization markdown-text-v1; see the module docstring).
    A statement's block (used by R3) is its own text."""

    def __init__(self, path, source):
        self.path, self.source = path, source
        self.statements, self.symbols = [], {'<file>'}
        stack, fence, current, first = [], None, [], None

        def symbol():
            return '::'.join(f'section {title}' for _, title in stack) or '<file>'

        def add(line, text, end=None):
            self.statements.append(Statement(path, line, symbol(), text, 'markdown', (line, end or line, text)))

        def flush(end):
            nonlocal current, first
            if current:
                add(first, ' '.join(current), end)
            current, first = [], None

        lines = source.split('\n')
        for number, raw in enumerate(lines, 1):
            text = raw.strip()
            if fence:
                if re.fullmatch(re.escape(fence[0]) + '{%d,}' % len(fence), text):
                    fence = None
                elif text:
                    add(number, text)
                continue
            opening = MD_FENCE.match(text)
            if opening:
                flush(number - 1)
                fence = opening.group(1)
                continue
            while text.startswith('>'):
                text = text[1:].strip()
            if not text or MD_BREAK.match(text):
                flush(number - 1)
                continue
            heading = MD_HEADING.match(text)
            if heading:
                flush(number - 1)
                level = len(heading.group(1))
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, heading.group(2).strip()))
                self.symbols.add(symbol())
                add(number, text)
                continue
            if text.startswith('|'):
                flush(number - 1)
                add(number, text)
                continue
            if MD_ITEM.match(text):
                flush(number - 1)
            if first is None:
                first = number
            current.append(text)
        flush(len(lines))


def source_file(path, text, exclude):
    """The statement extractor for `path`: Markdown prose for a `.md` file, the Swift lexer otherwise."""
    return MarkdownFile(path, text) if path.endswith('.md') else SourceFile(path, text, exclude)


def _label(decl):
    kind, name = decl.group(1), (decl.group(2) or '').strip('`')
    if kind in ('init', 'deinit', 'subscript') or not name or not re.match(r'[A-Za-z_]', name):
        return kind if not name or kind in ('init', 'deinit', 'subscript') else f'{kind} {name}'
    return f'{kind} {name.split(".")[0] if kind != "extension" else name}'


# --------------------------------------------------------------------------------------------- ledger

class Repo:
    """File reads rooted at `root` (the tests point it at temporary copies)."""

    def __init__(self, root):
        self.root = Path(root)
        self._cache = {}
        self._globs = {}

    def exists(self, path):
        return (self.root / path).is_file()

    def read(self, path):
        if path not in self._cache:
            self._cache[path] = (self.root / path).read_text(encoding='utf-8')
        return self._cache[path]

    def glob(self, pattern):
        """Files for one path pattern. A pattern without glob metacharacters is a literal path (returned when the
        file exists); otherwise the Swift files below its literal directory prefix that fnmatch the pattern."""
        if not GLOB_META.search(pattern):
            return [pattern] if self.exists(pattern) else []
        if pattern not in self._globs:   # memoized per Repo instance (one check)
            prefix = GLOB_META.split(pattern, 1)[0]
            base = self.root / prefix.rsplit('/', 1)[0] if '/' in prefix else self.root
            found = sorted(str(p.relative_to(self.root)) for p in base.rglob('*.swift')) if base.is_dir() else []
            self._globs[pattern] = [p for p in found if fnmatch.fnmatch(p, pattern)]
        return list(self._globs[pattern])


def tier_a(norm, vocabulary):
    return any(rx.search(norm) for rx in vocabulary)


class LiteralExclusion:
    """`excludeLiterals`, except that a literal matching any forbidden phrase (on its matching text) is never excluded."""

    def __init__(self, pattern, forbidden):
        self.regex = re.compile(pattern)
        self.forbidden = [re.compile(each) for each in forbidden]
        self.pattern = pattern + '\x00' + '\x00'.join(forbidden)   # parse-cache key

    def match(self, content):
        if not self.regex.match(content):
            return None
        text = neutralize(normalize(content))
        return None if any(rx.search(text) for rx in self.forbidden) else True


def load(repo, ledger_path=LEDGER):
    ledger = json.loads(repo.read(ledger_path))
    vocabulary = [re.compile(each) for each in ledger['vocabulary']['tierA']]
    exclude = LiteralExclusion(ledger['vocabulary']['excludeLiterals'],
                               [rule['pattern'] for rule in ledger.get('forbiddenPhrases', []) if rule.get('pattern')])
    return ledger, vocabulary, exclude


_PARSED = {}
_MASKS = {}


def parse(repo, path, exclude):
    """SourceFile for `path`, memoized by (path, content, exclusion) so repeated checks re-lex only changed files."""
    text = repo.read(path)
    key = (path, hashlib.sha256(text.encode('utf-8')).hexdigest(), exclude.pattern)
    if key not in _PARSED:
        _PARSED[key] = source_file(path, text, exclude)
    return _PARSED[key]


def _sentences(block_text):
    """Normalized sentences of a comment block: markers stripped, split into statements (paragraphs and bullets) as
    for extraction, then at sentence-ending punctuation followed by whitespace."""
    lines = [re.sub(r'^\s*(?:/{2,}|/\*+|\*+/|\*(?!/))', '', line) for line in block_text.split('\n')]
    out = []
    for _, text in _segment(lines):
        out.extend(part for part in SENTENCE_END.split(normalize(text)) if part)
    return out


def masked(text):
    """`Lexed(text).mask`, memoized by content."""
    key = hashlib.sha256(text.encode('utf-8')).hexdigest()
    if key not in _MASKS:
        _MASKS[key] = Lexed(text).mask
    return _MASKS[key]


def _test_exists(repo, reference):
    if '::' not in reference:
        return False, 'reference is not path::testName'
    path, name = reference.split('::', 1)
    name = name.split('.')[-1]
    if not repo.exists(path):
        return False, f'{path} does not exist'
    text = repo.read(path)
    if path.endswith('.py'):
        ok = name.startswith('test') and re.search(rf'(?m)^\s*def\s+{re.escape(name)}\s*\(', text)
    else:
        code = masked(text)
        match = re.search(rf'\bfunc\s+{re.escape(name)}\s*\(', code)
        ok = bool(match) and (name.startswith('test') or re.search(
            r'@Test\b[^{}]*$', code[max(0, match.start() - 400):match.start()]))
    return bool(ok), None if ok else f'no test func {name} in {path}'


def test_identity(repo, reference, exclude):
    """(path, suite, method) a test reference resolves to: the method is the part after the last `.`; among `func
    method(` declarations in the file, the one inside a type named by the qualifier when one is given, otherwise the
    first. Spellings `path::method` and `path::Suite.method` of one test give one identity. None if it does not resolve."""
    ok, _ = _test_exists(repo, reference)
    if not ok:
        return None
    path, name = reference.split('::', 1)
    parts = name.split('.')
    method, qualifier = parts[-1], (parts[-2] if len(parts) > 1 else None)
    if path.endswith('.py'):
        return (path, '', method)
    parsed = parse(repo, path, exclude)
    suites = []
    for found in re.finditer(rf'\bfunc\s+{re.escape(method)}\s*\(', parsed.lexed.mask):
        chain = parsed.chain(parsed.scope_at(found.start()))
        suites.append(chain[-1].split(' ', 1)[-1] if chain else '')
    suite = qualifier if qualifier in suites else suites[0]
    return (path, suite, method)


def section_scopes(ledger, problems):
    """{path: [section symbol, ...]} of the ledger's `sectionRegions` (T13 RD6); malformed items are R8 problems."""
    scopes = {}
    whole = {region['path'] for region in ledger.get('regions', [])}
    for item in ledger.get('sectionRegions', []):
        path = item.get('path') if isinstance(item, dict) else None
        scope = item.get('sections') if isinstance(item, dict) else None
        if not (isinstance(path, str) and path and isinstance(scope, list) and scope
                and all(isinstance(each, str) and each.startswith('section ') for each in scope) and item.get('reason')):
            problems.append(f'R8 sectionRegions {item!r:.80}: needs a path, a non-empty list of `section ...` symbols and a '
                            'reason')
            continue
        if path in whole or path in scopes:
            problems.append(f'R8 sectionRegions {path}: already a region')
            continue
        scopes[path] = scope
    return scopes


def in_scope(sections, path, symbol):
    """True unless `path` is a section region and `symbol` is outside every listed section (or its subsections)."""
    scope = sections.get(path)
    return scope is None or any(symbol == each or symbol.startswith(each + '::') for each in scope)


def check_contracts(ledger, entries):
    """R8 (T13 RD10): every non-null `acceptedContract`, split on `/`, names a key of the ledger's `contracts` table;
    every contract has a description and a source and is used by at least one entry."""
    problems, contracts, used = [], ledger.get('contracts', {}), set()
    if not isinstance(contracts, dict):
        return ['R8 contracts: must be an object of ID -> {description, source}']
    for key, value in contracts.items():
        if not (isinstance(value, dict) and isinstance(value.get('description'), str) and value['description'].strip()
                and isinstance(value.get('source'), str) and value['source'].strip()):
            problems.append(f'R8 contracts {key}: needs a non-empty description and source')
    for entry in entries:
        accepted = entry.get('acceptedContract')
        if accepted is None:
            continue
        if not (isinstance(accepted, str) and accepted.strip()):
            problems.append(f'R8 {entry.get("id", "<no id>")}: acceptedContract {accepted!r} is not a contract ID')
            continue
        for part in accepted.split('/'):
            used.add(part)
            if part not in contracts:
                problems.append(f'R8 {entry.get("id", "<no id>")}: acceptedContract {part!r} is not in contracts')
    for key in sorted(set(contracts) - used):
        problems.append(f'R8 contracts {key}: no entry names it (stale)')
    return problems


def check(repo, ledger_path=LEDGER):
    """List of problem strings (empty when the ledger and the regions agree)."""
    problems = []
    ledger, vocabulary, exclude = load(repo, ledger_path)
    entries = ledger.get('entries', [])
    region_paths = [region['path'] for region in ledger.get('regions', [])]
    sections = section_scopes(ledger, problems)
    region_paths += [path for path in sections if path not in region_paths]
    files = {}
    for path in region_paths:
        if not repo.exists(path):
            problems.append(f'R1 {path}: region file missing')
            continue
        files[path] = parse(repo, path, exclude)
    for path, scope in sections.items():
        for section in scope:
            if path in files and section not in files[path].symbols:
                problems.append(f'R8 sectionRegions {path}: section {section!r} is not in the file (stale scope)')

    # Body-pin tests (drift guards) by resolved identity; only the listed describers may cite one as named-test evidence.
    pinned = set()
    for entry in entries:
        evidence = entry.get('evidence') if isinstance(entry.get('evidence'), dict) else {}
        for reference in evidence.get('bodyPin') or []:
            if isinstance(reference, str):
                identity = test_identity(repo, reference, exclude)
                if identity:
                    pinned.add(identity)
    describers = set(ledger.get('bodyPinDescribers', []))
    described = set()

    # R8 integrity
    ids, keys, tomb_keys = Counter(), Counter(), Counter()
    for entry in entries:
        missing = [field for field in REQUIRED if field not in entry]
        label = entry.get('id', '<no id>')
        if missing:
            problems.append(f'R8 {label}: missing fields {missing}')
            continue
        ids[entry['id']] += 1
        if entry['disposition'] not in DISPOSITIONS:
            problems.append(f'R8 {label}: disposition {entry["disposition"]!r}')
        if entry['category'] not in CATEGORIES:
            problems.append(f'R8 {label}: category {entry["category"]!r}')
        evidence = entry['evidence']
        if not isinstance(evidence, dict) or not all(k in evidence for k in ('kind', 'reference', 'domain', 'limitations')):
            problems.append(f'R8 {label}: evidence needs kind, reference, domain and limitations')
            continue
        if evidence['kind'] not in EVIDENCE_FOR.get(entry['disposition'], ()):
            problems.append(f'R8 {label}: evidence kind {evidence["kind"]!r} not allowed for {entry["disposition"]}')
        if evidence['kind'] != 'none' and not evidence['reference']:
            problems.append(f'R8 {label}: empty evidence reference')
        cited = set()
        if evidence['kind'] == 'named-test':
            for reference in (evidence['reference'] if isinstance(evidence['reference'], list) else [evidence['reference']]):
                identity = test_identity(repo, reference, exclude) if isinstance(reference, str) else None
                if identity:
                    cited.add(identity)
            for identity in cited & pinned:
                if label in describers:
                    described.add(label)
                else:
                    problems.append(f'R8 {label}: cites body-pin test {identity[1]}.{identity[2]} as named-test evidence; '
                                    'a drift guard is evidence only for an entry listed in bodyPinDescribers')
        for field in ('bodyPin', 'flagTests'):
            if field not in evidence:
                continue
            pins = evidence[field]
            if not (isinstance(pins, list) and pins and all(isinstance(each, str) for each in pins)):
                problems.append(f'R8 {label}: evidence.{field} must be a non-empty list of path::testName')
                continue
            for reference in pins:
                ok, why = _test_exists(repo, reference)
                if not ok:
                    problems.append(f'R8 {label}: evidence.{field} does not resolve: {why}')
                if ok and test_identity(repo, reference, exclude) in cited:
                    problems.append(f'R8 {label}: {reference} is cited as named-test evidence and as evidence.{field}; '
                                    f'a {field} reference is not evidence for the statement')
        if entry['disposition'] == 'not-a-claim' and not entry.get('reason'):
            problems.append(f'R8 {label}: not-a-claim needs a reason')
        if entry['disposition'] == 'corrected':
            history = entry['history'] or {}
            if not (history.get('originalStatement') and history.get('originalFingerprint')):
                problems.append(f'R8 {label}: corrected entry needs history.originalStatement/originalFingerprint')
        if not re.fullmatch(r'[0-9a-f]{16}', str(entry['fingerprint'])):
            problems.append(f'R8 {label}: fingerprint is not 16 hex')
        if entry['disposition'] != 'removed' and fingerprint(entry['statement']) != entry['fingerprint']:
            problems.append(f'R8 {label}: statement text does not hash to its fingerprint')
        if entry['path'] not in files and entry['path'] in region_paths:
            continue
        if entry['path'] not in region_paths:
            problems.append(f'R8 {label}: path {entry["path"]} is not a ledger region')
            continue
        if not in_scope(sections, entry['path'], entry['symbol']):
            problems.append(f'R8 {label}: symbol {entry["symbol"]!r} is outside the sections of region {entry["path"]}')
        key = (entry['path'], entry['symbol'], entry['fingerprint'])
        if entry['disposition'] == 'removed':
            tomb_keys[(entry['path'], entry['fingerprint'])] += 1
        else:
            keys[key] += 1
    for value, count in ids.items():
        if count > 1:
            problems.append(f'R8 duplicate id {value}')
    problems.extend(check_contracts(ledger, entries))
    for label in sorted(describers - described):
        problems.append(f'R8 bodyPinDescribers {label}: not a named-test entry citing a body-pin test (stale)')
    for key, count in keys.items():
        if count > 1:
            problems.append(f'R8 fingerprint double-covered {key}')
    for key, count in tomb_keys.items():
        if count > 1:
            problems.append(f'R8 tombstone double-covered {key}')

    present = defaultdict(Counter)   # path -> Counter[(symbol, fp)]
    by_fp = defaultdict(set)         # path -> {fp}
    statements = defaultdict(list)   # (path, symbol, fp) -> [Statement]
    for path, parsed in files.items():
        for st in parsed.statements:
            by_fp[path].add(st.fp)
            statements[(path, st.symbol, st.fp)].append(st)
            if tier_a(st.match, vocabulary) and in_scope(sections, path, st.symbol):
                present[path][(st.symbol, st.fp)] += 1

    covered = defaultdict(Counter)
    for entry in entries:
        if any(field not in entry for field in REQUIRED) or entry['path'] not in files:
            continue
        label, path, evidence = entry['id'], entry['path'], entry['evidence']
        parsed = files[path]
        if entry['disposition'] == 'removed':
            # R2
            if entry['fingerprint'] in by_fp[path]:
                problems.append(f'R2 {label}: tombstoned statement is present again in {path}')
            phrase = entry.get('absentPhrase')
            if phrase and any(re.search(phrase, st.match) for st in parsed.statements):
                problems.append(f'R2 {label}: tombstoned phrase {phrase!r} is present again in {path}')
            continue
        # R1
        key = (path, entry['symbol'], entry['fingerprint'])
        if entry['symbol'] not in parsed.symbols:
            problems.append(f'R1 {label}: stale anchor: symbol {entry["symbol"]!r} not in {path}')
        elif not statements.get(key):
            problems.append(f'R1 {label}: stale anchor: no statement {entry["fingerprint"]} at {entry["symbol"]!r} in {path}')
        occurrences = entry.get('occurrences', 1)
        found = len(statements.get(key, []))
        if statements.get(key) and found != occurrences:
            problems.append(f'R1 {label}: {found} occurrences at the anchor, ledger says {occurrences}')
        covered[path][(entry['symbol'], entry['fingerprint'])] += occurrences
        original = (entry['history'] or {}).get('originalFingerprint')
        if original:
            if original != entry['fingerprint'] and original in by_fp[path]:
                problems.append(f'R2 {label}: tombstone violation: the corrected statement\'s original text is present again in {path}')
        for superseded in (entry['history'] or {}).get('superseded', []):
            if superseded.get('fingerprint') in by_fp[path]:
                problems.append(f'R2 {label}: tombstone violation: superseded text {superseded["fingerprint"]} '
                                f'({superseded.get("revision", "?")}) is present again in {path}')
        # R3
        if evidence['kind'] == 'named-test':
            references = evidence['reference'] if isinstance(evidence['reference'], list) else [evidence['reference']]
            for reference in references:
                ok, why = _test_exists(repo, reference)
                if not ok:
                    problems.append(f'R3 {label}: missing named evidence: {why}')
                    continue
                name = reference.split('::', 1)[1].split('.')[-1]
                # The bound statement's own block; when the text no longer matches its fingerprint, every
                # statement still attached to the entry's symbol (so a deleted citation is reported as R3).
                candidates = statements.get(key) or [st for st in parsed.statements if st.symbol == entry['symbol']]
                if not any(name in st.text or name in st.block[2] for st in candidates):
                    problems.append(f'R3 {label}: missing named evidence: no statement at {entry["symbol"]!r} names {name}')
        # R4
        if evidence['kind'] == 'dated-artifact':
            artifact = evidence['reference'] if isinstance(evidence['reference'], dict) else {}
            artifact_path = artifact.get('path', '')
            if not (artifact_path and repo.exists(artifact_path) and SHA256.match(artifact.get('sha256', ''))
                    and ISO_DATE.match(artifact.get('date', ''))):
                problems.append(f'R4 {label}: dated artifact needs an in-repo path, sha256 and date')
            elif hashlib.sha256((repo.root / artifact_path).read_bytes()).hexdigest() != artifact['sha256']:
                problems.append(f'R4 {label}: dated artifact sha256 mismatch')
        # R5
        if entry['disposition'] == 'unverified-contract':
            # A runtime string literal cannot carry the wording without changing output, so for a literal the
            # wording may sit in one sentence of its adjacent comment block that also refers to runtime text.
            ok = bool(UNVERIFIED_WORDING.search(normalize(entry['statement'])))
            if not ok:
                ok = any(UNVERIFIED_WORDING.search(sentence) and REFERS_TO_RUNTIME_TEXT.search(sentence)
                         for st in statements.get(key, []) if st.kind == 'string' for sentence in _sentences(st.block[2]))
            if not ok:
                problems.append(f'R5 {label}: unverified contract lacks explicit unverified wording')

    # R6
    for path, counts in present.items():
        for (symbol, fp), count in counts.items():
            if covered[path][(symbol, fp)] != count:
                st = statements[(path, symbol, fp)][0]
                problems.append(f'R6 {path}:{st.line}: uncovered claim at {symbol!r} ({fp}, '
                                f'{count} present, {covered[path][(symbol, fp)]} covered): {st.text[:100]}')

    # R7 (and the R8 integrity of each forbidden-phrase rule)
    for number, rule in enumerate(ledger.get('forbiddenPhrases', []), 1):
        name = f'forbidden-phrase rule {number} ({rule.get("pattern", "")[:60]!r})'
        pattern = re.compile(rule['pattern'])
        explicit_screen = isinstance(rule.get('screen'), str) and bool(rule['screen'].strip())
        if 'screen' in rule and not explicit_screen:
            problems.append(f'R8 {name}: screen must be a non-empty string (an empty screen is not a screen)')
        if not explicit_screen and SCREEN_NEEDED.search(rule['pattern']):
            problems.append(f'R8 {name}: a pattern with an anchor (^, $, \\A, \\Z) or a lookaround needs an explicit '
                            'keyword screen')
        try:
            screen = re.compile(rule['screen'] if explicit_screen else rule['pattern'])
        except re.error as error:
            problems.append(f'R8 {name}: screen does not compile ({error})')
            screen, explicit_screen = pattern, False
        fixtures = rule.get('fixtures') if isinstance(rule.get('fixtures'), dict) else {}
        if 'fixtures' in rule and not isinstance(rule['fixtures'], dict):
            problems.append(f'R8 {name}: fixtures must be an object with "fail" (and optionally "pass") lists')
        for kind in ('fail', 'pass'):
            if kind in fixtures and not isinstance(fixtures[kind], list):
                problems.append(f'R8 {name}: fixtures.{kind} must be a list of strings')
                fixtures = {**fixtures, kind: []}
            bad = [each for each in fixtures.get(kind, []) if not isinstance(each, str)]
            if bad:
                problems.append(f'R8 {name}: fixtures.{kind} has non-string entries {bad[:3]!r}')
                fixtures = {**fixtures, kind: [each for each in fixtures[kind] if isinstance(each, str)]}
        if not fixtures.get('fail'):
            problems.append(f'R8 {name}: needs at least one must-fail fixture (fixtures.fail)')
        for text in fixtures.get('fail', []):
            matching = neutralize(normalize(text))
            if not pattern.search(matching):
                problems.append(f'R8 {name}: must-fail fixture {text!r} does not match the pattern')
            elif not screen.search(matching):
                problems.append(f'R8 {name}: the screen rejects must-fail fixture {text!r} (the screen would disable the rule)')
        for text in fixtures.get('pass', []):
            if pattern.search(neutralize(normalize(text))):
                problems.append(f'R8 {name}: must-pass fixture {text!r} matches the pattern')
        if not rule.get('paths'):
            problems.append(f'R8 {name}: paths is missing or empty (the rule would apply to no file)')
        targets = set()
        for each in rule.get('paths') or []:
            matched = repo.glob(each)
            if not matched:
                problems.append(f'R8 {name}: path {each!r} matches no file (stale rule)')
            targets.update(matched)
        comment_only = set()
        for each in rule.get('commentOnlyPaths') or []:
            matched = repo.glob(each)
            if not matched:
                problems.append(f'R8 {name}: commentOnlyPaths entry {each!r} matches no file (stale rule)')
            comment_only.update(matched)
        comment_only -= targets
        for text in fixtures.get('passLiteral', []):
            if not rule.get('commentOnlyPaths'):
                problems.append(f'R8 {name}: fixtures.passLiteral needs commentOnlyPaths')
                break
            if not isinstance(text, str) or not pattern.search(neutralize(normalize(text))):
                problems.append(f'R8 {name}: passLiteral fixture {text!r} does not match the pattern, so it does not test the '
                                'literal scope')
        exempt = {}
        for item in rule.get('exempt', []):
            if not (isinstance(item, dict) and item.get('path') and re.fullmatch(r'[0-9a-f]{16}', str(item.get('fingerprint')))
                    and item.get('owner') and item.get('reason')):
                problems.append(f'R8 {name}: an exemption needs path, 16-hex fingerprint, owner and reason')
                continue
            exempt[(item['path'], item['fingerprint'])] = False
        for path in sorted(targets | comment_only):
            if path not in files and not path.endswith('.md') and not screen.search(_prefilter_text(repo.read(path))):
                continue  # cheap whole-file screen for Swift; only files that might match are lexed (Markdown is always parsed)
            parsed = files.get(path) or parse(repo, path, exclude)
            for st in parsed.statements:
                if path in comment_only and st.kind != 'comment':
                    continue   # comment-only path: string literals (runtime text, fixtures) are not checked
                if explicit_screen and not screen.search(st.match):
                    continue   # the screen is a necessary condition (validated against fixtures.fail above)
                if pattern.search(st.match):
                    if (path, st.fp) in exempt:
                        exempt[(path, st.fp)] = True
                        continue
                    problems.append(f'R7 {path}:{st.line}: forbidden phrase {rule["pattern"][:80]!r} ({rule["reason"]}): '
                                    f'{st.text[:100]}')
        for (path, fp), used in exempt.items():
            if not used:
                problems.append(f'R7 {name}: exemption {path} {fp} matches no present statement (stale exemption)')

    # Q17 recorded sites: lexical count of a pattern in the masked body of a symbol.
    for site in ledger.get('recordedSites', []):
        path = site['path']
        if not repo.exists(path):
            problems.append(f'Q17 {site["id"]}: recorded-site file {path} missing')
            continue
        parsed = files.get(path) or parse(repo, path, exclude)
        count = count_in_symbol(parsed, site['symbol'], site['pattern'])
        if count is None:
            problems.append(f'Q17 {site["id"]}: recorded-site symbol {site["symbol"]!r} not in {path}')
        elif count != site['count']:
            problems.append(f'Q17 {site["id"]}: recorded-site count {site["count"]} != {count} actual')

    # Q14 caller inventory: every use of a flag-gated helper outside its defining file, by file.
    for item in ledger.get('callerInventory', []):
        symbol, defined = item['symbol'], item['definedIn']
        word = re.compile(rf'(?<![\w]){re.escape(symbol)}(?![\w])')
        if not repo.exists(defined) or not re.search(rf'\b(?:func|var|let)\s+{re.escape(symbol)}\b', masked(repo.read(defined))):
            problems.append(f'Q14 {item["id"]}: {symbol} is not declared in {defined}')
        else:
            own = len(word.findall(masked(repo.read(defined))))
            if not isinstance(item.get('definedInOccurrences'), int):
                problems.append(f'Q14 {item["id"]}: definedInOccurrences (uses inside {defined}, declaration included) is missing')
            elif own != item['definedInOccurrences']:
                problems.append(f'Q14 {item["id"]}: occurrences of {symbol} in its defining file changed: ledger '
                                f'{item["definedInOccurrences"]}, actual {own}; re-derive the reachability of the gated path '
                                'before updating the inventory')
        actual = {}
        for path in sorted({path for glob in item['globs'] for path in repo.glob(glob)}):
            text = repo.read(path)
            if path == defined or symbol not in text:
                continue
            count = len(word.findall(masked(text)))
            if count:
                actual[path] = count
        if actual != item['callers']:
            problems.append(f'Q14 {item["id"]}: callers of {symbol} changed: ledger {item["callers"]}, actual {actual}; '
                            're-derive the reachability of the gated path before updating the inventory')
    return problems


_PREFILTER = {}


def _prefilter_text(text):
    """Comment markers and `" + "` literal joins removed, then normalized: a cheap superset screen for R7 (memoized by
    content)."""
    key = hashlib.sha256(text.encode('utf-8')).hexdigest()
    if key not in _PREFILTER:
        _PREFILTER[key] = neutralize(normalize(re.sub(r'"\s*\+\s*"', '', re.sub(r'(?m)^\s*(?:/{2,3}|\*)', ' ', text))))
    return _PREFILTER[key]


def count_in_symbol(parsed, symbol, pattern):
    """Matches of `pattern` in the masked text of every scope whose chain equals `symbol`."""
    total, seen = 0, False
    rx = re.compile(pattern)
    for scope in range(1, len(parsed.parent)):
        if '::'.join(parsed.chain(scope)) == symbol and parsed.label[scope]:
            seen = True
            total += len(rx.findall(parsed.lexed.mask[parsed.opens[scope]:parsed.closes[scope]]))
    return total if seen else None


def inventory(repo, paths, vocabulary, exclude, everything=False):
    """[(path, line, symbol, fp, kind, text)] of tier-A (or all) statements."""
    rows = []
    for path in paths:
        for st in parse(repo, path, exclude).statements:
            if everything or tier_a(st.match, vocabulary):
                rows.append({'path': path, 'line': st.line, 'symbol': st.symbol, 'fingerprint': st.fp,
                             'kind': st.kind, 'statement': st.text})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--ledger', default=LEDGER)
    parser.add_argument('--list', nargs='*', metavar='PATH', help='print tier-A statements as JSON lines')
    parser.add_argument('--all', action='store_true', help='with --list: every statement, not only tier A')
    args = parser.parse_args(argv)
    repo = Repo(args.root)
    if args.list is not None:
        ledger, vocabulary, exclude = load(Repo(ROOT), LEDGER) if not repo.exists(args.ledger) else load(repo, args.ledger)
        scopes = section_scopes(ledger, [])
        paths = args.list or [region['path'] for region in ledger['regions']] + list(scopes)
        for row in inventory(repo, paths, vocabulary, exclude, args.all):
            if in_scope(scopes, row['path'], row['symbol']) or args.all:
                print(json.dumps(row, ensure_ascii=False))
        return 0
    problems = check(repo, args.ledger)
    for problem in problems:
        print(problem)
    print(f'claim ledger: {len(problems)} problem(s)')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
