"""Policy (P02 phase C): lexical pins for Operations.findNearestBatch's CPU strategy and the explicit Gram route.

WHY: before P02, `findNearestBatch` switched to a GEMM route by itself at >= 8 queries and >= 256 candidates and ignored
`BatchOperations.Configuration.enableMatrixRouting`, so callers could neither see nor turn off a change of numerical
method (root ruling RD3). After P02 the Gram route runs only for `cpuStrategy: .gram(...)`, and the batch search reads
no BatchOperations setting at all. The Gram route also has class rules from the P02 brief that a lexical check can pin
(phase 1 check map S6, T24/T25, T44, T46): it must not rescore candidates with a per-pair kernel or `metric.distance`
(no rerank-and-claim-equivalence), must not build sort, pair or extractor arrays outside its counted payload, must not
slice its inputs, and must not suspend (no pointer may cross an `await`).

CHECKS, on `lex(text)`: comments and literal string text blanked, the code inside string interpolations KEPT, and
backtick escapes turned into spaces (Scripts/ci/swift_lexing._mask with keep_interpolations=True, and _unbacktick; P02
final review FA-2, FA-3), so
a call written inside `"\\(…)"` and an escaped name such as `` `gramBatch` `` are seen as code and as the plain name:
  C1 Operations.swift and GramBatchSearch.swift contain none of `BatchOperations`, `_configuration`,
     `enableMatrixRouting`, `matrixRoutingMinN`, `configuration(` (RD3: the batch search reads no BatchOperations
     setting; the behavioural check is P02GramRouteTests.matrixRoutingFlagDoesNotAffectBatchSearch, child-isolated);
  C2 Operations.swift reaches the Gram route only through `case let .gram(...)` of `switch cpuStrategy` (phase C
     review m9, phase D-F review A-2 and final review FA-2, the class "Gram search without `.gram`"):
     - structure: EXACTLY ONE `gramBatch` token in any use (a call, an unapplied reference such as
       `let f = gramBatch`, a qualified `Operations.gramBatch`, a backtick-escaped `` `gramBatch` ``, or any of these
       inside a string interpolation), and it is the call that is the first statement of the one
       `case let .gram(...):` label (optionally `return` and `try` before it; an escaped spelling of that one call is
       the same call); exactly one `.gram` token, escapes included (that label: no `.gram` strategy is built in this
       file) and exactly one `switch cpuStrategy {`. So a size gate in this file that calls or references `gramBatch`
       by any of those spellings (for example `queries.count > 7 && vectors.count > 255`,
       `` return try `gramBatch`(...) ``, or a call in a closure inside `"\\(…)"`) adds a second token and fails;
     - tokens: no `queries.count >=` / `vectors.count >=` comparison, no `gemmFindNearestBatch` / `gemmBatchRun` (the
       deleted implicit route), and no `gramSearch`, `gemmCrossTerm`, `MatrixDistance` or `cblas_` (the Gram tile code
       reached around `gramBatch`);
  C3 GramBatchSearch.swift calls no per-pair or metric kernel: no `.distance(`, `EuclideanKernels`, `CosineKernels`,
     `DotKernels`, `BatchKernels`, `metric.` (S6);
  C4 GramBatchSearch.swift builds no sort or extractor: no `sort(`, `sorted(`, `TopKSelection`, `extractSortedResult`,
     `toTuples`, `toResults` (T24/T25; selection is `TopKBuffer.drainWorstFirst`);
  C5 GramBatchSearch.swift slices nothing with a range subscript: no `[... ..< ...]` or `[... ... ...]` inside square
     brackets (T44);
  C6 GramBatchSearch.swift has no `await` and no `async` (T46: every scratch borrow is synchronous);
  C7 the body of `gemmPairwiseTyped` in BatchOperations.swift (the pairwise GEMM helper, P02 phase D, RD10; check map
     R1/R5/R6) builds no flat N·N buffer: no `candidates:` argument (the full-matrix `MatrixDistance` entries, allocating
     or `into:`, all take one) and no `n * n` product; it calls `MatrixDistance.prepare(` exactly once and a prepared
     entry (`prepared:`) at least once. The body must exist.
  C8 the hops of the Gram BLAS call that the trace seam cannot observe (P02 phase D-F review A-1, the second instance
     of the class "the seam claims to observe the BLAS call", after phase C review RA-1). The seam records the
     `MatrixDistance.GemmArguments` binding, so:
     - MatrixDistance.swift has exactly one `cblas_` token, a `cblas_sgemm(` call inside the body of the
       `func gemmCrossTerm` overload that takes `arguments: GemmArguments` (the private array wrapper of the same name
       builds a `GemmArguments` and calls it), and its argument list (whitespace removed) is exactly
       `CblasRowMajor, CblasNoTrans, CblasTrans, arguments.m, arguments.n, arguments.k, alpha, x, arguments.lda, y,
       arguments.ldb, 0, out, arguments.ldc` (the six fields in BLAS order, beta 0);
     - GramBatchSearch.swift has no `cblas_` token, at most one `gemmCrossTerm(` call, at most one
       `gramTileRecord(` call and at most one binding `let blas = plan.gemmArguments(` (final review C-M3: the
       arguments come from the plan's checked Int32 fields); every `gemmCrossTerm(` call passes `arguments: blas`,
       every `gramTileRecord(` call passes `blas: blas`, and the record copies each field as `f: Int(blas.f)`;
     - and EVERY `blas` token in that file is one of: that binding, the `arguments: blas` of the call, the two tokens
       of `blas: blas` in the record call, the `blas: MatrixDistance.GemmArguments` parameter of `func
       gramTileRecord`, or a field read `blas.m` … `blas.ldc` (final review FA-1). Any other binding of the name (a
       `var`, a second `let` in a nested scope, a `for … in` pattern, a closure parameter, a `case let` pattern)
       fails, so the value the call receives is the value the record copies (non-vacuity: the real file has exactly
       one call, one binding, one record call and 11 `blas` tokens).
Each check has must-fail and must-pass fixtures below; the real tree must pass. Filesystem reads only (git-less safe).

LIMITS (lexical, not a parser): a configuration read through another name (an alias, a helper in another file) is not
seen by C1. C2 sees only Operations.swift: a function in ANOTHER file that calls `gramBatch` or the tile code, or that
returns a `.gram` strategy which Operations then switches on, is not seen; its structural rule needs the `case let
.gram(...)` label on one line and recognises only `return`/`try` before the call (any other spelling of the one
admitted call fails, which is safe); its token rule sees only the listed spellings. C8 compares one argument list
textually: a field passed through a local alias inside `gemmCrossTerm` or a second BLAS wrapper in another file is not
seen, and its `blas` accounting is by name in GramBatchSearch.swift only (a value passed through a differently named
binding to a call that still spells `arguments: blas` cannot exist, since that call must name `blas`). The lexer does
not lex regex literals (`/…/`) and does not nest `/* */` comments. The behavioural backstop for a re-added size gate is the default-route suite at Q7/8 x N255/256
(P02DefaultRouteRedTests, P02PerQueryRouteTests), which a gate at a threshold above those shapes would evade without C2.
C3 does not see a rescoring call hidden behind a helper in another file or a closure parameter named otherwise; C4 sees
only the listed names; C5 sees only a range operator inside one pair of square brackets on one line; C6 is a token
check; C7 reads only the body of the function named `gemmPairwiseTyped` (a flat buffer built in a helper it calls, or
a product spelled other than `n * n`, is not seen; the behavioural pin is
P02PairwiseRowTests.rowsEqualPreparedIntoPerRow). Comments and literal string text are blanked first, so documentation
may name these symbols; code inside an interpolation is not documentation and is checked.
"""
import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location('swift_lexing', ROOT / 'Scripts/ci/swift_lexing.py')
lexing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lexing)

OPERATIONS = 'Sources/VectorCore/Operations/Operations.swift'
GRAM = 'Sources/VectorCore/Operations/GramBatchSearch.swift'
BATCH = 'Sources/VectorCore/Operations/BatchOperations.swift'

C1 = re.compile(r'\bBatchOperations\b|\b_configuration\b|\benableMatrixRouting\b|\bmatrixRoutingMinN\b|\bconfiguration\(')
C2 = re.compile(r'\b(?:queries|vectors)\.count\s*>=|\bgemmFindNearestBatch\b|\bgemmBatchRun\b|\bgramSearch\b'
                r'|\bgemmCrossTerm\b|\bMatrixDistance\b|\bcblas_')
C2_CALL = re.compile(r'\bgramBatch\s*\(')
# Phase D-F review A-2: every use of the identifier counts (call, reference, qualified, backtick-escaped).
C2_IDENT = re.compile(r'\bgramBatch\b')   # on lex() text: backtick escapes are spaces
C2_CASE_CALL = re.compile(r'\bcase\s+let\s+\.gram\s*\([^()\n]*\)\s*:\s*(?:return\s+)?(?:try\s+)?(?P<call>gramBatch)\s*\(')
C2_GRAM_TOKEN = re.compile(r'\.\s*`?gram\b`?')
C2_SWITCH = re.compile(r'\bswitch\s+cpuStrategy\s*\{')
C3 = re.compile(r'\.distance\(|\bEuclideanKernels\b|\bCosineKernels\b|\bDotKernels\b|\bBatchKernels\b|\bmetric\.')
C4 = re.compile(r'\bsort(?:ed)?\s*[({]|\bTopKSelection\b|\bextractSortedResult\b|\btoTuples\b|\btoResults\b')
C5 = re.compile(r'\[[^\[\]\n]*(?:\.\.<|\.\.\.)[^\[\]\n]*\]')
C6 = re.compile(r'\bawait\b|\basync\b')
C7_FUNC = re.compile(r'\bfunc\s+gemmPairwiseTyped\b')
C7 = re.compile(r'\bcandidates\s*:|\bn\s*\*\s*n\b')
C7_PREPARE = re.compile(r'\bMatrixDistance\s*\.\s*prepare\s*\(')
C7_PREPARED = re.compile(r'\bprepared\s*:')
MATRIX = 'Sources/VectorCore/Operations/MatrixDistance.swift'
C8_CBLAS = re.compile(r'\bcblas_\w*')
C8_SGEMM = re.compile(r'\bcblas_sgemm\s*\(')
C8_FUNC = re.compile(r'\bfunc\s+gemmCrossTerm\b')
C8_SIGNATURE = re.compile(r'\barguments\s*:\s*GemmArguments\b')
C8_ARGUMENTS = ('CblasRowMajor,CblasNoTrans,CblasTrans,arguments.m,arguments.n,arguments.k,alpha,x,arguments.lda,y,'
                'arguments.ldb,0,out,arguments.ldc')
C8_CROSS = re.compile(r'\bgemmCrossTerm\s*\(')
C8_BLAS = re.compile(r'\bblas\b')
# Final review C-M3: the one binding is built by the plan, from its checked Int32 fields.
C8_BINDING_VALUE = re.compile(r'\blet\s+(blas)\s*=\s*plan\s*\.\s*gemmArguments\s*\(')
C8_PARAMETER = re.compile(r'(?<![\w.])(blas)\s*:\s*MatrixDistance\s*\.\s*GemmArguments\b')
C8_READ = re.compile(r'\b(blas)\s*\.\s*(\w+)')
C8_RECORD = re.compile(r'\bgramTileRecord\s*\(')
C8_FIELDS = ('m', 'n', 'k', 'lda', 'ldb', 'ldc')


def _line(masked, offset):
    return masked.count('\n', 0, offset) + 1


def _c2_structure(path, masked):
    """C2's structural rule for Operations.swift (see the module docstring)."""
    found = []
    uses = list(C2_IDENT.finditer(masked))
    admitted = {match.start('call') for match in C2_CASE_CALL.finditer(masked)}
    if not uses:
        found.append(f'C2 {path}:1: no gramBatch( call inside `case let .gram(...)`')
    for match in uses:
        if match.start() not in admitted:
            found.append(f'C2 {path}:{_line(masked, match.start())}: `{match.group(0)}` outside the admitted '
                         f'`case let .gram(...): return try gramBatch(` call')
    if len(uses) > 1:
        found.append(f'C2 {path}:{_line(masked, uses[1].start())}: {len(uses)} gramBatch tokens (exactly one allowed)')
    tokens = list(C2_GRAM_TOKEN.finditer(masked))
    if len(tokens) != 1:
        where = _line(masked, tokens[1].start()) if len(tokens) > 1 else 1
        found.append(f'C2 {path}:{where}: {len(tokens)} `.gram` tokens (exactly one, the case label, allowed)')
    switches = list(C2_SWITCH.finditer(masked))
    if len(switches) != 1:
        found.append(f'C2 {path}:1: {len(switches)} `switch cpuStrategy {{` (exactly one allowed)')
    return found


def _body(masked, start):
    """(offset, text) of the brace-delimited body that opens after `start`, or None."""
    open_at = masked.find('{', start)
    if open_at < 0:
        return None
    depth = 0
    for i in range(open_at, len(masked)):
        depth += {'{': 1, '}': -1}.get(masked[i], 0)
        if depth == 0:
            return open_at, masked[open_at:i + 1]
    return None


def _c7(path, masked):
    """C7 for BatchOperations.swift (see the module docstring)."""
    found = []
    heads = list(C7_FUNC.finditer(masked))
    body = _body(masked, heads[0].end()) if len(heads) == 1 else None
    if body is None:
        return [f'C7 {path}:1: {len(heads)} `func gemmPairwiseTyped` with a body (exactly one expected)']
    offset, text = body
    for match in C7.finditer(text):
        found.append(f'C7 {path}:{_line(masked, offset + match.start())}: {match.group(0).strip()}')
    if len(C7_PREPARE.findall(text)) != 1:
        found.append(f'C7 {path}:{_line(masked, offset)}: MatrixDistance.prepare( is not called exactly once')
    if not C7_PREPARED.search(text):
        found.append(f'C7 {path}:{_line(masked, offset)}: no prepared/into call')
    return found


def _call_arguments(masked, match):
    """Text between the `(` that ends `match` and its balanced `)`, or None."""
    open_at = match.end() - 1
    depth = 0
    for i in range(open_at, len(masked)):
        depth += {'(': 1, ')': -1}.get(masked[i], 0)
        if depth == 0:
            return masked[open_at + 1:i]
    return None


def _calls(pattern, masked):
    """Matches of `pattern` that are uses, not the `func name(` declaration."""
    return [match for match in pattern.finditer(masked) if not masked[:match.start()].rstrip().endswith('func')]


def _c8_matrix(path, masked):
    """C8 for MatrixDistance.swift: the one BLAS call forwards the GemmArguments fields in BLAS order."""
    found = []
    tokens = list(C8_CBLAS.finditer(masked))
    calls = list(C8_SGEMM.finditer(masked))
    if len(tokens) != 1 or len(calls) != 1:
        where = _line(masked, tokens[1].start()) if len(tokens) > 1 else 1
        return [f'C8 {path}:{where}: {len(tokens)} cblas_ tokens and {len(calls)} cblas_sgemm( calls (exactly one)']
    call = calls[0]
    inside = False
    for head in C8_FUNC.finditer(masked):   # the pointer overload; the private array wrapper has the same name
        body = _body(masked, head.end())
        if body and C8_SIGNATURE.search(masked, head.end(), body[0]) and body[0] < call.start() < body[0] + len(body[1]):
            inside = True
    if not inside:
        found.append(f'C8 {path}:{_line(masked, call.start())}: cblas_sgemm( is not inside the `func gemmCrossTerm` '
                     f'that takes `arguments: GemmArguments`')
    arguments = _call_arguments(masked, call)
    if arguments is None or re.sub(r'\s+', '', arguments) != C8_ARGUMENTS:
        shown = ' '.join((arguments or '').split())
        found.append(f'C8 {path}:{_line(masked, call.start())}: cblas_sgemm arguments are not the GemmArguments '
                     f'fields in BLAS order: {shown}')
    return found


def _c8_gram(path, masked):
    """C8 for GramBatchSearch.swift: the BLAS call and the trace record receive the same `blas` binding, and the name
    `blas` has no other binding or use in the file (final review FA-1)."""
    found = []
    for match in C8_CBLAS.finditer(masked):
        found.append(f'C8 {path}:{_line(masked, match.start())}: {match.group(0)} (BLAS only through gemmCrossTerm)')
    crosses, records = _calls(C8_CROSS, masked), _calls(C8_RECORD, masked)
    bindings = list(C8_BINDING_VALUE.finditer(masked))
    if len(crosses) > 1 or len(records) > 1 or len(bindings) > 1:
        found.append(f'C8 {path}:1: {len(crosses)} gemmCrossTerm( calls, {len(records)} gramTileRecord( calls and '
                     f'{len(bindings)} `let blas = plan.gemmArguments(` bindings (at most one each)')
    admitted = {match.start(1) for match in bindings}
    for pattern, label in ((C8_CROSS, 'arguments'), (C8_RECORD, 'blas')):
        for match in _calls(pattern, masked):
            arguments = _call_arguments(masked, match) or ''
            hits = list(re.finditer(r'(?<![\w.])(' + label + r')\s*:\s*(blas)(?![\w.(])', arguments))
            if not hits:
                found.append(f'C8 {path}:{_line(masked, match.start())}: {match.group(0)} without `{label}: blas`')
            for hit in hits:
                admitted.add(match.end() + hit.start(2))
                if label == 'blas':
                    admitted.add(match.end() + hit.start(1))
    for head in re.finditer(r'\bfunc\s+gramTileRecord\b', masked):
        brace = masked.find('{', head.end())
        for hit in C8_PARAMETER.finditer(masked, head.end(), brace if brace >= 0 else len(masked)):
            admitted.add(hit.start(1))
    for hit in C8_READ.finditer(masked):
        if hit.group(2) in C8_FIELDS:
            admitted.add(hit.start(1))
    for match in C8_BLAS.finditer(masked):
        if match.start() not in admitted:
            found.append(f'C8 {path}:{_line(masked, match.start())}: `blas` that is not the one plan binding, its two '
                         f'uses, the record parameter or a field read (a rebinding: var, let, for-in, closure parameter '
                         f'or pattern)')
    for match in re.finditer(r'\b(\w+)\s*:\s*Int\s*\(\s*blas\s*\.\s*(\w+)\s*\)', masked):
        if match.group(1) != match.group(2) or match.group(1) not in C8_FIELDS:
            found.append(f'C8 {path}:{_line(masked, match.start())}: record field {match.group(0)}')
    return found


def lex(text):
    """The text every rule reads: comments and literal string text blanked, interpolated code kept, backtick escapes
    turned into spaces (Scripts/ci/swift_lexing._mask with keep_interpolations=True, and _unbacktick; final review
    FA-2, FA-3)."""
    return lexing._unbacktick(lexing._mask(text, keep_interpolations=True))


def problems(path, text):
    """Rule violations in `text` (the contents of `path`), as 'Cn path:line: token'."""
    masked = lex(text)
    found = []

    def scan(rule, pattern):
        for match in pattern.finditer(masked):
            found.append(f'{rule} {path}:{_line(masked, match.start())}: {match.group(0).strip()}')

    if path in (OPERATIONS, GRAM):
        scan('C1', C1)
    if path == OPERATIONS:
        scan('C2', C2)
        found += _c2_structure(path, masked)
    if path == BATCH:
        found += _c7(path, masked)
    if path == GRAM:
        for rule, pattern in (('C3', C3), ('C4', C4), ('C5', C5), ('C6', C6)):
            scan(rule, pattern)
        found += _c8_gram(path, masked)
    if path == MATRIX:
        found += _c8_matrix(path, masked)
    return found


# The admitted shape of the route switch (the real Operations.swift form). Operations fixtures are snippets placed
# before it, so each one fails or passes for its own text only.
SWITCH = ('        switch cpuStrategy {\n'
          '        case .perQuery:\n'
          '            return try await computeProvider.parallelExecute(items: 0..<queries.count) { i in try await f(i) }\n'
          '        case let .gram(queryTileRows, candidateTileRows, maximumTemporaryPayloadBytes):\n'
          '            return try gramBatch(queries: queries, candidates: vectors, k: k, metric: metric,\n'
          '                                 queryTileRows: queryTileRows, candidateTileRows: candidateTileRows,\n'
          '                                 maximumTemporaryPayloadBytes: maximumTemporaryPayloadBytes)\n'
          '        }\n')
GATE_CALL = ('gramBatch(queries: queries, candidates: vectors, k: k, metric: metric, queryTileRows: 8, '
             'candidateTileRows: 256, maximumTemporaryPayloadBytes: .max)')
# Phase C review m9: the old gate respelled with strict comparisons, routing to the Gram route.
RESPELLED_GATE = ('        let batchIsLarge = queries.count > 7 && vectors.count > 255\n'
                  f'        if batchIsLarge, let routed = try? {GATE_CALL} {{ return routed }}\n')


GATE_ARGUMENTS = GATE_CALL[len('gramBatch'):]
# Phase D-F review A-2: E1, a backtick-escaped call under a size gate (it compiles), and E3, an unapplied reference
# called under a size gate.
BACKTICK_GATE = f'        if queries.count > 63, vectors.count > 1023 {{ return try `gramBatch`{GATE_ARGUMENTS} }}\n'
# Final review FA-2: the call inside a string interpolation (it compiles; review A's mutant).
INTERPOLATED_GATE = ('        if queries.count > 63, vectors.count > 1023 {\n'
                     '            var routed: [[NearestNeighborResult]]?\n'
                     f'            _ = "\\({{ () -> Int in routed = try? {GATE_CALL}; return 0 }}())"\n'
                     '            if let routed { return routed }\n'
                     '        }\n')
REFERENCE_GATE = ('        let viaGram: ([V], [V], Int, M, Int, Int, Int) throws -> [[NearestNeighborResult]] = gramBatch\n'
                  '        if queries.count > 63 { return try viaGram(queries, vectors, k, metric, 8, 256, .max) }\n')


def ops(snippet=''):
    return snippet + SWITCH


# C8: the real shape of gemmCrossTerm (MatrixDistance.swift) and of the Gram tile's BLAS call and record
# (GramBatchSearch.swift).
CROSS = ('    internal static func gemmCrossTerm(\n'
         '        x: UnsafePointer<Float>, y: UnsafePointer<Float>, arguments: GemmArguments, alpha: Float,\n'
         '        into out: UnsafeMutablePointer<Float>\n'
         '    ) {\n'
         '        cblas_sgemm(\n'
         '            CblasRowMajor, CblasNoTrans, CblasTrans,\n'
         '            arguments.m, arguments.n, arguments.k,\n'
         '            alpha, x, arguments.lda,\n'
         '            y, arguments.ldb,\n'
         '            0, out, arguments.ldc\n'
         '        )\n'
         '    }\n')
TILE = ('                let blas = plan.gemmArguments(activeQueryRows: m, activeCandidateRows: n)\n'
        '                MatrixDistance.gemmCrossTerm(x: scratch.queryPack, y: scratch.candidatePack, arguments: blas,\n'
        '                                             alpha: euclidean ? -2 : 1, into: scratch.scores)\n'
        '                trace(.tile(gramTileRecord(scratch, before: before, queryTile: queryTile,\n'
        '                                           candidateTile: candidateTile, queryRows: queryRows,\n'
        '                                           candidateRows: candidateRows, blas: blas, ldc: ldc)))\n'
        '        return GramTileRecord(queryTile: queryTile, candidateTile: candidateTile, queryRows: queryRows,\n'
        '                              candidateRows: candidateRows, m: Int(blas.m), n: Int(blas.n), k: Int(blas.k),\n'
        '                              lda: Int(blas.lda), ldb: Int(blas.ldb), ldc: Int(blas.ldc),\n'
        '                              scores: scores, scoreTileElements: count, paddingSlotsChanged: changed)\n')
CROSS_CALL = ('                MatrixDistance.gemmCrossTerm(x: scratch.queryPack, y: scratch.candidatePack, arguments: blas,\n'
              '                                             alpha: euclidean ? -2 : 1, into: scratch.scores)\n')
OTHER_SGEMM = ('    static func other(x: UnsafePointer<Float>, y: UnsafePointer<Float>, out: UnsafeMutablePointer<Float>) {\n'
               '        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, 1, 1, 1, x, 1, y, 1, 0, out, 1)\n'
               '    }\n')
# Phase D-F review A-1's mutant A-RA1-callsite-n: BLAS N is the score-tile stride, changed at the call site only.
SWAPPED_N = CROSS.replace('arguments.m, arguments.n, arguments.k,', 'arguments.m, arguments.ldc, arguments.k,')


# C7: the per-row pairwise helper (the real shape) and the phase C reshape it replaced.
PAIRWISE = ('    private static func gemmPairwiseTyped<O: UnifiedVectorBuffer>(_ vs: [O], euclid: Bool) -> [[Float]] {\n'
            '        let n = vs.count\n'
            '        let prepared = MatrixDistance.prepare(vs, normalized: !euclid)\n'
            '        var rows = [[Float]]()\n'
            '        for i in 0..<n {\n'
            '            var row = [Float](repeating: 0, count: n)\n'
            '            if euclid {\n'
            '                MatrixDistance.euclideanSquaredMatrix(queries: [vs[i]], prepared: prepared, into: &row)\n'
            '            } else {\n'
            '                MatrixDistance.cosineDistanceMatrix(queries: [vs[i]], prepared: prepared, into: &row)\n'
            '            }\n'
            '            rows.append(row)\n'
            '        }\n'
            '        return rows\n'
            '    }\n')
OLD_PAIRWISE = ('    private static func gemmPairwiseTyped<O: UnifiedVectorBuffer>(_ vs: [O], euclid: Bool) -> [[Float]] {\n'
                '        let n = vs.count\n'
                '        let flat = euclid\n'
                '            ? MatrixDistance.euclideanSquaredMatrix(queries: vs, candidates: vs)\n'
                '            : MatrixDistance.cosineDistanceMatrix(queries: vs, candidates: vs)\n'
                '        var rows = [[Float]]()\n'
                '        for i in 0..<n {\n'
                '            var row = [Float](repeating: 0, count: n)\n'
                '            let base = i * n\n'
                '            for j in 0..<n { row[j] = flat[base + j] }\n'
                '            rows.append(row)\n'
                '        }\n'
                '        return rows\n'
                '    }\n')


class Fixtures(unittest.TestCase):
    MUST_FAIL = [
        (OPERATIONS, ops('let config = await BatchOperations.configuration()\n'), 'C1'),
        (OPERATIONS, ops('if _configuration.get().enableMatrixRouting { }\n'), 'C1'),
        (GRAM, 'guard n >= config.matrixRoutingMinN else { return }\n', 'C1'),
        (OPERATIONS, ops('guard queries.count >= 8, vectors.count >= 256 else { return nil }\n'), 'C2'),
        (OPERATIONS, ops('if let routed = gemmFindNearestBatch(queries: q, in: v, k: k, metric: m) { return routed }\n'),
         'C2'),
        (OPERATIONS, ops(RESPELLED_GATE), 'C2'),
        (OPERATIONS, ops(f'if ProcessInfo.processInfo.environment["GRAM"] != nil {{ return try {GATE_CALL} }}\n'), 'C2'),
        (OPERATIONS, ops('let rows = try gramSearch(queries: q, candidates: c, plan: plan)\n'), 'C2'),
        (OPERATIONS, ops('MatrixDistance.gemmCrossTerm(x: x, y: y, arguments: a, alpha: -2, into: out)\n'), 'C2'),
        (OPERATIONS, ops('let strategy = queries.count > 63 ? BatchSearchCPUStrategy.gram(queryTileRows: 8, '
                         'candidateTileRows: 256, maximumTemporaryPayloadBytes: .max) : cpuStrategy\n'), 'C2'),
        (OPERATIONS, SWITCH.replace('return try gramBatch(', 'return try perQueryBatch('), 'C2'),
        (OPERATIONS, SWITCH.replace('case .perQuery:\n', f'case .perQuery:\n            if k > 99 {{ return try {GATE_CALL} }}\n'),
         'C2'),
        (OPERATIONS, ops('switch cpuStrategy { default: break }\n'), 'C2'),
        (GRAM, 'let exact = metric.distance(queries[i], candidates[j])\n', 'C3'),
        (GRAM, 'let s = EuclideanKernels.squared512(a, b)\n', 'C3'),
        (GRAM, 'let row = pairs.sorted { $0.1 < $1.1 }\n', 'C4'),
        (GRAM, 'let sel = TopKSelection.select(k: k, from: base, count: n, ids: nil, tieBreaker: .smallerIndex)\n', 'C4'),
        (GRAM, 'let tile = Array(candidates[lower..<upper])\n', 'C5'),
        (GRAM, 'let tile = candidates[lower...upper]\n', 'C5'),
        (GRAM, 'try await Task.yield()\n', 'C6'),
        (GRAM, 'static func gramSearch() async throws {}\n', 'C6'),
        (BATCH, OLD_PAIRWISE, 'C7'),
        (BATCH, PAIRWISE.replace('        var rows = [[Float]]()\n',
                                 '        var flat = [Float](repeating: 0, count: n * n)\n'
                                 '        MatrixDistance.euclideanSquaredMatrix(queries: vs, prepared: prepared, into: &flat)\n'
                                 '        var rows = [[Float]]()\n'), 'C7'),
        (BATCH, PAIRWISE.replace('            var row = [Float](repeating: 0, count: n)\n',
                                 '            var row = [Float](repeating: 0, count: n)\n'
                                 '            let again = MatrixDistance.prepare(vs, normalized: !euclid)\n'), 'C7'),
        (BATCH, PAIRWISE.replace('prepared: prepared', 'candidates: vs'), 'C7'),
        (BATCH, 'static func pairwiseDistances() {}\n', 'C7'),
        # Phase D-F review A-2: every spelling of the identifier counts.
        (OPERATIONS, ops(BACKTICK_GATE), 'C2'),
        (OPERATIONS, ops(REFERENCE_GATE), 'C2'),
        (OPERATIONS, ops('        let route = Operations.gramBatch\n'), 'C2'),
        (OPERATIONS, ops('let strategy = queries.count > 63 ? BatchSearchCPUStrategy.`gram`(queryTileRows: 8, '
                         'candidateTileRows: 256, maximumTemporaryPayloadBytes: .max) : cpuStrategy\n'), 'C2'),
        # Final review FA-2: a gate whose call sits inside a string interpolation.
        (OPERATIONS, ops(INTERPOLATED_GATE), 'C2'),
        # Phase D-F review A-1: the BLAS forwarding and the binding.
        (MATRIX, SWAPPED_N, 'C8'),
        (MATRIX, CROSS.replace('alpha, x, arguments.lda,\n            y, arguments.ldb,',
                               'alpha, x, arguments.ldb,\n            y, arguments.lda,'), 'C8'),
        (MATRIX, CROSS.replace('0, out, arguments.ldc', '1, out, arguments.ldc'), 'C8'),
        (MATRIX, CROSS.replace('CblasNoTrans, CblasTrans', 'CblasNoTrans, CblasNoTrans'), 'C8'),
        (MATRIX, CROSS + OTHER_SGEMM, 'C8'),
        (MATRIX, OTHER_SGEMM, 'C8'),
        (MATRIX, CROSS.replace('cblas_sgemm(', 'cblas_dgemm('), 'C8'),
        (MATRIX, 'internal struct GemmArguments {}\n', 'C8'),
        (MATRIX, CROSS.replace('arguments: GemmArguments', 'arguments: GemmArgumentsV2'), 'C8'),
        (GRAM, TILE.replace('arguments: blas,', 'arguments: MatrixDistance.GemmArguments(rows: m, columns: ldc, d: d, '
                                                 'ldc: ldc),'), 'C8'),
        (GRAM, TILE.replace('blas: blas, ldc', 'blas: MatrixDistance.GemmArguments(rows: m, columns: n, d: d, ldc: ldc), '
                                               'ldc'), 'C8'),
        (GRAM, TILE + '                MatrixDistance.gemmCrossTerm(x: a, y: b, arguments: other, alpha: 1, into: out)\n',
         'C8'),
        (GRAM, TILE.replace('let blas = plan.gemmArguments(', 'let blas = makeArguments('), 'C8'),
        # Final review C-M3: the phase C binding, converted again from Int instead of taken from the plan.
        (GRAM, TILE.replace('let blas = plan.gemmArguments(activeQueryRows: m, activeCandidateRows: n)',
                            'let blas = MatrixDistance.GemmArguments(rows: m, columns: n, d: d, ldc: ldc)'), 'C8'),
        # Final review FA-1: another binding of `blas` feeds the BLAS call while the record keeps the plan's binding.
        (GRAM, TILE.replace(CROSS_CALL, '                do { var blas = MatrixDistance.GemmArguments(rows: m, '
                                        'columns: ldc, d: d, ldc: ldc)\n' + CROSS_CALL + '                }\n'), 'C8'),
        (GRAM, TILE.replace(CROSS_CALL, '                do { let blas = MatrixDistance.GemmArguments(rows: m, '
                                        'columns: ldc, d: d, ldc: ldc)\n' + CROSS_CALL + '                }\n'), 'C8'),
        (GRAM, TILE.replace(CROSS_CALL, '                for blas in [MatrixDistance.GemmArguments(rows: m, '
                                        'columns: ldc, d: d, ldc: ldc)] {\n' + CROSS_CALL + '                }\n'), 'C8'),
        (GRAM, TILE.replace(CROSS_CALL, '                _ = { (blas: MatrixDistance.GemmArguments) in\n' + CROSS_CALL
                                        + '                }(MatrixDistance.GemmArguments(rows: m, columns: ldc, d: d, '
                                        'ldc: ldc))\n'), 'C8'),
        (GRAM, TILE.replace(CROSS_CALL, '                if case let blas = MatrixDistance.GemmArguments(rows: m, '
                                        'columns: ldc, d: d, ldc: ldc) {\n' + CROSS_CALL + '                }\n'), 'C8'),
        (GRAM, TILE.replace('let blas = plan.gemmArguments(', 'var blas = plan.gemmArguments('), 'C8'),
        (GRAM, TILE.replace('arguments: blas,', 'arguments: `blas`,') + '                var `blas` = other\n', 'C8'),
        (GRAM, TILE.replace('n: Int(blas.n)', 'n: Int(blas.ldc)'), 'C8'),
        (GRAM, TILE + '        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, 1, 1, 1, x, 1, y, 1, 0, out, 1)\n',
         'C8'),
    ]
    MUST_PASS = [
        (OPERATIONS, ops()),
        (OPERATIONS, ops('/// `BatchOperations.Configuration` (`enableMatrixRouting`) does not affect this function.\n')),
        (OPERATIONS, ops('let message = "reads no BatchOperations setting"\n')),
        (OPERATIONS, ops('guard k > 0 else { throw VectorError.invalidDimension(k, reason: "k must be positive") }\n')),
        (OPERATIONS, ops('// A size gate such as `if queries.count > 7 { return try gramBatch(...) }` is not allowed.\n')),
        (OPERATIONS, ops('let note = "gramBatch( runs only for .gram"\n')),
        (OPERATIONS, SWITCH.replace('return try gramBatch(', 'try gramBatch(')),
        (GRAM, '// Never calls metric.distance or sorted(); see TopKSelection.\n'),
        (GRAM, 'heaps[i].pushIfBetter(val: scores[j], idx: candidateRows.lowerBound + j)\n'),
        (GRAM, 'for queryTile in 0..<plan.queryTileCount { }\n'),
        (GRAM, 'let rows = plan.queryTileRange(queryTile)\n'),
        (GRAM, 'MatrixDistance.cosinePostprocess(scratch.scores, rows: m, columns: n, ldc: ldc)\n'),
        (GRAM, 'case .scratch(queryPackElements: Int)\n'),
        (BATCH, PAIRWISE),
        (BATCH, 'let all = MatrixDistance.euclideanSquaredMatrix(queries: q, candidates: c)\n' + PAIRWISE),
        (BATCH, PAIRWISE.replace('        var rows = [[Float]]()\n',
                                 '        // Not MatrixDistance.euclideanSquaredMatrix(queries: vs, candidates: vs): no n * n buffer.\n'
                                 '        var rows = [[Float]]()\n')),
        (OPERATIONS, ops('        // A gate such as `return try `gramBatch`(...)` or `let f = gramBatch` is not allowed.\n')),
        (OPERATIONS, ops('        let note = "`gramBatch` and .`gram` are named only in strings"\n')),
        # A backtick-escaped spelling of the one admitted call is the same call (lex() normalizes escapes).
        (OPERATIONS, SWITCH.replace('return try gramBatch(', 'return try `gramBatch`(')),
        (OPERATIONS, ops('        let note = "\\(queries.count) queries; gramBatch( is named in literal text only"\n')),
        (MATRIX, CROSS),
        (MATRIX, '    /// One cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, ...) call; see cblas_dgemm.\n' + CROSS),
        (MATRIX, CROSS.replace('arguments.m, arguments.n, arguments.k,', 'arguments.m,\n  arguments.n,   arguments.k ,')),
        (GRAM, TILE),
        (GRAM, TILE.replace('arguments: blas,', 'arguments:\n blas,')),
        (GRAM, '// cblas_sgemm runs only inside MatrixDistance.gemmCrossTerm(arguments: other)\n' + TILE),
        (GRAM, TILE + '        let checked = Int(blas.m) + Int(blas.ldc)\n'),
    ]

    def test_must_fail(self):
        for path, text, rule in self.MUST_FAIL:
            with self.subTest(text=text):
                found = problems(path, text)
                self.assertTrue(any(item.startswith(rule + ' ') for item in found), (text, found))

    def test_must_pass(self):
        for path, text in self.MUST_PASS:
            with self.subTest(text=text):
                self.assertEqual(problems(path, text), [])

    def test_mutation_control(self):
        # Re-adding the deleted gate to a copy of the real Operations source is caught.
        text = (ROOT / OPERATIONS).read_text(encoding='utf-8')
        mutated = text.replace('switch cpuStrategy {',
                               'if queries.count >= 8, let r = gemmFindNearestBatch() { return r }\n        switch cpuStrategy {', 1)
        self.assertNotEqual(mutated, text)
        self.assertTrue(any(item.startswith('C2 ') for item in problems(OPERATIONS, mutated)))

    def test_respelled_gate_mutation_control(self):
        # Phase C review m9: the respelled gate (strict comparisons, routing to gramBatch) inserted into a copy of the
        # real Operations source is caught by the structural rule alone (the token rule does not see `> 7`).
        text = (ROOT / OPERATIONS).read_text(encoding='utf-8')
        mutated = text.replace('        switch cpuStrategy {', RESPELLED_GATE + '        switch cpuStrategy {', 1)
        self.assertNotEqual(mutated, text)
        masked = lex(mutated)
        self.assertEqual([m.group(0) for m in C2.finditer(masked)], [])
        self.assertTrue(any(item.startswith('C2 ') for item in _c2_structure(OPERATIONS, masked)),
                        problems(OPERATIONS, mutated))

    def test_escaped_and_referenced_gate_mutation_controls(self):
        # Phase D-F review A-2: E1 (backtick-escaped call) and E3 (unapplied reference) inserted into a copy of the real
        # Operations source. The token rule does not see either; the structural rule catches both.
        text = (ROOT / OPERATIONS).read_text(encoding='utf-8')
        for gate in (BACKTICK_GATE, REFERENCE_GATE):
            with self.subTest(gate=gate):
                mutated = text.replace('        switch cpuStrategy {', gate + '        switch cpuStrategy {', 1)
                self.assertNotEqual(mutated, text)
                masked = lex(mutated)
                self.assertEqual([m.group(0) for m in C2.finditer(masked)], [])
                # The phase C call count, on phase C's masking (no backtick normalization), misses both.
                self.assertEqual(len(C2_CALL.findall(lexing._mask(mutated, keep_interpolations=False))), 1)
                self.assertTrue(any(item.startswith('C2 ') for item in _c2_structure(OPERATIONS, masked)),
                                problems(OPERATIONS, mutated))

    def test_interpolated_gate_mutation_control(self):
        # Final review FA-2: the gate whose call sits in a string interpolation, inserted into a copy of the real
        # Operations source. The shared lexer keeps interpolated code, so C2's structure sees a second token; with the
        # former masking (interpolations blanked) the same structure sees only the admitted call.
        text = (ROOT / OPERATIONS).read_text(encoding='utf-8')
        mutated = text.replace('        switch cpuStrategy {', INTERPOLATED_GATE + '        switch cpuStrategy {', 1)
        self.assertNotEqual(mutated, text)
        self.assertTrue(any(item.startswith('C2 ') for item in _c2_structure(OPERATIONS, lex(mutated))),
                        problems(OPERATIONS, mutated))
        former = lexing._unbacktick(lexing._mask(mutated, keep_interpolations=False))
        self.assertEqual(_c2_structure(OPERATIONS, former), [])

    def test_shadowed_blas_mutation_control(self):
        # Final review FA-1 (review A's N1): the real BLAS call wrapped in a scope with its own `var blas`, while the
        # record keeps the outer binding. Each spelling is caught by C8 on a copy of the real GramBatchSearch source.
        gram = (ROOT / GRAM).read_text(encoding='utf-8')
        call = gram[gram.index('                MatrixDistance.gemmCrossTerm('):]
        call = call[:call.index(')\n') + 2]
        other = 'MatrixDistance.GemmArguments(rows: m, columns: ldc, d: d, ldc: ldc)'
        for head, tail in (('                do { var blas = ' + other + '\n', '                }\n'),
                           ('                for blas in [' + other + '] {\n', '                }\n'),
                           ('                _ = { (blas: MatrixDistance.GemmArguments) in\n',
                            '                }(' + other + ')\n')):
            with self.subTest(head=head):
                mutated = gram.replace(call, head + call + tail, 1)
                self.assertNotEqual(mutated, gram)
                self.assertTrue(any(item.startswith('C8 ') and 'rebinding' in item for item in problems(GRAM, mutated)),
                                problems(GRAM, mutated))

    def test_blas_forwarding_mutation_controls(self):
        # Phase D-F review A-1: the field swap at the cblas_sgemm call site (A-RA1-callsite-n) in a copy of the real
        # MatrixDistance source, and a call-site value other than `blas` in a copy of the real GramBatchSearch source.
        matrix = (ROOT / MATRIX).read_text(encoding='utf-8')
        swapped = matrix.replace('arguments.m, arguments.n, arguments.k,', 'arguments.m, arguments.ldc, arguments.k,', 1)
        self.assertNotEqual(swapped, matrix)
        self.assertTrue(any(item.startswith('C8 ') for item in problems(MATRIX, swapped)))
        gram = (ROOT / GRAM).read_text(encoding='utf-8')
        inline = gram.replace('arguments: blas,', 'arguments: MatrixDistance.GemmArguments(rows: m, columns: ldc, d: d, '
                                                  'ldc: ldc),', 1)
        self.assertNotEqual(inline, gram)
        self.assertTrue(any(item.startswith('C8 ') for item in problems(GRAM, inline)))

    def test_pairwise_flat_buffer_mutation_control(self):
        # Putting a flat N x N buffer back into the real gemmPairwiseTyped body (a copy) is caught by C7.
        text = (ROOT / BATCH).read_text(encoding='utf-8')
        head = 'let prepared = MatrixDistance.prepare(vs, normalized: !euclid)\n'
        mutated = text.replace(head, head + '        let flat = MatrixDistance.euclideanSquaredMatrix(queries: vs, '
                                            'candidates: vs)\n', 1)
        self.assertNotEqual(mutated, text)
        self.assertTrue(any(item.startswith('C7 ') for item in problems(BATCH, mutated)))


class RealTree(unittest.TestCase):
    def test_files_exist(self):
        for path in (OPERATIONS, GRAM, BATCH, MATRIX):
            self.assertTrue((ROOT / path).is_file(), path)

    def test_real_tree_passes(self):
        found = []
        for path in (OPERATIONS, GRAM, BATCH, MATRIX):
            found += problems(path, (ROOT / path).read_text(encoding='utf-8'))
        self.assertEqual(found, [])

    def test_real_tree_has_exactly_one_admitted_call(self):
        # Non-vacuity: the structural rule matches the real switch (one token, admitted by the case-label pattern).
        masked = lex((ROOT / OPERATIONS).read_text(encoding='utf-8'))
        self.assertEqual(len(C2_CALL.findall(masked)), 1)
        self.assertEqual(len(C2_IDENT.findall(masked)), 1)
        self.assertEqual(len(list(C2_CASE_CALL.finditer(masked))), 1)

    def test_real_tree_c8_is_not_vacuous(self):
        # Non-vacuity: the real Gram file has one BLAS call, one binding, one record call and the six copied fields;
        # the real MatrixDistance has one cblas_sgemm call with the expected argument list.
        gram = lex((ROOT / GRAM).read_text(encoding='utf-8'))
        self.assertEqual(len(_calls(C8_CROSS, gram)), 1)
        self.assertEqual(len(C8_BINDING_VALUE.findall(gram)), 1)
        # The binding, `arguments: blas`, `blas: blas` (two tokens), the record parameter and six field reads.
        self.assertEqual(len(C8_BLAS.findall(gram)), 11)
        self.assertEqual(len(C8_RECORD.findall(gram)), 2)   # the call and the declaration `func gramTileRecord(`
        self.assertEqual(len(_calls(C8_RECORD, gram)), 1)
        fields = re.findall(r'\b(\w+)\s*:\s*Int\s*\(\s*blas\s*\.\s*(\w+)\s*\)', gram)
        self.assertEqual(sorted(name for name, _ in fields), sorted(C8_FIELDS))
        matrix = lex((ROOT / MATRIX).read_text(encoding='utf-8'))
        call = C8_SGEMM.search(matrix)
        self.assertIsNotNone(call)
        self.assertEqual(re.sub(r'\s+', '', _call_arguments(matrix, call)), C8_ARGUMENTS)


if __name__ == '__main__':
    unittest.main()
