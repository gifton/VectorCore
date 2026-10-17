# N01 independent implementation review

## Frozen-source disposition

**Source review favorable; final acceptance pending runtime evidence.** No concrete numerical, ownership, aliasing, dispatch, policy or scope blocker was found in the frozen implementation. Final Debug03/Release/ASan records against these bytes remain required. This is not yet NUM-01/NUM-02 closure; API-39 downstream CPU/GPU parity remains unexecuted.

Reviewed execution POLICY, binding briefs/N01.md, N01-design-review.md including safeNormalized, N01-implementation.md, the source manifest and the complete 2,395-line N01-full.diff. Root explicitly authorized the additional two Accelerate binary division repairs after the retained isolated control failure. No reviewer builds, source/test edits, commits, index operations or delegation occurred.

## Identity and scope verification

- Base `028b4aacc80249b745a89454008c7d2b9e5da154`; archive `/private/tmp/vectorcore-stabilization-N01-implementation`.
- All 22 final path hashes match N01-source-manifest.json. Complete diff SHA256 is `213a43e5b1c58b630322dba48573f83d9b627d41032da578f5da0abc53025633`.
- Independently regenerated the full diff from base and archive, including all new files; bytes match the frozen diff exactly.
- Compared 240 baseline paths under Sources, Tests, Scripts and Package.swift with archive blobs and inventoried new files. None was missing, and changed/new paths equal the manifest exactly. No existing test expectation, Package, backend flag, scheduler, tuner or unrelated source change is included. The root-reported external temporary-file restoration is not classified as implementation scope.
- Independently recomputed all four retained binary64 normalization fixtures and four quotient fixtures using exact input-bit decoding and Decimal at both 2500 and 3000 digits. Every rounded output matches the stability artifact and checked-in Swift literals. The generator is offline and its fixture-writing code was not executed during review.

## Division and dispatch

Scalar operator/default/provider changes use direct division rather than a rounded reciprocal. VectorProtocol and VectorFactory / and /= paths retain their existing selection and factory behavior; direct inPlaceDivide preserves the S02 mutation/shape domain. All four concrete optimized / operators delegate to direct per-coordinate mutation of a copy. Zero scalar rejection occurs before mutation/factory work, including empty input and either signed zero. Nonzero nonfinite scalar behavior is not restricted by a new finiteness guard.

The inherited SIMDProvider implementation, SwiftFloat override and Accelerate Float/Double overrides check nonnegative count before constructing a range and access nothing for count zero. Each coordinate reads its source before writing its result, supporting exact aliasing. Binary Accelerate division similarly reads both operands at index i before writing index i, so either operand may exactly alias output. Partial overlap is not newly promised. This change follows the separately authorized same-class vDSP failure and does not introduce backend routing gates or change zero/nonfinite IEEE policy.

Both underlying array scalar adapters use direct division; SwiftSIMDProvider/forwarding routes remain intact. Tests cover inherited Float/default Double, Swift/Accelerate, both forwarding namespaces, all three array adapters, protocol/factory generic operators, concrete optimized widths and raw aliases/sentinels. The Float exponent sweep covers -149...127; Double closed forms and independent quotient bits supplement exceptional/near-boundary cases. The binary class matrix checks both exact aliases, tails, zero and special values. Expected direct Float quotients are widened independently where finite range matters; signed-zero bits/sign and NaN classification are distinguished.

## Shared normalization mathematics

NormalizationCore first rejects empty input, scans every coordinate needed to establish finiteness/max magnitude, then rejects all-zero input. No destination writes occur before a successful plan. Nonfinite input takes precedence over zero classification. The Float/Float16 branch widens coordinates exactly and accumulates squared values with compensation in Double. Finite Float squares and at most Int.max terms stay within the stated Double range; the final norm is never narrowed before coordinate division. The half-vector fixture with 65,536 ones directly checks that q is not accumulated in Float16.

The Double/native branch uses finite positive max scale and compensated sum of scaled squares, checks positive finite q, and returns `(x/scale)/sqrt(q)`. It forms neither the original-scale norm nor its reciprocal for output. Arbitrary Scalar values are not narrowed to Double: the widening branch is explicitly limited to Float and Float16. A nonrepresentable native sum fails before writeback. Discarding sufficiently tiny Double components through native underflow is consistent with the specified absolute/normwise accuracy contract, rather than a componentwise relative guarantee.

The plan stores scalar values, not a borrowed pointer or escaping getter. Generic methods compute it within a read borrow, then copy and mutate through a separate valid borrow. Optimized checked kernels validate their existing storage before any coordinate write; rejected inputs preserve raw bits. S01 storage-count properties and O04 pointer ownership/binding bodies outside this scope remain intact. Internal optimized magnitude helpers have only fixed-width callers in this scope and now accumulate the actual selected coordinates without SIMD rebinding. Their final Float magnitude/squared-magnitude projection can legitimately overflow while normalization still succeeds. floatSquares distinguishes NaN from infinity before compensated arithmetic, preserving NaN precedence.

The pointer normalization path checks negative dimension before work, returns immediately for zero, and uses scalar Float loads/stores after complete validation. This supports naturally aligned Float pointers at 0/4/8/12-byte offsets without requiring SIMD alignment or rebinding. No allocation is proportional to the external pointer count. Its unusable-pointer/count-zero fixture verifies the early no-access path. Typed unchecked paths validate with named fatal diagnostics in Debug and normal Release; no -Ounchecked or arbitrary lossy-storage accuracy promise is made.

## Policies and consumers

Permissive array normalization returns original input on failed planning, preserving empty/zero/nonfinite bits. Fast/tolerance methods validate tolerance before early returns, retain strict widened `abs(norm-1) < tolerance`, and make either signed zero disable the shortcut. Large finite tolerance and near-unit default no-ops remain documented and tested separately from executed-normalization norm bounds. isNormalized's existing squared-norm convention is untouched.

Both TransformOperations bodies call the stable permissive path and the required create(from:) factory for each row, preserving dimensions, ordering, error propagation and existing serial/parallel selection. Operations.normalize retains its task-local custom provider witness: sentinel-provider tests prove it is not bypassed. Boundary tests at 0/1/999/1000/1001 exercise both public batch APIs with varying dimensions, tiny/huge values and invalid rows; injected factory errors test both execution paths.

safeNormalized changes only the owned method. It applies original handleNonFinite first, respects its strict/propagate/replacement behavior, then returns unresolved nonfinite handled coordinates without further normalization. Exact all-zero checking preserves strict zero error, propagation all-NaN and unchanged signed-zero policy in order. The explicitly Result-typed normalized call prevents accidental selection of the permissive tolerance overload. Its independent raw-bit option oracle enumerates all 32 flag combinations at dimensions 4 and 512, across placements and input categories, and checks original error indices. It computes expected handling directly rather than calling production handleNonFinite; the short finite expected-vector oracle independently sums widened terms. Existing safeDivide is a control.

Matrix row normalization now uses the same plan, preserves zero rows and fills invalid rows with NaNs. Packing, GEMM and NaN-preserving distance clipping are unchanged. The tests execute both prepared and unprepared matrix entry points with invalid query/candidate rows, zero rows and unrelated valid pairs, checking propagation through actual matrix output instead of only row preparation.

Hint conveniences evaluate returned coordinates after actual writeback, using a finite widened norm check, then set true/cached unit magnitude only on success. The lossy custom conformer exercises a result that was mathematically normalized before writeback but fails after rounding; checked returns nil and unchecked returns false with nil cache. All four optimized widths, generic fixed and Dynamic routes plus wrapper conveniences are represented. Existing asserted-unit constructors and I01 invalidation/decoding remain unchanged.

NormCache normalized builders continue to preserve checked failures. Each preNormalized consumer explicitly produces NaN for a nonfinite dot before finite clipping; tests cover valid/invalid queries and candidates at 512/768/1536. Ordinary cache norms, cached cosine scaling, generic distance kernels, GPU/C routes and quantization fidelity are not repaired or claimed covered.

## Oracles, failure diagnostics and pending verification

The regression suite separates exact signed-basis/four-equal/Pythagorean expectations from arbitrary mixed Double references. Decimal references use exact binary inputs, two independently stable precisions and retained oracle residuals. Runtime Float validation checks actual finiteness before coordinate/residual comparisons and uses compensated Double residuals; Double output residuals use pairwise bounded-square summation, distinct from production max scaling/Kahan accumulation. Full coordinate comparisons catch wrong direction even if output norm is one. Named fixture acceptance thresholds are not presented as universal forward-error theorems.

The child helper installs its termination semaphore before run, waits up to 15 seconds, then bounds termination/SIGKILL cleanup to two one-second waits. It never calls waitUntilExit or blocks on a pipe. Logs go to unique files; timeout records failure and avoids treating termination status as a passing diagnostic. The 24 selected invalid cases require the intended fatal-error message and signal rather than merely any failure. ASan must exclude these deliberate guard children while retaining pointer/alias/consumer checks.

The handoff preserves separate initial division/normalization behavioral RED, the binary-provider control RED, compile-only failures and the diagnostic-stripping Release failure; it does not attribute the later expanded sweep to the initial minimal RED. Its migration section accurately describes checked/unchecked/permissive/tolerance/safeNormalized and hint behavior without claiming every norm/distance route is repaired.

Before final approval, inspect the frozen-byte Debug03/Release/ASan commands, exits, actual selected symbols and diagnostics; verify final source hashes still match this manifest. The currently favorable source finding does not certify pending runs, downstream VectorAccelerate behavior or production performance/scale readiness.

## Final evidence adjudication — ACCEPT

**ACCEPT the frozen N01 implementation for the bounded Core contract.** The previously pending runtime gates are satisfied by the final evidence. Root retains integration and NUM-01/NUM-02 closure decisions. API-39, downstream VectorAccelerate CPU/GPU execution, ordinary norm/cosine range and the other exclusions above remain outside this acceptance.

Read final N01-validation.json and the updated N01-implementation.md. N01-validation.json SHA256 at this review is `4942f2a5150885d8339263107230dc676ee18cca044937920037a8801b1735fc`. Rechecked every one of the 22 archive file hashes and the complete diff hash: all remain unchanged. The validation record's source-diff hash agrees with the source manifest and independently hashed diff, and it names the same isolated working directory, six-job limit and separate Debug/Release/ASan scratches.

Verified the full retained final-log hashes against the exact commands/status records:

| Run | Log SHA256 | Inspected result |
|---|---|---|
| Debug03 | `f354057d0aaa2c3df03531a4b6b9203fa6161eb7766eb2665464d82227c3fe92` | Exit 0 recorded; 100 reported functions in 10 suites passed |
| Release02 | `ce26fa018f8aa560cfc6218eb945bb8f6c052333a44ea82db18b89f1b52b1cb8` | Exit 0 recorded; same 100 functions in 10 suites passed |
| ASan01 | `4a1ba89fd43fae0a9c3625fa11eabacd862ec49f3f5400941c01b02e73222f28` | Exit 0 recorded; 35 reported functions in four suites passed; no sanitizer finding |

Both Debug and normal Release explicitly record invalidInputsRejectInChildren passing; the reviewed helper executes its 24 intended-diagnostic cases and checks the expected messages/signals. There are 144 retained child captures across all attempts, including earlier failures; this aggregate is not represented as 144 final acceptance cases. The final ASan command sets N01_SKIP_GUARDS=1. Its documented 35 reported functions comprise 33 exercised functions plus two explicit no-op guard entries, with zero deliberate child traps; those entries do not supply sanitizer coverage.

The final logs explicitly show passing safeNormalized flag enumeration, prepared/unprepared matrix propagation, binary Accelerate alias/class coverage, natural-pointer-offset controls, actual hint writeback/cache consumers, native Double independent references and both batch boundaries. This confirms execution of the meaningful new consumer routes rather than relying solely on suite names. Exact command filters retain the requested existing normalization/pointer/hint/shape/provider/TopK controls in Debug/Release; ASan selects the valid division/normalization/pointer suites. Oracle regeneration retained the exact fixture hash, consistent with this review's independent Decimal recomputation.

No runtime discrepancy or new unsupported completion claim was found. These are reviewed worker command/log records; no duplicate reviewer builds were launched. Complete source/diff/scope identity, independent arithmetic references, final guard diagnostics and valid-path sanitizer evidence now support acceptance within the stated boundaries.
