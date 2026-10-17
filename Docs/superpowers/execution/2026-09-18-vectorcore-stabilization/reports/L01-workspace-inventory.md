# L01 / NUM-35 workspace inventory

Read-only source inventory; no builds, large allocations, runtime corruption reproduction, arbitrary workload cap or implementation choice. Reviewed vc_lapack.c/.h, shared shape validation, LAPACK provider and existing LinearAlgebraProviderTests. Root-provided Netlib sufficient-workspace formulas are distinguished below from locally established Accelerate ABI facts.

## ABI and exposure

Sources/VectorCoreC/common/vc_lapack.c:18–26 and Package.swift:47–53 select **modern ACCELERATE_NEW_LAPACK**, without ACCELERATE_LAPACK_ILP64. This is the modern LP64 interface, not the deprecated clapack route. The installed SDK `Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/lapack_types.h` typedefs __LAPACK_int as int when ILP64 is absent; lapack_version.h aliases symbols to $NEWLAPACK. lapack.h:24395 (sgesdd_) and17662 (ssyevd_) expose signed __LAPACK_int workspace lengths and availability macOS13.3/iOS16.4. These local headers provide prototypes, not implementation arithmetic or a workspace-overflow guarantee. The source comment asserting Accelerate3.9.x and a fixed safety margin is not independently established by these headers.

The C entry points are exported by vc_lapack.h and do not themselves validate dimensions/leading dimensions/pointer capacities before their own allocation arithmetic. Swift provider callers have shared shape checks; a direct C caller does not inherit them. C pointers remain caller-owned capacity contracts, not checkable array lengths.

## Every local arithmetic/conversion boundary

| Site | Current behavior and limit |
| --- | --- |
| LinearAlgebraProvider.swift:161–200 | Positive rows/columns <=Int32.max, then rows*columns or dimension² equality. Two positive Int32 dimensions multiply safely in 64-bit Swift Int; this is not proof that Float payload bytes fit signed Int or LAPACK internal indexing. No workspace or byte validation. |
| LAPACKLinearAlgebraProvider.swift:29–57 QR | Int32(rows/columns) after shape check; copy-on-write input, tau count columns; R allocation columns² after first factorization; indices j*columns+i and j*rows+i. QR rows>=columns makes R element count <= input count. Array byte/allocation failure is not caught as a VectorError. |
| LAPACK provider:65–95 SVD | k=min(rows,columns), Int32 conversions; output counts k, rows*k, k*columns and input mutable copy before C. These element products fit 64-bit Int under dimension checks but no explicit byte/aggregate allocation guard exists. |
| LAPACK provider:100–123 eigen | Int32(dimension), mutable input copy and eigenvalue count dimension before shim; input square count already checked. No workspace preflight. |
| vc_lapack.c:47–51 shared float query conversion | ceilf(q) casts directly to int64_t before checking finiteness/range/sign, then r+=r/64+1 and clamps only above INT32_MAX. NaN/Inf/out-of-int64 conversion is not defined by safe C integer conversion; negative results can survive and later convert to huge size_t. Signed addition can overflow before clamp for enormous q. Saturating an excessive request to INT32_MAX hides nonrepresentability rather than establishing sufficient workspace. |
| vc_lapack.c:54–69 sgeqrf | Copies Int32 to LP64 __LAPACK_int; queries with lwork=-1 before validation of returned q; converts q via helper; malloc((size_t)lwork*sizeof(float)); calls again then frees. |
| vc_lapack.c:72–87 sorgqr | Same query/conversion/allocation pattern; k is also forwarded unchanged. |
| vc_lapack.c:90–116 sgesdd S | minmn=min(m,n); **8*minmn occurs as signed __LAPACK_int before size_t cast** at99, then multiplied by sizeof(__LAPACK_int). IWORK is allocated before query. Returned floating workspace uses the common helper, then float-byte allocation. Error paths free allocated IWORK. |
| vc_lapack.c:119–145 ssyevd | Dual query occurs first. Float query uses common helper; integer iquery is assigned without positive/minimum/range validation. Both counts cast to size_t and multiply by element size. Both buffers freed on allocation failure and normal return. |

For valid positive Int32 lwork on current 64-bit size_t platforms, multiplication by4 fits size_t; that fact does not validate the preceding signed count/query conversion. A portable checked byte helper can state the complete condition independently of host width. Similarly Float matrix bytes for the maximum Int32 square fit unsigned64 size_t but exceed signed64 Int, which matters to Swift allocation machinery. No practical matrix of that size is claimed to exist in memory.

## Error mapping

C returns raw LAPACK info for query/execution errors. Both unavailable backend stubs and malloc failures return VC_LAPACK_UNAVAILABLE=-1000. LAPACKLinearAlgebraProvider.swift:127–154 maps absent backend/-1000 to resourceUnavailable; other negative info to invalidOperation labelled provider bug; positive info to operationFailed labelled nonconvergence. There is currently no dedicated checked-workspace/size error code. Any new negative sentinel must be recognized before generic -info formatting (and avoid Int32.min negation). Shape errors precede backend availability checking. The header's -1000 description mentions unavailable backend but omits allocation failure; repair documentation/error distinctions alongside any checked helper.

## What can be established locally

Pure helpers can check positive dimensions and operation-specific leading-dimension relationships, widened workspace products/sums, count-to-byte representability, and query conversion without touching matrices. Use checked integer operations or promote before multiplication; reject impossible values rather than clamp to a falsely sufficient size. A query conversion helper can reject nonfinite/nonpositive/oversized values before any integer cast and use a documented conservative rounding policy (including Float spacing above2^24), with checked margin arithmetic if a margin is retained.

Root supplied primary Netlib contracts: SGESDD JOBZ S sufficient LWORK=4*k²+7*k, IWORK=8*k; SSYEVD JOBZ V LWORK=1+6*n+2*n², LIWORK=3+5*n (special small dimensions must retain their documented cases). These support derived representability checks, not an arbitrary production dimension cap. They should be calculated in a widened checked type before query and before Swift output allocation if the goal is early clean rejection. A shared/internal preflight exposed for tiny arithmetic tests can avoid giant arrays. Distinguish SSYEVD eigenvalues-only policy from eigenvectors; do not apply the V formula to every job without documenting conservative rejection.

**Limit:** A representable sufficient workspace bound, IWORK byte count and returned query do not prove all arithmetic *inside* Accelerate's workspace query is safe. Query execution has already occurred when its result is validated. Thin/tall SGESDD can have implementation/path-dependent optimal-workspace intermediates involving the large dimension even where a min-dimension bound fits. QR queries can use block-size products; local prototypes do not reveal their bounds. Do not claim arbitrary Int32-shape safety from postquery guards. Bounded options for root: implement local arithmetic/query checks and qualify remaining upstream scope; or establish a documented implementation-derived prequery envelope using primary algorithm/source evidence; or avoid optional queries for routines with a verified sufficient-workspace contract while still qualifying internal execution/index arithmetic. None alone proves arbitrary backend indexing safety, and no ABI switch to ILP64 is implied.

## Tests and scoped acceptance

Existing LinearAlgebraProviderTests exercises small QR/SVD/eigen reconstruction/orthogonality/spectra and shape errors (149/157/291), availability and backend parity (313). It contains no workspace conversion, byte/count overflow, query-result or extreme Int32 arithmetic test. Searches for vc_lapack routine/workspace tests found no dedicated boundary suite.

Add allocation-free helper tests around Int32 bounds, each widened polynomial/product cutoff, SIZE_MAX/element-size cutoffs, zero/negative counts, Float2^24 neighboring values, largest acceptable query, NaN/Inf/negative/fractional query and checked rounding/margin overflow. Assert exact typed/sentinel mapping and that rejected preflight does not call LAPACK or allocate; use explicit seams, not a fabricated huge matrix. Keep real small QR/SVD and eigenvectors-on/off controls to establish ordinary backend behavior. Shared Swift validation changes must preserve both providers' shape/error ordering deliberately; separate pure-helper acceptance from actual Accelerate runtime coverage.

NUM-35 remains a qualified extreme-shape issue. This inventory establishes local unsafe expressions and proof limits, not practical-size corruption, a measured allocation failure or a universal supported shape envelope.
