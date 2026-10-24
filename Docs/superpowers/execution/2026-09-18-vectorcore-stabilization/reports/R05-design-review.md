# R05 independent design review

Verdict: **ACCEPT** for implementation under the current R05 brief. No blocking mathematical, API, or integration gap found. This is design acceptance, not verified implementation or API-11 closure.

Reviewed execution POLICY, R05/R02 briefs, historical API-11, and current DistanceMetric, DistanceMetrics, SearchResult and VectorCollection source plus the approved computedScore test migration. No builds or production/test edits were needed.

## Mathematics and domain

The six nonnegative-distance families can retain inverse relevance without the old pole: restricting inputs to finite d >= 0 makes 1 + Double(d) positive and finite. At Float.max the reciprocal is approximately 2^-128, still a positive representable Float subnormal. At either signed zero it is exactly 1. Float rounding makes this mapping weakly decreasing, not strictly decreasing; tiny positive distances can also round to score 1. Cosine's inverse-distance score is a deliberate convention and must not be documented as cosine similarity. Hamming threshold and Minkowski p configuration remain part of their existing distance witnesses.

For negative-dot distance d, logistic(-d) = 1/(1+exp(d)) increases with the underlying dot product and decreases with distance. The two branches in the brief are algebraically equivalent and exponentiate only nonpositive arguments, preventing exponential overflow. Both signed zeros produce 0.5; d=-1 has finite score about 0.731. Large finite magnitudes may saturate to 0 or 1 after Float projection. Rejecting all nonfinite distances before branching prevents an overflowing/invalid distance calculation from becoming a confident endpoint score. This policy is intentionally different from accepting finite extremes.

The guarantees concern the delivered distance, not its numerical correctness or the meaning of an arbitrary configured metric. In particular, a kernel may return a finite value after encountering problematic coordinates; R05 does not validate those coordinates or repair the kernel. No calibration, probabilistic, or cross-metric comparability promise is justified by a bounded score range.

## API and dispatch

Adding relevanceScore as a protocol requirement with a public default nil implementation is the appropriate witness-dispatch design. A method only in the extension would fail the required existential/custom case. The requirement should remain expressed in Scalar, preserving existing non-Float conformers; the SearchResult interpretation method can be Float-specific because its stored distance already is Float. No requirement should be inferred from name/identifier, and no builtin downcast is needed.

SearchResult.computedScore(using:) must return a present score directly, including negative/out-of-range values, signed zero, NaN payloads and infinities. No finiteness check, clamp or arithmetic belongs on that branch. For absent score, call the supplied witness once and return its optional result unchanged. The method cannot validate that a caller supplied the original metric; documenting that responsibility is sufficient. A custom witness may legitimately return nil or values outside builtin ranges. Nil is absent interpretation, not a reason to invoke the historical fallback.

Removing the property is an authorized source break. The approved existing test migration preserves its Euclidean values while adopting the optional explicit-metric method. New defaulted conformance should compile for external consumers, with explicit/custom score dispatch verified through concrete, generic and existential uses. This is source compatibility for defaulted conformances, not a binary ABI guarantee.

## Producers, locks and preservation

Current source builds selected SearchResult values in the VectorCollection default's generic search helper. R02 will also own SimpleVectorCollection's snapshot implementation/helper. R05 must begin from accepted R02 and inspect both actual producer paths, even if R02 unifies them. Relevance must use the same metric value/witness and the selected stored distance, after TopK membership and ordering are fixed. No new distance evaluation or score-based ranking is needed.

Metric score methods are arbitrary caller code, just like distance methods. They must execute after the snapshot lock is released. A reentrant score witness that reads/mutates the collection is a useful direct check in the new score suite, complementing R02's reentrant distance test; any deadlock-capable child probe needs a bounded execution deadline. Tests should preserve captured result membership when score callbacks change live collection contents. These checks verify integration and do not broaden snapshot guarantees for externally aliased custom vectors.

Count score invocations only for selected results, including selected nonfinite distances; builtins can then return nil. k<=0 and empty selection call no score witness. Do not change distance-call counts, candidate counts, tie handling, or NaN ranking for this requirement. Custom search overrides retain responsibility for their stored scores; the three convenience searches must forward rather than reconstruct results or overwrite explicit values.

mapIDs already copies score, and TopK conversion already lacks metric context and leaves score nil. Preserve these paths. Synthesized Codable can retain the existing fields and optional decoding behavior without adding metric metadata. Producer-populated scores will cause ordinary encoded search results to contain score values where previous results omitted them; this is the intended semantic change within the same schema. Default JSON cannot represent nonfinite numbers, and a configured string strategy does not promise preservation of a NaN payload's bits. Test payload preservation in memory, and configured wire behavior by the strategy's actual guarantees.

## Regression and oracle quality

The actual historical query [1] with candidates [0.5], [1], [2] is an adequate behavioral RED independent of any API compile failure. Its distances and order must stay [-2,-1,-0.5] while scores become bounded and descending. This directly addresses API-11 through the producer instead of only a helper.

Direct Double logistic evaluation at modest d is independent of the production piecewise formula and safe there. At extreme Float inputs, use exact rounded endpoints or independent high-precision references; do not reuse the branches as an oracle. Literal inverse controls 0->1, 1->0.5 and 3->0.25 plus widened expected values adequately check that simple transform. Finiteness-first assertions, four-ULP ordinary tolerances and weak monotonicity cover numerical behavior without demanding impossible distinctions after Float rounding. Include the specified -1 neighbors, signed/tiny values and invalid classifications.

The required custom/default/spoofed-name witnesses, explicit-score noninvocation spy, all-seven-metric ordinary collection cases, nondefault Minkowski parameter and external consumer fixture close the dispatch/integration defect class. Keep raw distance expectations modest and independent of score generation; extreme score-domain tests can call the witness directly rather than exercising unrelated unstable kernels. Preserve finite wire round trips, absent/nil scores and mapIDs bit preservation.

## Scope and acceptance

The listed four source files and focused tests are sufficient. Keep distance/batch formulas, TopK, shape decoding, comparison semantics, Package and downstream packages untouched. Reserve VectorCollection against R02 and SearchResult against S03. At review time this worktree still shows pre-R02 mutation/default-search code, so accepted R02 integration remains an explicit dispatch prerequisite.

Final closure requires isolated runtime RED, focused Debug/Release suites against accepted R01/R02, public-consumer migration evidence, full changed/new diff and hashes, and independent implementation review of actual witnesses and producers. This review establishes no downstream relevance usage, ANN/GPU behavior, distance-kernel accuracy or calibration claim.
