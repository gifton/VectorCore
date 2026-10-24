# R04 final independent acceptance — frozen v1

**ACCEPT.** Accepted source/spec disposition from R04-v1-source-review.md remains unchanged; sealed runtime/evidence checks now satisfy the bounded final obligations. No remaining blocker found. No builds, test reruns, source edits, commits or delegation occurred.

Verified worker index SHA256 `b276f1cd648ae85ba0b1fa14cb9d941a3963fcd87fc94ca6f9b4b3a4c66fc21c` and implementation report SHA256 `ed643b8e2c5b2b83f612b6b5091d7cb8147650875f49f23bd3791f10539431b6`. All 26 worker records match both shared and canonical evidence-final-v1 copies. The accepted patch remains `a5dffcd7c82cad065528e1f32e66196cd8cf518cc2336a36fa5f5fa3f15e0049`; all 321 frozen source hashes still match. No post-test source revision is being inferred.

## Actual evidence and provenance

Independently rehashed the retained actual build entries: initial compile-failure Debug 4,259; behavioral RED Debug 4,351; final Debug 4,351; final Release 2,885. These are manifest-entry counts, not unique test executions. All 355 absolute source/runtime artifact-map entries match. Final Debug's three raw log chunks and Release's four chunks concatenate byte-for-byte to their sealed complete logs. Source-review base/final reconstruction remains the prior accepted 320/321 check; the broader root archive reconstruction is separate evidence.

The initial baseline log contains operator-inference and mutating-call macro compilation errors, not a runtime regression. The two interrupted approval attempts remain unexecuted setup history, with no invented status. Actual red3 exits 1 with four reported functions and 13 issues: four bulk/stream assertions across two orientations, two NaN-eviction assertions, six assertions from the two incorrectly successful negative-capacity children, and one timeout from negative empty bulk construction. The bulk child logs entry, does not log acceptance, and its parent records the 15-second timeout. This is a bounded reproduced hang, not a successful trap or a passing control. No baseline huge-capacity allocation or parent hang is claimed.

Final Debug and Release each exit 0 and report 72 Swift Testing functions in four suites under the exact filter `HeapMembershipContractTests|TopKNaN|IntegrationProtocolsSuite`. Checked actual passed-body lines for the generic/record/lifetime/nearest matrices, both bulk orientations, explicit NaN eviction and negative-capacity parent. Each total includes one no-op childProbe parent entry; negativeCapacity is one parameterized function launching three actual children, not three additional primary functions. Existing modern TopK and integration controls execute in the same focused runs. The separately logged scanner command exits 0 with seven actual Python checks; it is not a Swift-test total or full semantic proof.

## Nine child associations

Verified all three red3 and six final captures against original and retained copies, their embedded scenario/argv records, corresponding parent capture/result lines, actual copied helper host and separately loaded versioned test bundles. The host SHA256 is `da2601bc562f2e0dd0cd7e00e923aee763ce94bdbf3eddccdebf1a4b8d5d313d`. Loaded bundle hashes are red3 `f4d634e890933d08c2d5366c4f4e1d1c20c788e2f16a1c7854f83a49d4a8a823`, Debug `5b4a106ce8ace63236c56353040244ebc348eba1a1cebf7c3d75cc3904b2c52e`, Release `aafe08d30f0de5eb3f255abb62e9965a9a49861335f4ffa58ea1da43a51e93ca`. These are actual retained bytes, not a helper hash substituted for a test bundle.

Every final child reports status 5/reason 2, contains its intended entry marker and lacks the accepted marker. Debug captures also contain the named nonnegative-capacity diagnostic. Release uses actual signal/entry evidence plus the already reviewed frozen precondition guards; no assertion that optimized diagnostics survived is made. Red3 empty/nearest return status 0/reason 1 and contain accepted markers; red3 bulk alone is a timeout and has no fabricated terminal-status result.

The reviewed helper installs its handler before launch, captures to files, and waits 15 seconds plus at most two one-second cleanup intervals. It throws on timeout before status/reason access. This bounds the parent wait; it does not prove every hypothetical timed-out child was reaped or establish hard scheduling latency. Final actual children complete normally through the guard path rather than exercising timeout cleanup.

## Quality and closure limits

The source repair and final matrix establish the approved strict-weak-order retention law, lazy capacity behavior, copy/lifetime controls and explicit legacy numeric/NaN/index ordering. Independent sorting/classification oracles and permutation tables cover the repeated membership class without relying solely on production comparator logic. Public legacy signatures remain; tie/NaN behavior changes deliberately under the documented migration. Modern production routing remains unchanged and the removed ExecutionOperations route stays absent.

API-17/API-26 are ready for root closure on this evidence. No pointer change requires ASan here. This verdict does not claim benchmark speedup, measured allocation savings, arbitrary invalid-comparator semantics, stable identity among equivalent keys, downstream execution or production-scale readiness. Historical setup failures and the baseline timeout remain distinct from final acceptance.
