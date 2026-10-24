# V03 v3 scoped source re-review

**ACCEPT source correction; final runtime/evidence acceptance remains pending.** The disclosed v2 compile repair and all three IR1 timing-predicate sites plus R1/IR2 aggregate-policy gap are resolved in the frozen v3 source. This verdict permits the planned validation; it does not claim Swift compile/runtime, sanitizer or remote-CI success.

Independently verified frozen full patch SHA256 `e478c2eb1f11334066692e299c458b1ad59eabd0913f8052b8306279e404149d`, manifest `f95f009d83506b1b0bbbdc2827c034de878f82e9d3eeb089e30f2363292c0c9d`, and v2→v3 delta `42a728c25794cf31296ae87ee2956d442507ae7652c315be42964a594b36ed21`. Rehashed all 62 owned manifest entries, including the declared deletion, and compared v2/v3 manifests and files: precisely the seven disclosed files changed. All other owned bytes, including production and the v2 XCTest wrapper repair, remain identical.

## Timing-predicate correction

BatchKernels_SoATests now records numeric size/timing observations without computing the `allTimesReasonable` Boolean. The original threshold comparisons move into a lazy allSatisfy predicate under strict admission. Its short-circuit reduction preserves the prior pass/fail result; it does not change the measured operations or thresholds. The former performance-cliff loop is likewise expressed as a lazy allSatisfy predicate, with the same efficiency ratio and cutoff. Its existing nonempty fixed timing collection preserves the range domain.

MixedPrecisionKernelTests.testThroughputScaling now places `strictPerfGatesEnabled` first in the assertion-entry conjunction, so strict-off does not evaluate the clock-derived `previousThroughput > 0` condition. Numeric throughput observations and unconditional numerical controls remain intact. No threshold or correctness check was relaxed.

The maintained policy rejects bare precomputed Boolean expressions even if a future edit also changes the inventory to bless that form. Each of the two indirect sites has a mutation that changes both source and inventory and must fail. The entry-condition inventory pins the actual immediate enclosing conditional; mutations remove or reorder strict admission and require rejection. This closes the identified bounded syntax class, not arbitrary Swift dataflow. The scanner explicitly disclaims general dataflow analysis, appropriately.

## Aggregate policy correction

validate_github.rb now rejects aggregate job-level continue-on-error and any aggregate step-level continue-on-error before checking its exact always()/needs/result-command contract. The new matrix covers all eight jobs (seven prerequisites plus CI Required) at both job and step levels. Existing command-step checks and unconditioned exact aggregate result-command checks remain. The current workflow itself was already correctly configured; the correction closes the regression-guard hole.

Read the retained configured Python discovery command and output: actual exit 0, no timeout, 56 tests in 17.049 seconds, with the authorized 120-second outer deadline and two-second cleanup. Independently rehashed stdout/stderr against the command record. This is relevant policy runtime evidence only; it is not substituted for pending Swift/compiler, instrumented-body, lint or process-contract validation. No tests or builds were rerun by this reviewer.

Earlier source review findings and the v1 compile failure remain recorded in V03-source-review-v2.md. No additional blocker was found in this narrow correction review. Canonical copy: October2/reviews/V03-source-review-v3.md.
