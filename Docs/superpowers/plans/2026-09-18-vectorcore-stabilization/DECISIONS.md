# Planning decisions and dependency interpretation

[Planning draft](README.md) · [Work-item catalog](WORK_ITEMS.md)

These decisions organize future work. They do not authorize implementation or settle pending public compatibility choices.

## Orchestrator decisions

| Decision | Rationale | Cost if wrong / review point |
|---|---|---|
| Preserve all audit records and qualifications; add a separate planning overlay. | Historical observations must remain traceable after fixes. | Closure tracking must be maintained separately; stale links would impair review. |
| Use 12 workstreams and 43 candidate work items, then write bounded briefs only for the selected batch. | A whole-library executable plan would prematurely choose interfaces and algorithms. | Some items will split after design; preserve record ownership when splitting. |
| All workers/reviewers use Astra medium; the orchestrator retains difficult technical decisions. | Implements the user's explicit model/cost preference. | Root becomes a throughput bottleneck; narrow work and resolve decisions ahead of dispatch rather than silently increasing effort. |
| Design V01/O01/R01/I01 first, with at most two implementers later. | Addresses validation, value semantics and silent wrong-result paths with mostly separate files. | Shape/cache decisions remain urgent; this ordering must not defer their design indefinitely. |
| O01 completion does not close all OWN-05 acceptance. O02 provides remaining tier/bulk evidence. | The accepted class-wide matrix spans compact, standard and aligned storage. | Calling OWN-05 fixed before O02 would overclaim closure. Tracking records O02 explicitly. |
| Separate atomic settings updates (E03) from activation (E06). | Atomicity can be repaired before exposing dormant arithmetic/scheduler routes. | If updates reject a candidate, they must preserve all prior fields; mutating then validating is insufficient. |
| Numerical references are independent and range-aware. | Existing entropy/LA helpers can repeat the defect or overflow themselves. | A false oracle could certify a wrong implementation. Double extremes require scaled/exponent-based reference design. |
| Distinguish defect repair from profiling and contract investigation within each task. | The audit contains limitations, opportunities and unresolved hypotheses. | Broad scopes can become rewrites; the dispatch brief must select a testable outcome and exclude unjustified optimization. |

## Audit dependencies versus implementation ordering

The original register connects related findings, sometimes in cycles. It is not a build schedule. The catalog retains the original dependencies in `tracking.json`, adds an acyclic implementation/closure schedule, and uses explicit conditional prerequisites where the relation depends on the design.

| Audit relationship | Planning interpretation |
|---|---|
| API-03/API-07 depend on API-02 | Repair transactions and invalid-input arithmetic before activating live settings: E03/E01 before E06. E06 consumes and rechecks those contracts. |
| API-04 depends on API-24 | Bound scheduling in E02 without requiring P01's allocation optimizations. Use the established bound when measuring per-worker heap memory later. |
| API-16 depends on API-04 | R03 can repair numerical matrix semantics before E02 changes scheduling. Revalidate the integrated serial/parallel route afterward. |
| API-12 depends on API-01/API-02 | D02 follows metric/consumer work and E06; documentation of effective hooks must describe the final routes. |
| API-35 depends on API-11 | D01 explicitly waits for R05 score semantics. Direct VA conventions and the bridge remain separately specified. |
| NUM-23 and OWN-23 depend on numerical/storage repairs | V04's measurement setup can begin early. Final claims about L01/Q02/O02/O03 require those results; conditional prerequisites retain this distinction. |
| NUM-24 depends on OWN-17 | Q02 waits for O05 if introducing mutable conversion scratch or changing ownership/borrowing. Correcting an unsupported reuse claim alone can proceed independently. |
| NUM-27 depends on API-22 | L03 preserves documented Gram limitations unless changing that route. A new exactness/reference policy requires P02 coordination. |
| OWN-04 depends on OWN-07 | I02 requires verified exact identity even with digest collisions. P01's public hash-quality improvement is a separate performance task and cannot substitute for identity checks. |
| OWN-05 depends on OWN-11 | Compact-owner repair does not require the unrelated unused generic alignedCopy utility. O01/O02 still exercise every tier and sentinel lifetime; O04 separately resolves the unsafe copy contract. |
| OWN-12 depends on OWN-05/OWN-14 | HybridStorage has its own owner/mutation paths. O03 need not wait for triangular bulk fixes unless a design deliberately reuses that path; any such reuse adds O02 as a prerequisite before dispatch. |
| OWN-27 depends on OWN-06 | D02 follows S02 transitively through N01/D01. Generic dispatch documentation must describe the repaired guard behavior. |
| SER-03 depends on TOOL-07 | B01's compatible round-trip/test repairs may proceed; binary performance changes/claims require V04 evidence. |
| TOOL-10 depends on TOOL-01 | P02 follows V01 transitively through V02/V04. Scale measurements require a functioning correctness gate. |

## Independent planning review

Two Astra/medium agents reviewed ownership and numerical task boundaries; one also challenged execution/API sequencing. The orchestrator reviewed tooling, serialization, record coverage and the combined dependencies. No reviewer changed library code or ran tests/probes.

The final planning review produced four substantive refinements, incorporated into the documents:

1. Explicit scaled/exponent-based oracles for Double extremes in N01.
2. Conditional O05 prerequisite for Q02 mutable scratch/borrow changes.
3. Staged OWN-05 closure across O01/O02.
4. Candidate validation/publication semantics for E03, preserving old configuration on rejected updates.

The recommendations to preserve compatibility and define a workload envelope remain open for user review. The planning agents do not approve those product decisions.
