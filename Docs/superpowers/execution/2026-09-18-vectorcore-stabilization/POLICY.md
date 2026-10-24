# Current execution policy

The user's implementation authorization is preserved in ORIGINAL_EXECUTION_REQUEST.md. In the resumed session the user additionally stated: "i am completely fine with breaking changes - we can talk about performance targets and downstream intent later."

- Breaking source/API changes are authorized where they improve the stabilization design. Root decides and records the contract and migration consequences; another compatibility approval is not required.
- Workload, downstream intent and latency/memory targets remain unselected. Correctness and demonstrated work-amplification repairs proceed; do not claim production scale readiness.
- All delegated implementers and reviewers use gpt-6-astra, medium reasoning, fork_turns none. No nested delegation. At most two implementers and a reserved reviewer slot.
- Root owns local commits/index operations. Workers edit only their assigned files and do not commit. Review packages contain full task diffs, including newly created files.
- Retain historical audit/review/planning artifacts unchanged. This execution directory is the persistent record and is not deleted at completion.
- Keep changes local: no push, merge, release, publication or sibling-repository writes. VectorAccelerate can continue its independent SP0 work; downstream runtime validation remains separate.
- Preserve platform floors and avoid new third-party dependencies. Preserve existing SoA layout/ownership and defined Top-K NaN/tie contracts unless root explicitly records a necessary contract change.
- Any correctness/robustness claim in comments must name a test, follow directly from adjacent code, or be explicitly unverified. Close repeated defect classes mechanically. Re-derive reachability whenever a routing/configuration gate changes.
