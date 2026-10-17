# Benchmark test accounting erratum — October 2, 2026

This root correction supersedes only the parent no-op accounting in V02-review.md, the sealed V02 implementation report, milestone-10-review.md and earlier ledger entries. Historical reviews and sealed worker evidence remain unchanged.

`BenchmarkContractSwiftTestingEntry()` is a real test: it creates an invalid correctness result from NaN, encodes and decodes that result, then asserts the invalid status and absent numeric error. It is not a dispatcher no-op. Its source SHA256 is `fcce209dc5d8dfc3a93150aee80b4f5e380b84c1b5c02d2fb08f37a2dc66b5e4`, matching the V02 base inventory, the unchanged integration source and the milestone10 source inventory.

- V02 benchmark Debug and Release each report 20 XCTest methods, including one parent dispatcher no-op, and one real Swift Testing function. The 41 actual inherited V01 children per configuration are unchanged.
- Milestone10 Debug and Release each report 20 XCTest methods, including one parent dispatcher no-op, and 22 Swift Testing functions, including one parent no-op (`HeapMembershipContractTests.childProbe`). The 41 V01 and three R04 children per configuration are unchanged.
- The separate V02 C-on runs still report 10 XCTest methods and three Swift Testing functions including one Q01 parent no-op, plus 100 actual children per configuration. They are outside this correction.

Root corrected the maintained closure, state and milestone verifier, then reran the offline milestone evidence verifier successfully. Source, binaries, test logs, child counts and runtime results were not changed or rerun. The earlier verifier and JSON output are retained under milestone10 with `.before-accounting-erratum` suffixes. The original independently reviewed pass remains qualified by this explicit root accounting correction.
