# Q04 independent design review

**ACCEPT design-only.** Reviewed the complete candidate against Q04-design-inventory.md, NUM-22 and O05's approved scoped-borrow migration. No concrete blocker found; no builds/source edits/runtime claims.

The sliced-map mechanism preserves owner lifetime and compact-output semantics: validate original endpoints, select exactly that slice, map immutable storage handles and pass a rebased0..<count. It avoids whole-N metadata work without pretending to eliminate all allocations or copying FP16 payloads. The early return for valid empty ranges avoids cosine query work as well as metadata construction.

Validation order is safe even for Int.min..<Int.max: validate endpoint domain before count subtraction, then capacity, then empty return/work. Since a valid range lies within an actual Array count, count subtraction is representable. Shared named guard/fatalError diagnostics keep normal Release observable. The unsafe pointer's initialization/aliasing/capacity duties remain explicit; invalid-pointer validation is not claimed. Internal cores remain protected by their wrapper caller contract.

Nine-route finite references on actual stored FP16 values, distinct candidate patterns, nonzero starts, odd tails and offset sentinels detect rebasing/compact-output mistakes. Exceptional-value same-core parity is correctly separated from an independent accuracy oracle, preserving existing cosine denominator policy. Bounded invalid-input children require diagnostic identity, while no-write-before-rejection needs an actual observation rather than inference from a crash.

Fixed requested cardinalities with varying unrelated N, preconstructed inputs, identical linked Release fixtures, requested bytes plus events and calibrated controls are adequate to evaluate the metadata-work claim without a production latency target. Baseline optimizer-elimination remains an honest possible outcome requiring root adjudication. Existing value-range tests do not substitute for candidate-range coverage.

Scope is minimal: wrappers, validation helper and touched documentation only. No public signature migration, core arithmetic, diagnostic, SoA or escaped-pointer route is introduced. Q03 prerequisite and serialized Q02/O05 ownership remain required at dispatch; repeat caller inventory on that accepted base. Final implementation/evidence review remains separate.

Reviewed brief SHA256: `28dfd5e2b708bac3179ee9eed4b79e35678de99eb3b97d96f7b1d929c025d5b8`.
