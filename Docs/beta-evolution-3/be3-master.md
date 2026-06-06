Project: VectorCore Refactoring (High-Performance Vector Mathematics)
Goal: Resolve critical memory semantics, numerical instability, and performance bottlenecks identified during a principal-level architectural audit.
Global Constraints & Mechanics
Zero-Regression Mandate: VectorCore is a foundational compute library. Do not alter public API signatures or break downstream consumers.
Type-Based Alias Analysis (TBAA): Never permanently rebind memory owned by Swift standard library collections (e.g., ContiguousArray).
IEEE-754 Semantics: Subnormals, Infinities, and NaNs must be explicitly accounted for in all math kernels. Do not ignore mathematical overflows.
Minimal Invasiveness: Apply surgical fixes to the exact files and lines specified. Do not initiate sweeping rewrites of unrelated components.
🚨 REQUIRED EXECUTION ORDER
The agent MUST implement the changes in the following strict chronological sequence. Proceeding out of order risks masking crashes, introducing noisy performance profiling, or causing tests to infinitely hang.
PHASE 1: Numerical Rigor & Edge Cases (See Document 4)
Why first? Fixing mathematical overflows, NaN poisoning, and memory leaks ensures the test harness will not randomly trap, crash, or leak memory during subsequent phases. This establishes mathematical ground truth.
PHASE 2: Low-Level Swift & SIMD Optimization (See Document 1)
Why second? Resolves Undefined Behavior (UB) caused by strict-aliasing violations and fixes the algorithmic cache-thrashing that is currently starving the CPU memory bus.
PHASE 3: Accelerate Framework Integration (See Document 2)
Why third? Once memory is stable and caches are behaving sequentially, removing heap churn and leveraging vForce will unlock raw hardware throughput.
PHASE 4: Metal Compute Pipeline Prep (See Document 3)
Why last? Foundational memory alignment adjustments for zero-copy GPU bridging rely on the stability established in Phases 1–3.
