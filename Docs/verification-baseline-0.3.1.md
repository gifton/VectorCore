# Verification Baseline — 0.3.1 (`main` @ `0c9e588`)

**Date:** 2026-07-24 · **Machine:** Apple M3 Max, macOS (Darwin 25.5.0) · **Toolchain:** Swift 6.3.2 (swiftlang-6.3.2.1.108)

Supersedes the 2026-06-05 baseline (`rng-verify-full.log`, 996 tests), which predated all of 0.3.0's GEMM/SoA/GPU-seam work and 0.3.1's LAPACK/PCA/UMAP work. Tree under test includes PR #36 (AccelerateArraySIMDProvider + parity suite).

## Results

| Sweep | Scope | Result | Issues |
|---|---|---|---|
| Debug, full | 1117 swift-testing / 165 suites + 91 XCTest (27 skipped) | **green** | 0 |
| Release, full | same | green except 3 | 3, all classified below |
| ASan (debug), full | same, 4491 s | **0 sanitizer reports** | 16 test issues, all timing-class |
| TSan (debug), batch/GEMM/SoA/Matrix filter | 173 tests / 56 suites, 7242 s | **0 sanitizer reports** | 1 test issue (documented known-noise) |

**Memory/thread-safety verdict: clean.** The 0.3.x surfaces called out in the continuation plan all passed under ASan — every `LinearAlgebra` suite (SVD, QR, SymmetricEigen, parity/integration; the LAPACK shim), `UnifiedVectorBuffer / PageAlignedBuffer` (incl. `consumeAllocation()` ownership transfer), PCA, UMAP, and SoA suites. TSan covered GEMM routing, `MatrixDistance`, `BatchKernels SoA`, register blocking, tiled-kernel equivalence, mixed-precision batch kernels, and top-K with zero race reports.

## Issue classification (no code regressions found)

**A. Release-only test bugs — tests assert debug-only behavior (deterministic):**
- `ErrorHandlingTests.swift:1562` "Error description formatting" — expects `.swift` in the description, but the `[at file:line]` suffix is `#if DEBUG`-gated (`VectorError.swift:489`).
- `ErrorHandlingTests.swift:1269` "Dot product dimension mismatch" — expects `context.line > 0`, but source-location capture is `#if DEBUG`-gated (`VectorError.swift:47`).
- Fix: gate both expectations with `#if DEBUG` (test-only change).

**B. Perf assertion that now fails deterministically in Release on M3 Max:**
- `QuantizedKernelsTests.swift:627` `testQuantizedEuclideanDistance` — embedded micro-benchmark asserts INT8 `euclidean512` beats FP32 (`speedup > 1.0`); measures 0.91× reproducibly (3/3 isolated runs). Accuracy expectations green. Likely the 0.3.0 FP32 GEMM/SoA improvements flipped the ratio at dim 512.

**C. Timing assertions under sanitizer slowdown (expected noise, 16 under ASan / 1 under TSan):**
- The four families already documented in the continuation plan (`BatchKernels_SoATests.testBlockingEfficiency`, `QuantizedKernelsTests.testQuantizedBatchPerformance`, MixedPrecision/INT8 tolerance-timing checks), plus the wider set surfaced by this sweep: `QuantizedKernelsComprehensiveTests.swift:941,976`; `MixedPrecisionKernelsTests.swift:227,228,1091,1589–1591`; `MixedPrecisionKernelTests.swift:3681,4132`; `ErrorHandlingTests.swift:1714`; `BatchKernels_SoATests.swift:1086,1589,3079`.

## Recommendation

Convert the class-B and class-C wall-clock assertions into non-blocking benchmarks (report, don't `#expect`) so future sweeps can be strictly green, and apply the two class-A `#if DEBUG` gates. Until then, a sweep is "green" iff its only failures are in the lists above.

Raw logs (not committed): `baseline-0.3.1-release-tests.log`, `baseline-0.3.1-asan.log`, `baseline-0.3.1-tsan-batch-gemm.log` in the session job dir.
