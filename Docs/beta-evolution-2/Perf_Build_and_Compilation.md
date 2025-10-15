# VectorCore 0.2.0 — Build & Compilation

Focus: produce faster release binaries without sacrificing maintainability.

This document is the authoritative source for build flags, presets, and size/perf trade‑offs. Other docs should reference this to avoid drift.

## Goals
- Maximize cross‑module and whole‑module optimization benefits.
- Trim code size after more aggressive inlining/emit‑into‑client.
- Provide predictable, flag‑driven build presets for benching vs shipping.

## Opportunities

- Whole‑/Cross‑Module Optimization
  - Enable CMO for Release builds: SwiftPM `swiftSettings: [.unsafeFlags(["-cross-module-optimization"], .when(configuration: .release))]`.
  - Ensure WMO is active (default for library targets in Release).

- Library Evolution Toggle
  - Internal builds: evolution OFF for maximal optimization.
  - Optional evolution‑ON product for ABI stability; benchmark deltas.

- Link‑Time Optimization & Dead Stripping
  - Linker flags: `-Xlinker -dead_strip -Xlinker -dead_strip_dylibs`.
  - For future C kernels, enable Clang LTO where supported.

- Dual Release Presets (SwiftPM reality)
  - SwiftPM doesn’t have named custom configurations; model presets via scripts/Makefile/CI that pass flags.
  - `perf` preset: speed‑first (`-O`, CMO on, dead_strip, optional underscored attrs via `VC_ENABLE_UNDERSCORED=1`).
  - `size` preset: size‑first (`-Osize`, dead_strip, minimize emit‑into‑client/transparent usage).

- Arch‑Specific Builds (future C kernels)
  - arm64: NEON/SDOT dispatch; x86_64: AVX2 baseline, AVX‑512 optional.
  - Runtime CPU feature selection; keep a portable fallback.

- Package Flags & Presets
  - Centralize: `VC_ENABLE_UNDERSCORED`, `VC_ENABLE_FFI`, `VC_USE_C_KERNELS`.
  - Provide a single "perf" preset for benches that toggles these consistently.

## Presets & Tooling

- Make targets or scripts:
  - `make build-perf`: `swift build -c release` with CMO and dead_strip; exports `VC_ENABLE_UNDERSCORED=1` (optional).
  - `make build-size`: `swift build -c release` with `-Osize` and dead_strip; disables underscored emission.
  - `make bench-perf`: builds benches with perf preset and runs the configured matrix.
- CI jobs:
  - Build and record binary size for both presets; run benches and upload JSON results.
  - Maintain A/B runs that flip feature flags (underscored, C kernels, SoA preference).

- Codegen Hygiene
  - Ensure benches run optimized (Release, no `-enable-testing`).
  - Audit `@inlinable` on tiny cross‑module glue; prefer narrow scope.

## Validation
- Bench throughput vs `Release` baseline.
- Binary size deltas for ReleasePerf/ReleaseSize.
- Sanity: public API/ABI checks for evolution‑ON product.

## Checklist
- [ ] Add CMO flag to Package.swift (Release only)
- [ ] Add perf/size presets via Makefile/scripts and CI jobs
- [ ] Wire linker dead‑strip flags for production benches
- [ ] Introduce perf preset toggling feature flags
- [ ] Record size/perf metrics per preset and upload alongside bench results
