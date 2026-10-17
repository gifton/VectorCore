# L03 implementation and evidence

Status: final frozen v3 verification is complete. Scoped source review accepted v3; final evidence is ready for independent review and root integration.

## Source and scope

Base: 17c1af927b6d6b9f343583e34b330ac3c351dac8. The original full archive contains 5,992 regular files and was independently checked against base.tar (SHA256 787b8a639e81272424e3b852f032ceb616e2ba350c37d50be372cbc25e82373d). v3 contains 5,997 files, with 11 changed paths and five additions. The complete-source archive, full unified diff, full and code inventories, and per-file hashes are in v3/. The full diff SHA256 is 0e715eada09e7ca810aeeacb18fa520f696ea8d7dcbd58476e451df37afacdfb.

Production changes are confined to FuzzySimplicialSet.swift, UMAP.swift and KNNGraph.swift. PCA, providers, kernels, routing, package configuration and sibling packages are unchanged. The sole existing test migration is UMAPTests.swift: internal throwing calls and qualified Float Gram wording; existing numerical assertions remain unchanged. Public signatures remain unchanged. New public Swift 6 consumer code uses only public imports.

The binding original brief SHA256 is c489828874bce8ca6beea5056dde8a50b844e4b683a430a94c5f0111a17d4100. numerical-ruling-v3.md retains the root-approved strict compensated objective-change addendum. The original brief remains unchanged in the frozen source archive.

## Implemented numerical contract

Curve fitting uses the dimensionless 300-point target in positive logarithmic parameters, a deterministic Double LM system, a 200-iteration bound and explicit true SSE-gradient infinity norm <=1e-8. The finite strict-decrease test uses Neumaier compensation on `(newF-oldF)*((newF-target)+(oldF-target))`, with separately formed residuals. The returned Float exponent is rounded before physical coefficient conversion; finite positive coefficients must retain dimensionless curve fidelity within 64 Float ULPs on the calibration grid. Unsupported ranges throw rather than silently publishing the initial guess.

Smooth-kNN calibration retains Double rho and sigma through membership evaluation. It counts stored duplicate entries for calibration, then merges duplicate IDs by maximum membership in fusion. Solvable rows use an actual checked logarithmic bracket and a finite residual criterion in at most 64 bisections. Tied/infeasible/equality rows use the explicit zero pre-floor bandwidth; isolated rows remain zero. The existing local/global mean floor can override a solved target and is tested separately. Zero bandwidth gives unit membership only at zero delta. Fusion ordering and Float rounding/underflow policy remain.

Both public entry points validate finite floating configuration, checked output count/bytes and the exact-integer Double scheduling domain before layout allocation/iteration. Supplied initial rows are all shape-checked before copying or checking coordinate values. PCA initialization scales through a widened ratio. SGD preserves the Float equations and clipping policy while rejecting nonfinite intermediates before clipping and publication. Runtime failure may partially mutate an internal inout optimizer buffer; no transactional internal guarantee is added. Public input arrays are checked unchanged.

CSR construction checks every offset against the entry count and monotonicity before any row entry is read. Brute-force KNN checks local LP64/count/byte requirements before allocation or BLAS. Its Float Gram calculation and selection route are unchanged.

## Independent references

L03_generate_references.py imports no production solver/Jacobian. It uses three-start derivative-free Nelder–Mead over the dimensionless objective with math.fsum; stopping criteria, all starts and spread are retained. Seven exact Float ratio families each use five pinned power-of-two spreads; two additional references use the exact Float pairs .5/1.5 and .25/Float(.7). Smooth controls use physical-bandwidth Decimal bisection at 110 and 160 digits, with identical Double-rounded outputs, and retain exact Float input words. Analytic two-/three-level memberships, scale topology, tied/duplicate/all-zero/floor-dominated/isolated/unequal rows are separate controls. Nearly flat roots are judged by membership/residual rather than ill-conditioned sigma identity.

## Retained failures and versions

- RED: unchanged production. Actual scale-dependent curve SSE and smooth memberships failed; nonfinite options were accepted; CSR/dimension/rate children trapped; the Int.max epoch child hit its bounded deadline. Four child captures and exact commands are retained. Expected historical timeout/traps are evidence of defects, not final acceptance.
- v1: first implementation failed compilation on duplicate local `next`; no tests ran. The full source and 2,731 build files remain retained.
- v2: local rename only. Debug ran 38 functions in nine suites, with one existing fitQuality failure at .25/Float(.7). The new suite passed, including nine actual bounded children. All 4,375 build files remain retained.
- Diagnostic: exact v2 arithmetic extracted into standalone Swift with trace prints and message-only error stubs showed a candidate satisfying the gradient criterion rejected by separately rounded SSE totals. Initial sandbox cache failure and successful bounded escalated compile/run are both retained. Python proposal calculations are diagnostic feasibility only, not independent minimization references.
- v3: root-approved compensated objective change, two independent non-dyadic reference controls, public nongeneric DynamicVector correction and explanatory docs. Debug passes 39 functions in nine suites. Source was frozen before compilation; all 5,997 file hashes match afterward. Release, valid ASan and public Swift 6 checks also pass; their exact associations are listed below.

## Coverage and limitations

The final new suite has 14 registered functions: 12 numerical/public-boundary functions, one parent-context childProbe no-op, and one parent exercising nine bounded child scenarios. Counts must distinguish the parent no-op from actual child executions. Each child has file output, a handler installed before launch, 15 seconds plus at most two seconds cleanup, and throws on any timeout before reading termination status/reason. The malformed-offset child checks 19 invalid interior pairs with valid endpoints. Pure count/schedule tests do not allocate giant arrays or run huge accepted loops.

Finite extreme layout inputs may fail explicitly; no all-finite-input layout, global optimum, whole-layout scale equivariance, stochastic convergence, cross-platform bit parity, corpus memory budget, OOM recovery, vendor extreme-shape safety, GPU or performance claim is made. The near-offset Gram fixture reports computed neighbors [1,0,0] and zero distances on this host versus independent Double-difference nearest labels [2,2,0]. This is retained NUM-27 characterization, not an accuracy/routing repair. Robust routing remains deferred.

All task artifacts are canonical outside /tmp under .stabilization-recovery/2026-10-02/L03. Each tested source and scratch is separate and retained. The sealed worker index excludes root/reviewer records and itself. Input dispatch records and immutable base archives are retained as provenance.

## Final verification and exact associations

All final runtime evidence applies to frozen v3, not v1/v2. The current approved brief SHA256 is a57c9beaf4b6a3e0d7edf9d664bb32162f893834d87a25a4387da2d0df202e07; approved-brief-with-v3-addendum.md and brief-association.json bind it additively without rewriting the original frozen brief or historical manifests.

| Check | Result and coverage | Exact records |
| --- | --- | --- |
| Debug | Exit 0; 39 functions in 9 suites; 9 actual rejection children | v3/debug-command.json, debug.log, debug-children.json |
| Release | Exit 0; 39 functions in 9 suites; 9 actual rejection children | v3/release-command.json, release.log, release-children.json |
| Valid ASan via SwiftPM | Exit 0; exactly 12 functions in 1 suite; no child dispatcher or parent selected | v3/asan-command.json, asan.log |
| Direct ASan host proof | Exit 0; supplemental execution of the same 12 valid functions, not extra distinct coverage | v3/direct-asan/provenance.json and run.log |
| Public Swift 6 | Compile/link exit 0, executable exit 0; both public routes, default/Swift provider, typed boundary matrix and immutable input | v3/public/positive-compile.json, positive-run.json, inputs.json |
| Independent fixtures | Exit 0; 7 ratio families, 2 non-dyadic pairs, 3 starts each; 7 smooth rows at 110/160 digits | v3/reference-check.json, reference-check.log, fixture-scanner-commands.json |
| Maintained child scanner | Seven unittest checks pass, including repository scan | v3/scanner.log, fixture-scanner-commands.json |

Debug/Release report 39 registered functions: 38 exercised parent functions plus the childProbe parent-context no-op. The parameterized rejection parent contains nine actual child executions per mode (18 final ordinary children total). Each actual child executes childProbe with its own selected scenario. The ASan filter selects exactly the 12 numerical/public-boundary functions; it contains neither of those parent entries and uses no skip flag. The direct ASan replay is supplemental host/bundle proof, not an inflated function count. Historical RED four children and v2 nine children remain distinct from final acceptance.

The public consumer links 131 actual frozen v3 Release objects; 138 inputs including module sidecars, fixture, driver and executable are hashed in public/inputs.json. public-artifact-association.json ties these bytes to the source manifest and Release inventory. The original driver call waited for tool approval; pending functions cell 776 resumed as exec session 86127 and completed without duplicate launch. Its exact returned stdout/status is retained in public-driver.log/public-driver-command.json, while each compiler/executable child independently retained its file output, argv, PID and exit.

The test host is SwiftPM's swiftpm-testing-helper, not the loaded test code. Debug/Release child argv identify the actual VectorCorePackageTests Mach-O bundle, and debug-children.json/release-children.json bind both hashes; helper bytes are separately retained. The direct ASan record gives the actual argv, PID, bundle path/hash, framework/library paths and DYLD_INSERT_LIBRARIES preload. Both helper and the actual clang 21.0.0 ASan runtime are retained in direct-asan/. The bounded driver source is direct-runtime-v3.py. No framework path or host identity is inferred from a bundle directory name.

The six separate retained build inventories contain 4,375 RED files; 2,731 failed-v1 files; 4,375 v2 files; and v3 Debug 4,375, Release 2,897, ASan 4,376 files. These are actual retained bytes, not hashes referring to overwritten scratches. final-artifact-verification.json rechecks every inventory and all 5,997 frozen source entries. full.diff includes every new source/fixture/driver path. final-command-associations.json binds version, command, status and log hashes; no failed run is relabeled successful. The initial setup typo and Swift cache-denial attempts remain retained with their failure records.

Host qualification: Apple Swift 6.3.3, arm64 macOS 26.5.2, as captured in host-toolchain.json. This is focused CPU verification, not a universal platform, numerical-range, timing or downstream acceptance claim. The full artifacts, source snapshots and failed attempts remain canonical under /Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-02/L03. Shared reports receive copies only; no shared production or index/commit changes were made by this worker.
