# Actual unchanged-base RED

Production base101b3749bf531cfdbc42d61198b503e94af4aa5b. Added LinearAlgebraRangeTests.swift only. Command: `swift test --jobs 6 --scratch-path /private/tmp/vectorcore-L01-red-debug --filter LinearAlgebraRangeTests`. Working directory is the canonical L01 archive. Initial sandbox launch exited1 before compiling due to denied compiler cache. Escalated identical command exited1 after a successful build: four actual Swift Testing functions, 18 issues. XCTest wrapper reported zero tests separately.

QR [s,s] at1e-25 returned Q[1,0], R[s]; at1e20 returned NaN Q and -Inf R. Single-column SVD returned spectrum0 with unnormalized U at1e-25 and Inf with zero U at1e20. Golden full-rank SVD returned spectrum[sqrt2,1]*s and nonorthogonal U at both1e-12 and1e15. Reconstruction alone did not reject this golden fixture. The additional eigen1e20 control threw obsolete Frobenius invalidData. These are actual numerical failures, distinct from source-established C workspace hazards.

The entire scratch, actual executable/object/module bytes, logs and test source are copied before any rebuild to `.stabilization-recovery/2026-09-24/L01/red`. SHA256 inventory is L01-red-artifacts.json. These four finite, tiny inputs ran in-process; no unsafe pointer, nonfinite input, or huge-vendor probe was executed.
