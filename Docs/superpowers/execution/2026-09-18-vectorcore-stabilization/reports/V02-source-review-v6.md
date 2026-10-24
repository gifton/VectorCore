# V02 frozen-v6 scoped source review

Disposition: **ACCEPT — corrected source only.** No blocker in the fixture-only v5→v6 delta. Four-cell final runtime/evidence adjudication remains pending; no builds or probes were rerun.

Verified patch SHA256 `7306a8e830c6d06191251092985aa5fd1df35c1671df741dfee040b4fb8234df`, manifest `eec629ce6f6d80999d6565b96d0362f8d9dc849d3b50f70cc2c40f9b87a3c1be`, unchanged amended brief `5a1ca34f5a51833f177288d1a3054fda602bf1568ac2a80747413a1502af90e1`, and all31 final owned-file hashes. Complete owned-file comparison against v5 finds identical path sets and exactly one changed file, `Scripts/ci/fixtures/V02/public_routes.swift`; the other30 files are byte-identical.

## Correction and retained review history

The earlier source reviews missed that QuantizedKernels is internal despite its methods being accessible to in-module tests. The external consumer's direct references could not compile with ordinary import VectorCore. The subsequently reported v5 compile failure is a real fixture defect, and supersedes the earlier favorable public-consumer source disposition. It is not a failed numerical route or evidence of a Core regression. Preserve the original reviews and failed compile records; this correction does not retroactively make v5 executable.

The new fixture calls the actual public concrete `euclideanDistance(to:)`, `dotProduct(_:)` and `cosineDistance(to:)` methods at512/768/1536. Inspected QuantizedKernels.swift:965–1012: each public method directly forwards to its corresponding unchanged internal kernel. These are the intended public entry points, preserve the exact metric semantics and C-gate reachability, and require neither @testable nor a Core visibility change. This now also meets the brief's public-route requirement more directly.

The code-generator map is replaced by a typed loop with exactly the same candidate/query integer expressions in the same coordinate order. Its actual fixed width inputs keep arithmetic and Int8 conversion bounded. Storage packing, valid nonzero-zero-point fixture, independent Int64/Double oracle, finite-aware comparisons, reset/read sentinels,36 route cases and10 direct-tail controls are unchanged. No threshold, gate, numerical kernel, driver or build-mode policy changes.

Root reports a retained-object Debug C-off preflight passing36 routes/10 tails. That is useful fixture-validation evidence but is not final all-four-cell acceptance and was not rerun here. Final evidence must retain its exact source/object association and separately establish v6 Debug/Release × C-off/on completion with actual public consumer compile/link/run, forwarding counts, numerical checks and immutable ordinary build artifacts. Previous compile/discovery failures must remain setup history with no fabricated route completion.
