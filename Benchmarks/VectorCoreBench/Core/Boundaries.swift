import Foundation
import VectorCore
import VectorCoreBenchmarking

/// The single source of every catalog case's measurement boundary.
///
/// Every `BenchmarkCase` takes its record from `Boundaries.record(_:_:)` where the catalog creates the case, and
/// `CaseExecution` attaches that record to the case's results. The records are written by hand from reading the
/// bodies; nothing derives them from the bodies. What is mechanically checked (class `BenchmarkBoundaryDriftContractTests` in
/// `BenchmarkContractTests.swift`, and the catalog-table test in `BenchmarkSelectionContractTests`):
/// - every catalog case in every preset has a structurally valid boundary of the current version;
/// - an executed case reports the declared `unitsPerInvocation` (otherwise `CaseExecution` fails the case);
/// - `bodyFingerprints`: the comment-stripped, whitespace-run-collapsed source of every named declaration (`func`
///   overloads and computed `var`s) hashes to the recorded SHA-256, and every listed symbol must exist. The declarations
///   are the catalog functions, the `run*` helpers, the allocator functions and `ProviderObservations`/`OutputFold`
///   members that execute inside the warmup and timed bodies of `Suites/*` (`testBodyFingerprintsMatchRecordedHashes`);
/// - the sink calls in each case's bodies come from a closed vocabulary that maps to the declared `consumer`
///   (`testConsumerTokensMatchDeclaredConsumers`);
/// - each case's warmup and timed bodies run the same closure or helper (modulo the listed differences)
///   (`testWarmupAndTimedBodiesRunTheSameWork`).
///
/// NOT covered: code outside those declarations that also runs inside a timed body (`blackHole`, the `Harness` loops and
/// clock reads, `Boundaries` itself, library calls such as the kernels and `alignedCopy`); whether the inclusion flags,
/// routes or notes are still true after an edit. Updating a fingerprint is an assertion, not proof, that the record was
/// re-read against the new body.
enum Boundaries {
    static let fixtureVersion = "V02-inputs-1"
    static let lcgValues = "uniform[-1,1) LCG (InputFactory.randomArray)"

    // MARK: Inclusion helpers

    private typealias I = Inclusion

    private static func inclusions(
        construction: I = .excluded, preparation: I = .notApplicable, packing: I = .notApplicable,
        callerOutput: I = .notApplicable, internalAllocation: I = .notApplicable,
        scheduling: I = .notApplicable, observation: I = .notApplicable, kernel: I = .included,
        consumption: I = .included, validation: I = .excluded
    ) -> BoundaryInclusions {
        BoundaryInclusions(inputConstruction: construction, inputPreparation: preparation, routeInternalPacking: packing,
                           callerOutputAllocation: callerOutput, routeInternalAllocation: internalAllocation,
                           providerScheduling: scheduling, providerObservation: observation, kernelWork: kernel,
                           outputConsumption: consumption, correctnessValidation: validation)
    }

    private static func fixture(id: String, dimension: Int, count: Int, seeds: [String: String] = [:],
                                values: String = lcgValues, preparation: String = "none") -> FixtureIdentity {
        FixtureIdentity(id: id, version: fixtureVersion, dimension: dimension, count: count, seeds: seeds, values: values,
            preparation: preparation)
    }

    private static func genericType(_ dim: Int) -> String { "Vector<Dim\(dim)>" }
    private static func optimizedType(_ dim: Int) -> String { "Vector\(dim)Optimized" }

    // MARK: Body drift guard (fingerprints)

    /// SHA-256 of the comment- and whitespace-stripped source of the helpers that implement a family's warmup and timed
    /// bodies (`BodySource.strippedSource`). A body edit changes the hash (`testFingerprintNegativeControls`); updating it
    /// asserts the family's record was re-read against the new body. `file` is relative to `Benchmarks/VectorCoreBench`.
    struct BodyFingerprint: Sendable {
        let family: String
        let file: String
        let symbols: [String]
        let sha256: String
    }

    static let bodyFingerprints: [BodyFingerprint] = [
        .init(family: "dot",
              file: "Suites/DotProductBench.swift",
              symbols: ["catalog"],
              sha256: "d3eb21ca41deceeaf59f550873c06d13328ecb138f0ab6b0bc6569d41cfd1992"),
        .init(family: "distance",
              file: "Suites/DistanceBench.swift",
              symbols: ["catalog", "catalog512", "catalog768", "catalog1536"],
              sha256: "b5128c49200a9e2db1e4091fa045f507a52230d6f32e9f577fcdbadc0b120e34"),
        .init(family: "normalize",
              file: "Suites/NormalizationBench.swift",
              symbols: ["catalog", "catalog512", "catalog768", "catalog1536"],
              sha256: "dbe7948200595fb1609c05817ca0fabf61ed85caa1f87abf0787a60a71f601e3"),
        .init(family: "noop",
              file: "Suites/NoopBench.swift",
              symbols: ["catalog"],
              sha256: "75af337f83f8f07032993ba56c1e3c5e88f36db3e98a164672d76bb6b55d2d26"),
        .init(family: "matrix",
              file: "Suites/MatrixDistanceBench.swift",
              symbols: ["catalog", "measure", "consumedValue", "inputs"],
              sha256: "93e4e68aee2e7e8bc23e9b059b9d06bc0cbd07b2eaa8130e6ec754e67301ad96"),
        .init(family: "memory",
              file: "Suites/MemoryBench.swift",
              symbols: ["catalog", "alignedAllocate", "alignedRelease"],
              sha256: "081006cf564ee7550dbbdb6b81311da51907744bb4fcad18336aeffc80a3fe5a"),
        .init(family: "boundary",
              file: "Suites/BoundaryBench.swift",
              symbols: ["catalog", "caseName", "matrixConsumerName", "runMeasured", "validated", "validatedMatrix", "entries", "d512",
                  "d768", "d1536"],
              sha256: "32eae29409ce0209cb65eec9073820a55a5be74c00acae6ee4b52630937d2765"),
        .init(family: "batch.fixtures",
              file: "Core/BatchFixtures.swift",
              symbols: ["makeFixture"],
              sha256: "c37c8256369c02164c3ea5a9b3656cb5e53b8a1ca176a37cb14cd073734964e7"),
        .init(family: "fold",
              file: "Utils/OutputFold.swift",
              symbols: ["fold"],
              sha256: "0c5bc73b5aeabda9bb48260400aeeb8ddfbb1db01e53ede516eeb78d5a9a660d"),
        .init(family: "batch.catalog",
              file: "Suites/BatchBench.swift",
              symbols: ["catalog", "catalog512", "catalog768", "catalog1536", "providers"],
              sha256: "d2d20c09dfdeafa67f47aa1b57dcf4bacc93f04096ea5cc83297be0c15c8633f"),
        .init(family: "batch.observations",
              file: "Core/ProviderObservations.swift",
              symbols: ["observe", "statistics", "combining"],
              sha256: "28dfe486f2794c448c7429269594fa2aa57c7803b21d2ed9d56886d31f46be06"),
        .init(family: "batch.provider.euclid2.generic",
              file: "Suites/BatchBench.swift",
              symbols: ["runBatchGeneric"],
              sha256: "a6d6a309639b68ccfc40a57c778579c3c28249cecdaaa3bd8c5a6623cd69b030"),
        .init(family: "batch.provider.euclid.generic",
              file: "Suites/BatchBench.swift",
              symbols: ["runBatchGenericEuclid"],
              sha256: "77231b48c94e1c383ea8c78571e1f4c24750637a158fb57c4e8dc3b64d60dee0"),
        .init(family: "batch.provider.euclid2.optimized",
              file: "Suites/BatchBench.swift",
              symbols: ["runBatchOptimized"],
              sha256: "9cc5cdd3a652f579c46058315b293b10f8fddc18a6950ca64a747c977ab4d050"),
        .init(family: "batch.provider.euclid.optimized",
              file: "Suites/BatchBench.swift",
              symbols: ["runBatchOptimizedEuclid"],
              sha256: "b2a65ea90fb96f9248eb9350f63f5ac4e2f3f79c8dc34ba872acf5f62b4ed959"),
        .init(family: "batch.provider.cosine.fused",
              file: "Suites/BatchBench.swift",
              symbols: ["runBatchOptimizedCosineFused"],
              sha256: "cff60458082e2abd93342b1a7268412e62899519e8ca728c2a5aedd7c9d5a307"),
        .init(family: "batch.provider.cosine.preNorm",
              file: "Suites/BatchBench.swift",
              symbols: ["runBatchOptimizedCosinePreNorm"],
              sha256: "bd763e84100d7a8cb91249cfe720831a3f1d66c023891c5c93e98b52f7437d03"),
        .init(family: "batch.soa",
              file: "Suites/BatchBench.swift",
              symbols: ["runBatchSoA512", "runBatchSoA768", "runBatchSoA1536"],
              sha256: "db8a713ff4d285525fbed9c5f3f8d323afbacdeca0989ecb48c10a3ff2427ca7"),
        .init(family: "batch.fp16.euclid2",
              file: "Suites/BatchBench.swift",
              symbols: ["runBatchFP16_512", "runBatchFP16_768", "runBatchFP16_1536"],
              sha256: "619b17df0b6bbe956c995a85d8a9d364f1f29dc261facc3d8398816ef1de245e"),
        .init(family: "batch.fp16.cosine",
              file: "Suites/BatchBench.swift",
              symbols: ["runBatchCosineFP16_512", "runBatchCosineFP16_768", "runBatchCosineFP16_1536"],
              sha256: "7df442d31ef6e17dc1464426431e063cfd98e8f111990fc347aaaf6e9af65bdd")
    ]

    private static let batchFamilies: [String: [String]] = [
        "euclidean2.generic": ["batch.catalog", "batch.fixtures", "batch.observations", "batch.provider.euclid2.generic"],
        "euclidean.generic": ["batch.catalog", "batch.fixtures", "batch.observations", "batch.provider.euclid.generic"],
        "euclidean2.optimized": ["batch.catalog", "batch.fixtures", "batch.observations", "batch.provider.euclid2.optimized"],
        "euclidean.optimized": ["batch.catalog", "batch.fixtures", "batch.observations", "batch.provider.euclid.optimized"],
        "cosine.optimized-fused": ["batch.catalog", "batch.fixtures", "batch.observations", "batch.provider.cosine.fused"],
        "cosine.optimized-preNorm": ["batch.catalog", "batch.fixtures", "batch.observations", "batch.provider.cosine.preNorm"],
        "euclidean2.optimized-soa": ["batch.catalog", "batch.fixtures", "batch.soa"],
        "euclidean2.optimized-fp16": ["batch.catalog", "batch.fixtures", "batch.fp16.euclid2"],
        "cosine.optimized-fp16": ["batch.catalog", "batch.fixtures", "batch.fp16.cosine"]
    ]
    private static let kindFamilies: [String: [String]] = [
        "dot": ["dot"], "dist": ["distance"], "normalize": ["normalize", "fold"], "noop": ["noop"],
        "matrix": ["matrix", "fold"], "mem": ["memory"]
    ]

    /// Body-fingerprint families that implement the named case's bodies.
    static func bodyFamilies(forCaseName name: String) -> [String] {
        let parts = name.split(separator: ".").map(String.init)
        guard let kind = parts.first else { return [] }
        if kind == "batch", parts.count == 5, BoundaryBench.Variant(rawValue: parts[4]) != nil {
            return ["boundary", "batch.fixtures", "fold"]
        }
        if kind == "matrix", parts.last == "consumer-only" { return ["boundary", "matrix", "fold"] }
        if kind == "batch" { return parts.count >= 5 ? batchFamilies[parts[1] + "." + parts[4]] ?? [] : [] }
        return kindFamilies[kind] ?? []
    }

    // MARK: Entry point

    /// Boundary for a catalog case name. Traps for a name that has no declared family: a case without a
    /// boundary must not exist, and the contract tests enumerate the catalogs to prove none does.
    static func record(_ name: String, _ options: ValidatedOptions) -> MeasurementBoundary {
        guard let boundary = lookup(name, runSeed: options.runSeed) else {
            preconditionFailure("No measurement boundary family declared for case name \(name)")
        }
        return boundary
    }

    static func lookup(_ name: String, runSeed: UInt64) -> MeasurementBoundary? {
        let parts = name.split(separator: ".").map(String.init)
        guard let kind = parts.first else { return nil }
        let probe = AllocationProbe.resolutions[name]
        switch kind {
        case "dot": return parts.count == 3 ? dot(dim: Int(parts[1]), variant: parts[2], runSeed: runSeed, probe: probe) : nil
        case "dist":
            return parts.count == 4
                ? distance(metric: parts[1], dim: Int(parts[2]), variant: parts[3], runSeed: runSeed, probe: probe) : nil
        case "normalize":
            return parts.count == 4 ? normalize(status: parts[1], dim: Int(parts[2]), variant: parts[3], probe: probe) : nil
        case "batch": return batch(parts, runSeed: runSeed, probe: probe)
        case "matrix": return matrix(parts, runSeed: runSeed)
        case "mem": return memory(parts, probe: probe)
        case "noop": return name == "noop.scalar-add" ? noop() : nil
        default: return nil
        }
    }

    /// Boundary for results that did not come from a production catalog (injected test fixtures). A production registry
    /// rejects this scope (`testFixtureInjectedScopeIsRejectedInProductionRegistry`).
    ///
    /// Not verified: the runtime note's "never paired or compared" holds for comparison only because production
    /// registries reject this scope, while `Scripts/compare_benchmarks.swift` accepts it in a report (Q11 runtime text,
    /// kept unchanged because it is part of the boundary's identity; claim ledger V04-C-0607, D02 RD-R3).
    static func fixtureInjected(name: String) -> MeasurementBoundary {
        MeasurementBoundary(
            timingScope: .fixtureInjected, operation: "test.injected-fixture", route: "test fixture result; not a production catalog case",
            provider: nil, workUnit: "invocation", unitsPerInvocation: 1, inputRepresentation: "none", outputRepresentation: "none",
            fixture: fixture(id: "test-injected", dimension: 0, count: 0, values: "none"),
            inclusions: inclusions(construction: .notApplicable, kernel: .notApplicable, consumption: .notApplicable,
                validation: .notApplicable),
            consumer: "none",
                notes: ["Never paired or compared; results named differently from the case are accepted only for this adapter."])
    }

    // MARK: V04 phase D probe results

    /// A case's resolved `routeInternalAllocation` value and the note that names its record.
    struct AllocationResolution: Sendable, Equatable {
        let value: Inclusion
        let note: String
    }

    /// Resolved `routeInternalAllocation` values and notes from V04 phase D (probe D4), and FP16 notes from T12's rerun of that
    /// probe.
    ///
    /// Not verified in this repository: the runtime note text below cites records kept outside the repository
    /// (`V04/phaseD/D4/<set>/manifest.json` and `T12/phase2/D4/final/<set>/manifest.json`, 2026-10-04). Each probe counted
    /// allocation events (malloc_logger, every thread) in each case's timed window at 16 and at 32 fixed repeats, on one
    /// Release build (arm64, macOS 27.0.1), for the case IDs in `resolutions` and the FP16 cases its notes name only; a later
    /// library change can make a resolved value or a note stale.
    ///
    /// `testAllocationFlagsDoNotOverclaim` pins the resolved case IDs and values, the unprobed neighbours that keep
    /// `unverified`, and, for each resolved note, the one record set it names and its observation, against a table the
    /// test writes by hand from the D4 counts; it also checks literal counts and record paths in the FP16, generic-pair
    /// and unprobed-SoA notes. The test reads no record, so it cannot detect a record that disagrees with both.
    enum AllocationProbe {
        /// Resolved cases by exact case ID; every other case keeps its family's value. Not verified in this repository: the
        /// runtime note text built here cites probe records kept outside the repository.
        static let resolutions: [String: AllocationResolution] = {
            let none = "no allocation events"
            let tiled = "32 allocation events of 4,096 bytes per invocation; this N takes the SoA kernel's tiled path (N > 4096 at "
                + "D = 512), and 4,096 bytes matches one 256-candidate tile (attribution by size is an inference)"
            let metrics = ["euclidean", "cosine", "manhattan", "dot", "chebyshev", "hamming", "minkowski"]
            let scalar: [String] = ["dot.512.generic", "dot.512.optimized"]
                + metrics.flatMap { ["dist.\($0).512.generic", "dist.\($0).512.optimized"] }
            let rows: [(ids: [String], value: Inclusion, record: String, observation: String)] = [
                (["batch.euclidean2.512.N4096.aos-fp32-reused", "batch.euclidean2.512.N4096.soa-fp32-reused"], .notApplicable,
                 "primary", none),
                (["batch.euclidean2.512.N4096.aos-fp32-setup"], .notApplicable, "primary",
                 "1 allocation event per invocation, of 16,416 bytes: by size, the [Float](N) output counted under "
                     + "callerOutputAllocation (attribution by size is an inference); no other event"),
                (["batch.euclidean2.512.N4096.optimized-soa", "mem.copy.512"], .notApplicable, "unverified-a", none),
                (["batch.euclidean2.512.N8192.soa-fp32-reused", "batch.euclidean2.512.N8192.optimized-soa"], .included,
                 "unverified-a", tiled),
                (["normalize.zeroFail.512.generic", "normalize.zeroFail.512.optimized"], .included, "unverified-a",
                 "3 allocation events per invocation (136, 64 and 96 bytes)"),
                (["normalize.success.512.generic"], .included, "unverified-a", "2 allocation events per invocation (2,048 and 40 bytes)"),
                (["normalize.success.512.optimized"], .included, "unverified-a", "1 allocation event per invocation (2,080 bytes)"),
                (["mem.pool.acquire.512"], .included, "unverified-a",
                 "7 allocation events per invocation (five of 40 bytes, two of 80 bytes)"),
                (scalar, .notApplicable, "unverified-b", none)
            ]
            var table: [String: AllocationResolution] = [:]
            for row in rows {
                for id in row.ids {
                    table[id] = AllocationResolution(
                        value: row.value, note: "routeInternalAllocation \(row.value.rawValue): \(row.observation) "
                        + "in the timed window (V04 phase D allocation-event probe at 16 and at 32 invocations, record "
                        + "V04/phaseD/D4/\(row.record)/manifest.json, 2026-10-04; this case ID only).")
                }
            }
            return table
        }()

        /// Second note on SoA cases (both SoA routes) without a resolution.
        static let soaUnprobed = "The V04 phase D allocation-event probe measured the SoA routes at D = 512 only: no allocation events "
            + "in the timed window at N = 4096, and 32 events of 4,096 bytes per invocation at N = 8192, where the kernel takes its "
            + "tiled path (N > 4096 at D = 512) (records V04/phaseD/D4/primary/manifest.json and "
            + "V04/phaseD/D4/unverified-a/manifest.json, 2026-10-04). This (D, N) was not probed."

        /// Notes on the boundary-suite `aos-fp16-reused` and `aos-fp16-setup` rows at every width and N; each names the D and
        /// N its counts come from.
        static let fp16Reused = "At D = 512, 768 and 1536 with N = 4096 the T12 rerun of the V04 phase D allocation-event probe "
            + "recorded 1 event per invocation in the timed window, of 32,800 bytes (by size, the handle array; attribution by size "
            + "is an inference), and no other event (record T12/phase2/D4/final/fp16-boundary/manifest.json, 2026-10-04)."
        static let fp16Setup = "At D = 512, 768 and 1536 with N = 4096 the T12 rerun of the V04 phase D allocation-event probe "
            + "recorded 4,099 events per invocation in the timed window: 4,096 of 1,056 bytes at D = 512 (1,568 at D = 768, 3,104 "
            + "at D = 1536), two of 32,800 bytes and one of 16,416 bytes (record T12/phase2/D4/final/fp16-boundary/manifest.json, "
            + "2026-10-04)."

        /// Note on the default `optimized-fp16` rows, which T12's rerun of the V04 phase D probe measured at one N per width. Not
        /// verified in this repository: the returned note text cites probe records kept outside the repository.
        static func fp16DefaultRow(metric: String) -> String {
            let route = metric == "cosine" ? "range_cosine_mixed_*" : "range_euclid2_mixed_*"
            return "For \(route), the T12 rerun of the V04 phase D allocation-event probe recorded 1 allocation event per "
                + "invocation in the timed window, of 65,568 bytes at D = 512 and 768 with N = 8192 and of 32,800 bytes at D = 1536 "
                + "with N = 4096 (by size, the handle array; attribution by size is an inference), plus one 1,016-byte event per "
                + "timed window that does not grow with the invocation count (attributed by call site to swift_task_alloc at the "
                + "harness's awaited call in the timed loop) (records T12/phase2/D4/final/fp16-default/manifest.json and "
                + "T12/phase2/runs/attr-final-*/manifest.json, 2026-10-04). Other N were not probed."
        }

        /// Note on both sides of the generic `euclidean2_vs_euclidean` pairs whose squared side the probe measured (D = 512,
        /// N = 4096, sequential and parallel); nil elsewhere. Not verified in this repository: the returned note text cites a
        /// probe record kept outside the repository.
        static func genericPair(dim: Int, n: Int, provider: String) -> String? {
            guard dim == 512, n == 4096 else { return nil }
            let observed: String
            switch provider {
            case "sequential":
                observed = "12,290 allocation events per invocation: 3 per candidate (2,048, 2,080 and 40 bytes) plus one of 104 "
                    + "and one of 36 bytes"
            case "parallel":
                observed = "199,356 allocation events at 16 and 398,720 at 32 invocations, among them one 2,048-byte and one "
                    + "2,080-byte event per candidate per invocation"
            default:
                return nil
            }
            return "The V04 phase D allocation-event probe measured only the squared-Euclidean side of this pair "
                + "(batch.euclidean2.512.N4096.generic.\(provider)): \(observed) in the timed window (record "
                + "V04/phaseD/D4/unverified-a/manifest.json, 2026-10-04). The sqrt side was not probed, and both sides of a "
                + "declared pair carry the same inclusions, so routeInternalAllocation stays unverified on both."
        }
    }

    // MARK: Scalar suites

    private static func dot(dim: Int?, variant: String, runSeed: UInt64, probe: AllocationResolution?) -> MeasurementBoundary? {
        guard let dim, ["generic", "optimized"].contains(variant) else { return nil }
        let seeds = InputSeeds.dot(dim: dim, runSeed: runSeed)
        let type = variant == "generic" ? genericType(dim) : optimizedType(dim)
        return MeasurementBoundary(
            timingScope: .reusedInput, operation: "scalar.dot", route: "\(type).dotProduct(_:) inside withExtendedLifetime",
            provider: nil, workUnit: "invocation", unitsPerInvocation: 1,
            inputRepresentation: "fp32-vector-pair:\(type)", outputRepresentation: "scalar:Float",
            fixture: fixture(id: "V02-dot-pair", dimension: dim, count: 2, seeds: ["a": String(seeds.a), "b": String(seeds.b)]),
            inclusions: inclusions(internalAllocation: probe?.value ?? .unverified, validation: .notApplicable),
            consumer: "scalar-sink-v1",
            notes: ["No out-of-band validation exists for this case (correctnessValidation is notApplicable).",
                    probe?.note
                        ?? "Whether the library call allocates is not established from the benchmark source (routeInternalAllocation "
                        + "unverified)."])
    }

    private static func distance(metric: String, dim: Int?, variant: String, runSeed: UInt64,
                                 probe: AllocationResolution?) -> MeasurementBoundary? {
        guard let dim, ["generic", "optimized"].contains(variant) else { return nil }
        let types = ["euclidean": "EuclideanDistance", "cosine": "CosineDistance", "manhattan": "ManhattanDistance",
                     "dot": "DotProductDistance", "chebyshev": "ChebyshevDistance", "hamming": "HammingDistance",
                     "minkowski": "MinkowskiDistance(p: 3.0)"]
        guard let metricType = types[metric] else { return nil }
        let seeds = InputSeeds.distance(dim: dim, runSeed: runSeed)
        let type = variant == "generic" ? genericType(dim) : optimizedType(dim)
        return MeasurementBoundary(
            timingScope: .reusedInput, operation: "scalar.distance.\(metric)",
                route: "\(metricType).distance(_:_:) on \(type) inside withExtendedLifetime",
            provider: nil, workUnit: "invocation", unitsPerInvocation: 1,
            inputRepresentation: "fp32-vector-pair:\(type)", outputRepresentation: "scalar:Float",
            fixture: fixture(id: "V02-distance-pair", dimension: dim, count: 2, seeds: ["a": String(seeds.a), "b": String(seeds.b)]),
            inclusions: inclusions(internalAllocation: probe?.value ?? .unverified), consumer: "scalar-sink-v1",
            notes: [probe?.note
                ?? "Whether the library call allocates is not established from the benchmark source (routeInternalAllocation "
                + "unverified)."])
    }

    /// Notes checked by `testNormalizeSuccessFoldCoversEveryCoordinateAllWidthsAndVariants` (success fold) and
    /// `testConsumerTokensMatchDeclaredConsumers` (sink vocabulary); body drift by `testBodyFingerprintsMatchRecordedHashes`.
    private static func normalize(status: String, dim: Int?, variant: String,
                                  probe: AllocationResolution?) -> MeasurementBoundary? {
        guard let dim, ["generic", "optimized"].contains(variant), ["success", "zeroFail"].contains(status) else { return nil }
        let type = variant == "generic" ? genericType(dim) : optimizedType(dim)
        let success = status == "success"
        return MeasurementBoundary(
            timingScope: .reusedInput, operation: "vector.normalize",
            route: "\(type).normalized().get() via try? inside withExtendedLifetime",
            provider: nil, workUnit: "invocation", unitsPerInvocation: 1,
            inputRepresentation: success ? "fp32-vector:\(type) (all coordinates 2.0)" : "fp32-vector:\(type) (all coordinates 0.0)",
            outputRepresentation: success ? "fp32-vector:\(type) (unit vector, new value)" : "error-outcome:normalization failure",
            fixture: fixture(id: success ? "V02-normalize-constant-2" : "V02-normalize-zero", dimension: dim, count: 1,
                values: success ? "constant 2.0" : "constant 0.0"),
            inclusions: inclusions(internalAllocation: probe?.value ?? .unverified, validation: .notApplicable),
            consumer: success ? "u64-bitfold-sink-v1" : "error-outcome-sink-v1",
            notes: (success
                ? ["The consumer folds every coordinate of the produced unit vector."]
                : ["The produced result is the failure outcome; the consumer observes that outcome, so it is fully consumed."])
                + ["No out-of-band validation exists for this case (correctnessValidation is notApplicable).",
                   probe?.note ?? (success ? "Result storage allocation is likely but not established from the benchmark source "
                       + "(routeInternalAllocation unverified)."
                           : "Whether constructing the failure value allocates is not established from the benchmark source "
                               + "(routeInternalAllocation unverified).")])
    }

    // MARK: Batch suite

    private struct BatchShape {
        let dim: Int
        let n: Int
        let metric: String
        let variant: String
        let operation: String
        let seeds: [String: String]
        var generic: Bool { variant == "generic" }
        var type: String { generic ? Boundaries.genericType(dim) : Boundaries.optimizedType(dim) }
        var typeName: String { Boundaries.optimizedType(dim) }
    }

    private static func batch(_ parts: [String], runSeed: UInt64, probe: AllocationResolution?) -> MeasurementBoundary? {
        // batch.<metric>.<dim>.N<count>.<variant>[.<provider>]
        guard parts.count >= 5, parts.count <= 6, let dim = Int(parts[2]), parts[3].hasPrefix("N"),
              let n = Int(parts[3].dropFirst()) else { return nil }
        let operations = ["euclidean2": "batch.squared-euclidean", "euclidean": "batch.euclidean", "cosine": "batch.cosine-distance"]
        guard let operation = operations[parts[1]] else { return nil }
        let seeds = ["query": String(InputSeeds.batch(dim: dim, n: 0, runSeed: runSeed).query),
                     "candidateBase": String(InputSeeds.batch(dim: dim, n: n, runSeed: runSeed).baseCandidates)]
        let shape = BatchShape(dim: dim, n: n, metric: parts[1], variant: parts[4], operation: operation, seeds: seeds)
        if parts.count == 6 { return providerRow(shape, provider: parts[5]) }
        switch shape.variant {
        case "optimized-soa": return shape.metric == "euclidean2" ? soaRow(shape, probe: probe) : nil
        case "optimized-fp16": return shape.metric == "euclidean2" || shape.metric == "cosine" ? fp16Row(shape) : nil
        default: return shape.metric == "euclidean2" ? companionRow(shape, probe: probe) : nil
        }
    }

    // MARK: Boundary-suite companions

    /// Why the SoA-FP16 companions are unavailable (reported verbatim; the route is never substituted:
    /// `testExcludedAndUnavailableRoutesAreNeverInvoked`).
    static let soaFP16UnavailableReason =
        "No squared-Euclidean route exists for an FP32 query against SoA-FP16 candidates: every public SoA-FP16 Euclidean kernel returns "
        + "sqrt distances: MixedPrecisionKernels.batchEuclidean{512,768,1536}, batchEuclideanBlocked{512,768,1536}, the public "
        + "batchEuclideanSoA, and the deprecated batchEuclideanSquaredSoA, whose name is misleading (it returns the non-squared "
        + "distance; see the doc comments of batchEuclideanSoA and batchEuclideanSquaredSoA in MixedPrecisionKernels.swift). "
        + "No SoA-FP16 squared-Euclidean or cosine route exists."

    /// Rows of the opt-in `boundary` suite: reused-output counterparts, setup-inclusive companions and consumer-only records.
    /// Routes and notes are hand-written from `Suites/BoundaryBench.swift`: scopes and flags are checked by
    /// `testBoundaryRecordsAreHonestAboutScopeAndWork`, unavailable entries by `testExcludedAndUnavailableRoutesAreNeverInvoked`,
    /// fixture-only error reporting by `testExecutableCompanionsReportCorrectnessAndNoABRows`, and body drift by
    /// `testBodyFingerprintsMatchRecordedHashes` (which pins the body text; it does not test a note).
    ///
    /// Not verified: the runtime text of the routes, representations and notes below is written by hand from the route
    /// bodies; the tests named above check record fields and body text, not this text.
    private static func companionRow(_ shape: BatchShape, probe: AllocationResolution?) -> MeasurementBoundary? {
        let variant = shape.variant
        let scope: TimingScope
        if variant == "consumer-only" {
            scope = .consumerOnly
        } else if variant.hasSuffix("-reused") {
            scope = .reusedInput
        } else if variant.hasSuffix("-setup") {
            scope = .setupInclusive
        } else {
            return nil
        }
        let setup = scope == .setupInclusive
        let d = shape.dim
        let fixtureValues = fixture(id: "V02-batch-lcg", dimension: d, count: shape.n, seeds: shape.seeds,
                                    preparation: companionPreparation(variant))
        let soaEligible = BatchKernels_SoA.shouldUseSoA(candidateCount: shape.n, dimension: d)
        let halfEligible = MixedPrecisionKernels.shouldUseMixedPrecision(candidateCount: shape.n, dimension: d)
        let common = "kernel is called through a stored closure"
        let unverified = "(routeInternalAllocation unverified)"
        func row(route: String, input: String, flags: BoundaryInclusions, notes: [String], eligible: Bool? = nil) -> MeasurementBoundary {
            MeasurementBoundary(
                timingScope: scope, operation: shape.operation, route: route, provider: nil, workUnit: "candidate",
                unitsPerInvocation: shape.n, inputRepresentation: input,
                outputRepresentation: "scores:Float32[N]" + (setup ? " (created inside the invocation)" : " (caller-owned, reused)"),
                fixture: fixtureValues, inclusions: flags, consumer: "u64-bitfold-sink-v1", notes: notes, defaultEligible: eligible)
        }
        switch variant {
        case "aos-fp32-reused", "aos-fp32-setup":
            let kernel = "BatchKernels.range_euclid2_\(d)(query:candidates:range: 0..<N, out:)"
            return row(route: kernel + (setup ? " into a freshly allocated [Float](N), then a fold of all N outputs"
                                              : " into a caller-owned reusable buffer, then a fold of all N outputs"),
                       input: "aos-fp32:[\(shape.typeName)]",
                       flags: inclusions(callerOutput: setup ? .included : .excluded, internalAllocation: probe?.value ?? .unverified),
                       notes: ["Direct kernel: no provider scheduling or observation. The \(common).",
                               probe?.note ?? "Allocation inside the kernel is not established from the benchmark source \(unverified)."])
        case "soa-fp32-reused":
            return row(route: "BatchKernels_SoA.euclid2_\(d)(query:soa:out:) into a caller-owned reusable buffer, then a fold of all "
                + "N outputs",
                       input: "soa-fp32:SoA<\(shape.typeName)>",
                       flags: inclusions(preparation: .excluded, callerOutput: .excluded,
                                         internalAllocation: probe?.value ?? .unverified),
                       notes: ["The SoA is built outside timing. The \(common).",
                               probe?.note
                                   ?? "Allocation inside the SoA kernel is not established from the benchmark source \(unverified)."]
                           + (probe == nil ? [AllocationProbe.soaUnprobed] : []),
                       eligible: soaEligible)
        case "soa-fp32-setup":
            return row(route: "BatchKernels_SoA.batchEuclideanSquared\(d)(query:candidates:), which builds the SoA, allocates and "
                + "returns the output "
                              + "array, then a fold of all N outputs",
                       input: "aos-fp32:[\(shape.typeName)]",
                       flags: inclusions(packing: .included, internalAllocation: .included),
                       notes: ["API-call measurement: the public convenience route transposes the candidates and allocates its result "
                           + "on every call. "
                               + "Output allocation is the API's and is counted under routeInternalAllocation.", "The \(common)."],
                       eligible: soaEligible)
        case "aos-fp16-reused":
            return row(route: "MixedPrecisionKernels.range_euclid2_mixed_\(d)(query:candidatesFP16:range:out:) into a caller-owned "
                + "reusable buffer, "
                              + "then a fold of all N outputs",
                       input: "fp32-query + aos-fp16-candidates",
                       flags: inclusions(preparation: .excluded, callerOutput: .excluded, internalAllocation: .included),
                       notes: ["Candidates are converted to FP16 outside timing. Each call builds a temporary array of candidate handles "
                               + "(candidatesFP16[range].map { $0.storage }) inside the timing (routeInternalAllocation included).",
                               AllocationProbe.fp16Reused,
                               "Reduced precision: validation reports the measured error on this fixture only; equivalent requested "
                                   + "distances do not "
                               + "imply bitwise-equivalent arithmetic. The \(common)."],
                       eligible: halfEligible)
        case "aos-fp16-setup":
            return row(route: "MixedPrecisionKernels.convertToFP16_\(d)(candidates), then range_euclid2_mixed_\(d) into a freshly "
                + "allocated "
                              + "[Float](N), then a fold of all N outputs",
                       input: "aos-fp32:[\(shape.typeName)] (query stays FP32)",
                       flags: inclusions(preparation: .included, callerOutput: .included, internalAllocation: .included),
                       notes: ["Conversion of the candidates is inside the timing; the query is used as FP32. The kernel builds a "
                           + "temporary array of "
                               + "candidate handles on each call (routeInternalAllocation included).",
                               AllocationProbe.fp16Setup,
                               "Reduced precision: validation reports the measured error on this fixture only. The \(common)."],
                       eligible: halfEligible)
        case "soa-fp16-reused", "soa-fp16-setup":
            return MeasurementBoundary(
                timingScope: scope, operation: shape.operation, route: "unavailable: " + soaFP16UnavailableReason, provider: nil,
                workUnit: "candidate", unitsPerInvocation: shape.n, inputRepresentation: "none", outputRepresentation: "none",
                fixture: fixtureValues,
                inclusions: inclusions(construction: .notApplicable, preparation: .notApplicable, kernel: .notApplicable,
                                       consumption: .notApplicable, validation: .notApplicable),
                consumer: "none", notes: ["unavailable; never invoked, timed, validated or substituted."])
        case "consumer-only":
            return MeasurementBoundary(
                timingScope: .consumerOnly, operation: shape.operation, route: "OutputFold.fold over a precomputed score array (no kernel)",
                provider: nil, workUnit: "candidate", unitsPerInvocation: shape.n,
                inputRepresentation: "scores:Float32[N] (precomputed once outside timing by the aos-fp32 route)",
                outputRepresentation: "UInt64 fold", fixture: fixtureValues,
                inclusions: inclusions(kernel: .excluded), consumer: "u64-bitfold-sink-v1",
                notes: ["Consumer overhead for the same N and output representation as the batch companions. It is reported "
                    + "separately and is never "
                        + "subtracted or treated as zero."])
        default:
            return nil
        }
    }

    /// Fixture preparation label per companion variant (scopes checked by `testBoundaryRecordsAreHonestAboutScopeAndWork`).
    ///
    /// Not verified: the runtime text of these labels is written by hand from the route bodies; the test checks scope
    /// fields, not this text.
    private static func companionPreparation(_ variant: String) -> String {
        switch variant {
        case "soa-fp32-reused": return "soa-built-outside-timing"
        case "aos-fp16-reused": return "candidates-converted-to-fp16-outside-timing"
        case "aos-fp32-setup", "aos-fp32-reused", "consumer-only": return "none"
        default: return "none (transposition or conversion is part of the timed invocation)"
        }
    }

    private static func providerKernel(_ shape: BatchShape) -> String? {
        let scratch = " into a per-range scratch buffer"
        switch (shape.metric, shape.variant) {
        case ("euclidean2", "generic"): return "\(shape.type).euclideanDistanceSquared(to:) per candidate"
        case ("euclidean", "generic"): return "EuclideanDistance().distance(_:_:) per candidate"
        case ("euclidean2", "optimized"): return "BatchKernels.range_euclid2_\(shape.dim)" + scratch
        case ("euclidean", "optimized"): return "BatchKernels.range_euclid_\(shape.dim)" + scratch
        case ("cosine", "optimized-fused"): return "BatchKernels.range_cosine_fused_\(shape.dim)" + scratch
        case ("cosine", "optimized-preNorm"): return "BatchKernels.range_cosine_preNorm_\(shape.dim)" + scratch
        default: return nil
        }
    }

    /// Provider-scheduled AoS rows: scheduler, observation and intermediate allocations are inside the timing (not verified
    /// by a test: `testAllocationFlagsDoNotOverclaim` checks allocation flags and `testBodyFingerprintsMatchRecordedHashes`
    /// pins the body text).
    ///
    /// Not verified: the runtime text of the notes and the preparation label below is written by hand from the route
    /// bodies; no test checks this text.
    private static func providerRow(_ shape: BatchShape, provider: String) -> MeasurementBoundary? {
        guard let kernel = providerKernel(shape) else { return nil }
        let preNorm = shape.variant == "optimized-preNorm"
        var notes = ["Provider scheduler and child-task allocations are provider-defined and are inside the timing; "
                     + "they are not itemized here."]
        if shape.generic {
            notes.append("Allocation by the escaping closure context passed to the provider and inside the library call is not "
                         + "established (routeInternalAllocation unverified).")
            if let measured = AllocationProbe.genericPair(dim: shape.dim, n: shape.n, provider: provider) { notes.append(measured) }
        } else {
            notes.append("A [Float](range.count) scratch buffer is allocated inside each timed range (routeInternalAllocation included).")
        }
        let comparison: ComparisonMembership
        if shape.metric == "cosine" {
            comparison = ComparisonMembership(family: ComparisonFamilies.cosinePreNormVersusFused, side: preNorm ? "left" : "right",
                                              group: "\(shape.dim)|\(shape.n)|optimized|\(provider)")
        } else {
            comparison = ComparisonMembership(family: ComparisonFamilies.euclideanSquaredVersusEuclidean,
                                              side: shape.metric == "euclidean2" ? "left" : "right",
                                              group: "\(shape.dim)|\(shape.n)|\(shape.variant)|\(provider)")
        }
        let preparation = preNorm ? "l2-normalized-outside-timing (zero vectors keep the unnormalized input)" : "none"
        return MeasurementBoundary(
            timingScope: .reusedInput, operation: shape.operation,
            route: "ComputeProvider.parallelReduce(\(provider)) + \(kernel)",
            provider: provider, workUnit: "candidate", unitsPerInvocation: shape.n,
            inputRepresentation: "aos-fp32:[\(shape.type)]", outputRepresentation: "provider-range-partials:Float",
            fixture: fixture(id: "V02-batch-lcg", dimension: shape.dim, count: shape.n, seeds: shape.seeds, preparation: preparation),
            inclusions: inclusions(preparation: preNorm ? .excluded : .notApplicable,
                                   internalAllocation: shape.generic ? .unverified : .included,
                                   scheduling: .included, observation: .included),
            consumer: "provider-float-sum-observe-sink-v1", notes: notes, comparison: comparison)
    }

    private static func soaRow(_ shape: BatchShape, probe: AllocationResolution?) -> MeasurementBoundary {
        MeasurementBoundary(
            timingScope: .reusedInput, operation: shape.operation,
            route: "BatchKernels_SoA.euclid2_\(shape.dim)(query:soa:out:) into a caller-owned reusable buffer, "
                + "then a Float sum over all N outputs",
            provider: nil, workUnit: "candidate", unitsPerInvocation: shape.n,
            inputRepresentation: "soa-fp32:SoA<\(shape.type)>", outputRepresentation: "scores:Float32[N] (caller-owned, reused)",
            fixture: fixture(id: "V02-batch-lcg", dimension: shape.dim, count: shape.n, seeds: shape.seeds,
                             preparation: "soa-built-outside-timing"),
            inclusions: inclusions(preparation: .excluded, callerOutput: .excluded, internalAllocation: probe?.value ?? .unverified),
            consumer: "float-sum-all-outputs-sink-v1",
            notes: probe.map { [$0.note] }
                ?? ["Allocation inside the SoA kernel is not established for this (D, N) (routeInternalAllocation unverified).",
                    AllocationProbe.soaUnprobed])
    }

    /// Allocation flags checked by `testAllocationFlagsDoNotOverclaim`; body drift by `testBodyFingerprintsMatchRecordedHashes`.
    ///
    /// Not verified: the runtime text of the notes below is written by hand from the route bodies; no test checks this
    /// text.
    private static func fp16Row(_ shape: BatchShape) -> MeasurementBoundary {
        let routeMetric = shape.metric == "cosine" ? "cosine" : "euclid2"
        return MeasurementBoundary(
            timingScope: .reusedInput, operation: shape.operation,
            route: "MixedPrecisionKernels.range_\(routeMetric)_mixed_\(shape.dim)(query:candidatesFP16:range:out:) "
                + "into a caller-owned reusable buffer, then a Float sum over all N outputs",
            provider: nil, workUnit: "candidate", unitsPerInvocation: shape.n,
            inputRepresentation: "fp32-query + aos-fp16-candidates", outputRepresentation: "scores:Float32[N] (caller-owned, reused)",
            fixture: fixture(id: "V02-batch-lcg", dimension: shape.dim, count: shape.n, seeds: shape.seeds,
                             preparation: "candidates-converted-to-fp16-outside-timing"),
            inclusions: inclusions(preparation: .excluded, callerOutput: .excluded, internalAllocation: .included),
            consumer: "float-sum-all-outputs-sink-v1",
            notes: ["Each range_*_mixed_\(shape.dim) call builds a temporary array of candidate handles "
                    + "(candidatesFP16[range].map { $0.storage }) inside the timing (routeInternalAllocation included).",
                    AllocationProbe.fp16DefaultRow(metric: shape.metric),
                    "Reduced-precision candidates; validation reports error against a Double reference on this fixture only."])
    }

    // MARK: Matrix suite

    /// The prepacked-GEMM note is checked by `testReportRequiresBoundaryIdentity` and the allocation flags by
    /// `testAllocationFlagsDoNotOverclaim`; body drift by `testBodyFingerprintsMatchRecordedHashes`.
    ///
    /// Not verified: the runtime text of the second note below is written by hand from the MatrixDistance source; no test
    /// checks that text.
    private static func matrix(_ parts: [String], runSeed: UInt64) -> MeasurementBoundary? {
        // matrix.euclidean2.<dim>.N<count>.optimized
        guard parts.count == 5, parts[1] == "euclidean2", let dim = Int(parts[2]), parts[3].hasPrefix("N"),
              let n = Int(parts[3].dropFirst()), parts[4] == "optimized" || parts[4] == "consumer-only",
              let pairs = try? BenchmarkArithmetic.product(n, n) else { return nil }
        if parts[4] == "consumer-only" { return matrixConsumerOnly(dim: dim, n: n, pairs: pairs, runSeed: runSeed) }
        return MeasurementBoundary(
            timingScope: .reusedInput, operation: "matrix.squared-euclidean.public-api-call",
            route: "MatrixDistance.euclideanSquaredMatrix(queries:candidates:into:) with queries == candidates; the API packs rows "
                + "and norms and runs sgemm and the clamp pass on every call",
            provider: nil, workUnit: "produced-entry", unitsPerInvocation: pairs,
            inputRepresentation: "aos-fp32:[\(optimizedType(dim))] (queries == candidates)",
            outputRepresentation: "[Float] N*N row-major (caller-owned, reused)",
            fixture: fixture(id: "V02-matrix-rows", dimension: dim, count: n,
                             seeds: ["runSeed": String(runSeed), "row0": String(InputSeeds.matrix(dim: dim, row: 0, runSeed: runSeed))],
                             values: lcgValues + "; row seed = InputSeeds.matrix(dim,row,runSeed)"),
            inclusions: inclusions(packing: .included, callerOutput: .excluded, internalAllocation: .included),
            consumer: "u64-bitfold-sink-v1",
            notes: ["API-call measurement: not a prepacked GEMM kernel timing.",
                    "Routine-internal Float arrays are allocated by the API on each call."])
    }

    /// Scope checked by `testBoundaryRecordsAreHonestAboutScopeAndWork`; body drift by `testBodyFingerprintsMatchRecordedHashes`.
    ///
    /// Not verified: the runtime text of the representation and the note below is written by hand; no test checks this
    /// text.
    private static func matrixConsumerOnly(dim: Int, n: Int, pairs: Int, runSeed: UInt64) -> MeasurementBoundary {
        MeasurementBoundary(
            timingScope: .consumerOnly, operation: "matrix.squared-euclidean.consumer-only",
            route: "OutputFold.fold over a precomputed N*N output (no matrix API call)", provider: nil,
            workUnit: "produced-entry", unitsPerInvocation: pairs,
            inputRepresentation: "[Float] N*N row-major (precomputed once outside timing by MatrixDistance.euclideanSquaredMatrix)",
            outputRepresentation: "UInt64 fold",
            fixture: fixture(id: "V02-matrix-rows", dimension: dim, count: n,
                             seeds: ["runSeed": String(runSeed), "row0": String(InputSeeds.matrix(dim: dim, row: 0, runSeed: runSeed))],
                             values: lcgValues + "; row seed = InputSeeds.matrix(dim,row,runSeed)"),
            inclusions: inclusions(kernel: .excluded), consumer: "u64-bitfold-sink-v1",
            notes: ["Consumer overhead for the same N*N output representation as the matrix suite. It is reported separately and is never "
                    + "subtracted or treated as zero."])
    }

    // MARK: Memory and control suites

    /// Allocation flags checked by `testAllocationFlagsDoNotOverclaim`, the retention label by `testQ6ConsumerLabelsAreAccurate`;
    /// body drift by `testBodyFingerprintsMatchRecordedHashes`.
    ///
    /// Not verified: the runtime text of the routes, representations and notes below is written by hand from the suite
    /// bodies; the tests named above check record fields and labels, not this text.
    private static func memory(_ parts: [String], probe: AllocationResolution?) -> MeasurementBoundary? {
        guard let last = parts.last, let dim = Int(last), let count = try? BenchmarkArithmetic.product(dim, 1024) else { return nil }
        let joined = parts.dropLast().joined(separator: ".")
        let sizes = fixture(id: "V02-memory-sizes", dimension: dim, count: count, values: "constant (0.5 source, 0 or 0.123 destination)")
        let noValidation = "No out-of-band validation exists for this case (correctnessValidation is notApplicable)."
        let retention = "escaped-pointer-retention-v1"
        let limitation = "blackHole observes at most one element; retention of the work rests on the escaped allocator pointer, not "
            + "on traversal of the output."
        switch joined {
        case "mem.alloc.aligned":
            return MeasurementBoundary(
                timingScope: .reusedInput, operation: "memory.aligned-allocate-init8-release",
                route: "AlignedMemory.allocateAligned + initialize(min(8,count)) + deinitialize + deallocate", provider: nil,
                workUnit: "invocation", unitsPerInvocation: 1, inputRepresentation: "none",
                    outputRepresentation: "aligned float allocation (released in the invocation)",
                fixture: sizes, inclusions: inclusions(callerOutput: .included, internalAllocation: .notApplicable,
                                                       consumption: .notApplicable, validation: .notApplicable),
                consumer: retention, notes: [limitation, noValidation,
                    "The allocation is the operation and is counted once, under callerOutputAllocation."])
        case "mem.copy":
            return MeasurementBoundary(
                timingScope: .reusedInput, operation: "memory.aligned-copy", route: "alignedCopy(from:to:count:preferredAlignment:)",
                    provider: nil,
                workUnit: "element", unitsPerInvocation: count, inputRepresentation: "aligned float buffer (count elements)",
                outputRepresentation: "aligned float buffer (caller-owned, reused)", fixture: sizes,
                inclusions: inclusions(callerOutput: .excluded, internalAllocation: probe?.value ?? .unverified,
                    consumption: .notApplicable, validation: .notApplicable),
                consumer: retention, notes: [limitation, noValidation]
                    + (probe.map { ["Source and destination buffers are allocated outside timing.", $0.note] }
                        ?? ["Source and destination buffers are allocated outside timing; allocation inside alignedCopy is not established "
                            + "(routeInternalAllocation unverified)."]))
        case "mem.fill":
            return MeasurementBoundary(
                timingScope: .reusedInput, operation: "memory.scalar-fill-loop", route: "for-loop store of 0.123 into every element",
                    provider: nil,
                workUnit: "element", unitsPerInvocation: count, inputRepresentation: "none",
                outputRepresentation: "aligned float buffer (caller-owned, reused)", fixture: sizes,
                inclusions: inclusions(callerOutput: .excluded, internalAllocation: .notApplicable, consumption: .notApplicable,
                    validation: .notApplicable),
                consumer: retention, notes: [limitation, noValidation,
                    "The destination buffer is allocated outside timing; the loop allocates nothing."])
        case "mem.pool.acquire":
            return MeasurementBoundary(
                timingScope: .reusedInput, operation: "memory.pool-acquire-initialize1-release",
                route: "MemoryPool.shared.acquire + initialize(to:) + deinitialize(count: 1) + handle release at scope exit", provider: nil,
                workUnit: "invocation", unitsPerInvocation: 1, inputRepresentation: "none",
                    outputRepresentation: "pooled float buffer handle (released in the invocation)",
                fixture: sizes, inclusions: inclusions(callerOutput: .included, internalAllocation: probe?.value ?? .unverified,
                    consumption: .notApplicable, validation: .notApplicable),
                consumer: retention, notes: [limitation, noValidation, "The shared pool is warmed and quiesced before timing.",
                    probe?.note
                        ?? "Pool-internal bookkeeping allocation is not established from the benchmark source (routeInternalAllocation "
                        + "unverified)."])
        default:
            return nil
        }
    }

    private static func noop() -> MeasurementBoundary {
        MeasurementBoundary(
            timingScope: .reusedInput, operation: "control.scalar-add-256", route: "256-iteration Float accumulation loop over constants",
                provider: nil,
            workUnit: "invocation", unitsPerInvocation: 1, inputRepresentation: "none", outputRepresentation: "scalar:Float",
            fixture: fixture(id: "none", dimension: 0, count: 0, values: "constants"),
            inclusions: inclusions(construction: .notApplicable, kernel: .notApplicable, validation: .notApplicable),
            consumer: "scalar-sink-v1",
            notes: ["No out-of-band validation exists for this case (correctnessValidation is notApplicable).",
                    "Driver and sink overhead control; the loop is over compile-time constants and may be folded by the compiler."])
    }
}
