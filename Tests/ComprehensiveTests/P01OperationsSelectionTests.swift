import Foundation
import Testing
@testable import VectorCore

// P01 phase E: Operations validation, unbounded k, direct-route references and the Cosine 768/1536 fused route.
// Route observations under provider, partition, nesting and cancellation variation are in P01OperationsRouteTests.

@Suite("P01 Operations selection (phase E)")
struct P01OperationsSelectionTests {
    static let childFilter = "P01OperationsSelectionTests/p01OpsChild"

    // MARK: Validation (O1, O2, T10)

    /// A malformed row late in the candidates (or the queries) is rejected before any provider or metric work,
    /// with the error the unchanged base threw: kind, context information and the public entry point.
    @Test func lateMalformedRowsAreRejectedBeforeAnyWork() async throws {
        let calls = P01OpsCalls()
        let metric = P01OpsSpyMetric(calls: calls, token: 11)
        let good = p01TaggedRows((0..<40).map { Float($0) }, dimension: 3)
        var late = good
        late[39] = DynamicVector([1, 39])
        var middle = good
        middle[20] = DynamicVector([1, 20, 0, 0])
        let query = p01TaggedQuery(dimension: 3)
        let single = "findNearest(to:in:k:metric:)", batch = "findNearestBatch(queries:in:k:metric:cpuStrategy:)"

        for (rows, actual) in [(late, 2), (middle, 4)] {
            let expected = VectorError.dimensionMismatch(expected: 3, actual: actual)
            for providerIndex in 0..<4 {
                let kernelLog = P01KernelLog()
                let recording = P01RecordingProvider(inner: CPUComputeProvider(mode: .parallel))
                let providers: [any ComputeProvider] = [
                    CPUComputeProvider.automatic,
                    recording,
                    P01KernelSpy(log: kernelLog, reply: [(index: 0, distance: 0)]),
                    P01KernelBatchSpy(log: kernelLog, reply: [(index: 0, distance: 0)], batchReply: [])
                ]
                let provider = providers[providerIndex]
                calls.reset()
                let singleError = await p01ErrorIdentity {
                    _ = try await Operations.$computeProvider.withValue(provider) {
                        try await Operations.findNearest(to: query, in: rows, k: 1, metric: metric)
                    }
                }
                #expect(singleError == P01ErrorIdentity(expected: expected, function: single), "provider \(providerIndex)")
                let batchError = await p01ErrorIdentity {
                    _ = try await Operations.$computeProvider.withValue(provider) {
                        try await Operations.findNearestBatch(queries: [query, query, query], in: rows, k: 1, metric: metric)
                    }
                }
                #expect(batchError == P01ErrorIdentity(expected: expected, function: batch), "provider \(providerIndex)")
                #expect(calls.count == 0, "metric ran before validation finished (provider \(providerIndex))")
                #expect(recording.ledger.isUntouched)
                #expect(kernelLog.isUntouched, "BatchKernelProvider ran before validation finished")
            }
        }

        // A late malformed query row and a query that disagrees with the candidates, for the batch API.
        let kernelLog = P01KernelLog()
        let spy = P01KernelBatchSpy(log: kernelLog, reply: [], batchReply: [])
        let badQueries = [query, query, query, DynamicVector([0, -1])]
        let lateQuery = await p01ErrorIdentity {
            _ = try await Operations.$computeProvider.withValue(spy) {
                try await Operations.findNearestBatch(queries: badQueries, in: good, k: 1, metric: metric)
            }
        }
        #expect(lateQuery == P01ErrorIdentity(expected: .dimensionMismatch(expected: 3, actual: 2), function: batch))
        let crossed = await p01ErrorIdentity {
            _ = try await Operations.$computeProvider.withValue(spy) {
                try await Operations.findNearestBatch(queries: [DynamicVector([0, -1])], in: good, k: 1, metric: metric)
            }
        }
        #expect(crossed == P01ErrorIdentity(expected: .dimensionMismatch(expected: 3, actual: 2), function: batch))
        #expect(kernelLog.isUntouched)
        #expect(calls.count == 0)
    }

    /// Preflight order (O1): k first, then emptiness, then the query against row 0, then the rows.
    @Test func preflightOrderIsUnchanged() async throws {
        let calls = P01OpsCalls()
        let metric = P01OpsSpyMetric(calls: calls)
        var rows = p01TaggedRows((0..<40).map { Float($0) }, dimension: 3)
        rows[39] = DynamicVector([1, 39])
        let single = "findNearest(to:in:k:metric:)"
        for k in [0, -1, Int.min] {
            let kError = await p01ErrorIdentity {
                _ = try await Operations.findNearest(to: DynamicVector([0, -1]), in: rows, k: k, metric: metric)
            }
            #expect(kError == P01ErrorIdentity(expected: .invalidDimension(k, reason: "k must be positive"), function: single))
        }
        let emptyError = await p01ErrorIdentity {
            _ = try await Operations.findNearest(to: DynamicVector([0, -1]), in: [DynamicVector](), k: 1, metric: metric)
        }
        #expect(emptyError == P01ErrorIdentity(expected: .invalidDimension(0, reason: "Vector collection cannot be empty"),
                                               function: single))
        let queryError = await p01ErrorIdentity {
            _ = try await Operations.findNearest(to: DynamicVector([0, -1, 0, 0, 0]), in: rows, k: 1, metric: metric)
        }
        #expect(queryError == P01ErrorIdentity(expected: .dimensionMismatch(expected: 3, actual: 5), function: single))
        #expect(calls.count == 0)
    }

    // MARK: Unbounded k (O10)

    /// A user k is never a heap capacity: k = Int.max and k > n complete on every route family, and the partial
    /// capacities a provider sees never exceed the candidate count.
    @Test func unboundedKSizesNothing() async throws {
        let calls = P01OpsCalls()
        let generic = p01TaggedRows((0..<30).map { Float(29 - $0) })
        var random = P01OpsRandom(state: 7)
        let rows512 = (0..<30).map { _ in Vector512Optimized(generator: { _ in random.unit() }) }
        let rows768 = (0..<30).map { _ in Vector768Optimized(generator: { _ in random.unit() }) }
        for k in [30, 31, Int.max] {
            let recording = P01RecordingProvider(inner: CPUComputeProvider(mode: .parallel))
            try await Operations.$computeProvider.withValue(recording) {
                calls.reset()
                let a = try await Operations.findNearest(to: p01TaggedQuery(), in: generic, k: k, metric: P01OpsSpyMetric(calls: calls))
                #expect(a.map(\.index) == Array((0..<30).reversed()))
                let b = try await Operations.findNearest(to: rows512[0], in: rows512, k: k)
                #expect(b.count == 30 && b.first?.index == 0)
                let c = try await Operations.findNearest(to: rows768[0], in: rows768, k: k, metric: CosineDistance())
                #expect(c.count == 30)
                let d = try await Operations.findNearestBatch(queries: [rows512[1], rows512[2]], in: rows512, k: k)
                #expect(d.map(\.count) == [30, 30])
            }
            #expect(recording.ledger.partials.allSatisfy { $0.capacity <= 30 })
        }
        // The largest direct-route capacity: k = n - 1.
        let recording = P01RecordingProvider(inner: P01PlanProvider(label: "chunks", plan: P01PlanProvider.plans[3].1, fold: .left))
        let e = try await Operations.$computeProvider.withValue(recording) {
            try await Operations.findNearest(to: rows512[0], in: rows512, k: 29)
        }
        #expect(e.count == 29)
        #expect(recording.ledger.partials.map(\.capacity) == recording.ledger.partials.map { min(29, $0.range.count) })
        #expect(!recording.ledger.partials.isEmpty)
    }

    // MARK: Direct optimized routes: independent finite references and exceptional parity (O8, T7)

    /// The five direct routes keep their arithmetic and ranking score: finite fixtures agree with an independent
    /// Double reference (index order exactly, distances within a tolerance that no NaN can pass), and complete
    /// results (finite and exceptional) equal the kernel's own per-candidate scores selected by the independent
    /// oracle, bit for bit, under the sequential and parallel CPU providers.
    @Test func directRoutesMatchIndependentReferences() async throws {
        let count = 300, k = 5
        for exceptional in [false, true] {
            var random = P01OpsRandom(state: exceptional ? 99 : 42)
            func values(_ dimension: Int, _ row: Int) -> [Float] {
                var v = (0..<dimension).map { _ in random.unit() }
                if exceptional {
                    switch row % 11 {
                    case 3: v[1] = .nan
                    case 5: v[2] = .infinity
                    case 7: v = [Float](repeating: 0, count: dimension)
                    case 9: v = [Float](repeating: -0.0, count: dimension)
                    default: break
                    }
                }
                return v
            }
            let raw512 = (0..<count).map { values(512, $0) }
            let raw768 = (0..<count).map { values(768, $0) }
            let raw1536 = (0..<count).map { values(1536, $0) }
            let query512 = try Vector512Optimized((0..<512).map { _ in random.unit() })
            let query768 = try Vector768Optimized((0..<768).map { _ in random.unit() })
            let query1536 = try Vector1536Optimized((0..<1536).map { _ in random.unit() })
            let rows512 = try raw512.map { try Vector512Optimized($0) }
            let rows768 = try raw768.map { try Vector768Optimized($0) }
            let rows1536 = try raw1536.map { try Vector1536Optimized($0) }

            let squared512 = rows512.map { EuclideanKernels.squared512(query512, $0) }
            let squared768 = rows768.map { EuclideanKernels.squared768(query768, $0) }
            let squared1536 = rows1536.map { EuclideanKernels.squared1536(query1536, $0) }
            let cosine512 = rows512.map { CosineKernels.distance512_fused(query512, $0) }
            let negatedDot512 = rows512.map { -DotKernels.dot512(query512, $0) }

            for mode in [CPUComputeProvider.Mode.sequential, .parallel] {
                let provider = CPUComputeProvider(mode: mode)
                try await Operations.$computeProvider.withValue(provider) {
                    let e512 = try await Operations.findNearest(to: query512, in: rows512, k: k)
                    let e768 = try await Operations.findNearest(to: query768, in: rows768, k: k)
                    let e1536 = try await Operations.findNearest(to: query1536, in: rows1536, k: k)
                    let c512 = try await Operations.findNearest(to: query512, in: rows512, k: k, metric: CosineDistance())
                    let d512 = try await Operations.findNearest(to: query512, in: rows512, k: k, metric: DotProductDistance())
                    // Kernel parity: membership by the ranking score, then the returned score's own order.
                    #expect(p01Entries(e512) == P01OpsOracle.select(squared512, k: k, format: { $0.squareRoot() }), "\(mode)")
                    #expect(p01Entries(e768) == P01OpsOracle.select(squared768, k: k, format: { $0.squareRoot() }), "\(mode)")
                    #expect(p01Entries(e1536) == P01OpsOracle.select(squared1536, k: k, format: { $0.squareRoot() }), "\(mode)")
                    #expect(p01Entries(c512) == P01OpsOracle.select(cosine512, k: k), "\(mode)")
                    #expect(p01Entries(d512) == P01OpsOracle.select(negatedDot512, k: k), "\(mode)")

                    guard !exceptional else { return }
                    // Independent Double references.
                    func euclid(_ q: [Float], _ c: [Float]) -> Double {
                        zip(q, c).reduce(0) { $0 + pow(Double($1.0) - Double($1.1), 2) }.squareRoot()
                    }
                    func dot(_ q: [Float], _ c: [Float]) -> Double { zip(q, c).reduce(0) { $0 + Double($1.0) * Double($1.1) } }
                    func cosine(_ q: [Float], _ c: [Float]) -> Double { 1 - dot(q, c) / (dot(q, q).squareRoot() * dot(c, c).squareRoot()) }
                    let q512 = query512.toArray(), q768 = query768.toArray(), q1536 = query1536.toArray()
                    // Absolute tolerances for Float accumulation over 512-1536 terms of magnitude at most 1.
                    let references: [(String, [NearestNeighborResult], [Double], Double)] = [
                        ("euclid512", e512, raw512.map { euclid(q512, $0) }, 1e-3),
                        ("euclid768", e768, raw768.map { euclid(q768, $0) }, 1e-3),
                        ("euclid1536", e1536, raw1536.map { euclid(q1536, $0) }, 2e-3),
                        ("cosine512", c512, raw512.map { cosine(q512, $0) }, 1e-5),
                        ("dot512", d512, raw512.map { -dot(q512, $0) }, 2e-3)
                    ]
                    for (label, result, reference, tolerance) in references {
                        let order = reference.indices.sorted {
                            reference[$0] < reference[$1] || (reference[$0] == reference[$1] && $0 < $1)
                        }
                        // Fixture precondition: neighbours in the reference order are more than two tolerances
                        // apart, so values within tolerance of the reference cannot reorder them.
                        let gaps = (0..<k).map { reference[order[$0 + 1]] - reference[order[$0]] }
                        #expect(gaps.allSatisfy { $0 > 2 * tolerance }, "\(label): fixture has near-ties \(gaps)")
                        #expect(result.map(\.index) == Array(order.prefix(k)), "\(label) \(mode)")
                        for entry in result {
                            let distance = Double(entry.distance)
                            #expect(distance.isFinite && abs(distance - reference[entry.index]) <= tolerance,
                                    "\(label) \(mode): \(entry.distance) vs \(reference[entry.index])")
                        }
                    }
                }
            }
        }
    }

    // MARK: Cosine 768/1536 fused full-score route (O4)

    /// Cosine on Vector768Optimized/Vector1536Optimized keeps the fused `computeDistances` route for small k: a
    /// fresh child with a cold key sees calibration published, no generic stream, and complete scores (finite
    /// and exceptional) equal to `BatchKernels.range_cosine_fused_*` over the ranges the route actually used.
    /// Dot product on Vector768Optimized is the contrast: it streams and publishes nothing.
    @Test func cosine768And1536StayOnTheFusedFullScoreRoute() throws {
        let child = try p01RunChild("operations-cosine-fused", filter: Self.childFilter, removing: ["VC_AUTOTUNE"])
        #expect(child.run.terminationStatus == 0, "status=\(child.run.statusDescription): \(child.tail)")
        #expect(child.output.contains("P01_CHILD_COMPLETE operations-cosine-fused"), "\(child.tail)")
    }

    // MARK: Route matrix (review M4)

    /// `Operations.scoresWithBatchKernels` restates `computeDistances`' BatchKernels type tests; this keeps the two
    /// from drifting apart. For every built-in metric (plus a custom one) on the three optimized types and two
    /// generic types it observes, through the recording provider:
    /// - the BatchKernels branch of `computeDistances`, at large k over 4000 candidates (at or above every finite
    ///   calibrated break-even, so the branch is the parallel one): a hinted `parallelReduce`, against
    ///   `parallelExecute` for the per-candidate branch;
    /// - the small-k route over 64 candidates: a no-hint `parallelReduce` for a direct route or the stream, and no
    ///   provider call for the serial full-score route.
    /// The law: small k streams or takes a direct route exactly when the case has a direct route or no BatchKernels
    /// branch. A BatchKernels branch added to (or removed from) only one of the two functions breaks it for that
    /// case. For the five direct-route cases step 2 intercepts small k first, so the predicate's value there has no
    /// observable effect and the law holds whatever it is. `batchKernelCases` pins today's set.
    @Test func routeMatrixMatchesComputeDistancesBatchKernelBranches() async throws {
        var random = P01OpsRandom(state: 31)
        let count = 4000
        let rows512 = (0..<count).map { _ in Vector512Optimized(generator: { _ in random.unit() }) }
        let rows768 = (0..<count).map { _ in Vector768Optimized(generator: { _ in random.unit() }) }
        let rows1536 = (0..<count).map { _ in Vector1536Optimized(generator: { _ in random.unit() }) }
        let rows384 = (0..<count).map { _ in Vector384Optimized(generator: { _ in random.unit() }) }
        let dynamic = (0..<count).map { _ in DynamicVector((0..<32).map { _ in random.unit() }) }
        let metrics: [(String, any DistanceMetric<Float>)] = [
            ("euclidean", EuclideanDistance()), ("cosine", CosineDistance()), ("dot", DotProductDistance()),
            ("manhattan", ManhattanDistance()), ("chebyshev", ChebyshevDistance()), ("hamming", HammingDistance()),
            ("minkowski3", MinkowskiDistance(p: 3)), ("custom", P01RouteMatrixMetric())
        ]
        let batchKernelCases: Set<String> = ["euclidean/512", "euclidean/768", "euclidean/1536",
                                             "cosine/512", "cosine/768", "cosine/1536"]
        let directCases: Set<String> = ["euclidean/512", "euclidean/768", "euclidean/1536", "cosine/512", "dot/512"]
        var table: [String] = []
        for (metricName, metric) in metrics {
            let observations = [
                ("512", try await Self.observeRoutes(rows512, metric)),
                ("768", try await Self.observeRoutes(rows768, metric)),
                ("1536", try await Self.observeRoutes(rows1536, metric)),
                ("384", try await Self.observeRoutes(rows384, metric)),
                ("dynamic32", try await Self.observeRoutes(dynamic, metric))
            ]
            for (typeName, observed) in observations {
                let key = "\(metricName)/\(typeName)"
                let batchKernels = observed.hinted
                // Fixture precondition: large k took a parallel branch (the hinted or the per-candidate one).
                // Manhattan has no AutoTuning key, so ParallelHeuristic may keep its per-candidate branch serial.
                if metricName != "manhattan" {
                    #expect(observed.hinted != observed.executed, "\(key): large-k branch not observed \(observed)")
                }
                #expect(batchKernels == batchKernelCases.contains(key), "\(key): BatchKernels branch \(batchKernels)")
                let expectedSmallReduce = directCases.contains(key) || !batchKernels
                #expect(observed.smallReduced == expectedSmallReduce,
                        "\(key): small k reduced=\(observed.smallReduced), expected \(expectedSmallReduce)")
                table.append("\(key):bk=\(batchKernels),small=\(observed.smallReduced ? "reduce" : "serial")")
            }
        }
        print("P01_ROUTE_MATRIX \(table.joined(separator: " "))")
    }

    private static func observeRoutes<V: VectorProtocol>(
        _ rows: [V], _ metric: any DistanceMetric<Float>
    ) async throws -> (hinted: Bool, executed: Bool, smallReduced: Bool) where V.Scalar == Float {
        try await observeRoutes(rows, opened: metric)
    }

    private static func observeRoutes<V: VectorProtocol, M: DistanceMetric>(
        _ rows: [V], opened metric: M
    ) async throws -> (hinted: Bool, executed: Bool, smallReduced: Bool) where V.Scalar == Float, M.Scalar == Float {
        let large = P01RecordingProvider(inner: CPUComputeProvider(mode: .sequential))
        let full = try await Operations.$computeProvider.withValue(large) {
            try await Operations.findNearest(to: rows[0], in: rows, k: rows.count, metric: metric)
        }
        #expect(full.count == rows.count)
        let smallRows = Array(rows.prefix(64))
        let small = P01RecordingProvider(inner: CPUComputeProvider(mode: .sequential))
        let best = try await Operations.$computeProvider.withValue(small) {
            try await Operations.findNearest(to: smallRows[0], in: smallRows, k: 1, metric: metric)
        }
        #expect(best.count == 1)
        return (hinted: !large.ledger.reduceHints.isEmpty, executed: large.ledger.executeEntries == 1,
                smallReduced: small.ledger.reduceWithoutHint == 1 && small.ledger.reduceHints.isEmpty)
    }

    // MARK: Child entry

    /// Returns at once unless the parent selected a scenario.
    @Test func p01OpsChild() async throws {
        guard let scenario = ProcessInfo.processInfo.environment["P01_SCENARIO"] else { return }
        p01Marker("P01_CHILD_BEGIN \(scenario)")
        switch scenario {
        case "operations-cosine-fused":
            try await Self.fusedRoute(FusedCase(dimension: 768, make: { try Vector768Optimized($0) }, kernel: { range, query, rows, out in
                BatchKernels.range_cosine_fused_768(query: query, candidates: rows, range: range, out: out)
            }))
            try await Self.fusedRoute(FusedCase(dimension: 1536, make: { try Vector1536Optimized($0) }, kernel: { range, query, rows, out in
                BatchKernels.range_cosine_fused_1536(query: query, candidates: rows, range: range, out: out)
            }))
            try await Self.dotStreamsContrast()
        default:
            Issue.record("Unknown P01 scenario \(scenario)")
        }
        p01Marker("P01_CHILD_COMPLETE \(scenario)")
    }

    /// Row `row % 13` classes of the exceptional fixture: NaN, +inf, -inf, zero, negative zero and a magnitude
    /// whose squared norm overflows. Other rows (and every row of a finite fixture) are finite and moderate.
    private static let exceptionalClasses: Set<Int> = [1, 4, 6, 8, 10, 12]

    private static func fixture<V: VectorProtocol>(
        dimension: Int, count: Int, seed: UInt64, exceptional: Bool, make: ([Float]) throws -> V
    ) throws -> (query: V, rows: [V]) where V.Scalar == Float {
        var random = P01OpsRandom(state: seed)
        let query = try make((0..<dimension).map { _ in random.unit() })
        let rows = try (0..<count).map { row -> V in
            var v = (0..<dimension).map { _ in random.unit() * (1 + Float(row % 5)) }
            if exceptional {
                switch row % 13 {
                case 1: v[3] = .nan
                case 4: v[0] = .infinity
                case 6: v[5] = -.infinity
                case 8: v = [Float](repeating: 0, count: dimension)
                case 10: v = [Float](repeating: -0.0, count: dimension)
                case 12: v = v.map { $0 * 1e30 }
                default: break
                }
            }
            return try make(v)
        }
        return (query, rows)
    }

    /// One optimized type's fused route: its dimension, constructor and `BatchKernels` range helper.
    private struct FusedCase<V: VectorProtocol> where V.Scalar == Float {
        let dimension: Int
        let make: ([Float]) throws -> V
        let kernel: (Range<Int>, V, [V], UnsafeMutableBufferPointer<Float>) -> Void
    }

    /// Runs one cosine search and checks it against the fused kernel over the ranges the route used. The route is
    /// identified without score bits: the published calibration, no unhinted reduction or partial heap (no generic
    /// stream), and the branch and hint the ledger shows. Since T14 phase 3 the `CosineDistance` witness on these types
    /// computes the fused kernel's arithmetic, so a score cannot tell the fused kernel from `metric.distance`;
    /// `witnessDiffers` in the marker is informational.
    private static func checkFused<V: VectorProtocol>(
        _ route: FusedCase<V>, data: (query: V, rows: [V]), k: Int, inner: CPUComputeProvider, label: String
    ) async throws where V.Scalar == Float {
        let dimension = route.dimension, kernel = route.kernel
        let (query, rows) = data
        let count = rows.count
        let recording = P01RecordingProvider(inner: inner)
        let result = try await Operations.$computeProvider.withValue(recording) {
            try await Operations.findNearest(to: query, in: rows, k: k, metric: CosineDistance())
        }
        guard let tuning = AutoTuning.get(dim: dimension, kind: .cosineFused) else {
            Issue.record("cosineFused \(dimension): calibration was not published (the fused route did not run)")
            return
        }
        let ledger = recording.ledger
        #expect(ledger.reduceWithoutHint == 0 && ledger.partials.isEmpty, "\(label): the generic stream ran")
        // The branch the route took, as observed: the hinted reduction (parallel) or none (serial steps).
        let parallel = !ledger.reduceHints.isEmpty
        #expect(parallel == (count >= tuning.breakEvenN), "\(label): branch vs breakEvenN \(tuning.breakEvenN)")
        if parallel { #expect(ledger.reduceHints == [tuning.minChunk], "\(label)") }
        let ranges = parallel
            ? ledger.ranges
            : stride(from: 0, to: count, by: tuning.minChunk).map { $0..<min($0 + tuning.minChunk, count) }
        #expect(ranges.first?.lowerBound == 0 && ranges.last?.upperBound == count, "\(label): \(ranges)")
        #expect(zip(ranges, ranges.dropFirst()).allSatisfy { $0.upperBound == $1.lowerBound }, "\(label): \(ranges)")
        var scores = [Float](repeating: 0, count: count)
        scores.withUnsafeMutableBufferPointer { buffer in
            for range in ranges { kernel(range, query, rows, UnsafeMutableBufferPointer(rebasing: buffer[range])) }
        }
        #expect(p01Entries(result) == P01OpsOracle.select(scores, k: k), "\(label) parallel=\(parallel)")
        let differs = result.contains { p01Witness(CosineDistance(), query, rows[$0.index]).bitPattern != $0.distance.bitPattern }
        p01Marker("P01_OBSERVED cosine-fused \(label) parallel=\(parallel) breakEvenN=\(tuning.breakEvenN) "
                  + "minChunk=\(tuning.minChunk) ranges=\(ranges.count) noHintReduce=\(ledger.reduceWithoutHint) "
                  + "hints=\(ledger.reduceHints) witnessDiffers=\(differs) first=\(result.prefix(3).map(\.index))")
    }

    private static func fusedRoute<V: VectorProtocol>(_ route: FusedCase<V>) async throws where V.Scalar == Float {
        let dimension = route.dimension, make = route.make
        let wasCold = AutoTuning.get(dim: dimension, kind: .cosineFused) == nil
        #expect(wasCold, "cosineFused \(dimension) was warm in a fresh child")
        p01Marker("P01_OBSERVED cosine-fused D=\(dimension) cold=\(wasCold)")

        // Finite fixture, n = 64: below every calibrated break-even (at least 200), and the serial step (a multiple
        // of 64) makes 0..<64 one range. Seed 1. (Before T14 phase 3 the seed was chosen so that a fused score differed
        // from the CosineDistance witness's; the witness now computes the fused arithmetic, so the route is identified
        // by `checkFused`'s ledger and calibration observations instead.)
        let seed: UInt64 = 1
        let finite = try fixture(dimension: dimension, count: 64, seed: seed, exceptional: false, make: make)
        p01Marker("P01_OBSERVED cosine-fused D=\(dimension) finite seed=\(seed)")
        for k in [3, 64] {
            try await checkFused(route, data: finite, k: k, inner: CPUComputeProvider(mode: .sequential),
                                 label: "D=\(dimension) finite n=64 k=\(k)")
            // Independent Double reference for the finite fixture.
            let result = try await Operations.findNearest(to: finite.query, in: finite.rows, k: k, metric: CosineDistance())
            let q = finite.query.toArray()
            let qq = q.reduce(0.0) { $0 + Double($1) * Double($1) }
            for entry in result {
                let c = finite.rows[entry.index].toArray()
                let dot = zip(q, c).reduce(0.0) { $0 + Double($1.0) * Double($1.1) }
                let cc = c.reduce(0.0) { $0 + Double($1) * Double($1) }
                let reference = 1 - max(-1, min(1, dot / (qq.squareRoot() * cc.squareRoot())))
                let distance = Double(entry.distance)
                #expect(distance.isFinite && abs(distance - reference) <= 1e-5, "D=\(dimension) row \(entry.index)")
            }
        }

        // Exceptional fixtures: complete-score parity (k = n) and small k, serial (n = 64) and, normally, parallel
        // (n = 4000: AutoTuning clamps a finite break-even to at most 4000). `checkFused` reads the branch from the
        // provider ledger and checks it against the published break-even, so an unclamped tuning is still checked.
        for (count, inner) in [(64, CPUComputeProvider(mode: .sequential)), (4000, CPUComputeProvider(mode: .parallel))] {
            let data = try fixture(dimension: dimension, count: count, seed: UInt64(dimension + count),
                                   exceptional: true, make: make)
            for k in [3, count] {
                try await checkFused(route, data: data, k: k, inner: inner, label: "D=\(dimension) exceptional n=\(count) k=\(k)")
            }
        }
    }

    /// Contrast: dot product on Vector768Optimized has no fused route, so small k streams through the no-hint
    /// reduction, publishes no tuning, and returns the DotProductDistance witness's own score bits.
    private static func dotStreamsContrast() async throws {
        let wasCold = AutoTuning.get(dim: 768, kind: .dot) == nil
        let (query, rows) = try fixture(dimension: 768, count: 64, seed: 5, exceptional: true, make: { try Vector768Optimized($0) })
        let recording = P01RecordingProvider(inner: CPUComputeProvider(mode: .sequential))
        let result = try await Operations.$computeProvider.withValue(recording) {
            try await Operations.findNearest(to: query, in: rows, k: 3, metric: DotProductDistance())
        }
        let expected = P01OpsOracle.select(rows.map { p01Witness(DotProductDistance(), query, $0) }, k: 3)
        let published = AutoTuning.get(dim: 768, kind: .dot) != nil
        #expect(wasCold && !published, "dot 768 small k published a tuning")
        #expect(recording.ledger.reduceWithoutHint == 1 && recording.ledger.reduceHints.isEmpty)
        #expect(recording.ledger.partials.map(\.capacity) == recording.ledger.partials.map { min(3, $0.range.count) })
        #expect(p01Entries(result) == expected)
        p01Marker("P01_OBSERVED dot768-stream cold=\(wasCold) published=\(published) "
                  + "noHintReduce=\(recording.ledger.reduceWithoutHint) partials=\(recording.ledger.partials)")
    }
}

/// A custom witness for the route matrix: the sum of absolute coordinate differences.
private struct P01RouteMatrixMetric: DistanceMetric {
    typealias Scalar = Float
    var name: String { "P01 route matrix" }
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float {
        var sum: Float = 0
        for i in 0..<a.scalarCount { sum += abs(a[i] - b[i]) }
        return sum
    }
}
