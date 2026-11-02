import Foundation
import Testing
@testable import VectorCore

// P01 phase F route observations for `BatchOperations.findNearest` after the small-k / large-k entry split: the
// phase-1 inventory §5 retained-assertion checklist (six public routes, entry snapshot, partitions with minimum chunk
// size 1 and Int.max and oversubscription leastNonzero and 4, custom and parameterized witnesses, cancelled-task
// complete results, nested runs without submitted tasks, submitted <= limit with matched completions, P <= min(C, L)
// partial heaps, a logical range per chunk), the E5 small-k twin, the E6/E7 gaps, root-score admission (B3) and the
// Q17 routing-premise twins of `TopKNaNWrapperTests`. Each scenario runs in a fresh isolated child through
// `e06IsolatedScenario` (E06ConfigurationTestHelpers.swift, reused unchanged). Every optimized membership check is
// preceded by a complete per-candidate bit-parity check of the same fixture (B10 before membership).

@Suite("P01 BatchOperations routes (phase F)", .serialized)
struct P01BatchRouteTests {

    // MARK: Six optimized public routes (B1, B3-B8, B10-B16, B18)

    @Test func optimizedEuclid512() async throws {
        try await e06IsolatedScenario("p01-batch-euclid512", filter: "P01BatchRouteTests/optimizedEuclid512") {
            try await optimizedRoute(P01BatchCase(dim: 512, kind: .euclid))
        }
    }

    @Test func optimizedCosine512() async throws {
        try await e06IsolatedScenario("p01-batch-cosine512", filter: "P01BatchRouteTests/optimizedCosine512") {
            try await optimizedRoute(P01BatchCase(dim: 512, kind: .cosine))
        }
    }

    @Test func optimizedEuclid768() async throws {
        try await e06IsolatedScenario("p01-batch-euclid768", filter: "P01BatchRouteTests/optimizedEuclid768") {
            try await optimizedRoute(P01BatchCase(dim: 768, kind: .euclid))
        }
    }

    @Test func optimizedCosine768() async throws {
        try await e06IsolatedScenario("p01-batch-cosine768", filter: "P01BatchRouteTests/optimizedCosine768") {
            try await optimizedRoute(P01BatchCase(dim: 768, kind: .cosine))
        }
    }

    @Test func optimizedEuclid1536() async throws {
        try await e06IsolatedScenario("p01-batch-euclid1536", filter: "P01BatchRouteTests/optimizedEuclid1536") {
            try await optimizedRoute(P01BatchCase(dim: 1536, kind: .euclid))
        }
    }

    @Test func optimizedCosine1536() async throws {
        try await e06IsolatedScenario("p01-batch-cosine1536", filter: "P01BatchRouteTests/optimizedCosine1536") {
            try await optimizedRoute(P01BatchCase(dim: 1536, kind: .cosine))
        }
    }

    /// Settings: serial; E1's (8, 3, 4); chunk size 1 with oversubscription 4; one chunk (Int.max, leastNonzero).
    static let optimizedSettings: [(threshold: Int, minimum: Int, factor: Double)] = [
        (Int.max, 256, 2), (8, 3, 4), (8, 1, 4), (8, Int.max, .leastNonzeroMagnitude)
    ]

    /// One optimized fixture under test.
    struct OptimizedFixture {
        let c: P01BatchCase
        let fixture: P01BatchFixture
        let built: P01BatchRows
        let typed: P01BatchTyped
    }

    private func optimizedRoute(_ c: P01BatchCase) async throws {
        for fixture in [P01BatchFixture.finite, .exceptional] {
            for count in [260, 1_101] {
                let built = try p01BatchRows(fixture, case: c, count: count, seed: 0xF06 &+ UInt64(count))
                let subject = OptimizedFixture(c: c, fixture: fixture, built: built,
                                               typed: try p01BatchTyped(c, query: built.query, rows: built.rows))
                for setting in Self.optimizedSettings {
                    try await BatchOperations.updateConfiguration {
                        $0.parallelThreshold = setting.threshold; $0.minimumChunkSize = setting.minimum
                        $0.oversubscription = setting.factor
                    }
                    let mode = p01Mode(count: count, configuration: await BatchOperations.configuration())
                    try await checkOptimizedCalls(subject, mode: mode, label: "\(c) \(fixture) n=\(count) \(mode)")
                }
            }
        }
    }

    private func checkOptimizedCalls(_ subject: OptimizedFixture, mode: P01SmallKMode, label: String) async throws {
        let c = subject.c, typed = subject.typed, count = typed.count
        // B10 first: the seam's complete scores for this plan equal the untiled helper's.
        let seam = BatchOperations.smallKScores(route: typed.route, plan: p01Plan(typed.route, count: count, mode: mode))
        let reference = await p01Reference(typed, mode: mode, serialStep: c.serialStep)
        #expect(p01ScoreMismatches(seam, reference: reference).isEmpty, "\(label): parity")
        for k in [1, 3, count / 10 - 1] {
            let observed = await p01ObservedNearest(cancelled: true) { await typed.nearest(k) }
            #expect(p01BatchEntries(observed.result) == P01OpsOracle.select(reference, k: k), "\(label) k=\(k)")
            p01CheckSmallK(.optimized(c, count: count, k: k, mode: mode), observed, label: "\(label) k=\(k)")
            if subject.fixture == .finite {
                checkDoubleReference(c, built: subject.built, result: observed.result, label: "\(label) k=\(k)")
            }
        }
        // Large-k control: no small-k trace; its complete per-candidate scores are the untiled helper's output (the
        // private serial helper on the serial route), and they equal the tiled seam's scores bit for bit.
        let large = await p01ObservedNearest(cancelled: true) { await typed.nearest(Int.max) }
        #expect(large.trace.snapshot().isEmpty, "\(label): large k used the small-k route")
        var readout = [Float](repeating: .nan, count: count)
        for item in large.result { readout[item.index] = item.distance }
        #expect(Set(large.result.map(\.index)).count == count, "\(label): large-k coverage")
        #expect(p01ScoreMismatches(seam, reference: readout).isEmpty, "\(label): tiled vs large-k route scores")
        if case .serial = mode {
            #expect(large.events.snapshot().allSatisfy { $0.kind != .submitted }, "\(label): serial large k")
        } else {
            #expect(large.events.submitted(phase: c.phase) > 0, "\(label): parallel large k")
        }
    }

    /// T7: finite fixture scores against an independent Double formula (tolerances reject NaN).
    private func checkDoubleReference(_ c: P01BatchCase, built: P01BatchRows, result: [(index: Int, distance: Float)],
                                      label: String) {
        for item in result {
            let row = built.rows[item.index]
            var dot = 0.0, qq = 0.0, cc = 0.0, square = 0.0
            for i in 0..<c.dim {
                let a = Double(built.query[i]), b = Double(row[i])
                dot += a * b; qq += a * a; cc += b * b; square += (a - b) * (a - b)
            }
            let expected = c.kind == .euclid ? square.squareRoot() : 1 - dot / (qq.squareRoot() * cc.squareRoot())
            let tolerance = c.kind == .euclid ? 1e-5 * max(1, expected) : 1e-5
            #expect(item.distance.isFinite && abs(Double(item.distance) - expected) <= tolerance,
                    "\(label): i=\(item.index) \(item.distance) vs \(expected)")
        }
    }

    // MARK: Root-score admission (B3)

    @Test func rootTieAdmission() async throws {
        try await e06IsolatedScenario("p01-batch-root-tie", filter: "P01BatchRouteTests/rootTieAdmission") {
            for dim in [512, 768, 1536] {
                let c = P01BatchCase(dim: dim, kind: .euclid)
                let (b, a) = try p01RootTiePair(c)
                for (p, q) in [(0, 299), (130, 131), (255, 256), (298, 299)] {
                    var rows = (0..<300).map { index -> [Float] in
                        var row = [Float](repeating: 0, count: dim); row[0] = 4; row[2] = Float(index % 5); return row
                    }
                    rows[p] = b
                    rows[q] = a
                    let typed = try p01BatchTyped(c, query: [Float](repeating: 0, count: dim), rows: rows)
                    let squared = typed.squared([0..<300]), roots = typed.scores([0..<300])
                    #expect(squared[p] > squared[q] && roots[p].bitPattern == roots[q].bitPattern, "\(c) precondition (\(p), \(q))")
                    try await checkRootTie(c, typed: typed, p: p, q: q)
                }
            }
        }
    }

    private func checkRootTie(_ c: P01BatchCase, typed: P01BatchTyped, p: Int, q: Int) async throws {
        for setting in [(Int.max, 256, 2.0), (8, 3, 4.0), (8, 1, 4.0)] {
            try await BatchOperations.updateConfiguration {
                $0.parallelThreshold = setting.0; $0.minimumChunkSize = setting.1; $0.oversubscription = setting.2
            }
            let mode = p01Mode(count: 300, configuration: await BatchOperations.configuration())
            let label = "\(c) (\(p), \(q)) \(mode)"
            let reference = await p01Reference(typed, mode: mode, serialStep: c.serialStep)
            let seam = BatchOperations.smallKScores(route: typed.route, plan: p01Plan(typed.route, count: 300, mode: mode))
            #expect(p01ScoreMismatches(seam, reference: reference).isEmpty, "\(label): parity")
            for k in [1, 2] {
                let observed = await p01ObservedNearest(cancelled: false) { await typed.nearest(k) }
                // Root admission: equal roots tie and keep the smaller index p; squared admission would pick q.
                #expect(observed.result.map(\.index) == Array([p, q].prefix(k)), "\(label) k=\(k): \(observed.result)")
                #expect(observed.result.allSatisfy { $0.distance.bitPattern == reference[p].bitPattern }, "\(label) k=\(k)")
                p01CheckSmallK(.optimized(c, count: 300, k: k, mode: mode), observed, label: "\(label) k=\(k)")
            }
            let large = await typed.nearest(Int.max)
            #expect(Array(large.prefix(2).map(\.index)) == [p, q], "\(label): large-k control")
        }
    }

    // MARK: Generic partitions (E5 small-k twin; B11-B16)

    @Test func genericPartitions() async throws {
        try await e06IsolatedScenario("p01-batch-generic-partitions", filter: "P01BatchRouteTests/genericPartitions") {
            let rows = p01TaggedRows((0..<41).map { Float(($0 * 7) % 11) })
            let query = p01TaggedQuery()
            let scores = p01SpyScores(query, rows)
            try await BatchOperations.updateConfiguration { $0.parallelThreshold = 1 }
            let cores = BoundedExecution.hardwareLimit
            for minimum in [1, 3, Int.max] {
                for factor in [Double.leastNonzeroMagnitude, 1, 2, 4] {
                    try await BatchOperations.updateConfiguration { $0.minimumChunkSize = minimum; $0.oversubscription = factor }
                    // Independent stride (the E5 formula for 41 rows).
                    let target = factor == .leastNonzeroMagnitude ? 1 : min(41, cores * Int(factor))
                    let step = min(41, max(minimum, 41 / target + (41 % target == 0 ? 0 : 1)))
                    for k in [1, 3] {
                        let calls = P01OpsCalls()
                        let metric = P01OpsSpyMetric(calls: calls)
                        let observed = await p01ObservedNearest(cancelled: true) {
                            await BatchOperations.findNearest(to: query, in: rows, k: k, metric: metric)
                        }
                        let label = "min=\(minimum) factor=\(factor) k=\(k)"
                        #expect(p01BatchEntries(observed.result) == P01OpsOracle.select(scores, k: k), "\(label)")
                        p01CheckSmallK(.generic(count: 41, k: k, mode: .parallel(chunkSize: step)), observed, label: label)
                        p01CheckCallsPerChunk(calls.snapshot(), count: 41, chunk: step, label: label)
                        let chunks = (41 + step - 1) / step
                        print("P01F_E5_TWIN \(label) step=\(step) L=\(chunks) P=\(observed.trace.partials.count) " +
                              "submitted=\(observed.events.submitted(phase: "batch.nearest")) C=\(cores)")
                    }
                }
            }
        }
    }

    // MARK: Witnesses and threshold neighbors (E6, E7 gaps)

    @Test func genericWitnessesAndThresholds() async throws {
        try await e06IsolatedScenario("p01-batch-witnesses", filter: "P01BatchRouteTests/genericWitnessesAndThresholds") {
            // E7: a custom witness is called exactly once per candidate, in index order within each original range.
            let rows = p01TaggedRows((0..<30).map { Float(($0 * 5) % 7) })
            let query = p01TaggedQuery()
            for setting in [(Int.max, 256), (1, 1), (1, 4), (1, 256)] {
                try await BatchOperations.updateConfiguration { $0.parallelThreshold = setting.0; $0.minimumChunkSize = setting.1 }
                let mode = p01Mode(count: 30, configuration: await BatchOperations.configuration())
                for k in [1, 2] {
                    let calls = P01OpsCalls()
                    let metric = P01OpsSpyMetric(calls: calls)
                    let observed = await p01ObservedNearest(cancelled: false) {
                        await BatchOperations.findNearest(to: query, in: rows, k: k, metric: metric)
                    }
                    let label = "custom \(mode) k=\(k)"
                    #expect(p01BatchEntries(observed.result) == P01OpsOracle.select(p01SpyScores(query, rows), k: k), "\(label)")
                    p01CheckSmallK(.generic(count: 30, k: k, mode: mode), observed, label: label)
                    let chunk: Int
                    if case .parallel(let size) = mode { chunk = size } else { chunk = 30 }
                    p01CheckCallsPerChunk(calls.snapshot(), count: 30, chunk: chunk, label: label)
                }
            }
            try await parameterizedWitness()
            try await thresholdNeighbors()
        }
    }

    /// The parameterized `MinkowskiDistance(p: 3)` witness decides the answer (Euclidean would pick index 4).
    private func parameterizedWitness() async throws {
        func row(_ x: Float, _ y: Float, dim: Int) -> [Float] { var r = [Float](repeating: 0, count: dim); r[0] = x; r[1] = y; return r }
        for dim in [2, 768] {
            var values = (0..<30).map { row(10 + Float($0 % 3), 10, dim: dim) }
            values[4] = row(0, 4, dim: dim)
            values[17] = row(3, 3, dim: dim)
            let metric = MinkowskiDistance(p: 3)
            try await BatchOperations.updateConfiguration { $0.parallelThreshold = 1; $0.minimumChunkSize = 4 }
            let results: [[(index: Int, distance: Float)]]
            let expected: [Float]
            if dim == 2 {
                let vectors = values.map { DynamicVector($0) }, query = DynamicVector(row(0, 0, dim: 2))
                let minkowski = await BatchOperations.findNearest(to: query, in: vectors, k: 1, metric: metric)
                let euclidean = await BatchOperations.findNearest(to: query, in: vectors, k: 1, metric: EuclideanDistance())
                results = [minkowski, euclidean]
                expected = vectors.map { p01Witness(metric, query, $0) }
            } else {
                let vectors = try values.map { try Vector768Optimized($0) }, query = try Vector768Optimized(row(0, 0, dim: 768))
                let minkowski = await BatchOperations.findNearest(to: query, in: vectors, k: 1, metric: metric)
                let euclidean = await BatchOperations.findNearest(to: query, in: vectors, k: 1, metric: EuclideanDistance())
                results = [minkowski, euclidean]
                expected = vectors.map { p01Witness(metric, query, $0) }
            }
            #expect(p01BatchEntries(results[0]) == P01OpsOracle.select(expected, k: 1), "minkowski dim=\(dim): \(results[0])")
            #expect(results[0].first?.index == 17 && results[1].first?.index == 4, "dim=\(dim): \(results)")
            let reference = pow(54.0, 1.0 / 3.0)
            #expect(results[0].allSatisfy { $0.distance.isFinite && abs(Double($0.distance) - reference) < 1e-5 })
        }
    }

    /// E6: live threshold neighbors with k = 1 (all small k, since n / 10 >= 2), generic and optimized.
    private func thresholdNeighbors() async throws {
        try await BatchOperations.updateConfiguration { $0 = BatchOperations.Configuration(); $0.parallelThreshold = 30 }
        let c = P01BatchCase(dim: 512, kind: .euclid)
        for count in [29, 30, 31] {
            let mode = p01Mode(count: count, configuration: await BatchOperations.configuration())
            #expect((count >= 30) == (mode != .serial))
            let rows = p01TaggedRows((0..<count).map { Float(($0 * 3) % 5) })
            let query = p01TaggedQuery()
            let generic = await p01ObservedNearest(cancelled: true) {
                await BatchOperations.findNearest(to: query, in: rows, k: 1, metric: P01OpsSpyMetric(calls: P01OpsCalls()))
            }
            #expect(p01BatchEntries(generic.result) == P01OpsOracle.select(p01SpyScores(query, rows), k: 1))
            p01CheckSmallK(.generic(count: count, k: 1, mode: mode), generic, label: "threshold generic n=\(count)")
            let built = try p01BatchRows(.finite, case: c, count: count)
            let typed = try p01BatchTyped(c, query: built.query, rows: built.rows)
            let reference = await p01Reference(typed, mode: mode, serialStep: c.serialStep)
            let optimized = await p01ObservedNearest(cancelled: true) { await typed.nearest(1) }
            #expect(p01BatchEntries(optimized.result) == P01OpsOracle.select(reference, k: 1))
            p01CheckSmallK(.optimized(c, count: count, k: 1, mode: mode), optimized, label: "threshold euclid512 n=\(count)")
        }
    }

    // MARK: Cancellation and nesting (B2, B14)

    @Test func cancelledAndNested() async throws {
        try await e06IsolatedScenario("p01-batch-cancel-nested", filter: "P01BatchRouteTests/cancelledAndNested") {
            for setting in [(1, 7), (Int.max, 256)] {
                try await BatchOperations.updateConfiguration { $0.parallelThreshold = setting.0; $0.minimumChunkSize = setting.1 }
                let mode = p01Mode(count: 300, configuration: await BatchOperations.configuration())
                for c in P01BatchCase.all {
                    let built = try p01BatchRows(.exceptional, case: c, count: 300)
                    let typed = try p01BatchTyped(c, query: built.query, rows: built.rows)
                    let reference = await p01Reference(typed, mode: mode, serialStep: c.serialStep)
                    try await checkCancelledAndNested(.optimized(c, count: 300, k: 3, mode: mode),
                                                      expected: P01OpsOracle.select(reference, k: 3), label: "\(c)") {
                        await typed.nearest(3)
                    }
                }
                let rows = p01TaggedRows((0..<300).map { Float(($0 * 13) % 17) })
                let query = p01TaggedQuery()
                try await checkCancelledAndNested(.generic(count: 300, k: 3, mode: mode),
                                                  expected: P01OpsOracle.select(p01SpyScores(query, rows), k: 3), label: "generic") {
                    await BatchOperations.findNearest(to: query, in: rows, k: 3, metric: P01OpsSpyMetric(calls: P01OpsCalls()))
                }
            }
        }
    }

    private func checkCancelledAndNested(_ expect: P01Expect, expected: [P01OpsEntry], label: String,
                                         _ call: @Sendable @escaping () async -> [(index: Int, distance: Float)]) async throws {
        for cancelled in [false, true] {
            let observed = await p01ObservedNearest(cancelled: cancelled, call)
            #expect(p01BatchEntries(observed.result) == expected, "\(label) \(expect.mode) cancelled=\(cancelled)")
            p01CheckSmallK(expect, observed, label: "\(label) \(expect.mode) cancelled=\(cancelled)")
        }
        // Nested inside a library region: same partitions and result, no submitted tasks.
        let nestedMode: P01SmallKMode
        if case .parallel(let size) = expect.mode { nestedMode = .nested(chunkSize: size) } else { nestedMode = .serial }
        let nestedExpect = P01Expect(count: expect.count, k: expect.k, mode: nestedMode, phase: expect.phase,
                                     tiled: expect.tiled, serialStep: expect.serialStep)
        let nested = try await CPUComputeProvider.parallel.parallelExecute(items: 0..<1) { _ in
            await p01ObservedNearest(cancelled: true, call)
        }
        #expect(nested.count == 1)
        for observed in nested {
            #expect(p01BatchEntries(observed.result) == expected, "\(label) nested")
            p01CheckSmallK(nestedExpect, observed, label: "\(label) nested")
        }
    }

    // MARK: Entry snapshot (B1)

    @Test func entrySnapshot() async throws {
        try await e06IsolatedScenario("p01-batch-snapshot", filter: "P01BatchRouteTests/entrySnapshot") {
            try await BatchOperations.updateConfiguration { $0.parallelThreshold = 1; $0.minimumChunkSize = 5; $0.oversubscription = 4 }
            let rows = p01TaggedRows((0..<60).map { Float(($0 * 7) % 9) })
            let query = p01TaggedQuery()
            let gate = P01BatchGate()
            let task = Task {
                await p01ObservedNearest(cancelled: false) {
                    await BatchOperations.findNearest(to: query, in: rows, k: 2, metric: P01GatedMetric(gate: gate))
                }
            }
            await gate.suspendUntilEntered()
            do {
                try await BatchOperations.updateConfiguration { $0.minimumChunkSize = 13; $0.parallelThreshold = Int.max }
            } catch { gate.release(); _ = await task.value; throw error }
            gate.release()
            let observed = await task.value
            #expect(gate.timedOut == false)
            #expect(p01BatchEntries(observed.result) == P01OpsOracle.select(p01SpyScores(query, rows), k: 2))
            // The call keeps its entry snapshot: parallel, chunk size 5.
            p01CheckSmallK(.generic(count: 60, k: 2, mode: .parallel(chunkSize: 5)), observed, label: "snapshot")
            // A later call sees the new configuration: serial.
            let later = await p01ObservedNearest(cancelled: false) {
                await BatchOperations.findNearest(to: query, in: rows, k: 2, metric: P01OpsSpyMetric(calls: P01OpsCalls()))
            }
            p01CheckSmallK(.generic(count: 60, k: 2, mode: .serial), later, label: "later")
        }
    }

    // MARK: Strategy neighbors (B18, CS1)

    @Test func strategyNeighbors() async throws {
        try await e06IsolatedScenario("p01-batch-strategy", filter: "P01BatchRouteTests/strategyNeighbors") {
            let c = P01BatchCase(dim: 512, kind: .euclid)
            let cases: [(Int, [Int])] = [(20, [1, 2]), (100, [9, 10, 99, 100, 101, Int.max]), (109, [9, 10]), (110, [10, 11])]
            for setting in [(Int.max, 256), (1, 7)] {
                try await BatchOperations.updateConfiguration { $0.parallelThreshold = setting.0; $0.minimumChunkSize = setting.1 }
                for (count, ks) in cases {
                    let mode = p01Mode(count: count, configuration: await BatchOperations.configuration())
                    let built = try p01BatchRows(.exceptional, case: c, count: count)
                    let typed = try p01BatchTyped(c, query: built.query, rows: built.rows)
                    let reference = await p01Reference(typed, mode: mode, serialStep: c.serialStep)
                    let rows = p01TaggedRows((0..<count).map { Float(($0 * 3) % 4) })
                    let query = p01TaggedQuery()
                    for k in ks {
                        let small = min(k, count) < count / 10
                        let optimized = await p01ObservedNearest(cancelled: false) { await typed.nearest(k) }
                        let generic = await p01ObservedNearest(cancelled: false) {
                            await BatchOperations.findNearest(to: query, in: rows, k: k, metric: P01OpsSpyMetric(calls: P01OpsCalls()))
                        }
                        let label = "n=\(count) k=\(k) \(mode)"
                        #expect(p01BatchEntries(optimized.result) == P01OpsOracle.select(reference, k: k), "\(label)")
                        #expect(p01BatchEntries(generic.result) == P01OpsOracle.select(p01SpyScores(query, rows), k: k), "\(label)")
                        #expect(optimized.trace.snapshot().isEmpty == !small && generic.trace.snapshot().isEmpty == !small,
                                "\(label): small=\(small)")
                        if small {
                            p01CheckSmallK(.optimized(c, count: count, k: k, mode: mode), optimized, label: "\(label) optimized")
                            p01CheckSmallK(.generic(count: count, k: k, mode: mode), generic, label: "\(label) generic")
                        }
                    }
                }
            }
        }
    }

    // MARK: Q17 routing-premise twins of TopKNaNWrapperTests

    /// The same fixtures as `TopKNaNWrapperTests.batchSerialHeapAndSortKeepNaNsLast` and
    /// `.batchParallelHeapAndSortPreserveGlobalIndices`, under the default configuration, observing the real route:
    /// serial or parallel by `parallelThreshold`, heap (small-k trace) for k = 3 and sort (no trace) for the large k.
    /// No `Operations.computeProvider` is installed: BatchOperations does not read it (a premise pinned by
    /// `Scripts/ci/tests/test_inert_provider_wrappers.py`).
    @Test func q17PremiseTwins() async throws {
        try await e06IsolatedScenario("p01-batch-q17", filter: "P01BatchRouteTests/q17PremiseTwins") {
            try await BatchOperations.updateConfiguration { $0 = BatchOperations.Configuration() }
            try await q17SerialTwin()
            try await q17ParallelTwin()
        }
    }

    private func q17SerialTwin() async throws {
        let query = Vector<Dim2>(x: 0, y: 0)
        var inputs = Array(repeating: Vector<Dim2>(x: .nan, y: 0), count: 100)
        inputs[99] = query
        let rows = inputs
        for k in [3, 10] {
            let observed = await p01ObservedNearest(cancelled: false) {
                await BatchOperations.findNearest(to: query, in: rows, k: k)
            }
            #expect(observed.result.map(\.index) == [99] + Array(0..<(k - 1)) && observed.result.first?.distance == 0)
            #expect(observed.result.dropFirst().allSatisfy { $0.distance.isNaN })
            #expect(observed.events.snapshot().allSatisfy { $0.kind != .submitted }, "serial premise k=\(k)")
            #expect(observed.trace.partials.count == (k == 3 ? 1 : 0), "serial heap/sort premise k=\(k)")
            if k == 3 { p01CheckSmallK(.generic(count: 100, k: 3, mode: .serial), observed, label: "q17 serial") }
            print("P01F_Q17 fixture=batchSerialHeapAndSortKeepNaNsLast k=\(k) submitted=0 partials=\(observed.trace.partials.count)")
        }
    }

    private func q17ParallelTwin() async throws {
        let zero = try Vector<Dim512>([Float](repeating: 0, count: 512))
        let nan = try Vector<Dim512>([Float](repeating: .nan, count: 512))
        var inputs = Array(repeating: nan, count: 1_024)
        inputs[700] = zero
        inputs[900] = zero
        let rows = inputs
        let chunk = BatchOperations.logicalChunkSize(count: 1_024, cores: BoundedExecution.hardwareLimit,
                                                     configuration: BatchOperations.Configuration())
        for k in [3, 103] {
            let observed = await p01ObservedNearest(cancelled: false) {
                await BatchOperations.findNearest(to: zero, in: rows, k: k)
            }
            #expect(observed.result.map(\.index) == [700, 900] + Array(0..<(k - 2)))
            #expect(observed.result.dropFirst(2).allSatisfy { $0.distance.isNaN })
            let submitted = observed.events.submitted(phase: "batch.nearest")
            #expect(submitted > 0, "parallel premise k=\(k)")
            #expect(observed.trace.partials.isEmpty == (k == 103), "parallel heap/sort premise k=\(k)")
            if k == 3 {
                p01CheckSmallK(.generic(count: 1_024, k: 3, mode: .parallel(chunkSize: chunk)), observed, label: "q17 parallel")
            }
            print("P01F_Q17 fixture=batchParallelHeapAndSortPreserveGlobalIndices k=\(k) submitted=\(submitted) " +
                  "partials=\(observed.trace.partials.count)")
        }
    }
}

// MARK: - Helpers

func p01Mode(count: Int, configuration: BatchOperations.Configuration) -> P01SmallKMode {
    guard count >= configuration.parallelThreshold else { return .serial }
    return .parallel(chunkSize: BatchOperations.logicalChunkSize(count: count, cores: BoundedExecution.hardwareLimit,
                                                                 configuration: configuration))
}

func p01Plan(_ route: BatchOperations.SmallKRoute, count: Int, mode: P01SmallKMode) -> BatchOperations.SmallKPlan {
    switch mode {
    case .serial: return .serial(route: route, count: count)
    case .parallel(let size), .nested(let size):
        return BatchOperations.SmallKPlan(count: count, chunkSize: size, cores: BoundedExecution.hardwareLimit)
    }
}

/// The untiled helper over the same original ranges: the logical chunks, or the serial helper's steps.
func p01Reference(_ typed: P01BatchTyped, mode: P01SmallKMode, serialStep: Int) async -> [Float] {
    switch mode {
    case .serial: return await typed.untiled(serialStep, BoundedExecution.hardwareLimit)
    case .parallel(let size), .nested(let size): return await typed.untiled(size, BoundedExecution.hardwareLimit)
    }
}

/// Exactly one call per candidate; within each original range of `chunk` candidates, calls are in index order.
func p01CheckCallsPerChunk(_ log: [Int], count: Int, chunk: Int, label: String,
                           sourceLocation: SourceLocation = #_sourceLocation) {
    #expect(log.sorted() == Array(0..<count), "\(label): calls \(log.count)", sourceLocation: sourceLocation)
    var position = [Int: Int]()
    for (offset, index) in log.enumerated() { position[index] = offset }
    for start in stride(from: 0, to: count, by: chunk) {
        let order = (start..<min(start + chunk, count)).compactMap { position[$0] }
        #expect(order == order.sorted(), "\(label): chunk at \(start) out of order", sourceLocation: sourceLocation)
    }
}

/// Holds the first metric call until released (bounded by a 5 s wait).
final class P01BatchGate: @unchecked Sendable {
    private let lock = NSLock()
    private var entered = false
    private var continuation: CheckedContinuation<Void, Never>?
    private let releaseSignal = DispatchSemaphore(value: 0)
    private var expired = false
    var timedOut: Bool { lock.lock(); defer { lock.unlock() }; return expired }
    func suspendUntilEntered() async {
        await withCheckedContinuation { continuation in
            lock.lock()
            if entered { lock.unlock(); continuation.resume() } else { self.continuation = continuation; lock.unlock() }
        }
    }
    func holdFirst() {
        lock.lock()
        if entered { lock.unlock(); return }
        entered = true
        let continuation = self.continuation
        self.continuation = nil
        lock.unlock()
        continuation?.resume()
        if releaseSignal.wait(timeout: .now() + 5) == .timedOut { lock.lock(); expired = true; lock.unlock() }
    }
    func release() { releaseSignal.signal() }
}

/// The spy score `b[0] - a[0]`; the first call waits on the gate.
struct P01GatedMetric: DistanceMetric {
    typealias Scalar = Float
    let gate: P01BatchGate
    var name: String { "P01 gated" }
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float {
        gate.holdFirst()
        return b[0] - a[0]
    }
}
