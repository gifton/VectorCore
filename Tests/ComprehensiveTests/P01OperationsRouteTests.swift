import Foundation
import Testing
@testable import VectorCore

// P01 phase E route observations for `Operations.findNearest`/`findNearestBatch`: the Operations subset of the
// phase-1 inventory's retained-assertion checklist (provider precedence; serial, parallel, nested and cancelled
// execution; extreme partitions; custom and parameterized witnesses), observed through the public routes.
// Each scenario runs in a fresh isolated child through `e06IsolatedScenario` (E06ConfigurationTestHelpers.swift,
// reused unchanged), so process-global state such as AutoTuning starts from its defaults. Shared fixtures are in
// P01OperationsFixtures.swift.

@Suite("P01 Operations routes (phase E)", .serialized)
struct P01OperationsRouteTests {

    // MARK: Provider precedence (O3)

    @Test func providerPrecedence() async throws {
        try await e06IsolatedScenario("p01-ops-provider-precedence", filter: "P01OperationsRouteTests/providerPrecedence") {
            let calls = P01OpsCalls()
            let metric = P01OpsSpyMetric(calls: calls, token: 41)
            let rows = p01TaggedRows((0..<50).map { Float($0 % 7) })
            let query = p01TaggedQuery()
            // Not a CPU-shaped answer: unsorted, out of range, a repeated index, a NaN payload, -0 and more than k.
            let reply: [(index: Int, distance: Float)] = [
                (index: 9, distance: 3), (index: -4, distance: -0.0), (index: 9, distance: Float(bitPattern: 0x7FC0_1234)),
                (index: 1_000, distance: -.infinity), (index: 2, distance: 1)
            ]
            let replyEntries = reply.map { P01OpsEntry(index: $0.index, bits: $0.distance.bitPattern) }

            // Single query: the provider gets the original k and the caller's metric; its reply is returned verbatim
            // and the CPU routes never run.
            let log = P01KernelLog()
            let spy = P01KernelSpy(log: log, reply: reply)
            let ks = [1, 3, 49, 50, 51, Int.max]
            for k in ks {
                let result = try await Operations.$computeProvider.withValue(spy) {
                    try await Operations.findNearest(to: query, in: rows, k: k, metric: metric)
                }
                #expect(p01Entries(result) == replyEntries, "k=\(k)")
            }
            #expect(log.singleCalls == ks.map { P01KernelLog.Call(queries: 1, candidates: 50, k: $0, token: 41) })
            #expect(calls.count == 0)

            // Route 1 adds no cancellation check: this provider performs none, so a cancelled task gets its reply.
            let cancelled = await Task {
                withUnsafeCurrentTask { $0?.cancel() }
                return try await Operations.$computeProvider.withValue(spy) {
                    try await Operations.findNearest(to: query, in: rows, k: 3, metric: metric)
                }
            }.result
            #expect((try? cancelled.get()).map(p01Entries) == replyEntries, "cancelled: \(cancelled)")

            // Provider errors pass through unchanged.
            let failing = P01KernelSpy(log: P01KernelLog(), reply: reply, failure: P01OpsError(token: 7))
            await #expect(throws: P01OpsError(token: 7)) {
                _ = try await Operations.$computeProvider.withValue(failing) {
                    try await Operations.findNearest(to: query, in: rows, k: 3, metric: metric)
                }
            }

            // Batch without an override: the protocol default calls findNearest once per query, with the original k.
            let loopLog = P01KernelLog()
            let looped = try await Operations.$computeProvider.withValue(P01KernelSpy(log: loopLog, reply: reply)) {
                try await Operations.findNearestBatch(queries: [query, query, query], in: rows, k: 4, metric: metric)
            }
            #expect(looped.map(p01Entries) == Array(repeating: replyEntries, count: 3))
            #expect(loopLog.singleCalls == Array(repeating: P01KernelLog.Call(queries: 1, candidates: 50, k: 4, token: 41), count: 3))

            // Batch override: one call for the whole batch, ahead of every CPU route (without a provider this input,
            // 9 queries and 300 Vector512Optimized candidates, takes the `.perQuery` default; precedence over `.gram`
            // options is P02BatchProviderTests.providerPrecedesInvalidAndInfeasibleCPUOptions); its rows are returned
            // verbatim.
            var random = P01OpsRandom(state: 3)
            let gemmRows = (0..<300).map { _ in Vector512Optimized(generator: { _ in random.unit() }) }
            let batchLog = P01KernelLog()
            let rowsReply: [[(index: Int, distance: Float)]] = [
                [(index: 5, distance: 2)], [], [(index: 1, distance: .nan), (index: 1, distance: .nan)]
            ]
            let batched = try await Operations.$computeProvider.withValue(
                P01KernelBatchSpy(log: batchLog, reply: reply, batchReply: rowsReply)
            ) {
                try await Operations.findNearestBatch(queries: Array(gemmRows.prefix(9)), in: gemmRows, k: Int.max)
            }
            #expect(batched.map(p01Entries) == rowsReply.map { $0.map { P01OpsEntry(index: $0.index, bits: $0.distance.bitPattern) } })
            #expect(batchLog.batchCalls == [P01KernelLog.Call(queries: 9, candidates: 300, k: Int.max, token: nil)])
            #expect(batchLog.singleCalls.isEmpty)
            let failingBatch = P01KernelBatchSpy(log: P01KernelLog(), reply: reply, batchReply: [], failure: P01OpsError(token: 8))
            await #expect(throws: P01OpsError(token: 8)) {
                _ = try await Operations.$computeProvider.withValue(failingBatch) {
                    try await Operations.findNearestBatch(queries: [query], in: rows, k: 1, metric: metric)
                }
            }

            // Task-local nesting: the innermost binding decides; the outer binding applies again after the scope.
            let outer = P01RecordingProvider(inner: CPUComputeProvider(mode: .sequential))
            let innerLog = P01KernelLog()
            try await Operations.$computeProvider.withValue(outer) {
                let inside = try await Operations.$computeProvider.withValue(P01KernelSpy(log: innerLog, reply: reply)) {
                    try await Operations.findNearest(to: query, in: rows, k: 2, metric: metric)
                }
                #expect(p01Entries(inside) == replyEntries)
                #expect(outer.ledger.isUntouched)
                let after = try await Operations.findNearest(to: query, in: rows, k: 2, metric: metric)
                #expect(after.map(\.index) == [0, 7])
                #expect(outer.ledger.reduceWithoutHint == 1)
            }
            #expect(innerLog.singleCalls.count == 1)
            let cpuInside = P01RecordingProvider(inner: CPUComputeProvider(mode: .sequential))
            let outerLog = P01KernelLog()
            try await Operations.$computeProvider.withValue(P01KernelSpy(log: outerLog, reply: reply)) {
                let inside = try await Operations.$computeProvider.withValue(cpuInside) {
                    try await Operations.findNearest(to: query, in: rows, k: 2, metric: metric)
                }
                #expect(inside.map(\.index) == [0, 7])
            }
            #expect(outerLog.isUntouched)
            #expect(cpuInside.ledger.reduceWithoutHint == 1)
            print("P01_ROUTE provider-precedence single=\(log.singleCalls.count) loop=\(loopLog.singleCalls.count) "
                  + "override=\(batchLog.batchCalls.count)")
        }
    }

    // MARK: Serial, parallel, nested and cancelled execution (O5, H3)

    @Test func serialParallelNestedAndCancelled() async throws {
        try await e06IsolatedScenario("p01-ops-execution", filter: "P01OperationsRouteTests/serialParallelNestedAndCancelled") {
            let count = 4097, k = 3
            // Repeated values give exact ties across ranges; the minimum (0) occurs at several indices.
            let values = (0..<count).map { Float(($0 * 7919) % 1009) }
            let rows = p01TaggedRows(values)
            let query = p01TaggedQuery()
            let expected = P01OpsOracle.select(p01SpyScores(query, rows), k: k)
            let calls = P01OpsCalls()
            let metric = P01OpsSpyMetric(calls: calls)

            let modes: [(String, CPUComputeProvider)] = [
                ("sequential", CPUComputeProvider(mode: .sequential)),
                ("parallel", CPUComputeProvider(mode: .parallel)),
                ("automatic", CPUComputeProvider(mode: .automatic))
            ]
            for (label, inner) in modes {
                let recording = P01RecordingProvider(inner: inner)
                let events = P01OpsEvents()
                calls.reset()
                let result = try await BoundedExecution.$observation.withValue({ events.record($0) }, operation: {
                    try await Operations.$computeProvider.withValue(recording) {
                        try await Operations.findNearest(to: query, in: rows, k: k, metric: metric)
                    }
                })
                #expect(p01Entries(result) == expected, "\(label)")
                #expect(recording.ledger.reduceWithoutHint == 1 && recording.ledger.reduceHints.isEmpty, "\(label)")
                p01CheckStream(recording.ledger, calls: calls.snapshot(), count: count, k: k, label: label)
                events.verify()
                // CPU parallel mode splits into ceil(n / width) workers, width = max(256, ceil(n / min(n, limit)));
                // sequential and automatic below 50,000 candidates run the one range inline.
                let limit = inner.maxConcurrency
                let width = max(256, (count + min(count, limit) - 1) / min(count, limit))
                let workers = label == "parallel" ? (count + width - 1) / width : 0
                let submitted = events.submitted(phase: "cpu.reduce")
                #expect(submitted == workers, "\(label): submitted \(submitted), expected \(workers)")
                print("P01_ROUTE execution \(label) limit=\(limit) submitted=\(submitted) partials=\(recording.ledger.partials.count)")
            }

            // Nested: inside a library worker region the reduction runs inline, so it submits no workers. Both a
            // caller's region and the batch fallback's own per-query region are observed.
            let nestedEvents = P01OpsEvents()
            let nested = try await BoundedExecution.$observation.withValue({ nestedEvents.record($0) }, operation: {
                try await Operations.$computeProvider.withValue(CPUComputeProvider(mode: .parallel)) {
                    let inRegion = try await CPUComputeProvider(mode: .parallel).parallelExecute(items: 0..<2) { _ in
                        try await Operations.findNearest(to: query, in: rows, k: k, metric: P01OpsSpyMetric(calls: P01OpsCalls()))
                    }
                    let batch = try await Operations.findNearestBatch(queries: [query, query, query], in: rows, k: k,
                                                                      metric: P01OpsSpyMetric(calls: P01OpsCalls()))
                    return inRegion + batch
                }
            })
            #expect(nested.count == 5 && nested.allSatisfy { p01Entries($0) == expected })
            let reductions = nestedEvents.snapshot().filter { $0.phase == "cpu.reduce" }
            #expect(reductions.filter { $0.kind == .begin }.count == 5, "every nested search reached the provider")
            #expect(reductions.filter { $0.kind == .submitted }.isEmpty, "a nested search submitted workers")
            nestedEvents.verify()

            try await Self.cancellationByRoute(query: query, rows: rows, k: k, calls: calls)
        }
    }

    /// Cancellation per route of `findNearest` (its doc's "Cancellation differs by route" list), on the stream's
    /// fixture (`rows`, `k`) and on small route-2 and route-4 fixtures.
    private static func cancellationByRoute(query: DynamicVector, rows: [DynamicVector], k: Int, calls: P01OpsCalls) async throws {
        let metric = P01OpsSpyMetric(calls: calls)
        let modes: [(String, CPUComputeProvider)] = [
            ("sequential", CPUComputeProvider(mode: .sequential)),
            ("parallel", CPUComputeProvider(mode: .parallel)),
            ("automatic", CPUComputeProvider(mode: .automatic))
        ]
        // Cancelled before the call: the provider's entry check throws before any metric call.
        for (label, inner) in modes {
            calls.reset()
            let outcome = await Task {
                withUnsafeCurrentTask { $0?.cancel() }
                return try await Operations.$computeProvider.withValue(inner) {
                    try await Operations.findNearest(to: query, in: rows, k: k, metric: metric)
                }
            }.result
            #expect(Self.isCancellation(outcome), "\(label): \(outcome)")
            #expect(calls.count == 0, "\(label): \(calls.count) metric calls after cancellation")
        }
        // Cancelled, with n = 64: below every calibrated break-even (at least 200), where the pre-P01 full-score
        // route used a synchronous map without cancellation checks. The stream still throws before any call.
        let small = p01TaggedRows((0..<64).map { Float($0 % 9) })
        calls.reset()
        let smallOutcome = await Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await Operations.findNearest(to: query, in: small, k: 1, metric: metric)
        }.result
        #expect(Self.isCancellation(smallOutcome), "n=64: \(smallOutcome)")
        #expect(calls.count == 0, "n=64: \(calls.count) metric calls after cancellation")
        // Cancelled during the scan: the check before the next candidate throws, so no later call happens.
        calls.reset()
        let midScan = await Task {
            try await Operations.$computeProvider.withValue(CPUComputeProvider(mode: .sequential)) {
                try await Operations.findNearest(to: query, in: rows, k: k, metric: P01OpsSpyMetric(calls: calls, cancelAfter: 6))
            }
        }.result
        #expect(Self.isCancellation(midScan))
        #expect(calls.snapshot() == Array(0..<6))
        // Route 3 under a provider that performs no cancellation checks (P01UncheckedProviderProbe, not
        // conforming): the scan's own check still throws before the next metric call.
        let unchecked = P01UncheckedProviderProbe(plan: { [$0] })
        calls.reset()
        let uncheckedBefore = await Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await Operations.$computeProvider.withValue(unchecked) {
                try await Operations.findNearest(to: query, in: rows, k: k, metric: metric)
            }
        }.result
        #expect(Self.isCancellation(uncheckedBefore), "unchecked provider: \(uncheckedBefore)")
        #expect(calls.count == 0, "unchecked provider: \(calls.count) metric calls after cancellation")
        calls.reset()
        let uncheckedMidScan = await Task {
            try await Operations.$computeProvider.withValue(unchecked) {
                try await Operations.findNearest(to: query, in: rows, k: k, metric: P01OpsSpyMetric(calls: calls, cancelAfter: 6))
            }
        }.result
        #expect(Self.isCancellation(uncheckedMidScan))
        #expect(calls.snapshot() == Array(0..<6))

        // Route 2 (direct optimized) relies only on the provider: the built-in CPU providers throw, and the
        // unchecked probe lets a cancelled search complete.
        var random = P01OpsRandom(state: 11)
        let rows512 = (0..<64).map { _ in Vector512Optimized(generator: { _ in random.unit() }) }
        for (label, inner) in modes {
            let direct = await Task {
                withUnsafeCurrentTask { $0?.cancel() }
                return try await Operations.$computeProvider.withValue(inner) {
                    try await Operations.findNearest(to: rows512[0], in: rows512, k: 3)
                }
            }.result
            #expect(Self.isCancellation(direct), "direct \(label): \(direct)")
        }
        let directUnchecked = await Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await Operations.$computeProvider.withValue(unchecked) {
                try await Operations.findNearest(to: rows512[0], in: rows512, k: 3)
            }
        }.result
        #expect((try? directUnchecked.get())?.map(\.index).first == 0, "direct, unchecked provider: \(directUnchecked)")

        // Route 4 (full score) checks only on its provider-scheduled parallel branch. Below every calibrated
        // break-even (at least 200) its calibration and serial branch run to completion, so a cancelled task
        // gets a complete result: generic large k over 100 candidates (calibration may add up to 100 probe
        // calls on a cold key), and cosine on Vector768Optimized with k = 3 over 64 candidates.
        let hundred = p01TaggedRows((0..<100).map { Float(99 - $0) })
        calls.reset()
        let largeK = await Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await Operations.findNearest(to: query, in: hundred, k: 100, metric: metric)
        }.result
        #expect((try? largeK.get())?.map(\.index) == Array((0..<100).reversed()), "large k: \(largeK)")
        #expect([100, 200].contains(calls.count), "large k: \(calls.count) metric calls")
        let rows768 = (0..<64).map { _ in Vector768Optimized(generator: { _ in random.unit() }) }
        let cosine = await Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await Operations.findNearest(to: rows768[0], in: rows768, k: 3, metric: CosineDistance())
        }.result
        #expect((try? cosine.get())?.count == 3, "cosine 768: \(cosine)")
        print("P01_ROUTE cancellation large-k=\(Self.describe(largeK)) calls=\(calls.count) cosine768=\(Self.describe(cosine)) "
              + "direct-unchecked=\(Self.describe(directUnchecked)) stream-unchecked=\(Self.describe(uncheckedBefore))")

        // A provider error passes through the stream unchanged, before any metric call.
        calls.reset()
        let failing = P01UncheckedProviderProbe(plan: { [$0] }, failure: P01OpsError(token: 9))
        await #expect(throws: P01OpsError(token: 9)) {
            _ = try await Operations.$computeProvider.withValue(failing) {
                try await Operations.findNearest(to: query, in: rows, k: k, metric: metric)
            }
        }
        #expect(calls.count == 0)
    }

    private static func describe(_ outcome: Result<[NearestNeighborResult], any Error>) -> String {
        switch outcome {
        case .success(let results): return "success(\(results.count))"
        case .failure(let error): return "failure(\(type(of: error)))"
        }
    }

    private static func isCancellation<T>(_ outcome: Result<T, any Error>) -> Bool {
        if case .failure(let error) = outcome { return error is CancellationError }
        return false
    }

    // MARK: Extreme partitions and partial capacities (O8, O9, O14, Q5)

    @Test func extremePartitions() async throws {
        try await e06IsolatedScenario("p01-ops-partitions", filter: "P01OperationsRouteTests/extremePartitions") {
            let count = 41
            func providers() -> [(String, P01RecordingProvider)] {
                var all: [(String, P01RecordingProvider)] = []
                for (name, plan) in P01PlanProvider.plans {
                    for fold in P01PlanProvider.Fold.allCases {
                        all.append(("\(name)/\(fold)", P01RecordingProvider(inner: P01PlanProvider(label: name, plan: plan, fold: fold))))
                    }
                }
                all.append(("default-limit-max", P01RecordingProvider(inner: P01DefaultLimitProvider(limit: .max))))
                all.append(("default-limit-1", P01RecordingProvider(inner: P01DefaultLimitProvider(limit: 1))))
                all.append(("cpu-parallel", P01RecordingProvider(inner: CPUComputeProvider(mode: .parallel))))
                return all
            }
            var combines: [P01OpsLedger.Combine] = []

            // Generic stream (k' < 41 / 10 = 4), with ties across ranges, signed zeros, infinities and NaNs.
            var values = (0..<count).map { Float(($0 * 17) % 11) - 3 }
            values[4] = .nan; values[13] = -0.0; values[22] = 0; values[30] = -.infinity; values[31] = .nan; values[40] = -.infinity
            let rows = p01TaggedRows(values)
            let query = p01TaggedQuery()
            let calls = P01OpsCalls()
            for k in [1, 3] {
                let expected = P01OpsOracle.select(p01SpyScores(query, rows), k: k)
                for (label, provider) in providers() {
                    calls.reset()
                    let events = P01OpsEvents()
                    let result = try await BoundedExecution.$observation.withValue({ events.record($0) }, operation: {
                        try await Operations.$computeProvider.withValue(provider) {
                            try await Operations.findNearest(to: query, in: rows, k: k, metric: P01OpsSpyMetric(calls: calls))
                        }
                    })
                    #expect(p01Entries(result) == expected, "generic \(label) k=\(k)")
                    p01CheckStream(provider.ledger, calls: calls.snapshot(), count: count, k: k, label: "generic \(label) k=\(k)")
                    events.verify()
                    combines += provider.ledger.combines.filter { _ in k == 3 }
                    if label == "default-limit-max" {
                        #expect(events.submitted(phase: "default.reduce") == count && provider.ledger.partials.count == count)
                    }
                    if label == "default-limit-1" { #expect(events.submitted(phase: "default.reduce") == 1) }
                    if label.hasPrefix("singletons") {
                        #expect(provider.ledger.partials.filter { $0.range.isEmpty }.count == count + 1)
                    }
                }
            }
            // Both combine branches ran at k = 3: in place at full capacity, and a new heap for two smaller partials.
            #expect(combines.contains { $0.left == 3 } && combines.contains { $0.left < 3 && $0.out == min(3, $0.left + $0.right) })

            // Direct optimized routes under the same providers: capacities, empty ranges, the combine rule, and
            // results bit-equal to the oracle over each kernel's own per-candidate scores.
            var random = P01OpsRandom(state: 17)
            func rowValues(_ dimension: Int) -> [[Float]] {
                let base = (0..<13).map { _ in (0..<dimension).map { _ in random.unit() } }
                // Rows repeat with period 13, so equal scores occur in different ranges; one NaN and one zero row.
                return (0..<count).map { row in
                    if row == 7 { return [Float](repeating: .nan, count: dimension) }
                    if row == 19 { return [Float](repeating: 0, count: dimension) }
                    return base[row % 13]
                }
            }
            let raw512 = rowValues(512), raw768 = rowValues(768), raw1536 = rowValues(1536)
            let rows512 = try raw512.map { try Vector512Optimized($0) }
            let rows768 = try raw768.map { try Vector768Optimized($0) }
            let rows1536 = try raw1536.map { try Vector1536Optimized($0) }
            let q512 = Vector512Optimized(generator: { _ in random.unit() })
            let q768 = Vector768Optimized(generator: { _ in random.unit() })
            let q1536 = Vector1536Optimized(generator: { _ in random.unit() })
            let routes: [(String, [Float], (Float) -> Float, (Int) async throws -> [NearestNeighborResult])] = [
                ("euclid512", rows512.map { EuclideanKernels.squared512(q512, $0) }, { $0.squareRoot() },
                 { try await Operations.findNearest(to: q512, in: rows512, k: $0) }),
                ("euclid768", rows768.map { EuclideanKernels.squared768(q768, $0) }, { $0.squareRoot() },
                 { try await Operations.findNearest(to: q768, in: rows768, k: $0) }),
                ("euclid1536", rows1536.map { EuclideanKernels.squared1536(q1536, $0) }, { $0.squareRoot() },
                 { try await Operations.findNearest(to: q1536, in: rows1536, k: $0) }),
                ("cosine512", rows512.map { CosineKernels.distance512_fused(q512, $0) }, { $0 },
                 { try await Operations.findNearest(to: q512, in: rows512, k: $0, metric: CosineDistance()) }),
                ("dot512", rows512.map { -DotKernels.dot512(q512, $0) }, { $0 },
                 { try await Operations.findNearest(to: q512, in: rows512, k: $0, metric: DotProductDistance()) })
            ]
            for (name, scores, format, search) in routes {
                for k in [1, 3, 40] {
                    let expected = P01OpsOracle.select(scores, k: k, format: format)
                    for (label, provider) in providers() {
                        let result = try await Operations.$computeProvider.withValue(provider) { try await search(k) }
                        #expect(p01Entries(result) == expected, "\(name) \(label) k=\(k)")
                        p01CheckStream(provider.ledger, calls: nil, count: count, k: k, label: "\(name) \(label) k=\(k)")
                    }
                }
            }
            print("P01_ROUTE partitions providers=\(providers().count) combinesAtK3=\(combines.count) "
                  + "inPlace=\(combines.filter { $0.left == 3 }.count) newHeap=\(combines.filter { $0.left < 3 }.count)")
        }
    }

    // MARK: Custom and parameterized witnesses (CS12, O5, T7)

    @Test func customAndParameterizedWitnesses() async throws {
        try await e06IsolatedScenario("p01-ops-witnesses", filter: "P01OperationsRouteTests/customAndParameterizedWitnesses") {
            // A custom metric on Vector512Optimized is not replaced by a built-in kernel: it streams, called once
            // per candidate in index order within each range.
            let count = 64, k = 3
            var random = P01OpsRandom(state: 23)
            let values = (0..<count).map { _ in Float(Int(random.next() % 9)) }
            let rows = values.enumerated().map { index, value in
                Vector512Optimized(generator: { $0 == 0 ? value : ($0 == 1 ? Float(index) : random.unit()) })
            }
            let query = Vector512Optimized(generator: { $0 == 1 ? -1 : 0 })
            let calls = P01OpsCalls()
            let witnessProviders: [(String, any ComputeProvider)] = [
                ("cpu-parallel", CPUComputeProvider(mode: .parallel)),
                ("chunks-of-7", P01PlanProvider(label: "chunks", plan: P01PlanProvider.plans[3].1, fold: .balanced))
            ]
            for (label, inner) in witnessProviders {
                calls.reset()
                let recording = P01RecordingProvider(inner: inner)
                let result = try await Operations.$computeProvider.withValue(recording) {
                    try await Operations.findNearest(to: query, in: rows, k: k, metric: P01OpsSpyMetric(calls: calls))
                }
                #expect(p01Entries(result) == P01OpsOracle.select(p01SpyScores(query, rows), k: k), "\(label)")
                #expect(recording.ledger.reduceWithoutHint == 1, "\(label)")
                p01CheckStream(recording.ledger, calls: calls.snapshot(), count: count, k: k, label: "custom512 \(label)")
            }

            // A parameterized witness decides the answer: Minkowski p = 3 and Euclidean disagree on this fixture.
            func minkowskiRows<V: VectorProtocol>(_ make: ([Float]) throws -> V, dimension: Int) throws -> [V] where V.Scalar == Float {
                try (0..<30).map { row in
                    var v = [Float](repeating: 0, count: dimension)
                    switch row {
                    case 2: v[0] = 4                 // L2 = 4, L3 = 4
                    case 5: v[0] = 3; v[1] = 3       // L2 = 4.243, L3 = 3.780
                    default: v[0] = 10; v[1] = 10 + Float(row)
                    }
                    return try make(v)
                }
            }
            let minkowski = MinkowskiDistance(p: 3)
            func checkMinkowski<V: VectorProtocol>(_ rows: [V], _ query: V, _ label: String) async throws where V.Scalar == Float {
                let p3 = try await Operations.findNearest(to: query, in: rows, k: 1, metric: minkowski)
                let p2 = try await Operations.findNearest(to: query, in: rows, k: 1, metric: EuclideanDistance())
                #expect(p3.map(\.index) == [5] && p2.map(\.index) == [2], "\(label): \(p3) \(p2)")
                #expect(p3.first?.distance.bitPattern == minkowski.distance(query, rows[5]).bitPattern, "\(label)")
                let reference = pow(54.0, 1.0 / 3.0)
                #expect(abs(Double(p3[0].distance) - reference) <= 1e-5, "\(label): \(p3[0].distance) vs \(reference)")
            }
            try await checkMinkowski(try minkowskiRows({ DynamicVector($0) }, dimension: 4), DynamicVector([0, 0, 0, 0]), "dynamic")
            try await checkMinkowski(try minkowskiRows({ try Vector768Optimized($0) }, dimension: 768),
                                     try Vector768Optimized([Float](repeating: 0, count: 768)), "v768")

            // Built-in witnesses without an optimized route stream with their own protocol-witness scores (computed
            // here through `p01Witness`, as Operations' generic code calls them): Euclidean on Vector384Optimized,
            // Manhattan and Chebyshev on DynamicVector, dot product on Vector1536Optimized.
            let raw384 = (0..<200).map { _ in (0..<384).map { _ in random.unit() } }
            let rows384 = try raw384.map { try Vector384Optimized($0) }
            let q384 = Vector384Optimized(generator: { _ in random.unit() })
            let euclid384 = try await Operations.findNearest(to: q384, in: rows384, k: 5)
            #expect(p01Entries(euclid384) == P01OpsOracle.select(rows384.map { p01Witness(EuclideanDistance(), q384, $0) }, k: 5))
            let qArray = q384.toArray()
            for entry in euclid384 {
                let reference = zip(qArray, raw384[entry.index]).reduce(0.0) { $0 + pow(Double($1.0) - Double($1.1), 2) }.squareRoot()
                #expect(Double(entry.distance).isFinite && abs(Double(entry.distance) - reference) <= 1e-3)
            }
            let dynamicRows = raw384.map { DynamicVector(Array($0.prefix(16))) }
            let dynamicQuery = DynamicVector(Array(qArray.prefix(16)))
            for (label, witness) in [("manhattan", ManhattanDistance() as any DistanceMetric<Float>),
                                     ("chebyshev", ChebyshevDistance() as any DistanceMetric<Float>)] {
                let result = try await Self.search(dynamicQuery, dynamicRows, k: 4, witness)
                #expect(p01Entries(result) == P01OpsOracle.select(dynamicRows.map { witness.distance(dynamicQuery, $0) }, k: 4), "\(label)")
            }
            let rows1536 = (0..<40).map { _ in Vector1536Optimized(generator: { _ in random.unit() }) }
            let q1536 = Vector1536Optimized(generator: { _ in random.unit() })
            let dot1536 = try await Operations.findNearest(to: q1536, in: rows1536, k: 2, metric: DotProductDistance())
            #expect(p01Entries(dot1536) == P01OpsOracle.select(rows1536.map { p01Witness(DotProductDistance(), q1536, $0) }, k: 2))
            print("P01_ROUTE witnesses custom512 minkowski dynamic+v768 euclid384 manhattan chebyshev dot1536")
        }
    }

    /// Opens an existential witness so `findNearest` sees the concrete metric type.
    private static func search<M: DistanceMetric>(_ query: DynamicVector, _ rows: [DynamicVector], k: Int, _ metric: M)
        async throws -> [NearestNeighborResult] where M.Scalar == Float {
        try await Operations.findNearest(to: query, in: rows, k: k, metric: metric)
    }
}
