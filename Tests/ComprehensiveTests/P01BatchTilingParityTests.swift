import Foundation
import Testing
@testable import VectorCore

// P01 phase F, B10 gate: complete per-candidate scores that the small-k optimized routes feed their heaps (the
// `BatchOperations.smallKScores` seam, which runs the production `scorePartition` / `scoreRange` tiling) are
// bit-equal to the retained untiled helpers `computeDistances_*_parallel(chunkSize:cores:)` over the same original
// ranges. These tests run in-process (no child), so the same bodies run under AddressSanitizer.
//
// Score parity cannot see tile placement for ANY input, Euclidean or cosine: each `BatchKernels` pair loop and its tail
// kernel execute the same operation sequence (Euclidean: `EuclideanKernels.squared`, same lanes, order and reduction,
// and the SIMD4 and scalar square roots are both IEEE correctly rounded; cosine since T14 phase 3:
// `CosineKernels.fusedPair`, the 2-way blocked form of the tail kernel `CosineKernels.fused`). `recordDetectionPower`
// pins that fact (single-versus-pair differences must be 0). The tile log of `tiledScoresMatchUntiledHelpers` and the
// scratch-capacity checks of `P01BatchRouteTests` carry the placement coverage; phase F mutations MF1 and MF2 (and the
// review's RM1 and RM4b) leave score parity green and are caught by those checks.

@Suite("P01 BatchOperations tiling parity (phase F)")
struct P01BatchTilingParityTests {
    static let chunkSizes = [1, 3, 7, 255, 256, 257, 301, 512, 513]
    /// 1100 and 1101 give every chunk size above except 1 a last chunk of each parity.
    static let counts = [1_100, 1_101]

    /// B10: every candidate, every chunk size, odd and even tails, serial steps, three fixture classes.
    @Test(arguments: P01BatchCase.all)
    func tiledScoresMatchUntiledHelpers(_ c: P01BatchCase) async throws {
        let cores = BoundedExecution.hardwareLimit
        var compared = 0
        for fixture in P01BatchFixture.allCases {
            for count in Self.counts {
                let built = try p01BatchRows(fixture, case: c, count: count)
                let typed = try p01BatchTyped(c, query: built.query, rows: built.rows)
                #expect(typed.route.phase == c.phase && typed.route.serialStep == c.serialStep, "\(c) route")
                guard case .kernel(let kernel) = typed.route.scorer else {
                    Issue.record("\(c): optimized route without a kernel scorer"); continue
                }
                if fixture == .rootTie { try checkRootTiePreconditions(c, built, typed) }
                if fixture == .finite { await recordDetectionPower(c, typed) }
                var layouts: [(String, BatchOperations.SmallKPlan, Int)] = Self.chunkSizes.map {
                    ("cs\($0)", BatchOperations.SmallKPlan(count: count, chunkSize: $0, cores: cores), $0)
                }
                layouts.append(("serial", .serial(route: typed.route, count: count), c.serialStep))
                for (name, plan, referenceChunk) in layouts {
                    let tiles = P01TileLog()
                    let recording = BatchOperations.SmallKRoute(phase: typed.route.phase, serialStep: typed.route.serialStep,
                        scorer: .kernel { range, out in tiles.append(range, out.count); kernel(range, out) })
                    let seam = BatchOperations.smallKScores(route: recording, plan: plan)
                    // The untiled reference for the serial plan is the same helper over the serial helper's steps.
                    let reference = await typed.untiled(referenceChunk, cores)
                    let mismatches = p01ScoreMismatches(seam, reference: reference)
                    #expect(mismatches.isEmpty, "\(c) \(fixture) n=\(count) \(name): \(mismatches.prefix(6))")
                    #expect(tiles.snapshot() == expectedTiles(count: count, chunk: min(referenceChunk, count)),
                            "\(c) \(fixture) n=\(count) \(name): tiles")
                    compared += seam.count
                    print("P01F_PARITY case=\(c) fixture=\(fixture) n=\(count) layout=\(name) " +
                          "compared=\(seam.count) mismatches=\(mismatches.count)")
                }
            }
        }
        print("P01F_PARITY_TOTAL case=\(c) compared=\(compared)")
    }

    /// Independent tile list: each chunk `[c·chunk, min((c+1)·chunk, n))` in steps of 256 from its own lower bound.
    private func expectedTiles(count: Int, chunk: Int) -> [P01TileLog.Tile] {
        var tiles: [P01TileLog.Tile] = []
        var lower = 0
        while lower < count {
            let upper = min(lower + chunk, count)
            var start = lower
            while start < upper {
                let end = min(start + 256, upper)
                tiles.append(P01TileLog.Tile(range: start..<end, outCount: end - start))
                start = end
            }
            lower = upper
        }
        return tiles
    }

    /// Fixture precondition: every designed (b, a) pair has distinct kernel squared distances and bit-equal roots,
    /// with b (the smaller index) holding the larger squared distance.
    private func checkRootTiePreconditions(_ c: P01BatchCase, _ built: P01BatchRows, _ typed: P01BatchTyped) throws {
        let euclid = try p01BatchTyped(P01BatchCase(dim: c.dim, kind: .euclid), query: built.query, rows: built.rows)
        let squared = euclid.squared([0..<built.rows.count])
        let roots = euclid.scores([0..<built.rows.count])
        #expect(!built.tiePairs.isEmpty)
        for (b, a) in built.tiePairs {
            #expect(squared[b] > squared[a] && roots[b].bitPattern == roots[a].bitPattern,
                    "\(c) root-tie precondition at (\(b), \(a)): sq \(squared[b]) \(squared[a]) root \(roots[b]) \(roots[a])")
        }
    }

    /// Detection power, recorded: how many candidates' single-candidate (tail-kernel) scores differ in bits from their
    /// paired untiled scores. Euclidean and cosine must have none: their pair and tail kernels execute the same
    /// operations, so parity is blind to tile placement by construction (see the file header); a nonzero count means a
    /// kernel changed and that argument must be re-evaluated (T14 phase 3 made the cosine count 0).
    private func recordDetectionPower(_ c: P01BatchCase, _ typed: P01BatchTyped) async {
        let n = min(typed.count, 600)
        let untiled = await typed.untiled(n, 1)
        let singles = typed.scores((0..<n).map { $0..<($0 + 1) })
        let differing = (0..<n).filter { untiled[$0].bitPattern != singles[$0].bitPattern }.count
        print("P01F_DETECTION case=\(c) singleVsPairDiffer=\(differing)/\(n)")
        #expect(differing == 0, "\(c): pair and tail kernels no longer agree")
    }

    /// The pure plan mirrors `BoundedExecution.completeMap`'s grouping: the items each spawned task runs, observed
    /// by task identity, are exactly the plan's partitions, and the submitted count is the plan's partition count.
    @Test func planMirrorsCompleteMap() async {
        for chunkCount in [1, 2, 3, 5, 15, 16, 17, 31, 33, 100, 257] {
            for cores in [1, 2, 3, 7, 16, BoundedExecution.hardwareLimit, 64] {
                let plan = BatchOperations.SmallKPlan(count: chunkCount, chunkSize: 1, cores: cores)
                let events = P01OpsEvents()
                let owners = P01TaskOwners()
                _ = await BoundedExecution.$observation.withValue({ events.record($0) }, operation: {
                    await BoundedExecution.completeMap(0..<chunkCount, limit: cores, phase: "p01.plan.probe") { item in
                        withUnsafeCurrentTask { owners.record(item: item, task: $0?.hashValue ?? 0) }
                        return item
                    }
                })
                let groups = owners.groups()
                let expected = (0..<plan.partitionCount).map { plan.chunks(ofPartition: $0) }
                #expect(groups == expected, "L=\(chunkCount) C=\(cores): tasks ran \(groups), plan \(expected)")
                #expect(events.submitted(phase: "p01.plan.probe") == plan.partitionCount, "L=\(chunkCount) C=\(cores)")
                #expect(plan.partitionCount <= min(chunkCount, cores))
                events.verify()
            }
        }
    }

    /// Plan arithmetic at representable limits, without any per-chunk storage: partitions are contiguous, cover
    /// `0..<count`, start at multiples of the width, and number at most `min(L, C)`.
    @Test func planArithmeticAtRepresentableLimits() {
        for count in [1, 2, 17, 1_100, Int.max - 1, Int.max] {
            for chunkSize in [1, 3, 256, 513, Int.max] {
                for cores in [1, 2, 16, Int.max] {
                    let plan = BatchOperations.SmallKPlan(count: count, chunkSize: chunkSize, cores: cores)
                    let chunk = min(chunkSize, count)
                    let chunks = count / chunk + (count % chunk == 0 ? 0 : 1)
                    let groups = min(chunks, cores)
                    let width = chunks / groups + (chunks % groups == 0 ? 0 : 1)
                    let partitions = chunks / width + (chunks % width == 0 ? 0 : 1)
                    let label = "n=\(count) cs=\(chunkSize) C=\(cores)"
                    #expect(plan.chunkSize == chunk && plan.chunkCount == chunks && plan.width == width
                            && plan.partitionCount == partitions, "\(label): \(plan)")
                    #expect(partitions <= min(chunks, cores), "\(label)")
                    let samples = Set([0, 1, partitions / 2, partitions - 2, partitions - 1].filter { $0 >= 0 && $0 < partitions })
                    for p in samples.sorted() {
                        let range = plan.chunks(ofPartition: p)
                        #expect(range.lowerBound == p * width && !range.isEmpty, "\(label) p=\(p): \(range)")
                        let candidates = plan.candidates(ofPartition: p)
                        #expect(candidates.lowerBound == range.lowerBound * chunk, "\(label) p=\(p)")
                        if p + 1 < partitions {
                            #expect(candidates.upperBound == plan.candidates(ofPartition: p + 1).lowerBound, "\(label) p=\(p)")
                        }
                    }
                    #expect(plan.candidates(ofPartition: 0).lowerBound == 0, "\(label)")
                    #expect(plan.candidates(ofPartition: partitions - 1).upperBound == count, "\(label)")
                    #expect(plan.candidates(ofChunk: chunks - 1).upperBound == count, "\(label)")
                }
            }
        }
    }

    /// `scoreRange` with a recording kernel: tiles restart at the range's own lower bound (not a global multiple of
    /// 256), never cross its upper bound, write through a rebased buffer, and feed every value with its global index
    /// in order. The metric scorer calls once per candidate, in index order.
    @Test func scoreRangeTilesAndMetricOrder() {
        for range in [0..<0, 0..<1, 5..<6, 301..<602, 301..<1_100, 256..<513, 1_000..<1_257] {
            let tiles = P01TileLog()
            let route = BatchOperations.SmallKRoute(phase: "probe", serialStep: 256, scorer: .kernel { tile, out in
                tiles.append(tile, out.count)
                for offset in 0..<tile.count { out[offset] = Float(tile.lowerBound + offset) + 0.5 }
            })
            var scratch = [Float](repeating: .nan, count: min(256, range.count))
            var fed: [(Int, Float)] = []
            scratch.withUnsafeMutableBufferPointer { buffer in
                BatchOperations.scoreRange(range, route: route, scratch: buffer) { fed.append(($1, $0)) }
            }
            var expected: [P01TileLog.Tile] = []
            var start = range.lowerBound
            while start < range.upperBound {
                let end = min(start + 256, range.upperBound)
                expected.append(P01TileLog.Tile(range: start..<end, outCount: end - start))
                start = end
            }
            #expect(tiles.snapshot() == expected, "\(range)")
            #expect(fed.map(\.0) == Array(range), "\(range)")
            #expect(fed.allSatisfy { $0.1 == Float($0.0) + 0.5 }, "\(range)")
        }
        let calls = P01OpsCalls()
        let metricRoute = BatchOperations.SmallKRoute(phase: "probe", serialStep: .max, scorer: .metric { index in
            _ = calls.append(index); return Float(index)
        })
        var fed: [Int] = []
        BatchOperations.scoreRange(17..<41, route: metricRoute, scratch: UnsafeMutableBufferPointer(start: nil, count: 0)) {
            fed.append($1)
        }
        #expect(calls.snapshot() == Array(17..<41) && fed == Array(17..<41))
    }
}

/// Kernel calls of a recording route: each tile's range and the buffer length it received.
final class P01TileLog: @unchecked Sendable {
    struct Tile: Equatable, Sendable, CustomStringConvertible {
        let range: Range<Int>
        let outCount: Int
        var description: String { "\(range.lowerBound)..<\(range.upperBound)/\(outCount)" }
    }
    private let lock = NSLock()
    private var tiles: [Tile] = []
    func append(_ range: Range<Int>, _ outCount: Int) {
        lock.lock(); defer { lock.unlock() }; tiles.append(Tile(range: range, outCount: outCount))
    }
    func snapshot() -> [Tile] { lock.lock(); defer { lock.unlock() }; return tiles }
}

/// Which task ran each item, grouped into maximal per-task item ranges in item order.
final class P01TaskOwners: @unchecked Sendable {
    private let lock = NSLock()
    private var owners: [Int: Int] = [:]
    func record(item: Int, task: Int) { lock.lock(); defer { lock.unlock() }; owners[item] = task }
    func groups() -> [Range<Int>] {
        lock.lock(); defer { lock.unlock() }
        var result: [Range<Int>] = []
        var items = owners.keys.sorted()[...]
        while let first = items.first {
            let task = owners[first]
            var end = first + 1
            while let next = items.dropFirst(end - first).first, next == end, owners[next] == task { end += 1 }
            result.append(first..<end)
            items = items.dropFirst(end - first)
        }
        // Each task's items must be one contiguous block: a task appearing in two groups is reported as-is.
        return result
    }
}
