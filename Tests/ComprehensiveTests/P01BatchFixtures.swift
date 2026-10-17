import Foundation
import Testing
@testable import VectorCore

// P01 phase F fixtures for `BatchOperations.findNearest`: the six optimized routes as typed fixtures, deterministic
// finite, root-tie and exceptional rows, and recorders and checkers for the small-k route's observations
// (`BoundedExecution.observation` events and the `BatchOperations.smallKTrace` seam). Expectations are computed
// in test code from the stated formulas; production helpers are called only as the subject or as the retained
// untiled reference (`computeDistances_*_parallel`).

// MARK: - The six optimized routes

struct P01BatchCase: Sendable, CustomStringConvertible {
    enum Kind: String, Sendable, CaseIterable { case euclid, cosine }
    let dim: Int
    let kind: Kind
    var description: String { "\(kind.rawValue)\(dim)" }
    /// The accepted parallel worker phase of this route.
    var phase: String { "batch.\(kind == .euclid ? "euclid" : "cosine_fused")\(dim)" }
    /// The accepted serial helpers' step (`minChunk(forDim:)`): 512 for 1536, otherwise 256.
    var serialStep: Int { dim == 1536 ? 512 : 256 }
    static let all: [P01BatchCase] = [512, 768, 1536].flatMap { dim in Kind.allCases.map { P01BatchCase(dim: dim, kind: $0) } }
    static func cases(dim: Int) -> [P01BatchCase] { Kind.allCases.map { P01BatchCase(dim: dim, kind: $0) } }
}

/// Kernel entry points for one optimized vector type.
struct P01KernelBundle<O: Sendable>: Sendable {
    let euclidParallel: @Sendable (O, [O], Int, Int?, Int) async -> [Float]
    let cosineParallel: @Sendable (O, [O], Int, Int?, Int) async -> [Float]
    let euclid2: @Sendable (O, [O], Range<Int>, UnsafeMutableBufferPointer<Float>) -> Void
    let euclid: @Sendable (O, [O], Range<Int>, UnsafeMutableBufferPointer<Float>) -> Void
    let cosine: @Sendable (O, [O], Range<Int>, UnsafeMutableBufferPointer<Float>) -> Void
}

let p01Bundle512 = P01KernelBundle<Vector512Optimized>(
    euclidParallel: {
        await BatchOperations.computeDistances_euclid_optimized_parallel(query: $0, candidates: $1, dim: $2, chunkSize: $3, cores: $4)
    },
    cosineParallel: {
        await BatchOperations.computeDistances_cosine_fused_parallel(query: $0, candidates: $1, dim: $2, chunkSize: $3, cores: $4)
    },
    euclid2: { BatchKernels.range_euclid2_512(query: $0, candidates: $1, range: $2, out: $3) },
    euclid: { BatchKernels.range_euclid_512(query: $0, candidates: $1, range: $2, out: $3) },
    cosine: { BatchKernels.range_cosine_fused_512(query: $0, candidates: $1, range: $2, out: $3) })

let p01Bundle768 = P01KernelBundle<Vector768Optimized>(
    euclidParallel: {
        await BatchOperations.computeDistances_euclid_optimized_parallel(query: $0, candidates: $1, dim: $2, chunkSize: $3, cores: $4)
    },
    cosineParallel: {
        await BatchOperations.computeDistances_cosine_fused_parallel(query: $0, candidates: $1, dim: $2, chunkSize: $3, cores: $4)
    },
    euclid2: { BatchKernels.range_euclid2_768(query: $0, candidates: $1, range: $2, out: $3) },
    euclid: { BatchKernels.range_euclid_768(query: $0, candidates: $1, range: $2, out: $3) },
    cosine: { BatchKernels.range_cosine_fused_768(query: $0, candidates: $1, range: $2, out: $3) })

let p01Bundle1536 = P01KernelBundle<Vector1536Optimized>(
    euclidParallel: {
        await BatchOperations.computeDistances_euclid_optimized_parallel(query: $0, candidates: $1, dim: $2, chunkSize: $3, cores: $4)
    },
    cosineParallel: {
        await BatchOperations.computeDistances_cosine_fused_parallel(query: $0, candidates: $1, dim: $2, chunkSize: $3, cores: $4)
    },
    euclid2: { BatchKernels.range_euclid2_1536(query: $0, candidates: $1, range: $2, out: $3) },
    euclid: { BatchKernels.range_euclid_1536(query: $0, candidates: $1, range: $2, out: $3) },
    cosine: { BatchKernels.range_cosine_fused_1536(query: $0, candidates: $1, range: $2, out: $3) })

/// One fixture materialized as the case's optimized type.
struct P01BatchTyped: Sendable {
    let count: Int
    /// The production small-k route for this query, these rows and the case's metric.
    let route: BatchOperations.SmallKRoute
    /// The retained untiled helper `computeDistances_*_parallel(chunkSize:cores:)`; nil chunk size = its default.
    let untiled: @Sendable (Int?, Int) async -> [Float]
    /// `range_euclid2_D` (squared) over each range in turn, concatenated.
    let squared: @Sendable ([Range<Int>]) -> [Float]
    /// The case's score helper (`range_euclid_D` or `range_cosine_fused_D`) over each range in turn, concatenated.
    let scores: @Sendable ([Range<Int>]) -> [Float]
    /// Public `BatchOperations.findNearest` with the case's metric.
    let nearest: @Sendable (Int) async -> [(index: Int, distance: Float)]
}

private func p01RunRanges<O>(_ kernel: (O, [O], Range<Int>, UnsafeMutableBufferPointer<Float>) -> Void,
                             _ query: O, _ rows: [O], _ ranges: [Range<Int>]) -> [Float] {
    var output: [Float] = []
    for range in ranges {
        var part = [Float](repeating: -7, count: range.count)
        part.withUnsafeMutableBufferPointer { kernel(query, rows, range, $0) }
        output += part
    }
    return output
}

private func p01Typed<O: VectorProtocol & Sendable>(
    _ c: P01BatchCase, _ bundle: P01KernelBundle<O>, query: [Float], rows: [[Float]], make: ([Float]) throws -> O
) rethrows -> P01BatchTyped where O.Scalar == Float {
    let q = try make(query)
    let r = try rows.map(make)
    let dim = c.dim
    let euclid = c.kind == .euclid
    let route = euclid ? BatchOperations.smallKRoute(query: q, vectors: r, metric: EuclideanDistance())
        : BatchOperations.smallKRoute(query: q, vectors: r, metric: CosineDistance())
    return P01BatchTyped(
        count: rows.count,
        route: route,
        untiled: { cs, cores in
            euclid ? await bundle.euclidParallel(q, r, dim, cs, cores) : await bundle.cosineParallel(q, r, dim, cs, cores)
        },
        squared: { p01RunRanges(bundle.euclid2, q, r, $0) },
        scores: { p01RunRanges(euclid ? bundle.euclid : bundle.cosine, q, r, $0) },
        nearest: { k in
            euclid ? await BatchOperations.findNearest(to: q, in: r, k: k, metric: EuclideanDistance())
                : await BatchOperations.findNearest(to: q, in: r, k: k, metric: CosineDistance())
        })
}

func p01BatchTyped(_ c: P01BatchCase, query: [Float], rows: [[Float]]) throws -> P01BatchTyped {
    switch c.dim {
    case 512: return try p01Typed(c, p01Bundle512, query: query, rows: rows) { try Vector512Optimized($0) }
    case 768: return try p01Typed(c, p01Bundle768, query: query, rows: rows) { try Vector768Optimized($0) }
    default: return try p01Typed(c, p01Bundle1536, query: query, rows: rows) { try Vector1536Optimized($0) }
    }
}

// MARK: - Fixtures

enum P01BatchFixture: String, Sendable, CaseIterable {
    /// Uniform [-1, 1) coordinates (SplitMix64).
    case finite
    /// Query 0; rows alternate b, a and a random row, scaled by 2^(row/3 mod 5). Each (b, a) pair has distinct
    /// kernel squared distances and equal kernel roots (`p01RootTiePair`; checked again on the built rows).
    case rootTie
    /// NaN (two payloads), +inf, -inf, zero and -0 rows, overflowing squares, subnormals (2^-130 scale) and rows equal to the
    /// query, mixed with finite rows.
    case exceptional
}

struct P01BatchRows: Sendable {
    let query: [Float]
    let rows: [[Float]]
    /// rootTie only: the (b, a) index pairs that tie on the root with distinct squares, b first.
    let tiePairs: [(Int, Int)]
}

/// Deterministic search through the real kernels (query 0): `a = [1, 0, ...]`, `b = [1, j·2^-14, 0, ...]` for
/// j = 1, 2, ...; the first `b` whose squared distance differs from `a`'s while the root is bit-equal.
/// Returns (b, a): `b` has the larger squared distance.
func p01RootTiePair(_ c: P01BatchCase) throws -> (b: [Float], a: [Float]) {
    var a = [Float](repeating: 0, count: c.dim)
    a[0] = 1
    for j in 1...4_096 {
        var b = a
        b[1] = Float(j) * 0x1p-14
        let probe = try p01BatchTyped(P01BatchCase(dim: c.dim, kind: .euclid), query: [Float](repeating: 0, count: c.dim), rows: [b, a])
        let squared = probe.squared([0..<2])
        let roots = probe.scores([0..<2])
        if squared[0].bitPattern != squared[1].bitPattern, roots[0].bitPattern == roots[1].bitPattern, squared[1] < squared[0] {
            return (b, a)
        }
    }
    throw NSError(domain: "P01.rootTieSearch", code: c.dim)
}

func p01BatchRows(_ fixture: P01BatchFixture, case c: P01BatchCase, count: Int, seed: UInt64 = 0xF06) throws -> P01BatchRows {
    var random = P01OpsRandom(state: seed &+ UInt64(c.dim))
    func randomRow() -> [Float] { (0..<c.dim).map { _ in random.unit() } }
    switch fixture {
    case .finite:
        let query = randomRow()
        return P01BatchRows(query: query, rows: (0..<count).map { _ in randomRow() }, tiePairs: [])
    case .rootTie:
        let (b, a) = try p01RootTiePair(c)
        var rows: [[Float]] = []
        var pairs: [(Int, Int)] = []
        for index in 0..<count {
            let scale = Float(1 << ((index / 3) % 5))
            switch index % 3 {
            case 0: rows.append(b.map { $0 * scale })
            case 1: rows.append(a.map { $0 * scale }); pairs.append((index - 1, index))
            default: rows.append(randomRow().map { $0 * 4 })
            }
        }
        return P01BatchRows(query: [Float](repeating: 0, count: c.dim), rows: rows, tiePairs: pairs)
    case .exceptional:
        let query = randomRow()
        let rows: [[Float]] = (0..<count).map { p01ExceptionalRow($0, random: randomRow(), query: query) }
        return P01BatchRows(query: query, rows: rows, tiePairs: [])
    }
}

/// Row `index` of the exceptional fixture: one of 13 patterns applied to a random row.
private func p01ExceptionalRow(_ index: Int, random: [Float], query: [Float]) -> [Float] {
    var row = random
    switch index % 13 {
    case 0: row[0] = .nan
    case 1: row[3] = .infinity
    case 2: row[5] = -.infinity
    case 3: row = [Float](repeating: 0, count: row.count)
    case 4: row = [Float](repeating: -0.0, count: row.count)
    case 5: row[7] = 3e38
    case 6: row[1] = Float(bitPattern: 0x7FC0_1234)
    case 7: row[0] = -0.0
    case 8: row = row.map { $0 * 0x1p-130 }
    case 9: row = query
    case 10: row[2] = .infinity; row[4] = -.infinity
    default: break
    }
    return row
}

// MARK: - Score comparison

/// Bit-exact comparison of the seam's (index, score) stream with a complete per-candidate reference: the indices
/// must be exactly `0..<reference.count` in increasing order and every score's bit pattern (NaN payload and zero
/// sign included) must equal the reference's. Returns readable mismatches (empty when equal).
func p01ScoreMismatches(_ seam: [(index: Int, score: Float)], reference: [Float]) -> [String] {
    var problems: [String] = []
    if seam.map(\.index) != Array(reference.indices) {
        problems.append("index stream differs: count \(seam.count) vs \(reference.count), first=\(seam.prefix(4).map(\.index))")
    }
    for (index, score) in seam where reference.indices.contains(index) {
        if score.bitPattern != reference[index].bitPattern {
            let tiled = String(score.bitPattern, radix: 16), untiled = String(reference[index].bitPattern, radix: 16)
            problems.append("i=\(index) tiled=0x\(tiled) untiled=0x\(untiled)")
        }
    }
    return problems
}

func p01BatchEntries(_ results: [(index: Int, distance: Float)]) -> [P01OpsEntry] {
    results.map { P01OpsEntry(index: $0.index, bits: $0.distance.bitPattern) }
}

// MARK: - Small-k trace seam

final class P01BatchTrace: @unchecked Sendable {
    struct Partial: Equatable, Sendable {
        let partition: Int
        let candidates: Range<Int>
        let heapCapacity: Int
        let scratchCapacity: Int
        let retained: Int
    }
    struct Combine: Equatable, Sendable {
        let left: Int
        let right: Int
        let result: Int
    }
    private let lock = NSLock()
    private var events: [BatchOperations.SmallKTraceEvent] = []
    func record(_ event: BatchOperations.SmallKTraceEvent) { lock.lock(); defer { lock.unlock() }; events.append(event) }
    func snapshot() -> [BatchOperations.SmallKTraceEvent] { lock.lock(); defer { lock.unlock() }; return events }
    var partials: [Partial] {
        snapshot().compactMap { event in
            guard case let .partial(partition, candidates, heapCapacity, scratchCapacity, retained) = event else { return nil }
            return Partial(partition: partition, candidates: candidates, heapCapacity: heapCapacity,
                           scratchCapacity: scratchCapacity, retained: retained)
        }.sorted { $0.partition < $1.partition }
    }
    var combines: [Combine] {
        snapshot().compactMap { event in
            guard case let .combine(left, right, result) = event else { return nil }
            return Combine(left: left, right: right, result: result)
        }
    }
}

/// One observed public call: its result, its `BoundedExecution` events and its small-k trace.
struct P01Observed: Sendable {
    let result: [(index: Int, distance: Float)]
    let events: P01OpsEvents
    let trace: P01BatchTrace
}

/// Runs `body` with both observers installed, in a task that is cancelled before the call when `cancelled`.
func p01ObservedNearest(cancelled: Bool,
                        _ body: @Sendable @escaping () async -> [(index: Int, distance: Float)]) async -> P01Observed {
    let events = P01OpsEvents()
    let trace = P01BatchTrace()
    let result = await Task {
        if cancelled { withUnsafeCurrentTask { $0?.cancel() } }
        return await BoundedExecution.$observation.withValue({ events.record($0) }, operation: {
            await BatchOperations.$smallKTrace.withValue({ trace.record($0) }, operation: { await body() })
        })
    }.value
    return P01Observed(result: result, events: events, trace: trace)
}

/// How a small-k call is expected to run.
enum P01SmallKMode: Equatable, Sendable {
    case serial
    case parallel(chunkSize: Int)
    /// Parallel settings inside a library region: same partitions, no submitted tasks.
    case nested(chunkSize: Int)
}

/// What one small-k call should look like: `count` candidates, the caller's `k`, the expected mode and the route's
/// worker phase, whether it tiles (the six optimized routes) and its serial step.
struct P01Expect: Sendable {
    let count: Int
    let k: Int
    let mode: P01SmallKMode
    let phase: String
    let tiled: Bool
    let serialStep: Int

    static func generic(count: Int, k: Int, mode: P01SmallKMode) -> P01Expect {
        P01Expect(count: count, k: k, mode: mode, phase: "batch.nearest", tiled: false, serialStep: .max)
    }

    static func optimized(_ c: P01BatchCase, count: Int, k: Int, mode: P01SmallKMode) -> P01Expect {
        P01Expect(count: count, k: k, mode: mode, phase: c.phase, tiled: true, serialStep: c.serialStep)
    }
}

/// Checks one observed small-k call: the plan arithmetic against independent formulas, every partial heap's
/// candidates and capacities (B16), the combine rule (Q5 mirror), the per-chunk logical ranges (B12) and the
/// worker accounting (B13).
func p01CheckSmallK(_ expect: P01Expect, _ observed: P01Observed, label: String,
                    sourceLocation: SourceLocation = #_sourceLocation) {
    let n = expect.count, mode = expect.mode, phase = expect.phase, tiled = expect.tiled, serialStep = expect.serialStep
    let events = observed.events, trace = observed.trace
    let cores = BoundedExecution.hardwareLimit
    let actualK = min(expect.k, n)
    // Independent plan formulas: serial = one partition over steps of `serialStep`; parallel = completeMap grouping.
    let chunk: Int
    let partitionWidth: Int
    switch mode {
    case .serial: chunk = min(serialStep, n)
    case .parallel(let size), .nested(let size): chunk = size
    }
    let chunkCount = (n + chunk - 1) / chunk
    if case .serial = mode { partitionWidth = chunkCount } else {
        let groups = min(chunkCount, cores)
        partitionWidth = (chunkCount + groups - 1) / groups
    }
    let partitionCount = (chunkCount + partitionWidth - 1) / partitionWidth
    let plan = mode == .serial ? BatchOperations.SmallKPlan(count: n, chunkSize: serialStep, cores: 1)
        : BatchOperations.SmallKPlan(count: n, chunkSize: chunk, cores: cores)
    #expect(plan.chunkSize == chunk && plan.chunkCount == chunkCount && plan.width == partitionWidth
            && plan.partitionCount == partitionCount, "\(label): plan \(plan)", sourceLocation: sourceLocation)
    if mode != .serial {
        #expect(partitionCount <= min(chunkCount, cores), "\(label): P=\(partitionCount)", sourceLocation: sourceLocation)
    }

    // Partial heaps: one per partition, contiguous candidates, capacities bounded by actual work.
    let partials = trace.partials
    #expect(partials.map(\.partition) == Array(0..<partitionCount), "\(label): partitions \(partials.map(\.partition))",
            sourceLocation: sourceLocation)
    var expectedStart = 0
    for partial in partials {
        let last = min((partial.partition + 1) * partitionWidth, chunkCount)
        let end = min(last * chunk, n)
        #expect(partial.candidates == expectedStart..<end, "\(label): p\(partial.partition) \(partial.candidates)",
                sourceLocation: sourceLocation)
        let assigned = partial.candidates.count
        #expect(partial.heapCapacity == min(actualK, assigned) && partial.retained == partial.heapCapacity,
                "\(label): p\(partial.partition) heap \(partial)", sourceLocation: sourceLocation)
        let tile = tiled ? min(256, chunk, assigned) : 0
        #expect(partial.scratchCapacity == tile && partial.scratchCapacity <= 256,
                "\(label): p\(partial.partition) tile \(partial.scratchCapacity) expected \(tile)", sourceLocation: sourceLocation)
        expectedStart = end
    }
    #expect(expectedStart == n, "\(label): partitions cover \(expectedStart) of \(n)", sourceLocation: sourceLocation)
    #expect(partials.map(\.heapCapacity).reduce(0, +) <= n, sourceLocation: sourceLocation)

    // Combines replay the rule: in place at capacity k, else a new heap of min(k, left + right).
    let combines = trace.combines
    #expect(combines.count == max(0, partials.count - 1), "\(label): combines \(combines.count)", sourceLocation: sourceLocation)
    var accumulated = partials.first?.heapCapacity ?? 0
    for (offset, combine) in combines.enumerated() where offset + 1 < partials.count {
        let right = partials[offset + 1].heapCapacity
        let expected = accumulated == actualK ? actualK : min(actualK, accumulated + right)
        #expect(combine == P01BatchTrace.Combine(left: accumulated, right: right, result: expected),
                "\(label): combine \(offset) \(combine)", sourceLocation: sourceLocation)
        accumulated = expected
    }
    #expect(accumulated == actualK, "\(label): final capacity \(accumulated)", sourceLocation: sourceLocation)

    // Logical ranges (one per chunk, by chunk index) and worker events.
    let snapshot = events.snapshot()
    let logical = snapshot.filter { $0.phase == phase + ".logical" }.compactMap { event -> (Int, Range<Int>)? in
        if case .logicalRange(let range) = event.kind { return (event.worker, range) }
        return nil
    }.sorted { $0.0 < $1.0 }
    let submitted = events.submitted(phase: phase)
    switch mode {
    case .serial:
        #expect(logical.isEmpty && snapshot.allSatisfy { $0.kind != .submitted }, "\(label): serial events",
                sourceLocation: sourceLocation)
    case .parallel, .nested:
        let expectedRanges = (0..<chunkCount).map { ($0 * chunk)..<min(($0 + 1) * chunk, n) }
        #expect(logical.map(\.0) == Array(0..<chunkCount), "\(label): chunk indices", sourceLocation: sourceLocation)
        #expect(logical.map(\.1) == expectedRanges, "\(label): logical ranges", sourceLocation: sourceLocation)
        let expectedSubmitted = mode == .nested(chunkSize: chunk) ? 0 : partitionCount
        #expect(submitted == expectedSubmitted, "\(label): submitted \(submitted) expected \(expectedSubmitted)",
                sourceLocation: sourceLocation)
        if case .nested = mode {
            #expect(snapshot.allSatisfy { $0.kind != .submitted }, "\(label): nested submitted", sourceLocation: sourceLocation)
        }
    }
    events.verify(maximum: cores, sourceLocation: sourceLocation)
}
