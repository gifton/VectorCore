import Foundation
import Testing
@testable import VectorCore

// P01 phase E shared fixtures for `P01OperationsSelectionTests` and `P01OperationsRouteTests`.
//
// Providers are witnesses, not assumptions about any GPU run. Conformance to the documented `ComputeProvider`
// obligations, per fixture:
// - `P01RecordingProvider` delegates every requirement to the provider it wraps (so it conforms when that
//   provider does) and only observes the ranges, partial heaps and combines Operations hands it.
// - `P01PlanProvider` conforms: explicit disjoint ranges covering the input (empty ranges allowed), callbacks
//   and combines inside the library callback region with cancellation checked before and after each one, range
//   and `minChunk` validation, and a bracketing that keeps range order and uses `initial` once.
// - `P01DefaultLimitProvider` conforms through the protocol's default scheduling.
// - `P01UncheckedProviderProbe` deliberately does NOT conform: it never checks cancellation and can fail with
//   its own error. It exists to show what Operations does when a provider skips those obligations.
// - `P01KernelSpy` / `P01KernelBatchSpy` are BatchKernelProviders whose replies deliberately break the
//   protocol's documented result shape (unsorted, out of range, NaN payloads) to show verbatim passthrough.

// MARK: - Independent oracle

/// Best-first order used by every expectation: numeric scores ascending, NaNs after every number, exact ties
/// (including +0/-0 and two NaNs) by the smaller index. It uses no production comparator, heap or selector.
enum P01OpsOracle {
    static func before(_ a: (index: Int, value: Float), _ b: (index: Int, value: Float)) -> Bool {
        if a.value.isNaN != b.value.isNaN { return !a.value.isNaN }
        if !a.value.isNaN && a.value != b.value { return a.value < b.value }
        return a.index < b.index
    }

    /// The best `k` of `scores` (index = position), with the selected score bits.
    static func select(_ scores: [Float], k: Int) -> [P01OpsEntry] {
        let order = scores.indices.sorted { before(($0, scores[$0]), ($1, scores[$1])) }
        return order.prefix(max(0, k)).map { P01OpsEntry(index: $0, bits: scores[$0].bitPattern) }
    }

    /// Membership by `scores`, then each selected score mapped by `format` and the entries reordered by the formatted
    /// value (the direct Euclidean route selects squared distances and orders the returned roots).
    static func select(_ scores: [Float], k: Int, format: (Float) -> Float) -> [P01OpsEntry] {
        let members = select(scores, k: k).map { (index: $0.index, value: format(Float(bitPattern: $0.bits))) }
        return members.sorted { before($0, $1) }.map { P01OpsEntry(index: $0.index, bits: $0.value.bitPattern) }
    }
}

/// One result entry compared by index and exact score bits (no tolerance; NaN payload and zero sign included).
struct P01OpsEntry: Equatable, CustomStringConvertible, Sendable {
    let index: Int
    let bits: UInt32
    var description: String { "\(index):\(Float(bitPattern: bits))[0x\(String(bits, radix: 16))]" }
}

func p01Entries(_ results: [NearestNeighborResult]) -> [P01OpsEntry] {
    results.map { P01OpsEntry(index: $0.index, bits: $0.distance.bitPattern) }
}

// MARK: - Spies

/// Lock-protected call log with an O(1) count (the metric reads the count on every call).
final class P01OpsCalls: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Int] = []
    func append(_ value: Int) -> Int { lock.lock(); defer { lock.unlock() }; values.append(value); return values.count }
    func reset() { lock.lock(); defer { lock.unlock() }; values.removeAll() }
    func snapshot() -> [Int] { lock.lock(); defer { lock.unlock() }; return values }
    var count: Int { lock.lock(); defer { lock.unlock() }; return values.count }
}

/// Custom metric witness. Score = b[0] - a[0], so with a[0] == 0 the score is b[0] itself (the subtraction is
/// exact; the oracle repeats the same expression, so any NaN payload rule applies to both). Records b[1], the
/// candidate's index tag. `token` identifies the instance in provider passthrough checks; `cancelAfter` cancels
/// the current task once that many calls have been made.
struct P01OpsSpyMetric: DistanceMetric {
    typealias Scalar = Float
    let calls: P01OpsCalls
    var token: Int = 0
    var cancelAfter: Int?
    var name: String { "P01 ops spy \(token)" }
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float {
        let made = calls.append(Int(b[1]))
        if let cancelAfter, made == cancelAfter { withUnsafeCurrentTask { $0?.cancel() } }
        return b[0] - a[0]
    }
}

struct P01OpsError: Error, Equatable, Sendable {
    let token: Int
}

/// Rows tagged for `P01OpsSpyMetric`: row i is [values[i], i, 0...] with `dimension` coordinates.
func p01TaggedRows(_ values: [Float], dimension: Int = 2) -> [DynamicVector] {
    values.enumerated().map { index, value in
        var row = [Float](repeating: 0, count: dimension)
        row[0] = value
        row[1] = Float(index)
        return DynamicVector(row)
    }
}

func p01TaggedQuery(dimension: Int = 2) -> DynamicVector {
    var row = [Float](repeating: 0, count: dimension)
    row[1] = -1
    return DynamicVector(row)
}

/// The spy metric's scores for tagged rows and a tagged query, computed independently of Operations.
func p01SpyScores<V: VectorProtocol>(_ query: V, _ rows: [V]) -> [Float] where V.Scalar == Float {
    rows.map { $0[0] - query[0] }
}

/// The metric's score through its `DistanceMetric` requirement, which is what Operations' generic code calls.
/// For Euclidean, dot product, Manhattan and Chebyshev on the optimized types a direct call with concrete types binds
/// this witness too (T14 phase 2 removed those concrete overloads), and concrete, generic and existential calls agree
/// (`MetricWitnessEquivalenceTests`). P01 phase E run E1's dot1536 pair, 0xc1ce604c through Operations and 0xc1ce604e
/// through the removed overload, is replayed by `MetricWitnessEquivalenceTests.p01Dot1536Row18Replay`. T14 phase 3
/// removed the concrete cosine overloads too, so a direct call with concrete optimized types binds this witness for
/// every built-in metric.
func p01Witness<M: DistanceMetric, V: VectorProtocol>(_ metric: M, _ a: V, _ b: V) -> Float
    where M.Scalar == Float, V.Scalar == Float {
    metric.distance(a, b)
}

/// Deterministic generator (SplitMix64) for fixtures.
struct P01OpsRandom {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    /// Uniform in [-1, 1).
    mutating func unit() -> Float { Float(Double(next() >> 11) / Double(1 << 53) * 2 - 1) }
}

// MARK: - Provider ledger

/// What Operations handed a provider: entries per overload, every range given to a range callback, the
/// capacity and size of every `TopKBuffer` partial and every `TopKBuffer` combine.
final class P01OpsLedger: @unchecked Sendable {
    struct Partial: Sendable, CustomStringConvertible {
        let range: Range<Int>
        let capacity: Int
        let size: Int
        var description: String { "\(range)->k\(capacity)/s\(size)" }
    }
    struct Combine: Sendable, CustomStringConvertible {
        let left: Int
        let right: Int
        let out: Int
        let outSize: Int
        var description: String { "k\(left)+k\(right)->k\(out)/s\(outSize)" }
    }

    private let lock = NSLock()
    private var noHint = 0
    private var hints: [Int] = []
    private var executes = 0
    private var allRanges: [Range<Int>] = []
    private var heapPartials: [Partial] = []
    private var heapCombines: [Combine] = []

    func noteReduce(hint: Int?) {
        lock.lock(); defer { lock.unlock() }
        if let hint { hints.append(hint) } else { noHint += 1 }
    }
    func noteExecute() { lock.lock(); defer { lock.unlock() }; executes += 1 }
    func notePartial<R>(_ range: Range<Int>, _ part: R) {
        lock.lock(); defer { lock.unlock() }
        allRanges.append(range)
        if let heap = part as? TopKBuffer { heapPartials.append(Partial(range: range, capacity: heap.k, size: heap.size)) }
    }
    func noteCombine<R>(_ a: R, _ b: R, _ out: R) {
        guard let a = a as? TopKBuffer, let b = b as? TopKBuffer, let out = out as? TopKBuffer else { return }
        lock.lock(); defer { lock.unlock() }
        heapCombines.append(Combine(left: a.k, right: b.k, out: out.k, outSize: out.size))
    }
    func reset() {
        lock.lock(); defer { lock.unlock() }
        noHint = 0; hints = []; executes = 0; allRanges = []; heapPartials = []; heapCombines = []
    }

    var reduceWithoutHint: Int { lock.lock(); defer { lock.unlock() }; return noHint }
    var reduceHints: [Int] { lock.lock(); defer { lock.unlock() }; return hints }
    var executeEntries: Int { lock.lock(); defer { lock.unlock() }; return executes }
    var ranges: [Range<Int>] { lock.lock(); defer { lock.unlock() }; return allRanges.sorted { $0.lowerBound < $1.lowerBound } }
    var partials: [Partial] { lock.lock(); defer { lock.unlock() }; return heapPartials }
    var combines: [Combine] { lock.lock(); defer { lock.unlock() }; return heapCombines }
    var isUntouched: Bool {
        lock.lock(); defer { lock.unlock() }
        return noHint == 0 && hints.isEmpty && executes == 0 && allRanges.isEmpty
    }
}

/// Delegates every requirement to `inner` (so its scheduling, cancellation and nesting policies apply) and
/// records what Operations passed through the two reduction overloads and `parallelExecute`.
struct P01RecordingProvider: ComputeProvider {
    let inner: any ComputeProvider
    let ledger = P01OpsLedger()

    var device: ComputeDevice { inner.device }
    var maxConcurrency: Int { inner.maxConcurrency }
    var deviceInfo: ComputeDeviceInfo { inner.deviceInfo }

    func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T {
        try await inner.execute(work)
    }
    func parallelExecute<T: Sendable>(items: Range<Int>, _ work: @Sendable @escaping (Int) async throws -> T) async throws -> [T] {
        ledger.noteExecute()
        return try await inner.parallelExecute(items: items, work)
    }
    func parallelForEach(items: Range<Int>, _ body: @Sendable @escaping (Int) async throws -> Void) async throws {
        try await inner.parallelForEach(items: items, body)
    }
    func parallelForEach(items: Range<Int>, minChunk: Int, _ body: @Sendable @escaping (Int) async throws -> Void) async throws {
        try await inner.parallelForEach(items: items, minChunk: minChunk, body)
    }
    func parallelReduce<R: Sendable>(items: Range<Int>, initial: R,
                                     _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
                                     _ combine: @Sendable @escaping (R, R) -> R) async throws -> R {
        ledger.noteReduce(hint: nil)
        let ledger = self.ledger
        return try await inner.parallelReduce(items: items, initial: initial, { range in
            let part = try await rangeWork(range)
            ledger.notePartial(range, part)
            return part
        }, { a, b in
            let out = combine(a, b)
            ledger.noteCombine(a, b, out)
            return out
        })
    }
    func parallelReduce<R: Sendable>(items: Range<Int>, initial: R, minChunk: Int,
                                     _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
                                     _ combine: @Sendable @escaping (R, R) -> R) async throws -> R {
        ledger.noteReduce(hint: minChunk)
        let ledger = self.ledger
        return try await inner.parallelReduce(items: items, initial: initial, minChunk: minChunk, { range in
            let part = try await rangeWork(range)
            ledger.notePartial(range, part)
            return part
        }, { a, b in
            let out = combine(a, b)
            ledger.noteCombine(a, b, out)
            return out
        })
    }
}

/// A conforming custom provider with explicit partitions. `plan` returns disjoint ranges covering `items` in
/// order (empty ranges allowed); the range callbacks run in that order and the partials combine with `fold`,
/// a bracketing that keeps range order and uses `initial` exactly once. Like the protocol defaults, it rejects an
/// unrepresentable range length and (hinted overload) a nonpositive `minChunk`, returns `initial` for empty
/// `items` without callbacks, runs callbacks and combines inside the library callback region and checks
/// cancellation before and after each of them. `minChunk` is otherwise a hint it does not apply, because the plan
/// fixes the partition. Every other requirement uses the protocol defaults.
struct P01PlanProvider: ComputeProvider {
    enum Fold: String, Sendable, CaseIterable {
        case left       // ((initial + p0) + p1) + ...
        case balanced   // adjacent pairs, level by level
        case right      // initial + (p0 + (p1 + ...))
    }

    let label: String
    let plan: @Sendable (Range<Int>) -> [Range<Int>]
    let fold: Fold

    var device: ComputeDevice { .cpu }
    var maxConcurrency: Int { 1 }
    var deviceInfo: ComputeDeviceInfo {
        ComputeDeviceInfo(name: "p01-plan-\(label)", availableMemory: nil, maxThreads: 1, preferredChunkSize: 1)
    }

    func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T {
        try Task.checkCancellation()
        let result = try await BoundedExecution.$isInsideRegion.withValue(true) { try await work() }
        try Task.checkCancellation()
        return result
    }

    func parallelReduce<R: Sendable>(items: Range<Int>, initial: R,
                                     _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
                                     _ combine: @Sendable @escaping (R, R) -> R) async throws -> R {
        let count = try ComputeRangeBoundary.count(items)
        try Task.checkCancellation()
        guard count > 0 else {
            try Task.checkCancellation()
            return initial
        }
        var parts: [R] = []
        for range in plan(items) {
            try Task.checkCancellation()
            let part = try await BoundedExecution.$isInsideRegion.withValue(true) { try await rangeWork(range) }
            try Task.checkCancellation()
            parts.append(part)
        }
        return try BoundedExecution.$isInsideRegion.withValue(true) {
            let combined = try Self.combineAll(initial, parts, fold) { a, b in
                try Task.checkCancellation()
                let out = combine(a, b)
                try Task.checkCancellation()
                return out
            }
            try Task.checkCancellation()
            return combined
        }
    }
    func parallelReduce<R: Sendable>(items: Range<Int>, initial: R, minChunk: Int,
                                     _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
                                     _ combine: @Sendable @escaping (R, R) -> R) async throws -> R {
        _ = try ComputeRangeBoundary.count(items)
        try ComputeRangeBoundary.validate(minChunk: minChunk)
        return try await parallelReduce(items: items, initial: initial, rangeWork, combine)
    }

    static func combineAll<R>(_ initial: R, _ parts: [R], _ fold: Fold, _ combine: (R, R) throws -> R) rethrows -> R {
        switch fold {
        case .left:
            return try parts.reduce(initial, combine)
        case .balanced:
            var level = [initial] + parts
            while level.count > 1 {
                var next: [R] = []
                var index = 0
                while index < level.count {
                    next.append(index + 1 < level.count ? try combine(level[index], level[index + 1]) : level[index])
                    index += 2
                }
                level = next
            }
            return level[0]
        case .right:
            let all = [initial] + parts
            var accumulated = all[all.count - 1]
            for element in all.dropLast().reversed() { accumulated = try combine(element, accumulated) }
            return accumulated
        }
    }

    /// Partition plans over `0..<n`-style ranges, each a disjoint cover in order.
    static let plans: [(String, @Sendable (Range<Int>) -> [Range<Int>])] = [
        ("whole", { [$0] }),
        ("singletons-with-empties", { items in
            var ranges = [items.lowerBound..<items.lowerBound]
            for index in items { ranges.append(index..<(index + 1)); ranges.append((index + 1)..<(index + 1)) }
            return ranges
        }),
        ("uneven-with-empties", { items in
            let lo = items.lowerBound, hi = items.upperBound, n = items.count
            guard n >= 6 else { return [lo..<lo, items, hi..<hi] }
            let a = lo + 1, b = lo + n / 3, c = hi - 1
            return [lo..<lo, lo..<a, a..<a, a..<b, b..<b, b..<c, c..<hi, hi..<hi]
        }),
        ("chunks-of-7", { items in
            stride(from: items.lowerBound, to: items.upperBound, by: 7).map { $0..<min($0 + 7, items.upperBound) }
        })
    ]
}

/// Uses only the protocol's default scheduling (`BoundedExecution` with `maxConcurrency` workers and a minimum
/// chunk of 1), so `limit: .max` gives one range per candidate.
struct P01DefaultLimitProvider: ComputeProvider {
    let limit: Int
    var device: ComputeDevice { .cpu }
    var maxConcurrency: Int { limit }
    var deviceInfo: ComputeDeviceInfo {
        ComputeDeviceInfo(name: "p01-default-\(limit)", availableMemory: nil, maxThreads: 1, preferredChunkSize: 1)
    }
    func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T {
        try Task.checkCancellation()
        let result = try await work()
        try Task.checkCancellation()
        return result
    }
}

/// NOT a conforming provider (see the header): both reductions run every range of `plan` and combine left to
/// right without any cancellation check, or throw `failure` (a provider-side error) before any callback.
struct P01UncheckedProviderProbe: ComputeProvider {
    let plan: @Sendable (Range<Int>) -> [Range<Int>]
    var failure: P01OpsError?

    var device: ComputeDevice { .cpu }
    var maxConcurrency: Int { 1 }
    var deviceInfo: ComputeDeviceInfo {
        ComputeDeviceInfo(name: "p01-unchecked-probe", availableMemory: nil, maxThreads: 1, preferredChunkSize: 1)
    }
    func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T { try await work() }

    func parallelReduce<R: Sendable>(items: Range<Int>, initial: R,
                                     _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
                                     _ combine: @Sendable @escaping (R, R) -> R) async throws -> R {
        if let failure { throw failure }
        var result = initial
        for range in plan(items) { result = combine(result, try await rangeWork(range)) }
        return result
    }
    func parallelReduce<R: Sendable>(items: Range<Int>, initial: R, minChunk: Int,
                                     _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
                                     _ combine: @Sendable @escaping (R, R) -> R) async throws -> R {
        try await parallelReduce(items: items, initial: initial, rangeWork, combine)
    }
}

// MARK: - BatchKernelProvider spies

final class P01KernelLog: @unchecked Sendable {
    struct Call: Sendable, Equatable {
        let queries: Int
        let candidates: Int
        let k: Int
        let token: Int?
    }
    private let lock = NSLock()
    private var single: [Call] = []
    private var batch: [Call] = []
    private var distances = 0
    func noteSingle(_ call: Call) { lock.lock(); defer { lock.unlock() }; single.append(call) }
    func noteBatch(_ call: Call) { lock.lock(); defer { lock.unlock() }; batch.append(call) }
    func noteDistance() { lock.lock(); defer { lock.unlock() }; distances += 1 }
    var singleCalls: [Call] { lock.lock(); defer { lock.unlock() }; return single }
    var batchCalls: [Call] { lock.lock(); defer { lock.unlock() }; return batch }
    var isUntouched: Bool { lock.lock(); defer { lock.unlock() }; return single.isEmpty && batch.isEmpty && distances == 0 }
}

/// A BatchKernelProvider without a `findNearestBatch` override (the protocol default loops `findNearest`).
/// `reply` is returned verbatim; it deliberately breaks the documented result shape (unsorted, out of range,
/// NaN, -0), and `findNearest` performs no cancellation check, so Operations' passthrough is what is observed.
struct P01KernelSpy: BatchKernelProvider {
    let log: P01KernelLog
    var reply: [(index: Int, distance: Float)]
    var failure: P01OpsError?

    var device: ComputeDevice { .cpu }
    var maxConcurrency: Int { 2 }
    var deviceInfo: ComputeDeviceInfo {
        ComputeDeviceInfo(name: "p01-kernel-spy", availableMemory: nil, maxThreads: 2, preferredChunkSize: 1)
    }
    func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T {
        try Task.checkCancellation()
        let result = try await work()
        try Task.checkCancellation()
        return result
    }
    func batchDistance<V: VectorProtocol>(query: V, candidates: [V], metric: any DistanceMetric)
        async throws -> [Float] where V.Scalar == Float {
        log.noteDistance()
        return []
    }
    func findNearest<V: VectorProtocol>(query: V, candidates: [V], k: Int, metric: any DistanceMetric)
        async throws -> [(index: Int, distance: Float)] where V.Scalar == Float {
        log.noteSingle(.init(queries: 1, candidates: candidates.count, k: k, token: (metric as? P01OpsSpyMetric)?.token))
        if let failure { throw failure }
        return reply
    }
}

/// A BatchKernelProvider that overrides `findNearestBatch`; `batchReply[i]` answers query i verbatim. Its replies
/// deliberately break the documented result shape, as for `P01KernelSpy`.
struct P01KernelBatchSpy: BatchKernelProvider {
    let log: P01KernelLog
    var reply: [(index: Int, distance: Float)]
    var batchReply: [[(index: Int, distance: Float)]]
    var failure: P01OpsError?

    var device: ComputeDevice { .cpu }
    var maxConcurrency: Int { 2 }
    var deviceInfo: ComputeDeviceInfo {
        ComputeDeviceInfo(name: "p01-kernel-batch-spy", availableMemory: nil, maxThreads: 2, preferredChunkSize: 1)
    }
    func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T {
        try Task.checkCancellation()
        let result = try await work()
        try Task.checkCancellation()
        return result
    }
    func batchDistance<V: VectorProtocol>(query: V, candidates: [V], metric: any DistanceMetric)
        async throws -> [Float] where V.Scalar == Float {
        log.noteDistance()
        return []
    }
    func findNearest<V: VectorProtocol>(query: V, candidates: [V], k: Int, metric: any DistanceMetric)
        async throws -> [(index: Int, distance: Float)] where V.Scalar == Float {
        log.noteSingle(.init(queries: 1, candidates: candidates.count, k: k, token: (metric as? P01OpsSpyMetric)?.token))
        if let failure { throw failure }
        return reply
    }
    func findNearestBatch<V: VectorProtocol>(queries: [V], candidates: [V], k: Int, metric: any DistanceMetric)
        async throws -> [[(index: Int, distance: Float)]] where V.Scalar == Float {
        log.noteBatch(.init(queries: queries.count, candidates: candidates.count, k: k,
                            token: (metric as? P01OpsSpyMetric)?.token))
        if let failure { throw failure }
        return batchReply
    }
}

// MARK: - Scheduling events

final class P01OpsEvents: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [BoundedExecution.Event] = []
    func record(_ event: BoundedExecution.Event) { lock.lock(); defer { lock.unlock() }; events.append(event) }
    func snapshot() -> [BoundedExecution.Event] { lock.lock(); defer { lock.unlock() }; return events }
    func submitted(phase: String) -> Int { snapshot().filter { $0.phase == phase && $0.kind == .submitted }.count }

    /// Each scheduling phase begins and ends once, submits at most its limit (or `maximum`) workers and completes
    /// exactly the workers it submitted.
    func verify(maximum: Int? = nil, sourceLocation: SourceLocation = #_sourceLocation) {
        for (_, group) in Dictionary(grouping: snapshot(), by: \.id) {
            let submitted = group.filter { $0.kind == .submitted }.map(\.worker)
            let completed = group.filter { $0.kind == .completed }.map(\.worker)
            #expect(group.first?.kind == .begin && group.last?.kind == .end, sourceLocation: sourceLocation)
            #expect(submitted.count <= (maximum ?? group[0].limit), sourceLocation: sourceLocation)
            #expect(submitted.sorted() == completed.sorted(), sourceLocation: sourceLocation)
            #expect(Set(submitted).count == submitted.count, sourceLocation: sourceLocation)
        }
    }
}

// MARK: - Error identity

/// Kind, context information and throwing function of a VectorError (the timestamp and line are not identity).
struct P01ErrorIdentity: Equatable, CustomStringConvertible {
    let kind: VectorError.ErrorKind
    let info: [String: String]
    let function: String
    var description: String { "\(kind) \(info) in '\(function)'" }

    init(_ error: VectorError) {
        kind = error.kind
        info = error.context.additionalInfo
        function = "\(error.context.function)"
    }
    init(expected: VectorError, function: String) {
        kind = expected.kind
        info = expected.context.additionalInfo
        #if DEBUG
        self.function = function
        #else
        self.function = ""
        #endif
    }
}

/// Runs `body` and returns its VectorError identity, or records an issue.
func p01ErrorIdentity(sourceLocation: SourceLocation = #_sourceLocation,
                      _ body: () async throws -> Void) async -> P01ErrorIdentity? {
    do {
        try await body()
        Issue.record("expected a VectorError, but the call returned", sourceLocation: sourceLocation)
    } catch let error as VectorError {
        return P01ErrorIdentity(error)
    } catch {
        Issue.record("expected a VectorError, got \(error)", sourceLocation: sourceLocation)
    }
    return nil
}

// MARK: - Stream law

/// Checks one search's per-range heaps against the phase E law: every partial has capacity
/// `min(k, range.count)` and holds that many entries (an empty range: capacity 0, nothing held), the non-empty
/// ranges are a disjoint cover of `0..<count`, the capacities sum to at most `count`, and every combine follows
/// rule Q5 (`left == k` merges in place at capacity k, otherwise the new heap has capacity `min(k, left + right)`).
/// With `calls`, the metric ran once per candidate, in increasing index order within each range.
func p01CheckStream(_ ledger: P01OpsLedger, calls: [Int]?, count: Int, k: Int, label: String,
                    sourceLocation: SourceLocation = #_sourceLocation) {
    let partials = ledger.partials
    #expect(!partials.isEmpty, "\(label): no partial heaps were seen", sourceLocation: sourceLocation)
    for partial in partials {
        #expect(partial.capacity == min(k, partial.range.count) && partial.size == partial.capacity,
                "\(label): partial \(partial)", sourceLocation: sourceLocation)
    }
    let ranges = partials.map(\.range).filter { !$0.isEmpty }.sorted { $0.lowerBound < $1.lowerBound }
    let covers = ranges.first?.lowerBound == 0 && ranges.last?.upperBound == count
        && zip(ranges, ranges.dropFirst()).allSatisfy { $0.upperBound == $1.lowerBound }
    #expect(covers, "\(label): ranges \(ranges) do not tile 0..<\(count)", sourceLocation: sourceLocation)
    let capacitySum = partials.reduce(0) { $0 + $1.capacity }
    #expect(capacitySum <= count, "\(label): capacities sum to \(capacitySum) > \(count)", sourceLocation: sourceLocation)
    for combine in ledger.combines {
        let expected = combine.left == k ? k : min(k, combine.left + combine.right)
        #expect(combine.left <= k && combine.right <= k && combine.out == expected,
                "\(label): combine \(combine) (k \(k))", sourceLocation: sourceLocation)
    }
    guard let calls else { return }
    #expect(calls.count == count && Set(calls) == Set(0..<count),
            "\(label): \(calls.count) metric calls for \(count) candidates", sourceLocation: sourceLocation)
    for range in ranges {
        #expect(calls.filter { range.contains($0) } == Array(range), "\(label): call order in \(range)",
                sourceLocation: sourceLocation)
    }
}
