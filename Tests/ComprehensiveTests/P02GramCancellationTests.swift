import Foundation
import Testing
@testable import VectorCore

// P02 phase C: `.gram` cancellation (brief "Throwing search cancellation"; phase 1 check map X1-X6, X12-X14).
// The checks run after the plan and limit and before any scratch or result allocation, at each query-tile and
// candidate-tile boundary, after each tile's scores, before each result row and before returning. The seam's events
// are the observable work between checks; the trace closure cancels the running task itself (withUnsafeCurrentTask),
// after real work and before publication. No substitute kernel is used. Invalid, unsupported and infeasible requests
// in an already-cancelled task are in `P02GramRouteTests.validationPrecedence` and `batchValidationPrecedesStrategyChecks`.

/// Records the events and cancels the current task when event number `cancelAt` (0-based) arrives.
final class P02CancellingTrace: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [Operations.GramTraceEvent] = []
    let cancelAt: Int?
    init(cancelAt: Int?) { self.cancelAt = cancelAt }
    func observe(_ event: Operations.GramTraceEvent) {
        lock.lock()
        events.append(event)
        let fire = events.count - 1 == cancelAt
        lock.unlock()
        if fire { withUnsafeCurrentTask { $0?.cancel() } }
    }
    var labels: [String] { lock.lock(); defer { lock.unlock() }; return events.map(p02EventLabel) }
}

@Suite("P02 .gram cancellation")
struct P02GramCancellationTests {

    /// Runs `.gram` in a new task under `trace`; the task is cancelled before the search when `preCancelled`.
    private func run(_ queries: [Vector512Optimized], _ candidates: [Vector512Optimized], euclidean: Bool,
                     strategy: BatchSearchCPUStrategy, trace: P02CancellingTrace,
                     preCancelled: Bool = false) async -> Result<[[NearestNeighborResult]], any Error> {
        let task = Task {
            if preCancelled { withUnsafeCurrentTask { $0?.cancel() } }
            return try await Operations.$gramTrace.withValue({ trace.observe($0) }, operation: {
                try await p02Batch(queries, candidates, k: 3, euclidean: euclidean, strategy: strategy)
            })
        }
        return await task.result
    }

    /// X1, X12: a valid plan in an already-cancelled task throws `CancellationError` before any scratch, pack, BLAS
    /// call, heap or result: the seam sees no event at all. (The same request uncancelled does trace work.)
    @Test func preCancelledValidPlanDoesNoWork() async throws {
        let all = try p02Vectors(p02RandomRows(40, width: 512, seed: 21), as: Vector512Optimized.self)
        let queries = Array(all.prefix(5))
        for euclidean in [true, false] {
            for (r, c) in [(2, 4), (1, 1), (Int.max, Int.max)] {
                let trace = P02CancellingTrace(cancelAt: nil)
                let result = await run(queries, all, euclidean: euclidean, strategy: p02Gram(r, c), trace: trace,
                                       preCancelled: true)
                guard case .failure(let error) = result else { Issue.record("r\(r) c\(c): returned a result"); continue }
                #expect(error is CancellationError, "r\(r) c\(c): \(error)")
                #expect(trace.labels.isEmpty, "r\(r) c\(c): \(trace.labels)")
                let control = P02CancellingTrace(cancelAt: nil)
                _ = try await run(queries, all, euclidean: euclidean, strategy: p02Gram(r, c), trace: control).get()
                #expect(control.labels.prefix(2) == ["scratch", "heaps"])
            }
        }
    }

    /// X2-X6, X13: for EVERY event of an uncancelled run (scratch, each query and candidate pack, each tile, each
    /// selection and each result row, including the first tile, the last candidate tile of the first query tile, the
    /// final tile and the final row), cancelling the task when that event arrives makes the search throw
    /// `CancellationError` with no result, and no later work happens: the events end at the cancelling one, except
    /// that the scratch allocation is followed by the heap allocation, and a candidate pack by its own tile (each pair
    /// runs between the same two checks).
    @Test func cancellationAtEveryEventStopsAtTheNextCheckpoint() async throws {
        let shapes: [(q: Int, n: Int, r: Int, c: Int)] = [(5, 9, 2, 4), (2, 3, 1, 1), (3, 6, Int.max, Int.max)]
        for shape in shapes {
            let all = try p02Vectors(p02RandomRows(shape.n, width: 512, seed: UInt64(shape.n)), as: Vector512Optimized.self)
            let queries = try p02Vectors(p02RandomRows(shape.q, width: 512, seed: 77), as: Vector512Optimized.self)
            for euclidean in [true, false] {
                let baseline = P02CancellingTrace(cancelAt: nil)
                _ = try await run(queries, all, euclidean: euclidean, strategy: p02Gram(shape.r, shape.c), trace: baseline).get()
                let full = baseline.labels
                #expect(full.first == "scratch" && full.last == "row(\(shape.q - 1))", "\(full)")
                for index in full.indices {
                    let label = "Q\(shape.q) N\(shape.n) r\(shape.r) c\(shape.c) \(euclidean) cancel at \(full[index])"
                    let trace = P02CancellingTrace(cancelAt: index)
                    let result = await run(queries, all, euclidean: euclidean, strategy: p02Gram(shape.r, shape.c),
                                           trace: trace)
                    guard case .failure(let error) = result else { Issue.record("\(label): returned a result"); continue }
                    #expect(error is CancellationError, "\(label): \(error)")
                    let stop = full[index].hasPrefix("candidatePack") || full[index] == "scratch" ? index + 1 : index
                    #expect(trace.labels == Array(full[...stop]), "\(label): \(trace.labels.suffix(from: min(index, trace.labels.count)))")
                }
            }
        }
    }
}
