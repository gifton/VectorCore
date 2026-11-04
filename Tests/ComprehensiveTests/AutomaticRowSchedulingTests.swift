import Foundation
import Testing
@testable import VectorCore

private final class RowScheduleEvents: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [BoundedExecution.Event] = []
    func record(_ event: BoundedExecution.Event) {
        lock.lock(); defer { lock.unlock() }; events.append(event)
    }
    func verify(phase: String, maximum: Int, multiple: Bool = false, nested: Bool = false) {
        lock.lock(); let captured = events; lock.unlock()
        let submitted = captured.filter { $0.kind == .submitted }
        #expect(captured.first?.kind == .begin && captured.last?.kind == .end)
        #expect(captured.allSatisfy { $0.phase == phase })
        #expect(submitted.count <= maximum)
        #expect(submitted.count == captured.filter { $0.kind == .completed }.count)
        if multiple { #expect(submitted.count > 1) }
        if nested { #expect(submitted.isEmpty) }
        print("AB3 phase=\(phase) submitted=\(submitted.count) maximum=\(maximum)")
    }
}

private struct RowScheduleMetric: DistanceMetric {
    let name = "Row scheduling index distance"
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float {
        abs(a[0] - b[0])
    }
}

private struct RowScheduleContext: ExecutionContext {
    let maxThreadCount: Int
    let preferredChunkSize: Int
    var device: ComputeDevice { .cpu }
    func execute<T: Sendable>(_ work: @Sendable @escaping () throws -> T) async throws -> T { try work() }
}

@Suite(.serialized) struct AutomaticRowSchedulingTests {
    // Reverting the Float-element to row conversion must fail the multiple-submission assertions
    // on this multicore host; the observations come from the real CPUContext.automatic route.
    @Test func automaticWideRowsSubmitBoundedWorkers() async throws {
        for count in [1000, 2000, 4096] {
            let rows = (0..<count).map { index in
                var scalars = [Float](repeating: 1, count: 768)
                scalars[0] = Float(index)
                return DynamicVector(scalars)
            }
            print("AB3 rows=\(count) cores=\(CPUContext.automatic.maxThreadCount)")
            try await verifyOperations(rows, context: CPUContext.automatic,
                                       multiple: CPUContext.automatic.maxThreadCount > 1)
        }
    }

    @Test func singleCoreAndNestedWideRowsPreserveSerialControls() async throws {
        let rows = (0..<1000).map { DynamicVector([Float($0)] + [Float](repeating: 1, count: 767)) }
        try await verifyOperations(rows, context: CPUContext.sequential)
        try await BoundedExecution.$isInsideRegion.withValue(true) {
            try await verifyOperations(rows, context: CPUContext.automatic, nested: true)
        }
        try await CPUComputeProvider.$isInsideParallelRegion.withValue(true) {
            try await verifyOperations(rows, context: CPUContext.automatic, nested: true)
        }
    }

    private func verifyOperations(_ rows: [DynamicVector], context: any ExecutionContext,
                                  multiple: Bool = false, nested: Bool = false) async throws {
        let mapEvents = RowScheduleEvents()
        let mapped = try await BoundedExecution.$observation.withValue({ mapEvents.record($0) }, operation: {
            try await ExecutionOperations.map(rows, transform: { $0 * 2 + 1 }, context: context)
        })
        #expect(mapped.map { $0.toArray() } == rows.map { $0.toArray().map { $0 * 2 + 1 } })
        mapEvents.verify(phase: "transform.map", maximum: context.maxThreadCount, multiple: multiple, nested: nested)

        let normalizeEvents = RowScheduleEvents()
        let normalized = try await BoundedExecution.$observation.withValue({ normalizeEvents.record($0) }, operation: {
            try await ExecutionOperations.normalize(rows, context: context)
        })
        #expect(normalized.count == rows.count)
        for (index, row) in normalized.enumerated() {
            let denominator = sqrt(Double(index * index + 767))
            #expect(abs(Double(row[0]) - Double(index) / denominator) < 0.000001)
            #expect(row.toArray().dropFirst().allSatisfy { abs(Double($0) - 1 / denominator) < 0.000001 })
        }
        normalizeEvents.verify(phase: "transform.normalize", maximum: context.maxThreadCount, multiple: multiple, nested: nested)

        let matrixEvents = RowScheduleEvents()
        let matrix = try await BoundedExecution.$observation.withValue({ matrixEvents.record($0) }, operation: {
            try await ExecutionOperations.distanceMatrix(rows, metric: RowScheduleMetric(), context: context)
        })
        #expect(matrix.count == rows.count)
        for (index, row) in matrix.enumerated() {
            #expect(row.count == rows.count)
            #expect(row[0] == Float(index) && row[index] == 0 && row[rows.count - 1] == Float(rows.count - 1 - index))
        }
        matrixEvents.verify(phase: "matrix.distance", maximum: context.maxThreadCount, multiple: multiple, nested: nested)
    }

    @Test func emptyZeroWidthRaggedAndExtremeHints() async throws {
        for hint in [1, 4096, Int.max] {
            let context = RowScheduleContext(maxThreadCount: 2, preferredChunkSize: hint)
            let empty: [DynamicVector] = []
            #expect(try await ExecutionOperations.map(empty, transform: { $0 }, context: context).isEmpty)
            #expect(try await ExecutionOperations.normalize(empty, context: context).isEmpty)
            #expect(try await ExecutionOperations.distanceMatrix(empty, context: context).isEmpty)
            let zeroWidth = [DynamicVector](repeating: DynamicVector(), count: 1000)
            #expect(try await ExecutionOperations.map(zeroWidth, transform: { $0 }, context: context) == zeroWidth)
            #expect(try await ExecutionOperations.normalize(zeroWidth, context: context) == zeroWidth)
            let zeroMatrix = try await ExecutionOperations.distanceMatrix(zeroWidth, context: context)
            #expect(zeroMatrix.count == 1000 && zeroMatrix.allSatisfy { $0.count == 1000 && $0.allSatisfy { $0 == 0 } })
            let ragged = (0..<1000).map { DynamicVector($0 % 2 == 0 ? [3, 4] : [2]) }
            let mapped = try await ExecutionOperations.map(ragged, transform: { $0 + 1 }, context: context)
            #expect(mapped == ragged.map { DynamicVector($0.toArray().map { $0 + 1 }) })
            let normalized = try await ExecutionOperations.normalize(ragged, context: context)
            #expect(normalized.enumerated().allSatisfy { $0.element.toArray() == ($0.offset % 2 == 0 ? [0.6, 0.8] : [1]) })
            do {
                _ = try await ExecutionOperations.distanceMatrix(ragged, context: context)
                Issue.record("Ragged matrix accepted")
            } catch let error as VectorError { #expect(error.kind == .dimensionMismatch) }
        }
    }
}

private enum RowFactoryFailure: Error { case rejected }
private struct RejectingWideRow: VectorProtocol, VectorFactory {
    var storage: [Float]
    var scalarCount: Int { storage.count }
    subscript(index: Int) -> Float { storage[index] }
    init() { storage = [Float](repeating: 1, count: 768) }
    init(_ array: [Float]) { storage = array }
    init(repeating value: Float) { storage = [Float](repeating: value, count: 768) }
    func toArray() -> [Float] { storage }
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }
    mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer { try body($0) }
    }
    static func create(from scalars: [Float]) throws -> Self { throw RowFactoryFailure.rejected }
}

private struct CancellingRowMetric: DistanceMetric {
    let name = "Row scheduling cancellation"
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float {
        withUnsafeCurrentTask { $0?.cancel() }
        return abs(a[0] - b[0])
    }
}

extension AutomaticRowSchedulingTests {
    @Test func wideParallelFactoriesPropagateOriginalErrors() async {
        let rows = [RejectingWideRow](repeating: RejectingWideRow(), count: 1000)
        for normalize in [false, true] {
            let events = RowScheduleEvents()
            do {
                _ = try await BoundedExecution.$observation.withValue({ events.record($0) }, operation: {
                    if normalize { return try await ExecutionOperations.normalize(rows) }
                    return try await ExecutionOperations.map(rows, transform: { $0 + 1 })
                })
                Issue.record("Throwing factory accepted")
            } catch {
                if case RowFactoryFailure.rejected = error {} else { Issue.record("Factory error changed: \(error)") }
            }
            events.verify(phase: normalize ? "transform.normalize" : "transform.map",
                          maximum: CPUContext.automatic.maxThreadCount,
                          multiple: CPUContext.automatic.maxThreadCount > 1)
        }
    }

    @Test func wideParallelBodiesPropagateCancellation() async {
        let rows = [DynamicVector](repeating: DynamicVector([Float](repeating: 1, count: 768)), count: 1000)
        // Child tasks confine intentional self-cancellation to the operation being tested.
        for matrix in [false, true] {
            await Task {
                do {
                    if matrix {
                        _ = try await ExecutionOperations.distanceMatrix(rows, metric: CancellingRowMetric())
                    } else {
                        _ = try await ExecutionOperations.map(rows, transform: { value in
                            withUnsafeCurrentTask { $0?.cancel() }
                            return value
                        })
                    }
                    Issue.record("Cancelled body returned success")
                } catch { #expect(error is CancellationError) }
            }.value
        }
    }
}
