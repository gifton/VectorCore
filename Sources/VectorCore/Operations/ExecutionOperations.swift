// VectorCore retained execution-context operations.
import Foundation

internal enum ExecutionOperations {
    internal static let parallelThreshold = 1000
}

extension ExecutionOperations {
    /// Context fields are scheduling hints, not an executor/device guarantee.
    /// ExecutionSchedulingContractTests covers cancellation and throwing factories.
    internal static func map<V: VectorProtocol & VectorFactory>(
        _ vectors: [V], transform: @Sendable @escaping (Float) -> Float,
        context: any ExecutionContext = CPUContext.automatic
    ) async throws -> [V] where V.Scalar == Float {
        try await BoundedExecution.map(0..<vectors.count, limit: context.maxThreadCount,
            parallel: vectors.count >= parallelThreshold, minimum: context.preferredChunkSize,
            phase: "transform.map") { index in
            let vector = vectors[index]
            var result = Array(repeating: Float(0), count: vector.scalarCount)
            vector.withUnsafeBufferPointer { buffer in
                for i in buffer.indices { result[i] = transform(buffer[i]) }
            }
            return try V.create(from: result)
        }
    }

    /// Symmetric pairwise matrix including supplied self scores. Custom metrics
    /// must be symmetric; directed parity is not promised (MatrixMetricContractTests).
    internal static func distanceMatrix<V: VectorProtocol, M: DistanceMetric>(
        _ vectors: [V], metric: M = EuclideanDistance(),
        context: any ExecutionContext = CPUContext.automatic
    ) async throws -> [[Float]] where M.Scalar == Float, V.Scalar == Float {
        try validateConsistentDimensions(vectors)
        let n = vectors.count
        // n*n >= threshold, without forming an unrepresentable square.
        let parallel = n > 0 && n > (parallelThreshold - 1) / n
        let rows = try await BoundedExecution.map(0..<n, limit: context.maxThreadCount,
            parallel: parallel, minimum: context.preferredChunkSize, phase: "matrix.distance") { i in
            var row: [Float] = []
            row.reserveCapacity(n - i)
            for j in i..<n { row.append(metric.distance(vectors[i], vectors[j])) }
            return row
        }
        var matrix = Array(repeating: Array(repeating: Float(0), count: n), count: n)
        for i in 0..<n {
            for j in i..<n { matrix[i][j] = rows[i][j-i]; matrix[j][i] = rows[i][j-i] }
        }
        try Task.checkCancellation()
        return matrix
    }

    /// Exhaustive preflight before any matrix metric call, in Debug and Release.
    /// Exercised by ExecutionSchedulingContractTests.boundedRegressionChild.
    internal static func validateConsistentDimensions<V: VectorProtocol>(_ vectors: [V]) throws {
        guard let first = vectors.first else { return }
        for vector in vectors.dropFirst() {
            guard vector.scalarCount == first.scalarCount else {
                throw VectorError.dimensionMismatch(expected: first.scalarCount, actual: vector.scalarCount)
            }
        }
    }
}

// MARK: - Performance Context

/// Context for managing operation performance
internal struct OperationContext: Sendable {
    internal let bufferPool: BufferPool
    internal let executionContext: any ExecutionContext

    internal init(
        bufferPool: BufferPool = globalBufferPool,
        executionContext: any ExecutionContext = CPUContext.automatic
    ) {
        self.bufferPool = bufferPool
        self.executionContext = executionContext
    }

    /// Use temporary buffer for an operation
    internal func withTemporaryBuffer<T>(
        count: Int,
        _ body: (UnsafeMutableBufferPointer<Float>) async throws -> T
    ) async rethrows -> T where T: Sendable {
        let sendableBuffer = await bufferPool.acquire(count: count)
        do {
            let result = try await body(sendableBuffer.buffer)
            await bufferPool.release(sendableBuffer)
            return result
        } catch {
            await bufferPool.release(sendableBuffer)
            throw error
        }
    }
}
