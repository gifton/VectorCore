import Foundation

extension KNNGraph {
    /// Built-in metrics for exhaustive graph construction, independent of supplied CSR graphs.
    public enum Metric: Sendable, Equatable {
        case euclidean
        case cosine
    }

    /// Exhaustive graph construction with an explicit metric. O(n²d + n²k) work;
    /// packed O(nd), graph O(nk), and at most min(n, 256)·n Float Gram storage.
    ///
    /// Cosine normalizes a private packed copy once per row using Double norms and
    /// Float coordinates, then stores clamped `1 - dot` in [0, 2], without a square root.
    /// Rows sort by computed Float distance, then source index for computed ties.
    /// `KNNMetricTests` exercises bounded accuracy, ties, extreme values and input preservation.
    /// Near-tie order and exact words across BLAS/toolchain builds are not promised.
    /// Existing scalar/search cosine APIs have their own zero-vector and arithmetic rules.
    ///
    /// - Throws: The Euclidean overload's shape/count errors. Cosine additionally throws
    ///   `.invalidData` naming the first zero/nonfinite row (zero-based), after complete
    ///   shape validation; `.operationFailed` for nonfinite intermediate results.
    public static func bruteForce<V: VectorProtocol>(
        _ vectors: [V],
        neighbors k: Int,
        metric: Metric
    ) throws -> KNNGraph where V.Scalar == Float {
        try bruteForce(vectors, neighbors: k, metric: metric, execution: nil)
    }

    internal static func bruteForce<V: VectorProtocol>(
        _ vectors: [V], neighbors k: Int, metric: Metric, execution: UMAPExecutionContext?
    ) throws -> KNNGraph where V.Scalar == Float {
        switch metric {
        case .euclidean: return try bruteForce(vectors, neighbors: k, execution: execution)
        case .cosine: return try cosineGraph(vectors, neighbors: k, execution: execution)
        }
    }

    // Used before selection, including for self dots: no nonfinite candidate can be
    // hidden by the clamp or dropped by top k. Direct guard coverage: cosineRejectsNonfiniteIntermediates.
    internal static func cosineDistance(dot: Float) throws -> Float {
        guard dot.isFinite else {
            throw ErrorBuilder(.operationFailed).message("Cosine graph produced a nonfinite dot product").build()
        }
        return min(2, max(0, 1 - dot))
    }

    private static func cosineGraph<V: VectorProtocol>(
        _ vectors: [V], neighbors k: Int, execution: UMAPExecutionContext?
    ) throws -> KNNGraph where V.Scalar == Float {
        try execution?.checkCancellation()
        let n = vectors.count
        guard n >= 2 else { throw VectorError.invalidDimension(n, reason: "KNNGraph.bruteForce needs at least 2 points") }
        guard k >= 1, k < n else {
            throw VectorError.invalidDimension(k, reason: "neighbors must be in 1...\(n - 1) (n = \(n))")
        }
        let d = vectors[0].scalarCount
        let sizes = try checkedSizes(points: n, dimensions: d, neighbors: k)
        try execution?.report(.graph, completed: 0, total: n)
        var flat = try PCAModel.flattenColumnMajor(vectors, count: n, dimension: d)
        try execution?.checkCancellation()
        try flat.withUnsafeMutableBufferPointer { buffer in
            for row in 0..<n {
                let plan: NormalizationPlan<Float>
                switch NormalizationCore.plan(count: d, get: { buffer[$0 * n + row] }) {
                case .success(let value): plan = value
                case .failure(let error):
                    let reason = error.context.additionalInfo["message"] ?? "invalid vector"
                    throw VectorError.invalidData("Cosine graph row \(row): \(reason)")
                }
                for c in 0..<d {
                    let index = c * n + row
                    let value = plan.coordinate(buffer[index])
                    guard value.isFinite else {
                        throw ErrorBuilder(.operationFailed).message("Cosine graph normalization is nonfinite at row \(row)").build()
                    }
                    buffer[index] = value
                }
            }
        }
        try execution?.checkCancellation()
        var neighborIndices = [Int32](repeating: 0, count: sizes.edges)
        var distances = [Float](repeating: 0, count: sizes.edges)
        var bestDistance = [Float](repeating: .infinity, count: k)
        var bestIndex = [Int32](repeating: -1, count: k)
        let blockSize = min(n, 256)
        var start = 0
        while start < n {
            try execution?.checkCancellation()
            let blockRows = min(blockSize, n - start)
            let panelCount = try UMAPChecked.count([blockRows, d], stride: MemoryLayout<Float>.stride, label: "KNN block")
            _ = try UMAPChecked.count([blockRows, n], stride: MemoryLayout<Float>.stride, label: "KNN Gram panel")
            var block = [Float](repeating: 0, count: panelCount)
            for c in 0..<d {
                let source = c * n + start, target = c * blockRows
                for r in 0..<blockRows { block[target + r] = flat[source + r] }
            }
            let gram = LAMatMul.multiply(block, flat, m: blockRows, n: n, k: d, transposeB: true)
            for r in 0..<blockRows {
                let row = start + r
                for s in 0..<k { bestDistance[s] = .infinity; bestIndex[s] = -1 }
                for j in 0..<n {
                    let distance = try cosineDistance(dot: gram[j * blockRows + r])
                    if j == row || distance >= bestDistance[k - 1] { continue }
                    var pos = k - 1
                    while pos > 0 && bestDistance[pos - 1] > distance {
                        bestDistance[pos] = bestDistance[pos - 1]
                        bestIndex[pos] = bestIndex[pos - 1]
                        pos -= 1
                    }
                    bestDistance[pos] = distance
                    bestIndex[pos] = Int32(j)
                }
                let rowBase = row * k
                for s in 0..<k {
                    neighborIndices[rowBase + s] = bestIndex[s]
                    distances[rowBase + s] = bestDistance[s]
                }
            }
            start += blockRows
            try execution?.checkCancellation()
            if start < n { try execution?.report(.graph, completed: start, total: n) }
        }
        try execution?.checkCancellation()
        var offsets = [Int](repeating: 0, count: sizes.offsets)
        for i in 0...n { offsets[i] = i * k }
        try execution?.checkCancellation()
        let result = try KNNGraph(pointCount: n, rowOffsets: offsets, neighborIndices: neighborIndices, distances: distances)
        try execution?.report(.graph, completed: n, total: n)
        return result
    }
}
