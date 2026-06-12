//
//  KNNGraph.swift
//  VectorCore
//
//  CSR k-nearest-neighbor graph — the Core-owned interchange contract from
//  gap analysis §3.2. VectorCore declares zero package dependencies and the
//  dependency arrow points VectorIndex → VectorCore only, so graph-consuming
//  math (UMAP, §3.3) takes this struct as INPUT: VectorIndex (or any ANN
//  backend) populates it at corpus scale, and data flows Index → Core while
//  code never does. `bruteForce` below is the exact in-Core builder, so the
//  full UMAP pipeline also runs with no external producer at sample scale.
//

import Foundation

/// A k-nearest-neighbor graph in compressed-sparse-row form.
///
/// Row `i`'s neighbors occupy `neighborIndices[rowOffsets[i]..<rowOffsets[i+1]]`
/// with matching `distances`. The contract:
/// - no self-loops (`neighborIndices` in a row never names that row),
/// - distances are finite and non-negative (zero is legal: duplicate points),
/// - rows may have differing neighbor counts and need not be distance-sorted
///   (CSR generality — ANN backends prune asymmetrically).
public struct KNNGraph: Sendable {
    /// n — number of points (rows).
    public let pointCount: Int

    /// n+1 monotone offsets into the entry arrays; `rowOffsets[0] == 0`.
    public let rowOffsets: [Int]

    /// Column indices, one per stored edge. `Int32` keeps 5M×k graphs
    /// compact (and bounds `pointCount` at `Int32.max`).
    public let neighborIndices: [Int32]

    /// Edge lengths, parallel to `neighborIndices`.
    public let distances: [Float]

    /// Total number of stored directed edges (nnz).
    public var edgeCount: Int { neighborIndices.count }

    /// Validates the CSR invariants above and stores the arrays.
    ///
    /// - Throws: `VectorError.invalidData` for malformed structure,
    ///   `.invalidDimension` for a non-positive or over-`Int32` point count.
    public init(
        pointCount: Int,
        rowOffsets: [Int],
        neighborIndices: [Int32],
        distances: [Float]
    ) throws {
        guard pointCount >= 1, pointCount <= Int(Int32.max) else {
            throw VectorError.invalidDimension(
                pointCount, reason: "pointCount must be in 1...Int32.max")
        }
        guard rowOffsets.count == pointCount + 1 else {
            throw VectorError.invalidData(
                "rowOffsets has \(rowOffsets.count) entries, expected pointCount + 1 = \(pointCount + 1)")
        }
        guard rowOffsets[0] == 0, rowOffsets[pointCount] == neighborIndices.count else {
            throw VectorError.invalidData(
                "rowOffsets must run from 0 to edge count \(neighborIndices.count)")
        }
        guard neighborIndices.count == distances.count else {
            throw VectorError.invalidData(
                "neighborIndices (\(neighborIndices.count)) and distances (\(distances.count)) differ in length")
        }
        for i in 0..<pointCount {
            let low = rowOffsets[i]
            let high = rowOffsets[i + 1]
            guard low <= high else {
                throw VectorError.invalidData("rowOffsets must be non-decreasing (row \(i))")
            }
            for idx in low..<high {
                let j = Int(neighborIndices[idx])
                guard j >= 0, j < pointCount else {
                    throw VectorError.invalidData(
                        "neighbor index \(j) out of range 0..<\(pointCount) (row \(i))")
                }
                guard j != i else {
                    throw VectorError.invalidData("self-loop at row \(i)")
                }
                let dist = distances[idx]
                guard dist >= 0, dist.isFinite else {
                    throw VectorError.invalidData(
                        "distance \(dist) at row \(i) must be finite and non-negative")
                }
            }
        }
        self.pointCount = pointCount
        self.rowOffsets = rowOffsets
        self.neighborIndices = neighborIndices
        self.distances = distances
    }

    /// CSR storage range of row `i`'s entries.
    public func neighborRange(of i: Int) -> Range<Int> {
        rowOffsets[i]..<rowOffsets[i + 1]
    }
}

// MARK: - Exact construction

extension KNNGraph {
    /// Exact kNN by blocked all-pairs Euclidean distance — O(n²·d) GEMM
    /// passes plus O(n²) selection. This is the reference builder for
    /// samples and tests; at corpus scale build the graph with an ANN index
    /// (VectorIndex, §3.2) and pass it in.
    ///
    /// Distances are Euclidean via the Gram identity
    /// ‖x−y‖² = ‖x‖² + ‖y‖² − 2⟨x,y⟩. For unit-normalized embeddings this
    /// is monotone in cosine distance (‖x−y‖² = 2 − 2cosθ), so the neighbor
    /// sets match cosine kNN. Each row comes back distance-sorted ascending
    /// with index-order tie-breaking — fully deterministic.
    ///
    /// - Throws: `VectorError.invalidOperation` for n < 2,
    ///   `.invalidDimension` for k outside `1...n−1`,
    ///   `.dimensionMismatch` for ragged input.
    public static func bruteForce<V: VectorProtocol>(
        _ vectors: [V],
        neighbors k: Int
    ) throws -> KNNGraph where V.Scalar == Float {
        let n = vectors.count
        guard n >= 2 else {
            throw VectorError.invalidOperation(
                "KNNGraph.bruteForce", reason: "need at least 2 points, got \(n)")
        }
        guard k >= 1, k <= n - 1 else {
            throw VectorError.invalidDimension(
                k, reason: "neighbors must be in 1...\(n - 1) (n = \(n))")
        }
        let d = vectors[0].scalarCount
        let flat = try PCAModel.flattenColumnMajor(vectors, count: n, dimension: d)

        var norms = [Float](repeating: 0, count: n)
        flat.withUnsafeBufferPointer { fb in
            for c in 0..<d {
                let base = c * n
                for i in 0..<n { norms[i] += fb[base + i] * fb[base + i] }
            }
        }

        var neighborIndices = [Int32](repeating: 0, count: n * k)
        var distances = [Float](repeating: 0, count: n * k)
        var bestDistance = [Float](repeating: .infinity, count: k)
        var bestIndex = [Int32](repeating: -1, count: k)

        // Query-row blocks bound the Gram panel at blockRows×n floats.
        let blockSize = min(n, 256)
        var start = 0
        while start < n {
            let blockRows = min(blockSize, n - start)
            var block = [Float](repeating: 0, count: blockRows * d)
            for c in 0..<d {
                let source = c * n + start
                let target = c * blockRows
                for r in 0..<blockRows { block[target + r] = flat[source + r] }
            }
            // G (blockRows×n) = X_block · Xᵀ, all column-major.
            let gram = LAMatMul.multiply(block, flat, m: blockRows, n: n, k: d, transposeB: true)

            for r in 0..<blockRows {
                let gi = start + r
                for s in 0..<k {
                    bestDistance[s] = .infinity
                    bestIndex[s] = -1
                }
                for j in 0..<n where j != gi {
                    var squared = norms[gi] + norms[j] - 2 * gram[j * blockRows + r]
                    if squared < 0 { squared = 0 }
                    if squared >= bestDistance[k - 1] { continue }
                    var pos = k - 1
                    while pos > 0 && bestDistance[pos - 1] > squared {
                        bestDistance[pos] = bestDistance[pos - 1]
                        bestIndex[pos] = bestIndex[pos - 1]
                        pos -= 1
                    }
                    bestDistance[pos] = squared
                    bestIndex[pos] = Int32(j)
                }
                let rowBase = gi * k
                for s in 0..<k {
                    neighborIndices[rowBase + s] = bestIndex[s]
                    distances[rowBase + s] = bestDistance[s].squareRoot()
                }
            }
            start += blockRows
        }

        return try KNNGraph(
            pointCount: n,
            rowOffsets: (0...n).map { $0 * k },
            neighborIndices: neighborIndices,
            distances: distances)
    }
}
