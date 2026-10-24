//
//  KNNGraph.swift
//  VectorCore
//
//  CSR k-nearest-neighbor graph — the Core-owned interchange contract from
//  gap analysis §3.2. VectorCore declares zero package dependencies and the
//  dependency arrow points VectorIndex → VectorCore only, so graph-consuming
//  math (UMAP, §3.3) takes this struct as INPUT: VectorIndex (or any ANN
//  backend) populates it at corpus scale, and data flows Index → Core while
//  code never does. `bruteForce` exhaustively enumerates Float Gram scores, so the
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
        let expectedOffsets = try UMAPChecked.increment(pointCount, label: "CSR offsets")
        guard rowOffsets.count == expectedOffsets else {
            throw VectorError.invalidData(
                "rowOffsets has \(rowOffsets.count) entries, expected pointCount + 1 = \(expectedOffsets)")
        }
        guard rowOffsets[0] == 0, rowOffsets[pointCount] == neighborIndices.count else {
            throw VectorError.invalidData(
                "rowOffsets must run from 0 to edge count \(neighborIndices.count)")
        }
        guard neighborIndices.count == distances.count else {
            throw VectorError.invalidData(
                "neighborIndices (\(neighborIndices.count)) and distances (\(distances.count)) differ in length")
        }
        // Complete preflight before any row indexes either entry array.
        // UMAPRangeContractTests exercises malformed interior offsets with valid endpoints.
        var previous = 0
        for offset in rowOffsets {
            guard offset >= previous, offset >= 0, offset <= neighborIndices.count else {
                throw VectorError.invalidData("rowOffsets must be bounded by edge count and non-decreasing")
            }
            previous = offset
        }
        for i in 0..<pointCount {
            let low = rowOffsets[i]
            let high = rowOffsets[i + 1]
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

// MARK: - Exhaustive construction

extension KNNGraph {
    /// Exhaustive candidate enumeration using blocked Float Gram distances.
    /// Work is O(n²·d) GEMM plus O(n²·k) insertion selection; checked sizes are representation
    /// limits, not an allocation budget or corpus-scale readiness guarantee.
    ///
    /// Gram subtraction can cancel nearby distances under large common offsets,
    /// and extreme finite input can overflow Float intermediates. This builder is
    /// not a stable direct-distance oracle. UMAPRangeContractTests characterizes
    /// the offset fixture against independent Double differences; no reranking or
    /// routing repair is performed here. Ordinary controls remain in KNNGraphSuite.
    /// Rows sort by computed distance with index-order ties on the current route.
    ///
    /// Every query row scans all n − 1 candidates, so candidate work is
    /// quadratic in n for any k; each block's Gram panel holds min(n, 256)·n
    /// Floats next to the n×d packed copy (see the body). One-host times, taken
    /// in High Power Mode on synthetic data (conditions in the dated record cited
    /// by Docs/Performance_Guide.md, "Measured Workload Envelope (One Host)"),
    /// include 44.4 s at n = 100,000, d = 768, k = 15. There, time grew more than
    /// quadratically from n = 5,000 to 25,000 and from 25,000 to 100,000, but
    /// less than quadratically from 500 to 5,000; do not extrapolate. Larger n
    /// is not measured.
    ///
    /// - Throws: invalidDimension for n/d, k, LP64 or count/byte rejection (fewer than two points included; 0.3.3
    ///   threw invalidOperation for that case);
    ///   dimensionMismatch for ragged input; invalidData if computed entries do
    ///   not satisfy the checked CSR contract.
    public static func bruteForce<V: VectorProtocol>(
        _ vectors: [V],
        neighbors k: Int
    ) throws -> KNNGraph where V.Scalar == Float {
        let n = vectors.count
        guard n >= 2 else { throw VectorError.invalidDimension(n, reason: "KNNGraph.bruteForce needs at least 2 points") }
        guard k >= 1, k <= n - 1 else {
            throw VectorError.invalidDimension(
                k, reason: "neighbors must be in 1...\(n - 1) (n = \(n))")
        }
        let d = vectors[0].scalarCount
        let sizes = try checkedSizes(points: n, dimensions: d, neighbors: k)
        let flat = try PCAModel.flattenColumnMajor(vectors, count: n, dimension: d)

        var norms = [Float](repeating: 0, count: n)
        flat.withUnsafeBufferPointer { fb in
            for c in 0..<d {
                let base = c * n
                for i in 0..<n { norms[i] += fb[base + i] * fb[base + i] }
            }
        }

        var neighborIndices = [Int32](repeating: 0, count: sizes.edges)
        var distances = [Float](repeating: 0, count: sizes.edges)
        var bestDistance = [Float](repeating: .infinity, count: k)
        var bestIndex = [Int32](repeating: -1, count: k)

        // Query-row blocks bound the Gram panel at blockRows×n floats.
        let blockSize = min(n, 256)
        var start = 0
        while start < n {
            let blockRows = min(blockSize, n - start)
            let panelCount = try UMAPChecked.count([blockRows, d], stride: MemoryLayout<Float>.stride, label: "KNN block")
            _ = try UMAPChecked.count([blockRows, n], stride: MemoryLayout<Float>.stride, label: "KNN Gram panel")
            var block = [Float](repeating: 0, count: panelCount)
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

        var offsets = [Int](repeating: 0, count: sizes.offsets)
        for i in 0...n { offsets[i] = i * k } // i*k <= the checked n*k bound.
        return try KNNGraph(
            pointCount: n,
            rowOffsets: offsets,
            neighborIndices: neighborIndices,
            distances: distances)
    }

    /// Pure preflight used before allocation/BLAS and exercised at representability
    /// limits by UMAPRangeContractTests, without giant arrays or vendor calls.
    internal static func checkedSizes(points n: Int, dimensions d: Int, neighbors k: Int) throws -> (edges: Int, offsets: Int) {
        guard n >= 2 else { throw VectorError.invalidDimension(n, reason: "KNN needs at least 2 points") }
        try UMAPChecked.lp64(n, label: "KNN points")
        try UMAPChecked.lp64(d, label: "KNN dimensions")
        guard k >= 1, k < n else { throw VectorError.invalidDimension(k, reason: "KNN neighbors must be in 1..<points") }
        let edges = try UMAPChecked.count([n, k], stride: MemoryLayout<Int32>.stride, label: "KNN edges")
        _ = try UMAPChecked.count([n, k], stride: MemoryLayout<Float>.stride, label: "KNN distances")
        _ = try UMAPChecked.count([n, d], stride: MemoryLayout<Float>.stride, label: "KNN data")
        _ = try UMAPChecked.count([n], stride: MemoryLayout<Float>.stride, label: "KNN norms")
        _ = try UMAPChecked.count([k], stride: MemoryLayout<Float>.stride, label: "KNN selection")
        let block = min(n, 256)
        _ = try UMAPChecked.count([block, d], stride: MemoryLayout<Float>.stride, label: "KNN block")
        _ = try UMAPChecked.count([block, n], stride: MemoryLayout<Float>.stride, label: "KNN Gram panel")
        let offsets = try UMAPChecked.increment(n, label: "KNN offsets")
        _ = try UMAPChecked.count([offsets], stride: MemoryLayout<Int>.stride, label: "KNN offsets")
        return (edges, offsets)
    }

}
