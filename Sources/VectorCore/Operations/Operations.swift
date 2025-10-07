// VectorCore: Modern Operations API
//
// Clean API using provider abstraction - no legacy code
//

import Foundation

/// High-performance vector operations
///
/// Operations provides optimized vector computations with pluggable providers
/// for compute and SIMD operations. Default providers are pure Swift for
/// cross-platform compatibility.
///
/// ## Usage
/// ```swift
/// // Default: Pure Swift implementation
/// let results = try await Operations.findNearest(to: query, in: database, k: 10)
///
/// // With custom providers (e.g., from VectorAccelerate)
/// Operations.computeProvider = MetalComputeProvider()
/// Operations.simdProvider = AccelerateSIMDProvider()
/// ```
public enum Operations {

    // MARK: - Provider Configuration

    /// The compute provider for execution strategy
    @TaskLocal public static var computeProvider: any ComputeProvider = CPUComputeProvider.automatic

    /// The SIMD provider for vectorized operations
    @TaskLocal public static var simdProvider: any ArraySIMDProvider = SwiftSIMDProvider()

    /// The buffer provider for memory management
    @TaskLocal public static var bufferProvider: any BufferProvider = SwiftBufferPool.shared

    // MARK: - Nearest Neighbor Search

    /// Find k nearest neighbors to a query vector
    ///
    /// - Parameters:
    ///   - query: Query vector
    ///   - vectors: Search space
    ///   - k: Number of neighbors to find
    ///   - metric: Distance metric (default: Euclidean)
    /// - Returns: Nearest neighbors sorted by distance
    public static func findNearest<V: VectorProtocol, M: DistanceMetric>(
        to query: V,
        in vectors: [V],
        k: Int = 1,
        metric: M = EuclideanDistance()
    ) async throws -> [NearestNeighborResult] where V.Scalar == Float, M.Scalar == Float {
        // Validate inputs
        guard k > 0 else {
            throw VectorError.invalidDimension(k, reason: "k must be positive")
        }
        guard !vectors.isEmpty else {
            throw VectorError.invalidDimension(0, reason: "Vector collection cannot be empty")
        }
        let expectedDim = vectors[0].scalarCount
        guard query.scalarCount == expectedDim else {
            throw VectorError.dimensionMismatch(expected: expectedDim, actual: query.scalarCount)
        }
        // Ensure all candidate vectors have consistent dimension
        for v in vectors where v.scalarCount != expectedDim {
            throw VectorError.dimensionMismatch(expected: expectedDim, actual: v.scalarCount)
        }

        // Optimized Top‑K fast paths for Vector*Optimized + supported metrics
        if k < vectors.count { // only worth it when K < N
            // Euclidean distance Top‑K via squared kernels (monotonic → correct indices)
            if metric is EuclideanDistance {
                if let q = query as? Vector512Optimized, let c = vectors as? [Vector512Optimized] {
                    let results = try await topk_euclid_optimized(query: q, candidates: c, k: k, dim: 512)
                    return results
                }
                if let q = query as? Vector768Optimized, let c = vectors as? [Vector768Optimized] {
                    let results = try await topk_euclid_optimized(query: q, candidates: c, k: k, dim: 768)
                    return results
                }
                if let q = query as? Vector1536Optimized, let c = vectors as? [Vector1536Optimized] {
                    let results = try await topk_euclid_optimized(query: q, candidates: c, k: k, dim: 1536)
                    return results
                }
            }
            // Cosine fused Top‑K (512 supported)
            if metric is CosineDistance {
                if let q = query as? Vector512Optimized, let c = vectors as? [Vector512Optimized] {
                    let results = try await topk_cosine_fused_512(query: q, candidates: c, k: k)
                    return results
                }
            }
            // Dot product Top‑K (512 supported) — return negative dot as distance
            if metric is DotProductDistance {
                if let q = query as? Vector512Optimized, let c = vectors as? [Vector512Optimized] {
                    let results = try await topk_dot_512(query: q, candidates: c, k: k)
                    return results
                }
            }
        }

        let distances = try await computeDistances(
            from: query,
            to: vectors,
            metric: metric
        )

        // Find k smallest distances
        let results = distances
            .enumerated()
            .map { NearestNeighborResult(index: $0.offset, distance: $0.element) }
            .sorted { $0.distance < $1.distance }
            .prefix(min(k, vectors.count))

        return Array(results)
    }

    /// Find k nearest neighbors for multiple queries
    ///
    /// - Parameters:
    ///   - queries: Query vectors
    ///   - vectors: Search space
    ///   - k: Number of neighbors per query
    ///   - metric: Distance metric
    /// - Returns: Results for each query
    public static func findNearestBatch<V: VectorProtocol, M: DistanceMetric>(
        queries: [V],
        in vectors: [V],
        k: Int = 1,
        metric: M = EuclideanDistance()
    ) async throws -> [[NearestNeighborResult]] where V.Scalar == Float, M.Scalar == Float {
        // Validate inputs
        guard k > 0 else {
            throw VectorError.invalidDimension(k, reason: "k must be positive")
        }
        guard !queries.isEmpty else {
            throw VectorError.invalidDimension(0, reason: "queries cannot be empty")
        }
        guard !vectors.isEmpty else {
            throw VectorError.invalidDimension(0, reason: "vectors cannot be empty")
        }
        let qDim = queries[0].scalarCount
        let vDim = vectors[0].scalarCount
        // All queries same dim
        for q in queries where q.scalarCount != qDim {
            throw VectorError.dimensionMismatch(expected: qDim, actual: q.scalarCount)
        }
        // All vectors same dim
        for v in vectors where v.scalarCount != vDim {
            throw VectorError.dimensionMismatch(expected: vDim, actual: v.scalarCount)
        }
        // Queries dim must match vectors dim
        guard qDim == vDim else {
            throw VectorError.dimensionMismatch(expected: vDim, actual: qDim)
        }
        return try await computeProvider.parallelExecute(items: 0..<queries.count) { i in
            try await findNearest(
                to: queries[i],
                in: vectors,
                k: k,
                metric: metric
            )
        }
    }

    // MARK: - Optimized Top‑K helpers

    @usableFromInline
    static func mergeBuffers(_ a: TopKBuffer, _ b: TopKBuffer, isMinHeap: Bool, k: Int) -> TopKBuffer {
        var out = TopKBuffer(k: k, isMinHeap: isMinHeap)
        TopKSelectionKernels.mergeTopK(a, b, into: &out)
        return out
    }

    @inline(__always)
    private static func toResults(_ buf: TopKBuffer, sqrtValues: Bool = false, negateValues: Bool = false) -> [NearestNeighborResult] {
        let count = min(buf.size, buf.k)
        var pairs: [(Int, Float)] = []
        pairs.reserveCapacity(count)
        for i in 0..<count {
            var v = buf.vals[i]
            if sqrtValues { v = sqrt(v) }
            if negateValues { v = -v }
            pairs.append((buf.idxs[i], v))
        }
        return pairs.sorted { $0.1 < $1.1 }.map { NearestNeighborResult(index: $0.0, distance: $0.1) }
    }

    // Euclidean Top‑K using squared distance kernels for selection, sqrt at the end for distances
    private static func topk_euclid_optimized(
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        k: Int,
        dim: Int
    ) async throws -> [NearestNeighborResult] {
        let n = candidates.count
        let initial = TopKBuffer(k: k, isMinHeap: false)
        let buf = try await computeProvider.parallelReduce(items: 0..<n, initial: initial, { range in
            var local = TopKBuffer(k: k, isMinHeap: false)
            TopKSelectionKernels.range_topk_euclid2_512(query: query, candidates: candidates, range: range, k: k, localTopK: &local)
            return local
        }, { a, b in mergeBuffers(a, b, isMinHeap: false, k: k) })
        return toResults(buf, sqrtValues: true)
    }

    private static func topk_euclid_optimized(
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        k: Int,
        dim: Int
    ) async throws -> [NearestNeighborResult] {
        let n = candidates.count
        let initial = TopKBuffer(k: k, isMinHeap: false)
        let buf = try await computeProvider.parallelReduce(items: 0..<n, initial: initial, { range in
            var local = TopKBuffer(k: k, isMinHeap: false)
            TopKSelectionKernels.range_topk_euclid2_768(query: query, candidates: candidates, range: range, k: k, localTopK: &local)
            return local
        }, { a, b in mergeBuffers(a, b, isMinHeap: false, k: k) })
        return toResults(buf, sqrtValues: true)
    }

    private static func topk_euclid_optimized(
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        k: Int,
        dim: Int
    ) async throws -> [NearestNeighborResult] {
        let n = candidates.count
        let initial = TopKBuffer(k: k, isMinHeap: false)
        let buf = try await computeProvider.parallelReduce(items: 0..<n, initial: initial, { range in
            var local = TopKBuffer(k: k, isMinHeap: false)
            TopKSelectionKernels.range_topk_euclid2_1536(query: query, candidates: candidates, range: range, k: k, localTopK: &local)
            return local
        }, { a, b in mergeBuffers(a, b, isMinHeap: false, k: k) })
        return toResults(buf, sqrtValues: true)
    }

    private static func topk_cosine_fused_512(
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        k: Int
    ) async throws -> [NearestNeighborResult] {
        let n = candidates.count
        let initial = TopKBuffer(k: k, isMinHeap: false)
        let buf = try await computeProvider.parallelReduce(items: 0..<n, initial: initial, { range in
            var local = TopKBuffer(k: k, isMinHeap: false)
            TopKSelectionKernels.range_topk_cosine_fused_512(query: query, candidates: candidates, range: range, k: k, localTopK: &local)
            return local
        }, { a, b in mergeBuffers(a, b, isMinHeap: false, k: k) })
        return toResults(buf)
    }

    private static func topk_dot_512(
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        k: Int
    ) async throws -> [NearestNeighborResult] {
        let n = candidates.count
        let initial = TopKBuffer(k: k, isMinHeap: true)
        let buf = try await computeProvider.parallelReduce(items: 0..<n, initial: initial, { range in
            var local = TopKBuffer(k: k, isMinHeap: true)
            TopKSelectionKernels.range_topk_dot_512(query: query, candidates: candidates, range: range, k: k, localTopK: &local)
            return local
        }, { a, b in mergeBuffers(a, b, isMinHeap: true, k: k) })
        // Convert similarity to distance via negation
        return toResults(buf, negateValues: true)
    }

    // MARK: - Distance Computation

    /// Compute distances from query to all vectors
    private static func computeDistances<V: VectorProtocol, M: DistanceMetric>(
        from query: V,
        to vectors: [V],
        metric: M
    ) async throws -> [Float] where V.Scalar == Float, M.Scalar == Float {
        let dim = query.scalarCount
        let items = vectors.count

        // Infer variant (optimized vs generic) based on runtime type
        let variant: ParallelHeuristic.Variant = {
            if query is Vector512Optimized || query is Vector768Optimized || query is Vector1536Optimized {
                return .optimized
            }
            return .generic
        }()

        // Map metric type to metric class
        let metricClass: ParallelHeuristic.MetricClass = {
            switch metric {
            case is EuclideanDistance: return .euclideanLike
            case is DotProductDistance: return .dot
            case is CosineDistance: return .cosine
            case is ManhattanDistance: return .manhattan
            default: return .euclideanLike
            }
        }()

        // Decide parallelism using AutoTuning if available; otherwise fallback heuristic
        var tunedMinChunk: Int?
        var parallel: Bool
        if let kind = {
            switch metricClass {
            case .euclideanLike:
                // We are using euclidean (sqrt) at this level
                return KernelKind.euclid
            case .cosine:
                return KernelKind.cosineFused
            case .dot:
                return KernelKind.dot
            case .manhattan:
                return nil
            }
        }() {
            let cores = ProcessInfo.processInfo.activeProcessorCount
            // Calibrator: measure seq cost over a tiny sample; estimate parallel overhead and effective factor
            let tuning = AutoTuning.calibrateIfNeeded(dim: dim, kind: kind, providerCores: cores) {
                // Fix: Clamp M to actual array size to prevent out-of-bounds access when items < 32
                let M = min(max(32, min(items, 256)), items)
                var idx = 0
                let clock = ContinuousClock()
                let start = clock.now
                if M > 0 {
                    for _ in 0..<M {
                        let v = vectors[idx]
                        _ = metric.distance(query, v)
                        idx &+= 1
                    }
                }
                let elapsed = start.duration(to: clock.now).nanoseconds
                let a = (M > 0) ? (elapsed / Double(M)) : 1_000.0
                let Tp = ParallelHeuristic.parallelOverheadNs(items: M)
                let peff = Double(min(cores, 8)) * 0.7
                return CalibrationProbes(nsPerCandidateSeq: a, parallelOverheadNs: Tp, effectiveParallelFactor: peff)
            }
            tunedMinChunk = tuning.minChunk
            parallel = items >= tuning.breakEvenN
        } else {
            parallel = ParallelHeuristic.shouldParallelize(dim: dim, items: items, variant: variant, metric: metricClass)
        }

        // Fast paths using register-blocked batch kernels for optimized types
        if variant == .optimized {
            // Euclidean distance (sqrt)
            if metric is EuclideanDistance {
                if let q = query as? Vector512Optimized, let c = vectors as? [Vector512Optimized] {
                    return try await batchKernelDistances_euclid(query: q, candidates: c, dim: 512, parallel: parallel, tunedMinChunk: tunedMinChunk)
                }
                if let q = query as? Vector768Optimized, let c = vectors as? [Vector768Optimized] {
                    return try await batchKernelDistances_euclid(query: q, candidates: c, dim: 768, parallel: parallel, tunedMinChunk: tunedMinChunk)
                }
                if let q = query as? Vector1536Optimized, let c = vectors as? [Vector1536Optimized] {
                    return try await batchKernelDistances_euclid(query: q, candidates: c, dim: 1536, parallel: parallel, tunedMinChunk: tunedMinChunk)
                }
            }
            // Cosine distance (fused one-pass)
            if metric is CosineDistance {
                if let q = query as? Vector512Optimized, let c = vectors as? [Vector512Optimized] {
                    return try await batchKernelDistances_cosineFused(query: q, candidates: c, dim: 512, parallel: parallel, tunedMinChunk: tunedMinChunk)
                }
                if let q = query as? Vector768Optimized, let c = vectors as? [Vector768Optimized] {
                    return try await batchKernelDistances_cosineFused(query: q, candidates: c, dim: 768, parallel: parallel, tunedMinChunk: tunedMinChunk)
                }
                if let q = query as? Vector1536Optimized, let c = vectors as? [Vector1536Optimized] {
                    return try await batchKernelDistances_cosineFused(query: q, candidates: c, dim: 1536, parallel: parallel, tunedMinChunk: tunedMinChunk)
                }
            }
        }

        // Fallback: existing per-candidate path
        if parallel {
            return try await computeProvider.parallelExecute(items: 0..<vectors.count) { i in
                metric.distance(query, vectors[i])
            }
        } else {
            return vectors.map { metric.distance(query, $0) }
        }
    }

    // MARK: - Optimized batch-kernel helpers

    @usableFromInline
    struct _Chunk: Sendable { let start: Int; let values: [Float] }

    @inline(__always)
    private static func minChunk(forDim dim: Int) -> Int {
        switch dim {
        case 1536: return 512
        case 768: return 256
        default: return 256
        }
    }

    // Euclidean (sqrt) batch via BatchKernels
    private static func batchKernelDistances_euclid(
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        dim: Int,
        parallel: Bool,
        tunedMinChunk: Int? = nil
    ) async throws -> [Float] {
        let n = candidates.count
        if !parallel {
            var out = [Float](repeating: 0, count: n)
            out.withUnsafeMutableBufferPointer { buf in
                var start = 0
                let step = tunedMinChunk ?? minChunk(forDim: dim)
                while start < n {
                    let end = min(start + step, n)
                    let sub = UnsafeMutableBufferPointer<Float>(start: buf.baseAddress!.advanced(by: start), count: end - start)
                    BatchKernels.range_euclid_512(query: query, candidates: candidates, range: start..<end, out: sub)
                    start = end
                }
            }
            return out
        } else {
            let chunks: [_Chunk] = try await computeProvider.parallelReduce(
                items: 0..<n,
                initial: [],
                minChunk: tunedMinChunk ?? minChunk(forDim: dim),
                { range in
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_euclid_512(query: query, candidates: candidates, range: range, out: sub)
                    }
                    return [_Chunk(start: range.lowerBound, values: tmp)]
                },
                { $0 + $1 }
            )
            var out = [Float](repeating: 0, count: n)
            for ch in chunks { out.replaceSubrange(ch.start..<(ch.start + ch.values.count), with: ch.values) }
            return out
        }
    }

    private static func batchKernelDistances_euclid(
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        dim: Int,
        parallel: Bool,
        tunedMinChunk: Int? = nil
    ) async throws -> [Float] {
        let n = candidates.count
        if !parallel {
            var out = [Float](repeating: 0, count: n)
            out.withUnsafeMutableBufferPointer { buf in
                var start = 0
                let step = tunedMinChunk ?? minChunk(forDim: dim)
                while start < n {
                    let end = min(start + step, n)
                    let sub = UnsafeMutableBufferPointer<Float>(start: buf.baseAddress!.advanced(by: start), count: end - start)
                    BatchKernels.range_euclid_768(query: query, candidates: candidates, range: start..<end, out: sub)
                    start = end
                }
            }
            return out
        } else {
            let chunks: [_Chunk] = try await computeProvider.parallelReduce(
                items: 0..<n,
                initial: [],
                minChunk: tunedMinChunk ?? minChunk(forDim: dim),
                { range in
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_euclid_768(query: query, candidates: candidates, range: range, out: sub)
                    }
                    return [_Chunk(start: range.lowerBound, values: tmp)]
                },
                { $0 + $1 }
            )
            var out = [Float](repeating: 0, count: n)
            for ch in chunks { out.replaceSubrange(ch.start..<(ch.start + ch.values.count), with: ch.values) }
            return out
        }
    }

    private static func batchKernelDistances_euclid(
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        dim: Int,
        parallel: Bool,
        tunedMinChunk: Int? = nil
    ) async throws -> [Float] {
        let n = candidates.count
        if !parallel {
            var out = [Float](repeating: 0, count: n)
            out.withUnsafeMutableBufferPointer { buf in
                var start = 0
                let step = tunedMinChunk ?? minChunk(forDim: dim)
                while start < n {
                    let end = min(start + step, n)
                    let sub = UnsafeMutableBufferPointer<Float>(start: buf.baseAddress!.advanced(by: start), count: end - start)
                    BatchKernels.range_euclid_1536(query: query, candidates: candidates, range: start..<end, out: sub)
                    start = end
                }
            }
            return out
        } else {
            let chunks: [_Chunk] = try await computeProvider.parallelReduce(
                items: 0..<n,
                initial: [],
                minChunk: tunedMinChunk ?? minChunk(forDim: dim),
                { range in
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_euclid_1536(query: query, candidates: candidates, range: range, out: sub)
                    }
                    return [_Chunk(start: range.lowerBound, values: tmp)]
                },
                { $0 + $1 }
            )
            var out = [Float](repeating: 0, count: n)
            for ch in chunks { out.replaceSubrange(ch.start..<(ch.start + ch.values.count), with: ch.values) }
            return out
        }
    }

    // Cosine (fused) batch via BatchKernels
    private static func batchKernelDistances_cosineFused(
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        dim: Int,
        parallel: Bool,
        tunedMinChunk: Int? = nil
    ) async throws -> [Float] {
        let n = candidates.count
        if !parallel {
            var out = [Float](repeating: 0, count: n)
            out.withUnsafeMutableBufferPointer { buf in
                var start = 0
                let step = tunedMinChunk ?? minChunk(forDim: dim)
                while start < n {
                    let end = min(start + step, n)
                    let sub = UnsafeMutableBufferPointer<Float>(start: buf.baseAddress!.advanced(by: start), count: end - start)
                    BatchKernels.range_cosine_fused_512(query: query, candidates: candidates, range: start..<end, out: sub)
                    start = end
                }
            }
            return out
        } else {
            let chunks: [_Chunk] = try await computeProvider.parallelReduce(
                items: 0..<n,
                initial: [],
                minChunk: tunedMinChunk ?? minChunk(forDim: dim),
                { range in
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_cosine_fused_512(query: query, candidates: candidates, range: range, out: sub)
                    }
                    return [_Chunk(start: range.lowerBound, values: tmp)]
                },
                { $0 + $1 }
            )
            var out = [Float](repeating: 0, count: n)
            for ch in chunks { out.replaceSubrange(ch.start..<(ch.start + ch.values.count), with: ch.values) }
            return out
        }
    }

    private static func batchKernelDistances_cosineFused(
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        dim: Int,
        parallel: Bool,
        tunedMinChunk: Int? = nil
    ) async throws -> [Float] {
        let n = candidates.count
        if !parallel {
            var out = [Float](repeating: 0, count: n)
            out.withUnsafeMutableBufferPointer { buf in
                var start = 0
                let step = tunedMinChunk ?? minChunk(forDim: dim)
                while start < n {
                    let end = min(start + step, n)
                    let sub = UnsafeMutableBufferPointer<Float>(start: buf.baseAddress!.advanced(by: start), count: end - start)
                    BatchKernels.range_cosine_fused_768(query: query, candidates: candidates, range: start..<end, out: sub)
                    start = end
                }
            }
            return out
        } else {
            let chunks: [_Chunk] = try await computeProvider.parallelReduce(
                items: 0..<n,
                initial: [],
                minChunk: tunedMinChunk ?? minChunk(forDim: dim),
                { range in
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_cosine_fused_768(query: query, candidates: candidates, range: range, out: sub)
                    }
                    return [_Chunk(start: range.lowerBound, values: tmp)]
                },
                { $0 + $1 }
            )
            var out = [Float](repeating: 0, count: n)
            for ch in chunks { out.replaceSubrange(ch.start..<(ch.start + ch.values.count), with: ch.values) }
            return out
        }
    }

    private static func batchKernelDistances_cosineFused(
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        dim: Int,
        parallel: Bool,
        tunedMinChunk: Int? = nil
    ) async throws -> [Float] {
        let n = candidates.count
        if !parallel {
            var out = [Float](repeating: 0, count: n)
            out.withUnsafeMutableBufferPointer { buf in
                var start = 0
                let step = tunedMinChunk ?? minChunk(forDim: dim)
                while start < n {
                    let end = min(start + step, n)
                    let sub = UnsafeMutableBufferPointer<Float>(start: buf.baseAddress!.advanced(by: start), count: end - start)
                    BatchKernels.range_cosine_fused_1536(query: query, candidates: candidates, range: start..<end, out: sub)
                    start = end
                }
            }
            return out
        } else {
            let chunks: [_Chunk] = try await computeProvider.parallelReduce(
                items: 0..<n,
                initial: [],
                minChunk: tunedMinChunk ?? minChunk(forDim: dim),
                { range in
                    var tmp = [Float](repeating: 0, count: range.count)
                    tmp.withUnsafeMutableBufferPointer { sub in
                        BatchKernels.range_cosine_fused_1536(query: query, candidates: candidates, range: range, out: sub)
                    }
                    return [_Chunk(start: range.lowerBound, values: tmp)]
                },
                { $0 + $1 }
            )
            var out = [Float](repeating: 0, count: n)
            for ch in chunks { out.replaceSubrange(ch.start..<(ch.start + ch.values.count), with: ch.values) }
            return out
        }
    }

    /// Compute pairwise distance matrix
    ///
    /// - Parameters:
    ///   - setA: First vector set
    ///   - setB: Second vector set
    ///   - metric: Distance metric
    /// - Returns: Distance matrix [setA.count x setB.count]
    public static func distanceMatrix<V: VectorProtocol, M: DistanceMetric>(
        between setA: [V],
        and setB: [V],
        metric: M = EuclideanDistance()
    ) async throws -> [[Float]] where V.Scalar == Float, M.Scalar == Float {
        // If either set is empty, return an empty matrix with appropriate shape
        if setA.isEmpty || setB.isEmpty { return Array(repeating: [], count: setA.count) }
        let aDim = setA[0].scalarCount
        let bDim = setB[0].scalarCount
        // Validate dimensions within each set
        for v in setA where v.scalarCount != aDim {
            throw VectorError.dimensionMismatch(expected: aDim, actual: v.scalarCount)
        }
        for v in setB where v.scalarCount != bDim {
            throw VectorError.dimensionMismatch(expected: bDim, actual: v.scalarCount)
        }
        // Cross-set dimension match
        guard aDim == bDim else {
            throw VectorError.dimensionMismatch(expected: bDim, actual: aDim)
        }
        return try await computeProvider.parallelExecute(items: 0..<setA.count) { i in
            setB.map { metric.distance(setA[i], $0) }
        }
    }

    // MARK: - Vector Operations

    /// Compute centroid of vectors
    ///
    /// - Parameter vectors: Collection of vectors
    /// - Returns: Centroid vector
    public static func centroid<V: VectorProtocol & VectorFactory>(
        of vectors: [V]
    ) -> V where V.Scalar == Float {
        guard !vectors.isEmpty else {
            let dimension = vectors.first?.scalarCount ?? 0
            return try! V.create(from: [Float](repeating: 0, count: dimension))
        }

        let dimension = vectors.first!.scalarCount
        var sum = [Float](repeating: 0, count: dimension)

        // Sum all vectors
        for vector in vectors {
            let values = vector.toArray()
            sum = simdProvider.add(sum, values)
        }

        // Divide by count
        let scale = 1.0 / Float(vectors.count)
        let result = simdProvider.multiply(sum, by: scale)

        return try! V.create(from: result)
    }

    /// Normalize vectors to unit length
    ///
    /// - Parameter vectors: Vectors to normalize
    /// - Returns: Normalized vectors
    public static func normalize<V: VectorProtocol & VectorFactory>(
        _ vectors: [V]
    ) async throws -> [V] where V.Scalar == Float {
        try await computeProvider.parallelExecute(items: 0..<vectors.count) { i in
            let values = vectors[i].toArray()
            let normalized = simdProvider.normalize(values)
            return try! V.create(from: normalized)
        }
    }

    // MARK: - Statistics

    /// Compute statistics for vector collection
    ///
    /// - Parameter vectors: Vector collection
    /// - Returns: Statistical summary
    public static func statistics<V: VectorProtocol>(
        for vectors: [V]
    ) -> VectorStatistics where V.Scalar == Float {
        guard !vectors.isEmpty else {
            return VectorStatistics(
                count: 0,
                dimensions: vectors.first?.scalarCount ?? 0,
                mean: [],
                min: [],
                max: [],
                magnitudes: VectorStatistics.MagnitudeStats(min: 0, max: 0, mean: 0)
            )
        }

        let dimension = vectors.first!.scalarCount
        var mins = [Float](repeating: .infinity, count: dimension)
        var maxs = [Float](repeating: -.infinity, count: dimension)
        var sums = [Float](repeating: 0, count: dimension)
        var magnitudes: [Float] = []

        for vector in vectors {
            let values = vector.toArray()

            // Update component-wise stats
            for i in 0..<dimension {
                mins[i] = min(mins[i], values[i])
                maxs[i] = max(maxs[i], values[i])
                sums[i] += values[i]
            }

            // Track magnitude
            magnitudes.append(simdProvider.magnitude(values))
        }

        let count = Float(vectors.count)
        let mean = sums.map { $0 / count }

        let magMin = magnitudes.min() ?? 0
        let magMax = magnitudes.max() ?? 0
        let magMean = magnitudes.reduce(0, +) / count

        return VectorStatistics(
            count: vectors.count,
            dimensions: dimension,
            mean: mean,
            min: mins,
            max: maxs,
            magnitudes: VectorStatistics.MagnitudeStats(min: magMin, max: magMax, mean: magMean)
        )
    }
}

// MARK: - Result Types

/// Result of nearest neighbor search
public struct NearestNeighborResult: Sendable, Equatable {
    public let index: Int
    public let distance: Float

    public init(index: Int, distance: Float) {
        self.index = index
        self.distance = distance
    }
}

/// Statistical summary of vector collection
public struct VectorStatistics: Sendable {
    public let count: Int
    public let dimensions: Int
    public let mean: [Float]
    public let min: [Float]
    public let max: [Float]
    public let magnitudes: MagnitudeStats

    public struct MagnitudeStats: Sendable {
        public let min: Float
        public let max: Float
        public let mean: Float
    }
}
