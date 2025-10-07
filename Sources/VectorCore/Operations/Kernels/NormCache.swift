import Foundation
import simd

// MARK: - Norm Cache (per dimension)

internal struct NormCache512: Sendable {
    public let count: Int
    public let magnitudes: [Float]

    public static func build(candidates: [Vector512Optimized]) -> NormCache512 {
        let n = candidates.count
        var mags = [Float](repeating: 0, count: n)
        for i in 0..<n {
            // |b| = sqrt(dot(b,b))
            let sumBB = DotKernels.dot512(candidates[i], candidates[i])
            mags[i] = sumBB > 0 ? sqrt(sumBB) : 0
        }
        return NormCache512(count: n, magnitudes: mags)
    }
}

internal struct NormCache768: Sendable {
    public let count: Int
    public let magnitudes: [Float]

    public static func build(candidates: [Vector768Optimized]) -> NormCache768 {
        let n = candidates.count
        var mags = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let sumBB = DotKernels.dot768(candidates[i], candidates[i])
            mags[i] = sumBB > 0 ? sqrt(sumBB) : 0
        }
        return NormCache768(count: n, magnitudes: mags)
    }
}

internal struct NormCache1536: Sendable {
    public let count: Int
    public let magnitudes: [Float]

    public static func build(candidates: [Vector1536Optimized]) -> NormCache1536 {
        let n = candidates.count
        var mags = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let sumBB = DotKernels.dot1536(candidates[i], candidates[i])
            mags[i] = sumBB > 0 ? sqrt(sumBB) : 0
        }
        return NormCache1536(count: n, magnitudes: mags)
    }
}

// MARK: - Cosine Batch Helpers

internal enum CosineBatchHelpers {

    // Using precomputed magnitudes (NormCache)
    @inlinable
    public static func batch_cosine_withNormCache_512(
        query: Vector512Optimized,
        candidates: [Vector512Optimized],
        norms: NormCache512,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(norms.count == candidates.count)
        precondition(out.count >= candidates.count)
        // Precompute sumAA (no sqrt)
        let sumAA = DotKernels.dot512(query, query)
        for i in 0..<candidates.count {
            let sumBB = norms.magnitudes[i] * norms.magnitudes[i]
            let dot = DotKernels.dot512(query, candidates[i])
            out[i] = CosineKernels.calculateCosineDistance(dot: dot, sumAA: sumAA, sumBB: sumBB)
        }
    }

    @inlinable
    public static func batch_cosine_withNormCache_768(
        query: Vector768Optimized,
        candidates: [Vector768Optimized],
        norms: NormCache768,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(norms.count == candidates.count)
        precondition(out.count >= candidates.count)
        let sumAA = DotKernels.dot768(query, query)
        for i in 0..<candidates.count {
            let sumBB = norms.magnitudes[i] * norms.magnitudes[i]
            let dot = DotKernels.dot768(query, candidates[i])
            out[i] = CosineKernels.calculateCosineDistance(dot: dot, sumAA: sumAA, sumBB: sumBB)
        }
    }

    @inlinable
    public static func batch_cosine_withNormCache_1536(
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized],
        norms: NormCache1536,
        out: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(norms.count == candidates.count)
        precondition(out.count >= candidates.count)
        let sumAA = DotKernels.dot1536(query, query)
        for i in 0..<candidates.count {
            let sumBB = norms.magnitudes[i] * norms.magnitudes[i]
            let dot = DotKernels.dot1536(query, candidates[i])
            out[i] = CosineKernels.calculateCosineDistance(dot: dot, sumAA: sumAA, sumBB: sumBB)
        }
    }

    // Pre-normalized fast path: 1 − dot
    @inlinable
    public static func batch_cosine_preNormalized_512(
        queryNormalized: Vector512Optimized,
        candidatesNormalized: [Vector512Optimized],
        out: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(out.count >= candidatesNormalized.count)
        for i in 0..<candidatesNormalized.count {
            let dot = DotKernels.dot512(queryNormalized, candidatesNormalized[i])
            out[i] = 1.0 - max(-1.0, min(1.0, dot))
        }
    }

    @inlinable
    public static func batch_cosine_preNormalized_768(
        queryNormalized: Vector768Optimized,
        candidatesNormalized: [Vector768Optimized],
        out: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(out.count >= candidatesNormalized.count)
        for i in 0..<candidatesNormalized.count {
            let dot = DotKernels.dot768(queryNormalized, candidatesNormalized[i])
            out[i] = 1.0 - max(-1.0, min(1.0, dot))
        }
    }

    @inlinable
    public static func batch_cosine_preNormalized_1536(
        queryNormalized: Vector1536Optimized,
        candidatesNormalized: [Vector1536Optimized],
        out: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(out.count >= candidatesNormalized.count)
        for i in 0..<candidatesNormalized.count {
            let dot = DotKernels.dot1536(queryNormalized, candidatesNormalized[i])
            out[i] = 1.0 - max(-1.0, min(1.0, dot))
        }
    }

    // Build normalized copies (unit length). If magnitude is zero, keep original zero vector.
    public static func buildNormalized_512(candidates: [Vector512Optimized]) -> [Vector512Optimized] {
        candidates.map { (try? $0.normalized().get()) ?? $0 }
    }
    public static func buildNormalized_768(candidates: [Vector768Optimized]) -> [Vector768Optimized] {
        candidates.map { (try? $0.normalized().get()) ?? $0 }
    }
    public static func buildNormalized_1536(candidates: [Vector1536Optimized]) -> [Vector1536Optimized] {
        candidates.map { (try? $0.normalized().get()) ?? $0 }
    }
}
