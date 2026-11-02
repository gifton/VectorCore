import Foundation
import simd

// MARK: - Cosine Batch Helpers

internal enum CosineBatchHelpers {

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
            out[i] = dot.isFinite ? 1.0 - max(-1.0, min(1.0, dot)) : .nan
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
            out[i] = dot.isFinite ? 1.0 - max(-1.0, min(1.0, dot)) : .nan
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
            out[i] = dot.isFinite ? 1.0 - max(-1.0, min(1.0, dot)) : .nan
        }
    }

    // Build normalized finite nonzero copies; retain zero/nonfinite inputs on checked failure.
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
