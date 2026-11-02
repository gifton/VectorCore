// CPU normalization uses widened Float accumulation before output writes.
import Foundation
import simd

public enum NormalizeKernels {
    /// Rounds the compensated Double sum once; infinity can be a valid Float result.
    @usableFromInline
    static func magnitudeSquared(storage: ContiguousArray<SIMD4<Float>>, laneCount: Int) -> Float {
        Float(NormalizationCore.floatSquares(count: laneCount * 4) { storage[$0 / 4][$0 % 4] })
    }

    /// Rounds the widened norm once, independently of normalization's range.
    @usableFromInline
    static func magnitude(storage: ContiguousArray<SIMD4<Float>>, laneCount: Int) -> Float {
        Float(NormalizationCore.floatSquares(count: laneCount * 4) { storage[$0 / 4][$0 % 4] }.squareRoot())
    }

    @usableFromInline
    static func normalizeStorage(_ storage: inout ContiguousArray<SIMD4<Float>>) -> Result<Void, VectorError> {
        let calculation = NormalizationCore.plan(count: storage.count * 4) { storage[$0 / 4][$0 % 4] }
        guard case .success(let plan) = calculation else {
            if case .failure(let error) = calculation { return .failure(error) }
            fatalError("Unreachable normalization result")
        }
        for i in storage.indices {
            for lane in 0..<4 { storage[i][lane] = plan.coordinate(storage[i][lane]) }
        }
        return .success(())
    }

    @usableFromInline
    static func normalize384_inplace(_ value: inout Vector384Optimized) -> Result<Void, VectorError> {
        normalizeStorage(&value.storage)
    }

    @usableFromInline
    static func normalized384(_ value: Vector384Optimized) -> Result<Vector384Optimized, VectorError> {
        var copy = value
        switch normalize384_inplace(&copy) {
        case .success: return .success(copy)
        case .failure(let error): return .failure(error)
        }
    }

    /// Requires finite nonzero input; enforced in Debug and normal Release.
    @usableFromInline
    static func normalizedUnchecked384(_ value: Vector384Optimized) -> Vector384Optimized {
        var copy = value
        guard case .success = normalize384_inplace(&copy) else {
            fatalError("normalizedUnchecked: requires finite nonzero input and representable calculation")
        }
        return copy
    }

    @usableFromInline
    static func normalize512_inplace(_ value: inout Vector512Optimized) -> Result<Void, VectorError> {
        normalizeStorage(&value.storage)
    }

    @usableFromInline
    static func normalized512(_ value: Vector512Optimized) -> Result<Vector512Optimized, VectorError> {
        var copy = value
        switch normalize512_inplace(&copy) {
        case .success: return .success(copy)
        case .failure(let error): return .failure(error)
        }
    }

    /// Requires finite nonzero input; enforced in Debug and normal Release.
    @usableFromInline
    static func normalizedUnchecked512(_ value: Vector512Optimized) -> Vector512Optimized {
        var copy = value
        guard case .success = normalize512_inplace(&copy) else {
            fatalError("normalizedUnchecked: requires finite nonzero input and representable calculation")
        }
        return copy
    }

    @usableFromInline
    static func normalize768_inplace(_ value: inout Vector768Optimized) -> Result<Void, VectorError> {
        normalizeStorage(&value.storage)
    }

    @usableFromInline
    static func normalized768(_ value: Vector768Optimized) -> Result<Vector768Optimized, VectorError> {
        var copy = value
        switch normalize768_inplace(&copy) {
        case .success: return .success(copy)
        case .failure(let error): return .failure(error)
        }
    }

    /// Requires finite nonzero input; enforced in Debug and normal Release.
    @usableFromInline
    static func normalizedUnchecked768(_ value: Vector768Optimized) -> Vector768Optimized {
        var copy = value
        guard case .success = normalize768_inplace(&copy) else {
            fatalError("normalizedUnchecked: requires finite nonzero input and representable calculation")
        }
        return copy
    }

    @usableFromInline
    static func normalize1536_inplace(_ value: inout Vector1536Optimized) -> Result<Void, VectorError> {
        normalizeStorage(&value.storage)
    }

    @usableFromInline
    static func normalized1536(_ value: Vector1536Optimized) -> Result<Vector1536Optimized, VectorError> {
        var copy = value
        switch normalize1536_inplace(&copy) {
        case .success: return .success(copy)
        case .failure(let error): return .failure(error)
        }
    }

    /// Requires finite nonzero input; enforced in Debug and normal Release.
    @usableFromInline
    static func normalizedUnchecked1536(_ value: Vector1536Optimized) -> Vector1536Optimized {
        var copy = value
        guard case .success = normalize1536_inplace(&copy) else {
            fatalError("normalizedUnchecked: requires finite nonzero input and representable calculation")
        }
        return copy
    }

    /// Normalize naturally Float-aligned storage after validating every coordinate.
    /// Requires nonnegative dimension and finite nonzero nonempty input; dimension 0 accesses nothing.
    /// Exact in-place mutation is supported. The pointer must own dimension initialized Floats.
    @inlinable
    public static func normalizeUnchecked(_ buffer: UnsafeMutablePointer<Float>, dimension: Int) {
        guard dimension >= 0 else { fatalError("normalizeUnchecked: dimension must be nonnegative") }
        guard dimension > 0 else { return }
        guard case .success(let plan) = NormalizationCore.plan(count: dimension, get: { buffer[$0] }) else {
            fatalError("normalizeUnchecked: requires finite nonzero input and representable calculation")
        }
        for i in 0..<dimension { buffer[i] = plan.coordinate(buffer[i]) }
    }
}
