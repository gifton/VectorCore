//
//  ReductionNaN.swift
//  VectorCore
//
//  NaN helpers for the min/max-family reductions (T16 reduction NaN contract)
//

/// Helpers for the T16 reduction NaN contract: a NaN input element makes a min/max-family reduction, statistic or
/// element-wise min/max NaN.
///
/// `Swift.max(x, y)` and `Swift.min(x, y)` keep a NaN only in their FIRST argument (`max(1, .nan) == 1`), so a fold
/// written with them alone drops a NaN depending on its position. `propagatingMax`/`propagatingMin` check both
/// operands and otherwise evaluate the unchanged `Swift.max`/`Swift.min` expression, so non-NaN operands give the
/// same bits as `Swift.max`/`Swift.min`, ties and signed zeros included (follows from the expression). The result for
/// a NaN operand is `.nan`; its payload is unspecified.
///
/// `containsNaN` is the explicit scan the Accelerate reductions use (T16 ruling C1): the vDSP header describes
/// `maxv`/`minv`/`maxmgv` only as "the greatest/least value" and says nothing about NaN; on the T16 phase 1 host
/// `maxv`/`minv` dropped a NaN in their scalar tail and for counts up to 3.
///
/// Pinned by T16ReductionExceptionalTableTests.everyRowFollowsTheContract. This is an enum, not a `SIMD` or pointer
/// extension member named `min`/`max` (Scripts/ci/tests/test_no_stdlib_member_shadows.py).
@usableFromInline
internal enum ReductionNaN {
    @inlinable @inline(__always)
    static func propagatingMax<T: FloatingPoint>(_ x: T, _ y: T) -> T {
        (x.isNaN || y.isNaN) ? .nan : Swift.max(x, y)
    }

    @inlinable @inline(__always)
    static func propagatingMin<T: FloatingPoint>(_ x: T, _ y: T) -> T {
        (x.isNaN || y.isNaN) ? .nan : Swift.min(x, y)
    }

    /// True when any of the `count` elements at `a` is NaN (`x != x` holds only for NaN).
    @inlinable
    static func containsNaN(_ a: UnsafePointer<Float>, count: Int) -> Bool {
        var i = 0
        var lanes = SIMDMask<SIMD8<Int32>>(repeating: false)
        while i + 8 <= count {
            let v = SIMD8<Float>(a[i], a[i + 1], a[i + 2], a[i + 3], a[i + 4], a[i + 5], a[i + 6], a[i + 7])
            lanes = lanes .| (v .!= v)
            i += 8
        }
        var tail = false
        while i < count {
            tail = tail || a[i].isNaN
            i += 1
        }
        return any(lanes) || tail
    }

    /// True when any of the `count` elements at `a` is NaN (`x != x` holds only for NaN).
    @inlinable
    static func containsNaN(_ a: UnsafePointer<Double>, count: Int) -> Bool {
        var i = 0
        var lanes = SIMDMask<SIMD8<Int64>>(repeating: false)
        while i + 8 <= count {
            let v = SIMD8<Double>(a[i], a[i + 1], a[i + 2], a[i + 3], a[i + 4], a[i + 5], a[i + 6], a[i + 7])
            lanes = lanes .| (v .!= v)
            i += 8
        }
        var tail = false
        while i < count {
            tail = tail || a[i].isNaN
            i += 1
        }
        return any(lanes) || tail
    }
}
