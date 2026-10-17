// VectorCore: Vector Operations Protocol
//
// Protocol extensions for core operations
//

import Foundation

/// Protocol for vectors supporting core operations
public protocol VectorCoreOperations: VectorProtocol {
    /// Element-wise minimum
    func min(_ other: Self) -> Self

    /// Element-wise maximum
    func max(_ other: Self) -> Self

    /// Find minimum element and index. Requires a nonempty vector; `Vector` stops with a precondition failure at
    /// width 0 (D02DegenerateWidthTests).
    func minElement() -> (value: Float, index: Int)

    /// Find maximum element and index. Requires a nonempty vector; `Vector` stops with a precondition failure at
    /// width 0 (D02DegenerateWidthTests).
    func maxElement() -> (value: Float, index: Int)

    /// Clamp values to range
    func clamped(to range: ClosedRange<Float>) -> Self

    /// Linear interpolation
    func lerp(to other: Self, t: Float) -> Self

    /// Unclamped linear interpolation
    func lerpUnclamped(to other: Self, t: Float) -> Self

    /// Smooth interpolation
    func smoothstep(to other: Self, t: Float) -> Self

    /// Element-wise absolute value
    func absoluteValue() -> Self

    /// Element-wise square root
    func squareRoot() -> Self
}

// Every Vector<D> conforms, for every static dimension (D02 removed the storage gate).
extension Vector: VectorCoreOperations {}

// Convenience operators
infix operator .< : ComparisonPrecedence
infix operator .> : ComparisonPrecedence
