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

    /// Find minimum element and index
    func minElement() -> (value: Float, index: Int)

    /// Find maximum element and index
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

// Make Vector conform when storage supports operations
extension Vector: VectorCoreOperations where D.Storage: VectorStorageOperations {}

// Convenience operators
infix operator .< : ComparisonPrecedence
infix operator .> : ComparisonPrecedence
