// VectorCore: Core Vector Operations
//
// Element-wise operations, clamping, and interpolation
//

import Foundation

// MARK: - Element-wise Min/Max Operations

public extension Vector where D.Storage: VectorStorageOperations {
    
    /// Compute element-wise minimum between two vectors
    /// - Parameter other: The other vector
    /// - Returns: A new vector with the minimum values
    /// - Complexity: O(n) where n is the dimension
    @inlinable
    func min(_ other: Vector<D>) -> Vector<D> {
        let a = Array(self)
        let b = Array(other)
        let result = Operations.simdProvider.elementWiseMin(a, b)
        return try! Vector<D>(result)
    }
    
    /// Compute element-wise maximum between two vectors
    /// - Parameter other: The other vector
    /// - Returns: A new vector with the maximum values
    /// - Complexity: O(n) where n is the dimension
    @inlinable
    func max(_ other: Vector<D>) -> Vector<D> {
        let a = Array(self)
        let b = Array(other)
        let result = Operations.simdProvider.elementWiseMax(a, b)
        return try! Vector<D>(result)
    }
    
    /// Find the minimum element in the vector
    /// - Returns: The minimum value and its index
    /// - Complexity: O(n) where n is the dimension
    @inlinable
    func minElement() -> (value: Float, index: Int) {
        let array = Array(self)
        let index = Operations.simdProvider.minIndex(array)
        return (array[index], index)
    }
    
    /// Find the maximum element in the vector
    /// - Returns: The maximum value and its index
    /// - Complexity: O(n) where n is the dimension
    @inlinable
    func maxElement() -> (value: Float, index: Int) {
        let array = Array(self)
        let index = Operations.simdProvider.maxIndex(array)
        return (array[index], index)
    }
}

// MARK: - Clamp Operation

public extension Vector where D.Storage: VectorStorageOperations {
    
    /// Clamp all elements to a given range
    /// - Parameter range: The range to clamp values to
    /// - Returns: A new vector with clamped values
    /// - Complexity: O(n) where n is the dimension
    @inlinable
    func clamped(to range: ClosedRange<Float>) -> Vector<D> {
        let array = Array(self)
        let clamped = Operations.simdProvider.clip(array, min: range.lowerBound, max: range.upperBound)
        return try! Vector<D>(clamped)
    }
    
    /// Clamp all elements to a given range in place
    /// - Parameter range: The range to clamp values to
    /// - Complexity: O(n) where n is the dimension
    @inlinable
    mutating func clamp(to range: ClosedRange<Float>) {
        let array = Array(self)
        let clamped = Operations.simdProvider.clip(array, min: range.lowerBound, max: range.upperBound)
        self = try! Vector<D>(clamped)
    }
}

// MARK: - Linear Interpolation

public extension Vector where D.Storage: VectorStorageOperations {
    
    /// Linearly interpolate between two vectors
    /// - Parameters:
    ///   - other: The target vector
    ///   - t: The interpolation parameter [0, 1]
    /// - Returns: The interpolated vector
    /// - Complexity: O(n) where n is the dimension
    /// - Note: t is clamped to [0, 1] for safety
    @inlinable
    func lerp(to other: Vector<D>, t: Float) -> Vector<D> {
        // Clamp t to [0, 1]
        let clampedT = Swift.max(0, Swift.min(1, t))
        
        // Handle edge cases
        if clampedT == 0 { return self }
        if clampedT == 1 { return other }
        
        // Compute: result = self * (1 - t) + other * t
        // Which is equivalent to: result = self + (other - self) * t
        let diff = other - self
        return self + diff * clampedT
    }
    
    /// Linearly interpolate between two vectors (unclamped)
    /// - Parameters:
    ///   - other: The target vector
    ///   - t: The interpolation parameter (any value)
    /// - Returns: The interpolated vector
    /// - Complexity: O(n) where n is the dimension
    /// - Note: t can be any value, allowing extrapolation
    @inlinable
    func lerpUnclamped(to other: Vector<D>, t: Float) -> Vector<D> {
        // Handle edge cases for performance
        if t == 0 { return self }
        if t == 1 { return other }
        
        // Compute: result = self + (other - self) * t
        let diff = other - self
        return self + diff * t
    }
    
    /// Smoothly interpolate between two vectors using smoothstep
    /// - Parameters:
    ///   - other: The target vector
    ///   - t: The interpolation parameter [0, 1]
    /// - Returns: The interpolated vector with smooth acceleration/deceleration
    /// - Complexity: O(n) where n is the dimension
    @inlinable
    func smoothstep(to other: Vector<D>, t: Float) -> Vector<D> {
        // Clamp t to [0, 1]
        let clampedT = Swift.max(0, Swift.min(1, t))
        
        // Smoothstep function: 3t² - 2t³
        let smoothT = clampedT * clampedT * (3 - 2 * clampedT)
        
        return lerp(to: other, t: smoothT)
    }
}

// MARK: - Element-wise Operations

public extension Vector where D.Storage: VectorStorageOperations {
    
    /// Element-wise absolute value
    /// - Returns: A new vector with absolute values
    /// - Complexity: O(n) where n is the dimension
    @inlinable
    func absoluteValue() -> Vector<D> {
        let array = Array(self)
        let absArray = Operations.simdProvider.abs(array)
        return try! Vector<D>(absArray)
    }
    
    /// Element-wise square root
    /// - Returns: A new vector with square root of each element
    /// - Complexity: O(n) where n is the dimension
    /// - Note: Negative values will produce NaN
    @inlinable
    func squareRoot() -> Vector<D> {
        let array = Array(self)
        let sqrtArray = Operations.simdProvider.sqrt(array)
        return try! Vector<D>(sqrtArray)
    }
}

// MARK: - Operator Overloads

public extension Vector where D.Storage: VectorStorageOperations {
    
    /// Element-wise minimum operator
    @inlinable
    static func .< (lhs: Vector<D>, rhs: Vector<D>) -> Vector<D> {
        lhs.min(rhs)
    }
    
    /// Element-wise maximum operator
    @inlinable
    static func .> (lhs: Vector<D>, rhs: Vector<D>) -> Vector<D> {
        lhs.max(rhs)
    }
}

// MARK: - DynamicVector Extensions

public extension DynamicVector {
    
    /// Compute element-wise minimum between two vectors
    /// - Parameter other: The other vector
    /// - Returns: A new vector with the minimum values
    /// - Throws: VectorError.dimensionMismatch if dimensions don't match
    func min(_ other: DynamicVector) throws -> DynamicVector {
        guard dimension == other.dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: other.dimension)
        }
        
        // Create arrays, perform operation, then create result
        let arr1 = Array(self)
        let arr2 = Array(other)
        var result = [Float](repeating: 0, count: dimension)
        
        result = Operations.simdProvider.elementWiseMin(arr1, arr2)
        
        return DynamicVector(result)
    }
    
    /// Compute element-wise maximum between two vectors
    /// - Parameter other: The other vector
    /// - Returns: A new vector with the maximum values
    /// - Throws: VectorError.dimensionMismatch if dimensions don't match
    func max(_ other: DynamicVector) throws -> DynamicVector {
        guard dimension == other.dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: other.dimension)
        }
        
        // Create arrays, perform operation, then create result
        let arr1 = Array(self)
        let arr2 = Array(other)
        var result = [Float](repeating: 0, count: dimension)
        
        result = Operations.simdProvider.elementWiseMax(arr1, arr2)
        
        return DynamicVector(result)
    }
    
    /// Clamp all elements to a given range
    /// - Parameter range: The range to clamp values to
    /// - Returns: A new vector with clamped values
    func clamped(to range: ClosedRange<Float>) -> DynamicVector {
        // Create array, perform operation, then create result
        let arr = Array(self)
        var result = [Float](repeating: 0, count: dimension)
        let low = range.lowerBound
        let high = range.upperBound
        
        result = Operations.simdProvider.clip(arr, min: low, max: high)
        
        return DynamicVector(result)
    }
    
    /// Linearly interpolate between two vectors
    /// - Parameters:
    ///   - other: The target vector
    ///   - t: The interpolation parameter [0, 1]
    /// - Returns: The interpolated vector
    /// - Throws: VectorError.dimensionMismatch if dimensions don't match
    func lerp(to other: DynamicVector, t: Float) throws -> DynamicVector {
        guard dimension == other.dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: other.dimension)
        }
        
        // Clamp t to [0, 1]
        let clampedT = Swift.max(0, Swift.min(1, t))
        
        // Handle edge cases
        if clampedT == 0 { return self }
        if clampedT == 1 { return other }
        
        // Compute: result = self + (other - self) * t
        let diff = other - self
        return self + diff * clampedT
    }
}