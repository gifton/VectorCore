// VectorCore: Vector Factory
//
// Factory that picks a vector type from the dimension
//

import Foundation

/// Protocol for types that can be created by VectorFactory
///
/// VectorType provides a unified interface for all vector implementations
/// in VectorCore, allowing the factory to work with both fixed-size and
/// dynamic vectors uniformly.
public protocol VectorType: VectorProtocol where Scalar == Float {
    // VectorProtocol already provides:
    // - scalarCount
    // - toArray()
    // - dotProduct(_:)
    // - magnitude
    // - normalized()
    // - distance(to:)
    // - cosineSimilarity(to:)

    // Additional factory requirements
    init()
    init(repeating value: Float)
}

/// Factory that picks a vector type from the dimension: `Vector<DimN>` for 128, 256, 384, 512,
/// 768, 1536 and 3072, `DynamicVector` otherwise (see `vector(of:from:)`).
public enum VectorTypeFactory {

    /// Creates a strongly-typed vector with compile-time dimension checking.
    ///
    /// This method provides type-safe vector creation when the dimension is known
    /// at compile time, with compile-time dimension validation.
    ///
    /// ## Example Usage
    /// ```swift
    /// let values = Array(repeating: 0.5, count: 128)
    /// let vector = try VectorTypeFactory.create(Dim128.self, from: values)
    /// // vector is of type Vector<Dim128>
    /// ```
    ///
    /// - Parameters:
    ///   - type: The dimension type (e.g., `Dim128.self`)
    ///   - values: Array of Float values to initialize the vector
    /// - Returns: A strongly-typed Vector<D> instance
    /// - Throws: `VectorError.dimensionMismatch` if values.count != D.value
    public static func create<D: StaticDimension>(_ type: D.Type, from values: [Float]) throws -> Vector<D> {
        guard values.count == D.value else {
            throw VectorError.dimensionMismatch(expected: D.value, actual: values.count)
        }
        return try Vector<D>(values)
    }

    /// Create a vector of the specified dimension with given values
    ///
    /// - Parameters:
    ///   - dimension: The desired vector dimension
    ///   - values: Array of values (must match dimension)
    /// - Returns: `Vector<DimN>` for 128, 256, 384, 512, 768, 1536 and 3072; otherwise a `DynamicVector`
    /// - Throws: `.invalidDimension` for a negative dimension; `.dimensionMismatch` for a count mismatch.
    ///   Zero dimension with empty values is valid.
    public static func vector(of dimension: Int, from values: [Float]) throws -> any VectorType {
        guard dimension >= 0 else {
            throw VectorError.invalidDimension(dimension, reason: "Dimension must be nonnegative")
        }
        guard values.count == dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: values.count)
        }

        switch dimension {
        case 128:
            return try Vector<Dim128>(values)
        case 256:
            return try Vector<Dim256>(values)
        case 384:
            return try Vector<Dim384>(values)
        case 512:
            return try Vector<Dim512>(values)
        case 768:
            return try Vector<Dim768>(values)
        case 1536:
            return try Vector<Dim1536>(values)
        case 3072:
            return try Vector<Dim3072>(values)
        default:
            // For unsupported dimensions, use DynamicVector
            return DynamicVector(values)
        }
    }

    /// Create a random vector with values in the specified range
    ///
    /// - Parameters:
    ///   - dimension: The desired vector dimension
    ///   - range: Range for random values (default: -1...1)
    /// - Precondition: Dimension is nonnegative; zero produces an empty vector.
    /// - Returns: Vector with random values
    public static func random(
        dimension: Int,
        range: ClosedRange<Float> = -1...1
    ) -> any VectorType {
        guard dimension >= 0 else {
            fatalError("Dimension must be nonnegative")
        }
        let values = (0..<dimension).map { _ in Float.random(in: range) }
        return try! vector(of: dimension, from: values)
    }

    /// Create a zero vector of the specified dimension
    ///
    /// - Parameter dimension: The desired vector dimension
    /// - Precondition: Dimension is nonnegative; zero produces an empty vector.
    /// - Returns: Vector with all zeros
    public static func zeros(dimension: Int) -> any VectorType {
        guard dimension >= 0 else {
            fatalError("Dimension must be nonnegative")
        }
        switch dimension {
        case 128:
            return Vector<Dim128>()
        case 256:
            return Vector<Dim256>()
        case 384:
            return Vector<Dim384>()
        case 512:
            return Vector<Dim512>()
        case 768:
            return Vector<Dim768>()
        case 1536:
            return Vector<Dim1536>()
        case 3072:
            return Vector<Dim3072>()
        default:
            return DynamicVector(dimension: dimension)
        }
    }

    /// Create a vector of ones of the specified dimension
    ///
    /// - Parameter dimension: The desired vector dimension
    /// - Precondition: Dimension is nonnegative; zero produces an empty vector.
    /// - Returns: Vector with all ones
    public static func ones(dimension: Int) -> any VectorType {
        guard dimension >= 0 else {
            fatalError("Dimension must be nonnegative")
        }
        switch dimension {
        case 128:
            return Vector<Dim128>(repeating: 1)
        case 256:
            return Vector<Dim256>(repeating: 1)
        case 384:
            return Vector<Dim384>(repeating: 1)
        case 512:
            return Vector<Dim512>(repeating: 1)
        case 768:
            return Vector<Dim768>(repeating: 1)
        case 1536:
            return Vector<Dim1536>(repeating: 1)
        case 3072:
            return Vector<Dim3072>(repeating: 1)
        default:
            return DynamicVector(dimension: dimension, repeating: 1)
        }
    }

    /// Create a vector with a repeating pattern
    ///
    /// - Parameters:
    ///   - dimension: The desired vector dimension
    ///   - pattern: Function that generates value for each index
    /// - Precondition: Dimension is nonnegative; zero produces an empty vector.
    /// - Returns: Vector with pattern-generated values
    public static func withPattern(
        dimension: Int,
        pattern: (Int) -> Float
    ) -> any VectorType {
        guard dimension >= 0 else {
            fatalError("Dimension must be nonnegative")
        }
        let values = (0..<dimension).map(pattern)
        return try! vector(of: dimension, from: values)
    }

    // TODO: Fix normalized() on existential type
    // /// Create a normalized random vector (unit vector with random direction)
    // ///
    // /// - Parameter dimension: The desired vector dimension
    // /// - Returns: Normalized vector with random direction
    // public static func randomNormalized(dimension: Int) -> any VectorType {
    //     let random = self.random(dimension: dimension)
    //     // Try to normalize, return original if it fails (e.g., zero vector)
    //     if let result = try? random.normalized().get() {
    //         return result
    //     }
    //     return random
    // }

    /// The supported dimension nearest to a given approximate size
    ///
    /// - Parameter approximateSize: Desired approximate dimension
    /// - Returns: Nearest supported dimension, clamped at the endpoints; smaller width on ties.
    public static func optimalDimension(for approximateSize: Int) -> Int {
        let supportedDimensions = [128, 256, 384, 512, 768, 1536, 3072]

        // Clamp before subtraction so every Int input has bounded differences.
        if approximateSize <= 128 { return 128 }
        if approximateSize >= 3072 { return 3072 }

        // Ascending order and strict comparison preserve the smaller width on ties.
        // Find the closest supported dimension
        return supportedDimensions.min { abs($0 - approximateSize) < abs($1 - approximateSize) } ?? 384
    }

    /// Whether the factory maps `dimension` to a fixed-width `Vector<DimN>`
    ///
    /// `vector(of:from:)`, `zeros(dimension:)` and `ones(dimension:)` return `Vector<Dim128>`, `Vector<Dim256>`,
    /// `Vector<Dim384>`, `Vector<Dim512>`, `Vector<Dim768>`, `Vector<Dim1536>` or `Vector<Dim3072>` for exactly
    /// these seven dimensions and `DynamicVector` for every other nonnegative dimension. The factory never returns
    /// the separate `Vector384Optimized`, `Vector512Optimized`, `Vector768Optimized` or `Vector1536Optimized`
    /// types. No speed or storage comparison between `Vector<DimN>` and `DynamicVector` is measured in this
    /// repository (not verified); D02FactoryContractTests pins the concrete result types.
    ///
    /// - Parameter dimension: Dimension to check
    /// - Returns: true for 128, 256, 384, 512, 768, 1536 and 3072; false otherwise
    public static func isSupported(dimension: Int) -> Bool {
        switch dimension {
        case 128, 256, 384, 512, 768, 1536, 3072:
            return true
        default:
            return false
        }
    }

    /// Create a standard basis vector (one-hot encoded)
    ///
    /// - Parameters:
    ///   - dimension: The desired vector dimension
    ///   - index: Index of the non-zero element
    /// - Returns: Basis vector with 1.0 at the specified index
    /// - Throws: `.invalidDimension` unless dimension is positive, then `.indexOutOfBounds` for an invalid index.
    public static func basis(dimension: Int, index: Int) throws -> any VectorType {
        guard dimension > 0 else {
            throw VectorError.invalidDimension(dimension, reason: "Basis dimension must be positive")
        }
        guard index >= 0 && index < dimension else {
            throw VectorError.indexOutOfBounds(index: index, dimension: dimension)
        }

        switch dimension {
        case 128:
            return Vector<Dim128>.basis(axis: index)
        case 256:
            return Vector<Dim256>.basis(axis: index)
        case 384:
            return Vector<Dim384>.basis(axis: index)
        case 512:
            return Vector<Dim512>.basis(axis: index)
        case 768:
            return Vector<Dim768>.basis(axis: index)
        case 1536:
            return Vector<Dim1536>.basis(axis: index)
        case 3072:
            return Vector<Dim3072>.basis(axis: index)
        default:
            // Double-check bounds for safety in concurrent scenarios
            guard index < dimension else {
                throw VectorError.indexOutOfBounds(index: index, dimension: dimension)
            }
            var values = Array(repeating: Float(0), count: dimension)
            values[index] = 1.0
            return DynamicVector(values)
        }
    }

    /// Create a batch of vectors from a flat array
    ///
    /// - Parameters:
    ///   - dimension: Dimension of each vector
    ///   - values: Flat array of values (count must be multiple of dimension)
    /// - Returns: Array of vectors
    /// - Throws: `.invalidDimension` unless dimension is positive, then `.invalidData` for a nonmultiple count.
    ///   Empty values with a positive dimension produce an empty batch.
    public static func batch(dimension: Int, from values: [Float]) throws -> [any VectorType] {
        guard dimension > 0 else {
            throw VectorError.invalidDimension(dimension, reason: "Batch dimension must be positive")
        }
        guard values.count % dimension == 0 else {
            throw VectorError.invalidValues(
                indices: [],
                reason: "Values count (\(values.count)) must be multiple of dimension \(dimension)"
            )
        }

        let vectorCount = values.count / dimension
        var result: [any VectorType] = []
        result.reserveCapacity(vectorCount)

        for i in 0..<vectorCount {
            let start = i * dimension
            let end = start + dimension
            let vectorValues = Array(values[start..<end])
            result.append(try vector(of: dimension, from: vectorValues))
        }

        return result
    }
}

// MARK: - VectorType Extensions

// VectorType conformance is already declared in Vector.swift

// DynamicVector conformance is in DynamicVector.swift
