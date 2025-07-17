// VectorCore: Updated Vector Protocol
//
// Base protocol for vector operations without SIMD requirement
//

import Foundation

/// Base protocol for vector operations.
///
/// `BaseVectorProtocol` defines the fundamental interface for all vector types
/// in VectorCore. It provides the minimal set of requirements for vector-like
/// types while maintaining flexibility for different implementations.
///
/// ## Conforming Types
/// - `Vector<D>`: Strongly-typed vectors with compile-time dimensions
/// - `DynamicVector`: Runtime-determined dimension vectors
///
/// ## Example Implementation
/// ```swift
/// struct MyVector: BaseVectorProtocol {
///     typealias Scalar = Float
///     static let dimensions = 128
///     
///     private let storage: [Float]
///     var scalarCount: Int { storage.count }
///     
///     init(from array: [Float]) {
///         precondition(array.count == Self.dimensions)
///         self.storage = array
///     }
///     
///     func toArray() -> [Float] { storage }
///     subscript(index: Int) -> Float { storage[index] }
/// }
/// ```
public protocol BaseVectorProtocol: Sendable {
    /// The scalar type for vector elements (typically Float or Double).
    associatedtype Scalar: BinaryFloatingPoint
    
    /// Number of dimensions in the vector.
    ///
    /// For fixed-size vectors, this is a compile-time constant.
    /// For dynamic vectors, return 0 or use runtime checks.
    static var dimensions: Int { get }
    
    /// Number of elements in this vector instance.
    ///
    /// For fixed-size vectors, this equals `dimensions`.
    /// For dynamic vectors, this returns the actual size.
    var scalarCount: Int { get }
    
    /// Initialize from an array of scalars.
    ///
    /// - Parameter array: Source array with exactly `dimensions` elements
    /// - Precondition: array.count must equal expected dimensions
    init(from array: [Scalar])
    
    /// Convert vector to array representation.
    ///
    /// - Returns: Array containing all vector elements in order
    /// - Complexity: O(n) where n is the number of elements
    func toArray() -> [Scalar]
    
    /// Access vector elements by index.
    ///
    /// - Parameter index: Zero-based element index
    /// - Returns: Element value at the specified index
    /// - Precondition: 0 ≤ index < scalarCount
    subscript(index: Int) -> Scalar { get }
}

/// Extended vector protocol with mathematical operations.
///
/// `ExtendedVectorProtocol` builds upon `BaseVectorProtocol` to provide
/// common mathematical operations for vectors. All operations are optimized
/// for Float scalar types and leverage hardware acceleration where available.
///
/// ## Operations Provided
/// - **Dot Product**: Inner product of two vectors
/// - **Magnitude**: Euclidean norm (L2 norm)
/// - **Normalization**: Unit vector creation
/// - **Distance**: Euclidean distance between vectors
/// - **Similarity**: Cosine similarity measurement
///
/// ## Performance Notes
/// Implementations should use SIMD operations when available.
/// The Accelerate framework provides optimized implementations.
///
/// ## Example Usage
/// ```swift
/// let v1 = Vector<Dim128>.random(in: -1...1)
/// let v2 = Vector<Dim128>.random(in: -1...1)
/// 
/// let similarity = v1.cosineSimilarity(to: v2)
/// let distance = v1.distance(to: v2)
/// let unitVector = v1.normalized()
/// ```
public protocol ExtendedVectorProtocol: BaseVectorProtocol where Scalar == Float {
    /// Compute dot product with another vector.
    ///
    /// The dot product is defined as: a·b = Σ(aᵢ × bᵢ)
    ///
    /// - Parameter other: Vector to compute dot product with
    /// - Returns: Scalar dot product result
    /// - Complexity: O(n) where n is the number of dimensions
    func dotProduct(_ other: Self) -> Float
    
    /// Magnitude (L2 norm) of the vector.
    ///
    /// Computed as: ||v|| = √(Σvᵢ²)
    ///
    /// - Returns: Non-negative magnitude
    /// - Complexity: O(n) computation, may be cached
    var magnitude: Float { get }
    
    /// Create a normalized (unit) copy of the vector.
    ///
    /// Returns v/||v|| where ||v|| is the magnitude.
    /// Zero vectors return themselves unchanged.
    ///
    /// - Returns: New vector with magnitude 1.0 (or 0.0 for zero vectors)
    /// - Complexity: O(n) for computation
    func normalized() -> Self
    
    /// Compute Euclidean distance to another vector.
    ///
    /// Distance is defined as: d(a,b) = ||a - b||₂
    ///
    /// - Parameter other: Target vector for distance calculation
    /// - Returns: Non-negative distance value
    /// - Complexity: O(n) where n is the number of dimensions
    func distance(to other: Self) -> Float
    
    /// Compute cosine similarity with another vector.
    ///
    /// Cosine similarity is defined as: cos(θ) = (a·b)/(||a||×||b||)
    ///
    /// - Parameter other: Vector to compare with
    /// - Returns: Similarity in range [-1, 1] where:
    ///   - 1.0 = identical direction
    ///   - 0.0 = orthogonal
    ///   - -1.0 = opposite direction
    /// - Note: Returns 0.0 if either vector has zero magnitude
    /// - Complexity: O(n) where n is the number of dimensions
    func cosineSimilarity(to other: Self) -> Float
}

// MARK: - Helper Extensions

extension BaseVectorProtocol {
    /// Validate that an array has the correct dimensions for this vector type.
    ///
    /// - Parameter array: Array to validate
    /// - Returns: true if array.count matches expected dimensions
    ///
    /// ## Example
    /// ```swift
    /// let values = [1.0, 2.0, 3.0]
    /// if Vector<Dim3>.validate(values) {
    ///     let vector = Vector<Dim3>(values)
    /// }
    /// ```
    public static func validate(_ array: [Scalar]) -> Bool {
        array.count == dimensions
    }
    
    /// Create a vector from an array if dimensions match.
    ///
    /// This is a safe initialization method that returns nil
    /// instead of failing preconditions when dimensions don't match.
    ///
    /// - Parameter array: Source array of values
    /// - Returns: Vector instance if dimensions match, nil otherwise
    ///
    /// ## Example
    /// ```swift
    /// let values = loadValuesFromFile()
    /// if let vector = Vector<Dim128>.create(from: values) {
    ///     // Successfully created vector
    /// } else {
    ///     // Handle dimension mismatch
    /// }
    /// ```
    public static func create(from array: [Scalar]) -> Self? {
        guard validate(array) else { return nil }
        return Self(from: array)
    }
}