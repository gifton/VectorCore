// VectorCore: Dimension Protocol and Types
//
// Defines compile-time and runtime dimension specifications
//

import Foundation

/// Protocol for compile-time dimension specification
public protocol Dimension {
    /// The number of elements in vectors of this dimension
    static var value: Int { get }
    
    /// The storage type optimized for this dimension
    associatedtype Storage: VectorStorage
}

// MARK: - Standard Dimensions

/// A 32-dimensional vector type optimized for small embeddings.
///
/// `Dim32` provides compile-time dimension safety for 32-element vectors,
/// commonly used for simple feature vectors and low-dimensional embeddings.
/// Uses optimized storage for small vectors with minimal memory overhead.
///
/// ## Example Usage
/// ```swift
/// let embedding = Vector<Dim32>.random(in: -1...1)
/// let features = Vector<Dim32>.zeros()
/// ```
public struct Dim32: Dimension {
    public static let value = 32
    public typealias Storage = Storage32
}

/// A 64-dimensional vector type for compact embeddings.
///
/// `Dim64` provides compile-time dimension safety for 64-element vectors,
/// suitable for word embeddings and small feature representations.
/// Balances memory efficiency with expressiveness.
///
/// ## Example Usage
/// ```swift
/// let wordVector = Vector<Dim64>.ones()
/// let compressed = Vector<Dim64>.random(in: 0...1)
/// ```
public struct Dim64: Dimension {
    public static let value = 64
    public typealias Storage = Storage64
}

/// A 128-dimensional vector type for medium-sized embeddings.
///
/// `Dim128` provides compile-time dimension safety for 128-element vectors,
/// commonly used for face embeddings, audio features, and medium-complexity
/// representations. Offers good balance between expressiveness and efficiency.
///
/// ## Example Usage
/// ```swift
/// let faceEmbedding = Vector<Dim128>.normalized()
/// let audioFeatures = Vector<Dim128>.random(in: -1...1)
/// ```
public struct Dim128: Dimension {
    public static let value = 128
    public typealias Storage = Storage128
}

/// A 256-dimensional vector type for standard embeddings.
///
/// `Dim256` provides compile-time dimension safety for 256-element vectors,
/// ideal for image features, standard NLP embeddings, and general-purpose
/// vector representations. Popular choice for many ML applications.
///
/// ## Example Usage
/// ```swift
/// let imageFeatures = Vector<Dim256>.zeros()
/// let textEmbedding = Vector<Dim256>.random(in: -1...1)
/// ```
public struct Dim256: Dimension {
    public static let value = 256
    public typealias Storage = Storage256
}

/// A 512-dimensional vector type for rich embeddings.
///
/// `Dim512` provides compile-time dimension safety for 512-element vectors,
/// commonly used for sentence embeddings, BERT-style models, and complex
/// feature representations. Standard dimension for many transformer models.
///
/// ## Example Usage
/// ```swift
/// let sentenceEmbedding = Vector<Dim512>.ones()
/// let bertOutput = Vector<Dim512>.normalized()
/// ```
public struct Dim512: Dimension {
    public static let value = 512
    public typealias Storage = Storage512
}

/// A 768-dimensional vector type for transformer embeddings.
///
/// `Dim768` provides compile-time dimension safety for 768-element vectors,
/// the standard dimension for BERT base models and many transformer
/// architectures. Optimized for large-scale NLP applications.
///
/// ## Example Usage
/// ```swift
/// let bertEmbedding = Vector<Dim768>.zeros()
/// let transformerOutput = Vector<Dim768>.random(in: -1...1)
/// ```
public struct Dim768: Dimension {
    public static let value = 768
    public typealias Storage = Storage768
}

/// A 1536-dimensional vector type for large embeddings.
///
/// `Dim1536` provides compile-time dimension safety for 1536-element vectors,
/// commonly used for OpenAI embeddings and large language model outputs.
/// Optimized for high-dimensional representations with SIMD support.
///
/// ## Example Usage
/// ```swift
/// let gptEmbedding = Vector<Dim1536>.normalized()
/// let largeFeatures = Vector<Dim1536>.zeros()
/// ```
public struct Dim1536: Dimension {
    public static let value = 1536
    public typealias Storage = Storage1536
}

/// A 3072-dimensional vector type for very large embeddings.
///
/// `Dim3072` provides compile-time dimension safety for 3072-element vectors,
/// suitable for GPT-style embeddings, concatenated features, and
/// high-dimensional representations. Maximum SIMD optimization for performance.
///
/// ## Example Usage
/// ```swift
/// let gptLargeEmbedding = Vector<Dim3072>.ones()
/// let concatenatedFeatures = Vector<Dim3072>.random(in: -1...1)
/// ```
public struct Dim3072: Dimension {
    public static let value = 3072
    public typealias Storage = Storage3072
}

// MARK: - Dynamic Dimension Support

/// A dimension type for vectors with runtime-determined sizes.
///
/// `DynamicDimension` enables working with vectors whose size is not known
/// at compile time. While it provides flexibility, it sacrifices the compile-time
/// safety and some optimizations available to fixed-dimension vectors.
///
/// ## Usage Considerations
/// - Use when vector dimensions vary at runtime
/// - Slightly less performant than fixed-dimension vectors
/// - No compile-time dimension checking
/// - Suitable for variable-length embeddings
///
/// ## Example Usage
/// ```swift
/// let dim = DynamicDimension(256)
/// let vector = DynamicVector(dimension: 256, repeating: 0.0)
/// ```
///
/// - Note: For best performance, use fixed-dimension types (e.g., `Dim256`)
///   when the dimension is known at compile time.
public struct DynamicDimension: Dimension {
    /// The runtime-determined size of vectors with this dimension.
    public let size: Int
    
    /// Not available for dynamic dimensions.
    /// - Warning: This property will trigger a fatal error if accessed.
    ///   Use `instance.size` instead.
    public static var value: Int {
        fatalError("DynamicDimension value must be accessed via instance.size")
    }
    
    /// The storage type for dynamic vectors, using flexible array storage.
    public typealias Storage = ArrayStorage<DynamicDimension>
    
    /// Creates a new dynamic dimension with the specified size.
    ///
    /// - Parameter size: The number of elements in vectors of this dimension.
    ///   Must be greater than zero.
    /// - Precondition: `size > 0`
    public init(_ size: Int) {
        precondition(size > 0, "Dimension must be positive")
        self.size = size
    }
}