// VectorCore: Dimension Protocol and Types
//
// Defines compile-time and runtime dimension specifications
//

import Foundation

/// Protocol for compile-time dimension specification
///
/// A static dimension specifies only its compile-time element count (`value`) and `Sendable` conformance.
/// It names no storage: `Vector<D>.Storage` is `DimensionStorage<D, Float>` for every `D`, so code that
/// needs the storage type of a vector writes `Vector<D>.Storage`. Every `Vector<D>` member not specific to
/// `Dim2`, `Dim3` or `Dim4` (the members of `Vector.swift`'s `where D == Dim2/Dim3/Dim4` extensions) is
/// available for every static dimension, built-in or custom.
///
/// Domain of `value` (unverified contract beyond the widths D02DegenerateWidthTests covers): it must be `>= 0`.
/// Width 0 gives empty vectors and width 1 single-element vectors; members that need an element (for example
/// `minElement()`, `maxElement()`, `unitX`, `basis(axis:)`) stop with a precondition failure at width 0, and the
/// other width-0/width-1 results are the ones that test's table pins. A negative `value` is undefined and untested.
public protocol StaticDimension: Sendable {
    /// The number of elements in vectors of this dimension
    static var value: Int { get }
}

// MARK: - Standard Dimensions

/// A 2-dimensional vector type for 2D coordinates and pairs.
///
/// `Dim2` provides compile-time dimension safety for 2-element vectors,
/// commonly used for 2D points, complex numbers, and coordinate pairs.
///
/// ## Example Usage
/// ```swift
/// let point = Vector<Dim2>(x: 1.0, y: 2.0)
/// let direction = Vector<Dim2>.normalized()
/// ```
public struct Dim2: StaticDimension {
    public static let value = 2
}

/// A 3-dimensional vector type for 3D coordinates and RGB values.
///
/// `Dim3` provides compile-time dimension safety for 3-element vectors,
/// commonly used for 3D points, colors, and spatial coordinates.
///
/// ## Example Usage
/// ```swift
/// let position = Vector<Dim3>(x: 1.0, y: 2.0, z: 3.0)
/// let color = Vector<Dim3>(x: 0.5, y: 0.8, z: 1.0)
/// ```
public struct Dim3: StaticDimension {
    public static let value = 3
}

/// A 4-dimensional vector type for homogeneous coordinates and RGBA.
///
/// `Dim4` provides compile-time dimension safety for 4-element vectors,
/// commonly used for homogeneous 3D coordinates, quaternions, and RGBA colors.
///
/// ## Example Usage
/// ```swift
/// let quaternion = Vector<Dim4>(x: 0, y: 0, z: 0, w: 1)
/// let rgba = Vector<Dim4>(x: 1, y: 0, z: 0, w: 0.5)
/// ```
public struct Dim4: StaticDimension {
    public static let value = 4
}

/// A 8-dimensional vector type for small feature vectors.
///
/// `Dim8` provides compile-time dimension safety for 8-element vectors,
/// commonly used for small neural network layers and feature descriptors.
///
/// ## Example Usage
/// ```swift
/// let features = Vector<Dim8>.random(in: -1...1)
/// let weights = Vector<Dim8>.ones()
/// ```
public struct Dim8: StaticDimension {
    public static let value = 8
}

/// A 16-dimensional vector type for compact features.
///
/// `Dim16` provides compile-time dimension safety for 16-element vectors,
/// suitable for compact feature representations and small embeddings.
///
/// ## Example Usage
/// ```swift
/// let embedding = Vector<Dim16>.zeros()
/// let features = Vector<Dim16>.random(in: 0...1)
/// ```
public struct Dim16: StaticDimension {
    public static let value = 16
}

/// A 32-dimensional vector type for small embeddings.
///
/// `Dim32` provides compile-time dimension safety for 32-element vectors,
/// commonly used for simple feature vectors and low-dimensional embeddings.
/// Its 32 elements exceed the 16-element inline tier, so `DimensionStorage` keeps them in heap storage.
///
/// ## Example Usage
/// ```swift
/// let embedding = Vector<Dim32>.random(in: -1...1)
/// let features = Vector<Dim32>.zeros()
/// ```
public struct Dim32: StaticDimension {
    public static let value = 32
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
public struct Dim64: StaticDimension {
    public static let value = 64
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
public struct Dim128: StaticDimension {
    public static let value = 128
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
public struct Dim256: StaticDimension {
    public static let value = 256
}

/// A 384-dimensional vector type for MiniLM and SBERT embeddings.
///
/// `Dim384` provides compile-time dimension safety for 384-element vectors,
/// the standard dimension for MiniLM models (all-MiniLM-L6-v2) and many
/// Sentence-BERT variants.
///
/// ## Example Usage
/// ```swift
/// let miniLMEmbedding = Vector<Dim384>.normalized()
/// let sbertFeatures = Vector<Dim384>.random(in: -1...1)
/// ```
///
/// ## Layout Note
/// 384 dimensions = 96 SIMD4 lanes, a multiple of the stride-16 kernel pattern.
/// No speed comparison with other widths is measured (not verified).
public struct Dim384: StaticDimension, Sendable {
    public static let value = 384
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
public struct Dim512: StaticDimension, Sendable {
    public static let value = 512
}

/// A 768-dimensional vector type for transformer embeddings.
///
/// `Dim768` provides compile-time dimension safety for 768-element vectors,
/// the standard dimension for BERT base models and many transformer
/// architectures.
///
/// ## Example Usage
/// ```swift
/// let bertEmbedding = Vector<Dim768>.zeros()
/// let transformerOutput = Vector<Dim768>.random(in: -1...1)
/// ```
public struct Dim768: StaticDimension, Sendable {
    public static let value = 768
}

/// A 1024-dimensional vector type.
///
/// `Dim1024` provides compile-time dimension safety for 1024-element vectors,
/// suitable for medium-to-large ML models and neural network outputs.
///
/// ## Example Usage
/// ```swift
/// let features = Vector<Dim1024>.zeros()
/// let embeddings = Vector<Dim1024>.random(in: -1...1)
/// ```
public struct Dim1024: StaticDimension {
    public static let value = 1024
}

/// A 1536-dimensional vector type for large embeddings.
///
/// `Dim1536` provides compile-time dimension safety for 1536-element vectors,
/// commonly used for OpenAI embeddings and large language model outputs.
///
/// ## Example Usage
/// ```swift
/// let gptEmbedding = Vector<Dim1536>.normalized()
/// let largeFeatures = Vector<Dim1536>.zeros()
/// ```
public struct Dim1536: StaticDimension, Sendable {
    public static let value = 1536
}

/// A 2048-dimensional vector type.
///
/// `Dim2048` provides compile-time dimension safety for 2048-element vectors,
/// suitable for large model embeddings and high-dimensional feature vectors.
///
/// ## Example Usage
/// ```swift
/// let largeEmbedding = Vector<Dim2048>.zeros()
/// let features = Vector<Dim2048>.random(in: -1...1)
/// ```
public struct Dim2048: StaticDimension {
    public static let value = 2048
}

/// A 3072-dimensional vector type for very large embeddings.
///
/// `Dim3072` provides compile-time dimension safety for 3072-element vectors,
/// suitable for GPT-style embeddings, concatenated features, and
/// high-dimensional representations.
///
/// ## Example Usage
/// ```swift
/// let gptLargeEmbedding = Vector<Dim3072>.ones()
/// let concatenatedFeatures = Vector<Dim3072>.random(in: -1...1)
/// ```
public struct Dim3072: StaticDimension {
    public static let value = 3072
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
/// - Speed relative to fixed-dimension vectors is not measured in this repository
/// - No compile-time dimension checking
/// - Suitable for variable-length embeddings
///
/// ## Example Usage
/// ```swift
/// let dim = DynamicDimension(256)
/// let vector = DynamicVector(dimension: 256, repeating: 0.0)
/// ```
///
/// - Note: Prefer fixed-dimension types (e.g., `Dim256`) when the dimension is
///   known at compile time: they add compile-time dimension checking.
public struct DynamicDimension {
    /// The runtime-determined size of vectors with this dimension.
    public let size: Int

    /// Not available for dynamic dimensions.
    /// - Warning: This property will trigger a fatal error if accessed.
    ///   Use `instance.size` instead.
    @available(*, unavailable, message: "DynamicDimension has no static size; use instance.size instead")
    public static var value: Int { 0 }

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

    /// Throwing initializer that validates size and returns a DynamicDimension.
    /// - Parameter size: The desired dimension size. Must be > 0.
    /// - Throws: `VectorError.invalidDimension` if size <= 0.
    public init(validating size: Int) throws {
        guard size > 0 else {
            throw VectorError.invalidDimension(size, reason: "Dimension must be positive")
        }
        self.size = size
    }

    /// Factory method to create a validated dynamic dimension.
    /// - Parameter size: The desired dimension size. Must be > 0.
    /// - Throws: `VectorError.invalidDimension` if size <= 0.
    public static func make(_ size: Int) throws -> DynamicDimension {
        try DynamicDimension(validating: size)
    }
}
