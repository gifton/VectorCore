// VectorCore: Type Aliases
//
// Backward-compatible type aliases for common vector dimensions
//

import Foundation

// MARK: - Standard Vector Type Aliases

/// 128-dimensional vector (common for small embeddings)
public typealias Vector128 = Vector<Dim128>

/// 256-dimensional vector
public typealias Vector256 = Vector<Dim256>

/// 512-dimensional vector (common for BERT-style models)
public typealias Vector512 = Vector<Dim512>

/// 768-dimensional vector (BERT base)
public typealias Vector768 = Vector<Dim768>

/// 1536-dimensional vector (larger models)
public typealias Vector1536 = Vector<Dim1536>

/// 3072-dimensional vector (very large models)
public typealias Vector3072 = Vector<Dim3072>

// MARK: - Score Type Aliases

/// Type alias for distance scores
public typealias DistanceScore = Float

/// Type alias for similarity scores
public typealias SimilarityScore = Float

// MARK: - Collection Type Aliases

/// Type alias for a collection of Vector128
public typealias Vector128Collection = [Vector128]

/// Type alias for a collection of Vector256
public typealias Vector256Collection = [Vector256]

/// Type alias for a collection of Vector512
public typealias Vector512Collection = [Vector512]

/// Type alias for a collection of Vector768
public typealias Vector768Collection = [Vector768]

/// Type alias for a collection of Vector1536
public typealias Vector1536Collection = [Vector1536]
