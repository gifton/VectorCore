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

/// 128-dimensional vectors (common for small embeddings)
public struct Dim128: Dimension {
    public static let value = 128
    public typealias Storage = SIMDStorage128
}

/// 256-dimensional vectors
public struct Dim256: Dimension {
    public static let value = 256
    public typealias Storage = SIMDStorage256
}

/// 512-dimensional vectors (common for BERT-style models)
public struct Dim512: Dimension {
    public static let value = 512
    public typealias Storage = SIMDStorage512
}

/// 768-dimensional vectors (BERT base)
public struct Dim768: Dimension {
    public static let value = 768
    public typealias Storage = SIMDStorage768
}

/// 1536-dimensional vectors (larger models)
public struct Dim1536: Dimension {
    public static let value = 1536
    public typealias Storage = SIMDStorage1536
}

/// 3072-dimensional vectors (GPT-style embeddings)
public struct Dim3072: Dimension {
    public static let value = 3072
    public typealias Storage = SIMDStorage3072
}

// MARK: - Dynamic Dimension Support

/// Dimension type for runtime-determined sizes
public struct DynamicDimension: Dimension {
    public let size: Int
    
    public static var value: Int {
        fatalError("DynamicDimension value must be accessed via instance.size")
    }
    
    public typealias Storage = ArrayStorage<DynamicDimension>
    
    public init(_ size: Int) {
        precondition(size > 0, "Dimension must be positive")
        self.size = size
    }
}