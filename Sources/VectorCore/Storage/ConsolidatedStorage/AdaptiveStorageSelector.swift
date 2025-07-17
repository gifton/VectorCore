// VectorCore: Adaptive Storage Selector
//
// Automatic selection of optimal storage type based on dimension
//

import Foundation

/// Namespace for adaptive storage selection
public enum AdaptiveStorage {
    
    /// Select the optimal storage type for a given dimension
    public static func select(for dimension: Int) -> StorageType {
        switch dimension {
        case 1...64:
            return .small
        case 65...512:
            return .medium
        default:
            return .large
        }
    }
    
    /// Storage type categories
    public enum StorageType {
        case small   // 1-64 dimensions
        case medium  // 65-512 dimensions
        case large   // 513+ dimensions
    }
    
    /// Create storage with zeros for given dimension
    public static func zeros(dimension: Int) -> any VectorStorage {
        switch select(for: dimension) {
        case .small:
            return SmallVectorStorage(count: dimension)
        case .medium:
            return MediumVectorStorage(count: dimension)
        case .large:
            return LargeVectorStorage(count: dimension)
        }
    }
    
    /// Create storage with repeating value for given dimension
    public static func repeating(_ value: Float, dimension: Int) -> any VectorStorage {
        switch select(for: dimension) {
        case .small:
            return SmallVectorStorage(count: dimension, repeating: value)
        case .medium:
            return MediumVectorStorage(count: dimension, repeating: value)
        case .large:
            return LargeVectorStorage(count: dimension, repeating: value)
        }
    }
    
    /// Create storage from array
    public static func from(_ values: [Float]) -> any VectorStorage {
        switch select(for: values.count) {
        case .small:
            return SmallVectorStorage(from: values)
        case .medium:
            return MediumVectorStorage(from: values)
        case .large:
            return LargeVectorStorage(from: values)
        }
    }
    
    /// Get recommended storage type for a dimension
    public static func recommendedType(for dimension: Int) -> String {
        switch select(for: dimension) {
        case .small:
            return "SmallVectorStorage (SIMD64-based, \(dimension)/64 elements used)"
        case .medium:
            return "MediumVectorStorage (AlignedValueStorage, \(dimension)/512 elements used)"
        case .large:
            return "LargeVectorStorage (COWDynamicStorage, exact fit)"
        }
    }
}

// MARK: - Type-Erased Storage Operations

/// Protocol to enable operations on type-erased storage
public protocol AdaptiveVectorStorage: VectorStorage {
    func dotProduct(with other: any AdaptiveVectorStorage) -> Float?
}

// Extend our storage types to support adaptive operations
extension SmallVectorStorage: AdaptiveVectorStorage {
    public func dotProduct(with other: any AdaptiveVectorStorage) -> Float? {
        guard let otherSmall = other as? SmallVectorStorage,
              self.actualCount == otherSmall.actualCount else {
            return nil
        }
        return self.dotProduct(otherSmall)
    }
}

extension MediumVectorStorage: AdaptiveVectorStorage {
    public func dotProduct(with other: any AdaptiveVectorStorage) -> Float? {
        guard let otherMedium = other as? MediumVectorStorage,
              self.actualCount == otherMedium.actualCount else {
            return nil
        }
        return self.dotProduct(otherMedium)
    }
}

extension LargeVectorStorage: AdaptiveVectorStorage {
    public func dotProduct(with other: any AdaptiveVectorStorage) -> Float? {
        guard let otherLarge = other as? LargeVectorStorage,
              self.count == otherLarge.count else {
            return nil
        }
        return self.dotProduct(otherLarge)
    }
}