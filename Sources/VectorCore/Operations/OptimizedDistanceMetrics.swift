//
//  OptimizedDistanceMetrics.swift
//  VectorCore
//
//  Specialized distance metric implementations for optimized vector types
//  Provides significant performance improvements by avoiding array conversions
//

import Foundation
import simd

// MARK: - Optimized Euclidean Distance

extension EuclideanDistance {
    
    /// Specialized implementation for Vector512Optimized
    @inlinable
    @inline(__always)
    public func distance(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float {
        a.euclideanDistance(to: b)
    }
    
    /// Specialized implementation for Vector768Optimized
    @inlinable
    @inline(__always)
    public func distance(_ a: Vector768Optimized, _ b: Vector768Optimized) -> Float {
        a.euclideanDistance(to: b)
    }
    
    /// Specialized implementation for Vector1536Optimized
    @inlinable
    @inline(__always)
    public func distance(_ a: Vector1536Optimized, _ b: Vector1536Optimized) -> Float {
        a.euclideanDistance(to: b)
    }
    
    /// Optimized batch distance for Vector512Optimized
    @inlinable
    public func batchDistance(
        query: Vector512Optimized,
        candidates: [Vector512Optimized]
    ) -> [DistanceScore] {
        // For small batches, use direct computation
        guard candidates.count > 100 else {
            return candidates.map { query.euclideanDistance(to: $0) }
        }
        
        // For large batches, use chunked processing for better cache locality
        var results = [DistanceScore](repeating: 0, count: candidates.count)
        let chunkSize = 64 // Optimize for cache line
        
        for chunkStart in stride(from: 0, to: candidates.count, by: chunkSize) {
            let chunkEnd = min(chunkStart + chunkSize, candidates.count)
            for i in chunkStart..<chunkEnd {
                results[i] = query.euclideanDistance(to: candidates[i])
            }
        }
        
        return results
    }
    
    /// Optimized batch distance for Vector768Optimized
    @inlinable
    public func batchDistance(
        query: Vector768Optimized,
        candidates: [Vector768Optimized]
    ) -> [DistanceScore] {
        // For small batches, use direct computation
        guard candidates.count > 100 else {
            return candidates.map { query.euclideanDistance(to: $0) }
        }
        
        // For large batches, use chunked processing for better cache locality
        var results = [DistanceScore](repeating: 0, count: candidates.count)
        let chunkSize = 64 // Optimize for cache line
        
        for chunkStart in stride(from: 0, to: candidates.count, by: chunkSize) {
            let chunkEnd = min(chunkStart + chunkSize, candidates.count)
            for i in chunkStart..<chunkEnd {
                results[i] = query.euclideanDistance(to: candidates[i])
            }
        }
        
        return results
    }
    
    /// Optimized batch distance for Vector1536Optimized
    @inlinable
    public func batchDistance(
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized]
    ) -> [DistanceScore] {
        // For small batches, use direct computation
        guard candidates.count > 100 else {
            return candidates.map { query.euclideanDistance(to: $0) }
        }
        
        // For large batches, use chunked processing for better cache locality
        var results = [DistanceScore](repeating: 0, count: candidates.count)
        let chunkSize = 32 // Smaller chunk for larger vectors
        
        for chunkStart in stride(from: 0, to: candidates.count, by: chunkSize) {
            let chunkEnd = min(chunkStart + chunkSize, candidates.count)
            for i in chunkStart..<chunkEnd {
                results[i] = query.euclideanDistance(to: candidates[i])
            }
        }
        
        return results
    }
}

// MARK: - Optimized Cosine Distance

extension CosineDistance {
    
    /// Specialized implementation for Vector512Optimized
    @inlinable
    @inline(__always)
    public func distance(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float {
        1.0 - a.cosineSimilarity(to: b)
    }
    
    /// Specialized implementation for Vector768Optimized
    @inlinable
    @inline(__always)
    public func distance(_ a: Vector768Optimized, _ b: Vector768Optimized) -> Float {
        1.0 - a.cosineSimilarity(to: b)
    }
    
    /// Specialized implementation for Vector1536Optimized
    @inlinable
    @inline(__always)
    public func distance(_ a: Vector1536Optimized, _ b: Vector1536Optimized) -> Float {
        1.0 - a.cosineSimilarity(to: b)
    }
    
    /// Optimized batch distance for Vector512Optimized
    @inlinable
    public func batchDistance(
        query: Vector512Optimized,
        candidates: [Vector512Optimized]
    ) -> [DistanceScore] {
        candidates.map { 1.0 - query.cosineSimilarity(to: $0) }
    }
    
    /// Optimized batch distance for Vector768Optimized
    @inlinable
    public func batchDistance(
        query: Vector768Optimized,
        candidates: [Vector768Optimized]
    ) -> [DistanceScore] {
        candidates.map { 1.0 - query.cosineSimilarity(to: $0) }
    }
    
    /// Optimized batch distance for Vector1536Optimized
    @inlinable
    public func batchDistance(
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized]
    ) -> [DistanceScore] {
        candidates.map { 1.0 - query.cosineSimilarity(to: $0) }
    }
}

// MARK: - Optimized Dot Product Distance

extension DotProductDistance {
    
    /// Specialized implementation for Vector512Optimized
    @inlinable
    @inline(__always)
    public func distance(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float {
        -a.dotProduct(b)
    }
    
    /// Specialized implementation for Vector768Optimized
    @inlinable
    @inline(__always)
    public func distance(_ a: Vector768Optimized, _ b: Vector768Optimized) -> Float {
        -a.dotProduct(b)
    }
    
    /// Specialized implementation for Vector1536Optimized
    @inlinable
    @inline(__always)
    public func distance(_ a: Vector1536Optimized, _ b: Vector1536Optimized) -> Float {
        -a.dotProduct(b)
    }
    
    /// Optimized batch distance for Vector512Optimized
    @inlinable
    public func batchDistance(
        query: Vector512Optimized,
        candidates: [Vector512Optimized]
    ) -> [DistanceScore] {
        candidates.map { -query.dotProduct($0) }
    }
    
    /// Optimized batch distance for Vector768Optimized
    @inlinable
    public func batchDistance(
        query: Vector768Optimized,
        candidates: [Vector768Optimized]
    ) -> [DistanceScore] {
        candidates.map { -query.dotProduct($0) }
    }
    
    /// Optimized batch distance for Vector1536Optimized
    @inlinable
    public func batchDistance(
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized]
    ) -> [DistanceScore] {
        candidates.map { -query.dotProduct($0) }
    }
}

// MARK: - Optimized Manhattan Distance

extension ManhattanDistance {
    
    /// Specialized implementation for Vector512Optimized
    @inlinable
    public func distance(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float {
        var sum = SIMD4<Float>()
        
        for i in 0..<128 {
            let diff = a.storage[i] - b.storage[i]
            sum += abs(diff)
        }
        
        return sum.x + sum.y + sum.z + sum.w
    }
    
    /// Specialized implementation for Vector768Optimized
    @inlinable
    public func distance(_ a: Vector768Optimized, _ b: Vector768Optimized) -> Float {
        var sum = SIMD4<Float>()
        
        for i in 0..<192 {
            let diff = a.storage[i] - b.storage[i]
            sum += abs(diff)
        }
        
        return sum.x + sum.y + sum.z + sum.w
    }
    
    /// Specialized implementation for Vector1536Optimized
    @inlinable
    public func distance(_ a: Vector1536Optimized, _ b: Vector1536Optimized) -> Float {
        var sum = SIMD4<Float>()
        
        for i in 0..<384 {
            let diff = a.storage[i] - b.storage[i]
            sum += abs(diff)
        }
        
        return sum.x + sum.y + sum.z + sum.w
    }
    
    /// Optimized batch distance for Vector512Optimized
    @inlinable
    public func batchDistance(
        query: Vector512Optimized,
        candidates: [Vector512Optimized]
    ) -> [DistanceScore] {
        candidates.map { distance(query, $0) }
    }
    
    /// Optimized batch distance for Vector768Optimized
    @inlinable
    public func batchDistance(
        query: Vector768Optimized,
        candidates: [Vector768Optimized]
    ) -> [DistanceScore] {
        candidates.map { distance(query, $0) }
    }
    
    /// Optimized batch distance for Vector1536Optimized
    @inlinable
    public func batchDistance(
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized]
    ) -> [DistanceScore] {
        candidates.map { distance(query, $0) }
    }
}

// MARK: - Optimized Chebyshev Distance

extension ChebyshevDistance {
    
    /// Specialized implementation for Vector512Optimized
    @inlinable
    public func distance(_ a: Vector512Optimized, _ b: Vector512Optimized) -> Float {
        var maxDiff = SIMD4<Float>()
        
        for i in 0..<128 {
            let diff = abs(a.storage[i] - b.storage[i])
            maxDiff = max(maxDiff, diff)
        }
        
        return max(max(maxDiff.x, maxDiff.y), max(maxDiff.z, maxDiff.w))
    }
    
    /// Specialized implementation for Vector768Optimized
    @inlinable
    public func distance(_ a: Vector768Optimized, _ b: Vector768Optimized) -> Float {
        var maxDiff = SIMD4<Float>()
        
        for i in 0..<192 {
            let diff = abs(a.storage[i] - b.storage[i])
            maxDiff = max(maxDiff, diff)
        }
        
        return max(max(maxDiff.x, maxDiff.y), max(maxDiff.z, maxDiff.w))
    }
    
    /// Specialized implementation for Vector1536Optimized
    @inlinable
    public func distance(_ a: Vector1536Optimized, _ b: Vector1536Optimized) -> Float {
        var maxDiff = SIMD4<Float>()
        
        for i in 0..<384 {
            let diff = abs(a.storage[i] - b.storage[i])
            maxDiff = max(maxDiff, diff)
        }
        
        return max(max(maxDiff.x, maxDiff.y), max(maxDiff.z, maxDiff.w))
    }
    
    /// Optimized batch distance for Vector512Optimized
    @inlinable
    public func batchDistance(
        query: Vector512Optimized,
        candidates: [Vector512Optimized]
    ) -> [DistanceScore] {
        candidates.map { distance(query, $0) }
    }
    
    /// Optimized batch distance for Vector768Optimized
    @inlinable
    public func batchDistance(
        query: Vector768Optimized,
        candidates: [Vector768Optimized]
    ) -> [DistanceScore] {
        candidates.map { distance(query, $0) }
    }
    
    /// Optimized batch distance for Vector1536Optimized
    @inlinable
    public func batchDistance(
        query: Vector1536Optimized,
        candidates: [Vector1536Optimized]
    ) -> [DistanceScore] {
        candidates.map { distance(query, $0) }
    }
}

// MARK: - Distance Metric Factory

/// Factory for creating distance metrics with optimized implementations
public struct OptimizedDistanceMetrics {
    
    /// Available optimized metrics
    public enum MetricType: String, CaseIterable {
        case euclidean
        case cosine
        case dotProduct
        case manhattan
        case chebyshev
        case hamming
        
        /// Create the corresponding distance metric
        public func createMetric() -> any DistanceMetric {
            switch self {
            case .euclidean:
                return EuclideanDistance()
            case .cosine:
                return CosineDistance()
            case .dotProduct:
                return DotProductDistance()
            case .manhattan:
                return ManhattanDistance()
            case .chebyshev:
                return ChebyshevDistance()
            case .hamming:
                return HammingDistance()
            }
        }
    }
    
    /// Create a distance metric by type
    public static func create(_ type: MetricType) -> any DistanceMetric {
        type.createMetric()
    }
    
    /// Create a distance metric by name
    public static func create(named name: String) -> (any DistanceMetric)? {
        MetricType(rawValue: name.lowercased())?.createMetric()
    }
}