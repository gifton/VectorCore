//
//  SoAFP16Cache.swift
//  VectorCore
//
//  LRU cache for SoA FP16 conversions
//  Implements Phase 3 of kernel-specs/11-mixed-precision-kernels-part2.md
//

import Foundation

// MARK: - Cache Key

/// Cache key for SoA FP16 storage
///
/// Uses vector hashes for fast lookup while avoiding storing full vector data.
/// This enables efficient caching of expensive SoA transposition operations.
fileprivate struct SoAFP16CacheKey: Hashable, Sendable {
    let vectorHashes: [Int]
    let dimension: Int

    init(vectors: [Vector512Optimized], dimension: Int = 512) {
        self.vectorHashes = vectors.map { $0.hashValue }
        self.dimension = dimension
    }

    // TODO: Add init for Vector768Optimized and Vector1536Optimized when their caches are implemented
}

// MARK: - SoA FP16 Cache (512-dimensional)

/// Thread-safe LRU cache for SoA FP16 conversions (512-dimensional)
///
/// Caches the result of expensive AoS → SoA FP16 transpositions. This is beneficial when:
/// - The same candidate set is queried multiple times
/// - SoA creation overhead exceeds cache lookup cost
/// - Memory is available for caching (typical size: ~1KB per 100 vectors)
///
/// ## Usage Example
/// ```swift
/// let cache = SoAFP16Cache.shared
///
/// // Check cache first
/// if let cached = await cache.get(for: candidates) {
///     // Use cached SoA
/// } else {
///     // Create and store
///     let soa = MixedPrecisionKernels.createSoA512FP16(from: candidates)
///     await cache.store(soa, for: candidates)
/// }
/// ```
public actor SoAFP16Cache512 {

    // MARK: - Singleton

    public static let shared = SoAFP16Cache512()

    // MARK: - State

    private var cache: [SoAFP16CacheKey: MixedPrecisionKernels.SoA512FP16] = [:]
    private var accessOrder: [SoAFP16CacheKey] = []  // LRU tracking (most recent at end)
    private let maxSize: Int

    // MARK: - Statistics

    private var hitCount: Int = 0
    private var missCount: Int = 0

    // MARK: - Initialization

    public init(maxSize: Int = 100) {
        self.maxSize = maxSize
    }

    // MARK: - Cache Operations

    /// Get cached SoA for given vectors
    ///
    /// - Parameter vectors: Candidate vectors
    /// - Returns: Cached SoA if available, nil otherwise
    public func get(for vectors: [Vector512Optimized]) -> MixedPrecisionKernels.SoA512FP16? {
        let key = SoAFP16CacheKey(vectors: vectors)

        if let cached = cache[key] {
            // Update LRU access order
            if let index = accessOrder.firstIndex(of: key) {
                accessOrder.remove(at: index)
                accessOrder.append(key)
            }

            hitCount += 1
            return cached
        }

        missCount += 1
        return nil
    }

    /// Store SoA in cache
    ///
    /// - Parameters:
    ///   - soa: SoA FP16 structure to cache
    ///   - vectors: Original vectors (for key generation)
    public func store(
        _ soa: MixedPrecisionKernels.SoA512FP16,
        for vectors: [Vector512Optimized]
    ) {
        let key = SoAFP16CacheKey(vectors: vectors)

        // If key already exists, just update access order
        if cache[key] != nil {
            if let index = accessOrder.firstIndex(of: key) {
                accessOrder.remove(at: index)
            }
        } else {
            // Evict oldest if at capacity
            if cache.count >= maxSize {
                let oldestKey = accessOrder.removeFirst()
                cache.removeValue(forKey: oldestKey)
            }
        }

        cache[key] = soa
        accessOrder.append(key)
    }

    /// Clear all cached entries
    public func clear() {
        cache.removeAll()
        accessOrder.removeAll()
        hitCount = 0
        missCount = 0
    }

    // MARK: - Statistics

    /// Get cache hit rate
    ///
    /// - Returns: Hit rate in [0, 1] range, or 0 if no accesses yet
    public func hitRate() -> Double {
        let total = hitCount + missCount
        guard total > 0 else { return 0 }
        return Double(hitCount) / Double(total)
    }

    /// Get cache statistics
    ///
    /// - Returns: Tuple of (hits, misses, size, capacity)
    public func statistics() -> (hits: Int, misses: Int, size: Int, capacity: Int) {
        return (hitCount, missCount, cache.count, maxSize)
    }
}

// MARK: - Future: 768D and 1536D Caches

// TODO: Implement SoAFP16Cache768 and SoAFP16Cache1536 once
// MixedPrecisionKernels.createSoA768FP16() and createSoA1536FP16() are implemented.
//
// These will follow the same LRU cache pattern as SoAFP16Cache512.

// MARK: - Convenience Extensions

extension SoAFP16Cache512 {
    /// Get or create SoA with automatic caching
    ///
    /// - Parameter vectors: Candidate vectors
    /// - Returns: Cached or newly created SoA
    public func getOrCreate(for vectors: [Vector512Optimized]) -> MixedPrecisionKernels.SoA512FP16 {
        if let cached = get(for: vectors) {
            return cached
        }

        let soa = MixedPrecisionKernels.createSoA512FP16(from: vectors)
        store(soa, for: vectors)
        return soa
    }
}
