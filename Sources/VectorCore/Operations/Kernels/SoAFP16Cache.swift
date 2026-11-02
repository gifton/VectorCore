import Foundation

/// Hashing selects a bucket; equality compares the ordered Float bit snapshot.
private struct SoAFP16CacheKey: Hashable, Sendable {
    let bits: [UInt32]
    let vectorCount: Int
    let dimension: Int = 512
    let fingerprint: Int

    init(vectors: [Vector512Optimized], fingerprint: (@Sendable ([UInt32]) -> Int)?) {
        vectorCount = vectors.count
        // A materialized candidate array has 512 coordinates per fixed-shape vector.
        bits = [UInt32](unsafeUninitializedCapacity: vectors.count * 512) { buffer, initializedCount in
            var index = 0
            for vector in vectors {
                for coordinate in 0..<512 {
                    buffer[index] = vector[coordinate].bitPattern
                    index += 1
                }
            }
            initializedCount = index
        }
        if let fingerprint {
            self.fingerprint = fingerprint(bits)
        } else {
            var hasher = Hasher()
            hasher.combine(bits)
            self.fingerprint = hasher.finalize()
        }
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(vectorCount)
        hasher.combine(dimension)
        hasher.combine(fingerprint)
    }

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.vectorCount == rhs.vectorCount && lhs.dimension == rhs.dimension && lhs.bits == rhs.bits
    }
}

/// Actor-isolated LRU cache for exact 512-dimensional FP16 conversions.
///
/// Identity includes candidate order, count and every Float bit, including signed
/// zero and NaN payloads (SoAFP16CacheIdentityTests). The byte budget counts allocated
/// UInt32 snapshot and UInt16 SoA capacities. It excludes dictionary/LRU metadata,
/// transient lookup/conversion allocations and results retained only by callers;
/// it is not an RSS limit. Returned immutable SoA storage may outlive eviction.
///
/// Migration: replace `shared` with an owned `SoAFP16Cache512(maxBytes: budget)`.
/// Replace manual `store` calls with `getOrCreate(for:)`, which binds the conversion
/// to the same candidates as its identity snapshot.
public actor SoAFP16Cache512 {
    private struct Entry {
        let key: SoAFP16CacheKey
        let soa: MixedPrecisionKernels.SoA512FP16
        let bytes: Int
    }

    private var cache: [SoAFP16CacheKey: Entry] = [:]
    private var accessOrder: [SoAFP16CacheKey] = []
    private let maxSize: Int
    private let maxBytes: Int
    private let fingerprint: (@Sendable ([UInt32]) -> Int)?
    private var usedBytes = 0
    private var hitCount = 0
    private var missCount = 0

    /// Both limits must be nonnegative. Zero in either disables retention.
    /// Invalid limits trap at initialization with a named diagnostic in Debug/Release
    /// (SoAFP16CacheIdentityTests.negativeLimitsRejectAtInitialization).
    public init(maxSize: Int = 100, maxBytes: Int) {
        guard maxSize >= 0 else { fatalError("SoAFP16Cache512 maxSize must be nonnegative") }
        guard maxBytes >= 0 else { fatalError("SoAFP16Cache512 maxBytes must be nonnegative") }
        self.maxSize = maxSize
        self.maxBytes = maxBytes
        self.fingerprint = nil
    }

    // Collision injection exercises the actual dictionary route in
    // SoAFP16CacheIdentityTests.collisionBucketsStillRequireAllBits.
    internal init(maxSize: Int = 100, maxBytes: Int, fingerprint: @escaping @Sendable ([UInt32]) -> Int) {
        guard maxSize >= 0 else { fatalError("SoAFP16Cache512 maxSize must be nonnegative") }
        guard maxBytes >= 0 else { fatalError("SoAFP16Cache512 maxBytes must be nonnegative") }
        self.maxSize = maxSize
        self.maxBytes = maxBytes
        self.fingerprint = fingerprint
    }

    public func get(for vectors: [Vector512Optimized]) -> MixedPrecisionKernels.SoA512FP16? {
        lookup(SoAFP16CacheKey(vectors: vectors, fingerprint: fingerprint))
    }

    private func lookup(_ key: SoAFP16CacheKey) -> MixedPrecisionKernels.SoA512FP16? {
        guard let entry = cache[key] else {
            missCount += 1
            return nil
        }
        if let index = accessOrder.firstIndex(of: entry.key) {
            accessOrder.remove(at: index)
        }
        // Reuse the retained key's COW allocation, not the new lookup snapshot.
        accessOrder.append(entry.key)
        hitCount += 1
        return entry.soa
    }

    /// Return the exact conversion, retaining it only when both limits permit it.
    /// Oversized entries bypass retention without evicting existing entries.
    public func getOrCreate(for vectors: [Vector512Optimized]) -> MixedPrecisionKernels.SoA512FP16 {
        let key = SoAFP16CacheKey(vectors: vectors, fingerprint: fingerprint)
        if let cached = lookup(key) { return cached }
        let soa = MixedPrecisionKernels.createSoA512FP16(from: vectors)
        guard maxSize > 0, maxBytes > 0,
              let bytes = Self.retainedCost(identityCapacity: key.bits.capacity, soaCapacity: soa.storage.capacity),
              bytes <= maxBytes else { return soa }

        // Subtraction avoids overflowing usedBytes + bytes; both terms are bounded
        // by maxBytes. Every removed key has exactly one accounted entry.
        while cache.count >= maxSize || usedBytes > maxBytes - bytes {
            let oldest = accessOrder.removeFirst()
            if let removed = cache.removeValue(forKey: oldest) { usedBytes -= removed.bytes }
        }
        cache[key] = Entry(key: key, soa: soa, bytes: bytes)
        accessOrder.append(key)
        usedBytes += bytes
        return soa
    }

    /// Checked payload cost; nil means the capacities cannot be represented in bytes.
    internal nonisolated static func retainedCost(identityCapacity: Int, soaCapacity: Int) -> Int? {
        guard identityCapacity >= 0, soaCapacity >= 0 else { return nil }
        let identity = identityCapacity.multipliedReportingOverflow(by: MemoryLayout<UInt32>.stride)
        let soa = soaCapacity.multipliedReportingOverflow(by: MemoryLayout<UInt16>.stride)
        guard !identity.overflow, !soa.overflow else { return nil }
        let total = identity.partialValue.addingReportingOverflow(soa.partialValue)
        return total.overflow ? nil : total.partialValue
    }

    /// Remove retained entries and reset both retention and access statistics.
    public func clear() {
        cache.removeAll()
        accessOrder.removeAll()
        usedBytes = 0
        hitCount = 0
        missCount = 0
    }

    public func hitRate() -> Double {
        let total = Double(hitCount) + Double(missCount)
        return total > 0 ? Double(hitCount) / total : 0
    }

    public func statistics() -> (hits: Int, misses: Int, size: Int, capacity: Int) {
        (hitCount, missCount, cache.count, maxSize)
    }

    /// Allocated payload capacity retained by this actor, with the exclusions above.
    public func retentionStatistics() -> (usedBytes: Int, maxBytes: Int) {
        (usedBytes, maxBytes)
    }
}
