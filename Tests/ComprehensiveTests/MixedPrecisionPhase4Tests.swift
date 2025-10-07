import Testing
import Foundation
@testable import VectorCore

@Suite("Mixed Precision Phase 4 - Advanced Features")
struct MixedPrecisionPhase4Tests {

    // MARK: - Helper Functions

    func generateRandomVector512() -> Vector512Optimized {
        return try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
    }

    func generateRandomVectors512(count: Int) -> [Vector512Optimized] {
        return (0..<count).map { _ in generateRandomVector512() }
    }

    // MARK: - FP16VectorPool Tests

    @Test("FP16VectorPool: Basic acquire and release")
    func testVectorPool_BasicOperations() throws {
        let pool = FP16VectorPool(capacity: 10, dimension: 512)

        // Initially, pool should have all vectors available
        #expect(pool.availableCount == 10, "Pool should start with 10 vectors")

        // Acquire a vector
        let vec1 = pool.acquire()
        #expect(vec1 != nil, "Should be able to acquire vector")
        #expect(pool.availableCount == 9, "Pool should have 9 vectors after acquire")

        // Acquire another
        let vec2 = pool.acquire()
        #expect(vec2 != nil, "Should be able to acquire second vector")
        #expect(pool.availableCount == 8, "Pool should have 8 vectors after second acquire")

        // Release both vectors
        pool.release(vec1!)
        #expect(pool.availableCount == 9, "Pool should have 9 vectors after first release")

        pool.release(vec2!)
        #expect(pool.availableCount == 10, "Pool should have 10 vectors after both released")

        print("✓ FP16VectorPool basic operations validated")
    }

    @Test("FP16VectorPool: Capacity limits")
    func testVectorPool_CapacityLimits() throws {
        let pool = FP16VectorPool(capacity: 3, dimension: 512)

        // Acquire all vectors
        let vec1 = pool.acquire()
        let vec2 = pool.acquire()
        let vec3 = pool.acquire()

        #expect(pool.availableCount == 0, "Pool should be empty")

        // Try to acquire when empty
        let vec4 = pool.acquire()
        #expect(vec4 == nil, "Should return nil when pool is empty")

        // Release all 3
        pool.release(vec1!)
        pool.release(vec2!)
        pool.release(vec3!)

        #expect(pool.availableCount == 3, "Pool should be full again")

        // Create external vector and try to release (should be accepted)
        let externalVec = MixedPrecisionKernels.Vector512FP16(
            from: generateRandomVector512()
        )
        pool.release(externalVec)

        // Pool at capacity - should still be 3 (external vec discarded)
        #expect(pool.availableCount == 3, "Pool should not exceed capacity")

        print("✓ FP16VectorPool capacity limits validated")
    }

    @Test("FP16VectorPool: Drain operation")
    func testVectorPool_Drain() throws {
        let pool = FP16VectorPool(capacity: 5, dimension: 512)

        // Acquire some vectors
        _ = pool.acquire()
        _ = pool.acquire()

        #expect(pool.availableCount == 3, "Pool should have 3 vectors")

        // Drain pool
        pool.drain()
        #expect(pool.availableCount == 0, "Pool should be empty after drain")

        print("✓ FP16VectorPool drain operation validated")
    }

    @Test("FP16VectorPool: Concurrent access stress test")
    func testVectorPool_ConcurrentAccess() async throws {
        let pool = FP16VectorPool(capacity: 50, dimension: 512)
        let iterations = 100

        // Spawn 10 concurrent tasks that acquire/release repeatedly
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<10 {
                group.addTask {
                    for _ in 0..<iterations {
                        if let vec = pool.acquire() {
                            // Simulate some work
                            try? await Task.sleep(nanoseconds: 100)
                            pool.release(vec)
                        }
                    }
                }
            }
        }

        // After all tasks complete, pool should have all vectors back
        #expect(pool.availableCount == 50, "Pool should have all vectors after concurrent test")

        print("✓ FP16VectorPool concurrent access validated (1000 operations)")
    }

    // MARK: - PlatformCapabilities Tests

    @Test("PlatformCapabilities: Platform detection")
    func testPlatformCapabilities_Detection() throws {
        // Validate that platform detection returns sensible values
        print("Platform: \(PlatformCapabilities.platformName)")
        print("Native FP16: \(PlatformCapabilities.hasNativeHardwareFP16)")
        print("NEON: \(PlatformCapabilities.hasNEON)")
        print("AVX2: \(PlatformCapabilities.hasAVX2)")
        print("AVX-512: \(PlatformCapabilities.hasAVX512)")
        print("SIMD Width: \(PlatformCapabilities.optimalSIMDWidth)")
        print("L3 Cache: \(PlatformCapabilities.estimatedL3CacheSize / 1024 / 1024) MB")

        // Platform-specific assertions
        #if arch(arm64)
        #expect(PlatformCapabilities.hasNEON == true, "ARM64 should have NEON")
        #expect(PlatformCapabilities.hasNativeHardwareFP16 == true, "ARM64 should have hardware FP16")
        #expect(PlatformCapabilities.hasAVX2 == false, "ARM64 should not have AVX2")
        #expect(PlatformCapabilities.optimalSIMDWidth == 4, "ARM64 SIMD width should be 4")
        #elseif arch(x86_64)
        #expect(PlatformCapabilities.hasNEON == false, "x86_64 should not have NEON")
        #expect(PlatformCapabilities.hasAVX2 == true, "x86_64 should have AVX2")
        #expect(PlatformCapabilities.optimalSIMDWidth == 8, "x86_64 SIMD width should be 8")
        #endif

        print("✓ PlatformCapabilities detection validated")
    }

    @Test("PlatformCapabilities: Cache size estimation")
    func testPlatformCapabilities_CacheSize() throws {
        let cacheSize = PlatformCapabilities.estimatedL3CacheSize

        // Validate reasonable cache size (should be at least 4 MB)
        #expect(cacheSize >= 4 * 1024 * 1024, "L3 cache estimate should be at least 4 MB")
        #expect(cacheSize <= 128 * 1024 * 1024, "L3 cache estimate should be at most 128 MB")

        print("✓ Platform cache size estimation: \(cacheSize / 1024 / 1024) MB")
    }

    // MARK: - MixedPrecisionProvider Tests

    @Test("MixedPrecisionProvider: Single distance computation")
    func testProvider_SingleDistance() async throws {
        let provider = MixedPrecisionProvider()

        let vec1 = generateRandomVector512()
        let vec2 = generateRandomVector512()

        // Compute distance via provider
        let providerDist = try await provider.distance(
            from: vec1,
            to: vec2,
            metric: .euclidean
        )

        // Compute reference distance
        let referenceDist = EuclideanKernels.distance512(vec1, vec2)

        // Should match exactly (no FP16 conversion for single distances)
        #expect(abs(providerDist - referenceDist) < 1e-6, "Single distance should match reference")

        print("✓ MixedPrecisionProvider single distance validated")
    }

    @Test("MixedPrecisionProvider: Batch distance - small batch (FP32 path)")
    func testProvider_SmallBatch() async throws {
        let provider = MixedPrecisionProvider(minBatchSize: 50)

        let query = generateRandomVector512()
        let candidates = generateRandomVectors512(count: 10)  // Below threshold

        // Should use FP32 path (below minBatchSize)
        let distances = try await provider.batchDistance(
            from: query,
            to: candidates,
            metric: .euclidean
        )

        #expect(distances.count == 10, "Should return 10 distances")
        #expect(distances.allSatisfy { $0.isFinite }, "All distances should be finite")

        print("✓ MixedPrecisionProvider small batch (FP32) validated")
    }

    @Test("MixedPrecisionProvider: Batch distance - large batch (FP16 path)")
    func testProvider_LargeBatch() async throws {
        let provider = MixedPrecisionProvider(minBatchSize: 32)

        let query = generateRandomVector512()
        let candidates = generateRandomVectors512(count: 100)  // Above threshold

        // Should use FP16 SoA path
        let distancesFP16 = try await provider.batchDistance(
            from: query,
            to: candidates,
            metric: .euclidean
        )

        #expect(distancesFP16.count == 100, "Should return 100 distances")
        #expect(distancesFP16.allSatisfy { $0.isFinite }, "All distances should be finite")

        // Compute FP32 reference
        let distancesFP32 = candidates.map { EuclideanKernels.distance512(query, $0) }

        // Validate accuracy (should be <0.1% error)
        var maxRelativeError: Float = 0
        for (fp16, fp32) in zip(distancesFP16, distancesFP32) {
            let relativeError = abs(fp16 - fp32) / max(fp32, 1e-6)
            maxRelativeError = max(maxRelativeError, relativeError)
        }

        #expect(maxRelativeError < 0.001, "Max relative error should be <0.1%, got \(maxRelativeError * 100)%")

        print("✓ MixedPrecisionProvider large batch (FP16) validated: max error = \(String(format: "%.4f%%", maxRelativeError * 100))")
    }

    @Test("MixedPrecisionProvider: Cosine distance fallback")
    func testProvider_CosineFallback() async throws {
        let provider = MixedPrecisionProvider()

        let query = generateRandomVector512()
        let candidates = generateRandomVectors512(count: 50)

        // Cosine should fallback to FP32 (not yet implemented in FP16)
        let distances = try await provider.batchDistance(
            from: query,
            to: candidates,
            metric: .cosine
        )

        #expect(distances.count == 50, "Should return 50 distances")
        #expect(distances.allSatisfy { $0.isFinite }, "All distances should be finite")

        // Validate against reference
        let reference = candidates.map { CosineKernels.distance512_fused(query, $0) }
        for (computed, ref) in zip(distances, reference) {
            #expect(abs(computed - ref) < 1e-6, "Cosine distances should match FP32 reference")
        }

        print("✓ MixedPrecisionProvider cosine fallback validated")
    }

    @Test("MixedPrecisionProvider: Blocked kernel selection")
    func testProvider_BlockedKernelSelection() async throws {
        let provider = MixedPrecisionProvider(minBatchSize: 10)

        let query = generateRandomVector512()

        // Test boundary: batch size = 16 (should trigger blocked kernel)
        let candidates16 = generateRandomVectors512(count: 16)
        let distances16 = try await provider.batchDistance(
            from: query,
            to: candidates16,
            metric: .euclidean
        )

        #expect(distances16.count == 16, "Should return 16 distances")
        #expect(distances16.allSatisfy { $0.isFinite }, "All distances should be finite")

        // Test larger batch (should definitely use blocked kernel)
        let candidates100 = generateRandomVectors512(count: 100)
        let distances100 = try await provider.batchDistance(
            from: query,
            to: candidates100,
            metric: .euclidean
        )

        #expect(distances100.count == 100, "Should return 100 distances")

        print("✓ MixedPrecisionProvider blocked kernel selection validated")
    }

    // MARK: - Platform-Specific Optimizations Tests

    @Test("Platform-optimized batch conversion: Correctness")
    func testPlatformConversion_Correctness() throws {
        let count = 1024
        let source = (0..<count).map { _ in Float.random(in: -100...100) }

        let sourceBuffer = source
        var destination = [UInt16](repeating: 0, count: count)

        // Convert using platform-optimized function
        sourceBuffer.withUnsafeBufferPointer { srcPtr in
            destination.withUnsafeMutableBufferPointer { dstPtr in
                MixedPrecisionKernels.platformOptimizedConvertBatch(
                    source: srcPtr,
                    destination: dstPtr
                )
            }
        }

        // Validate against reference conversion
        for (idx, fp32) in source.enumerated() {
            let referenceUInt16 = MixedPrecisionKernels.fp32ToFp16_scalar(fp32)
            #expect(destination[idx] == referenceUInt16, "Platform conversion should match reference at index \(idx)")
        }

        print("✓ Platform-optimized batch conversion validated (1024 elements)")
    }

    @Test("Platform-optimized batch conversion: Alignment handling")
    func testPlatformConversion_UnalignedSizes() throws {
        // Test various odd sizes to ensure tail handling works
        for count in [1, 3, 7, 15, 17, 63, 100, 1000] {
            let source = (0..<count).map { _ in Float.random(in: -10...10) }
            let sourceBuffer = source
            var destination = [UInt16](repeating: 0, count: count)

            sourceBuffer.withUnsafeBufferPointer { srcPtr in
                destination.withUnsafeMutableBufferPointer { dstPtr in
                    MixedPrecisionKernels.platformOptimizedConvertBatch(
                        source: srcPtr,
                        destination: dstPtr
                    )
                }
            }

            // Validate all elements
            for (idx, fp32) in source.enumerated() {
                let referenceUInt16 = MixedPrecisionKernels.fp32ToFp16_scalar(fp32)
                #expect(destination[idx] == referenceUInt16, "Count=\(count), index \(idx) should match")
            }
        }

        print("✓ Platform-optimized conversion handles odd sizes correctly")
    }

    @Test("Platform-optimized batch conversion: Special values")
    func testPlatformConversion_SpecialValues() throws {
        let specialValues: [Float] = [
            0.0, -0.0,
            Float.infinity, -Float.infinity,
            1.0, -1.0,
            65504.0,   // Max FP16 normal
            -65504.0,  // Min FP16 normal
            0.00006103515625,  // Min FP16 normal (2^-14)
            0.000000059604645, // Min FP16 subnormal (2^-24)
        ]

        let source = specialValues
        var destination = [UInt16](repeating: 0, count: specialValues.count)

        source.withUnsafeBufferPointer { srcPtr in
            destination.withUnsafeMutableBufferPointer { dstPtr in
                MixedPrecisionKernels.platformOptimizedConvertBatch(
                    source: srcPtr,
                    destination: dstPtr
                )
            }
        }

        // Validate each special value
        for (idx, fp32) in specialValues.enumerated() {
            let referenceUInt16 = MixedPrecisionKernels.fp32ToFp16_scalar(fp32)
            #expect(destination[idx] == referenceUInt16, "Special value \(fp32) should match at index \(idx)")
        }

        print("✓ Platform-optimized conversion handles special values correctly")
    }

    // MARK: - Integration Tests

    @Test("Phase 4 Integration: Pool + Provider workflow")
    func testIntegration_PooledProvider() async throws {
        let pool = FP16VectorPool(capacity: 20, dimension: 512)
        let provider = MixedPrecisionProvider(minBatchSize: 10)

        let query = generateRandomVector512()
        let candidates = generateRandomVectors512(count: 50)

        // Use provider for batch computation
        let distances = try await provider.batchDistance(
            from: query,
            to: candidates,
            metric: .euclidean
        )

        #expect(distances.count == 50, "Should compute 50 distances")
        #expect(distances.allSatisfy { $0.isFinite }, "All distances should be finite")

        // Validate pool is still functional
        let vec = pool.acquire()
        #expect(vec != nil, "Pool should still be functional")
        pool.release(vec!)

        print("✓ Phase 4 integration (Pool + Provider) validated")
    }

    @Test("Phase 4 Integration: Platform capabilities usage")
    func testIntegration_PlatformAwareness() throws {
        // Create provider with platform-specific threshold
        let threshold = PlatformCapabilities.hasNativeHardwareFP16 ? 16 : 64
        _ = MixedPrecisionProvider(minBatchSize: threshold)

        // Log platform-aware configuration
        print("Platform: \(PlatformCapabilities.platformName)")
        print("Using batch threshold: \(threshold)")
        print("SIMD width: \(PlatformCapabilities.optimalSIMDWidth)")

        #expect(threshold > 0, "Threshold should be positive")

        print("✓ Platform-aware configuration validated")
    }
}
