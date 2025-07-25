//
//  TieredStorageTests.swift
//  VectorCoreTests
//
//  Tests for adaptive tiered storage implementation
//

import XCTest
@testable import VectorCore

final class TieredStorageTests: XCTestCase {
    
    // MARK: - Tier Selection Tests
    
    func testInlineTierSelection() {
        // Test that small counts use inline storage
        let storage1 = TieredStorage<Float>(capacity: 1)
        XCTAssertEqual(storage1.capacity, 4) // Inline buffer has fixed capacity
        
        let storage4 = TieredStorage<Float>(capacity: 4)
        XCTAssertEqual(storage4.capacity, 4)
    }
    
    func testCompactTierSelection() {
        // Test that medium counts use compact storage
        let storage = TieredStorage<Float>(capacity: 100)
        XCTAssertTrue(storage.capacity >= 100)
        XCTAssertTrue(storage.capacity % 16 == 0) // Should be cache-line aligned
    }
    
    func testStandardTierSelection() {
        // Test that larger counts use standard storage
        let storage = TieredStorage<Float>(capacity: 1000)
        XCTAssertTrue(storage.capacity >= 1000)
    }
    
    func testAlignedTierSelection() {
        // Test that very large counts use aligned storage
        let storage = TieredStorage<Float>(capacity: 5000)
        XCTAssertTrue(storage.capacity >= 5000)
        
        #if canImport(Accelerate)
        XCTAssertTrue(storage.isAlignedForSIMD)
        #endif
    }
    
    func testSIMDHintForcesAlignment() {
        // Test that SIMD hint forces aligned tier even for small sizes
        let storage = TieredStorage<Float>(capacity: 100, hint: .simd)
        
        #if canImport(Accelerate)
        XCTAssertTrue(storage.isAlignedForSIMD)
        #endif
    }
    
    // MARK: - Initialization Tests
    
    func testInitFromSequence() {
        // Test inline tier
        let small = TieredStorage([1.0, 2.0, 3.0])
        XCTAssertEqual(small.count, 3)
        XCTAssertEqual(small[0], 1.0)
        XCTAssertEqual(small[1], 2.0)
        XCTAssertEqual(small[2], 3.0)
        
        // Test compact tier
        let medium = TieredStorage(Array(repeating: 42.0, count: 100))
        XCTAssertEqual(medium.count, 100)
        XCTAssertEqual(medium[50], 42.0)
        
        // Test standard tier
        let large = TieredStorage(Array(repeating: 3.14, count: 1500))
        XCTAssertEqual(large.count, 1500)
        XCTAssertEqual(large[1000], 3.14)
        
        // Test aligned tier
        let veryLarge = TieredStorage(Array(repeating: 2.71, count: 3000))
        XCTAssertEqual(veryLarge.count, 3000)
        XCTAssertEqual(veryLarge[2500], 2.71)
    }
    
    // MARK: - Element Access Tests
    
    func testSubscriptAccess() {
        var storage = TieredStorage([10.0, 20.0, 30.0, 40.0])
        
        // Test read
        XCTAssertEqual(storage[0], 10.0)
        XCTAssertEqual(storage[3], 40.0)
        
        // Test write
        storage[1] = 25.0
        XCTAssertEqual(storage[1], 25.0)
        
        // Test bounds checking (should trap in debug)
        // XCTAssertThrowsError(storage[4]) // Would trap, not throw
    }
    
    func testMutationAcrossTiers() {
        // Test mutation in each tier
        
        // Inline
        var inline = TieredStorage([1.0, 2.0])
        inline[0] = 10.0
        XCTAssertEqual(inline[0], 10.0)
        
        // Compact
        var compact = TieredStorage(Array(repeating: 0.0, count: 50))
        compact[25] = 100.0
        XCTAssertEqual(compact[25], 100.0)
        
        // Standard
        var standard = TieredStorage(Array(repeating: 0.0, count: 1000))
        standard[500] = 200.0
        XCTAssertEqual(standard[500], 200.0)
        
        // Aligned
        var aligned = TieredStorage(Array(repeating: 0.0, count: 3000))
        aligned[2000] = 300.0
        XCTAssertEqual(aligned[2000], 300.0)
    }
    
    // MARK: - Buffer Pointer Tests
    
    func testUnsafeBufferPointerAccess() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0]
        let storage = TieredStorage(values)
        
        storage.withUnsafeBufferPointer { buffer in
            XCTAssertEqual(buffer.count, 5)
            for (i, value) in buffer.enumerated() {
                XCTAssertEqual(value, values[i])
            }
        }
    }
    
    func testUnsafeMutableBufferPointerAccess() {
        var storage = TieredStorage(Array(repeating: Float(0.0), count: 10))
        
        storage.withUnsafeMutableBufferPointer { buffer in
            for i in 0..<buffer.count {
                buffer[i] = Float(i * 10)
            }
        }
        
        for i in 0..<10 {
            XCTAssertEqual(storage[i], Float(i * 10))
        }
    }
    
    // MARK: - Copy-on-Write Tests
    
    func testCopyOnWriteSemantics() {
        // Test that COW works for reference-type tiers
        let original = TieredStorage(Array(repeating: 1.0, count: 100))
        var copy = original
        
        // Values should be equal
        XCTAssertEqual(original[50], copy[50])
        
        // Mutation should trigger COW
        copy[50] = 99.0
        
        // Original should be unchanged
        XCTAssertEqual(original[50], 1.0)
        XCTAssertEqual(copy[50], 99.0)
    }
    
    // MARK: - Performance Hint Tests
    
    func testOptimizeForSIMD() {
        var storage = TieredStorage(Array(repeating: 1.0, count: 100))
        
        #if canImport(Accelerate)
        XCTAssertFalse(storage.isAlignedForSIMD)
        #endif
        
        storage.optimizeFor(.simd)
        
        #if canImport(Accelerate)
        XCTAssertTrue(storage.isAlignedForSIMD)
        #endif
    }
    
    // MARK: - Collection Conformance Tests
    
    func testSequenceConformance() {
        let values: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let storage = TieredStorage(values)
        
        // Test iteration
        var result = [Float]()
        for value in storage {
            result.append(value)
        }
        
        XCTAssertEqual(result, values)
    }
    
    func testCollectionConformance() {
        let storage = TieredStorage([10.0, 20.0, 30.0])
        
        XCTAssertEqual(storage.startIndex, 0)
        XCTAssertEqual(storage.endIndex, 3)
        XCTAssertEqual(storage.index(after: 0), 1)
        XCTAssertEqual(storage.index(after: 1), 2)
    }
    
    // MARK: - Performance Tests
    
    func testInlinePerformance() {
        measure {
            var storage = TieredStorage<Float>(capacity: 4)
            for _ in 0..<100_000 {
                storage[0] = 1.0
                storage[1] = 2.0
                storage[2] = 3.0
                storage[3] = 4.0
                _ = storage[0] + storage[1] + storage[2] + storage[3]
            }
        }
    }
    
    func testCompactPerformance() {
        let size = 100
        measure {
            var storage = TieredStorage<Float>(capacity: size)
            for _ in 0..<1000 {
                for i in 0..<size {
                    storage[i] = Float(i)
                }
                var sum: Float = 0
                for i in 0..<size {
                    sum += storage[i]
                }
                _ = sum
            }
        }
    }
    
    func testAlignedPerformance() {
        let size = 10_000
        let values = Array(repeating: Float(1.0), count: size)
        
        measure {
            let storage = TieredStorage(values)
            var sum: Float = 0
            storage.withUnsafeBufferPointer { buffer in
                for value in buffer {
                    sum += value
                }
            }
            _ = sum
        }
    }
    
    // MARK: - Memory Tests
    
    func testMemoryEfficiency() {
        // Test that inline storage doesn't allocate
        let inline = TieredStorage([1.0, 2.0])
        XCTAssertEqual(inline.capacity, 4) // Fixed size, no allocation
        
        // Test that compact storage aligns to cache lines
        let compact = TieredStorage<Float>(capacity: 17)
        XCTAssertTrue(compact.capacity >= 17)
        XCTAssertTrue(compact.capacity % (64 / MemoryLayout<Float>.stride) == 0)
    }
    
    // MARK: - Edge Cases
    
    func testEmptyStorage() {
        let empty = TieredStorage<Float>([])
        XCTAssertEqual(empty.count, 0)
        XCTAssertEqual(empty.capacity, 4) // Still uses inline tier
    }
    
    func testBoundaryTransitions() {
        // Test exact boundary values
        
        // 4/5 boundary (inline/compact)
        let inline4 = TieredStorage(Array(repeating: 1.0, count: 4))
        XCTAssertEqual(inline4.capacity, 4)
        
        let compact5 = TieredStorage(Array(repeating: 1.0, count: 5))
        XCTAssertTrue(compact5.capacity >= 5)
        
        // 512/513 boundary (compact/standard)
        let _ = TieredStorage(Array(repeating: 1.0, count: 512))
        let _ = TieredStorage(Array(repeating: 1.0, count: 513))
        // Can't easily test the tier without exposing internals
        
        // 2048/2049 boundary (standard/aligned)
        let _ = TieredStorage(Array(repeating: 1.0, count: 2048))
        let aligned2049 = TieredStorage(Array(repeating: 1.0, count: 2049))
        
        #if canImport(Accelerate)
        XCTAssertTrue(aligned2049.isAlignedForSIMD)
        #endif
    }
    
    // MARK: - Thread Safety Tests (if needed)
    
    @MainActor
    func testConcurrentReads() {
        let storage = TieredStorage(Array(repeating: 42.0, count: 1000))
        let expectation = self.expectation(description: "Concurrent reads")
        expectation.expectedFulfillmentCount = 10
        
        DispatchQueue.concurrentPerform(iterations: 10) { i in
            let value = storage[i * 100]
            XCTAssertEqual(value, 42.0)
            expectation.fulfill()
        }
        
        waitForExpectations(timeout: 1.0)
    }
}

// MARK: - Test Utilities

extension TieredStorageTests {
    /// Create storage with known tier based on size
    func makeStorage(tier: TierType, count: Int) -> TieredStorage<Float> {
        switch tier {
        case .inline:
            return TieredStorage(Array(repeating: 1.0, count: min(count, 4)))
        case .compact:
            return TieredStorage(Array(repeating: 1.0, count: min(max(count, 5), 512)))
        case .standard:
            return TieredStorage(Array(repeating: 1.0, count: min(max(count, 513), 2048)))
        case .aligned:
            return TieredStorage(Array(repeating: 1.0, count: max(count, 2049)))
        }
    }
    
    enum TierType {
        case inline, compact, standard, aligned
    }
}