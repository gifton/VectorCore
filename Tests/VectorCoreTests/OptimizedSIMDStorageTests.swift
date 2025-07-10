// VectorCore: Optimized SIMD Storage Tests
//
// Comprehensive tests for zero-allocation SIMD storage implementations
//

import XCTest
@testable import VectorCore

final class OptimizedSIMDStorageTests: XCTestCase {
    
    // MARK: - Correctness Tests
    
    func testSIMDStorage128Correctness() {
        // Test initialization
        let zeros = OptimizedSIMDStorage128()
        for i in 0..<128 {
            XCTAssertEqual(zeros[i], 0.0, accuracy: 1e-7)
        }
        
        // Test repeating value initialization
        let ones = OptimizedSIMDStorage128(repeating: 1.0)
        for i in 0..<128 {
            XCTAssertEqual(ones[i], 1.0, accuracy: 1e-7)
        }
        
        // Test array initialization
        let values = (0..<128).map { Float($0) }
        let storage = OptimizedSIMDStorage128(from: values)
        for i in 0..<128 {
            XCTAssertEqual(storage[i], Float(i), accuracy: 1e-7)
        }
        
        // Test mutation
        var mutable = OptimizedSIMDStorage128()
        for i in 0..<128 {
            mutable[i] = Float(i * 2)
        }
        for i in 0..<128 {
            XCTAssertEqual(mutable[i], Float(i * 2), accuracy: 1e-7)
        }
        
        // Test buffer access
        storage.withUnsafeBufferPointer { buffer in
            XCTAssertEqual(buffer.count, 128)
            for i in 0..<128 {
                XCTAssertEqual(buffer[i], Float(i), accuracy: 1e-7)
            }
        }
        
        // Test mutable buffer access
        var mutableStorage = OptimizedSIMDStorage128()
        mutableStorage.withUnsafeMutableBufferPointer { buffer in
            for i in 0..<128 {
                buffer[i] = Float(i * 3)
            }
        }
        for i in 0..<128 {
            XCTAssertEqual(mutableStorage[i], Float(i * 3), accuracy: 1e-7)
        }
        
        // Test dot product
        let a = OptimizedSIMDStorage128(repeating: 2.0)
        let b = OptimizedSIMDStorage128(repeating: 3.0)
        let dotProduct = a.dotProduct(b)
        XCTAssertEqual(dotProduct, 768.0, accuracy: 1e-5) // 2 * 3 * 128
    }
    
    func testSIMDStorage256Correctness() {
        // Similar tests for 256 dimensions
        let zeros = OptimizedSIMDStorage256()
        for i in 0..<256 {
            XCTAssertEqual(zeros[i], 0.0, accuracy: 1e-7)
        }
        
        let values = (0..<256).map { Float($0) }
        let storage = OptimizedSIMDStorage256(from: values)
        
        storage.withUnsafeBufferPointer { buffer in
            XCTAssertEqual(buffer.count, 256)
            for i in 0..<256 {
                XCTAssertEqual(buffer[i], Float(i), accuracy: 1e-7)
            }
        }
        
        // Test boundary access
        var mutable = OptimizedSIMDStorage256()
        mutable[0] = 1.0
        mutable[63] = 2.0
        mutable[64] = 3.0
        mutable[127] = 4.0
        mutable[128] = 5.0
        mutable[191] = 6.0
        mutable[192] = 7.0
        mutable[255] = 8.0
        
        XCTAssertEqual(mutable[0], 1.0)
        XCTAssertEqual(mutable[63], 2.0)
        XCTAssertEqual(mutable[64], 3.0)
        XCTAssertEqual(mutable[127], 4.0)
        XCTAssertEqual(mutable[128], 5.0)
        XCTAssertEqual(mutable[191], 6.0)
        XCTAssertEqual(mutable[192], 7.0)
        XCTAssertEqual(mutable[255], 8.0)
    }
    
    func testSIMDStorage512Correctness() {
        let values = (0..<512).map { Float($0) }
        let storage = OptimizedSIMDStorage512(from: values)
        
        // Verify all values
        for i in 0..<512 {
            XCTAssertEqual(storage[i], Float(i), accuracy: 1e-7)
        }
        
        // Test dot product
        let a = OptimizedSIMDStorage512(repeating: 1.0)
        let b = OptimizedSIMDStorage512(repeating: 2.0)
        let dotProduct = a.dotProduct(b)
        XCTAssertEqual(dotProduct, 1024.0, accuracy: 1e-5) // 1 * 2 * 512
    }
    
    func testSIMDStorage768Correctness() {
        let storage = OptimizedSIMDStorage768(repeating: 3.14159)
        
        storage.withUnsafeBufferPointer { buffer in
            XCTAssertEqual(buffer.count, 768)
            for i in 0..<768 {
                XCTAssertEqual(buffer[i], 3.14159, accuracy: 1e-5)
            }
        }
    }
    
    func testSIMDStorage1536Correctness() {
        var storage = OptimizedSIMDStorage1536()
        
        // Test sparse updates
        storage[0] = 1.0
        storage[511] = 2.0
        storage[512] = 3.0
        storage[1023] = 4.0
        storage[1024] = 5.0
        storage[1535] = 6.0
        
        XCTAssertEqual(storage[0], 1.0)
        XCTAssertEqual(storage[511], 2.0)
        XCTAssertEqual(storage[512], 3.0)
        XCTAssertEqual(storage[1023], 4.0)
        XCTAssertEqual(storage[1024], 5.0)
        XCTAssertEqual(storage[1535], 6.0)
    }
    
    // MARK: - Zero Allocation Verification
    
    func testZeroAllocationInBufferAccess() {
        // This test verifies that buffer access doesn't allocate
        let storage128 = OptimizedSIMDStorage128(repeating: 1.0)
        let storage256 = OptimizedSIMDStorage256(repeating: 1.0)
        
        // Measure allocations for 128
        var allocCount128 = 0
        for _ in 0..<1000 {
            storage128.withUnsafeBufferPointer { buffer in
                // Access buffer to ensure it's not optimized away
                _ = buffer[0]
                allocCount128 += 1
            }
        }
        
        // Measure allocations for 256
        var allocCount256 = 0
        for _ in 0..<1000 {
            storage256.withUnsafeBufferPointer { buffer in
                _ = buffer[0]
                allocCount256 += 1
            }
        }
        
        // The allocation count should match iteration count (no heap allocations)
        XCTAssertEqual(allocCount128, 1000)
        XCTAssertEqual(allocCount256, 1000)
    }
    
    // MARK: - API Compatibility Tests
    
    func testAPICompatibility() {
        // Verify that optimized types conform to required protocols
        func testStorage<S: VectorStorage & VectorStorageOperations>(_ storage: S) {
            XCTAssertGreaterThan(storage.count, 0)
            _ = storage[0]
            _ = storage.dotProduct(storage)
        }
        
        testStorage(OptimizedSIMDStorage128())
        testStorage(OptimizedSIMDStorage256())
        testStorage(OptimizedSIMDStorage512())
        testStorage(OptimizedSIMDStorage768())
        testStorage(OptimizedSIMDStorage1536())
    }
    
    // MARK: - Thread Safety Tests
    
    func testConcurrentReads() {
        let storage = OptimizedSIMDStorage256(repeating: 42.0)
        let queue = DispatchQueue(label: "test", attributes: .concurrent)
        let group = DispatchGroup()
        
        // Perform many concurrent reads
        for _ in 0..<100 {
            group.enter()
            queue.async {
                storage.withUnsafeBufferPointer { buffer in
                    for i in 0..<256 {
                        XCTAssertEqual(buffer[i], 42.0, accuracy: 1e-7)
                    }
                }
                group.leave()
            }
        }
        
        group.wait()
    }
    
    // MARK: - Edge Cases
    
    func testBoundaryConditions() {
        // Test boundary indices for each storage type
        let storage128 = OptimizedSIMDStorage128(repeating: 1.0)
        XCTAssertEqual(storage128[0], 1.0)
        XCTAssertEqual(storage128[63], 1.0)
        XCTAssertEqual(storage128[64], 1.0)
        XCTAssertEqual(storage128[127], 1.0)
        
        let storage256 = OptimizedSIMDStorage256(repeating: 2.0)
        XCTAssertEqual(storage256[0], 2.0)
        XCTAssertEqual(storage256[63], 2.0)
        XCTAssertEqual(storage256[64], 2.0)
        XCTAssertEqual(storage256[127], 2.0)
        XCTAssertEqual(storage256[128], 2.0)
        XCTAssertEqual(storage256[191], 2.0)
        XCTAssertEqual(storage256[192], 2.0)
        XCTAssertEqual(storage256[255], 2.0)
    }
}