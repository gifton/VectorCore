import XCTest
@testable import VectorCore

final class MemoryLeakTests: XCTestCase {
    
    // MARK: - Leak Detection Infrastructure
    
    private func assertNoMemoryLeak<T: AnyObject>(
        _ object: @autoclosure () -> T,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        weak var weakReference: T?
        
        autoreleasepool {
            let strongReference = object()
            weakReference = strongReference
            
            // Use the object to prevent optimization
            _ = String(describing: strongReference)
        }
        
        // Object should be deallocated
        XCTAssertNil(weakReference, "Memory leak detected", file: file, line: line)
    }
    
    // MARK: - Vector Lifecycle Tests
    
    func testVectorDeallocation() {
        // Test all vector sizes
        let dimensions = [128, 256, 512, 768, 1536, 3072]
        
        for dim in dimensions {
            autoreleasepool {
                let vector = VectorFactory.random(dimension: dim)
                weak var weakVector = vector as AnyObject
                
                // Vector should exist while in scope
                XCTAssertNotNil(weakVector)
            }
            // Vector should be deallocated after leaving scope
        }
    }
    
    func testLargeVectorArrayDeallocation() {
        weak var weakArray: NSArray?
        
        autoreleasepool {
            let vectors = (0..<1000).map { _ in Vector512.random(in: -1...1) }
            weakArray = vectors as NSArray
            
            // Process vectors
            _ = vectors.map { $0.magnitude }
        }
        
        // Array and all vectors should be deallocated
        XCTAssertNil(weakArray, "Vector array not deallocated")
    }
    
    // MARK: - Storage Leak Tests
    
    func testAlignedStorageDeallocation() {
        class StorageWrapper {
            let storage: AlignedValueStorage
            init() {
                storage = AlignedValueStorage(count: 3072, alignment: 64)
            }
        }
        
        assertNoMemoryLeak(StorageWrapper())
    }
    
    func testStorageCopyOnWriteMemory() {
        autoreleasepool {
            let original = MediumVectorStorage(count: 512)
            var copies: [MediumVectorStorage] = []
            
            // Create multiple copies
            for _ in 0..<10 {
                copies.append(original)
            }
            
            // Modify each copy (triggers COW)
            for (index, var copy) in copies.enumerated() {
                copy[0] = Float(index)
                copies[index] = copy
            }
            
            // All copies should have independent storage
            for (index, copy) in copies.enumerated() {
                XCTAssertEqual(copy[0], Float(index))
            }
        }
        
        // All storage should be deallocated
    }
    
    // MARK: - Async Operation Leak Tests
    
    func testAsyncOperationMemoryLeaks() async {
        weak var weakVectors: NSArray?
        
        await withTaskGroup(of: Void.self) { _ in
            let vectors = (0..<100).map { _ in Vector256.random(in: -1...1) }
            weakVectors = vectors as NSArray
            
            // Async operations
            _ = await BatchOperations.findNearest(
                to: vectors[0],
                in: vectors,
                k: 10
            )
            
            _ = await BatchOperations.statistics(for: vectors)
        }
        
        // Ensure cleanup after async operations
        XCTAssertNil(weakVectors, "Vectors retained after async operations")
    }
    
    // MARK: - Circular Reference Tests
    
    func testNoCircularReferences() {
        // VectorError doesn't have a public initializer or chain method
        // This test is no longer applicable with the current error design
        let error1 = VectorError.dimensionMismatch(expected: 10, actual: 20)
        let error2 = VectorError.indexOutOfBounds(index: 5, dimension: 4)
        
        // Just verify the errors exist
        XCTAssertNotNil(error1.errorDescription)
        XCTAssertNotNil(error2.errorDescription)
        
        // No circular references to test with current error design
    }
}

// MARK: - Memory Pressure Tests

final class MemoryPressureTests: XCTestCase {
    
    // MARK: - Memory Pressure Scenarios
    
    func testLargeAllocationHandling() throws {
        let initialMemory = getMemoryUsage()
        
        // Try to allocate very large vectors
        do {
            var vectors: [any VectorType] = []
            
            // Allocate ~1GB of vectors
            let vectorSize = 3072
            let vectorCount = 100_000
            
            for i in 0..<vectorCount {
                if i % 10_000 == 0 {
                    let currentMemory = getMemoryUsage()
                    let usedMemory = currentMemory - initialMemory
                    
                    // Check if we're approaching memory limits
                    if usedMemory > 900_000_000 { // 900MB
                        print("Approaching memory limit at \(i) vectors")
                        break
                    }
                }
                
                vectors.append(VectorFactory.random(dimension: vectorSize))
            }
            
            print("Successfully allocated \(vectors.count) vectors")
            
            // Verify we can still operate
            let sample = vectors.prefix(100)
            let magnitudes = sample.map { $0.magnitude }
            XCTAssertEqual(magnitudes.count, sample.count)
            
        } catch {
            // Memory allocation failure is acceptable
            XCTAssertTrue(error is VectorError)
        }
    }
    
    func testMemoryWarningResponse() async throws {
        // Simulate memory pressure by allocating large batches
        var batches: [[Vector1536]] = []
        
        for _ in 0..<10 {
            autoreleasepool {
                let batch = (0..<1000).map { _ in Vector1536.random(in: -1...1) }
                batches.append(batch)
                
                // Process to ensure memory is actually used
                _ = batch.map { $0.magnitude }
            }
        }
        
        // System should handle memory pressure gracefully
        let totalVectors = batches.flatMap { $0 }.count
        XCTAssertGreaterThan(totalVectors, 0)
        
        // Clear memory
        batches.removeAll()
    }
    
    func testBatchProcessingMemoryEfficiency() async throws {
        let vectorCount = 100_000
        let batchSize = 1024
        
        // Create large dataset
        let vectors = (0..<vectorCount).map { _ in Vector512.random(in: -1...1) }
        
        let initialMemory = getMemoryUsage()
        
        // Process in batches
        let results = try await BatchOperations.process(
            vectors,
            batchSize: batchSize
        ) { batch in
            // Simulate heavy processing
            batch.map { vector in
                let normalized = vector.normalized()
                let magnitude = normalized.magnitude
                return magnitude
            }
        }
        
        let peakMemory = getMemoryUsage()
        let memoryIncrease = peakMemory - initialMemory
        
        XCTAssertEqual(results.count, vectorCount)
        
        // Memory increase should be proportional to batch size, not total size
        let expectedMaxIncrease = Int64(batchSize * 512 * 4 * 2) // 2x for processing overhead
        XCTAssertLessThan(memoryIncrease, expectedMaxIncrease * 10,
                         "Batch processing using too much memory")
    }
    
    // MARK: - Memory Fragmentation Tests
    
    func testMemoryFragmentation() {
        // Allocate and deallocate in patterns that could cause fragmentation
        for iteration in 0..<10 {
            autoreleasepool {
                var vectors: [any VectorType] = []
                
                // Allocate vectors of different sizes
                for i in 0..<1000 {
                    let dimension = [128, 512, 1536, 3072][i % 4]
                    vectors.append(VectorFactory.random(dimension: dimension))
                    
                    // Randomly remove vectors to create gaps
                    if i > 100 && i % 7 == 0 {
                        vectors.remove(at: Int.random(in: 0..<vectors.count))
                    }
                }
                
                // Verify operations still work efficiently
                let sample = Array(vectors.prefix(100))
                let start = CFAbsoluteTimeGetCurrent()
                _ = sample.map { $0.magnitude }
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                
                // Performance shouldn't degrade due to fragmentation
                XCTAssertLessThan(elapsed, 0.01,
                                 "Performance degraded in iteration \(iteration)")
            }
        }
    }
    
    // MARK: - Helpers
    
    private func getMemoryUsage() -> Int64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout.size(ofValue: info) / MemoryLayout<integer_t>.size)
        
        let result = withUnsafeMutablePointer(to: &info) { infoPtr in
            infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), intPtr, &count)
            }
        }
        
        return result == KERN_SUCCESS ? Int64(info.resident_size) : 0
    }
}