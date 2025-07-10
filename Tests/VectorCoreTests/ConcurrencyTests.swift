import XCTest
@testable import VectorCore

final class ConcurrencyTests: XCTestCase {
    
    // MARK: - Concurrent Vector Creation
    
    func testConcurrentVectorCreation() async {
        let iterations = 1000
        let tasks = 50
        
        await withTaskGroup(of: [any VectorType].self) { group in
            for _ in 0..<tasks {
                group.addTask {
                    var vectors: [any VectorType] = []
                    for _ in 0..<iterations {
                        let dimension = [128, 256, 512, 768].randomElement()!
                        let vector = VectorFactory.random(dimension: dimension)
                        vectors.append(vector)
                    }
                    return vectors
                }
            }
            
            var totalVectors = 0
            for await vectors in group {
                totalVectors += vectors.count
            }
            
            XCTAssertEqual(totalVectors, tasks * iterations)
        }
    }
    
    // MARK: - Concurrent Distance Calculations
    
    func testConcurrentDistanceCalculation() async {
        let vectors = (0..<100).map { _ in Vector512.random() }
        let queries = (0..<10).map { _ in Vector512.random() }
        
        // Calculate distances concurrently
        let results = await withTaskGroup(of: [(Int, Int, Float)].self) { group in
            for (i, query) in queries.enumerated() {
                group.addTask {
                    var distances: [(Int, Int, Float)] = []
                    for (j, vector) in vectors.enumerated() {
                        let distance = query.distance(to: vector)
                        distances.append((i, j, distance))
                    }
                    return distances
                }
            }
            
            var allDistances: [(Int, Int, Float)] = []
            for await distances in group {
                allDistances.append(contentsOf: distances)
            }
            return allDistances
        }
        
        // Verify all distances calculated
        XCTAssertEqual(results.count, queries.count * vectors.count)
        
        // Verify correctness by spot-checking
        for _ in 0..<10 {
            let (i, j, distance) = results.randomElement()!
            let expected = queries[i].distance(to: vectors[j])
            XCTAssertEqual(distance, expected, accuracy: 1e-5)
        }
    }
    
    // MARK: - Batch Operations Concurrency
    
    func testBatchOperationsConcurrency() async throws {
        let vectorGroups = (0..<10).map { _ in
            (0..<1000).map { _ in Vector256.random() }
        }
        
        // Process multiple batches concurrently
        let results = try await withTaskGroup(of: [Float].self) { group in
            for vectors in vectorGroups {
                group.addTask {
                    let magnitudes = try await BatchOperations.map(vectors) { $0.magnitude }
                    return magnitudes
                }
            }
            
            var allResults: [[Float]] = []
            for try await result in group {
                allResults.append(result)
            }
            return allResults
        }
        
        XCTAssertEqual(results.count, vectorGroups.count)
        for (i, result) in results.enumerated() {
            XCTAssertEqual(result.count, vectorGroups[i].count)
        }
    }
    
    // MARK: - Storage Thread Safety
    
    func testStorageThreadSafety() async {
        let storage = SIMDStorage1536()
        let iterations = 10000
        let writers = 5
        let readers = 20
        
        // Initialize storage
        for i in 0..<1536 {
            storage[i] = Float(i)
        }
        
        // Concurrent reads (should be safe)
        await withTaskGroup(of: Float.self) { group in
            for _ in 0..<readers {
                group.addTask {
                    var sum: Float = 0
                    for _ in 0..<iterations {
                        storage.withUnsafeBufferPointer { buffer in
                            sum += buffer.reduce(0, +)
                        }
                    }
                    return sum
                }
            }
            
            let expectedSum = Float(1536 * 1535 / 2) * Float(iterations)
            
            for await sum in group {
                XCTAssertEqual(sum, expectedSum, accuracy: 1e-3)
            }
        }
    }
    
    // MARK: - Race Condition Detection
    
    func testRaceConditionDetection() async {
        // Skip unsafe counter test in Swift 6 strict concurrency mode
        // The test was designed to intentionally demonstrate race conditions
        // which are now prevented by the compiler
        
        print("Note: Race condition detection test adapted for Swift 6 concurrency")
        
        // Instead test that proper actor usage prevents race conditions
        actor SafeCounter {
            private var value: Int = 0
            
            func increment() {
                value += 1
            }
            
            func getValue() -> Int {
                value
            }
        }
        
        let counter = SafeCounter()
        let iterations = 1000
        let tasks = 100
        
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<tasks {
                group.addTask {
                    for _ in 0..<iterations {
                        await counter.increment()
                    }
                }
            }
        }
        
        let finalValue = await counter.getValue()
        XCTAssertEqual(finalValue, tasks * iterations,
                      "Actor should prevent race conditions")
    }
    
    // MARK: - Actor Isolation Testing
    
    func testActorIsolation() async {
        // Safe counter using actor
        actor SafeCounter {
            private var value: Int = 0
            
            func increment() {
                value += 1
            }
            
            func getValue() -> Int {
                value
            }
        }
        
        let counter = SafeCounter()
        let iterations = 1000
        let tasks = 100
        
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<tasks {
                group.addTask {
                    for _ in 0..<iterations {
                        await counter.increment()
                    }
                }
            }
        }
        
        let finalValue = await counter.getValue()
        XCTAssertEqual(finalValue, tasks * iterations,
                      "Actor should prevent race conditions")
    }
    
    // MARK: - Stress Testing
    
    func testHighConcurrencyStress() async throws {
        let vectorCount = 10_000
        let queryCount = 100
        let k = 50
        
        // Create large dataset
        let vectors = (0..<vectorCount).map { _ in Vector768.random() }
        let queries = (0..<queryCount).map { _ in Vector768.random() }
        
        // Perform many k-NN searches concurrently
        let start = CFAbsoluteTimeGetCurrent()
        
        let results = await withTaskGroup(of: [(index: Int, distance: Float)].self) { group in
            for query in queries {
                group.addTask {
                    await BatchOperations.findNearest(to: query, in: vectors, k: k)
                }
            }
            
            var allResults: [[(index: Int, distance: Float)]] = []
            for await result in group {
                allResults.append(result)
            }
            return allResults
        }
        
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        
        print("Concurrent k-NN stress test: \(elapsed)s for \(queryCount) queries")
        XCTAssertEqual(results.count, queryCount)
        
        // Verify results
        for result in results {
            XCTAssertEqual(result.count, k)
            // Verify ordering
            for i in 1..<result.count {
                XCTAssertLessThanOrEqual(result[i-1].distance, result[i].distance)
            }
        }
    }
    
    // MARK: - Deadlock Prevention
    
    func testDeadlockPrevention() async {
        // Test that our async operations don't cause deadlocks
        let timeout: TimeInterval = 5.0
        let expectation = XCTestExpectation(description: "No deadlock")
        
        Task {
            // Nested async operations
            let vectors = (0..<100).map { _ in Vector512.random() }
            
            let _ = await BatchOperations.findNearest(
                to: vectors[0],
                in: vectors,
                k: 10
            )
            
            let _ = await BatchOperations.pairwiseDistances(
                Array(vectors.prefix(20))
            )
            
            expectation.fulfill()
        }
        
        // Should complete without timeout
        await fulfillment(of: [expectation], timeout: timeout)
    }
    
    // MARK: - Memory Consistency
    
    func testMemoryConsistency() async {
        // Test that concurrent operations maintain memory consistency
        let sharedVector = Vector1536.random()
        let iterations = 100
        
        await withTaskGroup(of: Bool.self) { group in
            // Multiple readers
            for _ in 0..<50 {
                group.addTask {
                    for _ in 0..<iterations {
                        let magnitude = sharedVector.magnitude
                        let normalized = sharedVector.normalized()
                        
                        // Verify consistency
                        if sharedVector.magnitude > 1e-6 {
                            XCTAssertEqual(normalized.magnitude, 1.0, accuracy: 1e-4)
                        }
                    }
                    return true
                }
            }
            
            for await _ in group {
                // All tasks completed successfully
            }
        }
    }
}