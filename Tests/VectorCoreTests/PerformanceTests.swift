// VectorCoreTests: Performance Tests
//
// Performance benchmarks for VectorCore
//

import XCTest
@testable import VectorCore

final class PerformanceTests: XCTestCase {
    
    // MARK: - Test Data
    
    let iterations = 1000
    let batchSize = 100
    
    // MARK: - Vector Operation Performance
    
    func testVector512DotProductPerformance() {
        let a = Vector512(repeating: 0.5)
        let b = Vector512(repeating: 0.7)
        
        measure {
            for _ in 0..<iterations {
                _ = a.dotProduct(b)
            }
        }
    }
    
    func testVector768DotProductPerformance() {
        let a = Vector768(repeating: 0.3)
        let b = Vector768(repeating: 0.8)
        
        measure {
            for _ in 0..<iterations {
                _ = a.dotProduct(b)
            }
        }
    }
    
    func testVector1536DotProductPerformance() {
        let a = Vector1536(repeating: 0.2)
        let b = Vector1536(repeating: 0.9)
        
        measure {
            for _ in 0..<iterations {
                _ = a.dotProduct(b)
            }
        }
    }
    
    func testVectorNormalizationPerformance() {
        let values = (0..<512).map { Float($0 + 1) }
        
        measure {
            for _ in 0..<iterations {
                var vector = Vector512(values)
                vector.normalize()
            }
        }
    }
    
    func testVectorDistancePerformance() {
        let a = Vector512(repeating: 0.1)
        let b = Vector512(repeating: 0.9)
        
        measure {
            for _ in 0..<iterations {
                _ = a.distance(to: b)
            }
        }
    }
    
    // MARK: - Distance Metric Performance
    
    func testEuclideanDistancePerformance() {
        let metric = EuclideanDistance()
        let vectors = (0..<batchSize).map { _ in
            Vector512(repeating: Float.random(in: 0...1))
        }
        let query = Vector512(repeating: 0.5)
        
        measure {
            for vector in vectors {
                _ = metric.distance(query, vector)
            }
        }
    }
    
    func testCosineDistancePerformance() {
        let metric = CosineDistance()
        let vectors = (0..<batchSize).map { _ in
            Vector768(repeating: Float.random(in: 0...1))
        }
        let query = Vector768(repeating: 0.5)
        
        measure {
            for vector in vectors {
                _ = metric.distance(query, vector)
            }
        }
    }
    
    func testBatchDistancePerformance() {
        let metric = EuclideanDistance()
        let query = Vector256(repeating: 0.5)
        let candidates = (0..<batchSize).map { _ in
            Vector256(repeating: Float.random(in: 0...1))
        }
        
        measure {
            _ = metric.batchDistance(query: query, candidates: candidates)
        }
    }
    
    // MARK: - Memory Operation Performance
    
    func testAlignedMemoryAllocationPerformance() {
        measure {
            for _ in 0..<iterations {
                let memory = Memory.AlignedBuffer<Float>(capacity: 512, alignment: 64)
                // Force deallocation
                _ = memory
            }
        }
    }
    
    // Removed: testVectorArrayLayoutPerformance - optimization code was removed
    
    func testMemoryPoolPerformance() {
        let pool = Memory.Pool<Float>(capacity: 512)
        
        measure {
            for _ in 0..<iterations {
                let buffer = pool.acquire()
                // Simulate some work
                buffer.baseAddress?[0] = 1.0
                pool.release(buffer)
            }
        }
    }
    
    // MARK: - Serialization Performance
    
    func testVector512SerializationPerformance() {
        let vector = Vector512(repeating: 0.5)
        
        measure {
            for _ in 0..<iterations {
                do {
                    let serialized = try vector.serialize()
                    _ = try Vector512.deserialize(from: serialized)
                } catch {
                    XCTFail("Serialization failed: \(error)")
                }
            }
        }
    }
    
    func testJSONSerializationPerformance() {
        let vector = Vector256(repeating: 0.7)
        
        measure {
            for _ in 0..<100 { // Fewer iterations for JSON
                do {
                    let json = try vector.serializeToJSON()
                    _ = try Vector256.deserializeFromJSON(json)
                } catch {
                    XCTFail("JSON serialization failed: \(error)")
                }
            }
        }
    }
    
    // MARK: - Batch Operation Performance
    
    func testBatchVectorCreationPerformance() {
        let flatArray = (0..<51200).map { Float($0) } // 100 Vector512s
        
        measure {
            _ = Vector512.createBatch(from: flatArray)
        }
    }
    
    // Removed: testOptimalBatchProcessingPerformance - optimization code was removed
    
    // MARK: - Optimization Utility Performance
    
    // Removed: testPrefetchPerformance - optimization code was removed
    
    // Removed: testCacheAwareMatrixMultiplyPerformance - optimization code was removed
    
    // MARK: - Comparison Tests
    
    func testOptimizedVsBasicDistance() {
        let a = Vector512(repeating: 0.3)
        let b = Vector512(repeating: 0.7)
        
        // Test that optimized version produces same result
        let euclidean = EuclideanDistance()
        let optimizedDist = euclidean.distance(a, b)
        
        // Manual calculation
        var sum: Float = 0
        for i in 0..<512 {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        let manualDist = sqrt(sum)
        
        XCTAssertEqual(optimizedDist, manualDist, accuracy: 0.0001)
        
        // Performance comparison
        let optimizedTime = measureTime {
            for _ in 0..<iterations {
                _ = euclidean.distance(a, b)
            }
        }
        
        let manualTime = measureTime {
            for _ in 0..<iterations {
                var localSum: Float = 0
                for i in 0..<512 {
                    let diff = a[i] - b[i]
                    localSum += diff * diff
                }
                _ = sqrt(localSum)
            }
        }
        
        print("Optimized: \(optimizedTime)s, Manual: \(manualTime)s")
        print("Speedup: \(manualTime / optimizedTime)x")
    }
    
    // MARK: - Utility Functions
    
    private func measureTime(block: () -> Void) -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        block()
        let end = CFAbsoluteTimeGetCurrent()
        return end - start
    }
}