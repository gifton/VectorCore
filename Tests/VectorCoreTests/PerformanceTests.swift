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
    
    // MARK: Vector Addition Performance Tests
    
    func testVector32AdditionPerformance() {
        let a = Vector<Dim32>.random(in: -1...1)
        let b = Vector<Dim32>.random(in: -1...1)
        
        measure {
            for _ in 0..<iterations {
                _ = a + b
            }
        }
    }
    
    func testVector128AdditionPerformance() {
        let a = Vector<Dim128>.random(in: -1...1)
        let b = Vector<Dim128>.random(in: -1...1)
        
        measure {
            for _ in 0..<iterations {
                _ = a + b
            }
        }
    }
    
    func testVector256AdditionPerformance() {
        let a = Vector<Dim256>.random(in: -1...1)
        let b = Vector<Dim256>.random(in: -1...1)
        
        measure {
            for _ in 0..<iterations {
                _ = a + b
            }
        }
    }
    
    func testVector512AdditionPerformance() {
        let a = Vector<Dim512>.random(in: -1...1)
        let b = Vector<Dim512>.random(in: -1...1)
        
        measure {
            for _ in 0..<iterations {
                _ = a + b
            }
        }
    }
    
    /// Compare addition performance to dot product baseline
    func testAdditionVsDotProductComparison() {
        print("\n=== Vector Addition vs Dot Product Performance ===")
        
        // Test different dimensions
        compareAdditionToDotProduct32()
        compareAdditionToDotProduct128()
        compareAdditionToDotProduct256()
        compareAdditionToDotProduct512()
    }
    
    private func compareAdditionToDotProduct32() {
        let a = Vector<Dim32>.random(in: -1...1)
        let b = Vector<Dim32>.random(in: -1...1)
        compareAdditionToDotProduct(a: a, b: b, dimension: 32)
    }
    
    private func compareAdditionToDotProduct128() {
        let a = Vector<Dim128>.random(in: -1...1)
        let b = Vector<Dim128>.random(in: -1...1)
        compareAdditionToDotProduct(a: a, b: b, dimension: 128)
    }
    
    private func compareAdditionToDotProduct256() {
        let a = Vector<Dim256>.random(in: -1...1)
        let b = Vector<Dim256>.random(in: -1...1)
        compareAdditionToDotProduct(a: a, b: b, dimension: 256)
    }
    
    private func compareAdditionToDotProduct512() {
        let a = Vector<Dim512>.random(in: -1...1)
        let b = Vector<Dim512>.random(in: -1...1)
        compareAdditionToDotProduct(a: a, b: b, dimension: 512)
    }
    
    private func compareAdditionToDotProduct<V: VectorType>(
        a: V,
        b: V,
        dimension: Int
    ) where V: ExtendedVectorProtocol {
        let testIterations = 5000
        
        // Measure addition - need to cast to specific Vector type
        var additionTime: TimeInterval = 0
        
        if let va = a as? Vector<Dim32>, let vb = b as? Vector<Dim32> {
            additionTime = measureTime {
                for _ in 0..<testIterations {
                    _ = va + vb
                }
            }
        } else if let va = a as? Vector<Dim128>, let vb = b as? Vector<Dim128> {
            additionTime = measureTime {
                for _ in 0..<testIterations {
                    _ = va + vb
                }
            }
        } else if let va = a as? Vector<Dim256>, let vb = b as? Vector<Dim256> {
            additionTime = measureTime {
                for _ in 0..<testIterations {
                    _ = va + vb
                }
            }
        } else if let va = a as? Vector<Dim512>, let vb = b as? Vector<Dim512> {
            additionTime = measureTime {
                for _ in 0..<testIterations {
                    _ = va + vb
                }
            }
        }
        
        // Measure dot product
        let dotProductTime = measureTime {
            for _ in 0..<testIterations {
                _ = a.dotProduct(b)
            }
        }
        
        let additionOpsPerSec = Double(testIterations) / additionTime
        let dotProductOpsPerSec = Double(testIterations) / dotProductTime
        let ratio = additionOpsPerSec / dotProductOpsPerSec
        
        print("\nDimension \(dimension):")
        print("  Addition: \(String(format: "%.2f", additionOpsPerSec / 1_000_000))M ops/sec")
        print("  Dot Product: \(String(format: "%.2f", dotProductOpsPerSec / 1_000_000))M ops/sec")
        print("  Ratio: \(String(format: "%.2f%%", ratio * 100)) (addition vs dot product)")
        
        if dimension >= 128 && ratio < 0.5 {
            print("  ⚠️ WARNING: Addition is \(String(format: "%.1f", 1.0/ratio))x slower!")
        }
    }
    
    private func createRandomVector<V: ExtendedVectorProtocol>(
        type: V.Type,
        dimension: Int
    ) -> V where V: Equatable {
        let values = (0..<dimension).map { _ in Float.random(in: -1...1) }
        return V(from: values)
    }
    
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
                // Use AlignedMemory from Storage/AlignedMemory.swift
                let pointer = AlignedMemory.allocateAligned(count: 512, alignment: 64)
                defer { pointer.deallocate() }
                // Force use
                pointer[0] = 1.0
            }
        }
    }
    
    // Removed: testVectorArrayLayoutPerformance - optimization code was removed
    
    func testMemoryPoolPerformance() {
        let pool = MemoryPool()
        
        measure {
            for _ in 0..<iterations {
                if let handle = pool.acquire(type: Float.self, count: 512) {
                    // Simulate some work
                    handle.pointer[0] = 1.0
                    // BufferHandle automatically releases on deinit
                }
            }
        }
    }
    
    // MARK: - Serialization Performance
    
    func testVector512SerializationPerformance() {
        let vector = Vector512(repeating: 0.5)
        
        measure {
            for _ in 0..<iterations {
                do {
                    let serialized = try vector.encodeBinary()
                    _ = try Vector512.decodeBinary(from: serialized)
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
                    // JSON serialization not implemented
                    let encoder = JSONEncoder()
                    let data = try encoder.encode(vector)
                    let decoder = JSONDecoder()
                    _ = try decoder.decode(Vector256.self, from: data)
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
            var vectors: [Vector512] = []
            vectors.reserveCapacity(flatArray.count / 512)
            for i in stride(from: 0, to: flatArray.count, by: 512) {
                let slice = Array(flatArray[i..<min(i+512, flatArray.count)])
                if slice.count == 512 {
                    vectors.append(Vector512(slice))
                }
            }
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