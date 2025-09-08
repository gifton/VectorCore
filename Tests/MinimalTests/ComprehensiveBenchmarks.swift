import XCTest
@testable import VectorCore

/// Comprehensive benchmark suite for VectorCore optimizations
final class ComprehensiveBenchmarks: XCTestCase {
    
    // MARK: - Test Configuration
    
    let iterations = 10_000
    let smallIterations = 1_000
    
    // MARK: - Vector512 Benchmarks
    
    func testVector512OptimizedDotProductPerformance() {
        let v1 = Vector512Optimized(repeating: 1.0)
        let v2 = Vector512Optimized(repeating: 2.0)
        
        measure {
            for _ in 0..<iterations {
                _ = v1.dotProduct(v2)
            }
        }
    }
    
    func testVector512OptimizedDistancePerformance() {
        let v1 = Vector512Optimized(repeating: 1.0)
        let v2 = Vector512Optimized(repeating: 2.0)
        
        measure {
            for _ in 0..<iterations {
                _ = v1.euclideanDistance(to: v2)
            }
        }
    }
    
    func testVector512GenericDotProductPerformance() {
        let v1 = Vector<Dim512>(repeating: 1.0)
        let v2 = Vector<Dim512>(repeating: 2.0)
        
        measure {
            for _ in 0..<iterations {
                _ = v1.dotProduct(v2)
            }
        }
    }
    
    func testVector512GenericDistancePerformance() {
        let v1 = Vector<Dim512>(repeating: 1.0)
        let v2 = Vector<Dim512>(repeating: 2.0)
        
        measure {
            for _ in 0..<iterations {
                _ = v1.euclideanDistance(to: v2)
            }
        }
    }
    
    // MARK: - Vector768 Benchmarks
    
    func testVector768OptimizedDotProductPerformance() {
        let v1 = Vector768Optimized(repeating: 1.0)
        let v2 = Vector768Optimized(repeating: 2.0)
        
        measure {
            for _ in 0..<iterations {
                _ = v1.dotProduct(v2)
            }
        }
    }
    
    func testVector768OptimizedDistancePerformance() {
        let v1 = Vector768Optimized(repeating: 1.0)
        let v2 = Vector768Optimized(repeating: 2.0)
        
        measure {
            for _ in 0..<iterations {
                _ = v1.euclideanDistance(to: v2)
            }
        }
    }
    
    func testVector768GenericDotProductPerformance() {
        let v1 = Vector<Dim768>(repeating: 1.0)
        let v2 = Vector<Dim768>(repeating: 2.0)
        
        measure {
            for _ in 0..<iterations {
                _ = v1.dotProduct(v2)
            }
        }
    }
    
    func testVector768GenericDistancePerformance() {
        let v1 = Vector<Dim768>(repeating: 1.0)
        let v2 = Vector<Dim768>(repeating: 2.0)
        
        measure {
            for _ in 0..<iterations {
                _ = v1.euclideanDistance(to: v2)
            }
        }
    }
    
    // MARK: - Vector1536 Benchmarks
    
    func testVector1536OptimizedDotProductPerformance() {
        let v1 = Vector1536Optimized(repeating: 1.0)
        let v2 = Vector1536Optimized(repeating: 2.0)
        
        measure {
            for _ in 0..<smallIterations {
                _ = v1.dotProduct(v2)
            }
        }
    }
    
    func testVector1536OptimizedDistancePerformance() {
        let v1 = Vector1536Optimized(repeating: 1.0)
        let v2 = Vector1536Optimized(repeating: 2.0)
        
        measure {
            for _ in 0..<smallIterations {
                _ = v1.euclideanDistance(to: v2)
            }
        }
    }
    
    func testVector1536GenericDotProductPerformance() {
        let v1 = Vector<Dim1536>(repeating: 1.0)
        let v2 = Vector<Dim1536>(repeating: 2.0)
        
        measure {
            for _ in 0..<smallIterations {
                _ = v1.dotProduct(v2)
            }
        }
    }
    
    func testVector1536GenericDistancePerformance() {
        let v1 = Vector<Dim1536>(repeating: 1.0)
        let v2 = Vector<Dim1536>(repeating: 2.0)
        
        measure {
            for _ in 0..<smallIterations {
                _ = v1.euclideanDistance(to: v2)
            }
        }
    }
    
    // MARK: - Distance Metric Benchmarks
    
    func testOptimizedEuclideanDistancePerformance() {
        let v1 = Vector512Optimized(repeating: 1.0)
        let v2 = Vector512Optimized(repeating: 2.0)
        let metric = EuclideanDistance()
        
        measure {
            for _ in 0..<iterations {
                _ = metric.distance(v1, v2)
            }
        }
    }
    
    func testOptimizedCosineDistancePerformance() {
        let v1 = Vector512Optimized(repeating: 1.0)
        let v2 = Vector512Optimized(repeating: 2.0)
        let metric = CosineDistance()
        
        measure {
            for _ in 0..<iterations {
                _ = metric.distance(v1, v2)
            }
        }
    }
    
    func testOptimizedManhattanDistancePerformance() {
        let v1 = Vector512Optimized(repeating: 1.0)
        let v2 = Vector512Optimized(repeating: 2.0)
        let metric = ManhattanDistance()
        
        measure {
            for _ in 0..<iterations {
                _ = metric.distance(v1, v2)
            }
        }
    }
    
    func testOptimizedDotProductDistancePerformance() {
        let v1 = Vector512Optimized(repeating: 1.0)
        let v2 = Vector512Optimized(repeating: 2.0)
        let metric = DotProductDistance()
        
        measure {
            for _ in 0..<iterations {
                _ = metric.distance(v1, v2)
            }
        }
    }
    
    // MARK: - Batch Operation Benchmarks
    
    func testBatchDistanceOptimizedPerformance() {
        let query = Vector512Optimized(repeating: 1.0)
        let candidates = (0..<100).map { _ in Vector512Optimized(repeating: 2.0) }
        let metric = EuclideanDistance()
        
        measure {
            _ = metric.batchDistance(query: query, candidates: candidates)
        }
    }
    
    func testBatchDistanceGenericPerformance() {
        let query = Vector<Dim512>(repeating: 1.0)
        let candidates = (0..<100).map { _ in Vector<Dim512>(repeating: 2.0) }
        let metric = EuclideanDistance()
        
        measure {
            _ = metric.batchDistance(query: query, candidates: candidates)
        }
    }
    
    // MARK: - Normalization Benchmarks
    
    func testVector512OptimizedNormalizationPerformance() {
        let vector = Vector512Optimized(repeating: 2.0)
        
        measure {
            for _ in 0..<iterations {
                _ = try? vector.normalizedThrowing()
            }
        }
    }
    
    func testVector768OptimizedNormalizationPerformance() {
        let vector = Vector768Optimized(repeating: 2.0)
        
        measure {
            for _ in 0..<iterations {
                _ = try? vector.normalizedThrowing()
            }
        }
    }
    
    func testVector1536OptimizedNormalizationPerformance() {
        let vector = Vector1536Optimized(repeating: 2.0)
        
        measure {
            for _ in 0..<smallIterations {
                _ = try? vector.normalizedThrowing()
            }
        }
    }
    
    // MARK: - Memory Performance
    
    func testVector512OptimizedMemoryFootprint() {
        let vectors = (0..<1000).map { _ in 
            Vector512Optimized(repeating: Float.random(in: -1...1))
        }
        
        measure {
            var sum: Float = 0
            for v in vectors {
                sum += v.magnitude
            }
            XCTAssertNotEqual(sum, 0) // Prevent optimization
        }
    }
    
    func testVector512GenericMemoryFootprint() {
        let vectors = (0..<1000).map { _ in 
            Vector<Dim512>(repeating: Float.random(in: -1...1))
        }
        
        measure {
            var sum: Float = 0
            for v in vectors {
                sum += v.magnitude
            }
            XCTAssertNotEqual(sum, 0) // Prevent optimization
        }
    }
    
    // MARK: - Arithmetic Operations
    
    func testVector512OptimizedAdditionPerformance() {
        let v1 = Vector512Optimized(repeating: 1.0)
        let v2 = Vector512Optimized(repeating: 2.0)
        
        measure {
            for _ in 0..<iterations {
                _ = v1 + v2
            }
        }
    }
    
    func testVector512OptimizedMultiplicationPerformance() {
        let v1 = Vector512Optimized(repeating: 1.0)
        
        measure {
            for _ in 0..<iterations {
                _ = v1 * 2.5
            }
        }
    }
    
    func testVector512OptimizedHadamardPerformance() {
        let v1 = Vector512Optimized(repeating: 1.0)
        let v2 = Vector512Optimized(repeating: 2.0)
        
        measure {
            for _ in 0..<iterations {
                _ = v1 .* v2
            }
        }
    }
}
