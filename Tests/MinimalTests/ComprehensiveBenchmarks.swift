import XCTest
@testable import VectorCore

/// Opt-in timing observations of VectorCore operations (VECTORCORE_TEST_EXTENDED=1). Most timed bodies discard their
/// results (`_ = ...`), so these are observations, not performance evidence (claim ledger recordedSites, Q17).
final class ComprehensiveBenchmarks: XCTestCase {
    // Skip these heavy benchmarks unless explicitly enabled
    override func setUpWithError() throws {
        let enabled = ProcessInfo.processInfo.environment["VECTORCORE_TEST_EXTENDED"] == "1"
        if !enabled {
            throw XCTSkip("Performance benchmarks disabled (set VECTORCORE_TEST_EXTENDED=1 to enable)")
        }
    }

    // MARK: - Test Configuration

    let iterations = 10_000
    let smallIterations = 1_000

    // MARK: - Vector512 Benchmarks

    func testVector512OptimizedDotProductPerformance() {
        let v1 = Vector512Optimized(repeating: 1.0)
        let v2 = Vector512Optimized(repeating: 2.0)

        observePerformance("ComprehensiveBenchmarks.testVector512OptimizedDotProductPerformance") {
            for _ in 0..<iterations {
                _ = v1.dotProduct(v2)
            }
        }
    }

    func testVector512OptimizedDistancePerformance() {
        let v1 = Vector512Optimized(repeating: 1.0)
        let v2 = Vector512Optimized(repeating: 2.0)

        observePerformance("ComprehensiveBenchmarks.testVector512OptimizedDistancePerformance") {
            for _ in 0..<iterations {
                _ = v1.euclideanDistance(to: v2)
            }
        }
    }

    func testVector512GenericDotProductPerformance() {
        let v1 = Vector<Dim512>(repeating: 1.0)
        let v2 = Vector<Dim512>(repeating: 2.0)

        observePerformance("ComprehensiveBenchmarks.testVector512GenericDotProductPerformance") {
            for _ in 0..<iterations {
                _ = v1.dotProduct(v2)
            }
        }
    }

    func testVector512GenericDistancePerformance() {
        let v1 = Vector<Dim512>(repeating: 1.0)
        let v2 = Vector<Dim512>(repeating: 2.0)

        observePerformance("ComprehensiveBenchmarks.testVector512GenericDistancePerformance") {
            for _ in 0..<iterations {
                _ = v1.euclideanDistance(to: v2)
            }
        }
    }

    // MARK: - Vector768 Benchmarks

    func testVector768OptimizedDotProductPerformance() {
        let v1 = Vector768Optimized(repeating: 1.0)
        let v2 = Vector768Optimized(repeating: 2.0)

        observePerformance("ComprehensiveBenchmarks.testVector768OptimizedDotProductPerformance") {
            for _ in 0..<iterations {
                _ = v1.dotProduct(v2)
            }
        }
    }

    func testVector768OptimizedDistancePerformance() {
        let v1 = Vector768Optimized(repeating: 1.0)
        let v2 = Vector768Optimized(repeating: 2.0)

        observePerformance("ComprehensiveBenchmarks.testVector768OptimizedDistancePerformance") {
            for _ in 0..<iterations {
                _ = v1.euclideanDistance(to: v2)
            }
        }
    }

    func testVector768GenericDotProductPerformance() {
        let v1 = Vector<Dim768>(repeating: 1.0)
        let v2 = Vector<Dim768>(repeating: 2.0)

        observePerformance("ComprehensiveBenchmarks.testVector768GenericDotProductPerformance") {
            for _ in 0..<iterations {
                _ = v1.dotProduct(v2)
            }
        }
    }

    func testVector768GenericDistancePerformance() {
        let v1 = Vector<Dim768>(repeating: 1.0)
        let v2 = Vector<Dim768>(repeating: 2.0)

        observePerformance("ComprehensiveBenchmarks.testVector768GenericDistancePerformance") {
            for _ in 0..<iterations {
                _ = v1.euclideanDistance(to: v2)
            }
        }
    }

    // MARK: - Vector1536 Benchmarks

    func testVector1536OptimizedDotProductPerformance() {
        let v1 = Vector1536Optimized(repeating: 1.0)
        let v2 = Vector1536Optimized(repeating: 2.0)

        observePerformance("ComprehensiveBenchmarks.testVector1536OptimizedDotProductPerformance") {
            for _ in 0..<smallIterations {
                _ = v1.dotProduct(v2)
            }
        }
    }

    func testVector1536OptimizedDistancePerformance() {
        let v1 = Vector1536Optimized(repeating: 1.0)
        let v2 = Vector1536Optimized(repeating: 2.0)

        observePerformance("ComprehensiveBenchmarks.testVector1536OptimizedDistancePerformance") {
            for _ in 0..<smallIterations {
                _ = v1.euclideanDistance(to: v2)
            }
        }
    }

    func testVector1536GenericDotProductPerformance() {
        let v1 = Vector<Dim1536>(repeating: 1.0)
        let v2 = Vector<Dim1536>(repeating: 2.0)

        observePerformance("ComprehensiveBenchmarks.testVector1536GenericDotProductPerformance") {
            for _ in 0..<smallIterations {
                _ = v1.dotProduct(v2)
            }
        }
    }

    func testVector1536GenericDistancePerformance() {
        let v1 = Vector<Dim1536>(repeating: 1.0)
        let v2 = Vector<Dim1536>(repeating: 2.0)

        observePerformance("ComprehensiveBenchmarks.testVector1536GenericDistancePerformance") {
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

        observePerformance("ComprehensiveBenchmarks.testOptimizedEuclideanDistancePerformance") {
            for _ in 0..<iterations {
                _ = metric.distance(v1, v2)
            }
        }
    }

    func testOptimizedCosineDistancePerformance() {
        let v1 = Vector512Optimized(repeating: 1.0)
        let v2 = Vector512Optimized(repeating: 2.0)
        let metric = CosineDistance()

        observePerformance("ComprehensiveBenchmarks.testOptimizedCosineDistancePerformance") {
            for _ in 0..<iterations {
                _ = metric.distance(v1, v2)
            }
        }
    }

    func testOptimizedManhattanDistancePerformance() {
        let v1 = Vector512Optimized(repeating: 1.0)
        let v2 = Vector512Optimized(repeating: 2.0)
        let metric = ManhattanDistance()

        observePerformance("ComprehensiveBenchmarks.testOptimizedManhattanDistancePerformance") {
            for _ in 0..<iterations {
                _ = metric.distance(v1, v2)
            }
        }
    }

    func testOptimizedDotProductDistancePerformance() {
        let v1 = Vector512Optimized(repeating: 1.0)
        let v2 = Vector512Optimized(repeating: 2.0)
        let metric = DotProductDistance()

        observePerformance("ComprehensiveBenchmarks.testOptimizedDotProductDistancePerformance") {
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

        observePerformance("ComprehensiveBenchmarks.testBatchDistanceOptimizedPerformance") {
            _ = metric.batchDistance(query: query, candidates: candidates)
        }
    }

    func testBatchDistanceGenericPerformance() {
        let query = Vector<Dim512>(repeating: 1.0)
        let candidates = (0..<100).map { _ in Vector<Dim512>(repeating: 2.0) }
        let metric = EuclideanDistance()

        observePerformance("ComprehensiveBenchmarks.testBatchDistanceGenericPerformance") {
            _ = metric.batchDistance(query: query, candidates: candidates)
        }
    }

    // MARK: - Normalization Benchmarks

    func testVector512OptimizedNormalizationPerformance() {
        let vector = Vector512Optimized(repeating: 2.0)

        observePerformance("ComprehensiveBenchmarks.testVector512OptimizedNormalizationPerformance") {
            for _ in 0..<iterations {
                _ = try? vector.normalized().get()
            }
        }
    }

    func testVector768OptimizedNormalizationPerformance() {
        let vector = Vector768Optimized(repeating: 2.0)

        observePerformance("ComprehensiveBenchmarks.testVector768OptimizedNormalizationPerformance") {
            for _ in 0..<iterations {
                _ = try? vector.normalized().get()
            }
        }
    }

    func testVector1536OptimizedNormalizationPerformance() {
        let vector = Vector1536Optimized(repeating: 2.0)

        observePerformance("ComprehensiveBenchmarks.testVector1536OptimizedNormalizationPerformance") {
            for _ in 0..<smallIterations {
                _ = try? vector.normalized().get()
            }
        }
    }

    // MARK: - Magnitude Traversal Timing

    func testVector512OptimizedMagnitudeTraversalTiming() {
        let vectors = (0..<1000).map { _ in
            Vector512Optimized(repeating: Float.random(in: -1...1))
        }

        observePerformance("ComprehensiveBenchmarks.testVector512OptimizedMagnitudeTraversalTiming") {
            var sum: Float = 0
            for v in vectors {
                sum += v.magnitude
            }
            XCTAssertNotEqual(sum, 0) // Anti-elision: the assertion reads the sum of every magnitude
        }
    }

    func testVector512GenericMagnitudeTraversalTiming() {
        let vectors = (0..<1000).map { _ in
            Vector<Dim512>(repeating: Float.random(in: -1...1))
        }

        observePerformance("ComprehensiveBenchmarks.testVector512GenericMagnitudeTraversalTiming") {
            var sum: Float = 0
            for v in vectors {
                sum += v.magnitude
            }
            XCTAssertNotEqual(sum, 0) // Anti-elision: the assertion reads the sum of every magnitude
        }
    }

    // MARK: - Arithmetic Operations

    func testVector512OptimizedAdditionPerformance() {
        let v1 = Vector512Optimized(repeating: 1.0)
        let v2 = Vector512Optimized(repeating: 2.0)

        observePerformance("ComprehensiveBenchmarks.testVector512OptimizedAdditionPerformance") {
            for _ in 0..<iterations {
                _ = v1 + v2
            }
        }
    }

    func testVector512OptimizedMultiplicationPerformance() {
        let v1 = Vector512Optimized(repeating: 1.0)

        observePerformance("ComprehensiveBenchmarks.testVector512OptimizedMultiplicationPerformance") {
            for _ in 0..<iterations {
                _ = v1 * 2.5
            }
        }
    }

    func testVector512OptimizedHadamardPerformance() {
        let v1 = Vector512Optimized(repeating: 1.0)
        let v2 = Vector512Optimized(repeating: 2.0)

        observePerformance("ComprehensiveBenchmarks.testVector512OptimizedHadamardPerformance") {
            for _ in 0..<iterations {
                _ = v1 .* v2
            }
        }
    }
}
