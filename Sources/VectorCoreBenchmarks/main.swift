// VectorCore Benchmarks
//
// Performance benchmarks for all distance metrics
//

import Foundation
import VectorCore

// MARK: - Benchmark Utilities

struct BenchmarkResult {
    let name: String
    let dimension: Int
    let iterations: Int
    let totalTime: TimeInterval
    let averageTime: TimeInterval
    let throughput: Double // operations per second
    
    var formattedReport: String {
        """
        \(name) (dim=\(dimension)):
          Total time: \(String(format: "%.3f", totalTime))s
          Avg time: \(String(format: "%.6f", averageTime * 1000))ms
          Throughput: \(String(format: "%.0f", throughput)) ops/sec
        """
    }
}

class Benchmark {
    static func measure(
        name: String,
        dimension: Int,
        iterations: Int = 10000,
        warmup: Int = 100,
        operation: () -> Void
    ) -> BenchmarkResult {
        // Warmup
        for _ in 0..<warmup {
            operation()
        }
        
        // Measure
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            operation()
        }
        let totalTime = CFAbsoluteTimeGetCurrent() - start
        
        return BenchmarkResult(
            name: name,
            dimension: dimension,
            iterations: iterations,
            totalTime: totalTime,
            averageTime: totalTime / Double(iterations),
            throughput: Double(iterations) / totalTime
        )
    }
}

// MARK: - Distance Metric Benchmarks

func benchmarkDistanceMetrics() {
    print("=== Distance Metric Benchmarks ===\n")
    
    // Test with different vector dimensions
    print("\n--- Vector128 (SIMD optimized) ---")
    let v128_1 = Vector128.random(in: -1...1)
    let v128_2 = Vector128.random(in: -1...1)
    let vectors128 = (0..<100).map { _ in Vector128.random(in: -1...1) }
    
    benchmarkMetricsForVectors(v128_1, v128_2, vectors: vectors128, dimension: 128)
    
    print("\n--- Vector256 (SIMD optimized) ---")
    let v256_1 = Vector256.random(in: -1...1)
    let v256_2 = Vector256.random(in: -1...1)
    let vectors256 = (0..<100).map { _ in Vector256.random(in: -1...1) }
    
    benchmarkMetricsForVectors(v256_1, v256_2, vectors: vectors256, dimension: 256)
    
    print("\n--- Vector512 (SIMD optimized) ---")
    let v512_1 = Vector512.random(in: -1...1)
    let v512_2 = Vector512.random(in: -1...1)
    let vectors512 = (0..<100).map { _ in Vector512.random(in: -1...1) }
    
    benchmarkMetricsForVectors(v512_1, v512_2, vectors: vectors512, dimension: 512)
}

func benchmarkMetricsForVectors<V: ExtendedVectorProtocol>(_ v1: V, _ v2: V, vectors: [V], dimension: Int) {
    let metrics: [(String, any DistanceMetric)] = [
        ("Euclidean", EuclideanDistance()),
        ("Cosine", CosineDistance()),
        ("DotProduct", DotProductDistance()),
        ("Manhattan", ManhattanDistance()),
        ("Chebyshev", ChebyshevDistance()),
        ("Hamming", HammingDistance()),
        ("Minkowski(p=3)", MinkowskiDistance(p: 3)),
        ("Jaccard", JaccardDistance())
    ]
    
    for (name, metric) in metrics {
        // Single distance computation
        let singleResult = Benchmark.measure(
            name: "\(name) (single)",
            dimension: dimension,
            iterations: 10000
        ) {
            _ = metric.distance(v1, v2)
        }
        print(singleResult.formattedReport)
        
        // Batch distance computation
        let batchResult = Benchmark.measure(
            name: "\(name) (batch-100)",
            dimension: dimension,
            iterations: 100
        ) {
            _ = metric.batchDistance(query: v1, candidates: vectors)
        }
        print(batchResult.formattedReport)
    }
}

// MARK: - Vector Operation Benchmarks

func benchmarkVectorOperations() {
    print("\n\n=== Vector Operation Benchmarks ===\n")
    
    // Test with Vector256 as representative
    let v1 = Vector256.random(in: -1...1)
    let v2 = Vector256.random(in: -1...1)
    
    let operations: [(String, () -> Void)] = [
        ("Addition", { _ = v1 + v2 }),
        ("Subtraction", { _ = v1 - v2 }),
        ("Scalar Multiply", { _ = v1 * 2.5 }),
        ("Scalar Divide", { _ = v1 / 2.5 }),
        ("Dot Product", { _ = v1.dotProduct(v2) }),
        ("Magnitude", { _ = v1.magnitude }),
        ("Normalize", { _ = v1.normalized() }),
        ("Distance", { _ = v1.distance(to: v2) }),
        ("Cosine Similarity", { _ = v1.cosineSimilarity(to: v2) }),
        ("Element-wise Multiply", { _ = v1 .* v2 }),
        ("Element-wise Divide", { _ = v1 ./ v2 }),
        ("L1 Norm", { _ = v1.l1Norm }),
        ("L∞ Norm", { _ = v1.lInfinityNorm }),
        ("Mean", { _ = v1.mean }),
        ("Variance", { _ = v1.variance }),
        ("Softmax", { _ = v1.softmax() }),
        ("Clamp", { _ = v1.clamped(to: -1...1) })
    ]
    
    for (name, operation) in operations {
        let result = Benchmark.measure(
            name: name,
            dimension: 256,
            iterations: 10000,
            operation: operation
        )
        print(result.formattedReport)
    }
}

// MARK: - Batch Operation Benchmarks

func benchmarkBatchOperations() {
    print("\n\n=== Batch Operation Benchmarks ===\n")
    
    let vectors = (0..<10000).map { _ in Vector256.random(in: -1...1) }
    let query = vectors[0]
    
    // k-NN search
    for k in [1, 10, 100] {
        let result = Benchmark.measure(
            name: "k-NN (k=\(k))",
            dimension: 256,
            iterations: 10
        ) {
            _ = BatchOperations.findNearest(
                to: query,
                in: vectors,
                k: k,
                metric: EuclideanDistance()
            )
        }
        print(result.formattedReport)
    }
    
    // Pairwise distances (smaller set)
    let smallSet = Array(vectors.prefix(100))
    let pairwiseResult = Benchmark.measure(
        name: "Pairwise Distances (100x100)",
        dimension: 256,
        iterations: 10
    ) {
        _ = BatchOperations.pairwiseDistances(smallSet)
    }
    print(pairwiseResult.formattedReport)
    
    // Batch transform
    let transformResult = Benchmark.measure(
        name: "Batch Normalize (10k vectors)",
        dimension: 256,
        iterations: 10
    ) {
        _ = BatchOperations.map(vectors, batchSize: 1024) { $0.normalized() }
    }
    print(transformResult.formattedReport)
}

// MARK: - Storage Type Benchmarks

func benchmarkStorageTypes() {
    print("\n\n=== Storage Type Benchmarks ===\n")
    
    // Compare different vector types for same dimension
    let dim128Values = (0..<128).map { _ in Float.random(in: -1...1) }
    
    // Type-safe Vector128
    let v128_1 = Vector128(dim128Values)
    let v128_2 = Vector128(dim128Values.reversed())
    
    let vector128Result = Benchmark.measure(
        name: "Vector128 (SIMD optimized)",
        dimension: 128,
        iterations: 100000
    ) {
        _ = v128_1.dotProduct(v128_2)
    }
    print(vector128Result.formattedReport)
    
    // Dynamic vector (using factory to ensure correct type)
    let dv1 = DynamicVector(dimension: 128, values: dim128Values)
    let dv2 = DynamicVector(dimension: 128, values: dim128Values.reversed())
    
    let dynamicResult = Benchmark.measure(
        name: "DynamicVector (array-based)",
        dimension: 128,
        iterations: 100000
    ) {
        _ = dv1.dotProduct(dv2)
    }
    print(dynamicResult.formattedReport)
    
    // Show speedup
    let speedup = dynamicResult.averageTime / vector128Result.averageTime
    print("  → Vector128 is \(String(format: "%.1fx", speedup)) faster than DynamicVector")
}

// MARK: - Memory Usage Analysis

func analyzeMemoryUsage() {
    print("\n\n=== Memory Usage Analysis ===\n")
    
    let dimensions = [32, 64, 128, 256, 512, 768, 1536, 2048]
    
    for dim in dimensions {
        // Calculate memory usage
        let floatSize = MemoryLayout<Float>.size
        let vectorSize = dim * floatSize
        let overhead: Int
        
        switch dim {
        case 1...64:
            // SmallVectorStorage uses SIMD64
            overhead = MemoryLayout<SmallVectorStorage>.size - (dim * floatSize)
        case 65...512:
            // MediumVectorStorage uses AlignedValueStorage
            overhead = MemoryLayout<MediumVectorStorage>.size - (dim * floatSize)
        default:
            // LargeVectorStorage uses COWDynamicStorage
            overhead = 24 // Approximate COW wrapper overhead
        }
        
        let totalSize = vectorSize + overhead
        let efficiency = Double(vectorSize) / Double(totalSize) * 100
        
        print("""
        Dimension \(dim):
          Data size: \(vectorSize) bytes
          Total size: ~\(totalSize) bytes
          Overhead: \(overhead) bytes
          Efficiency: \(String(format: "%.1f%%", efficiency))
        """)
    }
}

// MARK: - Performance Comparison

func compareImplementations() {
    print("\n\n=== Implementation Comparison ===\n")
    
    // Compare our Euclidean distance vs naive implementation
    let v1 = Vector256.random(in: -1...1)
    let v2 = Vector256.random(in: -1...1)
    
    // Naive implementation
    func naiveEuclidean(_ a: Vector256, _ b: Vector256) -> Float {
        var sum: Float = 0
        for i in 0..<256 {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }
    
    let naiveResult = Benchmark.measure(
        name: "Naive Euclidean",
        dimension: 256,
        iterations: 10000
    ) {
        _ = naiveEuclidean(v1, v2)
    }
    print(naiveResult.formattedReport)
    
    let optimizedResult = Benchmark.measure(
        name: "Optimized Euclidean",
        dimension: 256,
        iterations: 10000
    ) {
        _ = EuclideanDistance().distance(v1, v2)
    }
    print(optimizedResult.formattedReport)
    
    let speedup = naiveResult.averageTime / optimizedResult.averageTime
    print("  → Optimized is \(String(format: "%.1fx", speedup)) faster than naive")
    
    // Compare squared Euclidean
    let squaredResult = Benchmark.measure(
        name: "Squared Euclidean",
        dimension: 256,
        iterations: 10000
    ) {
        _ = DistanceCalculator.euclideanSquared(v1, v2)
    }
    print(squaredResult.formattedReport)
    
    let sqrtSpeedup = optimizedResult.averageTime / squaredResult.averageTime
    print("  → Squared is \(String(format: "%.1fx", sqrtSpeedup)) faster (no sqrt)")
}

// MARK: - SIMD Storage Benchmarks

func benchmarkSIMDStorageOptimizations() {
    print("\n\n=== Consolidated Storage Architecture Benchmarks ===\n")
    print("Benchmarking the three-tier storage system\n")
    
    // TODO: Implement consolidated storage benchmarks
    print("Consolidated storage benchmarks coming soon...")
}

// MARK: - Auto-Parallelization Benchmarks

func benchmarkAutoParallelization() async {
    print("\n\n=== Auto-Parallelization Benchmarks ===\n")
    
    await runAutoParallelizationBenchmarks()
}

// MARK: - Main

print("VectorCore Performance Benchmarks")
print("=================================")
print("Platform: \(ProcessInfo.processInfo.operatingSystemVersionString)")
print("Processor: \(ProcessInfo.processInfo.processorCount) cores")
print("Date: \(Date())\n")

// Check if we should run only specific benchmarks
let args = CommandLine.arguments
if args.count > 1 {
    switch args[1] {
    case "--simd-only":
        benchmarkSIMDStorageOptimizations()
    case "--parallel-only":
        await benchmarkAutoParallelization()
    default:
        // Run all benchmarks
        benchmarkDistanceMetrics()
        benchmarkVectorOperations()
        benchmarkBatchOperations()
        benchmarkStorageTypes()
        analyzeMemoryUsage()
        compareImplementations()
        benchmarkSIMDStorageOptimizations()
        await benchmarkAutoParallelization()
    }
} else {
    // Run all benchmarks
    benchmarkDistanceMetrics()
    benchmarkVectorOperations()
    benchmarkBatchOperations()
    benchmarkStorageTypes()
    analyzeMemoryUsage()
    compareImplementations()
    benchmarkSIMDStorageOptimizations()
    await benchmarkParallelBatchOperations()
}

print("\n\nBenchmark completed!")