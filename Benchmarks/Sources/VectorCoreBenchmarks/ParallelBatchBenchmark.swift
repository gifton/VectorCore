// VectorCore: Auto-Parallelization Benchmark
//
// Performance benchmarks demonstrating automatic parallelization behavior
//

import Foundation
@testable import VectorCore

/// Benchmark suite for auto-parallelization in BatchOperations
///
/// Demonstrates how BatchOperations automatically switches between serial and
/// parallel execution based on dataset size, achieving optimal performance
/// without manual intervention.
public struct AutoParallelizationBenchmark {
    
    // MARK: - Configuration
    
    public struct Configuration: Sendable {
        public let vectorDimensions: Int
        public let smallDatasetSizes: [Int]  // Below parallelization threshold
        public let largeDatasetSizes: [Int]  // Above parallelization threshold
        public let kValues: [Int]  // For k-NN benchmarks
        public let iterations: Int
        
        public static let `default` = Configuration(
            vectorDimensions: 256,
            smallDatasetSizes: [100, 500, 900],  // Below 1000 threshold
            largeDatasetSizes: [1500, 5000, 10000, 50000],  // Above 1000 threshold
            kValues: [10, 50, 100],
            iterations: 5
        )
        
        public init(vectorDimensions: Int, smallDatasetSizes: [Int], largeDatasetSizes: [Int], kValues: [Int], iterations: Int) {
            self.vectorDimensions = vectorDimensions
            self.smallDatasetSizes = smallDatasetSizes
            self.largeDatasetSizes = largeDatasetSizes
            self.kValues = kValues
            self.iterations = iterations
        }
    }
    
    // MARK: - Results
    
    public struct BenchmarkResult: Sendable {
        public let operation: String
        public let datasetSize: Int
        public let avgTime: TimeInterval
        public let isParallel: Bool  // Whether this used parallel execution
        public let opsPerSecond: Double
        
        public var description: String {
            String(format: "%@ (n=%d): %.3fs [%@] - %.0f ops/sec",
                   operation, datasetSize, avgTime, 
                   isParallel ? "Parallel" : "Serial",
                   opsPerSecond)
        }
    }
    
    // MARK: - Benchmarks
    
    /// Run all benchmarks and return results
    public static func runBenchmarks(configuration: Configuration = .default) async -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []
        let coreCount = ProcessInfo.processInfo.activeProcessorCount
        
        print("Running Auto-Parallelization Benchmarks")
        print("=====================================")
        print("System has \(coreCount) CPU cores")
        print("Vector dimensions: \(configuration.vectorDimensions)")
        print("Parallel threshold: \(BatchOperations.configuration.parallelThreshold) vectors")
        print("")
        
        // Test small datasets (should run serially)
        print("Small Datasets (Serial Execution Expected):")
        for size in configuration.smallDatasetSizes {
            let vectors = generateTestVectors(count: size, dimensions: configuration.vectorDimensions)
            
            if let result = await benchmarkFindNearest(vectors: vectors, k: 10, iterations: configuration.iterations) {
                results.append(result)
                print("  \(result.description)")
            }
            
            if let result = await benchmarkMap(vectors: vectors, iterations: configuration.iterations) {
                results.append(result)
                print("  \(result.description)")
            }
        }
        
        print("\nLarge Datasets (Parallel Execution Expected):")
        for size in configuration.largeDatasetSizes {
            let vectors = generateTestVectors(count: size, dimensions: configuration.vectorDimensions)
            
            for k in configuration.kValues where k < size {
                if let result = await benchmarkFindNearest(vectors: vectors, k: k, iterations: configuration.iterations) {
                    results.append(result)
                    print("  \(result.description)")
                }
            }
            
            if let result = await benchmarkMap(vectors: vectors, iterations: configuration.iterations) {
                results.append(result)
                print("  \(result.description)")
            }
        }
        
        // Test pairwise distances (threshold at 100)
        print("\nPairwise Distances (Threshold: 100 vectors):")
        for size in [50, 150, 300] {
            let vectors = Array(generateTestVectors(count: size, dimensions: configuration.vectorDimensions))
            if let result = await benchmarkPairwiseDistances(vectors: vectors, iterations: configuration.iterations) {
                results.append(result)
                print("  \(result.description)")
            }
        }
        
        // Print analysis
        printAnalysis(results: results)
        
        return results
    }
    
    // MARK: - Individual Benchmarks
    
    private static func benchmarkFindNearest(vectors: [Vector<Dim256>], k: Int, iterations: Int) async -> BenchmarkResult? {
        let query = Vector<Dim256>(repeating: 0.5)
        
        // Warm up
        _ = await BatchOperations.findNearest(to: query, in: vectors, k: k)
        
        // Benchmark
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = await BatchOperations.findNearest(to: query, in: vectors, k: k)
        }
        let avgTime = (CFAbsoluteTimeGetCurrent() - start) / Double(iterations)
        
        return BenchmarkResult(
            operation: "FindNearest(k=\(k))",
            datasetSize: vectors.count,
            avgTime: avgTime,
            isParallel: vectors.count >= BatchOperations.configuration.parallelThreshold,
            opsPerSecond: 1.0 / avgTime
        )
    }
    
    private static func benchmarkMap(vectors: [Vector<Dim256>], iterations: Int) async -> BenchmarkResult? {
        // Warm up
        _ = try? await BatchOperations.map(vectors) { $0.normalized() }
        
        // Benchmark
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = try? await BatchOperations.map(vectors) { $0.normalized() }
        }
        let avgTime = (CFAbsoluteTimeGetCurrent() - start) / Double(iterations)
        
        return BenchmarkResult(
            operation: "Map(normalize)",
            datasetSize: vectors.count,
            avgTime: avgTime,
            isParallel: vectors.count >= BatchOperations.configuration.parallelThreshold,
            opsPerSecond: Double(vectors.count) / avgTime
        )
    }
    
    private static func benchmarkPairwiseDistances(vectors: [Vector<Dim256>], iterations: Int) async -> BenchmarkResult? {
        // Warm up
        _ = await BatchOperations.pairwiseDistances(vectors)
        
        // Benchmark
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = await BatchOperations.pairwiseDistances(vectors)
        }
        let avgTime = (CFAbsoluteTimeGetCurrent() - start) / Double(iterations)
        
        let operationCount = vectors.count * vectors.count
        return BenchmarkResult(
            operation: "PairwiseDistances",
            datasetSize: vectors.count,
            avgTime: avgTime,
            isParallel: vectors.count >= 100,  // Pairwise has different threshold
            opsPerSecond: Double(operationCount) / avgTime
        )
    }
    
    // MARK: - Helpers
    
    private static func generateTestVectors(count: Int, dimensions: Int) -> [Vector<Dim256>] {
        guard dimensions == 256 else {
            fatalError("Benchmark currently supports only 256-dimensional vectors")
        }
        
        return (0..<count).map { i in
            let values = (0..<dimensions).map { j in
                sin(Float(i * dimensions + j) / 1000.0)
            }
            return Vector<Dim256>(values)
        }
    }
    
    private static func printAnalysis(results: [BenchmarkResult]) {
        print("\n=== Performance Analysis ===")
        
        // Analyze serial vs parallel performance
        let serialResults = results.filter { !$0.isParallel }
        let parallelResults = results.filter { $0.isParallel }
        
        if !serialResults.isEmpty && !parallelResults.isEmpty {
            // Find operations that exist in both serial and parallel
            let operations = Set(results.map { $0.operation.components(separatedBy: "(")[0] })
            
            for operation in operations.sorted() {
                let serialOps = serialResults.filter { $0.operation.starts(with: operation) }
                let parallelOps = parallelResults.filter { $0.operation.starts(with: operation) }
                
                if !serialOps.isEmpty && !parallelOps.isEmpty {
                    print("\n\(operation):")
                    
                    // Compare performance per vector
                    if let maxSerial = serialOps.max(by: { $0.datasetSize < $1.datasetSize }),
                       let minParallel = parallelOps.min(by: { $0.datasetSize < $1.datasetSize }) {
                        
                        let serialPerVector = maxSerial.avgTime / Double(maxSerial.datasetSize)
                        let parallelPerVector = minParallel.avgTime / Double(minParallel.datasetSize)
                        
                        print("  Serial (n=\(maxSerial.datasetSize)): \(String(format: "%.6f", serialPerVector))s per vector")
                        print("  Parallel (n=\(minParallel.datasetSize)): \(String(format: "%.6f", parallelPerVector))s per vector")
                        
                        if parallelPerVector < serialPerVector {
                            let improvement = (serialPerVector - parallelPerVector) / serialPerVector * 100
                            print("  Parallel is \(String(format: "%.1f%%", improvement)) faster per vector")
                        } else {
                            let overhead = (parallelPerVector - serialPerVector) / serialPerVector * 100
                            print("  Parallel has \(String(format: "%.1f%%", overhead)) overhead per vector")
                        }
                    }
                }
            }
        }
        
        print("\n=== Key Insights ===")
        print("- Operations below \(BatchOperations.configuration.parallelThreshold) vectors use serial execution")
        print("- Operations above \(BatchOperations.configuration.parallelThreshold) vectors use parallel execution")
        print("- Pairwise distances use parallel execution above 100 vectors")
        
        // Find crossover point efficiency
        let mapResults = results.filter { $0.operation.starts(with: "Map") }
        if let lastSerial = mapResults.filter({ !$0.isParallel }).last,
           let firstParallel = mapResults.filter({ $0.isParallel }).first {
            print("\nCrossover analysis for Map operation:")
            print("  Last serial: n=\(lastSerial.datasetSize), \(String(format: "%.3f", lastSerial.avgTime))s")
            print("  First parallel: n=\(firstParallel.datasetSize), \(String(format: "%.3f", firstParallel.avgTime))s")
        }
    }
}

// MARK: - Standalone Runner

/// Run benchmarks from command line or as part of test suite
public func runAutoParallelizationBenchmarks() async {
    print("VectorCore Auto-Parallelization Benchmark")
    print("========================================\n")
    
    let config = AutoParallelizationBenchmark.Configuration(
        vectorDimensions: 256,
        smallDatasetSizes: [100, 500, 900],
        largeDatasetSizes: [1100, 5000, 10000],
        kValues: [10, 100],
        iterations: 3
    )
    
    let results = await AutoParallelizationBenchmark.runBenchmarks(configuration: config)
    
    print("\n=== Benchmark Complete ===")
    print("Total benchmarks run: \(results.count)")
}