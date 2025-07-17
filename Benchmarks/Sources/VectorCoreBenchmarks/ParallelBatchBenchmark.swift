// VectorCore: Auto-Parallelization Benchmark
//
// Performance benchmarks demonstrating automatic parallelization behavior
//

import Foundation
import VectorCore

/// Benchmark suite for auto-parallelization in BatchOperations
///
/// Demonstrates how BatchOperations automatically switches between serial and
/// parallel execution based on dataset size, achieving optimal performance
/// without manual intervention.
public struct AutoParallelizationBenchmark {
    
    // MARK: - Configuration
    
    public struct Configuration: Sendable {
        public let vectorDimensions: [Int]  // Support multiple dimensions
        public let smallDatasetSizes: [Int]  // Below parallelization threshold
        public let largeDatasetSizes: [Int]  // Above parallelization threshold
        public let kValues: [Int]  // For k-NN benchmarks
        public let iterations: Int
        public let useOptimizedTypes: Bool  // Use Vector<D> when available
        public let includeDynamicBenchmarks: Bool  // Include DynamicVector benchmarks
        
        public static let `default` = Configuration(
            vectorDimensions: [256],
            smallDatasetSizes: [100, 500, 900],  // Below 1000 threshold
            largeDatasetSizes: [1500, 5000, 10000, 50000],  // Above 1000 threshold
            kValues: [10, 50, 100],
            iterations: 5,
            useOptimizedTypes: true,
            includeDynamicBenchmarks: true
        )
        
        public static let quick = Configuration(
            vectorDimensions: [64, 256],
            smallDatasetSizes: [100, 500],
            largeDatasetSizes: [1500, 5000],
            kValues: [10, 50],
            iterations: 2,
            useOptimizedTypes: true,
            includeDynamicBenchmarks: false
        )
        
        public static let comprehensive = Configuration(
            vectorDimensions: [32, 64, 128, 256, 512, 768, 1536],
            smallDatasetSizes: [100, 500, 900],
            largeDatasetSizes: [1500, 5000, 10000],
            kValues: [10, 50, 100],
            iterations: 3,
            useOptimizedTypes: true,
            includeDynamicBenchmarks: true
        )
        
        public init(
            vectorDimensions: [Int],
            smallDatasetSizes: [Int],
            largeDatasetSizes: [Int],
            kValues: [Int],
            iterations: Int,
            useOptimizedTypes: Bool = true,
            includeDynamicBenchmarks: Bool = true
        ) {
            self.vectorDimensions = vectorDimensions
            self.smallDatasetSizes = smallDatasetSizes
            self.largeDatasetSizes = largeDatasetSizes
            self.kValues = kValues
            self.iterations = iterations
            self.useOptimizedTypes = useOptimizedTypes
            self.includeDynamicBenchmarks = includeDynamicBenchmarks
        }
    }
    
    // MARK: - Results
    
    public struct BenchmarkResult: Sendable {
        public let operation: String
        public let datasetSize: Int
        public let dimension: Int
        public let vectorType: String
        public let avgTime: TimeInterval
        public let isParallel: Bool  // Whether this used parallel execution
        public let opsPerSecond: Double
        
        public var description: String {
            String(format: "%@ [%s,%dd] (n=%d): %.3fs [%@] - %.0f ops/sec",
                   operation, vectorType, dimension, datasetSize, avgTime, 
                   isParallel ? "Parallel" : "Serial",
                   opsPerSecond)
        }
    }
    
    // MARK: - Vector Type Wrapper
    
    /// Wrapper to handle different vector types uniformly
    private enum VectorCollection {
        case dynamic([DynamicVector])
        case dim32([Vector<Dim32>])
        case dim64([Vector<Dim64>])
        case dim128([Vector<Dim128>])
        case dim256([Vector<Dim256>])
        case dim512([Vector<Dim512>])
        case dim768([Vector<Dim768>])
        case dim1536([Vector<Dim1536>])
        case dim3072([Vector<Dim3072>])
        
        var count: Int {
            switch self {
            case .dynamic(let v): return v.count
            case .dim32(let v): return v.count
            case .dim64(let v): return v.count
            case .dim128(let v): return v.count
            case .dim256(let v): return v.count
            case .dim512(let v): return v.count
            case .dim768(let v): return v.count
            case .dim1536(let v): return v.count
            case .dim3072(let v): return v.count
            }
        }
        
        var dimension: Int {
            switch self {
            case .dynamic(let v): return v.first?.dimension ?? 0
            case .dim32: return 32
            case .dim64: return 64
            case .dim128: return 128
            case .dim256: return 256
            case .dim512: return 512
            case .dim768: return 768
            case .dim1536: return 1536
            case .dim3072: return 3072
            }
        }
        
        var typeName: String {
            switch self {
            case .dynamic: return "Dynamic"
            case .dim32: return "V32"
            case .dim64: return "V64"
            case .dim128: return "V128"
            case .dim256: return "V256"
            case .dim512: return "V512"
            case .dim768: return "V768"
            case .dim1536: return "V1536"
            case .dim3072: return "V3072"
            }
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
        print("Dimensions: \(configuration.vectorDimensions)")
        print("Parallel threshold: \(BatchOperations.configuration.parallelThreshold) vectors")
        print("")
        
        // Run benchmarks for each dimension
        for dimension in configuration.vectorDimensions {
            print("\n--- Dimension: \(dimension) ---")
            
            // Test small datasets (should run serially)
            print("Small Datasets (Serial Execution Expected):")
            for size in configuration.smallDatasetSizes {
                let vectors = generateTestVectors(count: size, dimensions: dimension, useOptimized: configuration.useOptimizedTypes)
                
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
                let vectors = generateTestVectors(count: size, dimensions: dimension, useOptimized: configuration.useOptimizedTypes)
                
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
                let vectors = generateTestVectors(count: size, dimensions: dimension, useOptimized: configuration.useOptimizedTypes)
                if let result = await benchmarkPairwiseDistances(vectors: vectors, iterations: configuration.iterations) {
                    results.append(result)
                    print("  \(result.description)")
                }
            }
        }
        
        // Print analysis
        printAnalysis(results: results)
        
        return results
    }
    
    // MARK: - Individual Benchmarks
    
    private static func benchmarkFindNearest(vectors: VectorCollection, k: Int, iterations: Int) async -> BenchmarkResult? {
        switch vectors {
        case .dynamic(let vecs):
            return await benchmarkFindNearestDynamic(vectors: vecs, k: k, iterations: iterations)
        case .dim32(let vecs):
            return await benchmarkFindNearestGeneric(vectors: vecs, k: k, iterations: iterations)
        case .dim64(let vecs):
            return await benchmarkFindNearestGeneric(vectors: vecs, k: k, iterations: iterations)
        case .dim128(let vecs):
            return await benchmarkFindNearestGeneric(vectors: vecs, k: k, iterations: iterations)
        case .dim256(let vecs):
            return await benchmarkFindNearestGeneric(vectors: vecs, k: k, iterations: iterations)
        case .dim512(let vecs):
            return await benchmarkFindNearestGeneric(vectors: vecs, k: k, iterations: iterations)
        case .dim768(let vecs):
            return await benchmarkFindNearestGeneric(vectors: vecs, k: k, iterations: iterations)
        case .dim1536(let vecs):
            return await benchmarkFindNearestGeneric(vectors: vecs, k: k, iterations: iterations)
        case .dim3072(let vecs):
            return await benchmarkFindNearestGeneric(vectors: vecs, k: k, iterations: iterations)
        }
    }
    
    private static func benchmarkFindNearestGeneric<V: ExtendedVectorProtocol & Sendable>(
        vectors: [V], 
        k: Int, 
        iterations: Int
    ) async -> BenchmarkResult? where V: Equatable {
        let query = vectors.first ?? V(from: Array(repeating: 0.5, count: vectors.first?.scalarCount ?? 1))
        
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
            dimension: query.scalarCount,
            vectorType: String(describing: type(of: query)),
            avgTime: avgTime,
            isParallel: vectors.count >= BatchOperations.configuration.parallelThreshold,
            opsPerSecond: 1.0 / avgTime
        )
    }
    
    private static func benchmarkFindNearestDynamic(
        vectors: [DynamicVector], 
        k: Int, 
        iterations: Int
    ) async -> BenchmarkResult? {
        let query = vectors.first ?? DynamicVector(dimension: vectors.first?.dimension ?? 256, repeating: 0.5)
        
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
            dimension: query.dimension,
            vectorType: "DynamicVector",
            avgTime: avgTime,
            isParallel: vectors.count >= BatchOperations.configuration.parallelThreshold,
            opsPerSecond: 1.0 / avgTime
        )
    }
    
    private static func benchmarkMap(vectors: VectorCollection, iterations: Int) async -> BenchmarkResult? {
        switch vectors {
        case .dynamic(let vecs):
            return await benchmarkMapDynamic(vectors: vecs, iterations: iterations)
        case .dim32(let vecs):
            return await benchmarkMapGeneric(vectors: vecs, iterations: iterations, typeName: vectors.typeName)
        case .dim64(let vecs):
            return await benchmarkMapGeneric(vectors: vecs, iterations: iterations, typeName: vectors.typeName)
        case .dim128(let vecs):
            return await benchmarkMapGeneric(vectors: vecs, iterations: iterations, typeName: vectors.typeName)
        case .dim256(let vecs):
            return await benchmarkMapGeneric(vectors: vecs, iterations: iterations, typeName: vectors.typeName)
        case .dim512(let vecs):
            return await benchmarkMapGeneric(vectors: vecs, iterations: iterations, typeName: vectors.typeName)
        case .dim768(let vecs):
            return await benchmarkMapGeneric(vectors: vecs, iterations: iterations, typeName: vectors.typeName)
        case .dim1536(let vecs):
            return await benchmarkMapGeneric(vectors: vecs, iterations: iterations, typeName: vectors.typeName)
        case .dim3072(let vecs):
            return await benchmarkMapGeneric(vectors: vecs, iterations: iterations, typeName: vectors.typeName)
        }
    }
    
    private static func benchmarkMapGeneric<V: ExtendedVectorProtocol & Sendable>(
        vectors: [V], 
        iterations: Int,
        typeName: String
    ) async -> BenchmarkResult? where V: Equatable {
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
            dimension: vectors.first?.scalarCount ?? 0,
            vectorType: typeName,
            avgTime: avgTime,
            isParallel: vectors.count >= BatchOperations.configuration.parallelThreshold,
            opsPerSecond: Double(vectors.count) / avgTime
        )
    }
    
    private static func benchmarkMapDynamic(
        vectors: [DynamicVector], 
        iterations: Int
    ) async -> BenchmarkResult? {
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
            dimension: vectors.first?.dimension ?? 0,
            vectorType: "DynamicVector",
            avgTime: avgTime,
            isParallel: vectors.count >= BatchOperations.configuration.parallelThreshold,
            opsPerSecond: Double(vectors.count) / avgTime
        )
    }
    
    private static func benchmarkPairwiseDistances(vectors: VectorCollection, iterations: Int) async -> BenchmarkResult? {
        switch vectors {
        case .dynamic(let vecs):
            return await benchmarkPairwiseDistancesDynamic(vectors: vecs, iterations: iterations)
        case .dim32(let vecs):
            return await benchmarkPairwiseDistancesGeneric(vectors: vecs, iterations: iterations, typeName: vectors.typeName)
        case .dim64(let vecs):
            return await benchmarkPairwiseDistancesGeneric(vectors: vecs, iterations: iterations, typeName: vectors.typeName)
        case .dim128(let vecs):
            return await benchmarkPairwiseDistancesGeneric(vectors: vecs, iterations: iterations, typeName: vectors.typeName)
        case .dim256(let vecs):
            return await benchmarkPairwiseDistancesGeneric(vectors: vecs, iterations: iterations, typeName: vectors.typeName)
        case .dim512(let vecs):
            return await benchmarkPairwiseDistancesGeneric(vectors: vecs, iterations: iterations, typeName: vectors.typeName)
        case .dim768(let vecs):
            return await benchmarkPairwiseDistancesGeneric(vectors: vecs, iterations: iterations, typeName: vectors.typeName)
        case .dim1536(let vecs):
            return await benchmarkPairwiseDistancesGeneric(vectors: vecs, iterations: iterations, typeName: vectors.typeName)
        case .dim3072(let vecs):
            return await benchmarkPairwiseDistancesGeneric(vectors: vecs, iterations: iterations, typeName: vectors.typeName)
        }
    }
    
    private static func benchmarkPairwiseDistancesGeneric<V: ExtendedVectorProtocol & Sendable>(
        vectors: [V], 
        iterations: Int,
        typeName: String
    ) async -> BenchmarkResult? where V: Equatable {
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
            dimension: vectors.first?.scalarCount ?? 0,
            vectorType: typeName,
            avgTime: avgTime,
            isParallel: vectors.count >= 100,  // Pairwise has different threshold
            opsPerSecond: Double(operationCount) / avgTime
        )
    }
    
    private static func benchmarkPairwiseDistancesDynamic(
        vectors: [DynamicVector], 
        iterations: Int
    ) async -> BenchmarkResult? {
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
            dimension: vectors.first?.dimension ?? 0,
            vectorType: "DynamicVector",
            avgTime: avgTime,
            isParallel: vectors.count >= 100,  // Pairwise has different threshold
            opsPerSecond: Double(operationCount) / avgTime
        )
    }
    
    // MARK: - Helpers
    
    private static func generateTestVectors(count: Int, dimensions: Int, useOptimized: Bool) -> VectorCollection {
        // Check if we should use optimized types
        if useOptimized {
            switch dimensions {
            case 32:
                return .dim32(generateOptimizedVectors(count: count, type: Vector<Dim32>.self))
            case 64:
                return .dim64(generateOptimizedVectors(count: count, type: Vector<Dim64>.self))
            case 128:
                return .dim128(generateOptimizedVectors(count: count, type: Vector<Dim128>.self))
            case 256:
                return .dim256(generateOptimizedVectors(count: count, type: Vector<Dim256>.self))
            case 512:
                return .dim512(generateOptimizedVectors(count: count, type: Vector<Dim512>.self))
            case 768:
                return .dim768(generateOptimizedVectors(count: count, type: Vector<Dim768>.self))
            case 1536:
                return .dim1536(generateOptimizedVectors(count: count, type: Vector<Dim1536>.self))
            case 3072:
                return .dim3072(generateOptimizedVectors(count: count, type: Vector<Dim3072>.self))
            default:
                // Fall back to dynamic for unsupported dimensions
                return .dynamic(generateDynamicVectors(count: count, dimensions: dimensions))
            }
        } else {
            // Always use dynamic vectors
            return .dynamic(generateDynamicVectors(count: count, dimensions: dimensions))
        }
    }
    
    private static func generateOptimizedVectors<V: ExtendedVectorProtocol>(
        count: Int,
        type: V.Type
    ) -> [V] where V: Equatable {
        let dimension = V.dimensions
        return (0..<count).map { i in
            let values = (0..<dimension).map { j in
                sin(Float(i * dimension + j) / 1000.0)
            }
            return V(from: values)
        }
    }
    
    private static func generateDynamicVectors(count: Int, dimensions: Int) -> [DynamicVector] {
        return (0..<count).map { i in
            let values = (0..<dimensions).map { j in
                sin(Float(i * dimensions + j) / 1000.0)
            }
            return DynamicVector(values)
        }
    }
    
    private static func printAnalysis(results: [BenchmarkResult]) {
        print("\n\nBenchmark Analysis")
        print("==================")
        
        // Group by dimension
        let byDimension = Dictionary(grouping: results) { $0.dimension }
        
        for (dimension, dimResults) in byDimension.sorted(by: { $0.key < $1.key }) {
            print("\nDimension \(dimension):")
            
            // Calculate serial vs parallel speedup
            let serialResults = dimResults.filter { !$0.isParallel }
            let parallelResults = dimResults.filter { $0.isParallel }
            
            if !serialResults.isEmpty && !parallelResults.isEmpty {
                let avgSerialOps = serialResults.map { $0.opsPerSecond }.reduce(0, +) / Double(serialResults.count)
                let avgParallelOps = parallelResults.map { $0.opsPerSecond }.reduce(0, +) / Double(parallelResults.count)
                let speedup = avgParallelOps / avgSerialOps
                
                print("  Average Serial: %.0f ops/sec", avgSerialOps)
                print("  Average Parallel: %.0f ops/sec", avgParallelOps)
                print("  Parallel Speedup: %.2fx", speedup)
            }
            
            // Show threshold effectiveness
            let threshold = BatchOperations.configuration.parallelThreshold
            let belowThreshold = dimResults.filter { $0.datasetSize < threshold }
            let aboveThreshold = dimResults.filter { $0.datasetSize >= threshold }
            
            let belowParallel = belowThreshold.filter { $0.isParallel }.count
            let aboveSerial = aboveThreshold.filter { !$0.isParallel }.count
            
            print("  Threshold Accuracy: %.1f%%",
                  100.0 * Double(belowThreshold.count - belowParallel + aboveThreshold.count - aboveSerial) /
                  Double(belowThreshold.count + aboveThreshold.count))
        }
    }
}