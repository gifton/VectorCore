// VectorCore: Benchmark Runner
//
// Executes VectorCore benchmarks and collects performance metrics
//

import Foundation

/// Main benchmark runner for VectorCore
public final class VectorCoreBenchmarkRunner: BenchmarkRunnerProtocol {
    
    /// Configuration for benchmark execution
    public let configuration: BenchmarkConfiguration
    
    /// Memory tracker for benchmarks
    private var memoryTracker: BenchmarkMemoryTracker?
    
    public init(configuration: BenchmarkConfiguration = .default) {
        self.configuration = configuration
    }
    
    /// Run all benchmarks
    public func run() async throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []
        
        // Vector operation benchmarks
        results.append(contentsOf: try await runVectorOperationBenchmarks())
        
        // Distance metric benchmarks
        results.append(contentsOf: try await runDistanceBenchmarks())
        
        // Storage benchmarks
        results.append(contentsOf: try await runStorageBenchmarks())
        
        // Batch operation benchmarks
        results.append(contentsOf: try await runBatchOperationBenchmarks())
        
        return results
    }
    
    /// Run specific benchmark by name
    public func run(benchmarkNamed name: String) async throws -> BenchmarkResult? {
        let allResults = try await run()
        return allResults.first { $0.name == name }
    }
    
    /// List of available benchmarks
    public var availableBenchmarks: [String] {
        [
            // Vector operations
            "Vector Addition - 128D",
            "Vector Addition - 768D",
            "Vector Addition - 1536D",
            "Vector Scalar Multiplication - 768D",
            "Vector Dot Product - 128D",
            "Vector Dot Product - 768D",
            "Vector Dot Product - 1536D",
            "Vector Normalization - 768D",
            "Vector Magnitude - 768D",
            
            // Distance metrics
            "Euclidean Distance - 128D",
            "Euclidean Distance - 768D",
            "Euclidean Distance - 1536D",
            "Cosine Similarity - 128D",
            "Cosine Similarity - 768D",
            "Cosine Similarity - 1536D",
            "Manhattan Distance - 128D",
            "Manhattan Distance - 768D",
            "Manhattan Distance - 1536D",
            
            // Dynamic vectors
            "DynamicVector Addition - 768D",
            "DynamicVector Dot Product - 768D",
            "DynamicVector Euclidean Distance - 768D",
            
            // Batch operations
            "Batch Euclidean Distance - 100x768D",
            "Batch Cosine Similarity - 100x768D"
        ]
    }
    
    // MARK: - Vector Operation Benchmarks
    
    private func runVectorOperationBenchmarks() async throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []
        
        // Addition benchmarks
        results.append(try await measureBenchmark(name: "Vector Addition - 128D") {
            let a = Vector<Dim128>.random(in: -1...1)
            let b = Vector<Dim128>.random(in: -1...1)
            return {
                _ = a + b
            }
        })
        
        results.append(try await measureBenchmark(name: "Vector Addition - 768D") {
            let a = Vector<Dim768>.random(in: -1...1)
            let b = Vector<Dim768>.random(in: -1...1)
            return {
                _ = a + b
            }
        })
        
        results.append(try await measureBenchmark(name: "Vector Addition - 1536D") {
            let a = Vector<Dim1536>.random(in: -1...1)
            let b = Vector<Dim1536>.random(in: -1...1)
            return {
                _ = a + b
            }
        })
        
        // Scalar multiplication
        results.append(try await measureBenchmark(name: "Vector Scalar Multiplication - 768D") {
            let a = Vector<Dim768>.random(in: -1...1)
            let scalar: Float = 2.5
            return {
                _ = a * scalar
            }
        })
        
        // Dot product
        results.append(try await measureBenchmark(name: "Vector Dot Product - 128D") {
            let a = Vector<Dim128>.random(in: -1...1)
            let b = Vector<Dim128>.random(in: -1...1)
            return {
                _ = a.dotProduct(b)
            }
        })
        
        results.append(try await measureBenchmark(name: "Vector Dot Product - 768D") {
            let a = Vector<Dim768>.random(in: -1...1)
            let b = Vector<Dim768>.random(in: -1...1)
            return {
                _ = a.dotProduct(b)
            }
        })
        
        results.append(try await measureBenchmark(name: "Vector Dot Product - 1536D") {
            let a = Vector<Dim1536>.random(in: -1...1)
            let b = Vector<Dim1536>.random(in: -1...1)
            return {
                _ = a.dotProduct(b)
            }
        })
        
        // Normalization
        results.append(try await measureBenchmark(name: "Vector Normalization - 768D") {
            let a = Vector<Dim768>.random(in: -1...1)
            return {
                _ = a.normalized()
            }
        })
        
        // Magnitude
        results.append(try await measureBenchmark(name: "Vector Magnitude - 768D") {
            let a = Vector<Dim768>.random(in: -1...1)
            return {
                _ = a.magnitude
            }
        })
        
        // Dynamic vectors
        results.append(try await measureBenchmark(name: "DynamicVector Addition - 768D") {
            let a = DynamicVector.random(dimension: 768, in: -1...1)
            let b = DynamicVector.random(dimension: 768, in: -1...1)
            return {
                _ = a + b
            }
        })
        
        results.append(try await measureBenchmark(name: "DynamicVector Dot Product - 768D") {
            let a = DynamicVector.random(dimension: 768, in: -1...1)
            let b = DynamicVector.random(dimension: 768, in: -1...1)
            return {
                _ = a.dotProduct(b)
            }
        })
        
        return results
    }
    
    // MARK: - Distance Metric Benchmarks
    
    private func runDistanceBenchmarks() async throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []
        
        let euclideanMetric = EuclideanDistance()
        let cosineMetric = CosineDistance()
        let manhattanMetric = ManhattanDistance()
        
        // Euclidean distance
        for (dim, dimType) in [(128, "128D"), (768, "768D"), (1536, "1536D")] {
            results.append(try await measureBenchmark(name: "Euclidean Distance - \(dimType)") {
                switch dim {
                case 128:
                    let a = Vector<Dim128>.random(in: -1...1)
                    let b = Vector<Dim128>.random(in: -1...1)
                    return { _ = euclideanMetric.distance(a, b) }
                case 768:
                    let a = Vector<Dim768>.random(in: -1...1)
                    let b = Vector<Dim768>.random(in: -1...1)
                    return { _ = euclideanMetric.distance(a, b) }
                case 1536:
                    let a = Vector<Dim1536>.random(in: -1...1)
                    let b = Vector<Dim1536>.random(in: -1...1)
                    return { _ = euclideanMetric.distance(a, b) }
                default:
                    fatalError("Unsupported dimension")
                }
            })
        }
        
        // Cosine similarity
        for (dim, dimType) in [(128, "128D"), (768, "768D"), (1536, "1536D")] {
            results.append(try await measureBenchmark(name: "Cosine Similarity - \(dimType)") {
                switch dim {
                case 128:
                    let a = Vector<Dim128>.random(in: -1...1)
                    let b = Vector<Dim128>.random(in: -1...1)
                    return { _ = cosineMetric.distance(a, b) }
                case 768:
                    let a = Vector<Dim768>.random(in: -1...1)
                    let b = Vector<Dim768>.random(in: -1...1)
                    return { _ = cosineMetric.distance(a, b) }
                case 1536:
                    let a = Vector<Dim1536>.random(in: -1...1)
                    let b = Vector<Dim1536>.random(in: -1...1)
                    return { _ = cosineMetric.distance(a, b) }
                default:
                    fatalError("Unsupported dimension")
                }
            })
        }
        
        // Manhattan distance
        for (dim, dimType) in [(128, "128D"), (768, "768D"), (1536, "1536D")] {
            results.append(try await measureBenchmark(name: "Manhattan Distance - \(dimType)") {
                switch dim {
                case 128:
                    let a = Vector<Dim128>.random(in: -1...1)
                    let b = Vector<Dim128>.random(in: -1...1)
                    return { _ = manhattanMetric.distance(a, b) }
                case 768:
                    let a = Vector<Dim768>.random(in: -1...1)
                    let b = Vector<Dim768>.random(in: -1...1)
                    return { _ = manhattanMetric.distance(a, b) }
                case 1536:
                    let a = Vector<Dim1536>.random(in: -1...1)
                    let b = Vector<Dim1536>.random(in: -1...1)
                    return { _ = manhattanMetric.distance(a, b) }
                default:
                    fatalError("Unsupported dimension")
                }
            })
        }
        
        // Dynamic vector distances
        results.append(try await measureBenchmark(name: "DynamicVector Euclidean Distance - 768D") {
            let a = DynamicVector.random(dimension: 768, in: -1...1)
            let b = DynamicVector.random(dimension: 768, in: -1...1)
            return {
                _ = euclideanMetric.distance(a, b)
            }
        })
        
        return results
    }
    
    // MARK: - Storage Benchmarks
    
    private func runStorageBenchmarks() async throws -> [BenchmarkResult] {
        // Placeholder - implement storage-specific benchmarks
        return []
    }
    
    // MARK: - Batch Operation Benchmarks
    
    private func runBatchOperationBenchmarks() async throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []
        
        let euclideanMetric = EuclideanDistance()
        let cosineMetric = CosineDistance()
        
        // Batch euclidean distance
        results.append(try await measureBenchmark(name: "Batch Euclidean Distance - 100x768D") {
            let vectors = (0..<100).map { _ in Vector<Dim768>.random(in: -1...1) }
            let target = Vector<Dim768>.random(in: -1...1)
            return {
                for vector in vectors {
                    _ = euclideanMetric.distance(vector, target)
                }
            }
        })
        
        // Batch cosine similarity
        results.append(try await measureBenchmark(name: "Batch Cosine Similarity - 100x768D") {
            let vectors = (0..<100).map { _ in Vector<Dim768>.random(in: -1...1) }
            let target = Vector<Dim768>.random(in: -1...1)
            return {
                for vector in vectors {
                    _ = cosineMetric.distance(vector, target)
                }
            }
        })
        
        return results
    }
    
    // MARK: - Measurement Helpers
    
    private func measureBenchmark(
        name: String,
        setup: () -> (() -> Void)
    ) async throws -> BenchmarkResult {
        // Setup
        let operation = setup()
        
        // Initialize memory tracking
        memoryTracker = BenchmarkMemoryTracker()
        
        // Warmup
        for _ in 0..<configuration.warmupIterations {
            operation()
        }
        
        // Measurement
        var times: [Double] = []
        times.reserveCapacity(configuration.measurementIterations)
        
        for _ in 0..<configuration.measurementIterations {
            let start = CFAbsoluteTimeGetCurrent()
            operation()
            let end = CFAbsoluteTimeGetCurrent()
            times.append(end - start)
            
            // Update memory tracking
            memoryTracker?.update()
        }
        
        // Calculate statistics
        let mean = times.reduce(0, +) / Double(times.count)
        let variance = times.map { pow($0 - mean, 2) }.reduce(0, +) / Double(times.count)
        let stdDev = sqrt(variance)
        _ = times.min() ?? 0
        _ = times.max() ?? 0
        let throughput = mean > 0 ? 1.0 / mean : 0
        
        return BenchmarkResult(
            name: name,
            iterations: configuration.measurementIterations,
            totalTime: mean * Double(configuration.measurementIterations),
            averageTime: mean,
            standardDeviation: stdDev,
            throughput: throughput,
            memoryAllocated: memoryTracker?.memoryUsed
        )
    }
}