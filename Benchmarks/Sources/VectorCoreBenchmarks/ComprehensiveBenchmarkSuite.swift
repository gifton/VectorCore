import Foundation
import VectorCore

/// Comprehensive benchmark suite for VectorCore
///
/// Provides detailed performance analysis including:
/// - Various vector dimensions (32 to 3072)
/// - All core operations with baseline comparisons
/// - Storage type performance analysis
/// - Memory allocation tracking
/// - Statistical analysis of results
public struct ComprehensiveBenchmarkSuite {
    
    // MARK: - Main Entry Point
    
    public static func run(configuration: BenchmarkFramework.Configuration = .default) {
        print("VectorCore Comprehensive Performance Benchmark Suite")
        print("===================================================")
        print("Date: \(Date())")
        print("")
        
        var allResults: [BenchmarkFramework.MeasurementResult] = []
        
        // Run benchmarks for each category
        allResults.append(contentsOf: benchmarkVectorCreation(configuration: configuration))
        allResults.append(contentsOf: benchmarkBasicOperations(configuration: configuration))
        allResults.append(contentsOf: benchmarkAdvancedOperations(configuration: configuration))
        allResults.append(contentsOf: benchmarkStorageTypes(configuration: configuration))
        allResults.append(contentsOf: benchmarkBatchOperations(configuration: configuration))
        
        // Generate summary report
        generateSummaryReport(results: allResults)
        
        // Save results if JSON output is configured
        if let jsonPath = configuration.jsonOutputPath {
            saveAllResults(allResults, to: jsonPath)
        }
    }
    
    // MARK: - Vector Creation Benchmarks
    
    private static func benchmarkVectorCreation(configuration: BenchmarkFramework.Configuration) -> [BenchmarkFramework.MeasurementResult] {
        print("\n=== Vector Creation Benchmarks ===\n")
        
        var benchmarks: [(name: String, category: String, vectorSize: Int, baseline: (() -> Void)?, block: () -> Void)] = []
        
        // Test different vector sizes
        let dimensions = [32, 64, 128, 256, 512, 768, 1536]
        
        for dim in dimensions {
            let values = (0..<dim).map { Float($0) }
            
            // From array creation
            benchmarks.append((
                name: "Vector\(dim) from array",
                category: "Creation",
                vectorSize: dim,
                baseline: {
                    _ = values  // Baseline: just use the array
                },
                block: {
                    switch dim {
                    case 32: _ = Vector<Dim32>(values)
                    case 64: _ = Vector<Dim64>(values)
                    case 128: _ = Vector128(values)
                    case 256: _ = Vector256(values)
                    case 512: _ = Vector512(values)
                    case 768: _ = Vector768(values)
                    case 1536: _ = Vector1536(values)
                    default: _ = DynamicVector(values)
                    }
                }
            ))
            
            // Zero initialization
            benchmarks.append((
                name: "Vector\(dim) zeros",
                category: "Creation",
                vectorSize: dim,
                baseline: {
                    _ = [Float](repeating: 0, count: dim)
                },
                block: {
                    switch dim {
                    case 32: _ = Vector<Dim32>.zero
                    case 64: _ = Vector<Dim64>.zero
                    case 128: _ = Vector128.zero
                    case 256: _ = Vector256.zero
                    case 512: _ = Vector512.zero
                    case 768: _ = Vector768.zero
                    case 1536: _ = Vector1536.zero
                    default: _ = DynamicVector(dimension: dim, repeating: 0)
                    }
                }
            ))
            
            // Random initialization
            benchmarks.append((
                name: "Vector\(dim) random",
                category: "Creation",
                vectorSize: dim,
                baseline: {
                    _ = (0..<dim).map { _ in Float.random(in: -1...1) }
                },
                block: {
                    switch dim {
                    case 32: _ = Vector<Dim32>.random(in: -1...1)
                    case 64: _ = Vector<Dim64>.random(in: -1...1)
                    case 128: _ = Vector128.random(in: -1...1)
                    case 256: _ = Vector256.random(in: -1...1)
                    case 512: _ = Vector512.random(in: -1...1)
                    case 768: _ = Vector768.random(in: -1...1)
                    case 1536: _ = Vector1536.random(in: -1...1)
                    default: _ = DynamicVector.random(dimension: dim, in: -1...1)
                    }
                }
            ))
        }
        
        return BenchmarkFramework.runBenchmarkSuite(
            name: "Vector Creation",
            configuration: configuration,
            benchmarks: benchmarks
        )
    }
    
    // MARK: - Basic Operations Benchmarks
    
    private static func benchmarkBasicOperations(configuration: BenchmarkFramework.Configuration) -> [BenchmarkFramework.MeasurementResult] {
        print("\n=== Basic Operations Benchmarks ===\n")
        
        var benchmarks: [(name: String, category: String, vectorSize: Int, baseline: (() -> Void)?, block: () -> Void)] = []
        
        // Test key dimensions
        let testDimensions = [64, 256, 768]
        
        for dim in testDimensions {
            // Create test vectors
            let array1 = (0..<dim).map { sin(Float($0) * 0.1) }
            let array2 = (0..<dim).map { cos(Float($0) * 0.1) }
            let scalar: Float = 2.5
            
            // Addition
            benchmarks.append((
                name: "Vector\(dim) addition",
                category: "Basic Operations",
                vectorSize: dim,
                baseline: {
                    _ = BaselineImplementations.add(array1, array2)
                },
                block: {
                    switch dim {
                    case 64:
                        let v1 = Vector<Dim64>(array1)
                        let v2 = Vector<Dim64>(array2)
                        _ = v1 + v2
                    case 256:
                        let v1 = Vector256(array1)
                        let v2 = Vector256(array2)
                        _ = v1 + v2
                    case 768:
                        let v1 = Vector768(array1)
                        let v2 = Vector768(array2)
                        _ = v1 + v2
                    default:
                        let v1 = DynamicVector(array1)
                        let v2 = DynamicVector(array2)
                        _ = v1 + v2
                    }
                }
            ))
            
            // Subtraction
            benchmarks.append((
                name: "Vector\(dim) subtraction",
                category: "Basic Operations",
                vectorSize: dim,
                baseline: {
                    _ = BaselineImplementations.add(array1, array2.map { -$0 })
                },
                block: {
                    switch dim {
                    case 64:
                        let v1 = Vector<Dim64>(array1)
                        let v2 = Vector<Dim64>(array2)
                        _ = v1 - v2
                    case 256:
                        let v1 = Vector256(array1)
                        let v2 = Vector256(array2)
                        _ = v1 - v2
                    case 768:
                        let v1 = Vector768(array1)
                        let v2 = Vector768(array2)
                        _ = v1 - v2
                    default:
                        let v1 = DynamicVector(array1)
                        let v2 = DynamicVector(array2)
                        _ = v1 - v2
                    }
                }
            ))
            
            // Scalar multiplication
            benchmarks.append((
                name: "Vector\(dim) scalar multiply",
                category: "Basic Operations",
                vectorSize: dim,
                baseline: {
                    _ = array1.map { $0 * scalar }
                },
                block: {
                    switch dim {
                    case 64:
                        let v = Vector<Dim64>(array1)
                        _ = v * scalar
                    case 256:
                        let v = Vector256(array1)
                        _ = v * scalar
                    case 768:
                        let v = Vector768(array1)
                        _ = v * scalar
                    default:
                        let v = DynamicVector(array1)
                        _ = v * scalar
                    }
                }
            ))
            
            // Element-wise multiplication
            benchmarks.append((
                name: "Vector\(dim) element multiply",
                category: "Basic Operations",
                vectorSize: dim,
                baseline: {
                    var result = [Float](repeating: 0, count: dim)
                    for i in 0..<dim {
                        result[i] = array1[i] * array2[i]
                    }
                    _ = result
                },
                block: {
                    switch dim {
                    case 64:
                        let v1 = Vector<Dim64>(array1)
                        let v2 = Vector<Dim64>(array2)
                        _ = v1 * v2
                    case 256:
                        let v1 = Vector256(array1)
                        let v2 = Vector256(array2)
                        _ = v1 * v2
                    case 768:
                        let v1 = Vector768(array1)
                        let v2 = Vector768(array2)
                        _ = v1 * v2
                    default:
                        let v1 = DynamicVector(array1)
                        let v2 = DynamicVector(array2)
                        _ = v1 * v2
                    }
                }
            ))
        }
        
        return BenchmarkFramework.runBenchmarkSuite(
            name: "Basic Operations",
            configuration: configuration,
            benchmarks: benchmarks
        )
    }
    
    // MARK: - Advanced Operations Benchmarks
    
    private static func benchmarkAdvancedOperations(configuration: BenchmarkFramework.Configuration) -> [BenchmarkFramework.MeasurementResult] {
        print("\n=== Advanced Operations Benchmarks ===\n")
        
        var benchmarks: [(name: String, category: String, vectorSize: Int, baseline: (() -> Void)?, block: () -> Void)] = []
        
        let testDimensions = [64, 256, 768]
        
        for dim in testDimensions {
            let array1 = (0..<dim).map { sin(Float($0) * 0.1) }
            let array2 = (0..<dim).map { cos(Float($0) * 0.1) }
            
            // Dot product
            benchmarks.append((
                name: "Vector\(dim) dot product",
                category: "Advanced Operations",
                vectorSize: dim,
                baseline: {
                    _ = BaselineImplementations.dotProduct(array1, array2)
                },
                block: {
                    switch dim {
                    case 64:
                        let v1 = Vector<Dim64>(array1)
                        let v2 = Vector<Dim64>(array2)
                        _ = v1.dotProduct(v2)
                    case 256:
                        let v1 = Vector256(array1)
                        let v2 = Vector256(array2)
                        _ = v1.dotProduct(v2)
                    case 768:
                        let v1 = Vector768(array1)
                        let v2 = Vector768(array2)
                        _ = v1.dotProduct(v2)
                    default:
                        let v1 = DynamicVector(array1)
                        let v2 = DynamicVector(array2)
                        _ = v1.dotProduct(v2)
                    }
                }
            ))
            
            // Magnitude
            benchmarks.append((
                name: "Vector\(dim) magnitude",
                category: "Advanced Operations",
                vectorSize: dim,
                baseline: {
                    _ = BaselineImplementations.magnitude(array1)
                },
                block: {
                    switch dim {
                    case 64:
                        let v = Vector<Dim64>(array1)
                        _ = v.magnitude
                    case 256:
                        let v = Vector256(array1)
                        _ = v.magnitude
                    case 768:
                        let v = Vector768(array1)
                        _ = v.magnitude
                    default:
                        let v = DynamicVector(array1)
                        _ = v.magnitude
                    }
                }
            ))
            
            // Normalization
            benchmarks.append((
                name: "Vector\(dim) normalize",
                category: "Advanced Operations",
                vectorSize: dim,
                baseline: {
                    _ = BaselineImplementations.normalize(array1)
                },
                block: {
                    switch dim {
                    case 64:
                        let v = Vector<Dim64>(array1)
                        _ = v.normalized()
                    case 256:
                        let v = Vector256(array1)
                        _ = v.normalized()
                    case 768:
                        let v = Vector768(array1)
                        _ = v.normalized()
                    default:
                        let v = DynamicVector(array1)
                        _ = v.normalized()
                    }
                }
            ))
            
            // Euclidean distance
            benchmarks.append((
                name: "Vector\(dim) distance",
                category: "Advanced Operations",
                vectorSize: dim,
                baseline: {
                    _ = BaselineImplementations.distance(array1, array2)
                },
                block: {
                    switch dim {
                    case 64:
                        let v1 = Vector<Dim64>(array1)
                        let v2 = Vector<Dim64>(array2)
                        _ = v1.distance(to: v2)
                    case 256:
                        let v1 = Vector256(array1)
                        let v2 = Vector256(array2)
                        _ = v1.distance(to: v2)
                    case 768:
                        let v1 = Vector768(array1)
                        let v2 = Vector768(array2)
                        _ = v1.distance(to: v2)
                    default:
                        let v1 = DynamicVector(array1)
                        let v2 = DynamicVector(array2)
                        _ = v1.distance(to: v2)
                    }
                }
            ))
            
            // Cosine similarity
            benchmarks.append((
                name: "Vector\(dim) cosine similarity",
                category: "Advanced Operations",
                vectorSize: dim,
                baseline: {
                    _ = BaselineImplementations.cosineSimilarity(array1, array2)
                },
                block: {
                    switch dim {
                    case 64:
                        let v1 = Vector<Dim64>(array1)
                        let v2 = Vector<Dim64>(array2)
                        _ = v1.cosineSimilarity(to: v2)
                    case 256:
                        let v1 = Vector256(array1)
                        let v2 = Vector256(array2)
                        _ = v1.cosineSimilarity(to: v2)
                    case 768:
                        let v1 = Vector768(array1)
                        let v2 = Vector768(array2)
                        _ = v1.cosineSimilarity(to: v2)
                    default:
                        let v1 = DynamicVector(array1)
                        let v2 = DynamicVector(array2)
                        _ = v1.cosineSimilarity(to: v2)
                    }
                }
            ))
        }
        
        return BenchmarkFramework.runBenchmarkSuite(
            name: "Advanced Operations",
            configuration: configuration,
            benchmarks: benchmarks
        )
    }
    
    // MARK: - Storage Type Benchmarks
    
    private static func benchmarkStorageTypes(configuration: BenchmarkFramework.Configuration) -> [BenchmarkFramework.MeasurementResult] {
        print("\n=== Storage Type Benchmarks ===\n")
        
        var benchmarks: [(name: String, category: String, vectorSize: Int, baseline: (() -> Void)?, block: () -> Void)] = []
        
        // Test different storage types
        let smallSize = 32
        let mediumSize = 256
        let largeSize = 1536
        
        // Small storage benchmarks
        let smallArray = (0..<smallSize).map { Float($0) }
        let smallStorage = SmallVectorStorage(count: smallSize, repeating: 1.0)
        
        benchmarks.append((
            name: "SmallStorage[\(smallSize)] access",
            category: "Storage",
            vectorSize: smallSize,
            baseline: {
                var sum: Float = 0
                for i in 0..<smallSize {
                    sum += smallArray[i]
                }
                _ = sum
            },
            block: {
                var sum: Float = 0
                for i in 0..<smallSize {
                    sum += smallStorage[i]
                }
                _ = sum
            }
        ))
        
        // Medium storage benchmarks
        let mediumArray = (0..<mediumSize).map { Float($0) }
        let mediumStorage = MediumVectorStorage(count: mediumSize, repeating: 1.0)
        
        benchmarks.append((
            name: "MediumStorage[\(mediumSize)] access",
            category: "Storage",
            vectorSize: mediumSize,
            baseline: {
                var sum: Float = 0
                for i in 0..<mediumSize {
                    sum += mediumArray[i]
                }
                _ = sum
            },
            block: {
                var sum: Float = 0
                for i in 0..<mediumSize {
                    sum += mediumStorage[i]
                }
                _ = sum
            }
        ))
        
        // Large storage benchmarks
        let largeArray = (0..<largeSize).map { Float($0) }
        let largeStorage = LargeVectorStorage(count: largeSize, repeating: 1.0)
        
        benchmarks.append((
            name: "LargeStorage[\(largeSize)] access",
            category: "Storage",
            vectorSize: largeSize,
            baseline: {
                var sum: Float = 0
                for i in 0..<largeSize {
                    sum += largeArray[i]
                }
                _ = sum
            },
            block: {
                var sum: Float = 0
                for i in 0..<largeSize {
                    sum += largeStorage[i]
                }
                _ = sum
            }
        ))
        
        // COW (Copy-on-Write) benchmarks
        benchmarks.append((
            name: "LargeStorage[\(largeSize)] COW copy",
            category: "Storage",
            vectorSize: largeSize,
            baseline: {
                var copy = largeArray
                copy[0] = 42.0
            },
            block: {
                var copy = largeStorage
                copy[0] = 42.0
            }
        ))
        
        return BenchmarkFramework.runBenchmarkSuite(
            name: "Storage Types",
            configuration: configuration,
            benchmarks: benchmarks
        )
    }
    
    // MARK: - Batch Operations Benchmarks
    
    private static func benchmarkBatchOperations(configuration: BenchmarkFramework.Configuration) -> [BenchmarkFramework.MeasurementResult] {
        print("\n=== Batch Operations Benchmarks ===\n")
        
        var benchmarks: [(name: String, category: String, vectorSize: Int, baseline: (() -> Void)?, block: () -> Void)] = []
        
        // Test batch operations with different dataset sizes
        let vectorDim = 256
        let datasetSizes = [100, 1000, 5000]  // Below and above parallel threshold
        
        for datasetSize in datasetSizes {
            // Generate test data
            let vectors = (0..<datasetSize).map { i in
                Vector256((0..<vectorDim).map { j in
                    sin(Float(i * vectorDim + j) / 1000.0)
                })
            }
            let arrays = vectors.map { v in
                (0..<vectorDim).map { i in v[i] }
            }
            
            let query = vectors[0]
            let queryArray = arrays[0]
            
            // Batch normalization
            benchmarks.append((
                name: "Batch normalize [\(datasetSize)x\(vectorDim)]",
                category: "Batch Operations",
                vectorSize: datasetSize * vectorDim,
                baseline: {
                    _ = arrays.map { BaselineImplementations.normalize($0) }
                },
                block: {
                    _ = try? SyncBatchOperations.map(vectors) { $0.normalized() }
                }
            ))
            
            // Batch distance calculation
            benchmarks.append((
                name: "Batch distances [\(datasetSize)x\(vectorDim)]",
                category: "Batch Operations",
                vectorSize: datasetSize * vectorDim,
                baseline: {
                    _ = arrays.map { BaselineImplementations.distance(queryArray, $0) }
                },
                block: {
                    _ = SyncBatchOperations.distances(from: query, to: vectors)
                }
            ))
            
            // K-nearest neighbors
            let k = min(10, datasetSize / 2)
            benchmarks.append((
                name: "Batch k-NN (k=\(k)) [\(datasetSize)x\(vectorDim)]",
                category: "Batch Operations",
                vectorSize: datasetSize * vectorDim,
                baseline: {
                    // Naive k-NN
                    let distances = arrays.enumerated().map { (i, arr) in
                        (index: i, distance: BaselineImplementations.distance(queryArray, arr))
                    }
                    _ = distances.sorted { $0.distance < $1.distance }.prefix(k)
                },
                block: {
                    _ = SyncBatchOperations.findNearest(to: query, in: vectors, k: k)
                }
            ))
        }
        
        return BenchmarkFramework.runBenchmarkSuite(
            name: "Batch Operations",
            configuration: configuration,
            benchmarks: benchmarks
        )
    }
    
    // MARK: - Summary Report
    
    private static func generateSummaryReport(results: [BenchmarkFramework.MeasurementResult]) {
        print("\n=== Performance Summary ===\n")
        
        // Group by category
        let byCategory = Dictionary(grouping: results) { $0.category }
        
        for (category, categoryResults) in byCategory.sorted(by: { $0.key < $1.key }) {
            print("\(category):")
            
            // Find best performers
            let sorted = categoryResults.sorted { $0.opsPerSecond > $1.opsPerSecond }
            if let best = sorted.first {
                print("  Best: \(best.name) - \(formatOps(best.opsPerSecond)) ops/sec")
            }
            
            // Average speedup vs baseline
            let withBaseline = categoryResults.filter { $0.baselineComparison != nil }
            if !withBaseline.isEmpty {
                let avgSpeedup = withBaseline.map { $0.baselineComparison! }.reduce(0, +) / Double(withBaseline.count)
                print("  Avg speedup vs baseline: \(String(format: "%.2f", avgSpeedup))x")
            }
            
            print("")
        }
        
        // Overall statistics
        print("Total benchmarks run: \(results.count)")
        
        let totalSpeedups = results.compactMap { $0.baselineComparison }
        if !totalSpeedups.isEmpty {
            let overallSpeedup = totalSpeedups.reduce(0, +) / Double(totalSpeedups.count)
            print("Overall speedup vs baseline: \(String(format: "%.2f", overallSpeedup))x")
        }
    }
    
    private static func saveAllResults(_ results: [BenchmarkFramework.MeasurementResult], to path: String) {
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            
            let data = try encoder.encode(results)
            try data.write(to: URL(fileURLWithPath: path))
            
            print("\nAll results saved to: \(path)")
        } catch {
            print("Failed to save results: \(error)")
        }
    }
    
    private static func formatOps(_ ops: Double) -> String {
        if ops > 1e9 {
            return String(format: "%.2fG", ops / 1e9)
        } else if ops > 1e6 {
            return String(format: "%.2fM", ops / 1e6)
        } else if ops > 1e3 {
            return String(format: "%.2fK", ops / 1e3)
        } else {
            return String(format: "%.0f", ops)
        }
    }
}