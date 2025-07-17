import Foundation
import VectorCore

/// Benchmark suite for comparing performance across different vector dimensions
///
/// This benchmark helps identify:
/// - Performance scaling characteristics
/// - Optimal dimension ranges for different operations
/// - SIMD utilization efficiency
/// - Cache effects on larger vectors
public struct DimensionComparisonBenchmark {
    
    // MARK: - Configuration
    
    public struct DimensionProfile {
        let dimension: Int
        let name: String
        let expectedUsage: String
        
        static let profiles = [
            DimensionProfile(dimension: 32, name: "Tiny", expectedUsage: "Small feature vectors"),
            DimensionProfile(dimension: 64, name: "Small", expectedUsage: "Compact embeddings"),
            DimensionProfile(dimension: 128, name: "Medium", expectedUsage: "Standard features"),
            DimensionProfile(dimension: 256, name: "Standard", expectedUsage: "Common embeddings"),
            DimensionProfile(dimension: 384, name: "BERT-Small", expectedUsage: "Small transformer models"),
            DimensionProfile(dimension: 512, name: "Large", expectedUsage: "Image embeddings"),
            DimensionProfile(dimension: 768, name: "BERT-Base", expectedUsage: "Standard transformer models"),
            DimensionProfile(dimension: 1024, name: "XLarge", expectedUsage: "Large language models"),
            DimensionProfile(dimension: 1536, name: "GPT", expectedUsage: "OpenAI embeddings"),
            DimensionProfile(dimension: 3072, name: "XXLarge", expectedUsage: "Very large models")
        ]
    }
    
    // MARK: - Performance Results
    
    public struct DimensionResult {
        let profile: DimensionProfile
        let operations: [String: OperationMetrics]
        let optimalBatchSize: Int
        let memoryEfficiency: Double
        let simdUtilization: Double
        
        struct OperationMetrics {
            let opsPerSecond: Double
            let bytesPerOp: Double
            let scalingFactor: Double  // Relative to dimension 32
        }
    }
    
    // MARK: - Main Benchmark
    
    public static func run(configuration: BenchmarkFramework.Configuration = .default) {
        print("\nVector Dimension Performance Comparison")
        print("======================================")
        print("Analyzing performance characteristics across dimensions...\n")
        
        var results: [DimensionResult] = []
        
        // Baseline metrics from dimension 32
        var baselineMetrics: [String: Double] = [:]
        
        for profile in DimensionProfile.profiles {
            print("Testing \(profile.name) (dim=\(profile.dimension)) - \(profile.expectedUsage)")
            
            let result = benchmarkDimension(
                profile: profile,
                configuration: configuration,
                baselineMetrics: baselineMetrics.isEmpty ? nil : baselineMetrics
            )
            
            // Store baseline metrics from first dimension
            if baselineMetrics.isEmpty {
                for (op, metrics) in result.operations {
                    baselineMetrics[op] = metrics.opsPerSecond
                }
            }
            
            results.append(result)
            printDimensionSummary(result)
            print("")
        }
        
        // Generate comparative analysis
        generateComparativeAnalysis(results)
    }
    
    // MARK: - Dimension Benchmarking
    
    private static func benchmarkDimension(
        profile: DimensionProfile,
        configuration: BenchmarkFramework.Configuration,
        baselineMetrics: [String: Double]?
    ) -> DimensionResult {
        let dim = profile.dimension
        var operations: [String: DimensionResult.OperationMetrics] = [:]
        
        // Generate test data
        let array1 = (0..<dim).map { sin(Float($0) * 0.1) }
        let array2 = (0..<dim).map { cos(Float($0) * 0.1) }
        
        // Test core operations
        let operationTests: [(name: String, block: () -> Void)] = [
            ("Addition", {
                performOperation(dimension: dim, array1: array1, array2: array2) { v1, v2 in
                    _ = v1 + v2
                }
            }),
            ("DotProduct", {
                performOperation(dimension: dim, array1: array1, array2: array2) { v1, v2 in
                    _ = v1.dotProduct(v2)
                }
            }),
            ("Magnitude", {
                performOperation(dimension: dim, array1: array1, array2: array2) { v1, _ in
                    _ = v1.magnitude
                }
            }),
            ("Normalize", {
                performOperation(dimension: dim, array1: array1, array2: array2) { v1, _ in
                    _ = v1.normalized()
                }
            }),
            ("Distance", {
                performOperation(dimension: dim, array1: array1, array2: array2) { v1, v2 in
                    _ = v1.distance(to: v2)
                }
            }),
            ("CosineSimilarity", {
                performOperation(dimension: dim, array1: array1, array2: array2) { v1, v2 in
                    _ = v1.cosineSimilarity(to: v2)
                }
            })
        ]
        
        // Benchmark each operation
        for (opName, block) in operationTests {
            let result = BenchmarkFramework.measure(
                name: "\(profile.name)-\(opName)",
                category: "Dimension",
                vectorSize: dim,
                configuration: configuration,
                block: block
            )
            
            let scalingFactor = baselineMetrics?[opName].map { baseline in
                result.opsPerSecond / baseline * (32.0 / Double(dim))
            } ?? 1.0
            
            operations[opName] = DimensionResult.OperationMetrics(
                opsPerSecond: result.opsPerSecond,
                bytesPerOp: result.bytesAllocatedPerOp ?? 0,
                scalingFactor: scalingFactor
            )
        }
        
        // Calculate optimal batch size
        let optimalBatchSize = calculateOptimalBatchSize(dimension: dim)
        
        // Calculate memory efficiency
        let memoryEfficiency = calculateMemoryEfficiency(dimension: dim)
        
        // Calculate SIMD utilization
        let simdUtilization = calculateSIMDUtilization(dimension: dim)
        
        return DimensionResult(
            profile: profile,
            operations: operations,
            optimalBatchSize: optimalBatchSize,
            memoryEfficiency: memoryEfficiency,
            simdUtilization: simdUtilization
        )
    }
    
    // MARK: - Helper Functions
    
    private static func performOperation(
        dimension: Int,
        array1: [Float],
        array2: [Float],
        operation: (any ExtendedVectorProtocol, any ExtendedVectorProtocol) -> Void
    ) {
        switch dimension {
        case 32:
            let v1 = Vector<Dim32>(array1)
            let v2 = Vector<Dim32>(array2)
            operation(v1, v2)
        case 64:
            let v1 = Vector<Dim64>(array1)
            let v2 = Vector<Dim64>(array2)
            operation(v1, v2)
        case 128:
            let v1 = Vector128(array1)
            let v2 = Vector128(array2)
            operation(v1, v2)
        case 256:
            let v1 = Vector256(array1)
            let v2 = Vector256(array2)
            operation(v1, v2)
        case 384:
            let v1 = Vector384(array1)
            let v2 = Vector384(array2)
            operation(v1, v2)
        case 512:
            let v1 = Vector512(array1)
            let v2 = Vector512(array2)
            operation(v1, v2)
        case 768:
            let v1 = Vector768(array1)
            let v2 = Vector768(array2)
            operation(v1, v2)
        case 1024:
            let v1 = Vector1024(array1)
            let v2 = Vector1024(array2)
            operation(v1, v2)
        case 1536:
            let v1 = Vector1536(array1)
            let v2 = Vector1536(array2)
            operation(v1, v2)
        case 3072:
            let v1 = Vector3072(array1)
            let v2 = Vector3072(array2)
            operation(v1, v2)
        default:
            let v1 = DynamicVector(array1)
            let v2 = DynamicVector(array2)
            operation(v1, v2)
        }
    }
    
    private static func calculateOptimalBatchSize(dimension: Int) -> Int {
        // Estimate based on L1/L2 cache sizes and vector dimension
        let l2CacheSize = 256 * 1024  // 256KB typical L2 cache
        let bytesPerVector = dimension * MemoryLayout<Float>.size
        let vectorsInL2 = l2CacheSize / bytesPerVector
        
        // Return a power of 2 that fits well in cache
        var batchSize = 1
        while batchSize * 2 <= vectorsInL2 && batchSize < 1024 {
            batchSize *= 2
        }
        return batchSize
    }
    
    private static func calculateMemoryEfficiency(dimension: Int) -> Double {
        // Calculate how efficiently the dimension uses memory alignment
        let floatsPerCacheLine = 64 / MemoryLayout<Float>.size  // 64-byte cache line
        let fullCacheLines = dimension / Int(floatsPerCacheLine)
        let remainder = dimension % Int(floatsPerCacheLine)
        
        if remainder == 0 {
            return 1.0
        } else {
            let usedBytes = dimension * MemoryLayout<Float>.size
            let allocatedBytes = (fullCacheLines + 1) * 64
            return Double(usedBytes) / Double(allocatedBytes)
        }
    }
    
    private static func calculateSIMDUtilization(dimension: Int) -> Double {
        // Calculate how well the dimension utilizes SIMD registers
        let simdWidth = 4  // Float32 SIMD width (4 for SSE, 8 for AVX)
        let fullSIMDOps = dimension / simdWidth
        let remainder = dimension % simdWidth
        
        if remainder == 0 {
            return 1.0
        } else {
            return Double(dimension) / Double((fullSIMDOps + 1) * simdWidth)
        }
    }
    
    // MARK: - Output Formatting
    
    private static func printDimensionSummary(_ result: DimensionResult) {
        print("  Operations:")
        for (op, metrics) in result.operations.sorted(by: { $0.key < $1.key }) {
            print("    \(op): \(formatOps(metrics.opsPerSecond)) ops/sec (scaling: \(String(format: "%.2f", metrics.scalingFactor)))")
        }
        print("  Optimal batch size: \(result.optimalBatchSize) vectors")
        print("  Memory efficiency: \(String(format: "%.1f%%", result.memoryEfficiency * 100))")
        print("  SIMD utilization: \(String(format: "%.1f%%", result.simdUtilization * 100))")
    }
    
    private static func generateComparativeAnalysis(_ results: [DimensionResult]) {
        print("\nComparative Analysis")
        print("===================")
        
        // Find best dimension for each operation
        print("\nBest dimensions by operation:")
        let allOps = Set(results.flatMap { $0.operations.keys })
        
        for op in allOps.sorted() {
            let bestDim = results.max { r1, r2 in
                let ops1 = r1.operations[op]?.opsPerSecond ?? 0
                let ops2 = r2.operations[op]?.opsPerSecond ?? 0
                return ops1 < ops2
            }
            
            if let best = bestDim, let metrics = best.operations[op] {
                print("  \(op): dim=\(best.profile.dimension) (\(formatOps(metrics.opsPerSecond)) ops/sec)")
            }
        }
        
        // Scaling analysis
        print("\nScaling characteristics:")
        for result in results {
            let avgScaling = result.operations.values.map { $0.scalingFactor }.reduce(0, +) / Double(result.operations.count)
            print("  dim=\(result.profile.dimension): \(String(format: "%.2fx", avgScaling)) average scaling factor")
        }
        
        // Memory and SIMD efficiency
        print("\nEfficiency metrics:")
        let mostMemEfficient = results.max { $0.memoryEfficiency < $1.memoryEfficiency }
        let mostSIMDEfficient = results.max { $0.simdUtilization < $1.simdUtilization }
        
        if let memBest = mostMemEfficient {
            print("  Best memory efficiency: dim=\(memBest.profile.dimension) (\(String(format: "%.1f%%", memBest.memoryEfficiency * 100)))")
        }
        if let simdBest = mostSIMDEfficient {
            print("  Best SIMD utilization: dim=\(simdBest.profile.dimension) (\(String(format: "%.1f%%", simdBest.simdUtilization * 100)))")
        }
        
        // Recommendations
        print("\nRecommendations:")
        print("  - For maximum throughput: Use dimensions that are multiples of 16 (4 floats × 4 SIMD lanes)")
        print("  - For cache efficiency: Consider dimensions that fit in L1/L2 cache")
        print("  - For batch operations: Use the calculated optimal batch sizes")
        
        // Performance sweet spots
        let sweetSpots = results.filter { $0.memoryEfficiency >= 0.95 && $0.simdUtilization >= 0.95 }
        if !sweetSpots.isEmpty {
            print("\nPerformance sweet spots (>95% efficiency):")
            for spot in sweetSpots {
                print("  - dim=\(spot.profile.dimension): \(spot.profile.expectedUsage)")
            }
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