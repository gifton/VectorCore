// VectorCore: Benchmark Adapter
//
// Bridges swift-benchmark package with VectorCore's performance baseline system
//

import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// Adapter to convert benchmark results to baseline metrics
public struct BenchmarkAdapter {
    
    /// Convert raw benchmark results to PerformanceBaseline
    public static func createBaseline(
        from benchmarkResults: [BenchmarkResult],
        platform: PlatformInfo? = nil,
        hardware: HardwareMetrics? = nil
    ) -> PerformanceBaseline {
        
        // Group results by operation type
        let throughputMetrics = extractThroughputMetrics(from: benchmarkResults)
        let memoryMetrics = extractMemoryMetrics(from: benchmarkResults)
        let parallelizationMetrics = extractParallelizationMetrics(from: benchmarkResults)
        
        return PerformanceBaseline(
            timestamp: Date(),
            swiftVersion: nil, // Will use getCurrentSwiftVersion()
            platform: platform ?? PlatformInfo.current,
            throughput: throughputMetrics,
            memory: memoryMetrics,
            parallelization: parallelizationMetrics,
            benchmarks: benchmarkResults,
            hardware: hardware
        )
    }
    
    /// Extract throughput metrics from benchmark results
    private static func extractThroughputMetrics(from results: [BenchmarkResult]) -> ThroughputMetrics {
        // Helper to find result by name pattern
        func findResult(matching pattern: String) -> Double {
            if let result = results.first(where: { $0.name.contains(pattern) }) {
                return result.throughput
            }
            return 0.0
        }
        
        return ThroughputMetrics(
            vectorAddition: findResult(matching: "Vector Addition - 768D"),
            vectorScalarMultiplication: findResult(matching: "Vector Scalar Multiplication"),
            vectorElementwiseMultiplication: findResult(matching: "Element-wise Multiplication"),
            dotProduct: findResult(matching: "Vector Dot Product - 768D"),
            euclideanDistance: findResult(matching: "Euclidean Distance - 768D"),
            cosineSimilarity: findResult(matching: "Cosine Similarity - 768D"),
            manhattanDistance: findResult(matching: "Manhattan Distance - 768D"),
            normalization: findResult(matching: "Vector Normalization")
        )
    }
    
    /// Extract memory metrics from benchmark results
    private static func extractMemoryMetrics(from results: [BenchmarkResult]) -> MemoryMetrics {
        // Track current memory usage for better peak estimation
        var memoryTracker = BenchmarkMemoryTracker()
        
        // Calculate average memory per operation
        let memoryResults = results.filter { $0.memoryAllocated != nil && $0.memoryAllocated! > 0 }
        let totalMemory = memoryResults.compactMap { $0.memoryAllocated }.reduce(0, +)
        let totalOps = memoryResults.map { $0.iterations }.reduce(0, +)
        let bytesPerOp = totalOps > 0 ? totalMemory / totalOps : 0
        
        // Get actual peak memory from results or use current memory tracking
        var peakMemory = 0
        for result in results {
            memoryTracker.update()
            if let allocated = result.memoryAllocated {
                peakMemory = max(peakMemory, allocated)
            }
        }
        
        // If no peak found, use tracker's peak
        if peakMemory == 0 {
            peakMemory = memoryTracker.peak
        }
        
        // Calculate allocation rate more accurately
        let totalTime = results.map { $0.totalTime }.reduce(0, +)
        let allocRate = totalTime > 0 ? Double(totalMemory) / totalTime : 0
        
        // Memory by dimension - calculate actual bytes per vector
        func memoryForDimension(_ dim: Int) -> Int {
            let pattern = "\\(dim)D"
            let dimResults = results.filter { $0.name.contains(pattern) }
            guard !dimResults.isEmpty else { 
                // Calculate theoretical memory for vector of given dimension
                return dim * MemoryLayout<Float>.size
            }
            
            // Find results with memory data
            let memResults = dimResults.filter { $0.memoryAllocated != nil && $0.memoryAllocated! > 0 }
            if memResults.isEmpty {
                // Use theoretical calculation
                return dim * MemoryLayout<Float>.size
            }
            
            // Average memory per operation for this dimension
            let totalMem = memResults.compactMap { $0.memoryAllocated }.reduce(0, +)
            let totalIterations = memResults.map { $0.iterations }.reduce(0, +)
            
            // Estimate bytes per vector based on memory per iteration
            if totalIterations > 0 {
                let memPerIteration = totalMem / totalIterations
                // Assume each iteration works with 2 vectors (common case)
                return memPerIteration / 2
            }
            
            return dim * MemoryLayout<Float>.size
        }
        
        return MemoryMetrics(
            bytesPerOperation: bytesPerOp,
            peakMemoryUsage: peakMemory,
            allocationRate: allocRate,
            bytesPerVector: [
                "128": memoryForDimension(128),
                "256": memoryForDimension(256),
                "512": memoryForDimension(512),
                "768": memoryForDimension(768),
                "1536": memoryForDimension(1536)
            ]
        )
    }
    
    /// Extract parallelization metrics from benchmark results
    private static func extractParallelizationMetrics(from results: [BenchmarkResult]) -> ParallelizationMetrics {
        // Look for parallel vs sequential benchmarks
        let batchResults = results.filter { $0.name.contains("Batch") }
        let sequentialResults = results.filter { !$0.name.contains("Batch") && !$0.name.contains("Parallel") }
        
        // Calculate speedup by comparing batch operations to sequential
        var speedup = 1.0
        var efficiency = 1.0
        
        // Find matching pairs of batch vs sequential operations
        for batchResult in batchResults {
            // Extract the operation type (e.g., "Euclidean Distance", "Cosine Similarity")
            let components = batchResult.name.components(separatedBy: " - ")
            if components.count >= 2 {
                let operation = components[0].replacingOccurrences(of: "Batch ", with: "")
                
                // Find corresponding sequential benchmark
                if let seqResult = sequentialResults.first(where: { $0.name.contains(operation) }) {
                    // Extract batch size from name (e.g., "100x768D" -> 100)
                    if let batchSizeMatch = batchResult.name.range(of: "(\\d+)x", options: .regularExpression) {
                        let batchSizeStr = String(batchResult.name[batchSizeMatch]).dropLast() // Remove 'x'
                        if let batchSize = Int(batchSizeStr) {
                            // Calculate operations per second for fair comparison
                            let batchOpsPerSec = batchResult.throughput * Double(batchSize)
                            let seqOpsPerSec = seqResult.throughput
                            
                            // Speedup is how much faster batch processing is
                            if seqOpsPerSec > 0 {
                                speedup = max(speedup, batchOpsPerSec / seqOpsPerSec)
                            }
                        }
                    }
                }
            }
        }
        
        // Calculate efficiency based on CPU cores and actual speedup
        let cpuCount = Double(ProcessInfo.processInfo.processorCount)
        efficiency = min(speedup / cpuCount, 1.0)
        
        // Find optimal batch size by analyzing throughput across different batch sizes
        var batchPerformance: [(size: Int, throughput: Double)] = []
        
        for result in batchResults {
            if let batchSizeMatch = result.name.range(of: "(\\d+)x", options: .regularExpression) {
                let batchSizeStr = String(result.name[batchSizeMatch]).dropLast()
                if let batchSize = Int(batchSizeStr) {
                    // Normalize throughput by batch size for fair comparison
                    let normalizedThroughput = result.throughput * Double(batchSize)
                    batchPerformance.append((size: batchSize, throughput: normalizedThroughput))
                }
            }
        }
        
        // Find optimal batch size
        let optimalBatch = batchPerformance.max(by: { $0.throughput < $1.throughput })?.size ?? 100
        
        // Calculate thread utilization based on actual performance
        // This is a more sophisticated calculation based on Amdahl's law
        let parallelFraction = 0.9 // Assume 90% of work can be parallelized
        let theoreticalSpeedup = 1.0 / ((1.0 - parallelFraction) + parallelFraction / cpuCount)
        let utilization = (speedup / theoreticalSpeedup) * 100.0
        
        return ParallelizationMetrics(
            parallelSpeedup: speedup,
            scalingEfficiency: efficiency,
            optimalBatchSize: optimalBatch,
            threadUtilization: min(utilization, 100.0)
        )
    }
}

// BenchmarkResult extension removed - using the existing API

/// Placeholder for benchmark statistics
/// This will be replaced with actual swift-benchmark types
public struct BenchmarkStatistics {
    public let mean: Double
    public let standardDeviation: Double
    public let min: Double
    public let max: Double
    public let throughput: Double
    
    public init(
        mean: Double,
        standardDeviation: Double,
        min: Double,
        max: Double,
        throughput: Double
    ) {
        self.mean = mean
        self.standardDeviation = standardDeviation
        self.min = min
        self.max = max
        self.throughput = throughput
    }
}

/// Memory tracking for benchmarks
public struct BenchmarkMemoryTracker {
    private var initialMemory: Int
    private var peakMemory: Int
    
    public init() {
        self.initialMemory = Self.currentMemoryUsage()
        self.peakMemory = self.initialMemory
    }
    
    public mutating func update() {
        let current = Self.currentMemoryUsage()
        peakMemory = max(peakMemory, current)
    }
    
    public var memoryUsed: Int {
        Self.currentMemoryUsage() - initialMemory
    }
    
    public var peak: Int {
        peakMemory - initialMemory
    }
    
    private static func currentMemoryUsage() -> Int {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        return result == KERN_SUCCESS ? Int(info.resident_size) : 0
    }
}

/// Protocol for benchmark runners to implement
public protocol BenchmarkRunnerProtocol {
    /// Run benchmarks and return results
    func run() async throws -> [BenchmarkResult]
    
    /// Run specific benchmark by name
    func run(benchmarkNamed name: String) async throws -> BenchmarkResult?
    
    /// List available benchmarks
    var availableBenchmarks: [String] { get }
}

// Hardware metrics collection is now implemented in HardwareMetricsCollector.swift