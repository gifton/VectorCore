// VectorCore: Performance Baseline
//
// Structures and utilities for capturing and comparing performance metrics
//

import Foundation

/// Represents a complete performance baseline for VectorCore operations
public struct PerformanceBaseline: Codable {
    /// Timestamp when the baseline was captured
    public let timestamp: Date
    
    /// Swift version used for the baseline
    public let swiftVersion: String
    
    /// Platform information
    public let platform: PlatformInfo
    
    /// Operation throughput metrics (operations per second)
    public let throughput: ThroughputMetrics
    
    /// Memory efficiency metrics
    public let memory: MemoryMetrics
    
    /// Parallelization efficiency metrics
    public let parallelization: ParallelizationMetrics
    
    /// Individual benchmark results
    public let benchmarks: [BenchmarkResult]
    
    /// Hardware performance metrics (optional for backward compatibility)
    public let hardware: HardwareMetrics?
    
    public init(
        timestamp: Date = Date(),
        swiftVersion: String? = nil,
        platform: PlatformInfo = PlatformInfo.current,
        throughput: ThroughputMetrics,
        memory: MemoryMetrics,
        parallelization: ParallelizationMetrics,
        benchmarks: [BenchmarkResult],
        hardware: HardwareMetrics? = nil
    ) {
        self.timestamp = timestamp
        self.swiftVersion = swiftVersion ?? getCurrentSwiftVersion()
        self.platform = platform
        self.throughput = throughput
        self.memory = memory
        self.parallelization = parallelization
        self.benchmarks = benchmarks
        self.hardware = hardware
    }
}

/// Platform information for baseline context
public struct PlatformInfo: Codable {
    public let os: String
    public let osVersion: String
    public let architecture: String
    public let cpuCores: Int
    public let memoryGB: Double
    
    /// Human-readable description of the platform
    public var description: String {
        "\(os) \(osVersion) (\(architecture)) - \(cpuCores) cores, \(String(format: "%.1f", memoryGB))GB RAM"
    }
    
    public static var current: PlatformInfo {
        #if os(macOS)
        let os = "macOS"
        #elseif os(iOS)
        let os = "iOS"
        #else
        let os = "Unknown"
        #endif
        
        let processInfo = ProcessInfo.processInfo
        
        return PlatformInfo(
            os: os,
            osVersion: processInfo.operatingSystemVersionString,
            architecture: getCurrentArchitecture(),
            cpuCores: processInfo.processorCount,
            memoryGB: Double(processInfo.physicalMemory) / 1_073_741_824.0
        )
    }
}

/// Throughput metrics for vector operations
public struct ThroughputMetrics: Codable {
    /// Vector addition operations per second
    public let vectorAddition: Double
    
    /// Vector scalar multiplication operations per second
    public let vectorScalarMultiplication: Double
    
    /// Vector element-wise multiplication operations per second
    public let vectorElementwiseMultiplication: Double
    
    /// Dot product operations per second
    public let dotProduct: Double
    
    /// Euclidean distance calculations per second
    public let euclideanDistance: Double
    
    /// Cosine similarity calculations per second
    public let cosineSimilarity: Double
    
    /// Manhattan distance calculations per second
    public let manhattanDistance: Double
    
    /// Vector normalization operations per second
    public let normalization: Double
}

/// Memory efficiency metrics
public struct MemoryMetrics: Codable {
    /// Average bytes allocated per vector operation
    public let bytesPerOperation: Int
    
    /// Peak memory usage during benchmarks (bytes)
    public let peakMemoryUsage: Int
    
    /// Memory allocation rate (allocations per second)
    public let allocationRate: Double
    
    /// Average bytes per vector storage
    public let bytesPerVector: [String: Int] // dimension -> bytes
}

/// Parallelization efficiency metrics
public struct ParallelizationMetrics: Codable {
    /// Speedup factor when using parallel operations
    public let parallelSpeedup: Double
    
    /// Scaling efficiency (0-1, where 1 is perfect scaling)
    public let scalingEfficiency: Double
    
    /// Optimal batch size for parallel operations
    public let optimalBatchSize: Int
    
    /// Thread utilization percentage
    public let threadUtilization: Double
}

/// Hardware-level performance metrics for deeper analysis
public struct HardwareMetrics: Codable {
    /// SIMD operation utilization (0-1, where 1 is full utilization)
    public let simdUtilization: Double
    
    /// L1 cache hit rate (0-1)
    public let l1CacheHitRate: Double
    
    /// L2 cache hit rate (0-1)
    public let l2CacheHitRate: Double
    
    /// L3 cache hit rate (0-1)
    public let l3CacheHitRate: Double
    
    /// Memory bandwidth utilization (GB/s)
    public let memoryBandwidthGBps: Double
    
    /// Average CPU frequency during benchmarks (GHz)
    public let avgCPUFrequencyGHz: Double
    
    /// CPU utilization percentage (0-100)
    public let cpuUtilization: Double
    
    /// Thread context switches per second
    public let contextSwitchesPerSec: Double
    
    public init(
        simdUtilization: Double,
        l1CacheHitRate: Double,
        l2CacheHitRate: Double,
        l3CacheHitRate: Double,
        memoryBandwidthGBps: Double,
        avgCPUFrequencyGHz: Double,
        cpuUtilization: Double,
        contextSwitchesPerSec: Double
    ) {
        self.simdUtilization = simdUtilization
        self.l1CacheHitRate = l1CacheHitRate
        self.l2CacheHitRate = l2CacheHitRate
        self.l3CacheHitRate = l3CacheHitRate
        self.memoryBandwidthGBps = memoryBandwidthGBps
        self.avgCPUFrequencyGHz = avgCPUFrequencyGHz
        self.cpuUtilization = cpuUtilization
        self.contextSwitchesPerSec = contextSwitchesPerSec
    }
}

/// Individual benchmark result
public struct BenchmarkResult: Codable {
    public let name: String
    public let iterations: Int
    public let totalTime: TimeInterval
    public let averageTime: TimeInterval
    public let standardDeviation: TimeInterval
    public let throughput: Double // ops/sec
    public let memoryAllocated: Int? // bytes
    
    public init(
        name: String,
        iterations: Int,
        totalTime: TimeInterval,
        averageTime: TimeInterval,
        standardDeviation: TimeInterval,
        throughput: Double,
        memoryAllocated: Int? = nil
    ) {
        self.name = name
        self.iterations = iterations
        self.totalTime = totalTime
        self.averageTime = averageTime
        self.standardDeviation = standardDeviation
        self.throughput = throughput
        self.memoryAllocated = memoryAllocated
    }
}

// MARK: - Helper Functions

internal func getCurrentSwiftVersion() -> String {
    #if swift(>=6.0)
    return "6.0"
    #elseif swift(>=5.9)
    return "5.9"
    #elseif swift(>=5.8)
    return "5.8"
    #elseif swift(>=5.7)
    return "5.7"
    #else
    return "Unknown"
    #endif
}

private func getCurrentArchitecture() -> String {
    #if arch(arm64)
    return "arm64"
    #elseif arch(x86_64)
    return "x86_64"
    #else
    return "Unknown"
    #endif
}

// MARK: - Baseline Comparison

public struct PerformanceComparison {
    public let baseline: PerformanceBaseline
    public let current: PerformanceBaseline
    public let regressionThreshold: Double // e.g., 0.05 for 5%
    
    public init(baseline: PerformanceBaseline, current: PerformanceBaseline, regressionThreshold: Double = 0.05) {
        self.baseline = baseline
        self.current = current
        self.regressionThreshold = regressionThreshold
    }
    
    /// Check if any regression exceeds the threshold
    public func hasRegression() -> Bool {
        let regressions = detectRegressions()
        return !regressions.isEmpty
    }
    
    /// Detect specific operations with performance regression
    public func detectRegressions() -> [BaselineRegressionResult] {
        var regressions: [BaselineRegressionResult] = []
        
        // Compare throughput metrics
        let throughputRegressions = compareThroughput()
        regressions.append(contentsOf: throughputRegressions)
        
        // Compare memory metrics
        let memoryRegressions = compareMemory()
        regressions.append(contentsOf: memoryRegressions)
        
        return regressions
    }
    
    private func compareThroughput() -> [BaselineRegressionResult] {
        var results: [BaselineRegressionResult] = []
        
        // Helper to check regression
        func checkRegression(name: String, baseline: Double, current: Double) {
            let percentChange = (current - baseline) / baseline
            if percentChange < -regressionThreshold {
                results.append(BaselineRegressionResult(
                    operation: name,
                    baselineValue: baseline,
                    currentValue: current,
                    percentChange: percentChange,
                    type: RegressionType.throughput
                ))
            }
        }
        
        checkRegression(
            name: "Vector Addition",
            baseline: baseline.throughput.vectorAddition,
            current: current.throughput.vectorAddition
        )
        
        checkRegression(
            name: "Dot Product",
            baseline: baseline.throughput.dotProduct,
            current: current.throughput.dotProduct
        )
        
        checkRegression(
            name: "Euclidean Distance",
            baseline: baseline.throughput.euclideanDistance,
            current: current.throughput.euclideanDistance
        )
        
        checkRegression(
            name: "Cosine Similarity",
            baseline: baseline.throughput.cosineSimilarity,
            current: current.throughput.cosineSimilarity
        )
        
        return results
    }
    
    private func compareMemory() -> [BaselineRegressionResult] {
        var results: [BaselineRegressionResult] = []
        
        // Check if memory usage increased beyond threshold
        let baseline = Double(baseline.memory.bytesPerOperation)
        let current = Double(current.memory.bytesPerOperation)
        let percentChange = (current - baseline) / baseline
        
        if percentChange > regressionThreshold {
            results.append(BaselineRegressionResult(
                operation: "Memory per Operation",
                baselineValue: baseline,
                currentValue: current,
                percentChange: percentChange,
                type: RegressionType.memory
            ))
        }
        
        return results
    }
}

public struct BaselineRegressionResult {
    public let operation: String
    public let baselineValue: Double
    public let currentValue: Double
    public let percentChange: Double
    public let type: RegressionType
    
    public init(
        operation: String,
        baselineValue: Double,
        currentValue: Double,
        percentChange: Double,
        type: RegressionType
    ) {
        self.operation = operation
        self.baselineValue = baselineValue
        self.currentValue = currentValue
        self.percentChange = percentChange
        self.type = type
    }
    
    public var description: String {
        let changeStr = String(format: "%.1f%%", percentChange * 100)
        return "\(operation): \(changeStr) regression (\(baselineValue) → \(currentValue))"
    }
}

public enum RegressionType {
    case throughput
    case memory
    case parallelization
}