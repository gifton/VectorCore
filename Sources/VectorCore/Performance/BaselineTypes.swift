// VectorCore: Baseline Types
//
// Shared types for performance baseline system
// This eliminates code duplication between scripts
//

import Foundation

// Note: These types are defined in PerformanceBaseline.swift in the same module
// No typealiases needed - they're already public and available
// HardwareMetrics is also already defined in PerformanceBaseline.swift

// MARK: - Benchmark Runner Protocol

/// Protocol for benchmark execution
public protocol BenchmarkRunner {
    /// Run all benchmarks and return results
    func runBenchmarks() async throws -> [BenchmarkResult]
    
    /// Run specific benchmark by name
    func runBenchmark(named: String) async throws -> BenchmarkResult?
    
    /// Get available benchmark names
    func availableBenchmarks() -> [String]
}

// MARK: - Extended Baseline
// Removed ExtendedPerformanceBaseline to avoid ambiguity issues

// MARK: - Benchmark Configuration

/// Configuration for benchmark execution
public struct BenchmarkConfiguration: Codable, Sendable {
    /// Number of warmup iterations
    public let warmupIterations: Int
    
    /// Number of measurement iterations
    public let measurementIterations: Int
    
    /// Timeout for individual benchmarks (seconds)
    public let timeoutSeconds: TimeInterval
    
    /// Whether to collect hardware metrics
    public let collectHardwareMetrics: Bool
    
    /// Whether to run benchmarks in parallel
    public let parallelExecution: Bool
    
    /// Specific benchmarks to run (nil = all)
    public let benchmarkFilter: [String]?
    
    public init(
        warmupIterations: Int = 10,
        measurementIterations: Int = 1000,
        timeoutSeconds: TimeInterval = 60,
        collectHardwareMetrics: Bool = true,
        parallelExecution: Bool = false,
        benchmarkFilter: [String]? = nil
    ) {
        self.warmupIterations = warmupIterations
        self.measurementIterations = measurementIterations
        self.timeoutSeconds = timeoutSeconds
        self.collectHardwareMetrics = collectHardwareMetrics
        self.parallelExecution = parallelExecution
        self.benchmarkFilter = benchmarkFilter
    }
    
    public static let `default` = BenchmarkConfiguration()
    
    public static let quick = BenchmarkConfiguration(
        warmupIterations: 5,
        measurementIterations: 100,
        timeoutSeconds: 10,
        collectHardwareMetrics: false
    )
    
    public static let comprehensive = BenchmarkConfiguration(
        warmupIterations: 50,
        measurementIterations: 10000,
        timeoutSeconds: 300,
        collectHardwareMetrics: true
    )
}

// MARK: - JSON Helpers

public extension JSONEncoder {
    /// Standard encoder configuration for baseline metrics
    static var baseline: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }
}

public extension JSONDecoder {
    /// Standard decoder configuration for baseline metrics
    static var baseline: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}