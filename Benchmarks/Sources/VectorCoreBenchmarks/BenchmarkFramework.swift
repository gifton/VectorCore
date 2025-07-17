import Foundation
import VectorCore

/// Core benchmark framework for VectorCore performance testing
///
/// Provides comprehensive performance measurement including:
/// - Execution time with statistical analysis
/// - Memory allocation tracking
/// - Throughput calculation
/// - JSON output for regression detection
public struct BenchmarkFramework {
    
    // MARK: - Configuration
    
    public struct Configuration {
        public let warmupIterations: Int
        public let measurementIterations: Int
        public let statisticalRuns: Int  // Number of runs for statistical significance
        public let includeMemoryMetrics: Bool
        public let jsonOutputPath: String?
        
        public static let `default` = Configuration(
            warmupIterations: 10,
            measurementIterations: 1000,
            statisticalRuns: 5,
            includeMemoryMetrics: true,
            jsonOutputPath: nil
        )
        
        public static let quick = Configuration(
            warmupIterations: 5,
            measurementIterations: 100,
            statisticalRuns: 3,
            includeMemoryMetrics: false,
            jsonOutputPath: nil
        )
        
        public static let comprehensive = Configuration(
            warmupIterations: 20,
            measurementIterations: 5000,
            statisticalRuns: 10,
            includeMemoryMetrics: true,
            jsonOutputPath: "benchmark_results.json"
        )
    }
    
    // MARK: - Measurement Results
    
    public struct MeasurementResult: Codable {
        public let name: String
        public let category: String
        public let vectorSize: Int
        public let iterations: Int
        
        // Time metrics (in seconds)
        public let meanTime: Double
        public let medianTime: Double
        public let minTime: Double
        public let maxTime: Double
        public let stdDeviation: Double
        public let percentile95: Double
        public let percentile99: Double
        
        // Throughput metrics
        public let opsPerSecond: Double
        public let elementsPerSecond: Double?
        
        // Memory metrics (optional)
        public let allocationsPerOp: Double?
        public let bytesAllocatedPerOp: Double?
        
        // Comparison metrics
        public let baselineComparison: Double?  // Speedup vs baseline
        
        // Metadata
        public let timestamp: Date
        public let platform: String
        public let swiftVersion: String
        
        public var formattedOutput: String {
            var output = "\(name):"
            output += "\n  Time: \(formatTime(meanTime)) (σ=\(formatTime(stdDeviation)))"
            output += "\n  Range: [\(formatTime(minTime)) - \(formatTime(maxTime))]"
            output += "\n  P95: \(formatTime(percentile95)), P99: \(formatTime(percentile99))"
            output += "\n  Throughput: \(formatOps(opsPerSecond)) ops/sec"
            
            if let elementsPerSec = elementsPerSecond {
                output += ", \(formatOps(elementsPerSec)) elements/sec"
            }
            
            if let allocsPerOp = allocationsPerOp, let bytesPerOp = bytesAllocatedPerOp {
                output += "\n  Memory: \(Int(allocsPerOp)) allocations, \(formatBytes(bytesPerOp))/op"
            }
            
            if let speedup = baselineComparison {
                output += "\n  Speedup: \(String(format: "%.2f", speedup))x vs baseline"
            }
            
            return output
        }
        
        private func formatTime(_ time: Double) -> String {
            if time < 1e-6 {
                return String(format: "%.1f ns", time * 1e9)
            } else if time < 1e-3 {
                return String(format: "%.1f µs", time * 1e6)
            } else if time < 1.0 {
                return String(format: "%.1f ms", time * 1e3)
            } else {
                return String(format: "%.2f s", time)
            }
        }
        
        private func formatOps(_ ops: Double) -> String {
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
        
        private func formatBytes(_ bytes: Double) -> String {
            if bytes < 1024 {
                return String(format: "%.0f B", bytes)
            } else if bytes < 1024 * 1024 {
                return String(format: "%.1f KB", bytes / 1024)
            } else {
                return String(format: "%.1f MB", bytes / (1024 * 1024))
            }
        }
    }
    
    // MARK: - Memory Tracking
    
    private struct MemoryInfo {
        let allocations: Int
        let bytesAllocated: Int
        
        static func capture() -> MemoryInfo {
            var info = mach_task_basic_info()
            var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
            
            let result = withUnsafeMutablePointer(to: &info) { pointer in
                pointer.withMemoryRebound(to: integer_t.self, capacity: 1) { intPointer in
                    task_info(mach_task_self_,
                             task_flavor_t(MACH_TASK_BASIC_INFO),
                             intPointer,
                             &count)
                }
            }
            
            if result == KERN_SUCCESS {
                return MemoryInfo(
                    allocations: 0,  // Would need malloc_zone tracking for accurate count
                    bytesAllocated: Int(info.resident_size)
                )
            }
            
            return MemoryInfo(allocations: 0, bytesAllocated: 0)
        }
        
        static func delta(from start: MemoryInfo, to end: MemoryInfo) -> MemoryInfo {
            return MemoryInfo(
                allocations: end.allocations - start.allocations,
                bytesAllocated: end.bytesAllocated - start.bytesAllocated
            )
        }
    }
    
    // MARK: - Benchmark Execution
    
    /// Execute a benchmark with comprehensive measurements
    public static func measure(
        name: String,
        category: String,
        vectorSize: Int,
        configuration: Configuration = .default,
        baseline: (() -> Void)? = nil,
        block: () -> Void
    ) -> MeasurementResult {
        // Warmup
        for _ in 0..<configuration.warmupIterations {
            block()
        }
        
        // Collect timing samples
        var timeSamples: [Double] = []
        var memorySamples: [(allocations: Int, bytes: Int)] = []
        
        for _ in 0..<configuration.statisticalRuns {
            let memStart = configuration.includeMemoryMetrics ? MemoryInfo.capture() : nil
            
            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<configuration.measurementIterations {
                block()
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            
            if let memStart = memStart {
                let memEnd = MemoryInfo.capture()
                let delta = MemoryInfo.delta(from: memStart, to: memEnd)
                memorySamples.append((delta.allocations, delta.bytesAllocated))
            }
            
            timeSamples.append(elapsed / Double(configuration.measurementIterations))
        }
        
        // Calculate statistics
        let sortedTimes = timeSamples.sorted()
        let meanTime = timeSamples.reduce(0, +) / Double(timeSamples.count)
        let medianTime = sortedTimes[sortedTimes.count / 2]
        let minTime = sortedTimes.first ?? 0
        let maxTime = sortedTimes.last ?? 0
        
        // Standard deviation
        let variance = timeSamples.map { pow($0 - meanTime, 2) }.reduce(0, +) / Double(timeSamples.count)
        let stdDeviation = sqrt(variance)
        
        // Percentiles
        let p95Index = Int(Double(sortedTimes.count) * 0.95)
        let p99Index = Int(Double(sortedTimes.count) * 0.99)
        let percentile95 = sortedTimes[min(p95Index, sortedTimes.count - 1)]
        let percentile99 = sortedTimes[min(p99Index, sortedTimes.count - 1)]
        
        // Memory statistics
        let allocationsPerOp: Double?
        let bytesPerOp: Double?
        if !memorySamples.isEmpty {
            allocationsPerOp = Double(memorySamples.map { $0.allocations }.reduce(0, +)) / 
                              Double(memorySamples.count * configuration.measurementIterations)
            bytesPerOp = Double(memorySamples.map { $0.bytes }.reduce(0, +)) / 
                        Double(memorySamples.count * configuration.measurementIterations)
        } else {
            allocationsPerOp = nil
            bytesPerOp = nil
        }
        
        // Baseline comparison
        let baselineComparison: Double?
        if let baseline = baseline {
            // Measure baseline
            let baselineResult = measure(
                name: "\(name)-baseline",
                category: category,
                vectorSize: vectorSize,
                configuration: Configuration(
                    warmupIterations: configuration.warmupIterations,
                    measurementIterations: configuration.measurementIterations,
                    statisticalRuns: min(3, configuration.statisticalRuns),
                    includeMemoryMetrics: false,
                    jsonOutputPath: nil
                ),
                baseline: nil,
                block: baseline
            )
            baselineComparison = baselineResult.meanTime / meanTime
        } else {
            baselineComparison = nil
        }
        
        // Create result
        return MeasurementResult(
            name: name,
            category: category,
            vectorSize: vectorSize,
            iterations: configuration.measurementIterations,
            meanTime: meanTime,
            medianTime: medianTime,
            minTime: minTime,
            maxTime: maxTime,
            stdDeviation: stdDeviation,
            percentile95: percentile95,
            percentile99: percentile99,
            opsPerSecond: 1.0 / meanTime,
            elementsPerSecond: Double(vectorSize) / meanTime,
            allocationsPerOp: allocationsPerOp,
            bytesAllocatedPerOp: bytesPerOp,
            baselineComparison: baselineComparison,
            timestamp: Date(),
            platform: getPlatformInfo(),
            swiftVersion: getSwiftVersion()
        )
    }
    
    /// Execute a batch of benchmarks
    public static func runBenchmarkSuite(
        name: String,
        configuration: Configuration = .default,
        benchmarks: [(name: String, category: String, vectorSize: Int, baseline: (() -> Void)?, block: () -> Void)]
    ) -> [MeasurementResult] {
        print("\(name)")
        print(String(repeating: "=", count: name.count))
        print("Platform: \(getPlatformInfo())")
        print("Swift: \(getSwiftVersion())")
        print("Configuration: \(configuration.measurementIterations) iterations, \(configuration.statisticalRuns) runs")
        print("")
        
        var results: [MeasurementResult] = []
        
        for benchmark in benchmarks {
            let result = measure(
                name: benchmark.name,
                category: benchmark.category,
                vectorSize: benchmark.vectorSize,
                configuration: configuration,
                baseline: benchmark.baseline,
                block: benchmark.block
            )
            
            print(result.formattedOutput)
            print("")
            
            results.append(result)
        }
        
        // Save JSON if requested
        if let jsonPath = configuration.jsonOutputPath {
            saveResults(results, to: jsonPath)
        }
        
        return results
    }
    
    // MARK: - JSON Export
    
    private static func saveResults(_ results: [MeasurementResult], to path: String) {
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            
            let data = try encoder.encode(results)
            try data.write(to: URL(fileURLWithPath: path))
            
            print("Results saved to: \(path)")
        } catch {
            print("Failed to save results: \(error)")
        }
    }
    
    // MARK: - Platform Info
    
    private static func getPlatformInfo() -> String {
        #if arch(arm64)
        return "Apple Silicon (ARM64)"
        #elseif arch(x86_64)
        return "Intel (x86_64)"
        #else
        return "Unknown"
        #endif
    }
    
    private static func getSwiftVersion() -> String {
        #if swift(>=6.0)
        return "6.0+"
        #elseif swift(>=5.10)
        return "5.10+"
        #elseif swift(>=5.9)
        return "5.9+"
        #else
        return "Unknown"
        #endif
    }
}

// MARK: - Baseline Implementations

/// Naive baseline implementations for performance comparison
public struct BaselineImplementations {
    
    /// Naive vector addition using Swift Arrays
    public static func add(_ a: [Float], _ b: [Float]) -> [Float] {
        guard a.count == b.count else { return [] }
        var result = [Float](repeating: 0, count: a.count)
        for i in 0..<a.count {
            result[i] = a[i] + b[i]
        }
        return result
    }
    
    /// Naive dot product using Swift Arrays
    public static func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }
        var sum: Float = 0
        for i in 0..<a.count {
            sum += a[i] * b[i]
        }
        return sum
    }
    
    /// Naive magnitude calculation
    public static func magnitude(_ a: [Float]) -> Float {
        var sum: Float = 0
        for value in a {
            sum += value * value
        }
        return sqrt(sum)
    }
    
    /// Naive normalization
    public static func normalize(_ a: [Float]) -> [Float] {
        let mag = magnitude(a)
        guard mag > 0 else { return a }
        return a.map { $0 / mag }
    }
    
    /// Naive euclidean distance
    public static func distance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }
        var sum: Float = 0
        for i in 0..<a.count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }
    
    /// Naive cosine similarity
    public static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        let dot = dotProduct(a, b)
        let magA = magnitude(a)
        let magB = magnitude(b)
        guard magA > 0 && magB > 0 else { return 0 }
        return dot / (magA * magB)
    }
}