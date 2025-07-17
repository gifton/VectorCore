import Foundation
import VectorCore

/// Regression detection benchmark for VectorCore
///
/// This benchmark:
/// - Loads baseline performance data from previous runs
/// - Compares current performance against baselines
/// - Detects performance regressions with configurable thresholds
/// - Generates detailed regression reports
public struct RegressionDetectionBenchmark {
    
    // MARK: - Configuration
    
    public struct Configuration {
        public let baselinePath: String
        public let regressionThreshold: Double  // Percentage slowdown to flag as regression
        public let improvementThreshold: Double  // Percentage speedup to flag as improvement
        public let criticalOperations: Set<String>  // Operations that should never regress
        
        public static let `default` = Configuration(
            baselinePath: "vectorcore_baseline.json",
            regressionThreshold: 0.05,  // 5% regression threshold
            improvementThreshold: 0.10,  // 10% improvement threshold
            criticalOperations: ["DotProduct", "Distance", "CosineSimilarity"]
        )
    }
    
    // MARK: - Regression Result
    
    public struct RegressionResult {
        public let operation: String
        public let dimension: Int
        public let baselineOpsPerSecond: Double
        public let currentOpsPerSecond: Double
        public let percentageChange: Double
        public let status: Status
        
        public enum Status {
            case regression(severity: Severity)
            case improvement
            case stable
            
            public enum Severity {
                case critical  // Critical operation regressed
                case major     // >10% regression
                case minor     // 5-10% regression
            }
        }
        
        public var description: String {
            let changeStr = percentageChange >= 0 ? 
                "+\(String(format: "%.1f", percentageChange * 100))%" : 
                "\(String(format: "%.1f", percentageChange * 100))%"
            
            let statusStr: String
            switch status {
            case .regression(let severity):
                switch severity {
                case .critical: statusStr = "⚠️  CRITICAL REGRESSION"
                case .major: statusStr = "❌ Major Regression"
                case .minor: statusStr = "⚠️  Minor Regression"
                }
            case .improvement:
                statusStr = "✅ Improvement"
            case .stable:
                statusStr = "✓ Stable"
            }
            
            return "\(operation) [dim=\(dimension)]: \(formatOps(currentOpsPerSecond)) ops/sec (\(changeStr)) - \(statusStr)"
        }
        
        private func formatOps(_ ops: Double) -> String {
            if ops > 1e6 {
                return String(format: "%.2fM", ops / 1e6)
            } else if ops > 1e3 {
                return String(format: "%.2fK", ops / 1e3)
            } else {
                return String(format: "%.0f", ops)
            }
        }
    }
    
    // MARK: - Main Entry Point
    
    public static func run(
        configuration: Configuration = .default,
        benchmarkConfig: BenchmarkFramework.Configuration = .default
    ) {
        print("\nVectorCore Regression Detection")
        print("==============================")
        
        // Load baseline data
        let baseline = loadBaseline(from: configuration.baselinePath)
        
        if baseline.isEmpty {
            print("No baseline found. Running benchmarks to establish baseline...")
            establishBaseline(
                path: configuration.baselinePath,
                benchmarkConfig: benchmarkConfig
            )
            return
        }
        
        print("Baseline loaded with \(baseline.count) measurements")
        print("Regression threshold: \(Int(configuration.regressionThreshold * 100))%")
        print("Critical operations: \(configuration.criticalOperations.joined(separator: ", "))")
        print("")
        
        // Run current benchmarks
        let currentResults = runCurrentBenchmarks(benchmarkConfig: benchmarkConfig)
        
        // Compare against baseline
        let regressions = detectRegressions(
            baseline: baseline,
            current: currentResults,
            configuration: configuration
        )
        
        // Generate report
        generateRegressionReport(regressions: regressions, configuration: configuration)
        
        // Update baseline if requested
        if CommandLine.arguments.contains("--update-baseline") {
            print("\nUpdating baseline...")
            saveBaseline(currentResults, to: configuration.baselinePath)
            print("Baseline updated successfully")
        }
    }
    
    // MARK: - Baseline Management
    
    private static func loadBaseline(from path: String) -> [BenchmarkFramework.MeasurementResult] {
        guard FileManager.default.fileExists(atPath: path) else {
            return []
        }
        
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            return try decoder.decode([BenchmarkFramework.MeasurementResult].self, from: data)
        } catch {
            print("Error loading baseline: \(error)")
            return []
        }
    }
    
    private static func saveBaseline(_ results: [BenchmarkFramework.MeasurementResult], to path: String) {
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(results)
            try data.write(to: URL(fileURLWithPath: path))
        } catch {
            print("Error saving baseline: \(error)")
        }
    }
    
    private static func establishBaseline(
        path: String,
        benchmarkConfig: BenchmarkFramework.Configuration
    ) {
        let results = runCurrentBenchmarks(benchmarkConfig: benchmarkConfig)
        saveBaseline(results, to: path)
        print("\nBaseline established with \(results.count) measurements")
        print("Run again to detect regressions")
    }
    
    // MARK: - Benchmark Execution
    
    private static func runCurrentBenchmarks(
        benchmarkConfig: BenchmarkFramework.Configuration
    ) -> [BenchmarkFramework.MeasurementResult] {
        print("Running performance benchmarks...")
        
        var results: [BenchmarkFramework.MeasurementResult] = []
        
        // Test dimensions
        let dimensions = [64, 256, 768]
        
        for dim in dimensions {
            // Generate test data
            let array1 = (0..<dim).map { sin(Float($0) * 0.1) }
            let array2 = (0..<dim).map { cos(Float($0) * 0.1) }
            
            // Core operations to benchmark
            let operations: [(name: String, block: () -> Void)] = [
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
            
            for (opName, block) in operations {
                let result = BenchmarkFramework.measure(
                    name: "\(opName)_\(dim)",
                    category: "Regression",
                    vectorSize: dim,
                    configuration: benchmarkConfig,
                    block: block
                )
                results.append(result)
            }
        }
        
        return results
    }
    
    private static func performOperation(
        dimension: Int,
        array1: [Float],
        array2: [Float],
        operation: (any ExtendedVectorProtocol, any ExtendedVectorProtocol) -> Void
    ) {
        switch dimension {
        case 64:
            let v1 = Vector<Dim64>(array1)
            let v2 = Vector<Dim64>(array2)
            operation(v1, v2)
        case 256:
            let v1 = Vector256(array1)
            let v2 = Vector256(array2)
            operation(v1, v2)
        case 768:
            let v1 = Vector768(array1)
            let v2 = Vector768(array2)
            operation(v1, v2)
        default:
            let v1 = DynamicVector(array1)
            let v2 = DynamicVector(array2)
            operation(v1, v2)
        }
    }
    
    // MARK: - Regression Detection
    
    private static func detectRegressions(
        baseline: [BenchmarkFramework.MeasurementResult],
        current: [BenchmarkFramework.MeasurementResult],
        configuration: Configuration
    ) -> [RegressionResult] {
        var regressions: [RegressionResult] = []
        
        // Create lookup for baseline results
        let baselineDict = Dictionary(
            uniqueKeysWithValues: baseline.map { ($0.name, $0) }
        )
        
        for currentResult in current {
            guard let baselineResult = baselineDict[currentResult.name] else {
                print("Warning: No baseline for \(currentResult.name)")
                continue
            }
            
            let percentageChange = (currentResult.opsPerSecond - baselineResult.opsPerSecond) / baselineResult.opsPerSecond
            
            // Extract operation name from result name (format: "Operation_Dimension")
            let components = currentResult.name.split(separator: "_")
            let operationName = String(components[0])
            
            // Determine status
            let status: RegressionResult.Status
            if percentageChange < -configuration.regressionThreshold {
                // Performance regression detected
                let isCritical = configuration.criticalOperations.contains(operationName)
                let severity: RegressionResult.Status.Severity
                if isCritical {
                    severity = .critical
                } else if percentageChange < -0.10 {
                    severity = .major
                } else {
                    severity = .minor
                }
                status = .regression(severity: severity)
            } else if percentageChange > configuration.improvementThreshold {
                status = .improvement
            } else {
                status = .stable
            }
            
            let result = RegressionResult(
                operation: operationName,
                dimension: currentResult.vectorSize,
                baselineOpsPerSecond: baselineResult.opsPerSecond,
                currentOpsPerSecond: currentResult.opsPerSecond,
                percentageChange: percentageChange,
                status: status
            )
            
            regressions.append(result)
        }
        
        return regressions.sorted { r1, r2 in
            // Sort by severity, then by percentage change
            switch (r1.status, r2.status) {
            case (.regression(let s1), .regression(let s2)):
                if s1.severity != s2.severity {
                    return s1.severity < s2.severity  // Critical first
                }
                return r1.percentageChange < r2.percentageChange
            case (.regression, _):
                return true
            case (_, .regression):
                return false
            default:
                return r1.percentageChange < r2.percentageChange
            }
        }
    }
    
    // MARK: - Report Generation
    
    private static func generateRegressionReport(
        regressions: [RegressionResult],
        configuration: Configuration
    ) {
        print("\nRegression Detection Results")
        print("===========================\n")
        
        // Count by status
        var regressionCount = 0
        var criticalCount = 0
        var improvementCount = 0
        var stableCount = 0
        
        for result in regressions {
            print(result.description)
            
            switch result.status {
            case .regression(let severity):
                regressionCount += 1
                if case .critical = severity {
                    criticalCount += 1
                }
            case .improvement:
                improvementCount += 1
            case .stable:
                stableCount += 1
            }
        }
        
        // Summary
        print("\nSummary")
        print("-------")
        print("Total operations tested: \(regressions.count)")
        print("Regressions found: \(regressionCount) (\(criticalCount) critical)")
        print("Improvements: \(improvementCount)")
        print("Stable: \(stableCount)")
        
        // Exit with error code if critical regressions found
        if criticalCount > 0 {
            print("\n❌ CRITICAL REGRESSIONS DETECTED - Build should fail")
            exit(1)
        } else if regressionCount > 0 {
            print("\n⚠️  Performance regressions detected")
        } else if improvementCount > 0 {
            print("\n✅ Performance improvements detected!")
        } else {
            print("\n✓ Performance is stable")
        }
    }
}