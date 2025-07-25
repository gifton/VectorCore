#!/usr/bin/env swift

// Script to compare current performance against baseline
// Usage: swift Scripts/compare_baseline.swift baseline.json current.json [--threshold 0.05]

import Foundation
import VectorCore

// MARK: - Comparison Logic

struct ComparisonResult {
    let metric: String
    let baselineValue: Double
    let currentValue: Double
    let percentChange: Double
    let isRegression: Bool
    
    var changeDescription: String {
        let sign = percentChange >= 0 ? "+" : ""
        return "\(sign)\(String(format: "%.1f", percentChange))%"
    }
    
    var status: String {
        if abs(percentChange) < 1.0 {
            return "✓ No Change"
        } else if isRegression {
            return "⚠️  REGRESSION"
        } else if percentChange > 0 {
            return "✅ Improvement"
        } else {
            return "✓ Acceptable"
        }
    }
}

// MARK: - Enhanced Comparison with Hardware Metrics

func compareMetrics(baseline: PerformanceBaseline, current: PerformanceBaseline, threshold: Double) -> [ComparisonResult] {
    var results: [ComparisonResult] = []
    
    // Compare throughput metrics
    results.append(contentsOf: compareThroughputMetrics(baseline, current, threshold))
    
    // Compare memory metrics
    results.append(contentsOf: compareMemoryMetrics(baseline, current, threshold))
    
    // Compare parallelization metrics
    results.append(contentsOf: compareParallelizationMetrics(baseline, current, threshold))
    
    // Compare hardware metrics if available
    if let baselineHW = baseline.hardware, let currentHW = current.hardware {
        results.append(contentsOf: compareHardwareMetrics(baselineHW, currentHW, threshold))
    }
    
    return results
}

func compareThroughputMetrics(_ baseline: PerformanceBaseline, _ current: PerformanceBaseline, _ threshold: Double) -> [ComparisonResult] {
    var results: [ComparisonResult] = []
    
    let metrics: [(String, KeyPath<ThroughputMetrics, Double>)] = [
        ("Vector Addition (768D)", \.vectorAddition),
        ("Vector Scalar Mult", \.vectorScalarMultiplication),
        ("Vector Elementwise Mult", \.vectorElementwiseMultiplication),
        ("Dot Product (768D)", \.dotProduct),
        ("Euclidean Distance", \.euclideanDistance),
        ("Cosine Similarity", \.cosineSimilarity),
        ("Manhattan Distance", \.manhattanDistance),
        ("Normalization", \.normalization)
    ]
    
    for (name, keyPath) in metrics {
        results.append(compareMetric(
            name,
            baseline: baseline.throughput[keyPath: keyPath],
            current: current.throughput[keyPath: keyPath],
            threshold: threshold,
            higherIsBetter: true
        ))
    }
    
    return results
}

func compareMemoryMetrics(_ baseline: PerformanceBaseline, _ current: PerformanceBaseline, _ threshold: Double) -> [ComparisonResult] {
    var results: [ComparisonResult] = []
    
    results.append(compareMetric(
        "Memory per Operation",
        baseline: Double(baseline.memory.bytesPerOperation),
        current: Double(current.memory.bytesPerOperation),
        threshold: threshold,
        higherIsBetter: false
    ))
    
    results.append(compareMetric(
        "Peak Memory Usage",
        baseline: Double(baseline.memory.peakMemoryUsage),
        current: Double(current.memory.peakMemoryUsage),
        threshold: threshold,
        higherIsBetter: false
    ))
    
    results.append(compareMetric(
        "Allocation Rate",
        baseline: baseline.memory.allocationRate,
        current: current.memory.allocationRate,
        threshold: threshold,
        higherIsBetter: false
    ))
    
    return results
}

func compareParallelizationMetrics(_ baseline: PerformanceBaseline, _ current: PerformanceBaseline, _ threshold: Double) -> [ComparisonResult] {
    var results: [ComparisonResult] = []
    
    results.append(compareMetric(
        "Parallel Speedup",
        baseline: baseline.parallelization.parallelSpeedup,
        current: current.parallelization.parallelSpeedup,
        threshold: threshold,
        higherIsBetter: true
    ))
    
    results.append(compareMetric(
        "Scaling Efficiency",
        baseline: baseline.parallelization.scalingEfficiency,
        current: current.parallelization.scalingEfficiency,
        threshold: threshold,
        higherIsBetter: true
    ))
    
    results.append(compareMetric(
        "Thread Utilization",
        baseline: baseline.parallelization.threadUtilization,
        current: current.parallelization.threadUtilization,
        threshold: threshold,
        higherIsBetter: true
    ))
    
    return results
}

func compareHardwareMetrics(_ baseline: HardwareMetrics, _ current: HardwareMetrics, _ threshold: Double) -> [ComparisonResult] {
    var results: [ComparisonResult] = []
    
    results.append(compareMetric(
        "SIMD Utilization",
        baseline: baseline.simdUtilization,
        current: current.simdUtilization,
        threshold: threshold,
        higherIsBetter: true
    ))
    
    results.append(compareMetric(
        "L1 Cache Hit Rate",
        baseline: baseline.l1CacheHitRate,
        current: current.l1CacheHitRate,
        threshold: threshold,
        higherIsBetter: true
    ))
    
    results.append(compareMetric(
        "Memory Bandwidth",
        baseline: baseline.memoryBandwidthGBps,
        current: current.memoryBandwidthGBps,
        threshold: threshold,
        higherIsBetter: true
    ))
    
    results.append(compareMetric(
        "CPU Utilization",
        baseline: baseline.cpuUtilization,
        current: current.cpuUtilization,
        threshold: threshold,
        higherIsBetter: true
    ))
    
    return results
}

func compareMetric(_ name: String, baseline: Double, current: Double, threshold: Double, higherIsBetter: Bool) -> ComparisonResult {
    let percentChange: Double
    
    if baseline == 0 {
        percentChange = current == 0 ? 0 : 100.0
    } else {
        percentChange = ((current - baseline) / baseline) * 100.0
    }
    
    let isRegression: Bool
    if higherIsBetter {
        // For metrics where higher is better (throughput), regression is negative change beyond threshold
        isRegression = percentChange < -threshold * 100
    } else {
        // For metrics where lower is better (memory), regression is positive change beyond threshold
        isRegression = percentChange > threshold * 100
    }
    
    return ComparisonResult(
        metric: name,
        baselineValue: baseline,
        currentValue: current,
        percentChange: percentChange,
        isRegression: isRegression
    )
}

// MARK: - Enhanced JSON Loading with Streaming Support

func loadBaseline(from path: String) throws -> PerformanceBaseline {
    let url = URL(fileURLWithPath: path)
    
    // For large files, use memory-mapped data
    let data: Data
    if let fileSize = try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int,
       fileSize > 10_000_000 { // 10MB threshold
        data = try Data(contentsOf: url, options: .mappedIfSafe)
    } else {
        data = try Data(contentsOf: url)
    }
    
    let decoder = JSONDecoder.baseline
    return try decoder.decode(PerformanceBaseline.self, from: data)
}

// MARK: - Main Execution

@main
struct CompareBaseline {
    static func main() async {
        // Parse command line arguments
        guard CommandLine.arguments.count >= 3 else {
            print("Usage: swift compare_baseline.swift baseline.json current.json [--threshold 0.05]")
            exit(1)
        }
        
        let baselinePath = CommandLine.arguments[1]
        let currentPath = CommandLine.arguments[2]
        var threshold = 0.05 // Default 5% regression threshold
        
        // Parse optional threshold
        if CommandLine.arguments.count >= 5 && CommandLine.arguments[3] == "--threshold" {
            if let thresholdValue = Double(CommandLine.arguments[4]) {
                threshold = thresholdValue
            }
        }
        
        do {
            // Load baselines
            let baseline = try loadBaseline(from: baselinePath)
            let current = try loadBaseline(from: currentPath)
            
            // Print header
            print("VectorCore Performance Comparison")
            print("=================================")
            print("Baseline: \(baselinePath) (\(baseline.swiftVersion))")
            print("Current:  \(currentPath) (\(current.swiftVersion))")
            print("Regression Threshold: \(Int(threshold * 100))%")
            print("")
            
            // Compare metrics
            let results = compareMetrics(baseline: baseline, current: current, threshold: threshold)
            
            // Print results table
            printResultsTable(results)
            
            // Analyze and report
            let regressions = results.filter { $0.isRegression }
            let improvements = results.filter { $0.percentChange > 1.0 && !$0.isRegression }
            
            print("")
            
            if !regressions.isEmpty {
                print("❌ PERFORMANCE REGRESSION DETECTED")
                print("   \(regressions.count) metric(s) have regressed beyond the \(Int(threshold * 100))% threshold:")
                for regression in regressions {
                    print("   - \(regression.metric): \(regression.changeDescription)")
                }
                exit(1)
            } else {
                print("✅ All metrics within acceptable range")
                
                if !improvements.isEmpty {
                    print("   \(improvements.count) metric(s) showed improvement:")
                    for improvement in improvements.prefix(3) {
                        print("   - \(improvement.metric): \(improvement.changeDescription)")
                    }
                    if improvements.count > 3 {
                        print("   ... and \(improvements.count - 3) more")
                    }
                }
            }
            
            // Save comparison output for CI
            if let outputPath = ProcessInfo.processInfo.environment["COMPARISON_OUTPUT"] {
                var output = ""
                output += "VectorCore Performance Comparison\n"
                output += "=================================\n"
                output += "Baseline: \(baselinePath) (\(baseline.swiftVersion))\n"
                output += "Current:  \(currentPath) (\(current.swiftVersion))\n"
                output += "Regression Threshold: \(Int(threshold * 100))%\n\n"
                
                for result in results {
                    output += String(format: "%-30s %15.0f → %15.0f %10s %s\n",
                                   result.metric,
                                   result.baselineValue,
                                   result.currentValue,
                                   result.changeDescription,
                                   result.status)
                }
                
                if !regressions.isEmpty {
                    output += "\n❌ PERFORMANCE REGRESSION DETECTED\n"
                    output += "   \(regressions.count) metric(s) have regressed beyond the \(Int(threshold * 100))% threshold\n"
                } else {
                    output += "\n✅ All metrics within acceptable range\n"
                    if !improvements.isEmpty {
                        output += "   \(improvements.count) metric(s) showed improvement\n"
                    }
                }
                
                try? output.write(toFile: outputPath, atomically: true, encoding: .utf8)
            }
            
        } catch {
            print("❌ Error: \(error)")
            exit(1)
        }
    }
    
    static func printResultsTable(_ results: [ComparisonResult]) {
        // Group results by category
        let categories = [
            "Throughput": results.filter { r in
                ["Addition", "Multiplication", "Dot", "Distance", "Similarity", "Normalization"]
                    .contains(where: { r.metric.contains($0) })
            },
            "Memory": results.filter { $0.metric.contains("Memory") || $0.metric.contains("Allocation") },
            "Parallelization": results.filter { $0.metric.contains("Parallel") || $0.metric.contains("Thread") || $0.metric.contains("Scaling") },
            "Hardware": results.filter { $0.metric.contains("SIMD") || $0.metric.contains("Cache") || $0.metric.contains("CPU") || $0.metric.contains("Bandwidth") }
        ]
        
        for (category, categoryResults) in categories where !categoryResults.isEmpty {
            print("\n\(category) Metrics:")
            print(String(format: "%-30s %15s %15s %10s %s", "Metric", "Baseline", "Current", "Change", "Status"))
            print(String(repeating: "-", count: 90))
            
            for result in categoryResults {
                let baselineStr = formatValue(result.baselineValue, category: category)
                let currentStr = formatValue(result.currentValue, category: category)
                
                print(String(format: "%-30s %15s %15s %10s %s",
                             result.metric,
                             baselineStr,
                             currentStr,
                             result.changeDescription,
                             result.status))
            }
        }
    }
    
    static func formatValue(_ value: Double, category: String) -> String {
        switch category {
        case "Memory":
            if value > 1_000_000 {
                return String(format: "%.1f MB", value / 1_048_576)
            } else if value > 1000 {
                return String(format: "%.1f KB", value / 1024)
            } else {
                return String(format: "%.0f B", value)
            }
        case "Hardware":
            if value < 1 {
                return String(format: "%.0f%%", value * 100)
            } else {
                return String(format: "%.1f", value)
            }
        default:
            return String(format: "%.0f", value)
        }
    }
}