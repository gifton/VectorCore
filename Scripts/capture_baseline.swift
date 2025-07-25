#!/usr/bin/env swift

// Script to capture performance baseline metrics
// Usage: swift Scripts/capture_baseline.swift [output_file]

import Foundation
import VectorCore

// MARK: - Benchmark Runner Implementation

struct SwiftBenchmarkRunner: BenchmarkRunner {
    private let runner = VectorCoreBenchmarkRunner()
    
    func runBenchmarks() async throws -> [BenchmarkResult] {
        // Use the real benchmark runner
        return try await runner.run()
    }
    
    func runBenchmark(named name: String) async throws -> BenchmarkResult? {
        return try await runner.run(benchmarkNamed: name)
    }
    
    func availableBenchmarks() -> [String] {
        return runner.availableBenchmarks
    }
}

// MARK: - Hardware Metrics Collection

func collectHardwareMetrics() -> HardwareMetrics {
    // Use the HardwareMetrics.collect() method from BenchmarkAdapter
    return HardwareMetrics.collect()
}

// BenchmarkAdapter handles all metric calculations now

// MARK: - Main Execution

@main
struct CaptureBaseline {
    static func main() async {
        let outputPath = CommandLine.arguments.count > 1 
            ? CommandLine.arguments[1] 
            : "baseline_metrics.json"
        
        // Print header
        print("VectorCore Performance Baseline Capture")
        print("======================================")
        print("Platform: \(PlatformInfo.current.os) \(PlatformInfo.current.osVersion)")
        print("Architecture: \(PlatformInfo.current.architecture)")
        print("CPU Cores: \(PlatformInfo.current.cpuCores)")
        print("Memory: \(String(format: "%.1f", PlatformInfo.current.memoryGB)) GB")
        print("")
        
        // Run benchmarks
        print("Running benchmarks...")
        let runner = SwiftBenchmarkRunner()
        
        do {
            let benchmarkResults = try await runner.runBenchmarks()
            print("✓ Completed \(benchmarkResults.count) benchmarks")
            
            // Collect hardware metrics
            print("Collecting hardware metrics...")
            let hardwareMetrics = collectHardwareMetrics()
            print("✓ Hardware metrics collected")
            
            // Use BenchmarkAdapter to create baseline from results
            let baseline = BenchmarkAdapter.createBaseline(
                from: benchmarkResults,
                platform: nil, // Will use PlatformInfo.current
                hardware: hardwareMetrics
            )
            
            // Save to file
            let encoder = JSONEncoder.baseline
            let data = try encoder.encode(baseline)
            try data.write(to: URL(fileURLWithPath: outputPath))
            
            print("\n✓ Baseline metrics saved to: \(outputPath)")
            
            // Print summary
            printSummary(baseline: baseline)
            
        } catch {
            print("❌ Error capturing baseline: \(error)")
            exit(1)
        }
    }
    
    static func printSummary(baseline: PerformanceBaseline) {
        print("\nPerformance Summary:")
        print("-------------------")
        print("Throughput Metrics:")
        print("  Vector Addition (768D): \(Int(baseline.throughput.vectorAddition)) ops/sec")
        print("  Dot Product (768D): \(Int(baseline.throughput.dotProduct)) ops/sec")
        print("  Euclidean Distance: \(Int(baseline.throughput.euclideanDistance)) ops/sec")
        print("  Cosine Similarity: \(Int(baseline.throughput.cosineSimilarity)) ops/sec")
        
        print("\nMemory Metrics:")
        print("  Bytes per Operation: \(baseline.memory.bytesPerOperation)")
        print("  Peak Memory: \(baseline.memory.peakMemoryUsage / 1_048_576) MB")
        
        print("\nParallelization:")
        print("  Speedup: \(String(format: "%.1fx", baseline.parallelization.parallelSpeedup))")
        print("  Efficiency: \(String(format: "%.0f%%", baseline.parallelization.scalingEfficiency * 100))")
        
        if let hardware = baseline.hardware {
            print("\nHardware Metrics:")
            print("  SIMD Utilization: \(String(format: "%.0f%%", hardware.simdUtilization * 100))")
            print("  L1 Cache Hit Rate: \(String(format: "%.0f%%", hardware.l1CacheHitRate * 100))")
            print("  Memory Bandwidth: \(String(format: "%.1f GB/s", hardware.memoryBandwidthGBps))")
            print("  CPU Utilization: \(String(format: "%.0f%%", hardware.cpuUtilization))")
        }
    }
}