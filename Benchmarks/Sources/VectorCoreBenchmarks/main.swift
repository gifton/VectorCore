// VectorCore Benchmark Runner
//
// Run performance benchmarks for VectorCore
//

import Foundation
import VectorCore

print("VectorCore Performance Benchmarks")
print("=================================")
print()

// Command line argument parsing
let arguments = CommandLine.arguments
let runAll = arguments.contains("--all") || arguments.count == 1
let runParallel = arguments.contains("--parallel") || runAll
let runComprehensive = arguments.contains("--comprehensive") || runAll
let runNewSuite = arguments.contains("--suite") || arguments.contains("--new")
let runQuick = arguments.contains("--quick")
let runWithBaseline = arguments.contains("--baseline")
let jsonOutput = arguments.contains("--json")
let runDimensionComparison = arguments.contains("--dimensions")
let runRegression = arguments.contains("--regression")
let updateBaseline = arguments.contains("--update-baseline")
let helpRequested = arguments.contains("--help") || arguments.contains("-h")

if helpRequested {
    print("""
    Usage: swift run VectorCoreBenchmarks [options]
    
    Options:
      --all                Run all benchmarks (default)
      --parallel           Run auto-parallelization benchmarks
      --comprehensive      Run comprehensive performance benchmarks
      --suite, --new       Run new comprehensive benchmark suite
      --dimensions         Run dimension comparison analysis
      --regression         Run regression detection
      --quick              Run quick benchmarks (fewer iterations)
      --baseline           Include baseline comparisons (slower but more informative)
      --json               Save results to JSON file
      --update-baseline    Update regression baseline after running
      --help, -h           Show this help message
    
    Examples:
      swift run VectorCoreBenchmarks --suite --baseline --json
      swift run VectorCoreBenchmarks --quick --parallel
      swift run VectorCoreBenchmarks --comprehensive
      swift run VectorCoreBenchmarks --dimensions
      swift run VectorCoreBenchmarks --regression --update-baseline
    """)
    exit(0)
}

// Run Regression Detection
if runRegression {
    print("\nRunning Regression Detection...")
    print("-------------------------------")
    
    let benchmarkConfig: BenchmarkFramework.Configuration = runQuick ? .quick : .default
    RegressionDetectionBenchmark.run(benchmarkConfig: benchmarkConfig)
    
    if !runAll {
        exit(0)
    }
}

// Run Dimension Comparison
if runDimensionComparison {
    print("\nRunning Dimension Comparison Analysis...")
    print("---------------------------------------")
    
    let configuration: BenchmarkFramework.Configuration = runQuick ? .quick : .default
    DimensionComparisonBenchmark.run(configuration: configuration)
    
    if !runAll {
        exit(0)
    }
}

// Run New Comprehensive Benchmark Suite
if runNewSuite {
    print("\nRunning New Comprehensive Benchmark Suite...")
    print("-------------------------------------------")
    
    // Configure based on command line flags
    let configuration: BenchmarkFramework.Configuration
    if runQuick {
        configuration = .quick
    } else if jsonOutput {
        configuration = BenchmarkFramework.Configuration(
            warmupIterations: 10,
            measurementIterations: 1000,
            statisticalRuns: 5,
            includeMemoryMetrics: runWithBaseline,
            jsonOutputPath: "vectorcore_benchmark_results.json"
        )
    } else {
        configuration = .default
    }
    
    ComprehensiveBenchmarkSuite.run(configuration: configuration)
    
    if !runAll {
        exit(0)
    }
}

// Run Original Comprehensive Benchmarks
if runComprehensive && !runParallel && !runNewSuite {
    ComprehensiveBenchmark.runBenchmarks()
}

// Run Auto-Parallelization Benchmarks
if runParallel {
    print("\nRunning Auto-Parallelization Benchmarks...")
    print("----------------------------------------")
    
    // Use async context for auto-parallelization benchmarks
    Task {
        // Use appropriate configuration based on flags
        let configuration: AutoParallelizationBenchmark.Configuration
        if runQuick {
            configuration = AutoParallelizationBenchmark.Configuration.quick
        } else if runComprehensive {
            configuration = AutoParallelizationBenchmark.Configuration.comprehensive
        } else {
            configuration = AutoParallelizationBenchmark.Configuration.default
        }
            
        let results = await AutoParallelizationBenchmark.runBenchmarks(configuration: configuration)
        
        print("\nBenchmark Summary:")
        print("==================")
        print("Total benchmarks run: \(results.count)")
        
        // Run comprehensive benchmarks after parallel if --all
        if runComprehensive && runAll && !runNewSuite {
            print("\n")
            ComprehensiveBenchmark.runBenchmarks()
        }
        
        exit(0)
    }
    
    // Keep the main thread alive for async work
    RunLoop.main.run()
} else {
    print("\nBenchmarks completed.")
}