// VectorCore Benchmark Runner
//
// Run performance benchmarks for VectorCore
//

import Foundation
@testable import VectorCore

print("VectorCore Performance Benchmarks")
print("=================================")
print()

// Command line argument parsing
let arguments = CommandLine.arguments
let runAll = arguments.contains("--all") || arguments.count == 1
let runSIMD = arguments.contains("--simd") || runAll
let runParallel = arguments.contains("--parallel") || runAll
let helpRequested = arguments.contains("--help") || arguments.contains("-h")

if helpRequested {
    print("""
    Usage: swift run VectorCoreBenchmarks [options]
    
    Options:
      --all        Run all benchmarks (default)
      --simd       Run SIMD storage benchmarks
      --parallel   Run auto-parallelization benchmarks
      --help, -h   Show this help message
    """)
    exit(0)
}

// Run SIMD Storage Benchmarks
if runSIMD {
    print("Running SIMD Storage Benchmarks...")
    print("---------------------------------")
    
    let simdBenchmark = SIMDStorageBenchmark()
    
    // Run buffer access benchmarks
    print("\nBuffer Access Operations:")
    let bufferResults = simdBenchmark.benchmarkBufferAccess()
    for result in bufferResults {
        print(result.formattedReport)
    }
    
    // Run dot product benchmarks
    print("\nDot Product Operations:")
    let dotResults = simdBenchmark.benchmarkDotProduct()
    for result in dotResults {
        print(result.formattedReport)
    }
    
    // Run vector addition benchmarks
    print("\nVector Addition Operations:")
    let addResults = simdBenchmark.benchmarkVectorAddition()
    for result in addResults {
        print(result.formattedReport)
    }
    
    // Run comparison benchmarks
    print("\nStorage Comparison:")
    let comparison = simdBenchmark.runComparisonBenchmark()
    print(comparison)
}

// Run Auto-Parallelization Benchmarks
if runParallel {
    print("\nRunning Auto-Parallelization Benchmarks...")
    print("----------------------------------------")
    
    // Use async context for auto-parallelization benchmarks
    Task {
        let results = await AutoParallelizationBenchmark.runBenchmarks()
        
        print("\nBenchmark Summary:")
        print("==================")
        print("Total benchmarks run: \(results.count)")
        
        exit(0)
    }
    
    // Keep the main thread alive for async work
    RunLoop.main.run()
}

print("\nBenchmarks completed.")