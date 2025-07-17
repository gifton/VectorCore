#!/usr/bin/swift

// VectorCore: Performance Regression Runner
//
// Command-line tool for running performance regression tests
//

import Foundation
import VectorCore

// MARK: - Command Line Arguments

struct Arguments {
    let iterations: Int
    let baselinePath: String?
    let outputPath: String?
    let outputFormat: RegressionTestConfig.OutputFormat
    let failOnRegression: Bool
    let saveBaseline: Bool
    let acceptableVariance: Double
    
    static func parse() -> Arguments {
        let args = CommandLine.arguments
        
        var iterations = 1000
        var baselinePath: String? = nil
        var outputPath: String? = nil
        var outputFormat = RegressionTestConfig.OutputFormat.plain
        var failOnRegression = true
        var saveBaseline = false
        var acceptableVariance = 0.1
        
        var i = 1
        while i < args.count {
            switch args[i] {
            case "--iterations", "-i":
                if i + 1 < args.count {
                    iterations = Int(args[i + 1]) ?? 1000
                    i += 1
                }
            case "--baseline", "-b":
                if i + 1 < args.count {
                    baselinePath = args[i + 1]
                    i += 1
                }
            case "--output", "-o":
                if i + 1 < args.count {
                    outputPath = args[i + 1]
                    i += 1
                }
            case "--format", "-f":
                if i + 1 < args.count {
                    switch args[i + 1] {
                    case "json":
                        outputFormat = .json
                    case "markdown", "md":
                        outputFormat = .markdown
                    default:
                        outputFormat = .plain
                    }
                    i += 1
                }
            case "--no-fail":
                failOnRegression = false
            case "--save-baseline":
                saveBaseline = true
            case "--variance", "-v":
                if i + 1 < args.count {
                    acceptableVariance = Double(args[i + 1]) ?? 0.1
                    i += 1
                }
            case "--help", "-h":
                printHelp()
                exit(0)
            default:
                break
            }
            i += 1
        }
        
        return Arguments(
            iterations: iterations,
            baselinePath: baselinePath,
            outputPath: outputPath,
            outputFormat: outputFormat,
            failOnRegression: failOnRegression,
            saveBaseline: saveBaseline,
            acceptableVariance: acceptableVariance
        )
    }
    
    static func printHelp() {
        print("""
        VectorCore Performance Regression Runner
        
        Usage: PerformanceRegressionRunner [options]
        
        Options:
          -i, --iterations <n>      Number of iterations per test (default: 1000)
          -b, --baseline <path>     Path to baseline JSON file
          -o, --output <path>       Path to save results
          -f, --format <format>     Output format: plain, json, markdown (default: plain)
          -v, --variance <n>        Acceptable variance percentage (default: 0.1 = 10%)
          --no-fail                 Don't fail on regression
          --save-baseline           Save current results as baseline
          -h, --help                Show this help message
        
        Examples:
          # Run tests and compare against baseline
          PerformanceRegressionRunner -b baseline.json
          
          # Create new baseline
          PerformanceRegressionRunner --save-baseline -o baseline.json -f json
          
          # Run with custom iterations and markdown output
          PerformanceRegressionRunner -i 5000 -f markdown -o results.md
        """)
    }
}

// MARK: - Main Execution

let args = Arguments.parse()

// Configure suite
let config = RegressionTestConfig(
    iterations: args.iterations,
    acceptableVariance: args.acceptableVariance,
    failOnRegression: args.failOnRegression,
    warmupIterations: max(10, args.iterations / 10),
    outputFormat: args.outputFormat
)

let suite = PerformanceRegressionSuite(config: config)

print("🚀 VectorCore Performance Regression Tests")
print("=========================================")
print("Configuration:")
print("  - Iterations: \(args.iterations)")
print("  - Acceptable Variance: \(args.acceptableVariance * 100)%")
print("  - Output Format: \(args.outputFormat)")
if let baseline = args.baselinePath {
    print("  - Baseline: \(baseline)")
}
print("")

// Run tests
print("Running performance tests...")
let startTime = Date()

do {
    let (results, regressions) = try suite.runAndCheckRegressions(
        baselineURL: args.baselinePath.map { URL(fileURLWithPath: $0) }
    )
    
    let elapsed = Date().timeIntervalSince(startTime)
    print("✅ Completed \(results.count) tests in \(String(format: "%.2f", elapsed)) seconds\n")
    
    // Output results
    let formattedResults = suite.formatResults(results)
    
    if let outputPath = args.outputPath {
        try formattedResults.write(toFile: outputPath, atomically: true, encoding: .utf8)
        print("Results saved to: \(outputPath)")
    } else {
        print(formattedResults)
    }
    
    // Show regression analysis if baseline was provided
    if let regressions = regressions {
        print("\n")
        print(suite.formatRegressionResults(regressions))
        
        let regressionCount = regressions.filter { $0.isRegression }.count
        if regressionCount > 0 && args.failOnRegression {
            print("\n❌ Build failed due to performance regressions")
            exit(1)
        }
    }
    
    // Save baseline if requested
    if args.saveBaseline {
        let baseline = suite.createBaseline()
        let baselineURL: URL
        
        if let outputPath = args.outputPath {
            baselineURL = URL(fileURLWithPath: outputPath)
        } else {
            baselineURL = URL(fileURLWithPath: "baseline.json")
        }
        
        try baseline.save(to: baselineURL)
        print("\n📊 Baseline saved to: \(baselineURL.path)")
    }
    
} catch RegressionError.regressionsDetected(let regressions) {
    print("\n❌ Performance regressions detected!")
    print(suite.formatRegressionResults(regressions))
    exit(1)
} catch {
    print("\n❌ Error: \(error)")
    exit(1)
}

print("\n✨ Performance tests completed successfully!")