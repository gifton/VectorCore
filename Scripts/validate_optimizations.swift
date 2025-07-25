#!/usr/bin/env swift

// Script to validate that optimizations are properly applied
// Usage: swift Scripts/validate_optimizations.swift

import Foundation

// MARK: - Validation Results

struct ValidationResult {
    let check: String
    let passed: Bool
    let details: String
    
    var icon: String {
        passed ? "✅" : "❌"
    }
}

// MARK: - Validation Functions

func validateBuildSettings() -> [ValidationResult] {
    var results: [ValidationResult] = []
    
    // Check if release build exists
    let buildDir = ".build/release"
    let releaseExists = FileManager.default.fileExists(atPath: buildDir)
    results.append(ValidationResult(
        check: "Release build exists",
        passed: releaseExists,
        details: releaseExists ? "Found at \(buildDir)" : "Run build_optimized.sh first"
    ))
    
    // Check library size (optimized builds should be smaller)
    if let libPath = findLibrary() {
        let attributes = try? FileManager.default.attributesOfItem(atPath: libPath)
        let size = attributes?[.size] as? Int ?? 0
        let sizeKB = size / 1024
        
        // Optimized builds should generally be smaller due to dead code elimination
        let sizeOptimal = sizeKB < 5000 // Less than 5MB is good
        results.append(ValidationResult(
            check: "Library size optimization",
            passed: sizeOptimal,
            details: "Library size: \(sizeKB)KB"
        ))
    }
    
    // Check Swift module exists
    let moduleExists = findSwiftModule() != nil
    results.append(ValidationResult(
        check: "Swift module generated",
        passed: moduleExists,
        details: moduleExists ? "Module interface available" : "Module not found"
    ))
    
    return results
}

func validateCompilerFlags() -> [ValidationResult] {
    var results: [ValidationResult] = []
    
    // Check Package.swift for optimization flags
    let packageContent = try? String(contentsOfFile: "Package.swift")
    let hasWMO = packageContent?.contains("whole-module-optimization") ?? false
    let hasCMO = packageContent?.contains("cross-module-optimization") ?? false
    let hasLTO = packageContent?.contains("-lto") ?? false
    
    // Check for problematic flags
    let hasDisabledARC = packageContent?.contains("disable-arc-opts") ?? false
    let hasDisabledOSSA = packageContent?.contains("disable-ossa-opts") ?? false
    let hasUncheckedExclusivity = packageContent?.contains("enforce-exclusivity=unchecked") ?? false
    
    results.append(ValidationResult(
        check: "Whole Module Optimization",
        passed: hasWMO,
        details: hasWMO ? "Enabled in Package.swift" : "Not configured"
    ))
    
    results.append(ValidationResult(
        check: "Cross Module Optimization",
        passed: hasCMO,
        details: hasCMO ? "Enabled in Package.swift" : "Not configured"
    ))
    
    results.append(ValidationResult(
        check: "Link Time Optimization",
        passed: hasLTO,
        details: hasLTO ? "Enabled in Package.swift" : "Not configured"
    ))
    
    // Check for problematic flags
    results.append(ValidationResult(
        check: "No disabled optimizations",
        passed: !hasDisabledARC && !hasDisabledOSSA,
        details: hasDisabledARC || hasDisabledOSSA ? "CRITICAL: Optimization-disabling flags found!" : "No problematic flags"
    ))
    
    results.append(ValidationResult(
        check: "Memory safety",
        passed: !hasUncheckedExclusivity,
        details: hasUncheckedExclusivity ? "WARNING: Exclusivity checking disabled" : "Memory safety checks enabled"
    ))
    
    return results
}

func validatePerformanceFeatures() -> [ValidationResult] {
    var results: [ValidationResult] = []
    
    // Check for SIMD usage in code
    let sourceFiles = findSwiftFiles(in: "Sources/VectorCore")
    var simdUsage = false
    var inlinableCount = 0
    var unsafeFlagsCount = 0
    
    for file in sourceFiles {
        if let content = try? String(contentsOfFile: file) {
            if content.contains("vDSP") || content.contains("SIMD") {
                simdUsage = true
            }
            inlinableCount += content.components(separatedBy: "@inlinable").count - 1
            inlinableCount += content.components(separatedBy: "@inline(__always)").count - 1
            unsafeFlagsCount += content.components(separatedBy: "Unsafe").count - 1
        }
    }
    
    results.append(ValidationResult(
        check: "SIMD utilization",
        passed: simdUsage,
        details: simdUsage ? "SIMD operations found" : "Consider using SIMD"
    ))
    
    results.append(ValidationResult(
        check: "Inlining hints",
        passed: inlinableCount > 10,
        details: "Found \(inlinableCount) inlining hints"
    ))
    
    results.append(ValidationResult(
        check: "Unsafe optimizations",
        passed: unsafeFlagsCount > 5,
        details: "Found \(unsafeFlagsCount) unsafe operations"
    ))
    
    return results
}

func validateBenchmarkPerformance() -> [ValidationResult] {
    var results: [ValidationResult] = []
    
    // Check if baseline exists
    let baselineExists = FileManager.default.fileExists(atPath: "baseline_metrics.json")
    results.append(ValidationResult(
        check: "Performance baseline",
        passed: baselineExists,
        details: baselineExists ? "Baseline available for comparison" : "Run capture_baseline.swift"
    ))
    
    // Check for benchmark executable
    let benchmarkExists = findBenchmarkExecutable() != nil
    results.append(ValidationResult(
        check: "Benchmark executable",
        passed: benchmarkExists,
        details: benchmarkExists ? "Ready to run benchmarks" : "Build benchmarks first"
    ))
    
    return results
}

// MARK: - Helper Functions

func findLibrary() -> String? {
    let paths = [
        ".build/release/libVectorCore.a",
        ".build/release/VectorCore.a",
        ".build/release/libVectorCore.dylib",
        ".build/release/VectorCore.framework/VectorCore"
    ]
    
    for path in paths {
        if FileManager.default.fileExists(atPath: path) {
            return path
        }
    }
    
    // Search for any library file
    let enumerator = FileManager.default.enumerator(atPath: ".build/release")
    while let element = enumerator?.nextObject() as? String {
        if element.contains("VectorCore") && 
           (element.hasSuffix(".a") || element.hasSuffix(".dylib")) {
            return ".build/release/\(element)"
        }
    }
    
    return nil
}

func findSwiftModule() -> String? {
    let modulePath = ".build/release/VectorCore.swiftmodule"
    if FileManager.default.fileExists(atPath: modulePath) {
        return modulePath
    }
    
    // Search for module
    let enumerator = FileManager.default.enumerator(atPath: ".build/release")
    while let element = enumerator?.nextObject() as? String {
        if element.hasSuffix(".swiftmodule") {
            return ".build/release/\(element)"
        }
    }
    
    return nil
}

func findBenchmarkExecutable() -> String? {
    let paths = [
        ".build/release/VectorCoreBenchmarks",
        ".build/release/VectorCoreBenchmarks.exe"
    ]
    
    for path in paths {
        if FileManager.default.fileExists(atPath: path) {
            return path
        }
    }
    
    return nil
}

func findSwiftFiles(in directory: String) -> [String] {
    var files: [String] = []
    
    let enumerator = FileManager.default.enumerator(atPath: directory)
    while let element = enumerator?.nextObject() as? String {
        if element.hasSuffix(".swift") {
            files.append("\(directory)/\(element)")
        }
    }
    
    return files
}

// MARK: - Main Execution

func main() {
    print("🔍 VectorCore Optimization Validator")
    print("====================================")
    print()
    
    var allResults: [(String, [ValidationResult])] = []
    
    // Run all validations
    allResults.append(("Build Settings", validateBuildSettings()))
    allResults.append(("Compiler Flags", validateCompilerFlags()))
    allResults.append(("Performance Features", validatePerformanceFeatures()))
    allResults.append(("Benchmarks", validateBenchmarkPerformance()))
    
    // Display results
    var totalPassed = 0
    var totalChecks = 0
    
    for (category, results) in allResults {
        print("\(category):")
        print(String(repeating: "-", count: category.count + 1))
        
        for result in results {
            print("\(result.icon) \(result.check)")
            print("   \(result.details)")
            
            totalChecks += 1
            if result.passed { totalPassed += 1 }
        }
        
        print()
    }
    
    // Summary
    let percentage = totalChecks > 0 ? (Double(totalPassed) / Double(totalChecks) * 100) : 0
    let summaryIcon = percentage >= 80 ? "✅" : "⚠️"
    
    print("Summary:")
    print("========")
    print("\(summaryIcon) \(totalPassed)/\(totalChecks) checks passed (\(Int(percentage))%)")
    
    if percentage < 100 {
        print("\nRecommendations:")
        if !FileManager.default.fileExists(atPath: ".build/release") {
            print("  • Run ./Scripts/build_optimized.sh to create optimized build")
        }
        if !FileManager.default.fileExists(atPath: "baseline_metrics.json") {
            print("  • Run ./Scripts/capture_baseline.swift to establish baseline")
        }
        if totalPassed < totalChecks {
            print("  • Review failed checks above for optimization opportunities")
        }
    } else {
        print("\n✨ All optimizations are properly configured!")
    }
    
    // Exit with appropriate code
    exit(percentage >= 80 ? 0 : 1)
}

// Run validation
main()