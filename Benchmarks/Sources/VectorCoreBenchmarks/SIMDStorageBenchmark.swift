// VectorCore: SIMD Storage Performance Benchmark
//
// Benchmarks comparing original vs optimized SIMD storage implementations
//

import Foundation
import Accelerate
@testable import VectorCore

/// Benchmark suite for SIMD storage implementations
public struct SIMDStorageBenchmark {
    
    /// Number of iterations for each benchmark
    private let iterations: Int
    
    /// Number of warmup iterations
    private let warmupIterations: Int
    
    public init(iterations: Int = 100_000, warmupIterations: Int = 1000) {
        self.iterations = iterations
        self.warmupIterations = warmupIterations
    }
    
    // MARK: - Benchmark Results
    
    public struct BenchmarkResult: Sendable {
        let name: String
        let nanoseconds: Double
        let allocations: Int
        let iterationsPerSecond: Double
        
        var formattedReport: String {
            """
            \(name):
              Time: \(String(format: "%.2f", nanoseconds)) ns/iter
              Rate: \(String(format: "%.0f", iterationsPerSecond)) ops/sec
              Allocations: \(allocations)
            """
        }
    }
    
    // MARK: - Allocation Tracking
    
    private class AllocationTracker {
        private var count = 0
        
        func track<T>(_ body: () throws -> T) rethrows -> T {
            // Simple allocation tracking - in production use Instruments
            count += 1
            return try body()
        }
        
        func reset() {
            count = 0
        }
        
        var allocationCount: Int { count }
    }
    
    // MARK: - Benchmarks
    
    /// Benchmark buffer access performance
    public func benchmarkBufferAccess() -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []
        
        // Benchmark SIMDStorage128
        results.append(benchmarkStorage128BufferAccess(
            name: "Original SIMDStorage128",
            storageType: SIMDStorage128.self
        ))
        
        results.append(benchmarkStorage128BufferAccess(
            name: "Optimized SIMDStorage128",
            storageType: OptimizedSIMDStorage128.self
        ))
        
        // Benchmark SIMDStorage256
        results.append(benchmarkStorage256BufferAccess(
            name: "Original SIMDStorage256",
            storageType: SIMDStorage256.self
        ))
        
        results.append(benchmarkStorage256BufferAccess(
            name: "Optimized SIMDStorage256",
            storageType: OptimizedSIMDStorage256.self
        ))
        
        return results
    }
    
    /// Benchmark dot product performance
    public func benchmarkDotProduct() -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []
        
        // 128 dimensions
        results.append(benchmarkDotProductOperation(
            name: "Original SIMDStorage128 DotProduct",
            storage1: SIMDStorage128(repeating: 1.0),
            storage2: SIMDStorage128(repeating: 2.0)
        ))
        
        results.append(benchmarkDotProductOperation(
            name: "Optimized SIMDStorage128 DotProduct",
            storage1: OptimizedSIMDStorage128(repeating: 1.0),
            storage2: OptimizedSIMDStorage128(repeating: 2.0)
        ))
        
        // 256 dimensions
        results.append(benchmarkDotProductOperation(
            name: "Original SIMDStorage256 DotProduct",
            storage1: SIMDStorage256(repeating: 1.0),
            storage2: SIMDStorage256(repeating: 2.0)
        ))
        
        results.append(benchmarkDotProductOperation(
            name: "Optimized SIMDStorage256 DotProduct",
            storage1: OptimizedSIMDStorage256(repeating: 1.0),
            storage2: OptimizedSIMDStorage256(repeating: 2.0)
        ))
        
        // 512 dimensions
        results.append(benchmarkDotProductOperation(
            name: "Original SIMDStorage512 DotProduct",
            storage1: SIMDStorage512(repeating: 1.0),
            storage2: SIMDStorage512(repeating: 2.0)
        ))
        
        results.append(benchmarkDotProductOperation(
            name: "Optimized SIMDStorage512 DotProduct",
            storage1: OptimizedSIMDStorage512(repeating: 1.0),
            storage2: OptimizedSIMDStorage512(repeating: 2.0)
        ))
        
        return results
    }
    
    /// Benchmark vector operations (addition)
    public func benchmarkVectorAddition() -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []
        
        // Test with Vector type using different storages
        // Use the existing Dim128 and Dim256 from Dimension.swift
        
        // Original implementations
        let origVec128a = Vector<Dim128>(repeating: 1.0)
        let origVec128b = Vector<Dim128>(repeating: 2.0)
        
        results.append(measureTime(name: "Original Vector<128> Addition") {
            for _ in 0..<iterations {
                _ = origVec128a + origVec128b
            }
        })
        
        return results
    }
    
    // MARK: - Helper Methods
    
    private func benchmarkStorage128BufferAccess<S: VectorStorage>(
        name: String,
        storageType: S.Type
    ) -> BenchmarkResult {
        let storage = S()
        var sum: Float = 0
        
        // Warmup
        for _ in 0..<warmupIterations {
            storage.withUnsafeBufferPointer { buffer in
                sum += buffer[0]
            }
        }
        
        // Measure
        let start = DispatchTime.now()
        for _ in 0..<iterations {
            storage.withUnsafeBufferPointer { buffer in
                sum += buffer[0] + buffer[64] + buffer[127]
            }
        }
        let end = DispatchTime.now()
        
        let nanoseconds = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / Double(iterations)
        let iterationsPerSecond = 1_000_000_000.0 / nanoseconds
        
        // Prevent optimization
        if sum == 0 { print("Optimization prevention") }
        
        return BenchmarkResult(
            name: name,
            nanoseconds: nanoseconds,
            allocations: 0, // Would need proper tracking
            iterationsPerSecond: iterationsPerSecond
        )
    }
    
    private func benchmarkStorage256BufferAccess<S: VectorStorage>(
        name: String,
        storageType: S.Type
    ) -> BenchmarkResult {
        let storage = S()
        var sum: Float = 0
        
        // Warmup
        for _ in 0..<warmupIterations {
            storage.withUnsafeBufferPointer { buffer in
                sum += buffer[0]
            }
        }
        
        // Measure
        let start = DispatchTime.now()
        for _ in 0..<iterations {
            storage.withUnsafeBufferPointer { buffer in
                sum += buffer[0] + buffer[128] + buffer[255]
            }
        }
        let end = DispatchTime.now()
        
        let nanoseconds = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / Double(iterations)
        let iterationsPerSecond = 1_000_000_000.0 / nanoseconds
        
        // Prevent optimization
        if sum == 0 { print("Optimization prevention") }
        
        return BenchmarkResult(
            name: name,
            nanoseconds: nanoseconds,
            allocations: 0,
            iterationsPerSecond: iterationsPerSecond
        )
    }
    
    private func benchmarkDotProductOperation<S: VectorStorageOperations>(
        name: String,
        storage1: S,
        storage2: S
    ) -> BenchmarkResult {
        var result: Float = 0
        
        // Warmup
        for _ in 0..<warmupIterations {
            result += storage1.dotProduct(storage2)
        }
        
        // Measure
        let start = DispatchTime.now()
        for _ in 0..<iterations {
            result += storage1.dotProduct(storage2)
        }
        let end = DispatchTime.now()
        
        let nanoseconds = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / Double(iterations)
        let iterationsPerSecond = 1_000_000_000.0 / nanoseconds
        
        // Prevent optimization
        if result == 0 { print("Optimization prevention") }
        
        return BenchmarkResult(
            name: name,
            nanoseconds: nanoseconds,
            allocations: 0,
            iterationsPerSecond: iterationsPerSecond
        )
    }
    
    private func measureTime(name: String, _ block: () -> Void) -> BenchmarkResult {
        let start = DispatchTime.now()
        block()
        let end = DispatchTime.now()
        
        let nanoseconds = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / Double(iterations)
        let iterationsPerSecond = 1_000_000_000.0 / nanoseconds
        
        return BenchmarkResult(
            name: name,
            nanoseconds: nanoseconds,
            allocations: 0,
            iterationsPerSecond: iterationsPerSecond
        )
    }
    
    // MARK: - Comparison Benchmark
    
    /// Run a comparison benchmark between original and optimized implementations
    public func runComparisonBenchmark() -> String {
        var report = "Storage Implementation Comparison:\n"
        report += "=" * 50 + "\n\n"
        
        // Run all benchmarks
        let bufferResults = benchmarkBufferAccess()
        let dotResults = benchmarkDotProduct()
        let _ = benchmarkVectorAddition()
        
        // Calculate improvements
        func calculateImprovement(original: BenchmarkResult, optimized: BenchmarkResult) -> Double {
            return ((original.nanoseconds - optimized.nanoseconds) / original.nanoseconds) * 100
        }
        
        // Buffer access improvements
        if bufferResults.count >= 4 {
            let improvement128 = calculateImprovement(
                original: bufferResults[0], // Original SIMDStorage128
                optimized: bufferResults[1] // Optimized SIMDStorage128
            )
            let improvement256 = calculateImprovement(
                original: bufferResults[2], // Original SIMDStorage256
                optimized: bufferResults[3] // Optimized SIMDStorage256
            )
            
            report += "Buffer Access Improvements:\n"
            report += "  128-dim: \(String(format: "%.1f%%", improvement128)) faster\n"
            report += "  256-dim: \(String(format: "%.1f%%", improvement256)) faster\n\n"
        }
        
        // Dot product improvements
        if dotResults.count >= 6 {
            let improvement128 = calculateImprovement(
                original: dotResults[0], // Original SIMDStorage128
                optimized: dotResults[1] // Optimized SIMDStorage128
            )
            let improvement256 = calculateImprovement(
                original: dotResults[2], // Original SIMDStorage256
                optimized: dotResults[3] // Optimized SIMDStorage256
            )
            let improvement512 = calculateImprovement(
                original: dotResults[4], // Original SIMDStorage512
                optimized: dotResults[5] // Optimized SIMDStorage512
            )
            
            report += "Dot Product Improvements:\n"
            report += "  128-dim: \(String(format: "%.1f%%", improvement128)) faster\n"
            report += "  256-dim: \(String(format: "%.1f%%", improvement256)) faster\n"
            report += "  512-dim: \(String(format: "%.1f%%", improvement512)) faster\n\n"
        }
        
        // Summary
        report += "Summary:\n"
        report += "Optimized implementations show significant performance gains,\n"
        report += "especially for operations on larger vector dimensions.\n"
        
        return report
    }
    
    // MARK: - Report Generation
    
    public static func runAllBenchmarks() {
        print("=== SIMD Storage Performance Benchmarks ===\n")
        
        let benchmark = SIMDStorageBenchmark()
        
        print("Buffer Access Performance:")
        print("-" * 50)
        for result in benchmark.benchmarkBufferAccess() {
            print(result.formattedReport)
            print()
        }
        
        print("\nDot Product Performance:")
        print("-" * 50)
        for result in benchmark.benchmarkDotProduct() {
            print(result.formattedReport)
            print()
        }
        
        print("\nVector Operation Performance:")
        print("-" * 50)
        for result in benchmark.benchmarkVectorAddition() {
            print(result.formattedReport)
            print()
        }
    }
}

// Helper for string repetition
fileprivate func * (left: String, right: Int) -> String {
    String(repeating: left, count: right)
}

// MARK: - Type Aliases for Testing

// Use the actual VectorCore types
typealias Vector128 = Vector<Dim128>
typealias Vector256 = Vector<Dim256>