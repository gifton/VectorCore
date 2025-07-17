// VectorCore: Performance Optimization Techniques
//
// This example demonstrates performance best practices and optimization strategies

import VectorCore
import Foundation

// MARK: - 1. Choosing the Right Vector Type

func vectorTypeSelection() {
    print("=== Choosing the Right Vector Type ===\n")
    
    // Benchmark different vector types
    let iterations = 10000
    
    // Strongly-typed vectors (fastest for known dimensions)
    let start128 = Date()
    for _ in 0..<iterations {
        let v1 = Vector128.random(in: -1...1)
        let v2 = Vector128.random(in: -1...1)
        let _ = v1.dotProduct(v2)
    }
    let time128 = Date().timeIntervalSince(start128)
    
    // Dynamic vectors (flexible but slower)
    let startDynamic = Date()
    for _ in 0..<iterations {
        let v1 = DynamicVector(dimension: 128, repeating: 0.5)
        let v2 = DynamicVector(dimension: 128, repeating: 0.5)
        let _ = v1.dotProduct(v2)
    }
    let timeDynamic = Date().timeIntervalSince(startDynamic)
    
    print("Performance comparison (\(iterations) iterations):")
    print("Vector128: \(String(format: "%.3f", time128))s")
    print("DynamicVector: \(String(format: "%.3f", timeDynamic))s")
    print("Speedup: \(String(format: "%.1fx", timeDynamic / time128))\n")
    
    // Memory layout optimization
    print("Memory characteristics:")
    print("Vector32 size: \(MemoryLayout<Vector32>.size) bytes")
    print("Vector128 size: \(MemoryLayout<Vector128>.size) bytes")
    print("Vector512 size: \(MemoryLayout<Vector512>.size) bytes")
    
    // Storage tier information
    print("\nStorage tiers:")
    print("Small vectors (≤64): Stack allocated, no heap overhead")
    print("Medium vectors (≤512): Single heap allocation")
    print("Large vectors (>512): Dynamic allocation")
}

// MARK: - 2. Batch Processing Optimization

func batchProcessingOptimization() {
    print("\n\n=== Batch Processing Optimization ===\n")
    
    let vectorCount = 1000
    let vectors = (0..<vectorCount).map { _ in Vector256.random(in: -1...1) }
    let query = Vector256.random(in: -1...1)
    
    // Method 1: Sequential processing
    let startSeq = Date()
    let seqDistances = vectors.map { $0.distance(to: query) }
    let timeSeq = Date().timeIntervalSince(startSeq)
    
    // Method 2: Batch operations
    let batchOps = SyncBatchOperations()
    let startBatch = Date()
    let batchDistances = batchOps.batchDistance(
        from: query,
        to: vectors,
        using: EuclideanDistance()
    )
    let timeBatch = Date().timeIntervalSince(startBatch)
    
    print("Processing \(vectorCount) vectors:")
    print("Sequential: \(String(format: "%.3f", timeSeq))s")
    print("Batch: \(String(format: "%.3f", timeBatch))s")
    print("Speedup: \(String(format: "%.1fx", timeSeq / timeBatch))")
    
    // Pre-normalization optimization
    print("\nPre-normalization optimization:")
    
    // Without pre-normalization
    let startWithout = Date()
    for _ in 0..<100 {
        let similarities = vectors.map { v in
            v.normalized().cosineSimilarity(to: query.normalized())
        }
    }
    let timeWithout = Date().timeIntervalSince(startWithout)
    
    // With pre-normalization
    let normalizedVectors = vectors.map { $0.normalized() }
    let normalizedQuery = query.normalized()
    let startWith = Date()
    for _ in 0..<100 {
        let similarities = normalizedVectors.map { v in
            v.cosineSimilarity(to: normalizedQuery)
        }
    }
    let timeWith = Date().timeIntervalSince(startWith)
    
    print("Without pre-normalization: \(String(format: "%.3f", timeWithout))s")
    print("With pre-normalization: \(String(format: "%.3f", timeWith))s")
    print("Speedup: \(String(format: "%.1fx", timeWithout / timeWith))")
}

// MARK: - 3. Memory-Efficient Operations

func memoryEfficientOperations() {
    print("\n\n=== Memory-Efficient Operations ===\n")
    
    // Copy-on-Write demonstration
    print("Copy-on-Write optimization:")
    let original = Vector512.random(in: -1...1)
    
    // Assignment doesn't copy immediately
    var copy1 = original
    var copy2 = original
    
    print("Created 2 copies (no actual copying yet)")
    
    // Modification triggers copy
    copy1[0] = 999.0  // This triggers COW
    print("Modified copy1 (COW triggered)")
    
    // Using in-place operations
    print("\nIn-place operations:")
    var mutable = Vector256.random(in: -1...1)
    
    // Less efficient: creates new vector
    let scaled1 = mutable * 2.0
    
    // More efficient: modify in place
    for i in 0..<256 {
        mutable[i] *= 2.0
    }
    
    print("In-place modification completed")
    
    // Memory pool pattern for temporary vectors
    print("\nMemory pool pattern:")
    
    class VectorPool {
        private var available: [Vector512] = []
        
        func acquire() -> Vector512 {
            if let vector = available.popLast() {
                return vector
            }
            return Vector512.zeros()
        }
        
        func release(_ vector: Vector512) {
            available.append(vector)
        }
    }
    
    let pool = VectorPool()
    
    // Use pool for temporary calculations
    for _ in 0..<10 {
        let temp = pool.acquire()
        // Use temp for calculations...
        pool.release(temp)
    }
    
    print("Vector pool reduces allocation overhead")
}

// MARK: - 4. SIMD and Accelerate Optimization

func simdOptimization() {
    print("\n\n=== SIMD and Accelerate Optimization ===\n")
    
    // Operations that benefit from SIMD
    let v1 = Vector1024.random(in: -1...1)
    let v2 = Vector1024.random(in: -1...1)
    
    print("SIMD-accelerated operations:")
    
    // These operations use vDSP under the hood
    let operations: [(String, () -> Float)] = [
        ("Dot product", { v1.dotProduct(v2) }),
        ("Sum", { v1.sum }),
        ("Mean", { v1.mean }),
        ("L1 norm", { v1.l1Norm }),
        ("L2 norm", { v1.l2Norm })
    ]
    
    for (name, operation) in operations {
        let start = Date()
        for _ in 0..<1000 {
            let _ = operation()
        }
        let time = Date().timeIntervalSince(start)
        print("  \(name): \(String(format: "%.4f", time))s for 1000 iterations")
    }
    
    // Fused operations
    print("\nFused operations (better cache usage):")
    
    // Less efficient: separate operations
    let start1 = Date()
    for _ in 0..<1000 {
        let normalized = v1.normalized()
        let scaled = normalized * 2.0
        let shifted = scaled + Vector1024(repeating: 1.0)
    }
    let time1 = Date().timeIntervalSince(start1)
    
    // More efficient: fused operation
    let start2 = Date()
    for _ in 0..<1000 {
        let magnitude = v1.magnitude
        let result = v1.toArray().map { ($0 / magnitude) * 2.0 + 1.0 }
        let _ = Vector1024(result)
    }
    let time2 = Date().timeIntervalSince(start2)
    
    print("Separate operations: \(String(format: "%.3f", time1))s")
    print("Fused operation: \(String(format: "%.3f", time2))s")
}

// MARK: - 5. Caching and Precomputation

func cachingStrategies() {
    print("\n\n=== Caching and Precomputation ===\n")
    
    // Magnitude caching example
    class CachedVector {
        private let vector: Vector512
        private var _magnitude: Float?
        
        init(_ vector: Vector512) {
            self.vector = vector
        }
        
        var magnitude: Float {
            if let cached = _magnitude {
                return cached
            }
            _magnitude = vector.magnitude
            return _magnitude!
        }
    }
    
    let cached = CachedVector(Vector512.random(in: -1...1))
    
    print("Magnitude caching:")
    let start1 = Date()
    for _ in 0..<10000 {
        let _ = cached.magnitude  // First call computes, rest use cache
    }
    let time1 = Date().timeIntervalSince(start1)
    
    let uncached = Vector512.random(in: -1...1)
    let start2 = Date()
    for _ in 0..<10000 {
        let _ = uncached.magnitude  // Computes every time
    }
    let time2 = Date().timeIntervalSince(start2)
    
    print("Cached: \(String(format: "%.4f", time1))s")
    print("Uncached: \(String(format: "%.4f", time2))s")
    print("Speedup: \(String(format: "%.0fx", time2 / time1))")
    
    // Precomputed lookup tables
    print("\nPrecomputed similarity matrix:")
    
    let database = (0..<100).map { _ in Vector128.random(in: -1...1).normalized() }
    
    // Precompute all pairwise similarities
    var similarityMatrix = [[Float]](
        repeating: [Float](repeating: 0, count: database.count),
        count: database.count
    )
    
    let startPrecompute = Date()
    for i in 0..<database.count {
        for j in i..<database.count {
            let similarity = database[i].cosineSimilarity(to: database[j])
            similarityMatrix[i][j] = similarity
            similarityMatrix[j][i] = similarity
        }
    }
    let precomputeTime = Date().timeIntervalSince(startPrecompute)
    
    print("Precomputed \(database.count * database.count) similarities in \(String(format: "%.3f", precomputeTime))s")
    print("Average per lookup: \(String(format: "%.6f", precomputeTime / Double(database.count * database.count)))s")
}

// MARK: - 6. Profiling and Measurement

func profilingAndMeasurement() {
    print("\n\n=== Profiling and Measurement ===\n")
    
    // Simple profiler
    class SimpleProfiler {
        private var measurements: [String: [TimeInterval]] = [:]
        
        func measure<T>(_ name: String, block: () throws -> T) rethrows -> T {
            let start = Date()
            let result = try block()
            let elapsed = Date().timeIntervalSince(start)
            measurements[name, default: []].append(elapsed)
            return result
        }
        
        func report() {
            print("\nProfile Report:")
            for (name, times) in measurements.sorted(by: { $0.key < $1.key }) {
                let total = times.reduce(0, +)
                let average = total / Double(times.count)
                print("  \(name):")
                print("    Calls: \(times.count)")
                print("    Total: \(String(format: "%.3f", total))s")
                print("    Average: \(String(format: "%.6f", average))s")
            }
        }
    }
    
    let profiler = SimpleProfiler()
    
    // Profile different operations
    let vectors = (0..<100).map { _ in Vector256.random(in: -1...1) }
    
    for vector in vectors {
        _ = profiler.measure("normalize") { vector.normalized() }
        _ = profiler.measure("magnitude") { vector.magnitude }
        _ = profiler.measure("entropy") { vector.entropy }
        _ = profiler.measure("quality") { vector.quality }
    }
    
    profiler.report()
}

// MARK: - Performance Tips Summary

func performanceTipsSummary() {
    print("\n\n=== Performance Tips Summary ===\n")
    
    print("""
    1. Vector Type Selection:
       - Use strongly-typed vectors (Vector128, Vector256, etc.) when dimensions are known
       - Use DynamicVector only when dimension is truly runtime-determined
       - Consider memory alignment for SIMD operations
    
    2. Batch Operations:
       - Process multiple vectors together using SyncBatchOperations
       - Pre-normalize vectors when doing multiple similarity computations
       - Use appropriate distance metrics (dot product is fastest)
    
    3. Memory Efficiency:
       - Leverage Copy-on-Write (COW) - avoid unnecessary modifications
       - Use in-place operations when possible
       - Consider memory pools for temporary vectors
    
    4. SIMD Optimization:
       - VectorCore automatically uses Accelerate framework
       - Fuse operations to improve cache locality
       - Process data in chunks that fit in cache
    
    5. Caching:
       - Cache expensive computations (magnitude, normalization)
       - Precompute similarity matrices for fixed databases
       - Use lookup tables for repeated calculations
    
    6. Profiling:
       - Measure before optimizing
       - Focus on hot paths identified by profiling
       - Use performance regression tests
    """)
}

// MARK: - Main

func main() {
    vectorTypeSelection()
    batchProcessingOptimization()
    memoryEfficientOperations()
    simdOptimization()
    cachingStrategies()
    profilingAndMeasurement()
    performanceTipsSummary()
}

// Run the example
main()