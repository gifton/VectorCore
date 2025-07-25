# Phase 5: Modern Operations Layer

## Overview

This final phase implements the ExecutionContext-based operations system, following patterns from Intel MKL and PyTorch. We'll create a unified API that eliminates sync/async duplication while preparing for GPU acceleration.

## Design Decisions Applied

From previous phases:
- Swift 6.0 with actor isolation for thread safety
- Safe/unsafe subscript variants with Swift naming conventions
- Sequence-only conformance for vectors
- Platform-specific optimizations for Apple Silicon
- 5% performance regression tolerance

## ExecutionContext Architecture

### Core Context Protocol

```swift
// Base protocol for all execution contexts
public protocol ExecutionContext: Sendable {
    var device: ComputeDevice { get }
    var maxThreadCount: Int { get }
    var preferredChunkSize: Int { get }
    
    // Execute work with context-specific optimization
    func execute<T>(_ work: @Sendable @escaping () throws -> T) async throws -> T
}

// Compute device enumeration
public enum ComputeDevice: Sendable, Hashable {
    case cpu
    case gpu(index: Int = 0)
    case neural
    
    public var isAccelerated: Bool {
        switch self {
        case .cpu: return false
        case .gpu, .neural: return true
        }
    }
}
```

### CPU Execution Context

```swift
public struct CPUContext: ExecutionContext {
    public let device = ComputeDevice.cpu
    public let maxThreadCount: Int
    public let preferredChunkSize: Int
    private let queue: DispatchQueue?
    
    // Preset configurations
    public static let sequential = CPUContext(threadCount: 1)
    public static let automatic = CPUContext(
        threadCount: ProcessInfo.processInfo.activeProcessorCount
    )
    
    public init(threadCount: Int? = nil, queue: DispatchQueue? = nil) {
        self.maxThreadCount = threadCount ?? ProcessInfo.processInfo.activeProcessorCount
        self.queue = queue
        
        // Optimize chunk size for cache efficiency
        #if os(iOS) || arch(arm64)
        // Apple Silicon: larger L1 cache, optimize for it
        self.preferredChunkSize = 16384 / MemoryLayout<Float>.size  // 16KB chunks
        #else
        // Intel: standard cache line optimization
        self.preferredChunkSize = 8192 / MemoryLayout<Float>.size   // 8KB chunks
        #endif
    }
    
    public func execute<T>(_ work: @Sendable @escaping () throws -> T) async throws -> T {
        if let queue = queue {
            return try await withCheckedThrowingContinuation { continuation in
                queue.async {
                    do {
                        let result = try work()
                        continuation.resume(returning: result)
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
        } else {
            return try work()
        }
    }
}
```

### GPU Context (Future-Ready)

```swift
// Placeholder for Metal implementation
public struct MetalContext: ExecutionContext {
    public let device: ComputeDevice
    public let maxThreadCount: Int
    public let preferredChunkSize: Int
    
    private let commandQueue: Any? // MTLCommandQueue
    
    public init(deviceIndex: Int = 0) async throws {
        self.device = .gpu(index: deviceIndex)
        
        // Metal setup would go here
        #if canImport(Metal)
        // Initialize Metal device and command queue
        self.maxThreadCount = 1024  // GPU thread groups
        self.preferredChunkSize = 65536  // Larger chunks for GPU
        #else
        throw VectorError.unsupportedDevice("Metal not available")
        #endif
        
        self.commandQueue = nil  // Placeholder
    }
    
    public func execute<T>(_ work: @Sendable @escaping () throws -> T) async throws -> T {
        // GPU execution would dispatch to Metal
        fatalError("GPU execution not yet implemented")
    }
}
```

### Default Context Strategy

Based on user decision: **Always parallel CPU** - matches expectations from NumPy, PyTorch, and other vector libraries.

```swift
public extension ExecutionContext where Self == CPUContext {
    /// Default execution context - automatically uses available CPU cores
    static var `default`: any ExecutionContext { CPUContext.automatic }
}

// Usage is simple and familiar:
let result = try await Operations.findNearest(to: query, in: vectors)
// Automatically parallel, just like NumPy/PyTorch
```

## Unified Operations Implementation

### Base Operations Structure

```swift
public enum Operations {
    // Thresholds based on empirical testing
    private static let parallelThreshold = 1000
    private static let vectorizedThreshold = 64
    
    // Thread-local storage for temporary buffers
    @TaskLocal static var temporaryBuffers: BufferPool?
}

// Buffer pool for reducing allocations
actor BufferPool {
    private var available: [Int: [UnsafeMutableBufferPointer<Float>]] = [:]
    
    func acquire(count: Int) -> UnsafeMutableBufferPointer<Float> {
        if let buffer = available[count]?.popLast() {
            return buffer
        }
        
        let memory = UnsafeMutablePointer<Float>.allocate(capacity: count)
        return UnsafeMutableBufferPointer(start: memory, count: count)
    }
    
    func release(_ buffer: UnsafeMutableBufferPointer<Float>) {
        available[buffer.count, default: []].append(buffer)
    }
}
```

### K-Nearest Neighbors Implementation

```swift
extension Operations {
    // Single API that handles both sync and async efficiently
    public static func findNearest<V: VectorProtocol>(
        to query: borrowing V,
        in vectors: borrowing [V],
        k: Int = 10,
        metric: any DistanceMetric = EuclideanDistance(),
        context: any ExecutionContext = .default
    ) async throws -> [(index: Int, distance: Float)] {
        
        let count = vectors.count
        guard count > 0 else { return [] }
        guard k > 0 else { return [] }
        
        // Choose execution strategy based on size
        if count < parallelThreshold {
            return try await context.execute {
                findNearestSequential(
                    query: query,
                    vectors: vectors,
                    k: k,
                    metric: metric
                )
            }
        }
        
        // Parallel execution for large datasets
        switch context.device {
        case .cpu:
            return try await findNearestParallelCPU(
                query: query,
                vectors: vectors,
                k: k,
                metric: metric,
                context: context
            )
            
        case .gpu:
            return try await findNearestGPU(
                query: query,
                vectors: vectors,
                k: k,
                metric: metric,
                context: context
            )
            
        case .neural:
            throw VectorError.unsupportedDevice("Neural Engine not yet supported")
        }
    }
    
    // Sequential implementation
    private static func findNearestSequential<V: VectorProtocol>(
        query: borrowing V,
        vectors: borrowing [V],
        k: Int,
        metric: any DistanceMetric
    ) -> [(index: Int, distance: Float)] {
        
        // Use a min-heap for efficient k-nearest tracking
        var heap = MinHeap<(index: Int, distance: Float)>(
            capacity: k,
            compare: { $0.distance < $1.distance }
        )
        
        // For small k, use heap; for large k, sort all
        if k < vectors.count / 10 {
            // Heap-based selection
            for (index, vector) in vectors.enumerated() {
                let distance = metric.distance(query, vector)
                heap.insert((index, distance))
            }
            return heap.sorted()
        } else {
            // Sort-based selection
            let distances = vectors.enumerated().map { index, vector in
                (index, metric.distance(query, vector))
            }
            return Array(distances.sorted(by: { $0.1 < $1.1 }).prefix(k))
        }
    }
    
    // Parallel CPU implementation
    private static func findNearestParallelCPU<V: VectorProtocol>(
        query: borrowing V,
        vectors: borrowing [V],
        k: Int,
        metric: any DistanceMetric,
        context: any ExecutionContext
    ) async throws -> [(index: Int, distance: Float)] {
        
        let count = vectors.count
        let chunkSize = max(context.preferredChunkSize, count / context.maxThreadCount)
        let chunkCount = (count + chunkSize - 1) / chunkSize
        
        // Process chunks in parallel
        let chunkResults = try await withThrowingTaskGroup(
            of: [(index: Int, distance: Float)].self
        ) { group in
            for chunkIndex in 0..<chunkCount {
                let start = chunkIndex * chunkSize
                let end = min(start + chunkSize, count)
                
                group.addTask {
                    let chunkVectors = Array(vectors[start..<end])
                    let chunkK = min(k, chunkVectors.count)
                    
                    var results = findNearestSequential(
                        query: query,
                        vectors: chunkVectors,
                        k: chunkK,
                        metric: metric
                    )
                    
                    // Adjust indices to global range
                    for i in results.indices {
                        results[i].index += start
                    }
                    
                    return results
                }
            }
            
            // Collect all chunk results
            var allResults: [(index: Int, distance: Float)] = []
            for try await chunkResult in group {
                allResults.append(contentsOf: chunkResult)
            }
            return allResults
        }
        
        // Merge chunk results
        return Array(chunkResults.sorted(by: { $0.distance < $1.distance }).prefix(k))
    }
}
```

### Batch Operations

```swift
extension Operations {
    // Process multiple queries efficiently
    public static func findNearestBatch<V: VectorProtocol>(
        queries: borrowing [V],
        in vectors: borrowing [V],
        k: Int = 10,
        metric: any DistanceMetric = EuclideanDistance(),
        context: any ExecutionContext = .default
    ) async throws -> [[(index: Int, distance: Float)]] {
        
        // For single query, use non-batch version
        if queries.count == 1 {
            let result = try await findNearest(
                to: queries[0],
                in: vectors,
                k: k,
                metric: metric,
                context: context
            )
            return [result]
        }
        
        // Process queries in parallel
        return try await withThrowingTaskGroup(
            of: (Int, [(index: Int, distance: Float)]).self
        ) { group in
            for (queryIndex, query) in queries.enumerated() {
                group.addTask {
                    let result = try await findNearest(
                        to: query,
                        in: vectors,
                        k: k,
                        metric: metric,
                        context: context
                    )
                    return (queryIndex, result)
                }
            }
            
            // Collect results in order
            var results = Array(repeating: [(index: Int, distance: Float)](), count: queries.count)
            for try await (index, result) in group {
                results[index] = result
            }
            return results
        }
    }
}
```

### Vector Transformations

```swift
extension Operations {
    // Map operation with automatic parallelization
    public static func map<V: VectorProtocol>(
        _ vectors: borrowing [V],
        transform: @escaping (Float) -> Float,
        context: any ExecutionContext = .default
    ) async throws -> [V] {
        
        if vectors.count < parallelThreshold {
            // Sequential for small arrays
            return try await context.execute {
                vectors.map { vector in
                    var result = V(storage: vector.storage.copy())
                    result.withUnsafeMutableBufferPointer { buffer in
                        for i in buffer.indices {
                            buffer[i] = transform(buffer[i])
                        }
                    }
                    return result
                }
            }
        }
        
        // Parallel for large arrays
        return try await withThrowingTaskGroup(of: (Int, V).self) { group in
            for (index, vector) in vectors.enumerated() {
                group.addTask {
                    var result = V(storage: vector.storage.copy())
                    result.withUnsafeMutableBufferPointer { buffer in
                        for i in buffer.indices {
                            buffer[i] = transform(buffer[i])
                        }
                    }
                    return (index, result)
                }
            }
            
            var results = Array<V?>(repeating: nil, count: vectors.count)
            for try await (index, result) in group {
                results[index] = result
            }
            return results.compactMap { $0 }
        }
    }
    
    // Reduce operation
    public static func reduce<V: VectorProtocol>(
        _ vectors: borrowing [V],
        _ initialResult: V,
        _ nextPartialResult: @escaping (V, V) -> V,
        context: any ExecutionContext = .default
    ) async throws -> V {
        
        guard !vectors.isEmpty else { return initialResult }
        
        if vectors.count < parallelThreshold {
            // Sequential reduction
            return try await context.execute {
                vectors.reduce(initialResult, nextPartialResult)
            }
        }
        
        // Parallel reduction using divide-and-conquer
        return try await parallelReduce(
            vectors,
            initialResult,
            nextPartialResult,
            context: context
        )
    }
}
```

### Distance Matrix Computation

```swift
extension Operations {
    // Compute pairwise distances efficiently
    public static func distanceMatrix<V: VectorProtocol>(
        _ vectors: borrowing [V],
        metric: any DistanceMetric = EuclideanDistance(),
        context: any ExecutionContext = .default
    ) async throws -> TriangularMatrix<Float> {
        
        let n = vectors.count
        var matrix = TriangularMatrix<Float>(dimension: n, defaultValue: 0)
        
        if n * n < parallelThreshold {
            // Sequential for small matrices
            return try await context.execute {
                for i in 0..<n {
                    for j in (i+1)..<n {
                        matrix[i, j] = metric.distance(vectors[i], vectors[j])
                    }
                }
                return matrix
            }
        }
        
        // Parallel computation
        try await withThrowingTaskGroup(of: Void.self) { group in
            for i in 0..<n {
                group.addTask {
                    for j in (i+1)..<n {
                        let distance = metric.distance(vectors[i], vectors[j])
                        await matrix.set(i, j, to: distance)
                    }
                }
            }
            
            try await group.waitForAll()
        }
        
        return matrix
    }
}
```

### Error Handling Strategy

Based on user decision: **Throw immediately** - matches NumPy's ValueError and PyTorch's runtime errors for dimension mismatches.

```swift
// Fail-fast with clear errors, just like other vector libraries
extension Operations {
    private static func validateDimensions<V: VectorProtocol>(_ vectors: borrowing [V]) throws {
        guard let first = vectors.first else { return }
        let expectedDim = first.scalarCount
        
        for (index, vector) in vectors.enumerated().dropFirst() {
            guard vector.scalarCount == expectedDim else {
                throw VectorError.dimensionMismatch(
                    expected: expectedDim,
                    actual: vector.scalarCount,
                    at: index
                )
            }
        }
    }
}

// Usage mirrors NumPy/PyTorch behavior:
// Python: a + b  # ValueError if shapes don't match
// Swift:  try a + b  // VectorError if dimensions don't match
```

## Performance Optimizations

### SIMD-Accelerated Distance Metrics

```swift
extension EuclideanDistance {
    // Optimized batch distance computation
    public func batchDistance<V: VectorProtocol>(
        from query: borrowing V,
        to candidates: borrowing [V],
        results: inout [Float]
    ) {
        precondition(results.count >= candidates.count)
        
        query.withUnsafeBufferPointer { queryBuffer in
            for (index, candidate) in candidates.enumerated() {
                candidate.withUnsafeBufferPointer { candidateBuffer in
                    var sum: Float = 0
                    vDSP_distancesq(
                        queryBuffer.baseAddress!, 1,
                        candidateBuffer.baseAddress!, 1,
                        &sum,
                        vDSP_Length(queryBuffer.count)
                    )
                    results[index] = sqrt(sum)
                }
            }
        }
    }
}
```

### Memory Management

```swift
// Pre-allocate buffers for operations
public struct OperationContext {
    let bufferPool = BufferPool()
    let temporaryVectors: [Any] = []
    
    public func withTemporaryBuffer<T>(
        count: Int,
        _ body: (UnsafeMutableBufferPointer<Float>) throws -> T
    ) async rethrows -> T {
        let buffer = await bufferPool.acquire(count: count)
        defer { Task { await bufferPool.release(buffer) } }
        return try body(buffer)
    }
}
```

## Validation and Testing

```swift
// Verify parallelization works correctly
func testParallelConsistency() async throws {
    let vectors = (0..<10000).map { _ in DynamicVector.random(dimension: 768) }
    let query = DynamicVector.random(dimension: 768)
    
    // Sequential result
    let sequential = try await Operations.findNearest(
        to: query,
        in: vectors,
        k: 100,
        context: CPUContext.sequential
    )
    
    // Parallel result
    let parallel = try await Operations.findNearest(
        to: query,
        in: vectors,
        k: 100,
        context: CPUContext.automatic
    )
    
    // Results should be identical
    assert(sequential.map(\.index) == parallel.map(\.index))
}
```

## Summary

This operations layer provides:
1. **Unified API**: No separate sync/async methods
2. **Automatic optimization**: Size-based parallelization
3. **Future-ready**: GPU context ready to implement
4. **Memory efficient**: Buffer pooling and reuse
5. **Type safe**: Swift 6.0 concurrency throughout

The ExecutionContext pattern allows users to control execution while providing sensible defaults that automatically parallelize when beneficial.

## Implementation Summary

With both decision points resolved, our operations layer provides:

1. **Familiar API** - Defaults match NumPy/PyTorch expectations
2. **Automatic parallelization** - Operations use all cores by default
3. **Clear error handling** - Immediate throws for dimension mismatches
4. **Future-ready** - GPU context ready for Metal implementation
5. **Performance-optimized** - SIMD operations and smart chunking

## Complete Architecture Decisions

All architectural decisions for the VectorCore refactoring are now finalized:

### From Previous Phases:
- ✅ **Swift 6.0** - 100% modern Swift with bleeding-edge features
- ✅ **Protocol-based consolidation** - Unified Vector/DynamicVector design
- ✅ **Generic storage with phantom types** - Type-safe dimensions
- ✅ **ExecutionContext pattern** - Following Intel MKL/PyTorch
- ✅ **Platform-specific alignment** - 16 bytes on Apple Silicon
- ✅ **Small vector optimization** - Stack allocation for ≤16 elements
- ✅ **Safe/unsafe subscript variants** - Swift-style API
- ✅ **Sequence-only conformance** - Not full Collection

### From This Phase:
- ✅ **Always parallel default** - Matches other vector libraries
- ✅ **Throw on errors** - Fail-fast like NumPy/PyTorch

## Next Steps

1. Begin implementation following the 5-phase plan
2. Set up benchmarking infrastructure first (Phase 1)
3. Implement protocols with Swift 6.0 features (Phase 2)
4. Build unified storage system (Phase 3)
5. Create vector types with zero duplication (Phase 4)
6. Implement modern operations layer (Phase 5)

The implementation guide is now complete with all architectural decisions documented!