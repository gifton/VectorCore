import Foundation
import VectorCore
import VectorCoreBenchmarking

/// Immutable per-catalog allocation seam; normal execution uses matched AlignedMemory calls.
struct BenchmarkAllocator: Sendable {
    let allocate: @Sendable (Int, Int) throws -> UnsafeMutablePointer<Float>
    let release: @Sendable (UnsafeMutablePointer<Float>, Int) -> Void
    // The two calls below run inside the timed (and warmup) body of the mem.alloc cases, so they are named functions
    // whose source is part of the `memory` body fingerprint (a `static let` closure pair would not be hashed).
    static func alignedAllocate(count: Int, alignment: Int) throws -> UnsafeMutablePointer<Float> {
        try AlignedMemory.allocateAligned(type: Float.self, count: count, alignment: alignment)
    }
    static func alignedRelease(_ pointer: UnsafeMutablePointer<Float>, _ initialized: Int) {
        pointer.deinitialize(count: initialized)
        AlignedMemory.deallocate(pointer)
    }
    static let aligned = BenchmarkAllocator(allocate: alignedAllocate, release: alignedRelease)
}

struct MemoryBench: BenchmarkSuite {
    static let name = "memory"
    static func catalog(options: ValidatedOptions, progress: BenchmarkProgressReporter) -> [BenchmarkCase] {
        catalog(options: options, allocator: .aligned)
    }
    static func catalog(options: ValidatedOptions, allocator: BenchmarkAllocator,
                        poolAcquire: @escaping @Sendable (Int, Int) -> MemoryPool.BufferHandle<Float>? = { MemoryPool.shared.acquire(type: Float.self, count: $0, alignment: $1) }) -> [BenchmarkCase] {
        var cases: [BenchmarkCase] = []
        for dim in options.dims {
            let count = dim * 1024 // ValidatedOptions checked the element and byte products.
            let alignment = AlignedMemory.optimalAlignment
            let allocName = "mem.alloc.aligned.\(dim)"
            cases.append(BenchmarkCase(allocName, boundary: Boundaries.record(allocName, options)) {
                func allocateOnce(_ value: Float) throws {
                    let pointer = try allocator.allocate(count, alignment)
                    let initialized = min(8, count)
                    pointer.initialize(repeating: value, count: initialized)
                    defer { allocator.release(pointer, initialized) }
                    blackHole(pointer[0])
                }
                try Harness.warmup { try allocateOnce(0) }
                return [try Harness.measure(name: allocName, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { try allocateOnce(1) }]
            })
            let copyName = "mem.copy.\(dim)"
            cases.append(BenchmarkCase(copyName, boundary: Boundaries.record(copyName, options)) {
                let source = try allocator.allocate(count, alignment)
                source.initialize(repeating: 0.5, count: count)
                defer { allocator.release(source, count) }
                let destination = try allocator.allocate(count, alignment)
                destination.initialize(repeating: 0, count: count)
                defer { allocator.release(destination, count) }
                let copy = {
                    alignedCopy(from: UnsafePointer(source), to: destination, count: count, preferredAlignment: alignment)
                    blackHole(destination[1])
                }
                try Harness.warmup(copy)
                return [try Harness.measure(name: copyName, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats,
                                            unitCount: count, samples: options.samples, copy)]
            })
            let fillName = "mem.fill.\(dim)"
            cases.append(BenchmarkCase(fillName, boundary: Boundaries.record(fillName, options)) {
                let destination = try allocator.allocate(count, alignment)
                destination.initialize(repeating: 0, count: count)
                defer { allocator.release(destination, count) }
                let fill = {
                    for index in 0..<count { destination[index] = 0.123 }
                    blackHole(destination[count - 1])
                }
                try Harness.warmup(fill)
                return [try Harness.measure(name: fillName, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: count, samples: options.samples, fill)]
            })
            let poolName = "mem.pool.acquire.\(dim)"
            cases.append(BenchmarkCase(poolName, boundary: Boundaries.record(poolName, options)) {
                func acquire(_ value: Float) throws {
                    guard let handle = poolAcquire(count, alignment) else {
                        throw BenchmarkConfigurationError("Memory pool acquisition failed")
                    }
                    handle.pointer.initialize(to: value)
                    defer { handle.pointer.deinitialize(count: 1) }
                    blackHole(handle.pointer[0])
                }
                try acquire(0)
                MemoryPool.shared.quiesce()
                try Harness.warmup { try acquire(1) }
                return [try Harness.measure(name: poolName, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) { try acquire(2) }]
            })
        }
        return cases
    }
}
