import Foundation
import VectorCore
import VectorCoreBenchmarking

struct MemoryBench: BenchmarkSuite {
    static let name = "memory"

    static func run(options: CLIOptions, progress: ProgressReporter) async -> [BenchResult] {
        var results: [BenchResult] = []
        for dim in options.dims {
            results += bench(forDim: dim, options: options)
        }
        return results
    }

    private static func bench(forDim dim: Int, options: CLIOptions) -> [BenchResult] {
        var out: [BenchResult] = []

        // Choose buffer length proportional to dim (e.g., dim * 1024 floats)
        let count = max(dim * 1024, 64)
        let alignment = AlignedMemory.optimalAlignment

        // MARK: - Aligned alloc/free per-iteration
        do {
            let label = "mem.alloc.aligned.\(dim)"
            Harness.warmup {
                if let p: UnsafeMutablePointer<Float> = try? AlignedMemory.allocateAligned(type: Float.self, count: count, alignment: alignment) {
                    p.initialize(repeating: 0.0, count: min(8, count))
                    p.deallocate()
                }
            }
            let r = Harness.measure(name: label, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
                if let p: UnsafeMutablePointer<Float> = try? AlignedMemory.allocateAligned(type: Float.self, count: count, alignment: alignment) {
                    p.initialize(repeating: 1.0, count: min(8, count))
                    p.deallocate()
                }
            }
            out.append(r)
        }

        // Allocate persistent src/dst for copy/fill
        let srcPtr: UnsafeMutablePointer<Float> = (try? AlignedMemory.allocateAligned(type: Float.self, count: count, alignment: alignment)) ?? UnsafeMutablePointer<Float>.allocate(capacity: count)
        let dstPtr: UnsafeMutablePointer<Float> = (try? AlignedMemory.allocateAligned(type: Float.self, count: count, alignment: alignment)) ?? UnsafeMutablePointer<Float>.allocate(capacity: count)
        srcPtr.initialize(repeating: 0.5, count: count)
        dstPtr.initialize(repeating: 0.0, count: count)

        // Ensure deallocation after benchmarks complete
        defer {
            srcPtr.deinitialize(count: count)
            dstPtr.deinitialize(count: count)
            srcPtr.deallocate()
            dstPtr.deallocate()
        }

        // MARK: - Copy throughput
        do {
            let label = "mem.copy.\(dim)"
            Harness.warmup {
                alignedCopy(from: UnsafePointer(srcPtr), to: dstPtr, count: count, preferredAlignment: alignment)
                blackHole(dstPtr[0])
            }
            let r = Harness.measure(name: label, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: count, samples: options.samples) {
                alignedCopy(from: UnsafePointer(srcPtr), to: dstPtr, count: count, preferredAlignment: alignment)
                blackHole(dstPtr[1])
            }
            out.append(r)
        }

        // MARK: - Fill throughput
        do {
            let label = "mem.fill.\(dim)"
            let value: Float = 0.123
            Harness.warmup {
                var i = 0
                while i < count { dstPtr[i] = value; i += 1 }
                blackHole(dstPtr[count - 1])
            }
            let r = Harness.measure(name: label, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: count, samples: options.samples) {
                var i = 0
                while i < count { dstPtr[i] = value; i += 1 }
                blackHole(dstPtr[count - 1])
            }
            out.append(r)
        }

        // MARK: - Pool acquire/return overhead
        do {
            let label = "mem.pool.acquire.\(dim)"
            // Seed the pool
            _ = MemoryPool.shared.acquire(type: Float.self, count: count, alignment: alignment)
            MemoryPool.shared.quiesce()

            Harness.warmup {
                if let handle = MemoryPool.shared.acquire(type: Float.self, count: count, alignment: alignment) {
                    handle.pointer[0] = 1.0
                    blackHole(handle.pointer[0])
                }
            }
            let r = Harness.measure(name: label, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, samples: options.samples) {
                if let handle = MemoryPool.shared.acquire(type: Float.self, count: count, alignment: alignment) {
                    handle.pointer[0] = 2.0
                    blackHole(handle.pointer[0])
                }
            }
            out.append(r)
        }

        return out
    }
}

