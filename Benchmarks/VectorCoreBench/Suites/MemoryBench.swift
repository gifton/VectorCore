import Foundation
import VectorCore
import VectorCoreBenchmarking

struct MemoryBench: BenchmarkSuite {
    static let name = "memory"

    static func run(options: CLIOptions, progress: ProgressReporter) async -> [BenchResult] {
        var results: [BenchResult] = []

        // Pre-count all cases for progress tracking
        var allCaseNames: [String] = []
        for dim in options.dims {
            allCaseNames += [
                "mem.alloc.aligned.\(dim)",
                "mem.copy.\(dim)",
                "mem.fill.\(dim)",
                "mem.pool.acquire.\(dim)"
            ]
        }
        let casesToRun = allCaseNames.filter { Filters.shouldRun(name: $0, options: options) }
        let totalCases = casesToRun.count
        var currentIndex = 0

        for dim in options.dims {
            results += bench(forDim: dim, options: options, progress: progress, totalCases: totalCases, currentIndex: &currentIndex)
        }
        return results
    }

    private static func bench(forDim dim: Int, options: CLIOptions, progress: ProgressReporter, totalCases: Int, currentIndex: inout Int) -> [BenchResult] {
        var out: [BenchResult] = []

        let count = max(dim * 1024, 64)
        let alignment = AlignedMemory.optimalAlignment

        // MARK: - Aligned alloc/free per-iteration
        do {
            let label = "mem.alloc.aligned.\(dim)"
            if Filters.shouldRun(name: label, options: options) {

            let caseStart = Date()
            progress.caseStarted(suite: Self.name, name: label, index: currentIndex, total: totalCases)

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

            let caseDuration = Date().timeIntervalSince(caseStart) * 1000.0
            progress.caseCompleted(suite: Self.name, name: label, index: currentIndex, total: totalCases, durationMs: caseDuration)
            currentIndex += 1
            }
        }

        // Allocate persistent src/dst for copy/fill
        let srcPtr: UnsafeMutablePointer<Float> = (try? AlignedMemory.allocateAligned(type: Float.self, count: count, alignment: alignment)) ?? UnsafeMutablePointer<Float>.allocate(capacity: count)
        let dstPtr: UnsafeMutablePointer<Float> = (try? AlignedMemory.allocateAligned(type: Float.self, count: count, alignment: alignment)) ?? UnsafeMutablePointer<Float>.allocate(capacity: count)
        srcPtr.initialize(repeating: 0.5, count: count)
        dstPtr.initialize(repeating: 0.0, count: count)

        defer {
            srcPtr.deinitialize(count: count)
            dstPtr.deinitialize(count: count)
            srcPtr.deallocate()
            dstPtr.deallocate()
        }

        // MARK: - Copy throughput
        do {
            let label = "mem.copy.\(dim)"
            if Filters.shouldRun(name: label, options: options) {

            let caseStart = Date()
            progress.caseStarted(suite: Self.name, name: label, index: currentIndex, total: totalCases)

            Harness.warmup {
                alignedCopy(from: UnsafePointer(srcPtr), to: dstPtr, count: count, preferredAlignment: alignment)
                blackHole(dstPtr[0])
            }
            let r = Harness.measure(name: label, minTimeSeconds: options.minTimeSeconds, repeats: options.repeats, unitCount: count, samples: options.samples) {
                alignedCopy(from: UnsafePointer(srcPtr), to: dstPtr, count: count, preferredAlignment: alignment)
                blackHole(dstPtr[1])
            }
            out.append(r)

            let caseDuration = Date().timeIntervalSince(caseStart) * 1000.0
            progress.caseCompleted(suite: Self.name, name: label, index: currentIndex, total: totalCases, durationMs: caseDuration)
            currentIndex += 1
            }
        }

        // MARK: - Fill throughput
        do {
            let label = "mem.fill.\(dim)"
            if Filters.shouldRun(name: label, options: options) {

            let caseStart = Date()
            progress.caseStarted(suite: Self.name, name: label, index: currentIndex, total: totalCases)

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

            let caseDuration = Date().timeIntervalSince(caseStart) * 1000.0
            progress.caseCompleted(suite: Self.name, name: label, index: currentIndex, total: totalCases, durationMs: caseDuration)
            currentIndex += 1
            }
        }

        // MARK: - Pool acquire/return overhead
        do {
            let label = "mem.pool.acquire.\(dim)"
            if Filters.shouldRun(name: label, options: options) {

            let caseStart = Date()
            progress.caseStarted(suite: Self.name, name: label, index: currentIndex, total: totalCases)

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

            let caseDuration = Date().timeIntervalSince(caseStart) * 1000.0
            progress.caseCompleted(suite: Self.name, name: label, index: currentIndex, total: totalCases, durationMs: caseDuration)
            currentIndex += 1
            }
        }

        return out
    }
}
