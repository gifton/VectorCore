import XCTest
import Foundation
import Darwin
@testable import VectorCore

private enum E01Sentinel: Error { case stop }
private final class E01Recorder: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Int] = []
    func append(_ value: Int) { lock.lock(); defer { lock.unlock() }; values.append(value) }
    func snapshot() -> [Int] { lock.lock(); defer { lock.unlock() }; return values }
}
private struct E01Defaults: ComputeProvider {
    let threads: Int
    var device: ComputeDevice { .cpu }
    var maxConcurrency: Int { threads }
    var deviceInfo: ComputeDeviceInfo {
        ComputeDeviceInfo(name: "Boundary defaults", maxThreads: threads, preferredChunkSize: 1)
    }
    func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T {
        try await work()
    }
}

final class ExecutionBoundaryTests: XCTestCase, @unchecked Sendable {
    private var providers: [any ComputeProvider] {
        [CPUComputeProvider(mode: .sequential), CPUComputeProvider(mode: .parallel),
         CPUComputeProvider(mode: .automatic, parallelizationThreshold: 1),
         CPUComputeProvider(mode: .automatic, parallelizationThreshold: 3)] +
        [0, -1, 1, Int.max].map { E01Defaults(threads: $0) as any ComputeProvider }
    }

    private func invoke(_ p: any ComputeProvider, method: Int, range: Range<Int>,
                        hint: Int, recorder: E01Recorder) async throws -> [Int] {
        switch method {
        case 0: return try await p.parallelExecute(items: range) { recorder.append($0); return $0 }
        case 1:
            try await p.parallelForEach(items: range) { recorder.append($0) }
            return recorder.snapshot().sorted()
        case 2:
            try await p.parallelForEach(items: range, minChunk: hint) { recorder.append($0) }
            return recorder.snapshot().sorted()
        case 3:
            return try await p.parallelReduce(items: range, initial: [Int](), { r in
                recorder.append(Int.min + 101)
                for i in r { recorder.append(i) }; return Array(r)
            }, { a, b in recorder.append(Int.min + 100); return a + b }).sorted()
        default:
            return try await p.parallelReduce(items: range, initial: [Int](), minChunk: hint, { r in
                recorder.append(Int.min + 101)
                for i in r { recorder.append(i) }; return Array(r)
            }, { a, b in recorder.append(Int.min + 100); return a + b }).sorted()
        }
    }

    @MainActor
    func testNegativeBatchRejectedBeforeDroppedWork() async throws {
        for n in [0, 1, 999, 1000] {
            let vectors = (0..<n).map { _ in Vector<Dim2>.zero }
            let calls = E01Recorder()
            do {
                let _: [Int] = try await BatchOperations.process(vectors, batchSize: -1) { _ in
                    calls.append(1); return [1]
                }
                XCTFail("negative batchSize must throw for N=\(n)")
            } catch let e as VectorError { XCTAssertEqual(e.kind, .invalidConfiguration) }
            XCTAssertEqual(calls.snapshot(), [])
        }
    }

    func testArbitraryLoopWidthsExactCoverage() {
        for count in 0...40 {
            for width in 1...17 {
                var indices: [Int] = []
                unrolledLoop(count: count, by: width) { indices.append($0) }
                XCTAssertEqual(indices, Array(0..<count), "count=\(count), width=\(width)")
                var ranges: [Range<Int>] = []
                vectorizedLoop(count: count, vectorWidth: width) { ranges.append($0) }
                XCTAssertTrue(ranges.allSatisfy { !$0.isEmpty })
                XCTAssertEqual(ranges.flatMap { Array($0) }, Array(0..<count))
            }
        }
    }

    func testLoopErrorsStopAtFirstInteriorAndTail() {
        for width in 1...17 {
            for stop in [0, 17, 39] {
                var seen: [Int] = []
                XCTAssertThrowsError(try unrolledLoop(count: 40, by: width) { i in
                    seen.append(i); if i == stop { throw E01Sentinel.stop }
                }) { XCTAssertTrue($0 is E01Sentinel) }
                XCTAssertEqual(seen, Array(0...stop))
                var seenRanges: [Range<Int>] = []
                XCTAssertThrowsError(try vectorizedLoop(count: 40, vectorWidth: width) { r in
                    seenRanges.append(r); if r.contains(stop) { throw E01Sentinel.stop }
                }) { XCTAssertTrue($0 is E01Sentinel) }
                XCTAssertTrue(seenRanges.last?.contains(stop) == true)
                XCTAssertEqual(seenRanges.flatMap { Array($0) }, Array(0..<(seenRanges.last!.upperBound)))
            }
        }
    }

    func testHintValidationBeforeEmptySequentialParallelAndNestedRouting() throws {
        try runChild("hints", expectsTrap: false)
    }

    @MainActor
    private func checkHints() async throws {
        for p in providers {
            for method in [2, 4] {
                for hint in [0, -1] {
                    for range in [0..<0, 0..<2] {
                        for nested in [false, true] {
                            let calls = E01Recorder()
                            do {
                                _ = try await CPUComputeProvider.$isInsideParallelRegion.withValue(nested) {
                                    try await invoke(p, method: method, range: range, hint: hint, recorder: calls)
                                }
                                XCTFail("invalid minChunk \(hint), method \(method), nested \(nested)")
                            } catch let e as VectorError { XCTAssertEqual(e.kind, .invalidConfiguration) }
                            XCTAssertEqual(calls.snapshot(), [])
                        }
                    }
                }
            }
        }
    }

    // Each dangerous family runs in a fresh child with a deadline; baseline traps cannot kill the suite.
    func testLargeRepresentableReductionLengthsInChild() throws {
        try runChild("largeLengths", expectsTrap: false)
    }

    func testDangerousBoundariesInChildren() throws {
        for mode in ["batch", "ranges", "unrepresentable", "hugeLoops"] {
            try runChild(mode, expectsTrap: false)
        }
        for loop in ["unrolled", "vectorized"] {
            for pair in ["0:0", "0:-1", "1:0", "1:-1", "-1:1"] {
                try runChild("domain:\(loop):\(pair)", expectsTrap: true)
            }
        }
    }

    private func runChild(_ mode: String, expectsTrap: Bool) throws {
        var env = ProcessInfo.processInfo.environment
        env["E01_CHILD"] = mode
        let output = FileManager.default.temporaryDirectory.appendingPathComponent("E01-\(UUID().uuidString).log")
        _ = FileManager.default.createFile(atPath: output.path, contents: nil)
        let handle = try FileHandle(forWritingTo: output)
        defer { try? handle.close() }
        let process = try launchChildProbe(.xctest(selector: "ComprehensiveTests.ExecutionBoundaryTests/testChildProbe",
                                                   bundlePath: Bundle(for: Self.self).bundlePath),
                                           environment: env, output: handle, deadline: 5, cleanup: .kill(seconds: 2))
        if process.timedOut {
            throw NSError(domain: "ExecutionBoundaryTests.childTimeout", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "\(mode): child deadline; cleanup=\(process.cleanup); log=\(output.path)"])
        }
        try handle.close()
        let log = try String(contentsOf: output, encoding: .utf8)
        if expectsTrap {
            XCTAssertEqual(process.terminationReason, .uncaughtSignal, "\(mode): \(log.prefix(1000))")
            XCTAssertFalse(log.contains("E01_BODY_REACHED"), "validation must precede body: \(mode)")
            #if DEBUG
            let expected = mode.hasSuffix("-1:1") ? "count must be nonnegative" : "width must be positive"
            XCTAssertTrue(log.contains(expected), "explicit domain rejection: \(mode): \(log.prefix(1000))")
            #endif
        } else {
            XCTAssertEqual(process.terminationStatus, 0, "\(mode): \(log.prefix(2000))")
        }
        try FileManager.default.removeItem(at: output)
    }

    @MainActor
    private func checkBatch() async throws {
        try await BatchOperations.updateConfiguration { $0.defaultBatchSize = 3 }
        for n in [0, 1, 999, 1000] {
            let vectors = try (0..<n).map { try Vector<Dim2>([Float($0), 0]) }
            for size in [0, -1, 1, n + 1, Int.max] {
                let calls = E01Recorder()
                do {
                    let actual: [Int] = try await BatchOperations.process(vectors, batchSize: size) { batch in
                        batch.flatMap { v in let i = Int(v[0]); calls.append(i); return [i, -i] }
                    }
                    XCTAssertGreaterThan(size, 0)
                    XCTAssertEqual(actual, (0..<n).flatMap { [$0, -$0] })
                    XCTAssertEqual(calls.snapshot().sorted(), Array(0..<n))
                } catch let e as VectorError {
                    XCTAssertLessThanOrEqual(size, 0); XCTAssertEqual(e.kind, .invalidConfiguration)
                    XCTAssertTrue(e.description.contains("batchSize")); XCTAssertEqual(calls.snapshot(), [])
                }
            }
            let nilCalls = E01Recorder()
            let nilBatchSizes = E01Recorder()
            let nilResult: [Int] = try await BatchOperations.process(vectors) { b in
                nilBatchSizes.append(b.count)
                return b.map { let i = Int($0[0]); nilCalls.append(i); return i }
            }
            XCTAssertEqual(nilBatchSizes.snapshot().sorted(), (0..<n).enumerated().filter { $0.offset % 3 == 0 }.map { min(3, n - $0.offset) }.sorted())
            XCTAssertEqual(nilResult, Array(0..<n)); XCTAssertEqual(nilCalls.snapshot().sorted(), Array(0..<n))
            if n > 0 {
                do {
                    let _: [Int] = try await BatchOperations.process(vectors, batchSize: 1) { _ in throw E01Sentinel.stop }
                    XCTFail("transform error swallowed")
                } catch { XCTAssertTrue(error is E01Sentinel) }
            }
        }
    }

    @MainActor
    func testChildProbe() async throws {
        guard let mode = ProcessInfo.processInfo.environment["E01_CHILD"] else { return }
        if mode.hasPrefix("domain:") {
            let parts = mode.split(separator: ":")
            let count = Int(parts[2])!, width = Int(parts[3])!
            if parts[1] == "unrolled" {
                try unrolledLoop(count: count, by: width) { _ in
                    FileHandle.standardError.write(Data("E01_BODY_REACHED\n".utf8)); throw E01Sentinel.stop
                }
            } else {
                try vectorizedLoop(count: count, vectorWidth: width) { _ in
                    FileHandle.standardError.write(Data("E01_BODY_REACHED\n".utf8)); throw E01Sentinel.stop
                }
            }
            XCTFail("invalid loop domain returned")
        } else if mode == "hints" {
            try await checkHints()
        } else if mode == "largeLengths" {
            let boundedProviders: [any ComputeProvider] = [
                CPUComputeProvider(mode: .sequential), CPUComputeProvider(mode: .parallel),
                CPUComputeProvider(mode: .automatic, parallelizationThreshold: 1),
                E01Defaults(threads: 1)
            ]
            for p in boundedProviders {
                for range in [0..<Int.max, Int.min ..< -1] {
                    let count = try await p.parallelReduce(items: range, initial: 0, { $0.count }, { $0 + $1 })
                    XCTAssertEqual(count, Int.max)
                    let hinted = try await p.parallelReduce(items: range, initial: 0, minChunk: Int.max, { $0.count }, { $0 + $1 })
                    XCTAssertEqual(hinted, Int.max)
                }
            }
            for limit in [0, -1] {
                let provider = E01Defaults(threads: limit)
                for range in [0..<0, 0..<Int.max, Int.min ..< -1] {
                    for method in 0..<5 {
                        let calls = E01Recorder()
                        do {
                            _ = try await invoke(provider, method: method, range: range, hint: Int.max, recorder: calls)
                            XCTFail("nonpositive concurrency accepted")
                        } catch let error as VectorError { XCTAssertEqual(error.kind, .invalidConfiguration) }
                        XCTAssertEqual(calls.snapshot(), [])
                    }
                }
            }
        } else if mode == "hugeLoops" {
            var ranges: [Range<Int>] = []
            vectorizedLoop(count: Int.max, vectorWidth: Int.max) { ranges.append($0) }
            XCTAssertEqual(ranges, [0..<Int.max])
            XCTAssertThrowsError(try unrolledLoop(count: Int.max, by: Int.max) { _ in throw E01Sentinel.stop }) {
                XCTAssertTrue($0 is E01Sentinel)
            }
        } else if mode == "batch" {
            // The restore helper is nonisolated; a @Sendable closure hops back to the main actor so
            // the assertions run in checkBatch(), which is main-actor isolated.
            try await e06WithRestoredConfiguration { @Sendable in try await self.checkBatch() }
        } else {
            let ranges: [Range<Int>] = mode == "unrepresentable" ? [Int.min..<Int.max] :
                [0..<0, Int.min..<Int.min, Int.max..<Int.max, 0..<2, 1..<3, -3 ..< -1,
                 (Int.max - 2)..<Int.max, Int.min..<(Int.min + 2)]
            for p in providers {
                for method in 0..<5 {
                    for hint in [1, 256, Int.max] {
                        for range in ranges {
                            let calls = E01Recorder()
                            do {
                                let result = try await invoke(p, method: method, range: range, hint: hint, recorder: calls)
                                XCTAssertGreaterThan(p.maxConcurrency, 0)
                                XCTAssertNotEqual(mode, "unrepresentable")
                                XCTAssertEqual(result, Array(range))
                                let visits = calls.snapshot().filter { $0 != Int.min + 100 && $0 != Int.min + 101 }.sorted()
                                XCTAssertEqual(visits, Array(range))
                                if range.isEmpty { XCTAssertEqual(calls.snapshot(), []) }
                            } catch let e as VectorError {
                                if mode == "unrepresentable" { XCTAssertEqual(e.kind, .invalidRange) }
                                else { XCTAssertLessThanOrEqual(p.maxConcurrency, 0); XCTAssertEqual(e.kind, .invalidConfiguration) }
                                XCTAssertEqual(calls.snapshot(), [])
                            }
                        }
                    }
                }
            }
        }
    }
}
