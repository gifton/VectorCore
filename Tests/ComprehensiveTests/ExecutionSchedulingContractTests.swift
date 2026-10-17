import Testing
import Foundation
import Darwin
@testable import VectorCore

private enum E02Failure: Error { case factory, body }
private struct E02Defaults: ComputeProvider {
    let maxConcurrency: Int
    var device: ComputeDevice { .cpu }
    var deviceInfo: ComputeDeviceInfo { .init(name: "E02", maxThreads: 99, preferredChunkSize: 1) }
    func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T { try await work() }
}
private final class E02Recorder: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Int] = []
    func append(_ value: Int) { lock.lock(); defer { lock.unlock() }; values.append(value) }
    func snapshot() -> [Int] { lock.lock(); defer { lock.unlock() }; return values }
}
private actor E02Gate {
    var entered = 0
    var peak = 0
    var active = 0
    var open = false
    var waiters: [CheckedContinuation<Void, Never>] = []
    func body() async {
        entered += 1; active += 1; peak = max(peak, active)
        if !open { await withCheckedContinuation { waiters.append($0) } }
        active -= 1
    }
    func release() { open = true; let pending = waiters; waiters = []; for w in pending { w.resume() } }
}
private struct E02RejectingVector: VectorProtocol, VectorFactory {
    var storage: [Float]
    var scalarCount: Int { storage.count }
    subscript(index: Int) -> Float { storage[index] }
    init() { storage = [1] }
    init(_ array: [Float]) { storage = array }
    init(repeating value: Float) { storage = [value] }
    func toArray() -> [Float] { storage }
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R { try storage.withUnsafeBufferPointer(body) }
    mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R { try storage.withUnsafeMutableBufferPointer { try body($0) } }
    static func create(from scalars: [Float]) throws -> Self { throw E02Failure.factory }
}
private struct E02SpyMetric: DistanceMetric {
    var name: String { "E02 spy" }
    let calls: E02Recorder
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float { calls.append(1); return 0 }
}

@Suite struct ExecutionSchedulingContractTests {
    @Test(arguments: ["defaultBound", "cancelSerial", "factory1", "factory1000", "shape", "batchControl", "gatedCancellation", "shortBatch", "reverseAndError"])
    func boundedRegressionChild(scenario: String) throws {
        var env = ProcessInfo.processInfo.environment
        env["E02_CHILD"] = scenario; env["SWIFT_BACKTRACE"] = "enable=no"
        let directory = env["E02_CAPTURE_DIRECTORY"] ?? FileManager.default.temporaryDirectory.path
        let url = URL(fileURLWithPath: directory).appendingPathComponent("E02-\(scenario)-\(UUID().uuidString).log")
        _ = FileManager.default.createFile(atPath: url.path, contents: nil)
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        let process = try launchChildProbe(.swiftTesting(filter: "ExecutionSchedulingContractTests/childProbe"),
                                           environment: env, output: handle, deadline: 15,
                                           cleanup: .terminateThenKill(grace: 1, seconds: 1))
        if process.timedOut {
            let cleanup = process.cleanup
            try handle.close()
            print("E02_CHILD_CAPTURE \(scenario) \(url.path)")
            print("E02_CHILD_TIMEOUT \(scenario) cleanupReaped=\(cleanup == .success) path=\(url.path)")
            // No termination property is read after a deadline, even if cleanup times out.
            throw NSError(domain: "E02.childTimeout", code: cleanup == .success ? 1 : 2,
                          userInfo: [NSLocalizedDescriptionKey: "\(scenario): child deadline; capture=\(url.path)"])
        }
        try handle.close()
        let output = try String(contentsOf: url, encoding: .utf8)
        print("E02_CHILD_CAPTURE \(scenario) \(url.path)")
        print("E02_CHILD_RESULT \(scenario) status=\(process.statusDescription) reason=\(process.reasonDescription) path=\(url.path)")
        if scenario == "shortBatch" {
            #expect(process.terminationReason == .uncaughtSignal)
            #expect(output.contains("executeBatch output capacity must cover candidates"))
            #expect(!output.contains("E02_KERNEL_REACHED"))
        } else {
            #expect(process.terminationStatus == 0, "\(scenario): \(output.prefix(3000))")
            #expect(output.contains("E02_COMPLETE \(scenario)"))
        }
    }

    @Test func childProbe() async throws {
        guard let scenario = ProcessInfo.processInfo.environment["E02_CHILD"] else { return }
        let data = try JSONSerialization.data(withJSONObject: ["argv": CommandLine.arguments, "scenario": scenario,
            "DYLD_FRAMEWORK_PATH": ProcessInfo.processInfo.environment["DYLD_FRAMEWORK_PATH"] ?? "",
            "DYLD_LIBRARY_PATH": ProcessInfo.processInfo.environment["DYLD_LIBRARY_PATH"] ?? ""], options: [.sortedKeys])
        FileHandle.standardOutput.write(Data("E02_INVOCATION \(String(decoding: data, as: UTF8.self))\nE02_BODY \(scenario)\n".utf8))
        switch scenario {
        case "watchdog":
            signal(SIGTERM, SIG_IGN)
            // Self-bound so a crashed parent cannot leave an orphan: exit when re-parented to
            // launchd (getppid() == 1) (unverified by an automated test; manual orphan check in
            // T04/fix/orphan2.out), or after a 120 s ceiling, well above the parent's deadline
            // (parent: 15 s wait, then SIGTERM + 1 s and SIGKILL + 1 s to the child's process group,
            // launchChildProbe(cleanup: .terminateThenKill(grace: 1, seconds: 1)) in boundedRegressionChild).
            let ceiling = ContinuousClock.now + .seconds(120)
            while getppid() != 1 && ContinuousClock.now < ceiling { try? await Task.sleep(for: .milliseconds(250)) }
            exit(3)
        case "reverseAndError":
            try await reverseCompletionReductionAndErrorCleanup()
        case "shortBatch":
            var output = [Int](repeating: -1, count: 1)
            output.withUnsafeMutableBufferPointer { out in
                CPUComputeProvider.parallel.executeBatch(query: 0, candidates: [1,2], kernelKind: .euclid, dimension: 1, results: out) { _,_,_,_ in
                    FileHandle.standardError.write(Data("E02_KERNEL_REACHED\n".utf8))
                }
            }
        case "gatedCancellation":
            await gatedCancellationJoinsIgnoringBodies()
        case "defaultBound":
            let gate = E02Gate()
            let task = Task { try await E02Defaults(maxConcurrency: 1).parallelExecute(items: 0..<16) { i in await gate.body(); return i } }
            while await gate.entered == 0 { await Task.yield() }
            for _ in 0..<100 { await Task.yield() }
            await gate.release()
            #expect(try await task.value == Array(0..<16))
            #expect(await gate.peak <= 1)
        case "cancelSerial":
            let calls = E02Recorder()
            let task = Task {
                withUnsafeCurrentTask { $0?.cancel() }
                do { _ = try await CPUComputeProvider.sequential.parallelExecute(items: 0..<3) { i in calls.append(i); return i }; Issue.record("Cancelled operation returned success") }
                catch { #expect(error is CancellationError) }
            }
            await task.value
            #expect(calls.snapshot().isEmpty)
        case "factory1", "factory1000":
            let n = scenario == "factory1" ? 1 : 1000
            do { _ = try await Array(repeating: E02RejectingVector(), count: n).mapElements { $0 + 1 }; Issue.record("factory accepted") }
            catch { #expect(error is E02Failure) }
        case "shape":
            let calls = E02Recorder()
            var rows = Array(repeating: DynamicVector([1,2]), count: 101)
            rows[1] = DynamicVector([1])
            do { _ = try await ExecutionOperations.distanceMatrix(rows, metric: E02SpyMetric(calls: calls)); Issue.record("shape accepted") }
            catch let error as VectorError { #expect(error.kind == .dimensionMismatch) }
            #expect(calls.snapshot().isEmpty)
        case "batchControl":
            for mode in [CPUComputeProvider.Mode.sequential, .parallel, .automatic] {
                let calls = E02Recorder(); var output = [Int](repeating: -1, count: 6)
                output.withUnsafeMutableBufferPointer { out in
                    CPUComputeProvider(mode: mode, parallelizationThreshold: 0, minChunk: 1).executeBatch(query: 0, candidates: [1,2,3], kernelKind: .euclid, dimension: 1,
                        results: out) { _, _, range, result in
                        calls.append(range.lowerBound); calls.append(range.upperBound)
                        for i in range { result[i] = i }
                    }
                }
                #expect(calls.snapshot() == [0,3]); #expect(output == [0,1,2,-1,-1,-1])
            }
        default: Issue.record("unknown child")
        }
        print("E02_COMPLETE \(scenario)")
    }
}

private final class E02Events: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [BoundedExecution.Event] = []
    func record(_ event: BoundedExecution.Event) { lock.lock(); defer { lock.unlock() }; events.append(event) }
    func snapshot() -> [BoundedExecution.Event] { lock.lock(); defer { lock.unlock() }; return events }
    func verify(maximum: Int? = nil) {
        let groups = Dictionary(grouping: snapshot(), by: \.id)
        for (_, events) in groups {
            let submissions = events.filter { $0.kind == .submitted }
            let completions = events.filter { $0.kind == .completed }
            #expect(events.first?.kind == .begin && events.last?.kind == .end)
            #expect(submissions.count <= (maximum ?? events[0].limit))
            #expect(submissions.map(\.worker).sorted() == completions.map(\.worker).sorted())
            #expect(Set(submissions.map(\.worker)).count == submissions.count)
        }
    }
}

private func e02Invoke(_ provider: any ComputeProvider, method: Int, range: Range<Int>, hint: Int,
                       calls: E02Recorder) async throws -> [Int] {
    switch method {
    case 0: return try await provider.parallelExecute(items: range) { calls.append($0); await Task.yield(); return $0 }
    case 1:
        try await provider.parallelForEach(items: range) { calls.append($0); await Task.yield() }
        return calls.snapshot().sorted()
    case 2:
        try await provider.parallelForEach(items: range, minChunk: hint) { calls.append($0); await Task.yield() }
        return calls.snapshot().sorted()
    case 3:
        return try await provider.parallelReduce(items: range, initial: [Int](), { range in
            for i in range { calls.append(i) }
            // Reverse completion pressure without changing the required merge order.
            if range.lowerBound % 2 == 0 { for _ in 0..<20 { await Task.yield() } }
            return Array(range)
        }, +)
    default:
        return try await provider.parallelReduce(items: range, initial: [Int](), minChunk: hint, { range in
            for i in range { calls.append(i) }; await Task.yield(); return Array(range)
        }, +)
    }
}

extension ExecutionSchedulingContractTests {
    @Test func providerOrderedCoverageAndWorkerBounds() async throws {
        let providers: [any ComputeProvider] = [1,2,4,Int.max].map { E02Defaults(maxConcurrency: $0) } +
            [CPUComputeProvider.sequential, CPUComputeProvider.parallel,
             CPUComputeProvider(mode: .automatic, parallelizationThreshold: 3)]
        let ranges = [0..<0, 0..<1, -7..<4, 0..<2, 0..<3, 0..<4, 0..<1025,
                      Int.min..<(Int.min + 3), (Int.max - 3)..<Int.max]
        for provider in providers {
            for method in 0..<5 {
                for range in ranges {
                    for hint in [1,3,Int.max] {
                        let calls = E02Recorder(), events = E02Events()
                        let actual = try await BoundedExecution.$observation.withValue({ events.record($0) }) {
                            try await e02Invoke(provider, method: method, range: range, hint: hint, calls: calls)
                        }
                        #expect(actual == Array(range))
                        #expect(calls.snapshot().sorted() == Array(range))
                        events.verify(maximum: min(provider.maxConcurrency, range.count))
                        if range.isEmpty { #expect(events.snapshot().allSatisfy { $0.kind != .submitted }) }
                    }
                }
            }
        }
    }

    @Test func invalidLimitsAndHintsPrecedeCallbacks() async {
        for limit in [0,-1] {
            for range in [0..<0, 0..<3, Int.min..<Int.max] {
                for method in 0..<5 {
                    let calls = E02Recorder(), events = E02Events()
                    do {
                        _ = try await BoundedExecution.$observation.withValue({ events.record($0) }) {
                            try await e02Invoke(E02Defaults(maxConcurrency: limit), method: method, range: range, hint: 1, calls: calls)
                        }
                        Issue.record("Invalid concurrency accepted")
                    } catch let error as VectorError {
                        #expect(error.kind == (range.lowerBound == Int.min ? .invalidRange : .invalidConfiguration))
                    } catch { Issue.record("Wrong error: \(error)") }
                    #expect(calls.snapshot().isEmpty); #expect(events.snapshot().isEmpty)
                }
            }
        }
        for provider: any ComputeProvider in [E02Defaults(maxConcurrency: 2), CPUComputeProvider.sequential, CPUComputeProvider.parallel] {
            for hint in [0,-1] {
                for range in [0..<0, 0..<3] {
                    for method in [2,4] {
                        let calls = E02Recorder()
                        do { _ = try await e02Invoke(provider, method: method, range: range, hint: hint, calls: calls); Issue.record("hint accepted") }
                        catch let e as VectorError { #expect(e.kind == .invalidConfiguration) }
                        catch { Issue.record("Wrong error") }
                        #expect(calls.snapshot().isEmpty)
                    }
                }
            }
        }
    }

    @Test func reductionIdentityAndOrderedAssociativity() async throws {
        for provider: any ComputeProvider in [E02Defaults(maxConcurrency: 4), CPUComputeProvider.sequential, CPUComputeProvider.parallel] {
            for count in [0,1,1025] {
                let calls = E02Recorder(), combines = E02Recorder()
                let result = try await provider.parallelReduce(items: 0..<count, initial: "", { range in
                    calls.append(range.count)
                    if range.lowerBound == 0 { for _ in 0..<100 { await Task.yield() } }
                    return range.map { "\($0)," }.joined()
                }, { a,b in combines.append(1); return a+b })
                #expect(result == (0..<count).map { "\($0)," }.joined())
                #expect(calls.snapshot().reduce(0,+) == count)
                #expect(combines.snapshot().count == calls.snapshot().count)
                if count == 0 { #expect(calls.snapshot().isEmpty && combines.snapshot().isEmpty) }
                let sum = try await provider.parallelReduce(items: 0..<count, initial: 0, minChunk: 3,
                    { $0.reduce(0,+) }, +)
                #expect(sum == count * (count - 1) / 2)
            }
        }
    }

    @Test func preCancelledThrowingRoutesAndExecute() async {
        for provider: any ComputeProvider in [E02Defaults(maxConcurrency: 2), CPUComputeProvider.sequential, CPUComputeProvider.parallel] {
            for count in [0,1,1025] {
                for method in 0..<5 {
                    let calls = E02Recorder(), events = E02Events()
                    await Task {
                        withUnsafeCurrentTask { $0?.cancel() }
                        do {
                            _ = try await BoundedExecution.$observation.withValue({ events.record($0) }) {
                                try await e02Invoke(provider, method: method, range: 0..<count, hint: 1, calls: calls)
                            }
                            Issue.record("Cancelled success")
                        } catch { #expect(error is CancellationError) }
                    }.value
                    #expect(calls.snapshot().isEmpty && events.snapshot().isEmpty)
                }
            }
        }
        let calls = E02Recorder()
        await Task {
            withUnsafeCurrentTask { $0?.cancel() }
            do { _ = try await CPUComputeProvider.sequential.execute { calls.append(1); return 1 }; Issue.record("Cancelled execute") }
            catch { #expect(error is CancellationError) }
        }.value
        #expect(calls.snapshot().isEmpty)
        // The body cancels its own task, so run it in a child Task: cancelling the test's task would make
        // Swift Testing 6.4 report the whole test as cancelled instead of passed.
        await Task {
            do {
                _ = try await CPUComputeProvider.sequential.execute { withUnsafeCurrentTask { $0?.cancel() }; return 1 }
                Issue.record("Missing execute post-check")
            } catch { #expect(error is CancellationError) }
        }.value
    }

    func gatedCancellationJoinsIgnoringBodies() async {
        for limit in [1,2,4] {
            let gate = E02Gate(), calls = E02Recorder(), events = E02Events()
            let task = Task {
                try await BoundedExecution.$observation.withValue({ events.record($0) }) {
                    try await E02Defaults(maxConcurrency: limit).parallelExecute(items: 0..<16) { i in
                        calls.append(i); await gate.body(); return i
                    }
                }
            }
            while await gate.entered < limit { await Task.yield() }
            task.cancel()
            // Bodies deliberately ignore cancellation while gated; cleanup requires release.
            await gate.release()
            do { _ = try await task.value; Issue.record("cancelled success") }
            catch { #expect(error is CancellationError) }
            #expect(calls.snapshot().count == limit)
            #expect(await gate.active == 0)
            events.verify(maximum: limit)
        }
    }

    @Test func cancellationBetweenUnitsAndOriginalBodyError() async {
        for provider: any ComputeProvider in [E02Defaults(maxConcurrency: 1), CPUComputeProvider.sequential, CPUComputeProvider.parallel] {
            let calls = E02Recorder(), events = E02Events()
            await Task {
            do {
                _ = try await BoundedExecution.$observation.withValue({ events.record($0) }) {
                    try await provider.parallelExecute(items: 0..<2048) { index in
                        calls.append(index)
                        withUnsafeCurrentTask { $0?.cancel() }
                        return index
                    }
                }
                Issue.record("cancellation ignored")
            } catch { #expect(error is CancellationError) }
            #expect(calls.snapshot().count <= provider.maxConcurrency)
            events.verify()
            }.value
        }
        let events = E02Events()
        do {
            _ = try await BoundedExecution.$observation.withValue({ events.record($0) }) {
                try await E02Defaults(maxConcurrency: 2).parallelExecute(items: 0..<100) { _ -> Int in throw E02Failure.body }
            }
            Issue.record("body failure swallowed")
        } catch { if case E02Failure.body = error {} else { Issue.record("Original body error lost") } }
        events.verify(maximum: 2)
    }

    @Test func nestedProvidersAndLegacySuppression() async throws {
        for outer: any ComputeProvider in [E02Defaults(maxConcurrency: 2), CPUComputeProvider.parallel, CPUComputeProvider.sequential] {
            let events = E02Events()
            let result = try await BoundedExecution.$observation.withValue({ events.record($0) }) {
                try await outer.parallelExecute(items: 0..<4) { index in
                    let inner = E02Events()
                    let values = try await BoundedExecution.$observation.withValue({ inner.record($0) }) {
                        let a = try await CPUComputeProvider.parallel.parallelExecute(items: 0..<1025) { $0 }
                        let b = try await E02Defaults(maxConcurrency: 4).parallelExecute(items: 0..<3) { $0 }
                        let matrix = await TriangularMatrix<Int>.buildParallel(size: 3) { $0+$1 }
                        #expect(matrix[0,2] == 2)
                        return a.count+b.count
                    }
                    #expect(inner.snapshot().allSatisfy { $0.kind != .submitted })
                    return index+values
                }
            }
            #expect(result == [1028,1029,1030,1031]); events.verify()
        }
        let events = E02Events()
        _ = try await BoundedExecution.$observation.withValue({ events.record($0) }) {
            try await CPUComputeProvider.$isInsideParallelRegion.withValue(true) {
                try await E02Defaults(maxConcurrency: 4).parallelExecute(items: 0..<7) { $0 }
            }
        }
        #expect(events.snapshot().allSatisfy { $0.kind != .submitted })
    }
}

private struct E02Context: ExecutionContext {
    let maxThreadCount: Int
    let preferredChunkSize: Int
    var device: ComputeDevice { .cpu }
    func execute<T: Sendable>(_ work: @Sendable @escaping () throws -> T) async throws -> T {
        Issue.record("Retained loops must not cross the context executor"); return try work()
    }
}
private struct E02BatchProvider: BatchKernelProvider {
    let maxConcurrency: Int
    let calls: E02Recorder
    var device: ComputeDevice { .cpu }
    var deviceInfo: ComputeDeviceInfo { .init(name: "E02 batch", maxThreads: 77, preferredChunkSize: 1) }
    func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T { try await work() }
    func batchDistance<V: VectorProtocol>(query: V, candidates: [V], metric: any DistanceMetric) async throws -> [Float] where V.Scalar == Float { candidates.map { _ in 0 } }
    func findNearest<V: VectorProtocol>(query: V, candidates: [V], k: Int, metric: any DistanceMetric) async throws -> [(index: Int, distance: Float)] where V.Scalar == Float {
        let index = Int(query[0]); calls.append(index); await Task.yield(); return [(index, Float(index))]
    }
}

extension ExecutionSchedulingContractTests {
    @Test func batchTransformsOrderingAndCancellation() async throws {
        for count in [0,1,999,1000,1001] {
            let rows = (0..<count).map { DynamicVector([Float($0)]) }
            let events = E02Events()
            try await BoundedExecution.$observation.withValue({ events.record($0) }) {
                let mapped = try await BatchOperations.map(rows) { DynamicVector([$0[0]+1]) }
                #expect(mapped.map { Int($0[0]) } == Array(1..<(count+1)))
                let filtered = try await BatchOperations.filter(rows) { Int($0[0]) % 2 == 0 }
                #expect(filtered.map { Int($0[0]) } == Array(0..<count).filter { $0 % 2 == 0 })
                for batchSize in [1,3,Int.max] {
                    let processed: [Int] = try await BatchOperations.process(rows, batchSize: batchSize) { batch in
                        #expect(BoundedExecution.isInsideRegion)
                        return batch.flatMap { [Int($0[0]), -Int($0[0])] }
                    }
                    #expect(processed == (0..<count).flatMap { [$0,-$0] })
                }
            }
            events.verify()
            for route in 0..<3 {
                let calls = E02Recorder()
                await Task {
                    withUnsafeCurrentTask { $0?.cancel() }
                    do {
                        switch route {
                        case 0: _ = try await BatchOperations.map(rows) { calls.append(1); return $0 }
                        case 1: _ = try await BatchOperations.filter(rows) { _ in calls.append(1); return true }
                        default: let _: [Int] = try await BatchOperations.process(rows, batchSize: 1) { _ in calls.append(1); return [1] }
                        }
                        Issue.record("Cancelled batch success")
                    } catch { #expect(error is CancellationError) }
                }.value
                #expect(calls.snapshot().isEmpty)
            }
        }
    }

    @Test func liveTransformsContextHintsAndFactoryIdentity() async throws {
        for count in [0,1,999,1000,1001] {
            let rows = (0..<count).map { DynamicVector($0 % 2 == 0 ? [3,4] : [2]) }
            let context = E02Context(maxThreadCount: 2, preferredChunkSize: 3)
            let events = E02Events()
            try await BoundedExecution.$observation.withValue({ events.record($0) }) {
                let mapped = try await ExecutionOperations.map(rows, transform: { $0+1 }, context: context)
                #expect(mapped.map { $0.toArray() } == rows.map { $0.toArray().map { $0+1 } })
                let normalized = try await ExecutionOperations.normalize(rows, context: context)
                for (i,row) in normalized.enumerated() { #expect(row.toArray() == (i % 2 == 0 ? [0.6,0.8] : [1])) }
                #expect(try await rows.adding(1) == mapped)
                #expect(try await rows.multiplying(by: 2) == rows.map { DynamicVector($0.toArray().map { $0*2 }) })
            }
            events.verify(maximum: BoundedExecution.hardwareLimit)
            for context in [E02Context(maxThreadCount: 0, preferredChunkSize: 1), E02Context(maxThreadCount: 2, preferredChunkSize: 0)] {
                do { _ = try await ExecutionOperations.map(rows, transform: { $0 }, context: context); Issue.record("invalid context") }
                catch let e as VectorError { #expect(e.kind == .invalidConfiguration) }
                do { _ = try await ExecutionOperations.normalize(rows, context: context); Issue.record("invalid context") }
                catch let e as VectorError { #expect(e.kind == .invalidConfiguration) }
            }
        }
        for count in [1,999,1000,1001] {
            let rows = Array(repeating: E02RejectingVector(), count: count)
            do { _ = try await rows.mapElements { $0 }; Issue.record("factory accepted") }
            catch { if case E02Failure.factory = error {} else { Issue.record("factory error changed") } }
        }
    }

    @Test func matricesStatisticsAndNonthrowingCancellation() async throws {
        for count in [0,1,31,32,99,100,101] {
            let rows = (0..<count).map { DynamicVector([Float($0),1]) }
            let events = E02Events()
            let result = await Task {
                withUnsafeCurrentTask { $0?.cancel() }
                return await BoundedExecution.$observation.withValue({ events.record($0) }) {
                    let matrix = await BatchOperations.pairwiseDistances(rows, metric: DotProductDistance())
                    let calls = E02Recorder()
                    let triangle = await TriangularMatrix<Int>.buildParallel(size: count) { i,j in calls.append(i*max(1,count)+j); return i+j }
                    var expectedVisits: [Int] = []
                    for i in 0..<count { for j in (i+1)..<count { expectedVisits.append(i*count+j) } }
                    #expect(calls.snapshot().sorted() == expectedVisits)
                    for i in 0..<count { for j in (i+1)..<count { #expect(triangle[i,j] == i+j) } }
                    let statistics = await BatchOperations.statistics(for: rows)
                    #expect(statistics.count == count)
                    let nearest = await BatchOperations.findNearest(to: DynamicVector([0,1]), in: rows, k: min(3,count))
                    #expect(nearest.map(\.index) == Array(0..<min(3,count)))
                    return matrix
                }
            }.value
            for i in 0..<count { for j in 0..<count { #expect(result[i][j] == -Float(i*j+1)) } }
            events.verify()
            let matrix = try await ExecutionOperations.distanceMatrix(rows, metric: DotProductDistance(), context: E02Context(maxThreadCount: 2, preferredChunkSize: 1))
            #expect(matrix == result)
        }
        let events = E02Events()
        let stats = await BoundedExecution.$observation.withValue({ events.record($0) }) {
            await BatchOperations.statisticsParallel(for: Array(repeating: DynamicVector([3,4]), count: 1025), chunkSize: 1)
        }
        #expect(stats.count == 1025 && stats.meanMagnitude == 5 && stats.stdMagnitude == 0)
        events.verify()
        #expect(events.snapshot().filter { $0.kind == .submitted }.count <= BoundedExecution.hardwareLimit)
        let phases = E02Events()
        let rows = Array(repeating: Vector<Dim768>.zero, count: 32)
        let matrix = try await BoundedExecution.$observation.withValue({ phases.record($0) }) { try await rows.cosineSimilarityMatrix() }
        #expect(matrix.count == 32 && matrix.flatMap { $0 }.allSatisfy { $0 == 0 })
        let begin = phases.snapshot().filter { $0.kind == .begin }
        #expect(begin.map(\.phase) == ["transform.normalize", "matrix.distance"])
        let eventsInOrder = phases.snapshot()
        if let normalizationEnd = eventsInOrder.firstIndex(where: { $0.phase == "transform.normalize" && $0.kind == .end }),
           let matrixBegin = eventsInOrder.firstIndex(where: { $0.phase == "matrix.distance" && $0.kind == .begin }) {
            #expect(normalizationEnd < matrixBegin)
        } else { Issue.record("missing phase boundaries") }
        phases.verify()
    }

    @Test func inheritedBatchQueryRoute() async throws {
        for limit in [1,2,4,Int.max] {
            let calls = E02Recorder(), events = E02Events()
            let rows = (0..<11).map { DynamicVector([Float($0)]) }
            let result = try await BoundedExecution.$observation.withValue({ events.record($0) }) {
                try await E02BatchProvider(maxConcurrency: limit, calls: calls).findNearestBatch(queries: rows, candidates: rows, k: 1, metric: EuclideanDistance())
            }
            #expect(result.map { $0[0].index } == Array(0..<11))
            #expect(calls.snapshot().sorted() == Array(0..<11)); events.verify(maximum: min(11,limit))
        }
    }

    @Test func sixDormantOptimizedHelpersDirectCoverage() async throws {
        for dimension in [512,768,1536] {
            for cosine in [false,true] {
                for count in [0,1,255,256,257,1025] {
                    var patterns = [[Float]]()
                    for i in 0..<count {
                        var row = [Float](repeating: 0, count: dimension)
                        row[0] = i % 7 == 0 ? .nan : Float(i % 3)
                        row[1] = 1
                        patterns.append(row)
                    }
                    var query = [Float](repeating: 0, count: dimension); query[0] = 1
                    let events = E02Events()
                    let (actual, kernelControl): ([Float], [Float]) = try await BoundedExecution.$observation.withValue({ events.record($0) }) {
                        switch dimension {
                        case 512:
                            let q = try Vector512Optimized(query), rows = try patterns.map { try Vector512Optimized($0) }
                            var control = [Float](repeating: -1, count: count)
                            control.withUnsafeMutableBufferPointer { out in
                                if cosine { BatchKernels.range_cosine_fused_512(query: q, candidates: rows, range: 0..<count, out: out) }
                                else { BatchKernels.range_euclid_512(query: q, candidates: rows, range: 0..<count, out: out) }
                            }
                            let values = cosine ? await BatchOperations.computeDistances_cosine_fused_parallel(query: q, candidates: rows, dim: dimension)
                                : await BatchOperations.computeDistances_euclid_optimized_parallel(query: q, candidates: rows, dim: dimension)
                            return (values, control)
                        case 768:
                            let q = try Vector768Optimized(query), rows = try patterns.map { try Vector768Optimized($0) }
                            var control = [Float](repeating: -1, count: count)
                            control.withUnsafeMutableBufferPointer { out in
                                if cosine { BatchKernels.range_cosine_fused_768(query: q, candidates: rows, range: 0..<count, out: out) }
                                else { BatchKernels.range_euclid_768(query: q, candidates: rows, range: 0..<count, out: out) }
                            }
                            let values = cosine ? await BatchOperations.computeDistances_cosine_fused_parallel(query: q, candidates: rows, dim: dimension)
                                : await BatchOperations.computeDistances_euclid_optimized_parallel(query: q, candidates: rows, dim: dimension)
                            return (values, control)
                        default:
                            let q = try Vector1536Optimized(query), rows = try patterns.map { try Vector1536Optimized($0) }
                            var control = [Float](repeating: -1, count: count)
                            control.withUnsafeMutableBufferPointer { out in
                                if cosine { BatchKernels.range_cosine_fused_1536(query: q, candidates: rows, range: 0..<count, out: out) }
                                else { BatchKernels.range_euclid_1536(query: q, candidates: rows, range: 0..<count, out: out) }
                            }
                            let values = cosine ? await BatchOperations.computeDistances_cosine_fused_parallel(query: q, candidates: rows, dim: dimension)
                                : await BatchOperations.computeDistances_euclid_optimized_parallel(query: q, candidates: rows, dim: dimension)
                            return (values, control)
                        }
                    }
                    #expect(actual.count == count)
                    for i in actual.indices {
                        let x = Double(patterns[i][0])
                        if x.isNaN {
                            // Preserve existing kernel behavior; E02 does not change numerical policy.
                            #expect(actual[i].isNaN == kernelControl[i].isNaN)
                            if !kernelControl[i].isNaN { #expect(actual[i].bitPattern == kernelControl[i].bitPattern) }
                        }
                        else {
                            let expected = cosine ? 1-x/sqrt(x*x+1) : sqrt((x-1)*(x-1)+1)
                            #expect(actual[i].isFinite && abs(Double(actual[i])-expected) <= 1e-5)
                            if i+3 < count && i % 7 != 0 && (i+3) % 7 != 0 { #expect(actual[i].bitPattern == actual[i+3].bitPattern) }
                        }
                    }
                    events.verify()
                    if count > 0 { #expect(events.snapshot().contains { $0.kind == .submitted }) }
                }
            }
        }
    }
}

extension ExecutionSchedulingContractTests {
    @Test func executeBatchModeSettingsAndEmptyControl() {
        for mode in [CPUComputeProvider.Mode.sequential, .parallel, .automatic] {
            for threshold in [nil,0,1,Int.max] as [Int?] {
                for hint in [nil,-1,1,Int.max] as [Int?] {
                    for count in [0,1,17] {
                        let calls = E02Recorder()
                        var output = [Int](repeating: -77, count: count+3)
                        output.withUnsafeMutableBufferPointer { out in
                            CPUComputeProvider(mode: mode, parallelizationThreshold: threshold, minChunk: hint).executeBatch(query: 0, candidates: Array(0..<count),
                                kernelKind: .dot, dimension: 1, results: out) { _,_,range,result in
                                calls.append(range.lowerBound); calls.append(range.upperBound)
                                for i in range { result[i] = i }
                            }
                        }
                        #expect(calls.snapshot() == (count == 0 ? [] : [0,count]))
                        #expect(output == Array(0..<count)+[-77,-77,-77])
                    }
                }
            }
        }
    }

    @Test func cancelledLiveTransformsAndMatrix() async {
        for count in [0,1,31,32,999,1000] {
            let rows = Array(repeating: DynamicVector([3,4]), count: count)
            for route in 0..<3 {
                let calls = E02Recorder(), events = E02Events()
                await Task {
                    withUnsafeCurrentTask { $0?.cancel() }
                    do {
                        try await BoundedExecution.$observation.withValue({ events.record($0) }) {
                            switch route {
                            case 0: _ = try await ExecutionOperations.map(rows, transform: { calls.append(1); return $0 })
                            case 1: _ = try await rows.normalized()
                            default: _ = try await ExecutionOperations.distanceMatrix(rows, metric: E02SpyMetric(calls: calls))
                            }
                        }
                        Issue.record("cancelled live operation returned success")
                    } catch { #expect(error is CancellationError) }
                }.value
                #expect(calls.snapshot().isEmpty && events.snapshot().isEmpty)
            }
        }
    }

    @Test func nestedMatrixCallbackCreatesNoLibraryWorkers() async throws {
        let events = E02Events(), inner = E02Events()
        let matrix = await BoundedExecution.$observation.withValue({ events.record($0) }) {
            await TriangularMatrix<Int>.buildParallel(size: 5) { i,j in
                await BoundedExecution.$observation.withValue({ inner.record($0) }) {
                    // Nonthrowing outer matrix guarantees a complete result; all inner work is valid.
                    let rows = [DynamicVector([Float(i)]), DynamicVector([Float(j)])]
                    let pairwise = await BatchOperations.pairwiseDistances(rows, metric: DotProductDistance())
                    let stats = await BatchOperations.statisticsParallel(for: rows, chunkSize: 1)
                    #expect(pairwise[0][1] == -Float(i*j) && stats.count == 2)
                    return i+j
                }
            }
        }
        #expect(matrix[1,4] == 5)
        events.verify(); inner.verify(maximum: 0)
    }
}

extension ExecutionSchedulingContractTests {
    @Test func eachRangeHelperChecksAfterBodyBeforeNextUnit() async {
        for provider: any ComputeProvider in [E02Defaults(maxConcurrency: 1), CPUComputeProvider.sequential, CPUComputeProvider.parallel] {
            for method in 0..<5 {
                let calls = E02Recorder()
                await Task {
                    let body: @Sendable (Int) async throws -> Void = { index in
                        calls.append(index); withUnsafeCurrentTask { $0?.cancel() }
                    }
                    do {
                        switch method {
                        case 0: _ = try await provider.parallelExecute(items: 0..<2048) { index in try await body(index); return index }
                        case 1: try await provider.parallelForEach(items: 0..<2048, body)
                        case 2: try await provider.parallelForEach(items: 0..<2048, minChunk: 1, body)
                        case 3: _ = try await provider.parallelReduce(items: 0..<2048, initial: 0, { range in try await body(range.lowerBound); return range.count }, +)
                        default: _ = try await provider.parallelReduce(items: 0..<2048, initial: 0, minChunk: 1, { range in try await body(range.lowerBound); return range.count }, +)
                        }
                        Issue.record("missing post-body cancellation check")
                    } catch { #expect(error is CancellationError) }
                }.value
                #expect(!calls.snapshot().isEmpty)
                #expect(calls.snapshot().count <= provider.maxConcurrency)
            }
        }
    }

    @Test func batchCancellationBetweenWorkUnits() async {
        for count in [3,1001] {
            let rows = Array(repeating: DynamicVector([1]), count: count)
            for route in 0..<3 {
                let calls = E02Recorder()
                await Task {
                    do {
                        switch route {
                        case 0: _ = try await BatchOperations.map(rows) { row in calls.append(1); withUnsafeCurrentTask { $0?.cancel() }; return row }
                        case 1: _ = try await BatchOperations.filter(rows) { _ in calls.append(1); withUnsafeCurrentTask { $0?.cancel() }; return true }
                        default: let _: [Int] = try await BatchOperations.process(rows, batchSize: 1) { _ in calls.append(1); withUnsafeCurrentTask { $0?.cancel() }; return [1] }
                        }
                        Issue.record("batch cancellation ignored")
                    } catch { #expect(error is CancellationError) }
                }.value
                #expect(!calls.snapshot().isEmpty)
                #expect(calls.snapshot().count <= (count < 1000 ? 1 : BoundedExecution.hardwareLimit))
            }
        }
    }
}

private final class E02Signal: @unchecked Sendable {
    private let lock = NSLock()
    private var released = false
    private var continuation: CheckedContinuation<Void, Never>?
    func suspendUntilReleased() async {
        await withCheckedContinuation { waiter in
            lock.lock()
            if released { lock.unlock(); waiter.resume() }
            else { continuation = waiter; lock.unlock() }
        }
    }
    func release() {
        lock.lock(); released = true; let waiter = continuation; continuation = nil; lock.unlock()
        waiter?.resume()
    }
}
extension ExecutionSchedulingContractTests {
    func reverseCompletionReductionAndErrorCleanup() async throws {
        let signal = E02Signal(), events = E02Events()
        let value = try await BoundedExecution.$observation.withValue({ event in
            events.record(event)
            if event.kind == .completed && event.worker == 1 { signal.release() }
        }) {
            try await E02Defaults(maxConcurrency: 2).parallelReduce(items: 0..<4, initial: "", { range in
                if range.lowerBound == 0 { await signal.suspendUntilReleased() }
                return range.map(String.init).joined()
            }, +)
        }
        signal.release()
        #expect(value == "0123")
        #expect(events.snapshot().filter { $0.kind == .completed }.map(\.worker) == [1,0])
        events.verify(maximum: 2)
        let gate = E02Gate(), cleanup = E02Events()
        let task = Task {
            try await BoundedExecution.$observation.withValue({ cleanup.record($0) }) {
                try await E02Defaults(maxConcurrency: 2).parallelExecute(items: 0..<4) { index in
                    await gate.body()
                    if index == 0 { throw E02Failure.body }
                    return index
                }
            }
        }
        while await gate.entered < 2 { await Task.yield() }
        await gate.release()
        do { _ = try await task.value; Issue.record("body error swallowed") }
        catch { if case E02Failure.body = error {} else { Issue.record("original error changed") } }
        #expect(await gate.active == 0)
        cleanup.verify(maximum: 2)
    }
}


extension ExecutionSchedulingContractTests {
    @Test func watchdogRejectsTimedOutChild() throws {
        do {
            try boundedRegressionChild(scenario: "watchdog")
            Issue.record("Watchdog child unexpectedly returned")
        } catch let error as NSError {
            #expect(error.domain == "E02.childTimeout")
            #expect(error.code == 1, "Owned child must be reaped within the cleanup bound")
        }
    }
}

extension ExecutionSchedulingContractTests {
    @Test func publicGenericNearestBoundaryWorkerObservation() async throws {
        try await e06IsolatedScenario("generic-nearest", filter: "ExecutionSchedulingContractTests/publicGenericNearestBoundaryWorkerObservation") {
        try await BatchOperations.updateConfiguration { $0.parallelThreshold = 35 }
        // Live threshold neighbors retain the independent numerical and worker controls.
        for count in [34,35,36,4097] {
            let parallel = count >= 35
            let query = DynamicVector(Array(repeating: Float(0), count: 64))
            for exceptional in [false,true] {
                let rows = (0..<count).map { index in
                    var values = [Float](repeating: 0, count: 64)
                    values[0] = exceptional && index % 7 == 0 ? .nan : Float(index % 3)
                    return DynamicVector(values)
                }
                let nanIndices = (0..<count).filter { exceptional && $0 % 7 == 0 }
                let numericIndices = (0..<count).filter { !(exceptional && $0 % 7 == 0) }.sorted {
                    $0 % 3 == $1 % 3 ? $0 < $1 : $0 % 3 < $1 % 3
                }
                let expectedIndices = numericIndices + nanIndices
                for k in [1,count] {
                    let events = E02Events()
                    let result = await Task {
                        withUnsafeCurrentTask { $0?.cancel() }
                        return await BoundedExecution.$observation.withValue({ events.record($0) }) {
                            await BatchOperations.findNearest(to: query, in: rows, k: k)
                        }
                    }.value
                    #expect(result.map(\.index) == Array(expectedIndices.prefix(k)))
                    #expect(result.count == k)
                    for item in result {
                        if exceptional && item.index % 7 == 0 { #expect(item.distance.isNaN) }
                        else {
                            // Independent scalar norm: only coordinate zero is nonzero.
                            let coordinate = Double(item.index % 3)
                            let expected = (coordinate * coordinate).squareRoot()
                            #expect(item.distance.isFinite && abs(Double(item.distance)-expected) <= 1e-6)
                        }
                    }
                    events.verify(maximum: BoundedExecution.hardwareLimit)
                    let submissions = events.snapshot().filter { $0.phase == "batch.nearest" && $0.kind == .submitted }.count
                    #expect(parallel ? submissions > 0 : submissions == 0)
                    #expect(submissions <= BoundedExecution.hardwareLimit)
                    print("E02_GENERIC_NEAREST N=\(count) exceptional=\(exceptional) k=\(k) parallel=\(parallel) submitted=\(submissions) limit=\(BoundedExecution.hardwareLimit)")
                }
            }
        }
    }
    }
}

extension ExecutionSchedulingContractTests {
    @Test func reductionCombineCancellationAndRegionAcrossProvidersAndOverloads() async {
        let providers: [any ComputeProvider] = [E02Defaults(maxConcurrency: 4), CPUComputeProvider.sequential,
            CPUComputeProvider.parallel, CPUComputeProvider(mode: .automatic, parallelizationThreshold: 1)]
        for provider in providers {
            for hinted in [false,true] {
                let calls = E02Recorder(), markers = E02Recorder()
                await Task {
                    let combine: @Sendable (Int, Int) -> Int = { first, second in
                        calls.append(1)
                        markers.append(BoundedExecution.isInsideRegion ? 1 : 0)
                        withUnsafeCurrentTask { $0?.cancel() }
                        return first + second
                    }
                    do {
                        if hinted {
                            _ = try await provider.parallelReduce(items: 0..<2048, initial: 0, minChunk: 3, { $0.count }, combine)
                        } else {
                            _ = try await provider.parallelReduce(items: 0..<2048, initial: 0, { $0.count }, combine)
                        }
                        Issue.record("Cancelled combine returned success")
                    } catch { #expect(error is CancellationError) }
                }.value
                #expect(calls.snapshot() == [1], "No later combine may run after the first cancels its task")
                #expect(markers.snapshot() == [1], "Every combine is a library callback region")
            }
        }
    }
}
