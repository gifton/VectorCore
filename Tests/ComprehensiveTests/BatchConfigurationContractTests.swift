import Foundation
import Darwin
import Testing
@testable import VectorCore

private final class E06Values<T: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [T] = []
    func append(_ value: T) { lock.lock(); defer { lock.unlock() }; values.append(value) }
    func snapshot() -> [T] { lock.lock(); defer { lock.unlock() }; return values }
}

@Suite struct BatchConfigurationContractTests {
    static func bits(_ c: BatchOperations.Configuration) -> [UInt64] {
        [UInt64(bitPattern: Int64(c.parallelThreshold)), c.oversubscription.bitPattern,
         UInt64(bitPattern: Int64(c.minimumChunkSize)), UInt64(bitPattern: Int64(c.defaultBatchSize)),
         c.enableMatrixRouting ? 1 : 0, UInt64(bitPattern: Int64(c.matrixRoutingMinN))]
    }

    @Test(.serialized, arguments: ["defaultBatch", "threshold", "chunks", "invalid", "domains", "linear", "partitions", "snapshot", "routes512", "routes768", "routes1536", "pairwise", "nested"])
    func configurationScenarioInBoundedChild(_ scenario: String) throws {
        var environment = ProcessInfo.processInfo.environment
        environment["E06_SCENARIO"] = scenario
        let directory = environment["E06_CAPTURE_DIRECTORY"].map { URL(fileURLWithPath: $0) } ?? FileManager.default.temporaryDirectory
        let path = directory.appendingPathComponent("E06-\(scenario)-\(UUID().uuidString).log")
        FileManager.default.createFile(atPath: path.path, contents: nil)
        let file = try FileHandle(forWritingTo: path)
        let process = try launchChildProbe(.swiftTesting(filter: "BatchConfigurationContractTests/configurationChild"),
                                           environment: environment, output: file, deadline: 15,
                                           cleanup: .terminateThenKill(grace: 1, seconds: 1))
        if process.timedOut {
            try file.close()
            print("E06_CHILD_CAPTURE \(scenario) \(path.path)")
            throw NSError(domain: "E06.childTimeout", code: 1)
        }
        try file.close()
        let output = try String(contentsOf: path, encoding: .utf8)
        print("E06_CHILD_CAPTURE \(scenario) \(path.path)")
        print("E06_CHILD_RESULT \(scenario) status=\(process.statusDescription) reason=\(process.reasonDescription)")
        #expect(process.terminationStatus == 0, "\(scenario): \(output.prefix(6000))")
        #expect(output.contains("E06_COMPLETE \(scenario)"))
    }

    @Test func configurationChild() async throws {
        guard let scenario = ProcessInfo.processInfo.environment["E06_SCENARIO"] else { return }
        let invocation: [String: Any] = ["argv": CommandLine.arguments, "scenario": scenario,
            "DYLD_FRAMEWORK_PATH": ProcessInfo.processInfo.environment["DYLD_FRAMEWORK_PATH"] ?? "",
            "DYLD_LIBRARY_PATH": ProcessInfo.processInfo.environment["DYLD_LIBRARY_PATH"] ?? ""]
        print("E06_INVOCATION " + String(decoding: try JSONSerialization.data(withJSONObject: invocation, options: [.sortedKeys]), as: UTF8.self))
        let original = await BatchOperations.configuration()
        do { try await run(scenario) }
        catch {
            try await BatchOperations.updateConfiguration { $0 = original }
            #expect(Self.bits(await BatchOperations.configuration()) == Self.bits(original))
            throw error
        }
        try await BatchOperations.updateConfiguration { $0 = original }
        #expect(Self.bits(await BatchOperations.configuration()) == Self.bits(original))
        print("E06_COMPLETE \(scenario)")
    }

    private func run(_ scenario: String) async throws {
        let vector = DynamicVector([Float](repeating: 1, count: 64))
        switch scenario {
        case "defaultBatch":
            try await BatchOperations.updateConfiguration { $0.defaultBatchSize = 3 }
            let shapes = E06Values<Int>()
            let result: [Int] = try await BatchOperations.process(Array(repeating: vector, count: 8)) { batch in
                shapes.append(batch.count); return [batch.count]
            }
            #expect(result == [3,3,2]); #expect(shapes.snapshot() == [3,3,2])
        case "threshold":
            try await BatchOperations.updateConfiguration { $0.parallelThreshold = 2 }
            let events = E06Values<BoundedExecution.Event>()
            _ = try await BoundedExecution.$observation.withValue({ events.append($0) }) {
                try await BatchOperations.map(Array(repeating: vector, count: 3)) { $0 }
            }
            #expect(events.snapshot().contains { $0.phase == "batch.map" && $0.kind == .submitted })
        case "chunks":
            try await BatchOperations.updateConfiguration { $0.minimumChunkSize = Int.max }
            let events = E06Values<BoundedExecution.Event>()
            _ = await BoundedExecution.$observation.withValue({ events.append($0) }) {
                await BatchOperations.statistics(for: Array(repeating: vector, count: 1025))
            }
            #expect(events.snapshot().filter { $0.kind == .submitted }.count == 1)
        case "invalid":
            let old = await BatchOperations.configuration()
            do {
                try await BatchOperations.updateConfiguration { value throws in value.parallelThreshold = 0 }
                Issue.record("Invalid live configuration was accepted")
            } catch let error as VectorError { #expect(error.kind == .invalidConfiguration) }
            #expect(Self.bits(await BatchOperations.configuration()) == Self.bits(old))
        case "domains": try await configurationDomains()
        case "linear": try await linearSettings()
        case "partitions": try await logicalPartitions()
        case "snapshot": try await snapshotConsistency()
        case "routes512": try await optimizedRoutes(Vector512Optimized.self, width: 512)
        case "routes768": try await optimizedRoutes(Vector768Optimized.self, width: 768)
        case "routes1536": try await optimizedRoutes(Vector1536Optimized.self, width: 1536)
        case "pairwise": try await pairwiseSettings()
        case "nested": try await nestedInvocation()
        default: Issue.record("Unknown scenario \(scenario)")
        }
    }
}

private final class E06Sentinel: Error, Sendable {}

extension BatchConfigurationContractTests {
    private func configurationDomains() async throws {
        let defaults = BatchOperations.Configuration()
        try await BatchOperations.updateConfiguration { $0 = defaults }
        #expect(Self.bits(await BatchOperations.configuration()) == Self.bits(defaults))
        let fields: [String] = ["parallelThreshold", "minimumChunkSize", "defaultBatchSize", "matrixRoutingMinN"]
        for field in fields {
            for invalid in [0,-1,Int.min] {
                let before = await BatchOperations.configuration()
                do {
                    try await BatchOperations.updateConfiguration { candidate in
                        candidate.enableMatrixRouting.toggle()
                        switch field {
                        case "parallelThreshold": candidate.parallelThreshold = invalid
                        case "minimumChunkSize": candidate.minimumChunkSize = invalid
                        case "defaultBatchSize": candidate.defaultBatchSize = invalid
                        default: candidate.matrixRoutingMinN = invalid
                        }
                    }
                    Issue.record("Accepted invalid \(field)=\(invalid)")
                } catch let error as VectorError {
                    #expect(error.kind == .invalidConfiguration && error.description.contains(field))
                }
                #expect(Self.bits(await BatchOperations.configuration()) == Self.bits(before))
            }
        }
        let processors = Double(max(1, ProcessInfo.processInfo.processorCount))
        let invalidFactors: [Double] = [0,-0.0,-1,.nan,.infinity,-.infinity,.greatestFiniteMagnitude,(Double(Int.max)/processors).nextUp]
        for factor in invalidFactors {
            let before = await BatchOperations.configuration()
            do {
                try await BatchOperations.updateConfiguration { $0.oversubscription = factor; $0.defaultBatchSize = 17 }
                Issue.record("Accepted invalid oversubscription \(factor)")
            } catch let error as VectorError { #expect(error.kind == .invalidConfiguration && error.description.contains("oversubscription")) }
            #expect(Self.bits(await BatchOperations.configuration()) == Self.bits(before))
        }
        // Validation follows declaration order, after any original user error.
        for first in ["parallelThreshold", "oversubscription", "minimumChunkSize", "defaultBatchSize", "matrixRoutingMinN"] {
            do {
                try await BatchOperations.updateConfiguration { c in
                    c.matrixRoutingMinN = 0
                    if first != "matrixRoutingMinN" { c.defaultBatchSize = 0 }
                    if first == "minimumChunkSize" || first == "oversubscription" || first == "parallelThreshold" { c.minimumChunkSize = 0 }
                    if first == "oversubscription" || first == "parallelThreshold" { c.oversubscription = .nan }
                    if first == "parallelThreshold" { c.parallelThreshold = 0 }
                }
                Issue.record("Invalid candidate accepted")
            } catch let error as VectorError { #expect(error.description.contains(first)) }
        }
        let sentinel = E06Sentinel(), before = await BatchOperations.configuration()
        do {
            try await BatchOperations.updateConfiguration { candidate in
                candidate.parallelThreshold = 0; candidate.oversubscription = .nan
                candidate.minimumChunkSize = -1; candidate.defaultBatchSize = -1
                candidate.enableMatrixRouting.toggle(); candidate.matrixRoutingMinN = -1
                throw sentinel
            }
            Issue.record("Original error lost")
        } catch { #expect((error as? E06Sentinel) === sentinel) }
        #expect(Self.bits(await BatchOperations.configuration()) == Self.bits(before))
        var boundary = Double(Int.max).nextDown / processors
        while processors * boundary > Double(Int.max).nextDown { boundary = boundary.nextDown }
        for factor in [Double.leastNonzeroMagnitude, 1,2,4,boundary] {
            try await BatchOperations.updateConfiguration { c in
                c.oversubscription = factor; c.parallelThreshold = 1
                c.minimumChunkSize = Int.max; c.defaultBatchSize = Int.max; c.matrixRoutingMinN = Int.max
            }
            let rows = [DynamicVector([Float(1)]), DynamicVector([Float(2)]), DynamicVector([Float(3)])]
            let sizes: [Int] = try await BatchOperations.process(rows) { [$0.count] }
            #expect(sizes == [3])
            let stats = await BatchOperations.statistics(for: rows)
            #expect(stats.count == 3 && stats.meanMagnitude == 2)
            #expect(await BatchOperations.findNearest(to: rows[0], in: rows, k: Int.max).map(\.index) == [0,1,2])
            #expect(await BatchOperations.pairwiseDistances(rows).count == 3)
        }
        try await BatchOperations.updateConfiguration { $0.parallelThreshold = Int.max }
        let events = E06Values<BoundedExecution.Event>()
        _ = try await BoundedExecution.$observation.withValue({ events.append($0) }) {
            try await BatchOperations.map([DynamicVector([Float(1)])]) { $0 }
        }
        #expect(events.snapshot().allSatisfy { $0.kind != .submitted })
    }

    @Test func partitionArithmeticBoundaries() {
        for count in [0,1,2,3,17,Int.max-1,Int.max] {
            for cores in [1,2,17,Int.max] {
                for minimum in [1,3,Int.max] {
                    for factor in [Double.leastNonzeroMagnitude,1,2,4] {
                        var c = BatchOperations.Configuration(); c.minimumChunkSize = minimum; c.oversubscription = factor
                        let chunk = BatchOperations.logicalChunkSize(count: count, cores: cores, configuration: c)
                        if count == 0 { #expect(chunk == 0); continue }
                        #expect(chunk > 0 && chunk <= count && chunk >= min(count,minimum))
                        if factor == Double.leastNonzeroMagnitude || minimum == Int.max { #expect(chunk == count) }
                        if cores >= count && factor >= 1 { #expect(chunk == min(count,minimum)) }
                        let chunks = count / chunk + (count % chunk == 0 ? 0 : 1)
                        let lastStart = (chunks-1) * chunk
                        let end = lastStart + min(chunk,count-lastStart)
                        #expect(end == count)
                    }
                }
            }
        }
        var c = BatchOperations.Configuration(); c.minimumChunkSize = 1; c.oversubscription = 2
        #expect(BatchOperations.logicalChunkSize(count: 17, cores: 2, configuration: c) == 5)
        c.minimumChunkSize = 7
        #expect(BatchOperations.logicalChunkSize(count: 17, cores: 2, configuration: c) == 7)
    }

    private func verifyEvents(_ events: [BoundedExecution.Event]) {
        for group in Dictionary(grouping: events, by: \.id).values {
            #expect(group.first?.kind == .begin && group.last?.kind == .end)
            let submitted = group.filter { $0.kind == .submitted }.map(\.worker)
            let completed = group.filter { $0.kind == .completed }.map(\.worker)
            #expect(submitted.count <= group[0].limit)
            #expect(submitted.sorted() == completed.sorted())
            #expect(Set(submitted).count == submitted.count)
        }
    }

    private func linearSettings() async throws {
        try await BatchOperations.updateConfiguration { $0.parallelThreshold = 7; $0.defaultBatchSize = 2; $0.minimumChunkSize = 1 }
        for count in [0,1,6,7,8] {
            let rows = (0..<count).map { DynamicVector([Float($0)]) }
            for operation in ["process","map","filter","statistics","nearest"] {
                let events = E06Values<BoundedExecution.Event>()
                try await BoundedExecution.$observation.withValue({ events.append($0) }) {
                    switch operation {
                    case "process":
                        let output: [Float] = try await BatchOperations.process(rows) { $0.map { $0[0] } }
                        #expect(output == rows.map { $0[0] })
                        let explicit: [Int] = try await BatchOperations.process(rows, batchSize: 3) { [$0.count] }
                        #expect(explicit == stride(from:0,to:count,by:3).map { min(3,count-$0) })
                    case "map": #expect(try await BatchOperations.map(rows) { $0 }.map { $0[0] } == rows.map { $0[0] })
                    case "filter": #expect(try await BatchOperations.filter(rows) { Int($0[0]) % 2 == 0 }.map { $0[0] } == rows.filter { Int($0[0]) % 2 == 0 }.map { $0[0] })
                    case "statistics": #expect(await BatchOperations.statistics(for: rows).count == count)
                    default: #expect(await BatchOperations.findNearest(to: DynamicVector([Float(0)]), in: rows, k: Int.max).count == count)
                    }
                }
                verifyEvents(events.snapshot())
                let submissions = events.snapshot().filter { $0.kind == .submitted }
                #expect(count >= 7 ? !submissions.isEmpty : submissions.isEmpty)
            }
        }
        for size in [0,-1,Int.min] {
            for empty in [false,true] {
                let calls = E06Values<Int>()
                await Task {
                    withUnsafeCurrentTask { $0?.cancel() }
                    do {
                        let _: [Int] = try await BatchOperations.process(empty ? [] : [DynamicVector([Float(1)])], batchSize: size) { _ in calls.append(1);return [] }
                        Issue.record("Invalid explicit batch accepted")
                    } catch let error as VectorError { #expect(error.kind == .invalidConfiguration && error.description.contains("batchSize")) }
                    catch { Issue.record("Explicit validation did not precede cancellation") }
                }.value
                #expect(calls.snapshot().isEmpty)
            }
        }
    }

    private func logicalPartitions() async throws {
        let rows = (0..<17).map { DynamicVector([Float($0)]) }, query = DynamicVector([Float(0)])
        try await BatchOperations.updateConfiguration { $0.parallelThreshold = 1 }
        for minimum in [1,3,Int.max] {
            try await BatchOperations.updateConfiguration { $0.minimumChunkSize = minimum }
            for factor in [Double.leastNonzeroMagnitude,1,2,4] {
                try await BatchOperations.updateConfiguration { $0.oversubscription = factor }
                let coreCount = BoundedExecution.hardwareLimit
                let target = factor == Double.leastNonzeroMagnitude ? 1 : min(17,coreCount * Int(factor))
                let ideal = (17 / target) + (17 % target == 0 ? 0 : 1)
                let step = min(17,max(minimum,ideal))
                let expected = stride(from:0,to:17,by:step).map { $0..<min(17,$0+step) }
                for operation in ["statistics","nearest"] {
                    let events = E06Values<BoundedExecution.Event>()
                    await BoundedExecution.$observation.withValue({ events.append($0) }) {
                        if operation == "statistics" {
                            let result = await BatchOperations.statistics(for: rows)
                            #expect(result.count == 17 && result.meanMagnitude.isFinite && result.meanMagnitude == 8)
                            #expect(result.stdMagnitude.isFinite && abs(Double(result.stdMagnitude)-sqrt(24)) < 1e-6)
                        } else { #expect(await BatchOperations.findNearest(to: query,in:rows,k:17).map(\.index) == Array(0..<17)) }
                    }
                    let ranges = events.snapshot().compactMap { event -> Range<Int>? in
                        if case .logicalRange(let range) = event.kind { return range };return nil
                    }.sorted { $0.lowerBound < $1.lowerBound }
                    #expect(ranges == expected)
                    verifyEvents(events.snapshot())
                }
            }
        }
    }
}

private struct E06Metric: DistanceMetric {
    typealias Scalar = Float
    let scale: Float
    let calls: E06Values<Int>
    var name: String { "E06 supplied metric" }
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float {
        calls.append(Int(b[0]))
        return scale * abs(a[0]-b[0]) + 7
    }
}

private final class E06OneGate: @unchecked Sendable {
    private let lock = NSLock()
    private var entered = false
    private var continuation: CheckedContinuation<Void,Never>?
    private let releaseSignal = DispatchSemaphore(value: 0)
    func suspendUntilEntered() async {
        await withCheckedContinuation { continuation in
            lock.lock()
            if entered { lock.unlock(); continuation.resume() }
            else { self.continuation = continuation; lock.unlock() }
        }
    }
    func holdFirst() throws {
        lock.lock()
        if entered { lock.unlock();return }
        entered = true
        let continuation = self.continuation; self.continuation = nil
        lock.unlock(); continuation?.resume()
        if releaseSignal.wait(timeout: .now() + 5) == .timedOut { throw E06Sentinel() }
    }
    func release() { releaseSignal.signal() }
}

extension BatchConfigurationContractTests {
    private func snapshotConsistency() async throws {
        try await BatchOperations.updateConfiguration { $0.parallelThreshold = 1; $0.defaultBatchSize = 3 }
        let rows = (0..<8).map { DynamicVector([Float($0)]) }
        let gate = E06OneGate(), shapes = E06Values<Int>()
        let task = Task {
            try await BatchOperations.process(rows) { batch -> [Float] in
                try gate.holdFirst();shapes.append(batch.count);return batch.map { $0[0] }
            }
        }
        await gate.suspendUntilEntered()
        do { try await BatchOperations.updateConfiguration { $0.defaultBatchSize = 2; $0.parallelThreshold = Int.max } }
        catch { gate.release();_ = try? await task.value;throw error }
        gate.release()
        #expect(try await task.value == Array(0..<8).map(Float.init))
        #expect(shapes.snapshot().sorted() == [2,3,3])
        let events = E06Values<BoundedExecution.Event>()
        let later: [Int] = try await BoundedExecution.$observation.withValue({ events.append($0) }) {
            try await BatchOperations.process(rows) { [$0.count] }
        }
        #expect(later == [2,2,2,2]);#expect(events.snapshot().allSatisfy { $0.kind != .submitted })
    }

    private func directScores<O: VectorProtocol>(query: O, rows: [O], cosine: Bool) -> [Float] where O.Scalar == Float {
        var scores = [Float](repeating: -1,count: rows.count)
        scores.withUnsafeMutableBufferPointer { out in
            if let q = query as? Vector512Optimized, let vectors = rows as? [Vector512Optimized] {
                if cosine { BatchKernels.range_cosine_fused_512(query:q,candidates:vectors,range:0..<rows.count,out:out) }
                else { BatchKernels.range_euclid_512(query:q,candidates:vectors,range:0..<rows.count,out:out) }
            } else if let q = query as? Vector768Optimized, let vectors = rows as? [Vector768Optimized] {
                if cosine { BatchKernels.range_cosine_fused_768(query:q,candidates:vectors,range:0..<rows.count,out:out) }
                else { BatchKernels.range_euclid_768(query:q,candidates:vectors,range:0..<rows.count,out:out) }
            } else if let q = query as? Vector1536Optimized, let vectors = rows as? [Vector1536Optimized] {
                if cosine { BatchKernels.range_cosine_fused_1536(query:q,candidates:vectors,range:0..<rows.count,out:out) }
                else { BatchKernels.range_euclid_1536(query:q,candidates:vectors,range:0..<rows.count,out:out) }
            } else { Issue.record("Missing optimized direct control") }
        }
        return scores
    }

    private func optimizedRoutes<O: VectorProtocol & Sendable>(_ type: O.Type, width: Int) async throws where O.Scalar == Float {
        try await BatchOperations.updateConfiguration { $0.parallelThreshold = 8; $0.minimumChunkSize = 3; $0.oversubscription = 4 }
        for count in [0,1,7,8,9,259] {
            for cosine in [false,true] {
                for zeroQuery in [false,true] {
                    var queryValues = [Float](repeating:0,count:width)
                    if !zeroQuery { queryValues[0] = 2;queryValues[1] = 1 }
                    let query = try O(queryValues)
                    let patterns: [[Float]] = (0..<count).map { index in
                        var a = [Float](repeating:0,count:width)
                        switch index % 9 {
                        case 0: a[0] = -0.0
                        case 1: a[0] = .nan;a[1] = 1
                        case 2: a[0] = .infinity;a[1] = 1
                        case 3: a[0] = -.infinity;a[1] = 1
                        default: a[0] = Float(index % 3);a[1] = 1
                        }
                        return a
                    }
                    let rows = try patterns.map(O.init)
                    let control = directScores(query:query,rows:rows,cosine:cosine)
                    // Independent finite ranking formula for the two populated coordinates.
                    // Nonfinite membership uses the direct-kernel policy.
                    // Typed parts: the inline literal-ternary form exceeded the Swift 6.0 type
                    // checker's limits in CI (Xcode 16.2). Same Double operations in the same order.
                    let queryX: Double = zeroQuery ? 0 : 2, queryY: Double = zeroQuery ? 0 : 1
                    let rankingScores: [Float] = patterns.enumerated().map { (index, values) -> Float in
                        guard values.allSatisfy(\.isFinite) else { return control[index] }
                        let x = Double(values[0]), y = Double(values[1])
                        if !cosine {
                            let dx: Double = queryX-x, dy: Double = queryY-y
                            return Float(sqrt(pow(dx,2) + pow(dy,2)))
                        }
                        if x == 0 && y == 0 { return 1 }
                        if zeroQuery { return 1 }
                        if x == 2 && y == 1 { return 0 }
                        let norm: Double = sqrt(5*(x*x+y*y))
                        return Float(1-(2*x+y)/norm)
                    }
                    let expectedOrder = Array(0..<count).sorted { first,second in
                        let a = rankingScores[first],b = rankingScores[second]
                        if a.isNaN != b.isNaN { return !a.isNaN }
                        if a.isNaN || a == b { return first < second }
                        return a < b
                    }
                    for k in [-1,0,1,count,count+1,Int.max] {
                        let events = E06Values<BoundedExecution.Event>()
                        let actual = await Task {
                            withUnsafeCurrentTask { $0?.cancel() }
                            return await BoundedExecution.$observation.withValue({ events.append($0) }) {
                                if cosine { return await BatchOperations.findNearest(to:query,in:rows,k:k,metric:CosineDistance()) }
                                return await BatchOperations.findNearest(to:query,in:rows,k:k,metric:EuclideanDistance())
                            }
                        }.value
                        #expect(actual.map(\.index) == Array(expectedOrder.prefix(max(0,min(k,count)))))
                        for item in actual {
                            let values = patterns[item.index]
                            if values.allSatisfy(\.isFinite) {
                                var dot: Double = 0, aa: Double = 0, bb: Double = 0, square: Double = 0
                                for i in 0..<width {
                                    let a = Double(queryValues[i]),b = Double(values[i]);dot += a*b;aa += a*a;bb += b*b;square += (a-b)*(a-b)
                                }
                                let expected: Double
                                if !cosine { expected = sqrt(square) }
                                else if aa == 0 || bb == 0 { expected = 1 }
                                else { expected = 1-dot/(sqrt(aa)*sqrt(bb)) }
                                #expect(item.distance.isFinite && abs(Double(item.distance)-expected) < 2e-6)
                            } else {
                                // E06 preserves the existing direct-kernel exceptional score policy.
                                let ref = control[item.index]
                                #expect(ref.isNaN ? item.distance.isNaN : item.distance.bitPattern == ref.bitPattern)
                            }
                        }
                        let phase = "batch.\(cosine ? "cosine_fused" : "euclid")\(width)"
                        let submissions = events.snapshot().filter { $0.phase == phase && $0.kind == .submitted }
                        #expect(count >= 8 && k > 0 ? !submissions.isEmpty : submissions.isEmpty)
                        verifyEvents(events.snapshot())
                    }
                    if cosine && !zeroQuery && count > 1 {
                        #expect(control[1].isNaN && expectedOrder.drop { $0 != 1 }.allSatisfy { rankingScores[$0].isNaN })
                    }
                }
            }
        }
        // Concrete and parameterized custom metrics retain their supplied witnesses.
        let query = try O(Array(repeating:0,count:width))
        let rows = try (0..<11).map { index -> O in var a = [Float](repeating:0,count:width);a[0] = Float(index);a[1] = Float(index % 2);return try O(a) }
        for threshold in [1,Int.max] {
            try await BatchOperations.updateConfiguration { $0.parallelThreshold = threshold }
            let calls = E06Values<Int>()
            let custom = await BatchOperations.findNearest(to:query,in:rows,k:Int.max,metric:E06Metric(scale:3,calls:calls))
            #expect(custom.map(\.index) == Array(0..<11));#expect(calls.snapshot().sorted() == Array(0..<11))
            for value in custom { #expect(value.distance == Float(value.index)*3+7) }
            let parameterized = await BatchOperations.findNearest(to:query,in:rows,k:Int.max,metric:MinkowskiDistance(p:3))
            #expect(parameterized.map(\.index) == Array(0..<11))
            for item in parameterized {
                let x = Double(item.index),y = Double(item.index % 2)
                let expected = pow(x*x*x+y*y*y,1.0/3.0)
                #expect(item.distance.isFinite && abs(Double(item.distance)-expected) < 2e-5)
            }
        }
    }

    private func nestedInvocation() async throws {
        try await BatchOperations.updateConfiguration { $0.parallelThreshold = 1; $0.defaultBatchSize = 3; $0.minimumChunkSize = 1 }
        let rows = (0..<9).map { DynamicVector([Float($0)]) }
        let events = E06Values<BoundedExecution.Event>()
        let result = try await CPUComputeProvider.parallel.parallelExecute(items:0..<1) { _ -> [Int] in
            // A nested public invocation takes a fresh snapshot while retaining the region.
            try await BatchOperations.updateConfiguration { $0.defaultBatchSize = 2 }
            return try await BoundedExecution.$observation.withValue({ events.append($0) }) {
                let shapes: [Int] = try await BatchOperations.process(rows) { [$0.count] }
                #expect(await BatchOperations.findNearest(to:rows[0],in:rows,k:Int.max).map(\.index) == Array(0..<9))
                return shapes
            }
        }
        #expect(result == [[2,2,2,2,1]])
        #expect(events.snapshot().allSatisfy { $0.kind != .submitted });verifyEvents(events.snapshot())
    }

    private func pairwiseSettings() async throws {
        for width in [512,768,1536] {
            for count in [2,3,4] {
                for enabled in [false,true] {
                    try await BatchOperations.updateConfiguration { $0.enableMatrixRouting = enabled; $0.matrixRoutingMinN = 3; $0.parallelThreshold = 1 }
                    let events = E06Values<BoundedExecution.Event>()
                    await BoundedExecution.$observation.withValue({ events.append($0) }) {
                        switch width {
                        case 512: await checkPairwise((0..<count).map { i in Vector512Optimized(repeating:Float(i)) })
                        case 768: await checkPairwise((0..<count).map { i in Vector768Optimized(repeating:Float(i)) })
                        default: await checkPairwise((0..<count).map { i in Vector1536Optimized(repeating:Float(i)) })
                        }
                    }
                    let gemm = events.snapshot().filter { $0.phase == "batch.pairwise.gemm" && $0.kind == .begin }
                    #expect(gemm.count == (enabled && count >= 3 ? 2 : 0))
                    verifyEvents(events.snapshot())
                }
            }
        }
        try await BatchOperations.updateConfiguration { $0.enableMatrixRouting = true; $0.matrixRoutingMinN = 1; $0.parallelThreshold = Int.max }
        for count in [99,100,101] {
            let rows = (0..<count).map { DynamicVector([Float($0)]) }, calls = E06Values<Int>()
            let events = E06Values<BoundedExecution.Event>()
            let matrix = await BoundedExecution.$observation.withValue({ events.append($0) }) {
                await BatchOperations.pairwiseDistances(rows,metric:E06Metric(scale:2,calls:calls))
            }
            #expect(matrix.count == count)
            for i in 0..<count { #expect(matrix[i][i] == 7) }
            #expect(events.snapshot().allSatisfy { $0.phase != "batch.pairwise.gemm" })
            #expect(count >= 100 ? events.snapshot().contains { $0.kind == .submitted } : events.snapshot().allSatisfy { $0.kind != .submitted })
        }
    }

    private func checkPairwise<O: VectorProtocol & Sendable>(_ rows: [O]) async where O.Scalar == Float {
        let euclidean = await BatchOperations.pairwiseDistances(rows,metric:EuclideanDistance())
        let cosine = await BatchOperations.pairwiseDistances(rows,metric:CosineDistance())
        for i in rows.indices {
            #expect(euclidean[i][i].bitPattern == EuclideanDistance().distance(rows[i],rows[i]).bitPattern)
            #expect(cosine[i][i].bitPattern == CosineDistance().distance(rows[i],rows[i]).bitPattern)
        }
        let calls = E06Values<Int>()
        let custom = await BatchOperations.pairwiseDistances(rows,metric:E06Metric(scale:2,calls:calls))
        #expect(custom.count == rows.count)
        for i in rows.indices { #expect(custom[i][i] == 7) }
    }
}
