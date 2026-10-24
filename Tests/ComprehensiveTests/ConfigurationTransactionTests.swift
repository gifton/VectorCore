import Foundation
import Darwin
import Testing
@testable import VectorCore

@Suite(.serialized)
struct ConfigurationTransactionTests {
    private func concurrentBatchIncrementsPreserveAllUpdatesBody() async throws {
        let original = await BatchOperations.configuration()
        for round in 0..<3 {
            try await BatchOperations.updateConfiguration { $0.parallelThreshold = 1 }
            try await withThrowingTaskGroup(of: Void.self) { group in
                for _ in 0..<100 {
                    group.addTask { try await BatchOperations.updateConfiguration { $0.parallelThreshold += 1 } }
                }
                try await group.waitForAll()
            }
            let result = await BatchOperations.configuration()
            print("E03 increment round=\(round) actual=\(result.parallelThreshold) expected=101")
            #expect(result.parallelThreshold == 101)
        }
        try await BatchOperations.updateConfiguration { $0 = original }
    }

    private func concurrentIndependentFieldsPreserveBothTotalsBody() async throws {
        let original = await BatchOperations.configuration()
        for round in 0..<3 {
            try await BatchOperations.updateConfiguration { $0.parallelThreshold = 1; $0.minimumChunkSize = 1 }
            try await withThrowingTaskGroup(of: Void.self) { group in
                for _ in 0..<100 {
                    group.addTask { try await BatchOperations.updateConfiguration { $0.parallelThreshold += 1 } }
                    group.addTask { try await BatchOperations.updateConfiguration { $0.minimumChunkSize += 1 } }
                }
                try await group.waitForAll()
            }
            let result = await BatchOperations.configuration()
            print("E03 fields round=\(round) threshold=\(result.parallelThreshold) chunk=\(result.minimumChunkSize) expected=101,101")
            #expect(result.parallelThreshold == 101)
            #expect(result.minimumChunkSize == 101)
        }
        try await BatchOperations.updateConfiguration { $0 = original }
    }
}

extension ConfigurationTransactionTests {
    private struct LocalValue: Sendable, Equatable {
        var first: Int = 0
        var second: Int = 0
    }
    private final class Sentinel: Error, Sendable {}

    private static func snapshot(_ value: BatchOperations.Configuration) -> [UInt64] {
        [UInt64(bitPattern: Int64(value.parallelThreshold)), value.oversubscription.bitPattern,
         UInt64(bitPattern: Int64(value.minimumChunkSize)), UInt64(bitPattern: Int64(value.defaultBatchSize)),
         value.enableMatrixRouting ? 1 : 0, UInt64(bitPattern: Int64(value.matrixRoutingMinN))]
    }

    private static func assign(_ value: inout BatchOperations.Configuration, generation: Int) {
        value.parallelThreshold = generation
        value.oversubscription = Double(generation)
        value.minimumChunkSize = generation + 1
        value.defaultBatchSize = generation + 2
        value.enableMatrixRouting = generation.isMultiple(of: 2)
        value.matrixRoutingMinN = generation + 3
    }

    @Test func actorModifySerializesIncrementsAndExistingControls() async {
        let configuration = ThreadSafeConfiguration(LocalValue())
        for round in 0..<3 {
            await configuration.update(LocalValue())
            await withTaskGroup(of: Void.self) { group in
                for _ in 0..<100 {
                    group.addTask { await configuration.modify { $0.first += 1 } }
                    group.addTask { await configuration.modify { $0.second += 1 } }
                }
            }
            let value = await configuration.get()
            print("E03 actor round=\(round) first=\(value.first) second=\(value.second) expected=100,100")
            #expect(value == LocalValue(first: 100, second: 100))
        }
        await configuration.update(LocalValue(first: 4, second: 5))
        await configuration.update(\.first, to: 9)
        #expect(await configuration.get() == LocalValue(first: 9, second: 5))
    }

    private func batchRollbackPreservesEveryFieldAndErrorIdentityBody() async throws {
        let original = await BatchOperations.configuration()
        try await BatchOperations.updateConfiguration { value in
            Self.assign(&value, generation: 11)
            value.oversubscription = 0.5
        }
        let before = await BatchOperations.configuration()
        let sentinel = Sentinel()
        do {
            try await BatchOperations.updateConfiguration { value in
                Self.assign(&value, generation: 24)
                throw sentinel
            }
            Issue.record("Expected the sentinel error")
        } catch {
            #expect((error as? Sentinel) === sentinel)
        }
        #expect(Self.snapshot(await BatchOperations.configuration()) == Self.snapshot(before))
        try await BatchOperations.updateConfiguration { Self.assign(&$0, generation: 31) }
        var expected = BatchOperations.Configuration()
        Self.assign(&expected, generation: 31)
        #expect(Self.snapshot(await BatchOperations.configuration()) == Self.snapshot(expected))
        try await BatchOperations.updateConfiguration { $0 = original }
        #expect(Self.snapshot(await BatchOperations.configuration()) == Self.snapshot(original))
        print("E03 rollback six-fields identity=true restored=true")
    }

    @Test func actorRollbackPreservesValueFields() async {
        var initial = BatchOperations.Configuration()
        Self.assign(&initial, generation: 13)
        initial.oversubscription = Double(bitPattern: 0x7ff8000000000001)
        let configuration = ThreadSafeConfiguration(initial)
        let sentinel = Sentinel()
        do {
            try await configuration.modify { value in
                Self.assign(&value, generation: 40)
                throw sentinel
            }
            Issue.record("Expected the sentinel error")
        } catch {
            #expect((error as? Sentinel) === sentinel)
        }
        #expect(Self.snapshot(await configuration.get()) == Self.snapshot(initial))
        await configuration.modify { Self.assign(&$0, generation: 9) }
        var expected = BatchOperations.Configuration()
        Self.assign(&expected, generation: 9)
        #expect(Self.snapshot(await configuration.get()) == Self.snapshot(expected))
    }

    private func concurrentSnapshotsObserveCompleteTuplesBody() async throws {
        let original = await BatchOperations.configuration()
        try await BatchOperations.updateConfiguration { Self.assign(&$0, generation: 1) }
        let valid = try await withThrowingTaskGroup(of: Bool.self, returning: Bool.self) { group in
            for generation in 1...100 {
                group.addTask {
                    try await BatchOperations.updateConfiguration { Self.assign(&$0, generation: generation) }
                    return true
                }
                group.addTask {
                    let value = await BatchOperations.configuration()
                    var expected = BatchOperations.Configuration()
                    Self.assign(&expected, generation: value.parallelThreshold)
                    return Self.snapshot(value) == Self.snapshot(expected)
                }
            }
            var allValid = true
            for try await observation in group { allValid = allValid && observation }
            return allValid
        }
        #expect(valid)
        print("E03 tuple observations=100 valid=\(valid)")
        try await BatchOperations.updateConfiguration { $0 = original }
    }
}


extension ConfigurationTransactionTests {
    private func runGlobalChild(_ scenario: String, result: String) throws {
        var environment = ProcessInfo.processInfo.environment
        environment["E03_GLOBAL_SCENARIO"] = scenario
        let directory = environment["E03_CHILD_LOG_DIR"] ?? "/private/tmp"
        let output = URL(fileURLWithPath: "\(directory)/E03-child-\(scenario)-\(UUID().uuidString).log")
        _ = FileManager.default.createFile(atPath: output.path, contents: nil)
        let file = try FileHandle(forWritingTo: output)
        let process = try launchChildProbe(.swiftTesting(filter: "ConfigurationTransactionTests/globalChildProbe"),
                                           environment: environment, output: file, deadline: 15, cleanup: .kill(seconds: 2))
        if process.timedOut {
            let cleaned = process.cleaned
            try file.close()
            Issue.record("E03 child deadline: \(output.path), cleaned=\(cleaned)")
            return
        }
        try file.close()
        let log = try String(contentsOf: output, encoding: .utf8)
        #expect(process.terminationStatus == 0, "\(output.path): \(log)")
        #expect(log.contains("E03_BODY_BEGIN \(scenario)"), "\(output.path): \(log)")
        #expect(log.contains("E03_BODY_END \(scenario)"), "\(output.path): \(log)")
        #expect(log.contains(result), "\(output.path): \(log)")
        print("E03 child \(scenario) exit=\(process.statusDescription) log=\(output.path)")
    }

    @Test func concurrentBatchIncrementsPreserveAllUpdates() throws {
        try runGlobalChild("increments", result: "E03 increment round=2 actual=101 expected=101")
    }

    @Test func concurrentIndependentFieldsPreserveBothTotals() throws {
        try runGlobalChild("independent", result: "E03 fields round=2 threshold=101 chunk=101 expected=101,101")
    }

    @Test func batchRollbackPreservesEveryFieldAndErrorIdentity() throws {
        try runGlobalChild("rollback", result: "E03 rollback six-fields identity=true restored=true")
    }

    @Test func concurrentSnapshotsObserveCompleteTuples() throws {
        try runGlobalChild("snapshots", result: "E03 tuple observations=100 valid=true")
    }

    @Test func globalChildProbe() async throws {
        guard let scenario = ProcessInfo.processInfo.environment["E03_GLOBAL_SCENARIO"] else { return }
        print("E03_BODY_BEGIN \(scenario)")
        try await e06WithRestoredConfiguration {
        switch scenario {
        case "increments": try await concurrentBatchIncrementsPreserveAllUpdatesBody()
        case "independent": try await concurrentIndependentFieldsPreserveBothTotalsBody()
        case "rollback": try await batchRollbackPreservesEveryFieldAndErrorIdentityBody()
        case "snapshots": try await concurrentSnapshotsObserveCompleteTuplesBody()
        default: Issue.record("Unknown E03 global scenario \(scenario)")
        }
        }
        print("E03_BODY_END \(scenario)")
    }
}
