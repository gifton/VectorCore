import Foundation
import Testing
import VectorCore

private final class DestructionHook: Sendable {
    let callback: @Sendable () -> Void
    init(_ callback: @escaping @Sendable () -> Void) { self.callback = callback }
    deinit { callback() }
}
private struct DestructionVector: IndexableVector {
    typealias Scalar = Float
    typealias Storage = [Float]
    var storage: [Float]
    let hook: DestructionHook?
    var scalarCount: Int { storage.count }
    init() { storage = []; hook = nil }
    init(repeating value: Float) { storage = [value]; hook = nil }
    init(_ array: [Float]) { storage = array; hook = nil }
    init(_ array: [Float], hook: DestructionHook) { storage = array; self.hook = hook }
    func toArray() -> [Float] { storage }
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }
    mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer { try body($0) }
    }
    static func == (a: Self, b: Self) -> Bool { a.storage == b.storage }
    func hash(into hasher: inout Hasher) { hasher.combine(storage) }
    init(from decoder: any Decoder) throws {
        storage = try decoder.singleValueContainer().decode([Float].self); hook = nil
    }
    func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer(); try c.encode(storage)
    }
}

@Suite struct CollectionDestructionContractTests {
    private func emit(_ text: String) {
        FileHandle.standardError.write(Data((text + "\n").utf8))
    }

    @inline(never)
    private func populate(_ collection: SimpleVectorCollection<DestructionVector>, scenario: String) throws {
        let ids = scenario == "removeAll" ? [0, 1, 2] : [0]
        for id in ids {
            try collection.add(id: id, vector: DestructionVector([1], hook: DestructionHook { [weak collection] in
                FileHandle.standardError.write(Data("F002_DEINIT_ENTERED\n".utf8))
                guard let collection else { Issue.record("Collection released before callback"); return }
                let count = collection.count
                if scenario == "update" {
                    #expect(count == 1 && collection.vector(for: 0)?.toArray() == [2])
                } else {
                    #expect(collection.allIDs.allSatisfy { $0 >= 100 })
                    #expect(count < ids.count)
                }
                do { try collection.add(id: 100 + id, vector: DestructionVector([3])) }
                catch { Issue.record("Reentrant insertion failed: \(error)") }
            }))
        }
    }

    @Test func destructionChild() throws {
        guard let scenario = ProcessInfo.processInfo.environment["F002_DESTRUCTION"] else { return }
        let collection = SimpleVectorCollection<DestructionVector>(dimension: 1)
        try populate(collection, scenario: scenario)
        switch scenario {
        case "update": try collection.update(id: 0, vector: DestructionVector([2]))
        case "removeAll": collection.removeAll()
        case "remove": _ = collection.remove(id: 0)
        default: Issue.record("Unknown destruction scenario"); return
        }
        #expect(Set(collection.allIDs) == (scenario == "update" ? [0, 100] : scenario == "removeAll" ? [100, 101, 102] : [100]))
        emit("F002_DESTRUCTION_COMPLETE")
    }

    @Test(arguments: ["update", "removeAll", "remove"])
    func destructionCanReenter(_ scenario: String) throws {
        let environment = ProcessInfo.processInfo.environment.merging(["F002_DESTRUCTION": scenario]) { _, new in new }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("F002-\(scenario)-\(UUID().uuidString).log")
        _ = FileManager.default.createFile(atPath: url.path, contents: nil)
        let output = try FileHandle(forWritingTo: url)
        defer { try? output.close() }
        let result = try launchChildProbe(.swiftTesting(filter: "CollectionDestructionContractTests/destructionChild"),
                                          environment: environment, output: output, deadline: 5, cleanup: .kill(seconds: 2))
        let diagnostic = try String(contentsOf: url, encoding: .utf8)
        #expect(diagnostic.contains("F002_DEINIT_ENTERED"), "Child never exercised destruction: \(url.path)")
        try #require(!result.timedOut, "Destructor re-entry deadlocked: \(url.path)")
        #expect(result.terminationStatus == 0 && result.terminationReason == .exit, "\(diagnostic)")
        #expect(diagnostic.contains("F002_DESTRUCTION_COMPLETE"))
        print("F002 child \(scenario): \(url.path)")
    }
}
