import Foundation
import Darwin
import Testing
@testable import VectorCore

func e06WithRestoredConfiguration(_ body: () async throws -> Void) async throws {
    let original = await BatchOperations.configuration()
    do { try await body() }
    catch {
        try await BatchOperations.updateConfiguration { $0 = original }
        #expect(BatchConfigurationContractTests.bits(await BatchOperations.configuration()) == BatchConfigurationContractTests.bits(original))
        throw error
    }
    try await BatchOperations.updateConfiguration { $0 = original }
    #expect(BatchConfigurationContractTests.bits(await BatchOperations.configuration()) == BatchConfigurationContractTests.bits(original))
}

func e06IsolatedScenario(_ scenario: String, filter: String, _ body: () async throws -> Void) async throws {
    if ProcessInfo.processInfo.environment["E06_ISOLATED"] == scenario {
        let invocation: [String: Any] = ["argv": CommandLine.arguments, "scenario": scenario,
            "DYLD_FRAMEWORK_PATH": ProcessInfo.processInfo.environment["DYLD_FRAMEWORK_PATH"] ?? "",
            "DYLD_LIBRARY_PATH": ProcessInfo.processInfo.environment["DYLD_LIBRARY_PATH"] ?? ""]
        print("E06_INVOCATION " + String(decoding: try JSONSerialization.data(withJSONObject: invocation, options: [.sortedKeys]), as: UTF8.self))
        try await e06WithRestoredConfiguration(body)
        print("E06_COMPLETE \(scenario)")
        return
    }
    try e06RunIsolatedChild(scenario, filter: filter)
}

private func e06RunIsolatedChild(_ scenario: String, filter: String) throws {
    var environment = ProcessInfo.processInfo.environment
    environment["E06_ISOLATED"] = scenario
    let directory = environment["E06_CAPTURE_DIRECTORY"].map { URL(fileURLWithPath: $0) } ?? FileManager.default.temporaryDirectory
    let path = directory.appendingPathComponent("E06-\(scenario)-\(UUID().uuidString).log")
    FileManager.default.createFile(atPath: path.path, contents: nil)
    let file = try FileHandle(forWritingTo: path)
    let process = try launchChildProbe(.swiftTesting(filter: filter), environment: environment, output: file,
                                       deadline: 15, cleanup: .terminateThenKill(grace: 1, seconds: 1))
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
