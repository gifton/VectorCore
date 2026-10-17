import Foundation
import Darwin

// Standalone driver: fixture path, module search directory, C module map, evidence directory.
// The Python caller (E03RunTypecheck.py) resolves the module directory and module map for the
// native or swiftbuild layout through Scripts/ci/swiftpm_products.py.
let fixture = CommandLine.arguments[1]
let modules = CommandLine.arguments[2]
let moduleMap = CommandLine.arguments[3]
let evidence = URL(fileURLWithPath: CommandLine.arguments[4], isDirectory: true)
try FileManager.default.createDirectory(at: evidence, withIntermediateDirectories: true)
let common = ["swiftc", "-typecheck", "-swift-version", "6", "-strict-concurrency=complete",
              "-target", "arm64-apple-macosx14.0", "-module-cache-path", evidence.appendingPathComponent("E03-consumer-cache").path,
              "-I", modules,
              "-Xcc", "-fmodule-map-file=" + moduleMap, fixture]
var records: [[String: Any]] = []
var valid = true
for negative in [false, true] {
    let name = negative ? "negative" : "positive"
    let output = evidence.appendingPathComponent("E03-consumer-\(name).log")
    _ = FileManager.default.createFile(atPath: output.path, contents: nil)
    let file = try FileHandle(forWritingTo: output)
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
    process.arguments = common + (negative ? ["-D", "E03_NEGATIVE_CAPTURE"] : [])
    process.standardOutput = file
    process.standardError = file
    let terminated = DispatchSemaphore(value: 0)
    process.terminationHandler = { _ in terminated.signal() }
    try process.run()
    if terminated.wait(timeout: .now() + 15) == .timedOut {
        if process.isRunning { kill(process.processIdentifier, SIGKILL) }
        let cleaned = terminated.wait(timeout: .now() + 2) == .success
        try file.close()
        records.append(["name": name, "argv": ["/usr/bin/xcrun"] + process.arguments!, "timedOut": true, "cleaned": cleaned])
        valid = false
        continue
    }
    try file.close()
    let diagnostic = try String(contentsOf: output, encoding: .utf8)
    let intended = diagnostic.contains("non-Sendable type 'MutableState'") && diagnostic.contains("'@Sendable' closure")
    valid = valid && (negative ? process.terminationStatus != 0 && intended : process.terminationStatus == 0)
    records.append(["name": name, "argv": ["/usr/bin/xcrun"] + process.arguments!,
                    "exit": process.terminationStatus, "intendedCaptureDiagnostic": intended, "log": output.path])
    print("E03 consumer \(name) exit=\(process.terminationStatus) intendedCaptureDiagnostic=\(intended)")
}
let data = try JSONSerialization.data(withJSONObject: records, options: [.prettyPrinted, .sortedKeys])
try data.write(to: evidence.appendingPathComponent("E03-consumer-results.json"))
exit(valid ? 0 : 1)
