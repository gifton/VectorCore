import Foundation
import Testing
import XCTest

/// Clock-derived assertions are opt-in. All values other than exactly "1" disable
/// predicate evaluation. Numerical checks and liveness deadlines remain independent.
/// The explicit site inventory is Scripts/ci/performance_sites.json.
let strictPerfGatesEnabled: Bool = ProcessInfo.processInfo.environment["VECTORCORE_STRICT_PERF"] == "1"

func expectPerformance(
    _ site: String,
    _ condition: @autoclosure () -> Bool,
    _ message: @autoclosure () -> String = "Performance predicate failed",
    sourceLocation: SourceLocation = #_sourceLocation
) {
    guard strictPerfGatesEnabled else { return }
    if !condition() {
        Issue.record("\(site): \(message())", sourceLocation: sourceLocation)
    }
}

func expectXCTestPerformance(
    _ site: String,
    _ condition: @autoclosure () -> Bool,
    _ message: @autoclosure () -> String = "Performance predicate failed",
    file: StaticString = #filePath,
    line: UInt = #line
) {
    guard strictPerfGatesEnabled else { return }
    if !condition() {
        XCTFail("\(site): \(message())", file: file, line: line)
    }
}
