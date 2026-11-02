import Foundation
import XCTest

extension XCTestCase {
    /// Strict-off observation executes the body once. Numeric body assertions stay active.
    /// Strict-on preserves XCTest measurement behavior, independently of test enablement.
    func observePerformance(
        _ site: String,
        block: () -> Void
    ) {
        if ProcessInfo.processInfo.environment["VECTORCORE_STRICT_PERF"] == "1" {
            measure(block)
        } else {
            let start = ProcessInfo.processInfo.systemUptime
            block()
            let elapsed = ProcessInfo.processInfo.systemUptime - start
            print("PERF_OBSERVATION \(site) seconds=\(elapsed) bodies=1")
        }
    }
}
