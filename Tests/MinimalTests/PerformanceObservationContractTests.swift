import Foundation
import XCTest

final class PerformanceObservationContractTests: XCTestCase {
    func testIsolatedObservation() {
        guard let mode = ProcessInfo.processInfo.environment["V03_OBSERVATION_CASE"] else { return }
        print("V03_OBSERVATION_BODY \(mode)")
        var count = 0
        observePerformance("V03.probe.observation") {
            count += 1
            XCTAssertEqual(mode == "numeric" ? 2 + 2 : 5, 5, "V03 body numerical control")
        }
        if ProcessInfo.processInfo.environment["VECTORCORE_STRICT_PERF"] == "1" {
            XCTAssertGreaterThan(count, 1, "Strict mode must delegate to XCTest measurement")
        } else {
            XCTAssertEqual(count, 1, "Disabled strict mode must execute the body exactly once")
        }
        print("V03_OBSERVATION_COMPLETE bodies=\(count)")
    }
}
