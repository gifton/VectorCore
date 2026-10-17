import Foundation
import Testing
import XCTest

private func gateProbeCondition() -> Bool {
    print("V03_PERFORMANCE_PREDICATE_EVALUATED")
    return ProcessInfo.processInfo.environment["V03_GATE_CASE"] == "true"
}

struct PerformanceGateContractTests {
    @Test func isolatedProbe() {
        guard let mode = ProcessInfo.processInfo.environment["V03_GATE_CASE"] else { return }
        print("V03_GATE_BODY swift \(mode) strict=\(strictPerfGatesEnabled)")
        if mode == "numeric" {
            #expect(2 + 2 == 5, "V03 numerical control must remain active")
        } else {
            expectPerformance("V03.probe.swift", gateProbeCondition())
        }
        print("V03_GATE_COMPLETE swift")
    }
}

final class XCTestPerformanceGateContractTests: XCTestCase {
    func testIsolatedProbe() {
        guard let mode = ProcessInfo.processInfo.environment["V03_GATE_CASE"] else { return }
        print("V03_GATE_BODY xctest \(mode) strict=\(strictPerfGatesEnabled)")
        if mode == "numeric" {
            XCTAssertEqual(2 + 2, 5, "V03 numerical control must remain active")
        } else {
            expectXCTestPerformance("V03.probe.xctest", gateProbeCondition())
        }
        print("V03_GATE_COMPLETE xctest")
    }
}
