import Foundation
import Testing
@testable import VectorCore

/// P01 TopKBuffer capacity behaviour (phase C), observed through the internal buffer in bounded children:
/// a zero-capacity buffer ignores pushes; a negative capacity traps before allocation (control).
/// Was RED before phase C; `P01RedGate.topKBuffer` is lifted, so the suite runs by default.
@Suite(.serialized, .enabled(if: P01RedGate.topKBuffer))
struct P01TopKBufferRedTests {
    static let childFilter = "P01TopKBufferRedTests/p01Child"

    @Test func zeroCapacityTopKBufferIgnoresPushInChild() throws {
        let child = try p01RunChild("topk-zero-capacity", filter: Self.childFilter)
        #expect(child.run.terminationStatus == 0 && child.run.terminationReason == .exit,
                "zero-capacity push child status=\(child.run.statusDescription) reason=\(child.run.reasonDescription): \(child.tail)")
        #expect(child.output.contains("P01_CHILD_COMPLETE topk-zero-capacity"), "\(child.tail)")
    }

    /// Control (green before and after P01): a negative capacity traps before any buffer exists.
    @Test func negativeCapacityTopKBufferTrapsControl() throws {
        let child = try p01RunChild("topk-negative-capacity", filter: Self.childFilter)
        #expect(child.run.terminationReason == .uncaughtSignal, "status=\(child.run.statusDescription): \(child.tail)")
        #expect(child.output.contains("P01_OBSERVED topk-negative-capacity before-init"))
        #expect(!child.output.contains("P01_OBSERVED topk-negative-capacity after-init"))
    }

    // MARK: Child entry

    /// Returns at once unless the parent selected a scenario.
    @Test func p01Child() async throws {
        guard let scenario = ProcessInfo.processInfo.environment["P01_SCENARIO"] else { return }
        p01Marker("P01_CHILD_BEGIN \(scenario)")
        switch scenario {
        case "topk-zero-capacity":
            for maximize in [false, true] {
                var buffer = TopKBuffer(k: 0, isMinHeap: maximize)
                p01Marker("P01_OBSERVED topk-zero-capacity maximize=\(maximize) before-push size=\(buffer.size)")
                buffer.pushIfBetter(val: 1, idx: 0)
                buffer.pushIfBetter(val: .nan, idx: 1)
                p01Marker("P01_OBSERVED topk-zero-capacity maximize=\(maximize) after-push size=\(buffer.size)")
                #expect(buffer.size == 0)
            }
        case "topk-negative-capacity":
            p01Marker("P01_OBSERVED topk-negative-capacity before-init")
            let buffer = TopKBuffer(k: -1, isMinHeap: false)
            p01Marker("P01_OBSERVED topk-negative-capacity after-init size=\(buffer.size)")
        default:
            Issue.record("Unknown P01 scenario \(scenario)")
        }
        p01Marker("P01_CHILD_COMPLETE \(scenario)")
    }
}
