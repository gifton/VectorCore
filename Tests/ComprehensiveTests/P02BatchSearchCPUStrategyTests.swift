import Foundation
import Testing
@testable import VectorCore

// P02 phase B: the public `BatchSearchCPUStrategy` enum (phase 1 check map P1). The exhaustive `switch` below has no
// `default`, so adding, removing or renaming a case, or changing the `.gram` labels, fails to compile; the generic
// `Sendable` requirement fails to compile if the conformance is dropped. The public (non-testable) view is checked
// by the phase B consumer type-check (P02/phaseB/consumer).

private func p02RequireSendable<T: Sendable>(_: T.Type) -> Bool { true }

private func p02Describe(_ strategy: BatchSearchCPUStrategy) -> String {
    switch strategy {
    case .perQuery:
        return "perQuery"
    case let .gram(queryTileRows: r, candidateTileRows: c, maximumTemporaryPayloadBytes: cap):
        return "gram(\(r),\(c),\(cap))"
    }
}

@Suite("P02 BatchSearchCPUStrategy")
struct P02BatchSearchCPUStrategyTests {
    @Test func exactlyTwoCasesAndSendable() {
        #expect(p02RequireSendable(BatchSearchCPUStrategy.self))
        let strategies: [BatchSearchCPUStrategy] = [
            .perQuery,
            .gram(queryTileRows: 8, candidateTileRows: 256, maximumTemporaryPayloadBytes: 1 << 20),
            .gram(queryTileRows: 0, candidateTileRows: -1, maximumTemporaryPayloadBytes: -1)
        ]
        #expect(strategies.map(p02Describe) == ["perQuery", "gram(8,256,1048576)", "gram(0,-1,-1)"])
    }
}
