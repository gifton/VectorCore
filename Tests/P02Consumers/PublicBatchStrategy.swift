// P02 strict Swift 6 public consumer (phase 1 check map S4, P1, P2): `import VectorCore` only, never @testable.
// Run by Scripts/ci/tests/P02RunConsumers.py (typecheck, link and run against a build's objects).
// - an exhaustive `switch` over BatchSearchCPUStrategy with exactly its two cases and no `default` (a third case or a
//   missing one fails to compile: the package is not built with library evolution, so the enum is exhaustive here);
// - a `Sendable` constraint on the enum;
// - both call forms of Operations.findNearestBatch: the four-argument form (default `.perQuery`) and the five-argument
//   form with `cpuStrategy:` as the final argument (ParameterOrder.swift in this directory is the negative: it must
//   not compile).
import VectorCore

func p02RequireSendable<T: Sendable>(_: T.Type) {}

func p02Describe(_ strategy: BatchSearchCPUStrategy) -> String {
    switch strategy {
    case .perQuery:
        return "perQuery"
    case let .gram(queryTileRows, candidateTileRows, maximumTemporaryPayloadBytes):
        return "gram(\(queryTileRows), \(candidateTileRows), \(maximumTemporaryPayloadBytes))"
    }
}

@main struct P02PublicBatchStrategyConsumer {
    static func main() async throws {
        p02RequireSendable(BatchSearchCPUStrategy.self)
        let gram = BatchSearchCPUStrategy.gram(queryTileRows: 2, candidateTileRows: 3, maximumTemporaryPayloadBytes: .max)
        precondition(p02Describe(.perQuery) == "perQuery")
        precondition(p02Describe(gram) == "gram(2, 3, \(Int.max))")

        let candidates = try (0..<5).map { i in
            try Vector512Optimized((0..<512).map { Float(($0 + 3 * i) % 7) - 3 })
        }
        let queries = Array(candidates.prefix(3))
        // Four arguments: the default CPU strategy (.perQuery), one row per query.
        let fourArguments = try await Operations.findNearestBatch(queries: queries, in: candidates, k: 2,
                                                                  metric: EuclideanDistance())
        // Five arguments: cpuStrategy is the final parameter.
        let explicitDefault = try await Operations.findNearestBatch(queries: queries, in: candidates, k: 2,
                                                                    metric: EuclideanDistance(), cpuStrategy: .perQuery)
        let viaGram = try await Operations.findNearestBatch(queries: queries, in: candidates, k: 2,
                                                            metric: EuclideanDistance(), cpuStrategy: gram)
        precondition(fourArguments == explicitDefault)
        precondition(fourArguments.count == 3 && viaGram.count == 3)
        precondition(fourArguments.allSatisfy { $0.count == 2 } && viaGram.allSatisfy { $0.count == 2 })
        // Each query is one of the candidates, so both routes put it first at index i (its own row, distance 0 on
        // .perQuery; the Gram score of an exact duplicate is within its documented bound, not necessarily 0).
        for i in 0..<3 {
            precondition(fourArguments[i][0].index == i && fourArguments[i][0].distance == 0)
            precondition(viaGram[i][0].index == i)
        }
        // A .gram request is checked before any work: a zero payload limit throws .resourceUnavailable.
        do {
            _ = try await Operations.findNearestBatch(
                queries: queries, in: candidates, k: 2, metric: EuclideanDistance(),
                cpuStrategy: .gram(queryTileRows: 2, candidateTileRows: 3, maximumTemporaryPayloadBytes: 0))
            preconditionFailure("a zero payload limit must throw")
        } catch let error as VectorError {
            precondition(error.kind == .resourceUnavailable)
        }
        print("P02_PUBLIC_BATCH_STRATEGY_COMPLETE")
    }
}
