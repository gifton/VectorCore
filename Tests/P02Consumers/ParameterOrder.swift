// P02 negative consumer (phase 1 check map P2): `cpuStrategy` is the final parameter of
// Operations.findNearestBatch, so an argument list that puts it before `metric:` must not compile.
// Scripts/ci/tests/P02RunConsumers.py expects exactly one error, naming `metric` and `cpuStrategy`.
import VectorCore

func p02StrategyBeforeMetric(_ rows: [Vector512Optimized]) async throws -> Int {
    try await Operations.findNearestBatch(queries: rows, in: rows, k: 1, cpuStrategy: .perQuery,
                                          metric: EuclideanDistance()).count
}
