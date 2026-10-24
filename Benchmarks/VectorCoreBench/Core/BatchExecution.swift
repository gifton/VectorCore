import VectorCoreBenchmarking

enum BatchExecution {
    static func run(name: String, operation: () async throws -> [BenchResult]) async -> [BenchResult] {
        do { return try await operation() }
        catch { return [BenchResult(name: name, iterations: 0, totalNanoseconds: 0, executionFailure: String(describing: error))] }
    }
}
