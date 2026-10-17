import VectorCore

struct DefaultScheduler: ComputeProvider {
    let maxConcurrency: Int
    var device: ComputeDevice { .cpu }
    var deviceInfo: ComputeDeviceInfo { .init(name: "External default", maxThreads: 64, preferredChunkSize: 1) }
    func execute<T: Sendable>(_ body: @Sendable @escaping () async throws -> T) async throws -> T {
        try Task.checkCancellation(); let result = try await body(); try Task.checkCancellation(); return result
    }
}
struct InheritedBatchScheduler: BatchKernelProvider {
    let maxConcurrency = 2
    var device: ComputeDevice { .cpu }
    var deviceInfo: ComputeDeviceInfo { .init(name: "External batch", maxThreads: 64, preferredChunkSize: 1) }
    func execute<T: Sendable>(_ body: @Sendable @escaping () async throws -> T) async throws -> T {
        try Task.checkCancellation(); let result = try await body(); try Task.checkCancellation(); return result
    }
    func batchDistance<V: VectorProtocol>(query: V, candidates: [V], metric: any DistanceMetric) async throws -> [Float] where V.Scalar == Float {
        candidates.map { _ in query[0] }
    }
    func findNearest<V: VectorProtocol>(query: V, candidates: [V], k: Int, metric: any DistanceMetric) async throws -> [(index: Int, distance: Float)] where V.Scalar == Float {
        [(Int(query[0]),query[0])]
    }
}
@main struct PublicSchedulers {
    static func main() async throws {
        let provider = DefaultScheduler(maxConcurrency: 2)
        let values = try await provider.parallelExecute(items: -3..<4) { index in
            try await CPUComputeProvider.parallel.parallelReduce(items: 0..<3, initial: 0, { range in range.count }, +) + index - 3
        }
        guard values == Array(-3..<4) else { fatalError("public ordering") }
        let text = try await provider.parallelReduce(items: 0..<13, initial: "", minChunk: 3,
            { $0.map(String.init).joined(separator: ",") + "," }, +)
        guard text == "0,1,2,3,4,5,6,7,8,9,10,11,12," else { fatalError("public reduction") }
        let queries = (0..<9).map { DynamicVector([Float($0)]) }
        let batch = try await InheritedBatchScheduler().findNearestBatch(queries: queries, candidates: queries, k: 1, metric: EuclideanDistance())
        guard batch.map({ $0[0].index }) == Array(0..<9) else { fatalError("inherited batch order") }
        do {
            _ = try await DefaultScheduler(maxConcurrency: 0).parallelExecute(items: 0..<0) { $0 }
            fatalError("nonpositive limit accepted")
        } catch let error as VectorError { guard error.kind == .invalidConfiguration else { throw error } }
        print("E02_PUBLIC_DEFAULT_AND_BATCH_COMPLETE")
    }
}
