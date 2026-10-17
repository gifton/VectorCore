import Foundation
import VectorCore

struct MinimalProvider: ComputeProvider {
    let device: ComputeDevice = .cpu
    var maxConcurrency: Int { 1 }
    var deviceInfo: ComputeDeviceInfo {
        .init(name: "single", maxThreads: 1, preferredChunkSize: 1)
    }
    func execute<T: Sendable>(_ body: @Sendable @escaping () async throws -> T) async throws -> T {
        try await body()
    }
}

actor Counter {
    var current = 0
    var maximum = 0
    func enter() { current += 1; maximum = max(maximum, current) }
    func leave() { current -= 1 }
}

@main struct Probe {
    static func main() async throws {
        try await run()
    }
    nonisolated static func run() async throws {
        let mode = CommandLine.arguments.dropFirst().first ?? "normal"
        if mode == "overflow" {
            let result = try await CPUComputeProvider.parallel.parallelExecute(items: (Int.max - 2)..<Int.max) { $0 }
            print(result)
            return
        }
        if mode == "zero-batch" {
            let result = try await BatchOperations.process([DynamicVector([1])], batchSize: 0) { $0 }
            print(result.count)
            return
        }
        let original = await BatchOperations.configuration()
        await BatchOperations.updateConfiguration { $0.defaultBatchSize = 2 }
        let sizes: [Int] = try await BatchOperations.process(Array(repeating: DynamicVector([1]), count: 5)) { [$0.count] }
        print("configuredBatch=2 observedBatches=\(sizes)")
        let negative: [Int] = try await BatchOperations.process([DynamicVector([1])], batchSize: -1) { [$0.count] }
        print("negativeBatchResults=\(negative)")
        await BatchOperations.updateConfiguration { $0.parallelThreshold = 0 }
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<100 {
                group.addTask { await BatchOperations.updateConfiguration { $0.parallelThreshold += 1 } }
            }
        }
        print("100ConcurrentConfigIncrements=\((await BatchOperations.configuration()).parallelThreshold)")
        await BatchOperations.updateConfiguration { $0 = original }
        let counter = Counter()
        try await MinimalProvider().parallelForEach(items: 0..<64) { _ in
            await counter.enter()
            try await Task.sleep(for: .milliseconds(5))
            await counter.leave()
        }
        print("providerMaxConcurrency=1 observedPeak=\(await counter.maximum)")
    }
}
