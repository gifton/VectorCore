import Foundation
import VectorCore

private struct Failure: Error { let message: String }
private func require(_ value: Bool, _ message: String) throws {
    guard value else { throw Failure(message: message) }
    print("Q02_CHECK \(message)")
    fflush(nil)
}
private func measured(_ width: Int, _ count: Int) -> WorkloadCharacteristics {
    WorkloadCharacteristics(dimension: width, batchSize: count, precisionMode: .fp32Full,
                            maximumProfiledMeanLatency: 1_000_000)
}
private func expectMiss(_ tuner: KernelAutoTuner, _ workload: WorkloadCharacteristics) async throws {
    do { _ = try await tuner.selectStrategy(for: workload); throw Failure(message: "expected profile miss") }
    catch let error as VectorError { try require(error.kind == .resourceUnavailable, "profile miss") }
}
private actor ActorConsumer {
    func select(_ tuner: KernelAutoTuner) async throws -> KernelStrategy {
        try await tuner.selectStrategy(for: WorkloadCharacteristics(dimension: 512, batchSize: 1, precisionMode: .fp32Full))
    }
}
@MainActor private func mainActorConsumer(_ tuner: MixedPrecisionAutoTuner) async throws -> MixedPrecisionStrategy {
    try await tuner.selectStrategy(candidateCount: 1)
}

@main struct Q02PublicConsumer {
    static func main() async throws {
        print("Q02_PUBLIC_BODY_BEGIN")
        fflush(nil)
        let directory = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        let url = directory.appendingPathComponent("profiles.json")
        let a = WorkloadCharacteristics(dimension: 512, batchSize: 1, precisionMode: .fp32Full)
        let b = WorkloadCharacteristics(dimension: 768, batchSize: 2, precisionMode: .fp32Full)
        let first = KernelAutoTuner(persistenceURL: url)
        try require(await first.persistenceStatus == .enabled, "explicit persistence enabled")
        let profilesA = try await first.profileWorkload(a, iterations: 2, warmupIterations: 1)
        try require(profilesA.count == 2, "actual calibration A")
        let selectedA = try await first.selectStrategy(for: measured(512,1))
        let override: KernelStrategy = selectedA == .cpuAoS ? .cpuSoA : .cpuAoS
        try await first.setOverride(override, for: measured(512,1))
        try require(try await first.selectStrategy(for: measured(512,1)) == override, "process local override set")
        let second = KernelAutoTuner(persistenceURL: url)
        let profilesB = try await second.profileWorkload(b, iterations: 2, warmupIterations: 1)
        try require(profilesB.count == 2, "actual calibration B before selection")
        let third = KernelAutoTuner(persistenceURL: url)
        try require(try await third.selectStrategy(for: measured(512,1)) == selectedA, "A survived B publication without override")
        _ = try await third.selectStrategy(for: measured(768,2))
        try require(true, "B survived reconstruction")
        let original = try Data(contentsOf: url)
        try await third.clearCache()
        try await expectMiss(third, measured(512,1))
        _ = try await KernelAutoTuner(persistenceURL: url).selectStrategy(for: measured(512,1))
        try require(true, "normal clear does not delete persisted file")

        for change in ["schema", "algorithm", "hardware", "boundary", "dataset", "legacy"] {
            var object = try JSONSerialization.jsonObject(with: original) as! [String: Any]
            if change == "schema" { object["schemaVersion"] = -1 }
            else if change == "legacy" { object = ["profiles": object["profiles"]!] }
            else {
                var provenance = object["provenance"] as! [String: Any]
                if change == "algorithm" { provenance["algorithmVersion"] = -1 }
                if change == "hardware" { provenance["hardwareModel"] = "Q02-mismatched-hardware" }
                if change == "boundary" { provenance["operationBoundary"] = "other-operation" }
                if change == "dataset" { provenance["datasetVersion"] = "other-dataset" }
                object["provenance"] = provenance
            }
            try JSONSerialization.data(withJSONObject: object).write(to: url)
            try await expectMiss(KernelAutoTuner(persistenceURL: url), measured(512,1))
            print("Q02_INVALIDATION \(change)")
        }
        try Data("malformed".utf8).write(to: url)
        try await expectMiss(KernelAutoTuner(persistenceURL: url), measured(512,1))
        let mutations = ["throughput", "mean", "p50", "p95", "p99", "estimate", "negativeScore",
                         "largeScore", "samples", "warmups", "auto", "incompatible", "profileProvenance", "count", "dimension"]
        for mutation in mutations {
            var object = try JSONSerialization.jsonObject(with: original) as! [String: Any]
            var records = object["profiles"] as! [[String: Any]]
            for index in records.indices {
                var workload = records[index]["workload"] as! [String: Any]
                guard workload["dimension"] as? Int == 512 else { continue }
                switch mutation {
                case "throughput": records[index]["throughput"] = 0
                case "mean": records[index]["latencyMean"] = -1
                case "p50": records[index]["latencyP50"] = 0
                case "p95": records[index]["latencyP50"] = 2; records[index]["latencyP95"] = 1
                case "p99": records[index]["latencyP95"] = 2; records[index]["latencyP99"] = 1
                case "estimate": records[index]["estimatedPayloadBytes"] = 1
                case "negativeScore": records[index]["syntheticAccuracyScore"] = -0.1
                case "largeScore": records[index]["syntheticAccuracyScore"] = 1.1
                case "samples": records[index]["measuredSamples"] = 0
                case "warmups": records[index]["warmupIterations"] = -1
                case "auto": records[index]["strategy"] = "auto"
                case "incompatible": records[index]["strategy"] = "cpuQuantized"
                case "profileProvenance":
                    var provenance = records[index]["provenance"] as! [String: Any]
                    provenance["algorithmVersion"] = -1; records[index]["provenance"] = provenance
                case "count": workload["batchSize"] = 0; records[index]["workload"] = workload
                case "dimension": workload["dimension"] = 256; records[index]["workload"] = workload
                default: throw Failure(message: "unknown mutation")
                }
            }
            object["profiles"] = records
            try JSONSerialization.data(withJSONObject: object).write(to: url)
            let partial = KernelAutoTuner(persistenceURL: url)
            try await expectMiss(partial, measured(512,1))
            _ = try await partial.selectStrategy(for: measured(768,2))
            try require(true, "invalid individual \(mutation) discarded; valid B retained")
        }

        let failingURL = directory.appendingPathComponent("save-failure")
        let failureTuner = KernelAutoTuner(persistenceURL: failingURL)
        _ = try await failureTuner.selectStrategy(for: a) // Loads the missing file before directory obstruction.
        try await failureTuner.setOverride(.cpuSoA, for: a)
        try FileManager.default.createDirectory(at: failingURL, withIntermediateDirectories: false)
        do {
            _ = try await failureTuner.profileWorkload(a, iterations: 1, warmupIterations: 0)
            throw Failure(message: "save unexpectedly succeeded")
        } catch let error as VectorError {
            try require(error.kind == .operationFailed, "save failure surfaced")
            try require(error.context.additionalInfo["reason"]?.contains("in-memory calibration published") == true, "save failure identifies committed memory")
        }
        _ = try await failureTuner.selectStrategy(for: measured(512,1))
        try require(try await failureTuner.selectStrategy(for: a) == .cpuSoA, "save failure preserves override")
        do { try await failureTuner.clearCache(removePersistedProfiles: true); throw Failure(message: "directory deleted") }
        catch let error as VectorError { try require(error.kind == .operationFailed, "directory deletion refused") }
        _ = try await failureTuner.selectStrategy(for: measured(512,1))
        try require(try await failureTuner.selectStrategy(for: a) == .cpuSoA, "failed deletion preserves state")
        do {
            _ = try await KernelAutoTuner(persistenceURL: failingURL).selectStrategy(for: a)
            throw Failure(message: "directory read accepted")
        } catch let error as VectorError { try require(error.kind == .operationFailed, "read I/O failure surfaced") }
        try original.write(to: url)
        try await KernelAutoTuner(persistenceURL: url).clearCache(removePersistedProfiles: true)
        try require(!FileManager.default.fileExists(atPath: url.path), "configured file removed")
        try require(FileManager.default.fileExists(atPath: failingURL.path), "unrelated directory retained")
        let shared = MixedPrecisionAutoTuner.shared
        await shared.clearCache()
        let sharedProfiles = try await shared.calibrate(candidateCount: 2, iterations: 1, warmupIterations: 0)
        let reduced = MixedPrecisionStrategy.queryFP32Blocked
        try require(sharedProfiles.contains { $0.strategy == reduced && $0.maxRelativeError <= 1 },
                    "actual qualifying shared reduced profile")
        try await shared.setOverride(reduced, candidateCount: 2, maximumSyntheticRelativeError: 1)
        try require(try await shared.selectStrategy(candidateCount: 2, maximumSyntheticRelativeError: 1) == reduced,
                    "shared reduced override is active")
        let bridge = KernelAutoTuner()
        try require(try await bridge.selectMixedPrecisionStrategy(for:
            WorkloadCharacteristics(dimension: 512, batchSize: 2, precisionMode: .fp32Full),
            maximumSyntheticRelativeError: 1) == .fullFP32, "full mode dominates qualifying shared reduced override")
        try require(try await bridge.selectMixedPrecisionStrategy(for:
            WorkloadCharacteristics(dimension: 512, batchSize: 2, precisionMode: .fp16Acceptable),
            maximumSyntheticRelativeError: 1) == reduced, "reduced bridge reaches active override")
        await shared.clearCache()
        _ = try await ActorConsumer().select(KernelAutoTuner())
        try require(try await mainActorConsumer(MixedPrecisionAutoTuner()) == .fullFP32, "MainActor consumer")
        _ = try QuantizedAutoTuner().suggestQuantization(for: "consumer", dimension: 512, candidateCount: 5001, relativeErrorHint: 0.1)
        print("Q02_PUBLIC_RESULT PASS")
        print("Q02_PUBLIC_BODY_END")
    }
}
