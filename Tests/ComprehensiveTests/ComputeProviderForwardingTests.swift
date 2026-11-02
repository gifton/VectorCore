import Foundation
import Testing
@testable import VectorCore

/// A 0.3.3-style conformer: it implements only the hint-less scheduling requirements, the overloads the 0.3.3
/// documentation told providers to implement, and reports `maxConcurrency` 0 for "unbounded" as a GPU provider may.
private final class ForwardingLedger: @unchecked Sendable {
    private let lock = NSLock()
    private var reached: [String] = []
    func note(_ witness: String) { lock.withLock { reached.append(witness) } }
    var witnesses: [String] { lock.withLock { reached } }
}

private struct HintlessOnlyProvider: ComputeProvider {
    let ledger: ForwardingLedger
    var device: ComputeDevice { .cpu }
    var maxConcurrency: Int { 0 }
    var deviceInfo: ComputeDeviceInfo { .init(name: "hintless-only", maxThreads: 1, preferredChunkSize: 1) }
    func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T { try await work() }
    func parallelForEach(items: Range<Int>, _ body: @Sendable @escaping (Int) async throws -> Void) async throws {
        ledger.note("forEach")
        for index in items { try await body(index) }
    }
    func parallelReduce<R: Sendable>(
        items: Range<Int>, initial: R,
        _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
        _ combine: @Sendable @escaping (R, R) -> R
    ) async throws -> R {
        ledger.note("reduce")
        return combine(initial, try await rangeWork(items))
    }
}

/// The chunk-hinted `ComputeProvider` defaults validate `minChunk` and then call the hint-less requirements, so a
/// conformer that implements only those is reached by every route that passes a hint (the direction BASE 0.3.3 used;
/// PR #56 review CM1). A hint-less override is therefore never bypassed, and its `maxConcurrency` is not consulted.
@Suite struct ComputeProviderForwardingTests {
    @Test func hintedDefaultsReachHintlessWitnesses() async throws {
        let ledger = ForwardingLedger()
        let provider: any ComputeProvider = HintlessOnlyProvider(ledger: ledger)
        let sum = try await provider.parallelReduce(items: 0..<10, initial: 0, minChunk: 4,
                                                    { $0.reduce(0, +) }, { $0 + $1 })
        #expect(sum == 45)
        let seen = ForwardingLedger()
        try await provider.parallelForEach(items: 0..<5, minChunk: 2) { seen.note("\($0)") }
        #expect(seen.witnesses.sorted() == ["0", "1", "2", "3", "4"])
        #expect(ledger.witnesses == ["reduce", "forEach"], "hinted defaults bypassed the hint-less witnesses: \(ledger.witnesses)")
    }

    @Test func hintedDefaultsStillRejectInvalidHints() async throws {
        let ledger = ForwardingLedger()
        let provider: any ComputeProvider = HintlessOnlyProvider(ledger: ledger)
        for hint in [0, -1] {
            do {
                _ = try await provider.parallelReduce(items: 0..<3, initial: 0, minChunk: hint, { $0.count }, { $0 + $1 })
                Issue.record("minChunk \(hint) accepted")
            } catch let error as VectorError { #expect(error.kind == .invalidConfiguration) }
            do {
                try await provider.parallelForEach(items: 0..<3, minChunk: hint) { _ in }
                Issue.record("minChunk \(hint) accepted")
            } catch let error as VectorError { #expect(error.kind == .invalidConfiguration) }
        }
        #expect(ledger.witnesses.isEmpty, "a rejected hint reached a witness: \(ledger.witnesses)")
    }
}
