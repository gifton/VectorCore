import Foundation
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif
import Testing
@testable import VectorCore

@Suite struct SoAFP16CacheIdentityTests {
    @Test func everyCoordinateMatchesUncachedConversion() async {
        let cache = SoAFP16Cache512(maxSize: 600, maxBytes: 16_000_000)
        let base = Vector512Optimized(repeating: 1)
        _ = await cache.getOrCreate(for: [base])
        for coordinate in 0..<512 {
            var changed = base
            changed[coordinate] = 2
            let actual = await cache.getOrCreate(for: [changed])
            let expected = MixedPrecisionKernels.createSoA512FP16(from: [changed])
            #expect(actual.storage == expected.storage, "coordinate \(coordinate)")
        }
        let stats = await cache.statistics()
        #expect(stats.hits == 0)
        #expect(stats.misses == 513)
    }

    private func candidates(_ count: Int, value: Float = 1) -> [Vector512Optimized] {
        (0..<count).map { Vector512Optimized(repeating: value + Float($0)) }
    }

    private func check(_ actual: MixedPrecisionKernels.SoA512FP16, _ vectors: [Vector512Optimized]) {
        let expected = MixedPrecisionKernels.createSoA512FP16(from: vectors)
        #expect(actual.vectorCount == expected.vectorCount)
        #expect(actual.dimension == expected.dimension)
        #expect(actual.groupCount == expected.groupCount)
        #expect(actual.storage == expected.storage)
    }

    // Independent allocation-capacity accounting; does not invoke the cache's cost helper.
    private func cost(_ vectors: [Vector512Optimized]) -> Int {
        let identity = [UInt32](repeating: 0, count: vectors.count * 512)
        let soa = MixedPrecisionKernels.createSoA512FP16(from: vectors)
        return identity.capacity * MemoryLayout<UInt32>.stride
            + soa.storage.capacity * MemoryLayout<UInt16>.stride
    }

    @Test func orderedCountPaddingFreshCopiesAndMutation() async {
        let cache = SoAFP16Cache512(maxBytes: 1_000_000)
        for count in [0, 1, 3, 4, 5, 7] {
            let original = candidates(count)
            check(await cache.getOrCreate(for: original), original)
            check(await cache.getOrCreate(for: candidates(count)), original)
        }
        let initial = await cache.statistics()
        #expect(initial.hits == 6)
        #expect(initial.misses == 6)
        var vectors = candidates(3)
        let baseline = vectors
        let retained = await cache.getOrCreate(for: baseline)
        vectors[1][250] = 9
        check(await cache.getOrCreate(for: vectors), vectors)
        check(retained, baseline)
        check(await cache.getOrCreate(for: baseline), baseline)
        let reversed = Array(baseline.reversed())
        check(await cache.getOrCreate(for: reversed), reversed)
        let final = await cache.statistics()
        #expect(final.misses == 8)
    }

    @Test func collisionBucketsStillRequireAllBits() async {
        let cache = SoAFP16Cache512(maxBytes: 1_000_000, fingerprint: { _ in 7 })
        let a = candidates(3)
        var b = a
        b[1][234] = 12
        for input in [a, b, a, b] { check(await cache.getOrCreate(for: input), input) }
        #expect(await cache.get(for: a) != nil)
        #expect(await cache.get(for: b) != nil)
        let stats = await cache.statistics()
        #expect(stats.size == 2)
        #expect(stats.hits == 4)
        #expect(stats.misses == 2)
    }

    @Test func signedZeroAndNaNPayloadIdentity() async {
        let cache = SoAFP16Cache512(maxBytes: 1_000_000, fingerprint: { _ in 0 })
        for bits: UInt32 in [0, 0x80000000, 0x7fc00001, 0x7fc00002] {
            var input = candidates(1)
            input[0][255] = Float(bitPattern: bits)
            check(await cache.getOrCreate(for: input), input)
            check(await cache.getOrCreate(for: input), input)
        }
        let stats = await cache.statistics()
        #expect(stats.size == 4)
        #expect(stats.hits == 4)
        #expect(stats.misses == 4)
    }

    @Test func zeroLimitsDisableRetentionIncludingEmptyInputs() async {
        for limits in [(0, 1_000_000), (100, 0)] {
            let cache = SoAFP16Cache512(maxSize: limits.0, maxBytes: limits.1)
            for count in [0, 1, 0, 1] {
                let input = candidates(count)
                check(await cache.getOrCreate(for: input), input)
            }
            let stats = await cache.statistics()
            #expect(stats.size == 0)
            #expect(stats.hits == 0)
            #expect(stats.misses == 4)
            #expect(await cache.retentionStatistics().usedBytes == 0)
        }
    }

    @Test func countLimitEvictionClearAndRetainedResult() async {
        let cache = SoAFP16Cache512(maxSize: 1, maxBytes: 1_000_000)
        let a = candidates(1)
        let retained = await cache.getOrCreate(for: a)
        for input in [candidates(0), candidates(1, value: 8), a] {
            check(await cache.getOrCreate(for: input), input)
            check(await cache.getOrCreate(for: input), input)
            #expect(await cache.statistics().size == 1)
        }
        check(retained, a)
        await cache.clear()
        let stats = await cache.statistics()
        #expect(stats.size == 0 && stats.hits == 0 && stats.misses == 0)
        #expect(await cache.retentionStatistics().usedBytes == 0)
        #expect(await cache.hitRate() == 0)
        check(retained, a)
    }

    @Test func byteExactFitOneShortAndOversizedBypass() async {
        for count in [0, 1, 3, 4, 5, 7] {
            let input = candidates(count)
            let bytes = cost(input)
            let exact = SoAFP16Cache512(maxBytes: max(1, bytes))
            check(await exact.getOrCreate(for: input), input)
            #expect(await exact.retentionStatistics().usedBytes == bytes)
            #expect(await exact.statistics().size == 1)
            if bytes > 0 {
                let short = SoAFP16Cache512(maxBytes: bytes - 1)
                check(await short.getOrCreate(for: input), input)
                #expect(await short.statistics().size == 0)
                #expect(await short.retentionStatistics().usedBytes == 0)
            }
        }
        let small = candidates(1)
        let budget = cost(small)
        let cache = SoAFP16Cache512(maxBytes: budget)
        _ = await cache.getOrCreate(for: small)
        check(await cache.getOrCreate(for: candidates(7)), candidates(7))
        #expect(await cache.get(for: small) != nil)
        let retention = await cache.retentionStatistics()
        #expect(retention.usedBytes == budget && retention.maxBytes == budget)
    }

    @Test func byteEvictionUsesRecencyAndCanRemoveMultipleEntries() async {
        let a = candidates(1, value: 1)
        let b = candidates(1, value: 2)
        let c = candidates(1, value: 3)
        let d = candidates(1, value: 4)
        let bytes = cost(a)
        let cache = SoAFP16Cache512(maxBytes: 3 * bytes)
        for input in [a, b, c] { _ = await cache.getOrCreate(for: input) }
        for _ in 0..<5 { _ = await cache.getOrCreate(for: a) }
        #expect(await cache.retentionStatistics().usedBytes == 3 * bytes)
        _ = await cache.getOrCreate(for: d)
        #expect(await cache.get(for: b) == nil)
        #expect(await cache.get(for: c) != nil)
        #expect(await cache.get(for: a) != nil)
        let larger = candidates(5)
        let largerBytes = cost(larger)
        #expect(largerBytes > 2 * bytes && largerBytes <= 3 * bytes)
        check(await cache.getOrCreate(for: larger), larger)
        #expect(await cache.statistics().size == 1)
        #expect(await cache.retentionStatistics().usedBytes == largerBytes)
        await cache.clear()
        #expect(await cache.retentionStatistics().usedBytes == 0)
    }

    @Test func checkedCapacityCostRejectsOverflow() {
        #expect(SoAFP16Cache512.retainedCost(identityCapacity: 2, soaCapacity: 3) == 14)
        #expect(SoAFP16Cache512.retainedCost(identityCapacity: 0, soaCapacity: 0) == 0)
        #expect(SoAFP16Cache512.retainedCost(identityCapacity: Int.max, soaCapacity: 0) == nil)
        #expect(SoAFP16Cache512.retainedCost(identityCapacity: 0, soaCapacity: Int.max) == nil)
        #expect(SoAFP16Cache512.retainedCost(identityCapacity: Int.max / 4, soaCapacity: 2) == nil)
        #expect(SoAFP16Cache512.retainedCost(identityCapacity: -1, soaCapacity: 0) == nil)
    }

    @Test func negativeLimitChild() {
        guard let limit = ProcessInfo.processInfo.environment["I02_NEGATIVE_LIMIT"] else { return }
        _ = SoAFP16Cache512(maxSize: limit == "maxSize" ? -1 : 1,
                           maxBytes: limit == "maxBytes" ? -1 : 1)
        print("I02_INVALID_LIMIT_ACCEPTED")
    }

    @Test func negativeLimitsRejectAtInitialization() throws {
        for limit in ["maxSize", "maxBytes"] {
            var environment = ProcessInfo.processInfo.environment
            environment["I02_NEGATIVE_LIMIT"] = limit
            environment["SWIFT_BACKTRACE"] = "enable=no"
            // File capture cannot block on a child keeping a pipe open. The launcher's semaphore
            // wait supplies a monotonic deadline independently of output or child progress.
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let outputURL = directory.appendingPathComponent("child.log")
            _ = FileManager.default.createFile(atPath: outputURL.path, contents: nil)
            let output = try FileHandle(forWritingTo: outputURL)
            defer { try? output.close() }
            // The launcher signals only the child's own process group; no process-wide name
            // matching is used.
            let process = try launchChildProbe(.swiftTesting(filter: "SoAFP16CacheIdentityTests/negativeLimitChild"),
                                               environment: environment, output: output, deadline: 15, cleanup: .kill(seconds: 2))
            let timedOut = process.timedOut
            let diagnostic = String(decoding: try Data(contentsOf: outputURL), as: UTF8.self)
            if timedOut {
                Issue.record("\(limit) child exceeded 15-second deadline. Output: \(diagnostic)")
                continue
            }
            #expect(process.terminationReason == .uncaughtSignal, "\(limit): \(diagnostic)")
            #expect(diagnostic.contains("SoAFP16Cache512 \(limit) must be nonnegative"), "\(diagnostic)")
            #expect(!diagnostic.contains("I02_INVALID_LIMIT_ACCEPTED"))
        }
    }
}
