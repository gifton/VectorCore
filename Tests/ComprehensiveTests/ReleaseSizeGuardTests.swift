import Foundation
import Testing
@testable import VectorCore

/// Release-active output-size guards (PR #56 review AP2, CV1; the class is closed for new sites by
/// Scripts/ci/tests/test_release_active_capacity_guards.py). Each scenario runs in a child process that prints a
/// marker and then calls one kernel with an output buffer shorter than the candidate count, or `MatrixDistance` with
/// rows of unequal length. The parent requires the child to die by signal, to print the family's named diagnostic, and
/// never to reach the line after the call. For the kernel families the output buffer is the first 4 slots of a 16-Float
/// shared mapping filled with a sentinel, so the parent also requires every slot (including the 4 "given" ones) to be
/// byte-identical afterwards: the guard runs before any write. The suite runs on the Debug and the Release CI rows;
/// the Release row is the one that exercised the former `#if DEBUG`-only asserts.
@Suite("Release-active size guards") struct ReleaseSizeGuardTests {
    static let sentinel: Float = -12345.0
    static let candidateCount = 8
    static let shortCount = 4

    static let kernelScenarios: [String] = [
        "mixed.euclidean.512.fp16q", "mixed.euclidean.512.fp32q",
        "mixed.euclidean.768.fp16q", "mixed.euclidean.768.fp32q",
        "mixed.euclidean.1536.fp16q", "mixed.euclidean.1536.fp32q",
        "mixed.dot.512", "mixed.dot.768", "mixed.dot.1536",
        "mixed.dotFP16.512", "mixed.dotFP16.768", "mixed.dotFP16.1536",
        "mixed.dotMixed.512", "mixed.dotMixed.768", "mixed.dotMixed.1536",
        "mixed.euclideanBlocked.512.fp16q", "mixed.euclideanBlocked.512.fp32q",
        "mixed.euclideanBlocked.768.fp16q", "mixed.euclideanBlocked.768.fp32q",
        "mixed.euclideanBlocked.1536.fp16q", "mixed.euclideanBlocked.1536.fp32q",
        "soa.euclid2_blocked", "soa.euclid2_blocked_4way", "soa.euclid2_tiled",
        "soa.dot_blocked_2way", "soa.dot_blocked_4way", "soa.dot_tiled",
        "soa.cosine_fused_blocked_2way", "soa.cosine_fused_blocked_4way", "soa.cosine_fused_tiled"
    ]
    static let matrixScenarios: [String] = [
        "matrix.euclidean.shortCandidateRow", "matrix.euclidean.shortQueryRow", "matrix.cosine.shortCandidateRow"
    ]

    static func diagnostic(for scenario: String) -> String {
        if scenario.hasPrefix("mixed.dotFP16.") || scenario.hasPrefix("mixed.dotMixed.") {
            return "MixedPrecision batch output capacity must cover the candidate count"
        }
        if scenario.hasPrefix("mixed.") { return "MixedPrecision batch results capacity must cover vectorCount" }
        if scenario.hasPrefix("soa.") { return "SoA batch output capacity must cover the candidate count" }
        return "MatrixDistance rows must share a dimension"
    }

    @Test(arguments: kernelScenarios)
    func kernelTrapsBeforeWriting(scenario: String) throws { try runChild(scenario, checksSentinel: true) }

    @Test(arguments: matrixScenarios)
    func matrixRowsOfUnequalLengthTrap(scenario: String) throws { try runChild(scenario, checksSentinel: false) }

    private func runChild(_ scenario: String, checksSentinel: Bool) throws {
        let stem = FileManager.default.temporaryDirectory
            .appendingPathComponent("SizeGuard-\(scenario)-\(UUID().uuidString)").path
        let shared = stem + ".buffer"
        let initial = Array(repeating: Self.sentinel, count: 16).withUnsafeBytes { Data($0) }
        try initial.write(to: URL(fileURLWithPath: shared))
        defer { try? FileManager.default.removeItem(atPath: shared) }
        var env = ProcessInfo.processInfo.environment
        env["SIZE_GUARD_CHILD"] = scenario
        env["SIZE_GUARD_SHARED"] = shared
        env["SWIFT_BACKTRACE"] = "enable=no"
        _ = FileManager.default.createFile(atPath: stem + ".log", contents: nil)
        let output = try FileHandle(forWritingTo: URL(fileURLWithPath: stem + ".log"))
        let process = try launchChildProbe(.swiftTesting(filter: "ReleaseSizeGuardTests/childProbe"),
                                           environment: env, output: output, deadline: 30, cleanup: .kill(seconds: 2))
        try output.close()
        let log = try String(contentsOfFile: stem + ".log", encoding: .utf8)
        try #require(!process.timedOut, "\(scenario): child deadline\n\(log.suffix(1500))")
        let outcome = "\(scenario): status \(process.statusDescription) reason \(process.reasonDescription)\n\(log.suffix(1500))"
        #expect(process.terminationReason == .uncaughtSignal, "\(outcome)")
        #expect(log.contains("SIZE_GUARD_BEGIN \(scenario)"), "\(scenario): the child never reached the call\n\(log.suffix(1500))")
        #expect(!log.contains("SIZE_GUARD_RETURNED"), "\(scenario): the call returned")
        #expect(log.contains(Self.diagnostic(for: scenario)), "\(scenario): diagnostic missing\n\(log.suffix(1500))")
        if checksSentinel {
            #expect(try Data(contentsOf: URL(fileURLWithPath: shared)) == initial, "\(scenario): output written before the guard")
        }
        try? FileManager.default.removeItem(atPath: stem + ".log")
    }

    // MARK: Child

    @Test func childProbe() throws {
        guard let scenario = ProcessInfo.processInfo.environment["SIZE_GUARD_CHILD"] else { return }
        let fd = open(ProcessInfo.processInfo.environment["SIZE_GUARD_SHARED"]!, O_RDWR)
        guard fd >= 0 else { throw NSError(domain: "SizeGuard open", code: 1) }
        defer { close(fd) }
        let mapping = mmap(nil, 64, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        guard mapping != MAP_FAILED, let mapping else { throw NSError(domain: "SizeGuard mmap", code: 1) }
        defer { munmap(mapping, 64) }
        let short = UnsafeMutableBufferPointer(start: mapping.assumingMemoryBound(to: Float.self), count: Self.shortCount)
        print("SIZE_GUARD_BEGIN \(scenario)"); fflush(stdout)
        try Self.run(scenario, into: short)
        print("SIZE_GUARD_RETURNED \(scenario)"); fflush(stdout)
    }

    private static func values(_ width: Int, _ k: Int) -> [Float] { (0..<width).map { Float(($0 + k) % 7) * 0.25 } }

    private static func unknown(_ scenario: String) -> Error { NSError(domain: "SizeGuard unknown scenario \(scenario)", code: 1) }

    /// `mixed.<kind>.<dimension>[.<query>]` dispatches on the dimension; `soa.<kernel>`; `matrix.<metric>.<shape>`.
    private static func run(_ scenario: String, into results: UnsafeMutableBufferPointer<Float>) throws {
        let parts = scenario.split(separator: ".").map(String.init)
        guard let family = parts.first else { throw unknown(scenario) }
        let rest = Array(parts.dropFirst())
        let kind = rest.filter { Int($0) == nil }.joined(separator: ".")
        switch family {
        case "mixed":
            switch rest.first(where: { Int($0) != nil }) {
            case "512": try runMixed512(kind, into: results)
            case "768": try runMixed768(kind, into: results)
            case "1536": try runMixed1536(kind, into: results)
            default: throw unknown(scenario)
            }
        case "soa": try runSoA(kind, into: results)
        case "matrix": try runMatrix(kind)
        default: throw unknown(scenario)
        }
    }

    private static func runMixed512(_ kind: String, into results: UnsafeMutableBufferPointer<Float>) throws {
        typealias MP = MixedPrecisionKernels
        let rows = try (0..<candidateCount).map { try Vector512Optimized(values(512, $0)) }
        let soa = try MP.SoA512FP16(vectors: rows)
        let fp16 = rows.map { MP.Vector512FP16(from: $0) }
        switch kind {
        case "euclidean.fp16q": MP.batchEuclidean512(query: fp16[0], candidates: soa, results: results)
        case "euclidean.fp32q": MP.batchEuclidean512(query: rows[0], candidates: soa, results: results)
        case "dot": MP.batchDotProductSoA(query: rows[0], candidates: soa, results: results)
        case "euclideanBlocked.fp16q": MP.batchEuclideanBlocked512(query: fp16[0], candidates: soa, results: results)
        case "euclideanBlocked.fp32q": MP.batchEuclideanBlocked512(query: rows[0], candidates: soa, results: results)
        case "dotFP16": MP.batchDotFP16_512(query: rows[0], candidates: fp16, out: results)
        case "dotMixed": MP.batchDotMixed512(query: rows[0], candidates: fp16, out: results)
        default: throw unknown("mixed.\(kind).512")
        }
    }

    private static func runMixed768(_ kind: String, into results: UnsafeMutableBufferPointer<Float>) throws {
        typealias MP = MixedPrecisionKernels
        let rows = try (0..<candidateCount).map { try Vector768Optimized(values(768, $0)) }
        let soa = try MP.SoA768FP16(vectors: rows)
        let fp16 = rows.map { MP.Vector768FP16(from: $0) }
        switch kind {
        case "euclidean.fp16q": MP.batchEuclidean768(query: fp16[0], candidates: soa, results: results)
        case "euclidean.fp32q": MP.batchEuclidean768(query: rows[0], candidates: soa, results: results)
        case "dot": MP.batchDotProductSoA(query: rows[0], candidates: soa, results: results)
        case "euclideanBlocked.fp16q": MP.batchEuclideanBlocked768(query: fp16[0], candidates: soa, results: results)
        case "euclideanBlocked.fp32q": MP.batchEuclideanBlocked768(query: rows[0], candidates: soa, results: results)
        case "dotFP16": MP.batchDotFP16_768(query: rows[0], candidates: fp16, out: results)
        case "dotMixed": MP.batchDotMixed768(query: rows[0], candidates: fp16, out: results)
        default: throw unknown("mixed.\(kind).768")
        }
    }

    private static func runMixed1536(_ kind: String, into results: UnsafeMutableBufferPointer<Float>) throws {
        typealias MP = MixedPrecisionKernels
        let rows = try (0..<candidateCount).map { try Vector1536Optimized(values(1536, $0)) }
        let soa = try MP.SoA1536FP16(vectors: rows)
        let fp16 = rows.map { MP.Vector1536FP16(from: $0) }
        switch kind {
        case "euclidean.fp16q": MP.batchEuclidean1536(query: fp16[0], candidates: soa, results: results)
        case "euclidean.fp32q": MP.batchEuclidean1536(query: rows[0], candidates: soa, results: results)
        case "dot": MP.batchDotProductSoA(query: rows[0], candidates: soa, results: results)
        case "euclideanBlocked.fp16q": MP.batchEuclideanBlocked1536(query: fp16[0], candidates: soa, results: results)
        case "euclideanBlocked.fp32q": MP.batchEuclideanBlocked1536(query: rows[0], candidates: soa, results: results)
        case "dotFP16": MP.batchDotFP16_1536(query: rows[0], candidates: fp16, out: results)
        case "dotMixed": MP.batchDotMixed1536(query: rows[0], candidates: fp16, out: results)
        default: throw unknown("mixed.\(kind).1536")
        }
    }

    private static func runSoA(_ kernel: String, into out: UnsafeMutableBufferPointer<Float>) throws {
        let rows = try (0..<candidateCount).map { try Vector512Optimized(values(512, $0)) }
        let soa = SoA<Vector512Optimized>(vectors: rows)
        let q = rows[0]
        switch kernel {
        case "euclid2_blocked": BatchKernels_SoA.euclid2_blocked(query: q, soa: soa, out: out)
        case "euclid2_blocked_4way": BatchKernels_SoA.euclid2_blocked_4way(query: q, soa: soa, out: out)
        case "euclid2_tiled": BatchKernels_SoA.euclid2_tiled(query: q, soa: soa, out: out)
        case "dot_blocked_2way": BatchKernels_SoA.dot_blocked_2way(query: q, soa: soa, out: out)
        case "dot_blocked_4way": BatchKernels_SoA.dot_blocked_4way(query: q, soa: soa, out: out)
        case "dot_tiled": BatchKernels_SoA.dot_tiled(query: q, soa: soa, out: out)
        case "cosine_fused_blocked_2way": BatchKernels_SoA.cosine_fused_blocked_2way(query: q, queryNorm: 1, soa: soa, out: out)
        case "cosine_fused_blocked_4way": BatchKernels_SoA.cosine_fused_blocked_4way(query: q, queryNorm: 1, soa: soa, out: out)
        case "cosine_fused_tiled": BatchKernels_SoA.cosine_fused_tiled(query: q, queryNorm: 1, soa: soa, out: out)
        default: throw unknown("soa.\(kernel)")
        }
    }

    private static func runMatrix(_ shape: String) throws {
        let full = DynamicVector([1, 2, 3]), short = DynamicVector([1, 2])
        switch shape {
        case "euclidean.shortCandidateRow": _ = MatrixDistance.euclideanSquaredMatrix(queries: [full], candidates: [full, short])
        case "euclidean.shortQueryRow": _ = MatrixDistance.euclideanSquaredMatrix(queries: [full, short], candidates: [full])
        case "cosine.shortCandidateRow": _ = MatrixDistance.cosineDistanceMatrix(queries: [full], candidates: [full, short])
        default: throw unknown("matrix.\(shape)")
        }
    }
}
