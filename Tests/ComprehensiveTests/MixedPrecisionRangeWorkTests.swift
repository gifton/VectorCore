import Foundation
import Darwin
import CryptoKit
import Testing
@testable import VectorCore

private func q04Marker(_ text: String) { FileHandle.standardOutput.write(Data((text + "\n").utf8)) }
private func q04Invoke(width: Int, metric: Int, query: [Float], candidates: [[UInt16]], range: Range<Int>, out: UnsafeMutableBufferPointer<Float>, core: Bool = false) throws {
    if width == 512 {
        let q = try Vector512Optimized(query)
        let c = candidates.map { MixedPrecisionKernels.Vector512FP16(fp16Values: $0) }
        if core {
            let storage = c.map { $0.storage }
            switch metric {
            case 0: MixedPrecisionKernels.range_euclid2_mixed_core(queryStorage: q.storage, candidatesArray: storage, range: range, out: out, lanes: Vector512Optimized.lanes)
            case 1: MixedPrecisionKernels.range_cosine_mixed_core(queryStorage: q.storage, candidatesArray: storage, range: range, out: out, lanes: Vector512Optimized.lanes)
            case 2: MixedPrecisionKernels.range_dot_mixed_core(queryStorage: q.storage, candidatesArray: storage, range: range, out: out, lanes: Vector512Optimized.lanes)
            default: fatalError("metric")
            }
        } else {
            switch metric {
            case 0: MixedPrecisionKernels.range_euclid2_mixed_512(query: q, candidatesFP16: c, range: range, out: out)
            case 1: MixedPrecisionKernels.range_cosine_mixed_512(query: q, candidatesFP16: c, range: range, out: out)
            case 2: MixedPrecisionKernels.range_dot_mixed_512(query: q, candidatesFP16: c, range: range, out: out)
            default: fatalError("metric")
            }
        }
        return
    }
    if width == 768 {
        let q = try Vector768Optimized(query)
        let c = candidates.map { MixedPrecisionKernels.Vector768FP16(fp16Values: $0) }
        if core {
            let storage = c.map { $0.storage }
            switch metric {
            case 0: MixedPrecisionKernels.range_euclid2_mixed_core(queryStorage: q.storage, candidatesArray: storage, range: range, out: out, lanes: Vector768Optimized.lanes)
            case 1: MixedPrecisionKernels.range_cosine_mixed_core(queryStorage: q.storage, candidatesArray: storage, range: range, out: out, lanes: Vector768Optimized.lanes)
            case 2: MixedPrecisionKernels.range_dot_mixed_core(queryStorage: q.storage, candidatesArray: storage, range: range, out: out, lanes: Vector768Optimized.lanes)
            default: fatalError("metric")
            }
        } else {
            switch metric {
            case 0: MixedPrecisionKernels.range_euclid2_mixed_768(query: q, candidatesFP16: c, range: range, out: out)
            case 1: MixedPrecisionKernels.range_cosine_mixed_768(query: q, candidatesFP16: c, range: range, out: out)
            case 2: MixedPrecisionKernels.range_dot_mixed_768(query: q, candidatesFP16: c, range: range, out: out)
            default: fatalError("metric")
            }
        }
        return
    }
    if width == 1536 {
        let q = try Vector1536Optimized(query)
        let c = candidates.map { MixedPrecisionKernels.Vector1536FP16(fp16Values: $0) }
        if core {
            let storage = c.map { $0.storage }
            switch metric {
            case 0: MixedPrecisionKernels.range_euclid2_mixed_core(queryStorage: q.storage, candidatesArray: storage, range: range, out: out, lanes: Vector1536Optimized.lanes)
            case 1: MixedPrecisionKernels.range_cosine_mixed_core(queryStorage: q.storage, candidatesArray: storage, range: range, out: out, lanes: Vector1536Optimized.lanes)
            case 2: MixedPrecisionKernels.range_dot_mixed_core(queryStorage: q.storage, candidatesArray: storage, range: range, out: out, lanes: Vector1536Optimized.lanes)
            default: fatalError("metric")
            }
        } else {
            switch metric {
            case 0: MixedPrecisionKernels.range_euclid2_mixed_1536(query: q, candidatesFP16: c, range: range, out: out)
            case 1: MixedPrecisionKernels.range_cosine_mixed_1536(query: q, candidatesFP16: c, range: range, out: out)
            case 2: MixedPrecisionKernels.range_dot_mixed_1536(query: q, candidatesFP16: c, range: range, out: out)
            default: fatalError("metric")
            }
        }
        return
    }
    fatalError("width")
}

/// Float16-free IEEE binary16 encoding of numerator/32 for integer |numerator| <= 2047 (exact when
/// the numerator needs at most 11 significant bits; checked by precondition).
/// Used so the finite range matrix compiles on Intel macOS, where Float16 does not exist.
private func q04HalfBits(numeratorOver32 n: Int) -> UInt16 {
    if n == 0 { return 0 }
    let magnitude = UInt32(abs(n))
    let top = 31 - magnitude.leadingZeroBitCount            // floor(log2 magnitude)
    precondition(top <= 10 || magnitude & ((1 << (top - 10)) - 1) == 0, "not exactly representable")
    let exponent = top - 5 + 15                               // magnitude/32 = 2^(top-5) * (1 + f)
    precondition(exponent >= 1 && exponent <= 30, "outside normal binary16 range")
    let mantissa = top <= 10 ? (magnitude - (1 << top)) << (10 - top) : (magnitude - (1 << top)) >> (top - 10)
    return (n < 0 ? 0x8000 : 0) | UInt16(exponent << 10) | UInt16(mantissa)
}

/// Float16-free exact decode of a zero or normal binary16 pattern (sign, 5-bit exponent 1...30, mantissa).
private func q04HalfValue(bits: UInt16) -> Double {
    if bits & 0x7fff == 0 { return 0 }
    let exponent = Int((bits >> 10) & 0x1f), mantissa = Double(bits & 0x3ff)
    precondition(exponent >= 1 && exponent <= 30, "helper decodes normal values only")
    let magnitude = (1 + mantissa / 1024) * pow(2.0, Double(exponent - 15))
    return bits & 0x8000 != 0 ? -magnitude : magnitude
}

@Suite(.serialized) struct MixedPrecisionRangeWorkTests {
    private let sentinel: UInt32 = 0x4b123456

    @Test func finiteRangeMatrix() throws {
        for width in [512, 768, 1536] {
            let query = (0..<width).map { Float(($0 % 13) - 4) / 16 }
            for n in [0, 1, 2, 3, 11] {
                var candidates: [[UInt16]] = []
                for index in 0..<n {
                    var bits: [UInt16] = []
                    for coordinate in 0..<width {
                        // value = (4k + index) / 32 with k = ((coordinate + 3 index) % 17) - 6: exact in binary16.
                        let k = ((coordinate + index * 3) % 17) - 6
                        bits.append(q04HalfBits(numeratorOver32: 4 * k + index))
                    }
                    candidates.append(bits)
                }
                var ranges = [0..<0, (n/2)..<(n/2), n..<n, 0..<n]
                if n > 0 { ranges.append((n-1)..<n) }
                if n >= 2 { ranges.append((n-2)..<n) }
                if n >= 3 { ranges.append((n-3)..<n) }
                if n > 3 { ranges += [2..<5, 1..<2, 3..<8] }
                for metric in 0..<3 {
                    for range in ranges {
                        var parent = Array(repeating: Float(bitPattern: sentinel), count: range.count + 7)
                        try parent.withUnsafeMutableBufferPointer { buffer in
                            let out = UnsafeMutableBufferPointer(rebasing: buffer[2..<(buffer.count-2)])
                            try q04Invoke(width: width, metric: metric, query: query, candidates: candidates, range: range, out: out)
                        }
                        #expect(parent.prefix(2).allSatisfy { $0.bitPattern == sentinel })
                        #expect(parent.suffix(from: 2 + range.count).allSatisfy { $0.bitPattern == sentinel })
                        for j in 0..<range.count {
                            let candidate = candidates[range.lowerBound+j].map { q04HalfValue(bits: $0) }
                            var dot = 0.0, square = 0.0, qnorm = 0.0, cnorm = 0.0, absProducts = 0.0
                            for i in 0..<width {
                                let q = Double(query[i]), c = candidate[i]
                                dot += q*c; square += (q-c)*(q-c); qnorm += q*q; cnorm += c*c; absProducts += abs(q*c)
                            }
                            let expected = metric == 0 ? square : (metric == 1 ? 1-dot/sqrt(qnorm*cnorm) : dot)
                            // At most width Float additions per stream: 16*n*epsilon is
                            // a conservative forward-error allowance, scaled by absolute terms.
                            let scale = metric == 0 ? square : (metric == 1 ? 4 : absProducts)
                            let tolerance = 16 * Double(width) * Double(Float.ulpOfOne) * max(1, scale)
                            let actual = parent[2+j]
                            #expect(actual.isFinite && expected.isFinite)
                            #expect(abs(Double(actual)-expected) <= tolerance)
                        }
                        if range.isEmpty {
                            try q04Invoke(width: width, metric: metric, query: query, candidates: candidates, range: range,
                                          out: UnsafeMutableBufferPointer(start: nil, count: 0))
                        }
                    }
                }
            }
        }
    }

    @Test func exceptionalSameCoreParity() throws {
        for width in [512, 768, 1536] {
            for special: Float in [0, -0.0, 1e-12, .infinity, -.infinity, .nan, .greatestFiniteMagnitude] {
                var query = Array(repeating: special.isFinite && abs(special) < 1e-9 ? special : Float(1), count: width)
                query[7] = special
                // Exact binary16 bit patterns (Float16 is unavailable on Intel macOS): (index+1)/4 for index 0...6,
                // +inf, a quiet NaN and the least subnormal.
                let quarterBits: [UInt16] = [0x3400, 0x3800, 0x3A00, 0x3C00, 0x3D00, 0x3E00, 0x3F00]
                var candidates = quarterBits.map { Array(repeating: $0, count: width) }
                candidates[2][3] = 0x7C00
                candidates[3][5] = 0x7E00
                candidates[4] = Array(repeating: 0, count: width)
                candidates[5] = Array(repeating: 0x0001, count: width)
                for metric in 0..<3 {
                    for range in [1..<2, 1..<3, 1..<4, 1..<6, 0..<7] {
                        var actual = Array(repeating: Float(0), count: range.count)
                        var expected = actual
                        try actual.withUnsafeMutableBufferPointer { try q04Invoke(width: width, metric: metric, query: query, candidates: candidates, range: range, out: $0) }
                        try expected.withUnsafeMutableBufferPointer { try q04Invoke(width: width, metric: metric, query: query, candidates: candidates, range: range, out: $0, core: true) }
                        for i in actual.indices {
                            #expect(actual[i].bitPattern == expected[i].bitPattern)
                        }
                    }
                }
            }
        }
    }

    @Test(arguments: [512, 768, 1536], [0, 1, 2])
    func namedRejectionMatrix(width: Int, metric: Int) throws {
        for invalid in ["negative", "upper", "empty", "extreme", "capacity"] { try runChild(width: width, metric: metric, invalid: invalid) }
    }

    private func runChild(width: Int, metric: Int, invalid: String) throws {
        let stem = "/private/tmp/Q04-child-\(width)-\(metric)-\(invalid)-\(UUID().uuidString)"
        let shared = stem + ".buffer"
        let initial = Array(repeating: sentinel, count: 16).withUnsafeBytes { Data($0) }
        try initial.write(to: URL(fileURLWithPath: shared))
        var env = ProcessInfo.processInfo.environment
        env["Q04_CHILD"] = "\(width),\(metric),\(invalid)"; env["Q04_SHARED"] = shared
        _ = FileManager.default.createFile(atPath: stem + ".log", contents: nil)
        let output = try FileHandle(forWritingTo: URL(fileURLWithPath: stem + ".log"))
        let process = try launchChildProbe(.swiftTesting(filter: "MixedPrecisionRangeWorkTests/childProbe"),
                                           environment: env, output: output, deadline: 15, cleanup: .kill(seconds: 2))
        let args = process.arguments, timedOut = process.timedOut, cleaned = process.cleaned
        try output.close()
        func hash(_ path: String) -> String {
            guard let bytes = try? Data(contentsOf: URL(fileURLWithPath: path)) else { return "unreadable" }
            return SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        }
        let log = try String(contentsOfFile: stem + ".log", encoding: .utf8)
        let bundle = args.firstIndex(of: "--test-bundle-path").map { args[$0+1] } ?? CommandLine.arguments[0]
        var metadata: [String: Any] = ["host": CommandLine.arguments[0], "hostSHA256": hash(CommandLine.arguments[0]),
            "loadedBundle": bundle, "loadedBundleSHA256": hash(bundle), "arguments": args, "scenario": env["Q04_CHILD"]!,
            "pid": process.processIdentifier, "timeout": timedOut, "cleaned": cleaned, "executionDeadlineSeconds": 15,
            "cleanupDeadlineSeconds": 2, "log": stem + ".log", "logSHA256": hash(stem + ".log"), "sharedBuffer": shared]
        if timedOut {
            try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted,.sortedKeys]).write(to: URL(fileURLWithPath: stem + ".json"))
            throw NSError(domain: "Q04 child deadline", code: 1)
        }
        metadata["exit"] = process.terminationStatus; metadata["terminationReason"] = process.terminationReason?.rawValue
        try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted,.sortedKeys]).write(to: URL(fileURLWithPath: stem + ".json"))
        print("Q04 capture=\(stem).log")
        #expect(process.terminationStatus != nil && process.terminationStatus != 0)
        #expect(log.contains("Q04_BODY_BEGIN"))
        #expect(log.contains(invalid == "capacity" ? "MixedPrecision range output capacity violation" : "MixedPrecision candidate range violation"), "\(log)")
        #expect(try Data(contentsOf: URL(fileURLWithPath: shared)) == initial, "Output changed before rejection")
    }

    @Test func childProbe() throws {
        guard let scenario = ProcessInfo.processInfo.environment["Q04_CHILD"] else { return }
        let parts = scenario.split(separator: ",").map(String.init)
        let width = Int(parts[0])!, metric = Int(parts[1])!, invalid = parts[2]
        let fd = open(ProcessInfo.processInfo.environment["Q04_SHARED"]!, O_RDWR)
        guard fd >= 0 else { throw NSError(domain: "Q04 open", code: 1) }
        defer { close(fd) }
        let mapping = mmap(nil, 64, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        guard mapping != MAP_FAILED, let mapping else { throw NSError(domain: "Q04 mmap", code: 1) }
        defer { munmap(mapping, 64) }
        let pointer = mapping.assumingMemoryBound(to: Float.self)
        let range: Range<Int>
        switch invalid {
        case "negative": range = -1..<1
        case "upper": range = 0..<4
        case "empty": range = 4..<4
        case "extreme": range = Int.min..<Int.max
        default: range = 0..<3
        }
        let query = Array(repeating: Float(1), count: width)
        // 0x4000 is the IEEE binary16 bit pattern of 2.0 (Float16 is unavailable on Intel macOS).
        let candidates = Array(repeating: Array(repeating: UInt16(0x4000), count: width), count: 3)
        q04Marker("Q04_BODY_BEGIN \(scenario)")
        try q04Invoke(width: width, metric: metric, query: query, candidates: candidates, range: range,
                      out: UnsafeMutableBufferPointer(start: pointer.advanced(by: 2), count: invalid == "capacity" ? 2 : 10))
        q04Marker("Q04_UNEXPECTED_RETURN")
    }

    #if !(arch(x86_64) && (os(macOS) || targetEnvironment(macCatalyst)))
    /// Pins the Float16-free helpers and the literal bit constants above against the native type (arm64 / iOS family).
    @Test func float16FreeFixturesMatchNativeFloat16() {
        for n in -64...64 where n != 0 {
            #expect(q04HalfBits(numeratorOver32: n) == Float16(Float(n) / 32).bitPattern, "n=\(n)")
            #expect(q04HalfValue(bits: q04HalfBits(numeratorOver32: n)) == Double(n) / 32)
        }
        for n in [100, 640, 2047] {
            #expect(q04HalfBits(numeratorOver32: n) == Float16(Float(n) / 32).bitPattern)
        }
        #expect([1, 2, 3, 4, 5, 6, 7].map { Float16(Float($0) / 4).bitPattern } == [0x3400, 0x3800, 0x3A00, 0x3C00, 0x3D00, 0x3E00, 0x3F00])
        #expect(Float16.infinity.bitPattern == 0x7C00 && Float16.nan.bitPattern == 0x7E00 && Float16.leastNonzeroMagnitude.bitPattern == 1)
        #expect(Float16(2).bitPattern == 0x4000)
    }
    #endif
}
