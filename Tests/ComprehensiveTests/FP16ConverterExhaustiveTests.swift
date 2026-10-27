// Exhaustive, host-independent tests of the FP16 software converters and of every FP32/FP16 conversion path.
//
// The references in this file are integer bit manipulations written here; they never call a production converter. The
// file uses no half-precision Swift type, so it needs no platform guard. Its Intel macOS compilation is exercised by
// `swift build --build-tests --triple x86_64-apple-macosx14.0` (the CI platforms job); no test build for Mac Catalyst
// is exercised.
//
// NaN policy pinned here: encoding (FP32 to FP16) maps every NaN to sign | 0x7E00, keeping the sign and dropping the
// payload; decoding (FP16 to FP32) maps every NaN to 0x7FC00000, dropping the sign and the payload.

import Testing
@testable import VectorCore

// MARK: - Independent references

/// FP32 bits of an FP16 pattern, by integer bit widening. Every NaN pattern gives the decode policy value 0x7FC00000.
private func referenceDecode(_ h: UInt16) -> UInt32 {
    let sign = UInt32(h >> 15) << 31
    let exponent = UInt32((h >> 10) & 0x1F)
    let mantissa = UInt32(h & 0x3FF)
    if exponent == 0x1F { return mantissa == 0 ? sign | 0x7F80_0000 : 0x7FC0_0000 }
    if exponent != 0 { return sign | ((exponent + 112) << 23) | (mantissa << 13) }   // rebias 15 -> 127
    if mantissa == 0 { return sign }
    // Subnormal m * 2^-24: with the leading one of m at bit p, the value is 2^(p - 24) * (m / 2^p).
    let p = UInt32(31 - mantissa.leadingZeroBitCount)
    return sign | ((p + 103) << 23) | ((mantissa << (23 - p)) & 0x7F_FFFF)
}

/// FP16 bits of FP32 bits, by integer round-to-nearest-even of the exact value. Every NaN gives sign | 0x7E00.
private func referenceEncode(_ f: UInt32) -> UInt16 {
    let sign = UInt16(f >> 31) << 15
    let biased = Int((f >> 23) & 0xFF)
    let fraction = UInt64(f & 0x7F_FFFF)
    if biased == 0xFF { return sign | (fraction == 0 ? 0x7C00 : 0x7E00) }
    // |x| = significand * 2^scale, exactly.
    let significand = biased == 0 ? fraction : fraction | 0x80_0000
    let scale = biased == 0 ? -149 : biased - 150
    if significand == 0 { return sign }
    let log2 = 63 - significand.leadingZeroBitCount + scale      // floor(log2 |x|)
    if log2 > 15 { return sign | 0x7C00 }
    let quantum = max(log2, -14) - 10                              // FP16 spacing in that binade is 2^quantum
    // n = |x| / 2^quantum, rounded to nearest with ties to even.
    let drop = quantum - scale
    var n: UInt64
    if drop <= 0 {
        n = significand << UInt64(-drop)
    } else if drop > 40 {
        n = 0                                                      // |x| < 2^(quantum - 16): below half a spacing
    } else {
        n = significand >> UInt64(drop)
        let remainder = significand & ((UInt64(1) << UInt64(drop)) - 1)
        let half = UInt64(1) << UInt64(drop - 1)
        if remainder > half || (remainder == half && n & 1 == 1) { n += 1 }
    }
    if log2 < -14 { return sign | UInt16(n) }                      // n <= 1024; 1024 encodes the smallest normal, 0x0400
    var exponent = log2
    if n == 2048 { n = 1024; exponent += 1 }                       // the rounding carried into the next binade
    if exponent > 15 { return sign | 0x7C00 }
    return sign | UInt16(exponent + 15) << 10 | UInt16(n - 1024)
}

/// Exact value of a finite FP16 pattern from its fields, in Double: (-1)^s * m * 2^-24 for a subnormal (or zero), else
/// (-1)^s * (1024 + m) * 2^(e - 25). Used only to check `referenceDecode`.
private func exactValue(_ h: UInt16) -> Double {
    let exponent = Int((h >> 10) & 0x1F), mantissa = Int(h & 0x3FF)
    let magnitude = exponent == 0
        ? Double(sign: .plus, exponent: -24, significand: Double(mantissa))
        : Double(sign: .plus, exponent: exponent - 25, significand: Double(1024 + mantissa))
    return h >> 15 == 1 ? -magnitude : magnitude
}

private func isNaNPattern(_ h: UInt16) -> Bool { h & 0x7C00 == 0x7C00 && h & 0x03FF != 0 }

/// The encode policy value for a NaN with these FP32 bits.
private func canonicalNaN(_ f: UInt32) -> UInt16 { UInt16(f >> 31) << 15 | 0x7E00 }

private enum Fixture {
    static let allPatterns: [UInt16] = Array(0...UInt16.max)
    static let nonNaNPatterns: [UInt16] = allPatterns.filter { !isNaNPattern($0) }
    static let nanPatterns: [UInt16] = allPatterns.filter(isNaNPattern)

    /// The rounding inputs. For each of the 31,744 adjacent pairs (k, k + 1) of non-negative finite FP16 values
    /// (k = 0x0000 ... 0x7BFF, where the upper neighbour of 0x7BFF is 2^16, the value 0x7C00 would have with one more
    /// exponent step): the FP32 midpoint and its two FP32 neighbours, in both signs. Expected values come from the pair,
    /// not from a converter: the midpoint ties to the even member (so 65520 gives 0x7C00), the value just below the
    /// midpoint gives k, and the value just above it gives k + 1.
    static let roundingCases: [(input: UInt32, expected: UInt16, kind: String)] = {
        var cases: [(input: UInt32, expected: UInt16, kind: String)] = []
        cases.reserveCapacity(190_464)
        for k in UInt16(0)...0x7BFF {
            let upper = k == 0x7BFF ? 65_536.0 : exactValue(k + 1)
            let midpoint = (exactValue(k) + upper) / 2                       // 12 significant bits: exact in Double
            let bits = Float(midpoint).bitPattern
            precondition(Double(Float(midpoint)) == midpoint, "midpoint of \(k) is not exact in FP32")
            let even = k & 1 == 0 ? k : k + 1
            for sign: (bits: UInt32, half: UInt16) in [(0, 0), (0x8000_0000, 0x8000)] {
                cases.append((sign.bits | bits, sign.half | even, "tie"))
                cases.append((sign.bits | (bits - 1), sign.half | k, "below"))
                cases.append((sign.bits | (bits + 1), sign.half | (k + 1), "above"))
            }
        }
        return cases
    }()

    /// Named FP32 NaN patterns from the T12 review and probes (quiet and signalling, both signs, several payloads).
    static let namedNaNInputs: [UInt32] = [0x7FC1_2345, 0xFFC1_2345, 0x7F80_0001, 0xFF80_0001, 0x7FBF_FFFF, 0x7FFF_FFFF,
                                           0x7FA0_0000, 0x7F80_2000, 0xFFFF_E000, 0x7FC0_0000, 0xFFC0_0000]

    /// FP32 NaN inputs: every value of the ten high mantissa bits (bits 13-22, the bits an FP16 mantissa has room for),
    /// each with the low 13 bits 0, 1 or 0x1FFF, in both signs, except the two infinities (6,142 inputs); then the 11
    /// named patterns.
    static let nanInputs: [UInt32] = {
        var bits: [UInt32] = []
        for sign: UInt32 in [0, 0x8000_0000] {
            for high: UInt32 in 0..<1024 {
                for low: UInt32 in [0, 1, 0x1FFF] where high != 0 || low != 0 {
                    bits.append(sign | 0x7F80_0000 | high << 13 | low)
                }
            }
        }
        return bits + namedNaNInputs
    }()

    /// Extremes with hand-written expected encodings (not from either reference).
    static let extremes: [(input: UInt32, expected: UInt16)] = [
        (0x0000_0000, 0x0000), (0x8000_0000, 0x8000),        // +0, -0
        (0x0000_0001, 0x0000), (0x8000_0001, 0x8000),        // smallest FP32 subnormal
        (0x007F_FFFF, 0x0000), (0x0080_0000, 0x0000),        // largest FP32 subnormal, smallest FP32 normal
        (0x32FF_FFFF, 0x0000), (0x3300_0000, 0x0000),        // just below 2^-25; 2^-25 ties to even (0)
        (0x3300_0001, 0x0001), (0xB300_0001, 0x8001),        // just above 2^-25
        (0x3380_0000, 0x0001), (0x33C0_0000, 0x0002),        // 2^-24; 1.5 * 2^-24 ties to even (2)
        (0x387F_C000, 0x03FF), (0x3880_0000, 0x0400),        // largest FP16 subnormal; smallest FP16 normal 2^-14
        (0x3F80_0000, 0x3C00), (0x3F80_1000, 0x3C00),        // 1; 1 + 2^-11 ties to even
        (0x3F80_1001, 0x3C01), (0x3F80_3000, 0x3C02),        // just above that tie; 1 + 3 * 2^-11 ties to even
        (0x477F_E000, 0x7BFF), (0x477F_EFFF, 0x7BFF),        // 65504; just below 65520
        (0x477F_F000, 0x7C00), (0xC77F_F000, 0xFC00),        // 65520 ties to even (infinity)
        (0x4780_0000, 0x7C00), (0x7F7F_FFFF, 0x7C00),        // 65536; greatest finite FP32
        (0xFF7F_FFFF, 0xFC00),
        (0x7F80_0000, 0x7C00), (0xFF80_0000, 0xFC00)         // +infinity, -infinity
    ]
}

// MARK: - Production paths

private func floats(_ bits: [UInt32]) -> [Float] { bits.map(Float.init(bitPattern:)) }

/// SIMD4 lanes of `values`, padded with +0 to a multiple of `multiple` scalars.
private func lanes(_ values: [Float], paddedTo multiple: Int) -> ContiguousArray<SIMD4<Float>> {
    let total = (values.count + multiple - 1) / multiple * multiple
    let padded = values + [Float](repeating: 0, count: total - values.count)
    return ContiguousArray(stride(from: 0, to: total, by: 4).map {
        SIMD4<Float>(padded[$0], padded[$0 + 1], padded[$0 + 2], padded[$0 + 3])
    })
}

/// `values` split into vectors of `width` scalars (the last one padded with +0).
private func vectors<V>(_ values: [Float], width: Int, _ make: (ContiguousArray<SIMD4<Float>>) -> V) -> [V] {
    let storage = lanes(values, paddedTo: width)
    let lanesPerVector = width / 4
    return stride(from: 0, to: storage.count, by: lanesPerVector).map {
        make(ContiguousArray(storage[$0..<($0 + lanesPerVector)]))
    }
}

/// `values` split into arrays of `width` scalars (the last one padded with +0).
private func scalarChunks(_ values: [Float], width: Int) -> [[Float]] {
    stride(from: 0, to: values.count, by: width).map {
        Array(values[$0..<min($0 + width, values.count)]) + [Float](repeating: 0, count: max(0, $0 + width - values.count))
    }
}

/// The SoA storage read back in vector order: element d of vector v sits at group v / 4, offset d * 4 + v % 4.
private func soaRows(_ storage: ContiguousArray<UInt16>, vectorCount: Int, dimension: Int) -> [UInt16] {
    var rows: [UInt16] = []
    rows.reserveCapacity(vectorCount * dimension)
    for v in 0..<vectorCount {
        for d in 0..<dimension { rows.append(storage[(v / 4) * dimension * 4 + d * 4 + v % 4]) }
    }
    return rows
}

private func platformBatch(_ values: [Float]) -> [UInt16] {
    var out = [UInt16](repeating: 0xDEAD, count: values.count)
    values.withUnsafeBufferPointer { source in
        out.withUnsafeMutableBufferPointer { destination in
            MixedPrecisionKernels.platformOptimizedConvertBatch(source: source, destination: destination)
        }
    }
    return out
}

/// Production encode paths. Each maps FP32 values to FP16 bit patterns, element for element.
private enum EncodePath: String, CaseIterable {
    case scalar = "fp32ToFp16_scalar"
    case convertToFP16 = "convertToFP16"
    case vector512 = "Vector512FP16(from:)"
    case vector768 = "Vector768FP16(from:)"
    case vector1536 = "Vector1536FP16(from:)"
    case batch512 = "convertToFP16_512"
    case batch768 = "convertToFP16_768"
    case batch1536 = "convertToFP16_1536"
    case soa512 = "createSoA512FP16"
    case soa768 = "createSoA768FP16"
    case soa1536 = "createSoA1536FP16"
    case soaInit = "SoAFP16<Vector512Optimized>(vectors:)"
    case factory512 = "MixedPrecisionFactory.createVector512FP16(from:)"
    case factory768 = "MixedPrecisionFactory.createVector768FP16(from:)"
    case factory1536 = "MixedPrecisionFactory.createVector1536FP16(from:)"
    /// `platformOptimizedConvertBatch` with every element in the SIMD4 chunk loop, which runs while i + 3 < count
    /// (`convertBatchNEON`, `convertBatchAVX2`): the buffer is padded with 1.0 to a multiple of 4.
    case platformChunk = "platformOptimizedConvertBatch chunk"
    /// `platformOptimizedConvertBatch` with every element in the scalar tail: buffers of at most 3 elements, for which
    /// the chunk loop condition i + 3 < count never holds.
    case platformTail = "platformOptimizedConvertBatch tail"

    /// The paths the round-trip and rounding tests run (one per distinct conversion loop, plus the vector initializer and
    /// the SoA creator that wrap them).
    static let rounding: [EncodePath] = [.scalar, .convertToFP16, .vector512, .soa512, .platformChunk, .platformTail]

    func encode(_ values: [Float]) -> [UInt16] {
        switch self {
        case .scalar:
            return values.map { MixedPrecisionKernels.fp32ToFp16_scalar($0) }
        case .convertToFP16:
            return Array(MixedPrecisionKernels.convertToFP16(lanes(values, paddedTo: 4)).prefix(values.count))
        case .vector512:
            return trim(vectors(values, width: 512, Vector512Optimized.init(fromSIMDStorage:)).flatMap {
                MixedPrecisionKernels.Vector512FP16(from: $0).withUnsafeFP16BufferPointer { Array($0) }
            }, values.count)
        case .vector768:
            return trim(vectors(values, width: 768, Vector768Optimized.init(fromSIMDStorage:)).flatMap {
                MixedPrecisionKernels.Vector768FP16(from: $0).withUnsafeFP16BufferPointer { Array($0) }
            }, values.count)
        case .vector1536:
            return trim(vectors(values, width: 1536, Vector1536Optimized.init(fromSIMDStorage:)).flatMap {
                MixedPrecisionKernels.Vector1536FP16(from: $0).withUnsafeFP16BufferPointer { Array($0) }
            }, values.count)
        case .batch512:
            let input = vectors(values, width: 512, Vector512Optimized.init(fromSIMDStorage:))
            return trim(MixedPrecisionKernels.convertToFP16_512(input).flatMap {
                $0.withUnsafeFP16BufferPointer { Array($0) }
            }, values.count)
        case .batch768:
            let input = vectors(values, width: 768, Vector768Optimized.init(fromSIMDStorage:))
            return trim(MixedPrecisionKernels.convertToFP16_768(input).flatMap {
                $0.withUnsafeFP16BufferPointer { Array($0) }
            }, values.count)
        case .batch1536:
            let input = vectors(values, width: 1536, Vector1536Optimized.init(fromSIMDStorage:))
            return trim(MixedPrecisionKernels.convertToFP16_1536(input).flatMap {
                $0.withUnsafeFP16BufferPointer { Array($0) }
            }, values.count)
        case .soa512:
            let input = vectors(values, width: 512, Vector512Optimized.init(fromSIMDStorage:))
            let soa = MixedPrecisionKernels.createSoA512FP16(from: input)
            return trim(soaRows(soa.storage, vectorCount: soa.vectorCount, dimension: 512), values.count)
        case .soa768:
            let input = vectors(values, width: 768, Vector768Optimized.init(fromSIMDStorage:))
            let soa = MixedPrecisionKernels.createSoA768FP16(from: input)
            return trim(soaRows(soa.storage, vectorCount: soa.vectorCount, dimension: 768), values.count)
        case .soa1536:
            let input = vectors(values, width: 1536, Vector1536Optimized.init(fromSIMDStorage:))
            let soa = MixedPrecisionKernels.createSoA1536FP16(from: input)
            return trim(soaRows(soa.storage, vectorCount: soa.vectorCount, dimension: 1536), values.count)
        case .soaInit:
            let input = vectors(values, width: 512, Vector512Optimized.init(fromSIMDStorage:))
            do {
                let soa = try MixedPrecisionKernels.SoAFP16<Vector512Optimized>(vectors: input)
                return trim(soaRows(soa.storage, vectorCount: soa.vectorCount, dimension: 512), values.count)
            } catch {
                Issue.record("SoAFP16(vectors:) threw \(error)")
                return []
            }
        case .factory512:
            return trim(scalarChunks(values, width: 512).flatMap {
                MixedPrecisionFactory.createVector512FP16(from: $0).withUnsafeFP16BufferPointer { Array($0) }
            }, values.count)
        case .factory768:
            return trim(scalarChunks(values, width: 768).flatMap {
                MixedPrecisionFactory.createVector768FP16(from: $0).withUnsafeFP16BufferPointer { Array($0) }
            }, values.count)
        case .factory1536:
            return trim(scalarChunks(values, width: 1536).flatMap {
                MixedPrecisionFactory.createVector1536FP16(from: $0).withUnsafeFP16BufferPointer { Array($0) }
            }, values.count)
        case .platformChunk:
            return EncodePath.chunk(values, lead: 0)
        case .platformTail:
            return stride(from: 0, to: values.count, by: 3).flatMap {
                platformBatch(Array(values[$0..<min($0 + 3, values.count)]))
            }
        }
    }

    /// The chunk path with `lead` finite values (1.0) in front, which moves each input to another lane of its chunk.
    static func chunk(_ values: [Float], lead: Int) -> [UInt16] {
        let total = (lead + values.count + 3) / 4 * 4
        let buffer = [Float](repeating: 1, count: lead) + values
            + [Float](repeating: 1, count: total - lead - values.count)
        return Array(platformBatch(buffer)[lead..<(lead + values.count)])
    }

    private func trim(_ bits: [UInt16], _ count: Int) -> [UInt16] { Array(bits.prefix(count)) }
}

/// Production decode paths. Each maps FP16 bit patterns to FP32 bit patterns, element for element.
private enum DecodePath: String, CaseIterable {
    case scalar = "fp16ToFp32_scalar"
    case simd8 = "fp16ToFp32_simd8"
    case convertToFP32 = "convertToFP32"
    /// `convertToFP32` with 12 elements per call, so the last 4 of each call take its scalar remainder loop.
    case convertToFP32Remainder = "convertToFP32 remainder"
    case vector512 = "Vector512FP16.toFP32()"
    case vector768 = "Vector768FP16.toFP32()"
    case vector1536 = "Vector1536FP16.toFP32()"
    case soaGetVector = "SoAFP16.getVector(at:)"

    func decode(_ patterns: [UInt16]) -> [UInt32] {
        switch self {
        case .scalar:
            return patterns.map { MixedPrecisionKernels.fp16ToFp32_scalar($0).bitPattern }
        case .simd8:
            let padded = patterns + [UInt16](repeating: 0, count: (8 - patterns.count % 8) % 8)
            return Array(stride(from: 0, to: padded.count, by: 8).flatMap { i -> [UInt32] in
                let lanes = MixedPrecisionKernels.fp16ToFp32_simd8(SIMD8<UInt16>(padded[i..<(i + 8)]))
                return (0..<8).map { lanes[$0].bitPattern }
            }.prefix(patterns.count))
        case .convertToFP32:
            return DecodePath.viaConvertToFP32(patterns, elementsPerCall: (patterns.count + 7) / 8 * 8)
        case .convertToFP32Remainder:
            return DecodePath.viaConvertToFP32(patterns, elementsPerCall: 12)
        case .vector512:
            return DecodePath.viaVectors(patterns, width: 512) {
                MixedPrecisionKernels.Vector512FP16(fp16Values: $0).toFP32().storage
            }
        case .vector768:
            return DecodePath.viaVectors(patterns, width: 768) {
                MixedPrecisionKernels.Vector768FP16(fp16Values: $0).toFP32().storage
            }
        case .vector1536:
            return DecodePath.viaVectors(patterns, width: 1536) {
                MixedPrecisionKernels.Vector1536FP16(fp16Values: $0).toFP32().storage
            }
        case .soaGetVector:
            // Storage for 4 vectors of 512 dimensions; pattern j sits at vector j % 4, dimension j / 4 of its group.
            let dimension = 512, perGroup = 4 * dimension
            var result: [UInt32] = []
            for start in stride(from: 0, to: patterns.count, by: perGroup) {
                let group = Array(patterns[start..<min(start + perGroup, patterns.count)])
                let storage = ContiguousArray(group + [UInt16](repeating: 0, count: perGroup - group.count))
                let soa = MixedPrecisionKernels.SoAFP16<Vector512Optimized>(storage: storage, vectorCount: 4,
                                                                           dimension: dimension)
                do {
                    let rows = try (0..<4).map { try soa.getVector(at: $0) }
                    for j in 0..<group.count { result.append(rows[j % 4][j / 4].bitPattern) }
                } catch {
                    Issue.record("getVector threw \(error)")
                }
            }
            return result
        }
    }

    private static func viaConvertToFP32(_ patterns: [UInt16], elementsPerCall: Int) -> [UInt32] {
        var result: [UInt32] = []
        for start in stride(from: 0, to: patterns.count, by: elementsPerCall) {
            let chunk = Array(patterns[start..<min(start + elementsPerCall, patterns.count)])
            let padded = chunk + [UInt16](repeating: 0, count: (4 - chunk.count % 4) % 4)
            let lanes = MixedPrecisionKernels.convertToFP32(ContiguousArray(padded), laneCount: padded.count / 4)
            let scalars = lanes.flatMap { [$0[0], $0[1], $0[2], $0[3]] }
            result += scalars.prefix(chunk.count).map(\.bitPattern)
        }
        return result
    }

    private static func viaVectors(_ patterns: [UInt16], width: Int,
                                   _ decode: ([UInt16]) -> ContiguousArray<SIMD4<Float>>) -> [UInt32] {
        var result: [UInt32] = []
        for start in stride(from: 0, to: patterns.count, by: width) {
            let chunk = Array(patterns[start..<min(start + width, patterns.count)])
            let lanes = decode(chunk + [UInt16](repeating: 0, count: width - chunk.count))
            result += lanes.flatMap { [$0[0], $0[1], $0[2], $0[3]] }.prefix(chunk.count).map(\.bitPattern)
        }
        return result
    }
}

// MARK: - Comparison

private func hex<T: FixedWidthInteger>(_ value: T) -> String {
    let digits = String(value, radix: 16)
    return "0x" + String(repeating: "0", count: max(0, T.bitWidth / 4 - digits.count)) + digits
}

/// One expectation per call: the number of mismatches of `actual` against `expected`, with the first eight in hex.
private func expectBits<Input: FixedWidthInteger, Output: FixedWidthInteger>(
    _ path: String, inputs: [Input], actual: [Output], expected: [Output]
) {
    #expect(actual.count == expected.count, "\(path): \(actual.count) outputs for \(expected.count) inputs")
    var mismatches = 0
    var examples: [String] = []
    for i in 0..<min(actual.count, expected.count) where actual[i] != expected[i] {
        mismatches += 1
        if examples.count < 8 { examples.append("\(hex(inputs[i])) -> \(hex(actual[i])) (expected \(hex(expected[i])))") }
    }
    #expect(mismatches == 0, "\(path): \(mismatches) of \(expected.count) mismatch; first: \(examples.joined(separator: ", "))")
}

// MARK: - Tests

@Suite("FP16 converters: exhaustive checks and the NaN policy")
struct FP16ConverterExhaustiveTests {

    // MARK: References

    @Test("The decode reference equals the exact value of every finite pattern, and maps the infinities")
    func referenceDecodeIsExact() {
        let finite = Fixture.allPatterns.filter { $0 & 0x7C00 != 0x7C00 }
        #expect(finite.count == 63_488)
        expectBits("referenceDecode vs exact Double value", inputs: finite,
                   actual: finite.map { Double(Float(bitPattern: referenceDecode($0))).bitPattern },
                   expected: finite.map { exactValue($0).bitPattern })
        #expect(referenceDecode(0x7C00) == 0x7F80_0000 && referenceDecode(0xFC00) == 0xFF80_0000)
        // The encode reference inverts the decode reference on every non-NaN pattern.
        expectBits("referenceEncode(referenceDecode(h))", inputs: Fixture.nonNaNPatterns,
                   actual: Fixture.nonNaNPatterns.map { referenceEncode(referenceDecode($0)) },
                   expected: Fixture.nonNaNPatterns)
    }

    // MARK: Decode

    @Test("Every FP16 pattern decodes to the bit-widening reference (NaN: 0x7FC00000)")
    func decodeAllPatterns() {
        #expect(Fixture.allPatterns.count == 65_536)
        let expected = Fixture.allPatterns.map(referenceDecode)
        for path in [DecodePath.scalar, .simd8, .convertToFP32] {
            expectBits(path.rawValue, inputs: Fixture.allPatterns, actual: path.decode(Fixture.allPatterns),
                       expected: expected)
        }
    }

    // MARK: Encode

    @Test("Every non-NaN pattern survives decode then encode on every rounding path")
    func roundTripAllNonNaNPatterns() {
        #expect(Fixture.nonNaNPatterns.count == 63_490)
        let decoded = Fixture.nonNaNPatterns.map { MixedPrecisionKernels.fp16ToFp32_scalar($0) }
        for path in EncodePath.rounding {
            expectBits(path.rawValue, inputs: Fixture.nonNaNPatterns, actual: path.encode(decoded),
                       expected: Fixture.nonNaNPatterns)
        }
    }

    @Test("Midpoints tie to even and their neighbours round to the nearer value, on every rounding path")
    func encodeRoundToNearestEvenNeighbours() {
        let cases = Fixture.roundingCases
        #expect(cases.count == 190_464)
        #expect(cases.filter { $0.kind == "tie" }.count == 63_488)
        let inputs = cases.map { $0.input }, expected = cases.map { $0.expected }
        expectBits("referenceEncode", inputs: inputs, actual: inputs.map(referenceEncode), expected: expected)
        let values = floats(inputs)
        for path in EncodePath.rounding {
            expectBits(path.rawValue, inputs: inputs, actual: path.encode(values), expected: expected)
        }
    }

    @Test("Zeros, FP32 subnormals, the smallest and largest FP16 values, overflow and infinities")
    func encodeExtremes() {
        let inputs = Fixture.extremes.map { $0.input }, expected = Fixture.extremes.map { $0.expected }
        expectBits("referenceEncode", inputs: inputs, actual: inputs.map(referenceEncode), expected: expected)
        for path in EncodePath.allCases {
            expectBits(path.rawValue, inputs: inputs, actual: path.encode(floats(inputs)), expected: expected)
        }
    }

    // MARK: NaN policy, encode: every NaN becomes sign | 0x7E00

    private func expectCanonicalEncode(_ paths: [EncodePath]) {
        let inputs = Fixture.nanInputs, expected = inputs.map(canonicalNaN)
        #expect(inputs.count == 6_153)
        expectBits("referenceEncode", inputs: inputs, actual: inputs.map(referenceEncode), expected: expected)
        for path in paths {
            expectBits(path.rawValue, inputs: inputs, actual: path.encode(floats(inputs)), expected: expected)
            let named = Fixture.namedNaNInputs
            expectBits("\(path.rawValue), named patterns", inputs: named, actual: path.encode(floats(named)),
                       expected: named.map(canonicalNaN))
        }
    }

    @Test("NaN encode: fp32ToFp16_scalar")
    func nanEncodeScalar() { expectCanonicalEncode([.scalar]) }

    @Test("NaN encode: convertToFP16")
    func nanEncodeConvertToFP16() { expectCanonicalEncode([.convertToFP16]) }

    @Test("NaN encode: Vector{512,768,1536}FP16(from:), convertToFP16_{512,768,1536} and MixedPrecisionFactory")
    func nanEncodeVectorInitializers() {
        expectCanonicalEncode([.vector512, .vector768, .vector1536, .batch512, .batch768, .batch1536,
                               .factory512, .factory768, .factory1536])
    }

    @Test("NaN encode: createSoA{512,768,1536}FP16 and SoAFP16(vectors:)")
    func nanEncodeSoACreators() { expectCanonicalEncode([.soa512, .soa768, .soa1536, .soaInit]) }

    @Test("NaN encode: platformOptimizedConvertBatch SIMD4 chunks, every lane position")
    func nanEncodePlatformBatchChunks() {
        let inputs = Fixture.nanInputs, expected = inputs.map(canonicalNaN), values = floats(inputs)
        for lead in 0..<4 {
            expectBits("platformOptimizedConvertBatch chunk, lead \(lead)", inputs: inputs,
                       actual: EncodePath.chunk(values, lead: lead), expected: expected)
            let named = Fixture.namedNaNInputs
            expectBits("platformOptimizedConvertBatch chunk, lead \(lead), named patterns", inputs: named,
                       actual: EncodePath.chunk(floats(named), lead: lead), expected: named.map(canonicalNaN))
        }
    }

    @Test("NaN encode: platformOptimizedConvertBatch scalar tail")
    func nanEncodePlatformBatchTail() { expectCanonicalEncode([.platformTail]) }

    // MARK: NaN policy, decode: every NaN becomes 0x7FC00000

    private func expectCanonicalDecode(_ paths: [DecodePath]) {
        let patterns = Fixture.nanPatterns
        #expect(patterns.count == 2_046)
        let expected = [UInt32](repeating: 0x7FC0_0000, count: patterns.count)
        for path in paths {
            expectBits(path.rawValue, inputs: patterns, actual: path.decode(patterns), expected: expected)
        }
    }

    @Test("NaN decode: fp16ToFp32_scalar")
    func nanDecodeScalar() { expectCanonicalDecode([.scalar]) }

    @Test("NaN decode: fp16ToFp32_simd8")
    func nanDecodeSIMD8() { expectCanonicalDecode([.simd8]) }

    @Test("NaN decode: convertToFP32 (SIMD8 loop and remainder loop) and Vector{512,768,1536}FP16.toFP32()")
    func nanDecodeConvertToFP32() {
        expectCanonicalDecode([.convertToFP32, .convertToFP32Remainder, .vector512, .vector768, .vector1536])
    }

    @Test("NaN decode: SoAFP16.getVector(at:)")
    func nanDecodeSoAGetVector() { expectCanonicalDecode([.soaGetVector]) }
}
