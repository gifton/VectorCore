import Testing
@testable import VectorCore

@Suite("FP16 comparison helper contracts")
struct FP16ComparisonHelperTests {
    private let helpers = MixedPrecisionKernelTests()
    private let values: [Float] = [-.infinity, -1, -0.0, 0.0, 1, .infinity, .nan]
    // Literal expectations, independent of the helpers' classification branches.
    private let expected: [[Bool]] = [
        [true,  false, false, false, false, false, false],
        [false, true,  false, false, false, false, false],
        [false, false, true,  true,  false, false, false],
        [false, false, true,  true,  false, false, false],
        [false, false, false, false, true,  false, false],
        [false, false, false, false, false, true,  false],
        [false, false, false, false, false, false, true]
    ]

    @Test func testMixedClassification() {
        for i in values.indices {
            for j in values.indices {
                #expect(helpers.compareMixedVsFullPrecision(
                    mixed: [values[i]], full: [values[j]], relativeTolerance: 0, absoluteTolerance: 0
                ) == expected[i][j], "mixed/full classes \(i), \(j)")
            }
        }
    }

    @Test func testMixedShapesAndAcceptedPrefixes() {
        #expect(helpers.compareMixedVsFullPrecision(mixed: [], full: [], relativeTolerance: 0, absoluteTolerance: 0))
        #expect(!helpers.compareMixedVsFullPrecision(mixed: [], full: [0], relativeTolerance: 0, absoluteTolerance: 0))
        #expect(!helpers.compareMixedVsFullPrecision(mixed: [0], full: [], relativeTolerance: 0, absoluteTolerance: 0))
        let prefix: [Float] = [.nan, .infinity, -.infinity, -0.0, 0.0]
        #expect(helpers.compareMixedVsFullPrecision(mixed: prefix, full: prefix, relativeTolerance: 0, absoluteTolerance: 0))
        for bad: Float in [2, .nan, .infinity, -.infinity] {
            #expect(!helpers.compareMixedVsFullPrecision(
                mixed: prefix + [bad], full: prefix + [1], relativeTolerance: 0, absoluteTolerance: 0))
            #expect(!helpers.compareMixedVsFullPrecision(
                mixed: prefix + [1], full: prefix + [bad], relativeTolerance: 0, absoluteTolerance: 0))
        }
    }

    @Test func testMixedFiniteToleranceBoundaries() {
        for (mixed, full): (Float, Float) in [(1, 1.25), (1.25, 1)] {
            #expect(helpers.compareMixedVsFullPrecision(
                mixed: [mixed], full: [full], relativeTolerance: 0, absoluteTolerance: 0.25))
            #expect(!helpers.compareMixedVsFullPrecision(
                mixed: [mixed], full: [full], relativeTolerance: 0, absoluteTolerance: Float(0.25).nextDown))
        }
        for (mixed, full, tolerance): (Float, Float, Float) in [(4, 2, 1), (2, 4, 0.5), (0.25, 0, 0.25)] {
            #expect(helpers.compareMixedVsFullPrecision(
                mixed: [mixed], full: [full], relativeTolerance: tolerance, absoluteTolerance: 0))
            #expect(!helpers.compareMixedVsFullPrecision(
                mixed: [mixed], full: [full], relativeTolerance: tolerance.nextDown, absoluteTolerance: 0))
        }
        // A finite-input relative error can overflow while the absolute arm still accepts.
        #expect(helpers.compareMixedVsFullPrecision(
            mixed: [1], full: [.leastNonzeroMagnitude], relativeTolerance: 0, absoluteTolerance: 1))
        for sign: Float in [-1, 1] {
            #expect(!helpers.compareMixedVsFullPrecision(
                mixed: [sign * Float.greatestFiniteMagnitude], full: [-sign * Float.greatestFiniteMagnitude],
                relativeTolerance: 1, absoluteTolerance: 1))
        }
    }

    @Test func testMixedRejectsCorruptedKernelOutputsInEveryLane() throws {
        let query = try Vector512Optimized([Float](repeating: 1, count: 512))
        let candidates = try [Float(2), 3, 4, 5, 6].map {
            try Vector512Optimized([Float](repeating: $0, count: 512))
        }
        let converted = MixedPrecisionKernels.convertToFP16_512(candidates)
        var actual = [Float](repeating: 0, count: candidates.count)
        actual.withUnsafeMutableBufferPointer {
            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: query, candidatesFP16: converted, range: candidates.indices, out: $0)
        }
        let reference = candidates.map { query.euclideanDistanceSquared(to: $0) }
        #expect(actual.allSatisfy { $0.isFinite })
        #expect(helpers.compareMixedVsFullPrecision(mixed: actual, full: reference, relativeTolerance: 0, absoluteTolerance: 0))
        for replacement: Float in [.nan, .infinity, -.infinity] {
            for index in actual.indices {
                var corrupted = actual
                corrupted[index] = replacement
                #expect(!helpers.compareMixedVsFullPrecision(
                    mixed: corrupted, full: reference, relativeTolerance: 0.005, absoluteTolerance: 0))
                #expect(!helpers.compareMixedVsFullPrecision(
                    mixed: reference, full: corrupted, relativeTolerance: 0.005, absoluteTolerance: 0))
            }
        }
    }

    #if !(arch(x86_64) && (os(macOS) || targetEnvironment(macCatalyst)))
    @Test func testNativeClassification() {
        let halfValues: [Float16] = [-.infinity, -1, -0.0, 0.0, 1, .infinity, .nan]
        for i in values.indices {
            for j in halfValues.indices {
                #expect(helpers.verifyFP16Accuracy(original: [values[i]], converted: [halfValues[j]], tolerance: 0)
                        == expected[i][j], "original/converted classes \(i), \(j)")
            }
        }
    }

    @Test func testNativeShapesAndAcceptedPrefixes() {
        #expect(helpers.verifyFP16Accuracy(original: [], converted: [], tolerance: 0))
        #expect(!helpers.verifyFP16Accuracy(original: [], converted: [0], tolerance: 0))
        #expect(!helpers.verifyFP16Accuracy(original: [0], converted: [], tolerance: 0))
        let originalPrefix: [Float] = [.nan, .infinity, -.infinity, 0x1p-15, -0.0, 0.0]
        let convertedPrefix: [Float16] = [.nan, .infinity, -.infinity, 0, -0.0, 0.0]
        #expect(helpers.verifyFP16Accuracy(original: originalPrefix, converted: convertedPrefix, tolerance: 0))
        for bad: Float in [2, .nan, .infinity, -.infinity] {
            #expect(!helpers.verifyFP16Accuracy(original: originalPrefix + [bad], converted: convertedPrefix + [1], tolerance: 0))
            #expect(!helpers.verifyFP16Accuracy(original: originalPrefix + [1], converted: convertedPrefix + [Float16(bad)], tolerance: 0))
        }
    }

    @Test func testNativeFlushBoundaryAndDirection() {
        for tiny: Float in [-0x1p-15, 0x1p-15] {
            for zero: Float16 in [-0.0, 0.0] {
                #expect(helpers.verifyFP16Accuracy(original: [tiny], converted: [zero], tolerance: 0))
                #expect(!helpers.verifyFP16Accuracy(original: [Float(zero)], converted: [Float16(tiny)], tolerance: 0))
            }
        }
        for boundary: Float in [-0x1p-14, 0x1p-14] {
            for zero: Float16 in [-0.0, 0.0] {
                #expect(!helpers.verifyFP16Accuracy(original: [boundary], converted: [zero], tolerance: 0))
            }
        }
    }

    @Test func testNativeFiniteToleranceBoundaries() {
        for (original, converted): (Float, Float16) in [(0.5, 0.75), (0.75, 0.5)] {
            #expect(helpers.verifyFP16Accuracy(original: [original], converted: [converted], tolerance: 0.25))
            #expect(!helpers.verifyFP16Accuracy(original: [original], converted: [converted], tolerance: Float(0.25).nextDown))
        }
        for (original, converted, tolerance): (Float, Float16, Float) in [(2, 4, 1), (4, 2, 0.5), (0, 0.25, 0.25)] {
            #expect(helpers.verifyFP16Accuracy(original: [original], converted: [converted], tolerance: tolerance))
            #expect(!helpers.verifyFP16Accuracy(original: [original], converted: [converted], tolerance: tolerance.nextDown))
        }
        #expect(helpers.verifyFP16Accuracy(original: [.leastNonzeroMagnitude], converted: [1], tolerance: 1))
    }

    @Test func testNativeRejectsCorruptedConversionOutputsInEveryLane() throws {
        let original = (0..<512).map { Float($0 + 1) / 8 }
        let vector = try Vector512Optimized(original)
        let converted = MixedPrecisionKernels.Vector512FP16(from: vector).storage.map { Float16(bitPattern: $0) }
        #expect(helpers.verifyFP16Accuracy(original: original, converted: converted, tolerance: 0))
        for replacement: Float16 in [.nan, .infinity, -.infinity] {
            for index in converted.indices {
                var corruptOutput = converted
                corruptOutput[index] = replacement
                #expect(!helpers.verifyFP16Accuracy(original: original, converted: corruptOutput, tolerance: 0.001))
                var corruptReference = original
                corruptReference[index] = Float(replacement)
                #expect(!helpers.verifyFP16Accuracy(original: corruptReference, converted: converted, tolerance: 0.001))
            }
        }
    }
    #endif
}
