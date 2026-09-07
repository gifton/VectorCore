import Testing
@testable import VectorCore

@Suite("FP16 range validation")
struct MixedPrecisionRangeValidationTests {
    @Test("Finite FP16 bit patterns are accepted", arguments: [512, 768, 1536])
    func acceptsFinitePatterns(dimension: Int) {
        let finite: [UInt16] = [
            0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF,
            0x0400, 0x8400, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF
        ]
        let values: [UInt16] = (0..<dimension).map { finite[$0 % finite.count] }
        #expect(validate(values, dimension: dimension))
    }

    @Test("Both infinities and NaN encodings are rejected at every boundary", arguments: [512, 768, 1536])
    func rejectsNonFinitePatterns(dimension: Int) {
        let nonFinite: [UInt16] = [0x7C00, 0xFC00, 0x7C01, 0xFC01, 0x7E00, 0xFE00, 0x7FFF, 0xFFFF]
        for bits in nonFinite {
            for index in [0, dimension / 2, dimension - 1] {
                var values: [UInt16] = Array(repeating: 0x3C00, count: dimension)
                values[index] = bits
                #expect(!validate(values, dimension: dimension), "bits=\(bits), index=\(index)")
            }
        }
    }

    @Test("FP32 range checks preserve finite endpoints and reject overflow")
    func checksFP32Range() {
        let finite: [Float] = [0, -0.0, 65504, -65504, 0x1p-24, -0x1p-24]
        #expect(MixedPrecisionKernels.Vector512FP16.canRepresent(finite))
        #expect(MixedPrecisionKernels.Vector768FP16.canRepresent(finite))
        #expect(MixedPrecisionKernels.Vector1536FP16.canRepresent(finite))
        let invalid: [Float] = [65505, -65505, .infinity, -.infinity, .nan]
        for value in invalid {
            #expect(!MixedPrecisionKernels.Vector512FP16.canRepresent([value]))
            #expect(!MixedPrecisionKernels.Vector768FP16.canRepresent([value]))
            #expect(!MixedPrecisionKernels.Vector1536FP16.canRepresent([value]))
        }
        let result = MixedPrecisionKernels.validateBatch(values: finite + invalid)
        #expect(!result.allValid)
        #expect(result.overflowCount == invalid.count)
        #expect(result.firstOverflowIndex == finite.count)
    }

    #if !(arch(x86_64) && (os(macOS) || targetEnvironment(macCatalyst)))
    @Test("Native overflow conversion preserves endpoints and signed zero")
    func checksNativeOverflowConversion() throws {
        #expect(try #require(MixedPrecisionKernels.detectOverflow(value: 65504)).bitPattern == 0x7BFF)
        #expect(try #require(MixedPrecisionKernels.detectOverflow(value: -65504)).bitPattern == 0xFBFF)
        #expect(try #require(MixedPrecisionKernels.detectOverflow(value: -0.0)).bitPattern == 0x8000)
        for value: Float in [65505, -65505, .infinity, -.infinity, .nan] {
            #expect(MixedPrecisionKernels.detectOverflow(value: value) == nil)
        }
    }
    #endif

    private func validate(_ values: [UInt16], dimension: Int) -> Bool {
        switch dimension {
        case 512: MixedPrecisionKernels.Vector512FP16(fp16Values: values).validateRange()
        case 768: MixedPrecisionKernels.Vector768FP16(fp16Values: values).validateRange()
        case 1536: MixedPrecisionKernels.Vector1536FP16(fp16Values: values).validateRange()
        default: preconditionFailure("Unsupported test dimension")
        }
    }
}
