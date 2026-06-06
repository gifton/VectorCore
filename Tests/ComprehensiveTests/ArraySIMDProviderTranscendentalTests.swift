import Foundation
import Testing
@testable import VectorCore

/// Tests for the transcendental array operations (`abs`, `sqrt`, `log`) and
/// `softmax`, which route through vForce (Accelerate) when available and fall
/// back to scalar `map` otherwise. Each test asserts numerical equivalence to a
/// scalar reference within float tolerance (vForce may differ by ~1 ULP) and
/// covers the empty-array edge case.
@Suite("ArraySIMD Transcendentals (FIX 2.2)")
struct ArraySIMDProviderTranscendentalSuite {

    // Exercise both concrete providers to ensure the vForce path is hit
    // regardless of which one is in use.
    private let providers: [any ArraySIMDProvider] = [
        DefaultArraySIMDProvider(),
        SwiftSIMDProvider()
    ]

    @Test
    func testSqrtMatchesScalarReference() {
        let input: [Float] = [0, 0.25, 1, 2, 4, 9, 16, 1e-4, 1e4, 123.456]
        for provider in providers {
            let result = provider.sqrt(input)
            #expect(result.count == input.count)
            for i in 0..<input.count {
                #expect(abs(result[i] - Foundation.sqrt(input[i])) < 1e-5,
                        "sqrt mismatch at \(i): \(result[i]) vs \(Foundation.sqrt(input[i]))")
            }
        }
    }

    @Test
    func testLogMatchesScalarReference() {
        // Strictly positive inputs (log domain).
        let input: [Float] = [1e-4, 0.5, 1, 2, Float(M_E), 10, 100, 1234.5]
        for provider in providers {
            let result = provider.log(input)
            #expect(result.count == input.count)
            for i in 0..<input.count {
                #expect(abs(result[i] - Foundation.log(input[i])) < 1e-5,
                        "log mismatch at \(i): \(result[i]) vs \(Foundation.log(input[i]))")
            }
        }
    }

    @Test
    func testAbsMatchesScalarReference() {
        let input: [Float] = [-3, -2.5, -1e-4, 0, 1e-4, 2.5, 3, -1234.5, 9999]
        for provider in providers {
            let result = provider.abs(input)
            #expect(result.count == input.count)
            for i in 0..<input.count {
                #expect(abs(result[i] - Swift.abs(input[i])) < 1e-5,
                        "abs mismatch at \(i): \(result[i]) vs \(Swift.abs(input[i]))")
            }
        }
    }

    @Test
    func testTranscendentalsEmptyArray() {
        let empty: [Float] = []
        for provider in providers {
            #expect(provider.sqrt(empty).isEmpty)
            #expect(provider.log(empty).isEmpty)
            #expect(provider.abs(empty).isEmpty)
        }
    }

    @Test
    func testSoftmaxMatchesScalarReference() {
        let logits: [Float] = [1, 2, 3, -1, 0.5, 10, -10, 4.25]
        let v = try! Vector<Dim8>(logits)
        let result = v.softmax().toArray()

        // Scalar reference: shift by max, exp, normalize.
        let maxVal = logits.max()!
        let exps = logits.map { Foundation.exp($0 - maxVal) }
        let sum = exps.reduce(0, +)
        let reference = exps.map { $0 / sum }

        #expect(result.count == reference.count)
        for i in 0..<reference.count {
            #expect(abs(result[i] - reference[i]) < 1e-5,
                    "softmax mismatch at \(i): \(result[i]) vs \(reference[i])")
        }
        // Probability distribution sanity: sums to 1.
        #expect(abs(result.reduce(0, +) - 1) < 1e-5)
    }
}
