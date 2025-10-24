//
//  NumericalStabilityHelpersValidation.swift
//  VectorCore
//
//  Quick validation tests for numerical stability helper functions
//  Ensures helpers work correctly before using them in main test suite
//

import Testing
import Foundation
@testable import VectorCore

@Suite("Helper Function Validation")
struct NumericalStabilityHelperValidationTests {

    @Test("Vector generation helpers work correctly")
    func testVectorGenerationHelpers() async throws {
        // Test createUniformLargeVector
        let uniform = NumericalStabilityHelpers.createUniformLargeVector(
            dimension: Dim32.self,
            value: 1000.0
        )
        #expect(uniform.scalarCount == 32)
        #expect(uniform[0] == 1000.0)
        #expect(uniform[31] == 1000.0)

        // Test createSingleLargeValueVector
        let single = try NumericalStabilityHelpers.createSingleLargeValueVector(
            dimension: Dim32.self,
            value: 5000.0,
            at: 10
        )
        #expect(single[10] == 5000.0)
        #expect(single[0] == 0.0)
        #expect(single[31] == 0.0)

        // Test createMixedScaleVector
        let mixed = try NumericalStabilityHelpers.createMixedScaleVector(
            dimension: Dim32.self,
            largeValue: 1e10,
            smallValue: 1e-5,
            largeCount: 5
        )
        #expect(mixed[0] == 1e10)
        #expect(mixed[4] == 1e10)
        #expect(mixed[5] == 1e-5)
        #expect(mixed[31] == 1e-5)

        // Test createRandomExtremeVector
        let random = try NumericalStabilityHelpers.createRandomExtremeVector(
            dimension: Dim32.self,
            minExponent: -5,
            maxExponent: 5
        )
        #expect(random.scalarCount == 32)
        // All values should be in range [1e-5, 1e+5]
        for i in 0..<32 {
            let absValue = abs(random[i])
            #expect(absValue >= 1e-6 && absValue <= 1e+6,
                    "Random value \(random[i]) outside expected range")
        }
    }

    @Test("Magnitude calculation helpers work correctly")
    func testMagnitudeCalculationHelpers() async throws {
        // Test computeUniformVectorMagnitude
        let expected = NumericalStabilityHelpers.computeUniformVectorMagnitude(
            value: 2.0,
            dimension: 100
        )
        let expectedValue: Float = 2.0 * sqrt(100.0)
        #expect(abs(expected - expectedValue) < 1e-6)

        // Test computeExpectedMagnitude with simple vector
        let vector = try Vector<Dim4>([3, 4, 0, 0])
        let magnitude = NumericalStabilityHelpers.computeExpectedMagnitude(vector)
        #expect(abs(magnitude - 5.0) < 1e-6, "Expected magnitude 5.0, got \(magnitude)")

        // Test with zero vector
        let zeroVector = Vector<Dim4>.zero
        let zeroMagnitude = NumericalStabilityHelpers.computeExpectedMagnitude(zeroVector)
        #expect(zeroMagnitude == 0.0)

        // Test stable magnitude with large values
        let largeVector = try NumericalStabilityHelpers.createSingleLargeValueVector(
            dimension: Dim32.self,
            value: 1e19,
            at: 0
        )
        let largeMagnitude = NumericalStabilityHelpers.computeExpectedMagnitude(largeVector)
        #expect(largeMagnitude.isFinite, "Stable magnitude should be finite for large values")
        #expect(abs(largeMagnitude - 1e19) / 1e19 < 1e-5, "Magnitude should be accurate")
    }

    @Test("Validation helpers work correctly")
    func testValidationHelpers() async throws {
        // Test isWithinRelativeTolerance
        #expect(NumericalStabilityHelpers.isWithinRelativeTolerance(
            actual: 1.0,
            expected: 1.0,
            relativeTolerance: 1e-5
        ))

        #expect(NumericalStabilityHelpers.isWithinRelativeTolerance(
            actual: 1.00001,
            expected: 1.0,
            relativeTolerance: 1e-4
        ))

        #expect(!NumericalStabilityHelpers.isWithinRelativeTolerance(
            actual: 1.1,
            expected: 1.0,
            relativeTolerance: 1e-5
        ))

        // Test relativeError
        let error = NumericalStabilityHelpers.relativeError(
            actual: 1.05,
            expected: 1.0
        )
        #expect(abs(error - 0.05) < 1e-6)

        // Test isUnitVector
        let unitVector = try Vector<Dim4>([1, 0, 0, 0])
        #expect(NumericalStabilityHelpers.isUnitVector(unitVector))

        let nonUnitVector = try Vector<Dim4>([2, 0, 0, 0])
        #expect(!NumericalStabilityHelpers.isUnitVector(nonUnitVector))

        // Test isEffectivelyZero
        let zeroVector = Vector<Dim4>.zero
        #expect(NumericalStabilityHelpers.isEffectivelyZero(zeroVector))

        let nonZeroVector = try Vector<Dim4>([1e-5, 0, 0, 0])
        #expect(!NumericalStabilityHelpers.isEffectivelyZero(nonZeroVector, epsilon: 1e-10))
    }

    @Test("Overflow detection helpers work correctly")
    func testOverflowDetectionHelpers() async throws {
        // Test willOverflowWhenSquared
        #expect(!NumericalStabilityHelpers.willOverflowWhenSquared(1.0))
        #expect(!NumericalStabilityHelpers.willOverflowWhenSquared(1e10))
        #expect(!NumericalStabilityHelpers.willOverflowWhenSquared(1e18))

        // This should overflow
        #expect(NumericalStabilityHelpers.willOverflowWhenSquared(2e19))
        #expect(NumericalStabilityHelpers.willOverflowWhenSquared(Float.greatestFiniteMagnitude))

        // Test willMagnitudeOverflow
        let safeVector = try Vector<Dim4>([1e10, 1e10, 1e10, 1e10])
        #expect(!NumericalStabilityHelpers.willMagnitudeOverflow(safeVector))

        let unsafeVector = try NumericalStabilityHelpers.createSingleLargeValueVector(
            dimension: Dim4.self,
            value: 2e19,
            at: 0
        )
        #expect(NumericalStabilityHelpers.willMagnitudeOverflow(unsafeVector))
    }

    @Test("Diagnostic helper produces useful output")
    func testDiagnosticHelper() async throws {
        let vector = try Vector<Dim4>([3, 4, 0, 0])
        let diagnostic = NumericalStabilityHelpers.magnitudeDiagnostic(
            vector: vector,
            actual: 5.0,
            expected: 5.0
        )

        #expect(diagnostic.contains("Dimension: 4"))
        #expect(diagnostic.contains("Expected:  5.0"))
        #expect(diagnostic.contains("Actual:    5.0"))

        // Test with overflow scenario
        let largeVector = try NumericalStabilityHelpers.createSingleLargeValueVector(
            dimension: Dim4.self,
            value: 3e19,
            at: 0
        )
        let overflowDiagnostic = NumericalStabilityHelpers.magnitudeDiagnostic(
            vector: largeVector,
            actual: Float.infinity,
            expected: 3e19
        )

        #expect(overflowDiagnostic.contains("WARNING"))
        #expect(overflowDiagnostic.contains("INFINITE"))
    }

    @Test("Stable magnitude handles extreme values correctly")
    func testStableMagnitudeExtremeValues() async throws {
        // Test with values that would overflow naive implementation
        let largeValue: Float = 1.5e19  // Near sqrt(Float.max)
        let vector = NumericalStabilityHelpers.createUniformLargeVector(
            dimension: Dim512.self,
            value: largeValue
        )

        // Stable magnitude should work
        let stableMag = NumericalStabilityHelpers.computeExpectedMagnitude(vector)
        #expect(stableMag.isFinite, "Stable magnitude should be finite")

        // Expected: largeValue * sqrt(512)
        let expected = largeValue * sqrt(512.0)
        let relError = abs(stableMag - expected) / expected
        #expect(relError < 0.01, "Stable magnitude should be accurate within 1%")

        // Current VectorCore magnitude will overflow
        let naiveMag = vector.magnitude
        // This will be inf in current implementation
        print("DEBUG: Stable mag = \(stableMag), Naive mag = \(naiveMag)")
    }
}
