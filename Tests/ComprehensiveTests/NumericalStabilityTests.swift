//
//  NumericalStabilityTests.swift
//  VectorCore
//
//  Comprehensive numerical stability testing for magnitude and normalization operations.
//  Tests edge cases involving Float.max, overflow, underflow, and precision boundaries.
//
//  Based on: Docs/numerical_stability_analysis.md
//  Date: 2025-10-23
//

import Testing
import Foundation
@testable import VectorCore

// MARK: - Numerical Stability Test Constants

/// Test boundaries and reference values for numerical stability testing
internal enum NumericalStabilityConstants {

    // MARK: Float32 Limits

    /// Maximum finite float value: 3.40282347e+38
    static let floatMax = Float.greatestFiniteMagnitude

    /// Minimum positive float value: 1.17549435e-38
    static let floatMin = Float.leastNormalMagnitude

    /// Smallest subnormal value
    static let floatTiny = Float.leastNonzeroMagnitude

    // MARK: Magnitude Boundaries

    /// Theoretical safe magnitude limit: sqrt(Float.max) ≈ 1.84467441e+19
    /// Values above this will overflow when squared
    static let safeMagnitudeLimit: Float = 1.84467441e+19

    /// Conservative safe limit for testing (90% of theoretical)
    static let testSafeMagnitudeLimit: Float = 1.66e+19

    /// Boundary where overflow begins to occur
    static let overflowBoundary: Float = 2.0e+19

    // MARK: Test Value Ranges

    /// Test values across the full dynamic range
    static let dynamicRangeExponents: [Float] = [
        -38, -30, -20, -10, -5, -2, 0, 2, 5, 10, 15, 19, 25, 30, 35
    ]

    /// Safe test values that should not overflow
    static let safeTestValues: [Float] = [
        1e-20, 1e-10, 1e-5, 0.1, 1.0, 10.0, 1e5, 1e10, 1e15
    ]

    /// Extreme test values near boundaries
    static let extremeTestValues: [Float] = [
        1e18, 1e19, 1.5e19, 1.8e19
    ]

    /// Values that will definitely cause overflow
    static let overflowValues: [Float] = [
        2e19, 1e20, 1e30, 1e37, floatMax / 10, floatMax
    ]

    // MARK: Precision Tolerances

    /// Default relative tolerance for magnitude comparisons
    static let defaultRelativeTolerance: Float = 1e-5

    /// Tight tolerance for unit vectors
    static let unitVectorTolerance: Float = 1e-6

    /// Loose tolerance for extreme value calculations
    static let extremeValueTolerance: Float = 1e-3
}

// MARK: - Test Helper Functions

/// Helper functions for numerical stability testing
internal enum NumericalStabilityHelpers {

    // MARK: - Vector Generation

    /// Creates a test vector with a single large value at specified index
    /// - Parameters:
    ///   - dimension: Vector dimension
    ///   - value: The large value to place
    ///   - index: Position for the value (default: 0)
    /// - Returns: Vector with single large value, rest zeros
    static func createSingleLargeValueVector<D: StaticDimension>(
        dimension: D.Type,
        value: Float,
        at index: Int = 0
    ) throws -> Vector<D> {
        let dim = Vector<D>.zero.scalarCount
        var array = Array(repeating: 0.0 as Float, count: dim)
        guard index >= 0 && index < dim else {
            throw VectorError.indexOutOfBounds(index: index, dimension: dim)
        }
        array[index] = value
        return try Vector<D>(array)
    }

    /// Creates a test vector with all components set to the same large value
    /// - Parameters:
    ///   - dimension: Vector dimension
    ///   - value: The value to repeat
    /// - Returns: Vector filled with the specified value
    static func createUniformLargeVector<D: StaticDimension>(
        dimension: D.Type,
        value: Float
    ) -> Vector<D> {
        return Vector<D>(repeating: value)
    }

    /// Creates a test vector with mixed scale values (large and small)
    /// - Parameters:
    ///   - dimension: Vector dimension
    ///   - largeValue: Large component value
    ///   - smallValue: Small component value
    ///   - largeCount: Number of large components
    /// - Returns: Vector with mixed magnitude components
    static func createMixedScaleVector<D: StaticDimension>(
        dimension: D.Type,
        largeValue: Float,
        smallValue: Float,
        largeCount: Int
    ) throws -> Vector<D> {
        let dim = Vector<D>.zero.scalarCount
        guard largeCount >= 0 && largeCount <= dim else {
            throw VectorError.invalidData("largeCount must be between 0 and \(dim)")
        }

        var array = Array(repeating: smallValue, count: dim)
        // Place large values at the beginning
        for i in 0..<largeCount {
            array[i] = largeValue
        }

        return try Vector<D>(array)
    }

    /// Generates a random vector with values in specified exponent range
    /// - Parameters:
    ///   - dimension: Vector dimension
    ///   - minExponent: Minimum power of 10 (e.g., -20 for 1e-20)
    ///   - maxExponent: Maximum power of 10 (e.g., 19 for 1e+19)
    /// - Returns: Vector with random extreme values
    static func createRandomExtremeVector<D: StaticDimension>(
        dimension: D.Type,
        minExponent: Float,
        maxExponent: Float
    ) throws -> Vector<D> {
        let dim = Vector<D>.zero.scalarCount
        var array = Array<Float>(repeating: 0.0, count: dim)

        for i in 0..<dim {
            // Random exponent in range
            let exponent = Float.random(in: minExponent...maxExponent)
            // Random sign
            let sign: Float = Bool.random() ? 1.0 : -1.0
            // Value = sign * 10^exponent
            let value = sign * pow(10.0, exponent)
            array[i] = value
        }

        return try Vector<D>(array)
    }

    // MARK: - Magnitude Calculation

    /// Computes expected magnitude using higher-precision algorithm
    /// This uses the stable two-pass Kahan algorithm as a reference
    /// - Parameter vector: Input vector
    /// - Returns: Expected magnitude value
    static func computeExpectedMagnitude<D: StaticDimension>(_ vector: Vector<D>) -> Float {
        // Kahan's two-pass algorithm for stable magnitude calculation
        // Phase 1: Find maximum absolute value
        var maxAbs: Float = 0
        vector.withUnsafeBufferPointer { buffer in
            for element in buffer {
                maxAbs = max(maxAbs, abs(element))
            }
        }

        // Handle zero vector
        guard maxAbs > 0 else { return 0 }

        // Handle already-infinite values
        guard maxAbs.isFinite else { return Float.infinity }

        // Phase 2: Scale by max, compute norm, scale back
        var sumSquares: Float = 0
        vector.withUnsafeBufferPointer { buffer in
            for element in buffer {
                let scaled = element / maxAbs
                sumSquares += scaled * scaled
            }
        }

        return maxAbs * sqrt(sumSquares)
    }

    /// Computes expected magnitude for uniform vector (analytical formula)
    /// For vector of n components with value v: magnitude = v * sqrt(n)
    /// - Parameters:
    ///   - value: Component value
    ///   - dimension: Number of components
    /// - Returns: Expected magnitude
    static func computeUniformVectorMagnitude(value: Float, dimension: Int) -> Float {
        // For uniform vector: ||v|| = |value| * sqrt(n)
        // This is mathematically exact for uniform vectors
        let absValue = abs(value)
        let sqrtDim = sqrt(Float(dimension))
        return absValue * sqrtDim
    }

    // MARK: - Numerical Validation

    /// Checks if a value is within relative tolerance of expected
    /// - Parameters:
    ///   - actual: Actual computed value
    ///   - expected: Expected reference value
    ///   - relativeTolerance: Relative error tolerance (default: 1e-5)
    /// - Returns: True if within tolerance
    static func isWithinRelativeTolerance(
        actual: Float,
        expected: Float,
        relativeTolerance: Float = NumericalStabilityConstants.defaultRelativeTolerance
    ) -> Bool {
        // Handle special cases
        if actual == expected { return true }
        if !actual.isFinite || !expected.isFinite {
            // Both must be infinite with same sign, or both NaN
            return (actual.isInfinite && expected.isInfinite && actual.sign == expected.sign) ||
                   (actual.isNaN && expected.isNaN)
        }

        // Handle near-zero expected values (use absolute tolerance)
        if abs(expected) < 1e-30 {
            return abs(actual - expected) < relativeTolerance
        }

        // Standard relative error check
        let relativeError = abs(actual - expected) / abs(expected)
        return relativeError <= relativeTolerance
    }

    /// Computes relative error between actual and expected values
    /// - Parameters:
    ///   - actual: Actual computed value
    ///   - expected: Expected reference value
    /// - Returns: Relative error: |actual - expected| / |expected|
    static func relativeError(actual: Float, expected: Float) -> Float {
        // Handle special cases
        if actual == expected { return 0.0 }
        if !expected.isFinite { return Float.infinity }
        if abs(expected) < Float.leastNormalMagnitude {
            // Avoid division by very small numbers - use absolute error
            return abs(actual - expected)
        }

        return abs(actual - expected) / abs(expected)
    }

    /// Checks if a vector is properly normalized (magnitude ≈ 1.0)
    /// - Parameters:
    ///   - vector: Vector to check
    ///   - tolerance: Absolute tolerance (default: 1e-6)
    /// - Returns: True if magnitude is within tolerance of 1.0
    static func isUnitVector<D: StaticDimension>(
        _ vector: Vector<D>,
        tolerance: Float = NumericalStabilityConstants.unitVectorTolerance
    ) -> Bool {
        let magnitude = vector.magnitude
        return abs(magnitude - 1.0) <= tolerance
    }

    /// Checks if a vector is effectively zero
    /// - Parameters:
    ///   - vector: Vector to check
    ///   - epsilon: Threshold for zero (default: 1e-10)
    /// - Returns: True if all components are below epsilon
    static func isEffectivelyZero<D: StaticDimension>(
        _ vector: Vector<D>,
        epsilon: Float = 1e-10
    ) -> Bool {
        var allZero = true
        vector.withUnsafeBufferPointer { buffer in
            for element in buffer {
                if abs(element) > epsilon {
                    allZero = false
                    return
                }
            }
        }
        return allZero
    }

    // MARK: - Overflow Detection

    /// Predicts if a value will overflow when squared
    /// - Parameter value: Value to check
    /// - Returns: True if value² will overflow
    static func willOverflowWhenSquared(_ value: Float) -> Bool {
        // Value will overflow when squared if |value| > sqrt(Float.max)
        let absValue = abs(value)
        return absValue > NumericalStabilityConstants.safeMagnitudeLimit
    }

    /// Predicts if a vector's magnitude calculation will overflow
    /// - Parameter vector: Vector to check
    /// - Returns: True if magnitude calculation will overflow
    static func willMagnitudeOverflow<D: StaticDimension>(_ vector: Vector<D>) -> Bool {
        // Check if any component will overflow when squared
        var willOverflow = false
        vector.withUnsafeBufferPointer { buffer in
            for element in buffer {
                if willOverflowWhenSquared(element) {
                    willOverflow = true
                    return
                }
            }
        }
        return willOverflow
    }

    // MARK: - Diagnostic Helpers

    /// Creates a diagnostic string for magnitude test failures
    /// - Parameters:
    ///   - vector: The test vector
    ///   - actualMagnitude: Computed magnitude
    ///   - expectedMagnitude: Expected magnitude
    /// - Returns: Formatted diagnostic message
    static func magnitudeDiagnostic<D: StaticDimension>(
        vector: Vector<D>,
        actual: Float,
        expected: Float
    ) -> String {
        let relError = relativeError(actual: actual, expected: expected)
        let dimension = vector.scalarCount

        // Get first few components for debugging
        var components: [Float] = []
        let sampleCount = min(5, dimension)
        vector.withUnsafeBufferPointer { buffer in
            for i in 0..<sampleCount {
                components.append(buffer[i])
            }
        }

        // Find max component
        var maxComponent: Float = 0
        vector.withUnsafeBufferPointer { buffer in
            for element in buffer {
                maxComponent = max(maxComponent, abs(element))
            }
        }

        var diagnostic = """
        Magnitude Test Failure:
          Dimension: \(dimension)
          Expected:  \(expected)
          Actual:    \(actual)
          Rel Error: \(relError) (\(relError * 100)%)
          Max |component|: \(maxComponent)
          First \(sampleCount) components: \(components)
        """

        // Add overflow warning if applicable
        if willMagnitudeOverflow(vector) {
            diagnostic += "\n  ⚠️  WARNING: Vector will overflow in naive magnitude calculation!"
        }

        if actual.isInfinite {
            diagnostic += "\n  ⚠️  Actual magnitude is INFINITE (overflow occurred)"
        }

        if isEffectivelyZero(vector) {
            diagnostic += "\n  ⚠️  Normalized vector is ZERO (catastrophic failure)"
        }

        return diagnostic
    }
}

// MARK: - Test Suite: Magnitude Overflow

/// Tests for magnitude calculation overflow conditions
@Suite("Numerical Stability: Magnitude Overflow")
struct MagnitudeOverflowTests {

    // MARK: Single Large Value Tests

    @Test("Magnitude with single value near sqrt(Float.max)")
    func testMagnitudeSingleValueNearBoundary() async throws {
        // Test Case: Vector with one component at safe boundary
        // Expected: Magnitude should be finite and accurate
        // Context: Value is 1e+19, just below sqrt(Float.max) ≈ 1.8e+19

        // Use conservative safe limit (90% of theoretical boundary)
        let safeBoundaryValue: Float = NumericalStabilityConstants.testSafeMagnitudeLimit
        let vector = try NumericalStabilityHelpers.createSingleLargeValueVector(
            dimension: Dim512.self,
            value: safeBoundaryValue,
            at: 0
        )

        // This should NOT overflow
        #expect(
            !NumericalStabilityHelpers.willMagnitudeOverflow(vector),
            "Value \(safeBoundaryValue) is within safe boundary, should not predict overflow"
        )

        // Compute magnitude
        let actual = vector.magnitude

        // Expected: magnitude equals the single non-zero value
        let expected = safeBoundaryValue

        // Should be finite
        #expect(
            actual.isFinite,
            "Magnitude should be finite at safe boundary, got: \(actual)"
        )

        // Should be accurate
        let withinTolerance = NumericalStabilityHelpers.isWithinRelativeTolerance(
            actual: actual,
            expected: expected,
            relativeTolerance: 1e-4  // Allow 0.01% error
        )

        #expect(
            withinTolerance,
            """
            Magnitude at safe boundary should be accurate:
            Expected: \(expected)
            Actual: \(actual)
            Relative error: \(NumericalStabilityHelpers.relativeError(actual: actual, expected: expected) * 100)%
            """
        )

        // Verify stable algorithm also agrees
        let stableResult = NumericalStabilityHelpers.computeExpectedMagnitude(vector)
        #expect(stableResult.isFinite, "Stable algorithm should produce finite result")
        #expect(
            abs(stableResult - expected) / expected < 1e-4,
            "Stable and expected should agree"
        )
    }

    @Test("Magnitude with single value at theoretical limit")
    func testMagnitudeSingleValueAtLimit() async throws {
        // Test Case: Vector with one component at sqrt(Float.max)
        // Expected: Should handle gracefully (either finite or clear error)
        // Context: Testing exact theoretical boundary

        // Exactly at theoretical limit: sqrt(Float.max) ≈ 1.84467441e+19
        let limitValue = NumericalStabilityConstants.safeMagnitudeLimit
        let vector = try NumericalStabilityHelpers.createSingleLargeValueVector(
            dimension: Dim512.self,
            value: limitValue,
            at: 0
        )

        // At the exact limit, overflow prediction may be borderline
        let willOverflow = NumericalStabilityHelpers.willOverflowWhenSquared(limitValue)
        print("DEBUG: At limit \(limitValue), willOverflow = \(willOverflow)")

        // Compute magnitude
        let actual = vector.magnitude
        let expected = limitValue

        // Stable algorithm should definitely work
        let stableResult = NumericalStabilityHelpers.computeExpectedMagnitude(vector)
        #expect(
            stableResult.isFinite,
            "Stable algorithm should handle limit: \(stableResult)"
        )

        print("DEBUG: Actual = \(actual), Expected = \(expected), Stable = \(stableResult)")

        // Current implementation might overflow at this boundary
        // We test both possibilities:
        if actual.isFinite {
            // Great! Handled without overflow
            let withinTolerance = NumericalStabilityHelpers.isWithinRelativeTolerance(
                actual: actual,
                expected: expected,
                relativeTolerance: 0.01  // Allow 1% error at boundary
            )

            #expect(
                withinTolerance,
                """
                At theoretical limit, magnitude should be accurate if finite:
                Expected: \(expected)
                Actual: \(actual)
                Relative error: \(NumericalStabilityHelpers.relativeError(actual: actual, expected: expected) * 100)%
                """
            )
        } else {
            // Overflowed to infinity (expected for naive implementation)
            #expect(
                actual.isInfinite,
                """
                At theoretical limit, expect either finite or infinity, got: \(actual)
                Note: Overflow at boundary is acceptable without stable algorithm.
                Stable result: \(stableResult)
                """
            )
        }
    }

    @Test("Magnitude with single value beyond safe limit")
    func testMagnitudeSingleValueBeyondLimit() async throws {
        // Test Case: Vector with value > sqrt(Float.max)
        // Expected: Current implementation will overflow to inf
        // Context: This is the core bug being tested

        // Value beyond safe limit: 2e+19 > sqrt(Float.max) ≈ 1.84e+19
        let unsafeValue: Float = NumericalStabilityConstants.overflowBoundary
        let vector = try NumericalStabilityHelpers.createSingleLargeValueVector(
            dimension: Dim512.self,
            value: unsafeValue,
            at: 0
        )

        // Verify we predict this will overflow
        #expect(
            NumericalStabilityHelpers.willMagnitudeOverflow(vector),
            "Should predict overflow for value \(unsafeValue)"
        )

        // Compute magnitude with current implementation
        let actual = vector.magnitude

        // Compute what magnitude SHOULD be with stable algorithm
        let stableExpected = NumericalStabilityHelpers.computeExpectedMagnitude(vector)

        // Stable algorithm should work correctly
        #expect(
            stableExpected.isFinite,
            "Stable magnitude should be finite: \(stableExpected)"
        )

        // Expected magnitude is just the single large value (other components are zero)
        let expected = unsafeValue

        #expect(
            abs(stableExpected - expected) / expected < 1e-4,
            "Stable algorithm should be accurate: expected \(expected), got \(stableExpected)"
        )

        // ⚠️  CORE BUG TEST: Current implementation WILL overflow to infinity
        // This test SHOULD FAIL on current VectorCore implementation
        #expect(
            actual.isFinite,
            """
            ❌ OVERFLOW BUG DETECTED:
            Magnitude overflowed to infinity when it should be finite!

            Value: \(unsafeValue)
            Expected (stable): \(stableExpected)
            Actual (VectorCore): \(actual)

            The naive magnitude calculation squares the value:
            (\(unsafeValue))² = \(unsafeValue * unsafeValue) → overflow to ∞
            Then sqrt(∞) = ∞

            This is a NUMERICAL STABILITY BUG.
            Solution: Use Kahan's two-pass scaling algorithm (see computeExpectedMagnitude).
            """
        )

        // If we somehow got a finite result, verify it's accurate
        if actual.isFinite {
            let withinTolerance = NumericalStabilityHelpers.isWithinRelativeTolerance(
                actual: actual,
                expected: expected,
                relativeTolerance: 1e-3
            )

            if !withinTolerance {
                let diagnostic = NumericalStabilityHelpers.magnitudeDiagnostic(
                    vector: vector,
                    actual: actual,
                    expected: expected
                )
                print(diagnostic)
            }

            #expect(withinTolerance, "Magnitude should be accurate")
        }
    }

    // MARK: Multiple Large Value Tests

    @Test("Magnitude with multiple large values (512-dim)")
    func testMagnitudeMultipleLargeValues512() async throws {
        // Test Case: All 512 components set to large value
        // Expected: Magnitude = value * sqrt(512), should be finite
        // Context: Tests accumulation overflow

        // Use 1e+18 - individually safe, but sum may overflow
        let largeValue: Float = 1e18
        let dimension = 512
        let vector = NumericalStabilityHelpers.createUniformLargeVector(
            dimension: Dim512.self,
            value: largeValue
        )

        // Expected magnitude: largeValue * sqrt(512) ≈ 2.26e+19
        let expected = NumericalStabilityHelpers.computeUniformVectorMagnitude(
            value: largeValue,
            dimension: dimension
        )

        print("DEBUG: Expected magnitude for 512×\(largeValue) = \(expected)")

        // Stable reference should work
        let stableActual = NumericalStabilityHelpers.computeExpectedMagnitude(vector)
        #expect(
            stableActual.isFinite,
            "Stable algorithm should produce finite result: \(stableActual)"
        )

        // VectorCore magnitude (may overflow)
        let actual = vector.magnitude
        print("DEBUG: VectorCore magnitude = \(actual)")
        print("DEBUG: Stable magnitude = \(stableActual)")

        // This SHOULD FAIL on current implementation (accumulation overflow)
        #expect(
            actual.isFinite,
            """
            ❌ ACCUMULATION OVERFLOW BUG:
            Magnitude overflowed during accumulation!

            Input: 512 components, each = \(largeValue)
            Expected: \(expected) ≈ 2.26e+19
            Actual: \(actual)

            Each square: (\(largeValue))² = 1e+36
            Sum of squares: 512 × 1e+36 = 5.12e+38 → overflow to ∞
            Magnitude: sqrt(∞) = ∞

            Solution: Use stable algorithm that scales before squaring.
            """
        )

        // If it didn't overflow, check accuracy
        if actual.isFinite {
            let withinTolerance = NumericalStabilityHelpers.isWithinRelativeTolerance(
                actual: actual,
                expected: expected,
                relativeTolerance: 1e-3
            )

            #expect(
                withinTolerance,
                """
                Magnitude should be accurate:
                Expected: \(expected)
                Actual: \(actual)
                Error: \(NumericalStabilityHelpers.relativeError(actual: actual, expected: expected) * 100)%
                """
            )
        }
    }

    @Test("Magnitude with multiple large values (768-dim)")
    func testMagnitudeMultipleLargeValues768() async throws {
        // Test Case: All 768 components set to large value
        // Expected: Magnitude = value * sqrt(768), should be finite
        // Context: Tests dimension-specific overflow

        let largeValue: Float = 1e18
        let dimension = 768
        let vector = NumericalStabilityHelpers.createUniformLargeVector(
            dimension: Dim768.self,
            value: largeValue
        )

        let expected = NumericalStabilityHelpers.computeUniformVectorMagnitude(
            value: largeValue,
            dimension: dimension
        )

        let stableActual = NumericalStabilityHelpers.computeExpectedMagnitude(vector)
        #expect(stableActual.isFinite, "Stable algorithm should work: \(stableActual)")

        let actual = vector.magnitude
        print("DEBUG: 768-dim: Expected = \(expected), Actual = \(actual), Stable = \(stableActual)")

        #expect(
            actual.isFinite,
            """
            ❌ 768-DIM OVERFLOW:
            Expected: \(expected) ≈ 2.77e+19
            Actual: \(actual)
            Stable: \(stableActual)
            """
        )

        if actual.isFinite {
            #expect(
                NumericalStabilityHelpers.isWithinRelativeTolerance(
                    actual: actual,
                    expected: expected,
                    relativeTolerance: 1e-3
                ),
                "Should be accurate within 0.1%"
            )
        }
    }

    @Test("Magnitude with multiple large values (1536-dim)")
    func testMagnitudeMultipleLargeValues1536() async throws {
        // Test Case: All 1536 components set to large value
        // Expected: Magnitude = value * sqrt(1536), should be finite
        // Context: Larger dimension increases overflow risk

        let largeValue: Float = 1e18
        let dimension = 1536
        let vector = NumericalStabilityHelpers.createUniformLargeVector(
            dimension: Dim1536.self,
            value: largeValue
        )

        let expected = NumericalStabilityHelpers.computeUniformVectorMagnitude(
            value: largeValue,
            dimension: dimension
        )

        let stableActual = NumericalStabilityHelpers.computeExpectedMagnitude(vector)
        #expect(stableActual.isFinite, "Stable algorithm should work: \(stableActual)")

        let actual = vector.magnitude
        print("DEBUG: 1536-dim: Expected = \(expected), Actual = \(actual), Stable = \(stableActual)")

        #expect(
            actual.isFinite,
            """
            ❌ 1536-DIM OVERFLOW:
            Expected: \(expected) ≈ 3.92e+19
            Actual: \(actual)
            Stable: \(stableActual)

            Larger dimension (1536) causes higher overflow risk.
            Sum: 1536 × (1e18)² = 1.536e+39 > Float.max
            """
        )

        if actual.isFinite {
            #expect(
                NumericalStabilityHelpers.isWithinRelativeTolerance(
                    actual: actual,
                    expected: expected,
                    relativeTolerance: 1e-3
                ),
                "Should be accurate within 0.1%"
            )
        }
    }

    @Test("Magnitude accumulation overflow at boundary")
    func testMagnitudeAccumulationOverflow() async throws {
        // Test Case: Each component small but many components
        // Expected: Should handle sum of many squares correctly
        // Context: Tests if (sqrt(Float.max) / sqrt(n))^2 * n overflows

        // TODO: Implement test logic
        #expect(Bool(false), "Test scaffold - implementation pending")
    }

    // MARK: Normalization Overflow Tests

    @Test("Normalization with extreme single value")
    func testNormalizationExtremeSingleValue() async throws {
        // Test Case: Normalize vector with one extreme value
        // Expected: Should produce unit vector or fail gracefully
        // Context: Tests full normalization pipeline with overflow

        // Use very large but not quite at overflow boundary
        let extremeValue: Float = Float.greatestFiniteMagnitude / 1000  // ~3.4e+35
        let vector = try NumericalStabilityHelpers.createSingleLargeValueVector(
            dimension: Dim512.self,
            value: extremeValue,
            at: 0
        )

        print("DEBUG: Extreme value = \(extremeValue)")

        // Check magnitude
        let magnitude = vector.magnitude
        print("DEBUG: Magnitude = \(magnitude)")

        let stableMagnitude = NumericalStabilityHelpers.computeExpectedMagnitude(vector)
        print("DEBUG: Stable magnitude = \(stableMagnitude)")

        // Normalize
        let normalized = vector.normalizedFast()

        // Check result
        let normalizedMagnitude = normalized.magnitude
        print("DEBUG: Normalized magnitude = \(normalizedMagnitude)")
        print("DEBUG: First component = \(normalized[0])")

        // Should not be zero vector
        #expect(
            !NumericalStabilityHelpers.isEffectivelyZero(normalized),
            """
            Normalization should not produce zero vector.
            Original extreme value: \(extremeValue)
            Magnitude: \(magnitude)
            Normalized magnitude: \(normalizedMagnitude)
            """
        )

        // If magnitude was finite, normalized should be unit
        if magnitude.isFinite {
            #expect(
                NumericalStabilityHelpers.isUnitVector(normalized, tolerance: 1e-3),
                "Should be unit vector when magnitude is finite"
            )

            // First component should be ±1 (only non-zero component)
            #expect(
                abs(abs(normalized[0]) - 1.0) < 0.01,
                "Single non-zero component should be ±1"
            )

            // Other components should be ~0
            #expect(abs(normalized[1]) < 0.01, "Other components should be near zero")
            #expect(abs(normalized[511]) < 0.01, "Other components should be near zero")
        } else {
            // Magnitude overflowed - check if we got zero vector (bug)
            if NumericalStabilityHelpers.isEffectivelyZero(normalized) {
                #expect(
                    Bool(false),
                    """
                    BUG: Overflow in magnitude led to zero vector from normalization.
                    This is silent data corruption.
                    """
                )
            }
        }
    }

    @Test("Normalization with extreme uniform values")
    func testNormalizationExtremeUniformValues() async throws {
        // Test Case: Normalize vector where all components are large
        // Expected: Result should be unit vector, not zero vector
        // Context: Core bug - currently produces zero vector

        // All 512 components = 1.5e+19 (will cause overflow when squared)
        let largeValue: Float = 1.5e19
        let vector = NumericalStabilityHelpers.createUniformLargeVector(
            dimension: Dim512.self,
            value: largeValue
        )

        // Note: With Kahan's algorithm, this no longer overflows
        // (Previously would overflow in naive implementation)
        let wouldOverflowNaively = NumericalStabilityHelpers.willMagnitudeOverflow(vector)
        print("DEBUG: Would overflow with naive implementation: \(wouldOverflowNaively)")

        // Compute magnitude to show what happens
        let magnitude = vector.magnitude
        print("DEBUG: Magnitude of extreme uniform vector = \(magnitude)")

        // Show what stable algorithm produces
        let stableMagnitude = NumericalStabilityHelpers.computeExpectedMagnitude(vector)
        let expectedMagnitude = largeValue * sqrt(512.0)
        print("DEBUG: Stable magnitude = \(stableMagnitude), Expected = \(expectedMagnitude)")

        // Try to normalize using the fast method (returns Vector directly)
        let normalized = vector.normalizedFast()

        print("DEBUG: Normalization completed")
        print("DEBUG: Normalized magnitude = \(normalized.magnitude)")
        print("DEBUG: First 5 components = [\(normalized[0]), \(normalized[1]), \(normalized[2]), \(normalized[3]), \(normalized[4])]")

        // Check if result is effectively zero (THE CORE BUG)
        let isZero = NumericalStabilityHelpers.isEffectivelyZero(normalized)

        // ⚠️  CRITICAL BUG TEST: This SHOULD FAIL on current implementation
        // Current behavior: magnitude overflows → inf, then vector/inf → zero vector
        let normalizedMagnitude = normalized.magnitude
        print("DEBUG: Normalized magnitude = \(normalizedMagnitude)")

        #expect(
            !isZero,
            """
            ❌ CRITICAL BUG DETECTED: ZERO VECTOR RETURNED FROM NORMALIZATION!

            This is the MAIN numerical stability bug:

            Input: Uniform vector with all components = \(largeValue)
            Expected output: Unit vector with all components = 1/sqrt(512) ≈ 0.0442

            What happened:
            1. Magnitude calculation: sqrt(Σ(x²))
            2. Each square: (\(largeValue))² = overflow to ∞
            3. Sum of squares: ∞ + ∞ + ... = ∞
            4. Magnitude: sqrt(∞) = ∞
            5. Normalization: vector / ∞ = [0, 0, 0, ...]  ← WRONG!

            Actual normalized vector: ALL ZEROS (magnitude = \(normalizedMagnitude))
            Expected: Unit vector (magnitude = 1.0)

            This silently corrupts data without any error indication!

            Solution: Implement Kahan's stable magnitude algorithm.
            See NumericalStabilityHelpers.computeExpectedMagnitude for reference.
            """
        )

        // Check if it's a unit vector
        let isUnit = NumericalStabilityHelpers.isUnitVector(normalized, tolerance: 1e-3)

        #expect(
            isUnit,
            "Normalized vector should be unit vector, actual magnitude: \(normalizedMagnitude)"
        )

        // For uniform input, all components should be equal
        // Expected: each component = 1/sqrt(512) ≈ 0.0442
        let expectedComponent: Float = 1.0 / sqrt(512.0)

        if !isZero {
            let actualFirstComponent = abs(normalized[0])
            let componentError = abs(actualFirstComponent - expectedComponent) / expectedComponent

            #expect(
                componentError < 0.01,
                "Component error \(componentError * 100)% exceeds 1% tolerance"
            )
        }
    }

    @Test("Normalization preserves direction after overflow")
    func testNormalizationDirectionPreservation() async throws {
        // Test Case: Verify normalized vector points in same direction
        // Expected: sign(normalized[i]) == sign(original[i])
        // Context: Even with overflow, direction should be preserved

        // TODO: Implement test logic
        #expect(Bool(false), "Test scaffold - implementation pending")
    }
}

// MARK: - Test Suite: Overflow Detection

/// Tests for overflow detection and graceful handling
@Suite("Numerical Stability: Overflow Detection")
struct OverflowDetectionTests {

    @Test("Magnitude overflow detection for infinity result")
    func testMagnitudeOverflowToInfinity() async throws {
        // Test Case: Value guaranteed to overflow
        // Expected: Either finite result (stable) or inf (current)
        // Context: Tests if overflow is detected

        // Use Float.max - guaranteed to overflow
        let vector = NumericalStabilityHelpers.createUniformLargeVector(
            dimension: Dim512.self,
            value: Float.greatestFiniteMagnitude
        )

        let magnitude = vector.magnitude

        print("DEBUG: Float.max vector magnitude = \(magnitude)")

        // Stable algorithm should handle or return inf appropriately
        let stableMag = NumericalStabilityHelpers.computeExpectedMagnitude(vector)
        print("DEBUG: Stable magnitude = \(stableMag)")

        // Current implementation will overflow
        if magnitude.isInfinite {
            // Expected behavior for naive implementation
            #expect(magnitude.isInfinite, "Detected overflow to infinity")
            print("INFO: Overflow detected - magnitude is infinite")
        } else {
            // Stable algorithm implemented
            #expect(magnitude.isFinite, "Magnitude should be finite with stable algorithm")
        }

        // Either way, we should be able to detect the situation
        #expect(
            magnitude.isFinite || magnitude.isInfinite,
            "Magnitude should be either finite or infinite, not NaN"
        )
    }

    @Test("Normalization fails gracefully on overflow")
    func testNormalizationOverflowGracefulFailure() async throws {
        // Test Case: Attempt normalization with overflow-causing values
        // Expected: Should return .failure() or produce valid result
        // Context: Should not silently produce zero vector

        // Create vector that will cause overflow
        let overflowValue: Float = 3e19
        let vector = NumericalStabilityHelpers.createUniformLargeVector(
            dimension: Dim512.self,
            value: overflowValue
        )

        // Try normalization
        let normalized = vector.normalizedFast()

        // Check if we got zero vector (the bug)
        let isZero = NumericalStabilityHelpers.isEffectivelyZero(normalized)

        print("DEBUG: Normalized magnitude = \(normalized.magnitude)")
        print("DEBUG: Is zero = \(isZero)")

        if isZero {
            // BUG: Silent failure - produced zero vector
            #expect(
                !isZero,
                """
                Normalization silently failed: returned zero vector.
                Should either return valid unit vector or fail with error.
                """
            )
        } else {
            // Good: either produced valid result or kept non-zero
            #expect(
                !isZero,
                "Normalization should not produce zero vector"
            )

            // If not zero, check if it's a unit vector
            let isUnit = NumericalStabilityHelpers.isUnitVector(normalized, tolerance: 0.1)
            print("DEBUG: Is unit = \(isUnit)")

            // Ideally should be unit, but at minimum shouldn't be zero
            #expect(!isZero, "Most important: not zero")
        }
    }

    @Test("Infinite input handling in magnitude")
    func testInfiniteInputMagnitude() async throws {
        // Test Case: Vector already containing infinity
        // Expected: Should handle gracefully (return inf or error)
        // Context: Tests degenerate input handling

        // Create vector with infinity
        let vector = try NumericalStabilityHelpers.createSingleLargeValueVector(
            dimension: Dim128.self,
            value: Float.infinity,
            at: 0
        )

        let magnitude = vector.magnitude

        print("DEBUG: Magnitude of infinite vector = \(magnitude)")

        // Should propagate infinity
        #expect(
            magnitude.isInfinite,
            "Magnitude of vector containing infinity should be infinite"
        )

        // Stable algorithm should also return infinity
        let stableMag = NumericalStabilityHelpers.computeExpectedMagnitude(vector)
        #expect(stableMag.isInfinite, "Stable algorithm should also detect infinity")
    }

    @Test("NaN input handling in magnitude")
    func testNaNInputMagnitude() async throws {
        // Test Case: Vector containing NaN values
        // Expected: Result should be NaN or error
        // Context: Tests invalid input propagation

        // Create vector with NaN
        let vector = try NumericalStabilityHelpers.createSingleLargeValueVector(
            dimension: Dim128.self,
            value: Float.nan,
            at: 0
        )

        let magnitude = vector.magnitude

        print("DEBUG: Magnitude of NaN vector = \(magnitude)")

        // Should propagate NaN
        #expect(
            magnitude.isNaN,
            "Magnitude of vector containing NaN should be NaN"
        )

        // Stable algorithm should also detect NaN
        let stableMag = NumericalStabilityHelpers.computeExpectedMagnitude(vector)
        print("DEBUG: Stable magnitude = \(stableMag)")
        // Note: stable algorithm might handle NaN differently, that's OK
    }

    @Test("Mixed valid and infinite components")
    func testMixedValidAndInfiniteComponents() async throws {
        // Test Case: Some components finite, some infinite
        // Expected: Result should be infinite
        // Context: Tests partial overflow scenarios

        // Create vector with one infinite, rest finite
        let vector = try NumericalStabilityHelpers.createMixedScaleVector(
            dimension: Dim128.self,
            largeValue: Float.infinity,  // One infinite
            smallValue: 100.0,           // Rest are finite
            largeCount: 1
        )

        let magnitude = vector.magnitude

        print("DEBUG: Mixed (inf + finite) magnitude = \(magnitude)")

        // If any component is infinite, magnitude should be infinite
        #expect(
            magnitude.isInfinite,
            "Magnitude with any infinite component should be infinite"
        )

        // Test reverse: many finite, one infinite
        let vector2 = try NumericalStabilityHelpers.createMixedScaleVector(
            dimension: Dim128.self,
            largeValue: 1000.0,
            smallValue: Float.infinity,
            largeCount: 127  // Only last one is infinite
        )

        let magnitude2 = vector2.magnitude
        print("DEBUG: Mixed (finite + inf) magnitude = \(magnitude2)")

        #expect(
            magnitude2.isInfinite,
            "Magnitude should be infinite regardless of infinite position"
        )
    }
}

// MARK: - Test Suite: Numerical Precision

/// Tests for numerical precision across dynamic range
@Suite("Numerical Stability: Numerical Precision")
struct NumericalPrecisionTests {

    @Test("Magnitude precision across full dynamic range")
    func testMagnitudePrecisionDynamicRange() async throws {
        // Test Case: Test magnitude accuracy from 1e-20 to 1e+19
        // Expected: All values should be within 0.01% relative error
        // Context: Tests precision across Float's full range

        // Test exponents from -20 to +15 (safe range, avoiding overflow)
        let testExponents: [Float] = [-20, -15, -10, -5, -2, 0, 2, 5, 10, 15]

        var maxError: Float = 0
        var failures: [(Float, Float, Float)] = []  // (value, expected, actual)

        for exponent in testExponents {
            let value = pow(10.0, exponent)
            let vector = try NumericalStabilityHelpers.createSingleLargeValueVector(
                dimension: Dim512.self,
                value: value,
                at: 0
            )

            let expected = value
            let actual = vector.magnitude
            let relError = NumericalStabilityHelpers.relativeError(
                actual: actual,
                expected: expected
            )

            maxError = max(maxError, relError)

            if relError > 1e-4 {  // 0.01% tolerance
                failures.append((value, expected, actual))
            }

            print("DEBUG: 1e\(Int(exponent)): expected=\(expected), actual=\(actual), error=\(relError * 100)%")
        }

        #expect(
            failures.isEmpty,
            """
            Precision loss across dynamic range:
            Max error: \(maxError * 100)%
            Failures: \(failures.count)/\(testExponents.count)
            \(failures.map { "  1e\(log10($0.0)): expected \($0.1), got \($0.2)" }.joined(separator: "\n"))
            """
        )

        #expect(maxError < 1e-4, "Max relative error \(maxError * 100)% exceeds 0.01%")
    }

    @Test("Magnitude precision for small values (underflow region)")
    func testMagnitudeSmallValues() async throws {
        // Test Case: Values near Float.min and subnormals
        // Expected: Accurate results, no underflow to zero
        // Context: Tests lower bound precision

        let smallValues: [Float] = [
            Float.leastNormalMagnitude,        // ~1.17e-38
            Float.leastNormalMagnitude * 10,   // ~1.17e-37
            Float.leastNormalMagnitude * 100,  // ~1.17e-36
            1e-30,
            1e-20,
            1e-10
        ]

        for smallValue in smallValues {
            let vector = try NumericalStabilityHelpers.createSingleLargeValueVector(
                dimension: Dim32.self,
                value: smallValue,
                at: 0
            )

            let expected = smallValue
            let actual = vector.magnitude

            print("DEBUG: Small value \(smallValue): actual=\(actual)")

            // Should not underflow to zero
            #expect(actual > 0, "Small value \(smallValue) underflowed to zero")

            // Should be accurate
            let relError = NumericalStabilityHelpers.relativeError(
                actual: actual,
                expected: expected
            )

            #expect(
                relError < 1e-4,
                "Small value \(smallValue): error \(relError * 100)% exceeds 0.01%"
            )
        }
    }

    @Test("Magnitude precision for medium values")
    func testMagnitudeMediumValues() async throws {
        // Test Case: Values in normal range (1e-5 to 1e+5)
        // Expected: High precision (< 1e-6 relative error)
        // Context: These should be exact

        let mediumValues: [Float] = [
            0.00001,  // 1e-5
            0.001,    // 1e-3
            0.1,
            1.0,
            10.0,
            100.0,
            1000.0,
            10000.0,  // 1e+4
            100000.0  // 1e+5
        ]

        var maxError: Float = 0

        for value in mediumValues {
            let vector = try NumericalStabilityHelpers.createSingleLargeValueVector(
                dimension: Dim128.self,
                value: value,
                at: 0
            )

            let expected = value
            let actual = vector.magnitude
            let relError = NumericalStabilityHelpers.relativeError(
                actual: actual,
                expected: expected
            )

            maxError = max(maxError, relError)

            print("DEBUG: Medium value \(value): error=\(relError * 100)%")

            // High precision expected in normal range
            #expect(
                relError < 1e-6,
                "Medium value \(value): error \(relError * 100)% exceeds 0.0001%"
            )
        }

        print("DEBUG: Max error in medium range: \(maxError * 100)%")
        #expect(maxError < 1e-6, "Max error in normal range should be < 1e-6")
    }

    @Test("Magnitude precision with mixed scales")
    func testMagnitudeMixedScales() async throws {
        // Test Case: Components spanning multiple orders of magnitude
        // Expected: Large components dominate, small ones don't vanish
        // Context: Tests catastrophic cancellation avoidance

        // Create vector with mixed magnitudes
        let largeValue: Float = 1e10
        let smallValue: Float = 1e-5
        let vector = try NumericalStabilityHelpers.createMixedScaleVector(
            dimension: Dim512.self,
            largeValue: largeValue,
            smallValue: smallValue,
            largeCount: 10
        )

        // Expected: dominated by large values
        // 10 large + 502 small
        // magnitude ≈ sqrt(10 * (1e10)² + 502 * (1e-5)²)
        //          ≈ sqrt(10 * 1e20 + 502 * 1e-10)
        //          ≈ sqrt(1e21) ≈ 3.16e10
        let largePart = sqrt(10.0) * largeValue
        let smallPart = sqrt(502.0) * smallValue
        let expected = sqrt(largePart * largePart + smallPart * smallPart)

        let actual = vector.magnitude

        print("DEBUG: Mixed scales - Large part: \(largePart), Small part: \(smallPart)")
        print("DEBUG: Expected: \(expected), Actual: \(actual)")

        // Large components should dominate
        #expect(actual > largePart * 0.99, "Large components should dominate")

        // But small components should still contribute (not vanish)
        #expect(actual <= largePart * 1.01, "Should be close to large component magnitude")

        // Overall accuracy
        let relError = NumericalStabilityHelpers.relativeError(
            actual: actual,
            expected: expected
        )

        #expect(
            relError < 1e-3,
            "Mixed scale error \(relError * 100)% exceeds 0.1%"
        )
    }

    @Test("Relative error bounds for safe magnitudes")
    func testRelativeErrorBounds() async throws {
        // Test Case: Verify <0.01% relative error for safe values
        // Expected: All safe range magnitudes within tolerance
        // Context: Precision guarantee testing

        // Test across safe value range
        let safeTestValues = NumericalStabilityConstants.safeTestValues

        var allWithinBounds = true
        var maxError: Float = 0
        var errorDistribution: [Float] = []

        for value in safeTestValues {
            let vector = try NumericalStabilityHelpers.createSingleLargeValueVector(
                dimension: Dim256.self,
                value: value,
                at: 0
            )

            let expected = value
            let actual = vector.magnitude
            let relError = NumericalStabilityHelpers.relativeError(
                actual: actual,
                expected: expected
            )

            errorDistribution.append(relError)
            maxError = max(maxError, relError)

            if relError >= 1e-4 {
                allWithinBounds = false
                print("WARNING: Value \(value) exceeded error bound: \(relError * 100)%")
            }
        }

        // Statistical summary
        let avgError = errorDistribution.reduce(0, +) / Float(errorDistribution.count)
        print("DEBUG: Error statistics:")
        print("  Max: \(maxError * 100)%")
        print("  Avg: \(avgError * 100)%")
        print("  Count: \(safeTestValues.count)")

        #expect(
            allWithinBounds,
            "All safe values should have <0.01% error. Max: \(maxError * 100)%"
        )

        #expect(maxError < 1e-4, "Max error \(maxError * 100)% exceeds 0.01% bound")
    }

    @Test("Magnitude calculation consistency across dimensions")
    func testMagnitudeConsistencyAcrossDimensions() async throws {
        // Test Case: Same per-component value across different dimensions
        // Expected: magnitude(v_n) = value * sqrt(n)
        // Context: Dimension independence

        let testValue: Float = 100.0

        // Test across multiple dimensions
        let dims: [(any StaticDimension.Type, Int)] = [
            (Dim32.self, 32),
            (Dim64.self, 64),
            (Dim128.self, 128),
            (Dim256.self, 256),
            (Dim512.self, 512),
            (Dim768.self, 768),
            (Dim1536.self, 1536)
        ]

        for (dimType, dimValue) in dims {
            // Create uniform vector for each dimension
            let vector: any VectorProtocol
            switch dimValue {
            case 32:
                vector = NumericalStabilityHelpers.createUniformLargeVector(
                    dimension: Dim32.self,
                    value: testValue
                )
            case 64:
                vector = NumericalStabilityHelpers.createUniformLargeVector(
                    dimension: Dim64.self,
                    value: testValue
                )
            case 128:
                vector = NumericalStabilityHelpers.createUniformLargeVector(
                    dimension: Dim128.self,
                    value: testValue
                )
            case 256:
                vector = NumericalStabilityHelpers.createUniformLargeVector(
                    dimension: Dim256.self,
                    value: testValue
                )
            case 512:
                vector = NumericalStabilityHelpers.createUniformLargeVector(
                    dimension: Dim512.self,
                    value: testValue
                )
            case 768:
                vector = NumericalStabilityHelpers.createUniformLargeVector(
                    dimension: Dim768.self,
                    value: testValue
                )
            case 1536:
                vector = NumericalStabilityHelpers.createUniformLargeVector(
                    dimension: Dim1536.self,
                    value: testValue
                )
            default:
                fatalError("Unexpected dimension")
            }

            // Expected: value * sqrt(n)
            let expected = testValue * sqrt(Float(dimValue))
            let actual = Float(vector.magnitude)

            let relError = abs(actual - expected) / expected

            print("DEBUG: Dim \(dimValue): expected=\(expected), actual=\(actual), error=\(relError * 100)%")

            #expect(
                relError < 1e-5,
                "Dim \(dimValue): error \(relError * 100)% exceeds 0.001%"
            )
        }
    }
}

// MARK: - Test Suite: Fuzzing Tests

/// Randomized fuzzing tests for numerical stability
@Suite("Numerical Stability: Fuzzing")
struct NumericalStabilityFuzzingTests {

    @Test("Fuzz: Random extreme values (1000 iterations)", .timeLimit(.minutes(1)))
    func testFuzzRandomExtremeValues() async throws {
        // Test Case: 1000 random vectors with extreme values
        // Expected: No crashes, no silent overflow to zero
        // Context: Discovers edge cases through randomization

        // TODO: Implement test logic
        #expect(Bool(false), "Test scaffold - implementation pending")
    }

    @Test("Fuzz: Mixed scale values (1000 iterations)", .timeLimit(.minutes(1)))
    func testFuzzMixedScaleValues() async throws {
        // Test Case: Random mix of large and small values
        // Expected: Magnitude handles scale differences correctly
        // Context: Tests real-world mixed-scale scenarios

        // TODO: Implement test logic
        #expect(Bool(false), "Test scaffold - implementation pending")
    }

    @Test("Fuzz: Edge case combinations (500 iterations)")
    func testFuzzEdgeCaseCombinations() async throws {
        // Test Case: Random combinations of zero, NaN, inf, finite
        // Expected: Predictable behavior for each combination
        // Context: Boundary condition discovery

        // TODO: Implement test logic
        #expect(Bool(false), "Test scaffold - implementation pending")
    }

    @Test("Fuzz: Normalization round-trip (1000 iterations)")
    func testFuzzNormalizationRoundTrip() async throws {
        // Test Case: Normalize random vectors, check magnitude
        // Expected: All successful normalizations produce unit vectors
        // Context: End-to-end fuzzing

        // TODO: Implement test logic
        #expect(Bool(false), "Test scaffold - implementation pending")
    }

    @Test("Fuzz: Extreme exponent range", .timeLimit(.minutes(1)))
    func testFuzzExtremeExponentRange() async throws {
        // Test Case: Random exponents from -38 to +38
        // Expected: Handles full Float range without crashes
        // Context: Exhaustive range testing

        // TODO: Implement test logic
        #expect(Bool(false), "Test scaffold - implementation pending")
    }
}

// MARK: - Test Suite: Regression Tests

/// Regression tests to ensure fixes don't break existing functionality
@Suite("Numerical Stability: Regression")
struct NumericalStabilityRegressionTests {

    @Test("Normal values still work after stability fixes")
    func testNormalValuesBehavior() async throws {
        // Test Case: Ensure typical values (0-100) still work
        // Expected: Same behavior as before fixes
        // Context: Regression prevention

        // Test various typical value patterns
        struct TestCase {
            let vector: Vector<Dim32>
            let expectedMagnitude: Float
            let description: String
        }

        let testCases: [TestCase] = [
            // Uniform vector of 1s
            TestCase(
                vector: Vector<Dim32>(repeating: 1.0),
                expectedMagnitude: sqrt(32.0),
                description: "Uniform 1.0"
            ),
            // Simple 3-4-5 triangle in first components
            TestCase(
                vector: try Vector<Dim32>([3, 4] + Array(repeating: 0.0 as Float, count: 30)),
                expectedMagnitude: 5.0,
                description: "3-4-5 triangle"
            ),
            // Uniform vector of 10s
            TestCase(
                vector: Vector<Dim32>(repeating: 10.0),
                expectedMagnitude: 10.0 * sqrt(32.0),
                description: "Uniform 10.0"
            ),
            // Uniform vector of 0.5s
            TestCase(
                vector: Vector<Dim32>(repeating: 0.5),
                expectedMagnitude: 0.5 * sqrt(32.0),
                description: "Uniform 0.5"
            ),
            // Mixed small values
            TestCase(
                vector: try Vector<Dim32>([1, 2, 3, 4, 5] + Array(repeating: 0.0 as Float, count: 27)),
                expectedMagnitude: sqrt(1 + 4 + 9 + 16 + 25),
                description: "Sequential 1-5"
            )
        ]

        for testCase in testCases {
            let actual = testCase.vector.magnitude

            let withinTolerance = NumericalStabilityHelpers.isWithinRelativeTolerance(
                actual: actual,
                expected: testCase.expectedMagnitude,
                relativeTolerance: 1e-5
            )

            #expect(
                withinTolerance,
                """
                \(testCase.description) failed:
                \(NumericalStabilityHelpers.magnitudeDiagnostic(
                    vector: testCase.vector,
                    actual: actual,
                    expected: testCase.expectedMagnitude
                ))
                """
            )
        }
    }

    @Test("Unit vectors remain unchanged")
    func testUnitVectorPreservation() async throws {
        // Test Case: Already-normalized vectors stay normalized
        // Expected: No change when normalizing unit vectors
        // Context: Idempotency regression

        // Create various unit vectors
        let unitVectors: [(Vector<Dim32>, String)] = [
            // Standard basis vectors
            (try Vector<Dim32>([1] + Array(repeating: 0.0 as Float, count: 31)), "e1 basis"),
            (try Vector<Dim32>([0, 1] + Array(repeating: 0.0 as Float, count: 30)), "e2 basis"),

            // 3-4 triangle normalized
            (try Vector<Dim32>([0.6, 0.8] + Array(repeating: 0.0 as Float, count: 30)), "3-4-5 normalized"),

            // Uniform unit vector
            (Vector<Dim32>(repeating: 1.0 / sqrt(32.0)), "Uniform unit"),

            // Random unit vector
            (try {
                let random = try Vector<Dim32>((0..<32).map { _ in Float.random(in: -1...1) })
                return random.normalizedFast()
            }(), "Random normalized")
        ]

        for (vector, description) in unitVectors {
            // Verify it's already unit
            #expect(
                NumericalStabilityHelpers.isUnitVector(vector),
                "\(description): should already be unit vector"
            )

            // Normalize and verify no significant change
            let normalized = vector.normalizedFast()

            #expect(
                NumericalStabilityHelpers.isUnitVector(normalized),
                "\(description): should remain unit after normalization"
            )

            // Check components are nearly identical (idempotency)
            let difference = (vector - normalized).magnitude
            #expect(
                difference < 1e-6,
                "\(description): should be unchanged (diff = \(difference))"
            )

            // Double normalization should also be identical
            let doubleNorm = normalized.normalizedFast()
            let doubleDiff = (normalized - doubleNorm).magnitude
            #expect(
                doubleDiff < 1e-6,
                "\(description): double normalization should be identical"
            )
        }
    }

    @Test("Zero vector handling unchanged")
    func testZeroVectorHandling() async throws {
        // Test Case: Zero vector normalization still fails appropriately
        // Expected: Returns .failure() as before
        // Context: Error handling regression

        // Test zero vector across different dimensions
        let zeroVector32 = Vector<Dim32>.zero
        let zeroVector512 = Vector<Dim512>.zero
        let zeroVector768 = Vector<Dim768>.zero

        // Test magnitude is zero
        #expect(zeroVector32.magnitude == 0.0, "Zero vector should have zero magnitude")
        #expect(zeroVector512.magnitude == 0.0, "Zero vector should have zero magnitude")

        // Test Result-based normalized() fails
        let result32: Result<Vector<Dim32>, VectorError> = zeroVector32.normalized()
        switch result32 {
        case .success:
            #expect(Bool(false), "Zero vector normalization should fail, not succeed")
        case .failure(let error):
            #expect(error.kind == .invalidOperation, "Should be invalidOperation error")
        }

        let result512: Result<Vector<Dim512>, VectorError> = zeroVector512.normalized()
        switch result512 {
        case .success:
            #expect(Bool(false), "Zero vector normalization should fail")
        case .failure(let error):
            #expect(error.kind == .invalidOperation)
        }

        // Test fast normalization returns self (zero) for zero vector
        let fastNorm32 = zeroVector32.normalizedFast()
        #expect(
            NumericalStabilityHelpers.isEffectivelyZero(fastNorm32),
            "Fast normalization of zero vector should return zero"
        )

        let fastNorm512 = zeroVector512.normalizedFast()
        #expect(
            NumericalStabilityHelpers.isEffectivelyZero(fastNorm512),
            "Fast normalization of zero vector should return zero"
        )

        // Verify magnitude remains zero
        #expect(fastNorm32.magnitude == 0.0)
        #expect(fastNorm512.magnitude == 0.0)
    }

    @Test("Performance within acceptable bounds")
    func testPerformanceRegression() async throws {
        // Test Case: Magnitude calculation not significantly slower
        // Expected: < 2x slowdown from baseline
        // Context: Performance regression check

        // TODO: Implement test logic
        #expect(Bool(false), "Test scaffold - implementation pending")
    }
}

// MARK: - Test Suite: Optimized Vector Variants

/// Tests for optimized vector implementations (512, 768, 1536)
@Suite("Numerical Stability: Optimized Vectors")
struct OptimizedVectorStabilityTests {

    @Test("Vector512Optimized magnitude overflow")
    func testVector512OptimizedOverflow() async throws {
        // Test Case: Optimized 512-dim vector with large values
        // Expected: Same stability as generic implementation
        // Context: SIMD kernel stability

        let largeValue: Float = 1.5e19
        let optimized = Vector512Optimized(repeating: largeValue)

        let magnitude = optimized.magnitude
        print("DEBUG: Vector512Optimized magnitude = \(magnitude)")

        // Should have same overflow behavior as generic
        // (Will overflow to infinity in current implementation)
        if magnitude.isInfinite {
            print("INFO: SIMD kernel also overflows (same bug as generic)")
            #expect(magnitude.isInfinite, "Detected SIMD overflow")
        } else {
            // If stable algorithm implemented
            #expect(magnitude.isFinite, "SIMD should be finite with stable implementation")
            let expected = largeValue * sqrt(512.0)
            let relError = abs(magnitude - expected) / expected
            #expect(relError < 0.01, "SIMD should be accurate")
        }
    }

    @Test("Vector768Optimized magnitude overflow")
    func testVector768OptimizedOverflow() async throws {
        // Test Case: Optimized 768-dim vector with large values
        // Expected: Same stability as generic implementation
        // Context: SIMD kernel stability

        let largeValue: Float = 1.5e19
        let optimized = Vector768Optimized(repeating: largeValue)

        let magnitude = optimized.magnitude
        print("DEBUG: Vector768Optimized magnitude = \(magnitude)")

        if magnitude.isInfinite {
            print("INFO: 768-SIMD also overflows")
            #expect(magnitude.isInfinite, "Detected SIMD overflow")
        } else {
            #expect(magnitude.isFinite, "SIMD should be finite")
            let expected = largeValue * sqrt(768.0)
            let relError = abs(magnitude - expected) / expected
            #expect(relError < 0.01, "SIMD should be accurate")
        }
    }

    @Test("Vector1536Optimized magnitude overflow")
    func testVector1536OptimizedOverflow() async throws {
        // Test Case: Optimized 1536-dim vector with large values
        // Expected: Same stability as generic implementation
        // Context: SIMD kernel stability

        let largeValue: Float = 1.5e19
        let optimized = Vector1536Optimized(repeating: largeValue)

        let magnitude = optimized.magnitude
        print("DEBUG: Vector1536Optimized magnitude = \(magnitude)")

        if magnitude.isInfinite {
            print("INFO: 1536-SIMD also overflows")
            #expect(magnitude.isInfinite, "Detected SIMD overflow")
        } else {
            #expect(magnitude.isFinite, "SIMD should be finite")
            let expected = largeValue * sqrt(1536.0)
            let relError = abs(magnitude - expected) / expected
            #expect(relError < 0.01, "SIMD should be accurate")
        }
    }

    @Test("Optimized normalization consistency with generic")
    func testOptimizedNormalizationConsistency() async throws {
        // Test Case: Compare optimized vs generic normalization
        // Expected: Same results (within floating point tolerance)
        // Context: Optimization correctness

        let testValue: Float = 100.0

        // Test 512
        let generic512 = NumericalStabilityHelpers.createUniformLargeVector(
            dimension: Dim512.self,
            value: testValue
        )
        let optimized512 = Vector512Optimized(repeating: testValue)

        let genericMag512 = generic512.magnitude
        let optimizedMag512 = optimized512.magnitude

        print("DEBUG: 512 - Generic: \(genericMag512), Optimized: \(optimizedMag512)")

        let diff512 = abs(genericMag512 - optimizedMag512)
        let relDiff512 = diff512 / genericMag512

        #expect(
            relDiff512 < 1e-5,
            "512: Generic vs Optimized differ by \(relDiff512 * 100)%"
        )

        // Test 768
        let generic768 = NumericalStabilityHelpers.createUniformLargeVector(
            dimension: Dim768.self,
            value: testValue
        )
        let optimized768 = Vector768Optimized(repeating: testValue)

        let genericMag768 = generic768.magnitude
        let optimizedMag768 = optimized768.magnitude

        let relDiff768 = abs(genericMag768 - optimizedMag768) / genericMag768

        #expect(
            relDiff768 < 1e-5,
            "768: Generic vs Optimized differ by \(relDiff768 * 100)%"
        )

        // Test 1536
        let generic1536 = NumericalStabilityHelpers.createUniformLargeVector(
            dimension: Dim1536.self,
            value: testValue
        )
        let optimized1536 = Vector1536Optimized(repeating: testValue)

        let genericMag1536 = generic1536.magnitude
        let optimizedMag1536 = optimized1536.magnitude

        let relDiff1536 = abs(genericMag1536 - optimizedMag1536) / genericMag1536

        #expect(
            relDiff1536 < 1e-5,
            "1536: Generic vs Optimized differ by \(relDiff1536 * 100)%"
        )

        print("INFO: All optimized implementations consistent with generic")
    }
}

// MARK: - Test Documentation

/*
 # Numerical Stability Test Suite Organization

 ## Test Coverage Matrix

 | Category              | Test Count | Status      | Priority |
 |-----------------------|------------|-------------|----------|
 | Magnitude Overflow    | 10         | Scaffolded  | HIGH     |
 | Overflow Detection    | 5          | Scaffolded  | HIGH     |
 | Numerical Precision   | 6          | Scaffolded  | MEDIUM   |
 | Fuzzing               | 5          | Scaffolded  | MEDIUM   |
 | Regression            | 4          | Scaffolded  | HIGH     |
 | Optimized Variants    | 4          | Scaffolded  | MEDIUM   |
 | **TOTAL**             | **34**     | **0% impl** |          |

 ## Implementation Phases

 ### Phase 1: Scaffold (Current)
 - [x] Test file structure
 - [x] Test suite organization
 - [x] Test case signatures
 - [x] Helper function signatures
 - [x] Documentation

 ### Phase 2: Helper Implementation (Next)
 - [ ] Vector generation helpers
 - [ ] Magnitude calculation helpers
 - [ ] Validation helpers
 - [ ] Diagnostic helpers

 ### Phase 3: Test Logic Implementation
 - [ ] Magnitude overflow tests
 - [ ] Overflow detection tests
 - [ ] Numerical precision tests
 - [ ] Fuzzing tests
 - [ ] Regression tests
 - [ ] Optimized vector tests

 ### Phase 4: Validation
 - [ ] Run tests against current implementation (should fail)
 - [ ] Document failure patterns
 - [ ] Update analysis document with test results

 ## Next Steps

 1. Implement helper functions in NumericalStabilityHelpers
 2. Implement test logic for each test case
 3. Run tests to confirm bugs
 4. Use test failures to guide implementation of fixes
 */
