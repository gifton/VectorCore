// VectorCore: NaN and Infinity Handling Tests
//
// Tests for comprehensive non-finite value handling
//

import XCTest
@testable import VectorCore

final class NaNInfinityHandlingTests: XCTestCase {
    
    // MARK: - Basic Detection Tests
    
    func testNaNDetection() {
        // Create vector with NaN
        var values = Array(repeating: Float(1.0), count: 32)
        values[5] = .nan
        values[15] = .nan
        let vector = Vector<Dim32>(values)
        
        XCTAssertTrue(vector.hasNaN)
        XCTAssertFalse(vector.isFinite)
        
        let check = vector.checkNonFinite()
        XCTAssertTrue(check.hasNaN)
        XCTAssertFalse(check.hasInfinity)
        XCTAssertFalse(check.hasNegativeInfinity)
        XCTAssertEqual(check.nanIndices, [5, 15])
        XCTAssertEqual(check.totalNonFiniteCount, 2)
    }
    
    func testInfinityDetection() {
        // Create vector with infinity values
        var values = Array(repeating: Float(1.0), count: 32)
        values[3] = .infinity
        values[7] = -.infinity
        values[20] = .infinity
        let vector = Vector<Dim32>(values)
        
        XCTAssertTrue(vector.hasInfinity)
        XCTAssertFalse(vector.isFinite)
        
        let check = vector.checkNonFinite()
        XCTAssertFalse(check.hasNaN)
        XCTAssertTrue(check.hasInfinity)
        XCTAssertTrue(check.hasNegativeInfinity)
        XCTAssertEqual(check.infinityIndices, [3, 20])
        XCTAssertEqual(check.negativeInfinityIndices, [7])
    }
    
    func testMixedNonFiniteDetection() {
        // Create vector with mixed non-finite values
        var values = Array(repeating: Float(1.0), count: 64)
        values[5] = .nan
        values[10] = .infinity
        values[15] = -.infinity
        values[25] = .nan
        values[30] = .infinity
        let vector = Vector<Dim64>(values)
        
        let check = vector.checkNonFinite()
        XCTAssertTrue(check.hasNonFinite)
        XCTAssertEqual(check.nanIndices, [5, 25])
        XCTAssertEqual(check.infinityIndices, [10, 30])
        XCTAssertEqual(check.negativeInfinityIndices, [15])
        XCTAssertEqual(check.totalNonFiniteCount, 5)
    }
    
    func testFiniteVector() {
        let vector = Vector<Dim128>.random(in: -10...10)
        
        XCTAssertTrue(vector.isFinite)
        XCTAssertFalse(vector.hasNaN)
        XCTAssertFalse(vector.hasInfinity)
        
        let check = vector.checkNonFinite()
        XCTAssertFalse(check.hasNonFinite)
        XCTAssertEqual(check.totalNonFiniteCount, 0)
    }
    
    // MARK: - Handling Options Tests
    
    func testReplaceNaNWithZero() throws {
        var values = Array(repeating: Float(1.0), count: 32)
        values[5] = .nan
        values[10] = .nan
        let vector = Vector<Dim32>(values)
        
        let handled = try vector.handleNonFinite(options: .replaceNaNWithZero)
        
        XCTAssertEqual(handled[5], 0)
        XCTAssertEqual(handled[10], 0)
        XCTAssertTrue(handled.isFinite)
    }
    
    func testReplaceInfinityWithMax() throws {
        var values = Array(repeating: Float(1.0), count: 32)
        values[5] = .infinity
        values[10] = -.infinity
        let vector = Vector<Dim32>(values)
        
        let handled = try vector.handleNonFinite(options: [.replaceInfinityWithMax, .replaceNegInfinityWithMin])
        
        XCTAssertEqual(handled[5], Float.greatestFiniteMagnitude)
        XCTAssertEqual(handled[10], -Float.greatestFiniteMagnitude)
        XCTAssertTrue(handled.isFinite)
    }
    
    func testReplaceAll() throws {
        var values = Array(repeating: Float(1.0), count: 32)
        values[5] = .nan
        values[10] = .infinity
        values[15] = -.infinity
        let vector = Vector<Dim32>(values)
        
        let handled = try vector.handleNonFinite(options: .replaceAll)
        
        XCTAssertEqual(handled[5], 0)
        XCTAssertEqual(handled[10], Float.greatestFiniteMagnitude)
        XCTAssertEqual(handled[15], -Float.greatestFiniteMagnitude)
        XCTAssertTrue(handled.isFinite)
    }
    
    func testPropagateNaN() throws {
        var values = Array(repeating: Float(1.0), count: 32)
        values[5] = .nan
        let vector = Vector<Dim32>(values)
        
        let handled = try vector.handleNonFinite(options: .propagateNaN)
        
        // All values should be NaN
        for i in 0..<32 {
            XCTAssertTrue(handled[i].isNaN)
        }
    }
    
    func testThrowOnNonFinite() {
        var values = Array(repeating: Float(1.0), count: 32)
        values[5] = .nan
        values[10] = .infinity
        let vector = Vector<Dim32>(values)
        
        XCTAssertThrowsError(try vector.handleNonFinite(options: .strict)) { error in
            guard let nonFiniteError = error as? NonFiniteError,
                  case .mixedNonFiniteValues(let nanIndices, let infIndices, _) = nonFiniteError else {
                XCTFail("Wrong error type")
                return
            }
            XCTAssertEqual(nanIndices, [5])
            XCTAssertEqual(infIndices, [10])
        }
    }
    
    // MARK: - Replacement Tests
    
    func testReplacingNonFinite() {
        var values = Array(repeating: Float(1.0), count: 32)
        values[5] = .nan
        values[10] = .infinity
        values[15] = -.infinity
        let vector = Vector<Dim32>(values)
        
        let replaced = vector.replacingNonFinite(with: 99.0)
        
        XCTAssertEqual(replaced[5], 99.0)
        XCTAssertEqual(replaced[10], 99.0)
        XCTAssertEqual(replaced[15], 99.0)
        XCTAssertEqual(replaced[0], 1.0) // Unchanged
        XCTAssertTrue(replaced.isFinite)
    }
    
    func testFiniteValues() {
        var values = Array(repeating: Float(0), count: 10)
        for i in 0..<10 {
            values[i] = Float(i)
        }
        values[3] = .nan
        values[6] = .infinity
        values[8] = -.infinity
        
        let vector = Vector<Dim32>(values + Array(repeating: 0, count: 22))
        let (finiteVals, indices) = vector.finiteValues()
        
        XCTAssertEqual(finiteVals.count, 29) // 32 - 3 non-finite
        XCTAssertEqual(finiteVals[0], 0)
        XCTAssertEqual(finiteVals[1], 1)
        XCTAssertEqual(finiteVals[2], 2)
        XCTAssertEqual(finiteVals[3], 4) // Skipped index 3 (NaN)
        
        XCTAssertEqual(indices[0], 0)
        XCTAssertEqual(indices[1], 1)
        XCTAssertEqual(indices[2], 2)
        XCTAssertEqual(indices[3], 4) // Skipped index 3
    }
    
    // MARK: - Safe Operation Tests
    
    func testSafeDivision() throws {
        let vector = Vector<Dim32>.ones()
        
        // Normal division
        let result1 = try vector.safeDivide(by: 2.0)
        for i in 0..<32 {
            XCTAssertEqual(result1[i], 0.5)
        }
        
        // Division by zero with replacement
        let result2 = try vector.safeDivide(by: 0, options: .replaceNaNWithZero)
        for i in 0..<32 {
            XCTAssertEqual(result2[i], 0)
        }
        
        // Division by zero with propagation
        let result3 = try vector.safeDivide(by: 0, options: .propagateNaN)
        for i in 0..<32 {
            XCTAssertTrue(result3[i].isNaN)
        }
        
        // Division by zero with throw
        XCTAssertThrowsError(try vector.safeDivide(by: 0, options: .throwOnNonFinite))
    }
    
    func testSafeNormalization() throws {
        // Normal vector
        let v1 = Vector<Dim32>.ones()
        let norm1 = try v1.safeNormalized()
        XCTAssertEqual(norm1.magnitude, 1.0, accuracy: 1e-6)
        
        // Zero vector with replacement
        let v2 = Vector<Dim32>.zeros()
        let norm2 = try v2.safeNormalized(options: .replaceAll)
        XCTAssertEqual(norm2, v2) // Should return zero vector
        
        // Zero vector with throw
        XCTAssertThrowsError(try v2.safeNormalized(options: .throwOnNonFinite))
    }
    
    func testSafeLog() throws {
        var values: [Float] = [2.0, 1.0, 0.0, -1.0, 5.0]
        values.append(contentsOf: Array(repeating: Float(1.0), count: 27))
        let vector = Vector<Dim32>(values)
        
        // With replacement
        let result1 = try vector.safeLog(options: NonFiniteHandling.replaceNaNWithZero)
        XCTAssertEqual(result1[0], log(2.0), accuracy: 1e-6)
        XCTAssertEqual(result1[1], log(1.0), accuracy: 1e-6)
        XCTAssertEqual(result1[2], 0) // log(0) replaced with 0
        XCTAssertEqual(result1[3], 0) // log(-1) replaced with 0
        XCTAssertEqual(result1[4], log(5.0), accuracy: 1e-6)
        
        // With throw
        XCTAssertThrowsError(try vector.safeLog(options: NonFiniteHandling.throwOnNonFinite))
    }
    
    // MARK: - DynamicVector Tests
    
    func testDynamicVectorNaNHandling() throws {
        var values = Array(repeating: Float(1.0), count: 50)
        values[10] = .nan
        values[20] = .infinity
        let vector = DynamicVector(dimension: 50, values: values)
        
        XCTAssertFalse(vector.isFinite)
        
        let check = vector.checkNonFinite()
        XCTAssertTrue(check.hasNaN)
        XCTAssertTrue(check.hasInfinity)
        XCTAssertEqual(check.nanIndices, [10])
        XCTAssertEqual(check.infinityIndices, [20])
        
        let handled = try vector.handleNonFinite(options: .replaceAll)
        XCTAssertTrue(handled.isFinite)
        XCTAssertEqual(handled[10], 0)
        XCTAssertEqual(handled[20], Float.greatestFiniteMagnitude)
    }
    
    // MARK: - Batch Operation Tests
    
    func testFilterFinite() {
        let vectors = [
            Vector<Dim32>.ones(),
            Vector<Dim32>([.nan] + Array(repeating: 1.0, count: 31)),
            Vector<Dim32>.ones() * 2,
            Vector<Dim32>([.infinity] + Array(repeating: 1.0, count: 31)),
            Vector<Dim32>.ones() * 3
        ]
        
        let finite = SyncBatchOperations.filterFinite(vectors)
        XCTAssertEqual(finite.count, 3) // Only vectors without NaN/Inf
        XCTAssertEqual(finite[0][0], 1.0)
        XCTAssertEqual(finite[1][0], 2.0)
        XCTAssertEqual(finite[2][0], 3.0)
    }
    
    func testFindNonFinite() {
        let vectors = [
            Vector<Dim32>.ones(),
            Vector<Dim32>([.nan] + Array(repeating: 1.0, count: 31)),
            Vector<Dim32>.ones() * 2,
            Vector<Dim32>([.infinity] + Array(repeating: 1.0, count: 31)),
            Vector<Dim32>(Array(repeating: -.infinity, count: 32))
        ]
        
        let nonFiniteIndices = SyncBatchOperations.findNonFinite(vectors)
        XCTAssertEqual(nonFiniteIndices, [1, 3, 4])
    }
    
    func testFiniteStatistics() {
        let vectors = [
            Vector<Dim32>.ones(),
            Vector<Dim32>([.nan] + Array(repeating: 1.0, count: 31)),
            Vector<Dim32>.ones() * 2,
            Vector<Dim32>([.infinity] + Array(repeating: 1.0, count: 31)),
            Vector<Dim32>.ones() * 3
        ]
        
        let stats = SyncBatchOperations.finiteStatistics(for: vectors)
        XCTAssertEqual(stats.count, 3) // Only finite vectors counted
        
        // Mean should be (1 + 2 + 3) / 3 = 2.0
        let expectedMean = sqrt(Float(32)) * 2.0
        XCTAssertEqual(stats.meanMagnitude, expectedMean, accuracy: 1e-5)
    }
    
    // MARK: - Global Validation Tests
    
    func testValidateFiniteValue() {
        XCTAssertNoThrow(try validateFinite(1.0))
        XCTAssertNoThrow(try validateFinite(-100.0))
        XCTAssertNoThrow(try validateFinite(0.0))
        
        XCTAssertThrowsError(try validateFinite(.nan))
        XCTAssertThrowsError(try validateFinite(.infinity))
        XCTAssertThrowsError(try validateFinite(-.infinity))
    }
    
    func testValidateFiniteArray() {
        XCTAssertNoThrow(try validateFinite([1.0, 2.0, 3.0]))
        
        XCTAssertThrowsError(try validateFinite([1.0, .nan, 3.0])) { error in
            guard let nonFiniteError = error as? NonFiniteError,
                  case .mixedNonFiniteValues(let nanIndices, _, _) = nonFiniteError else {
                XCTFail("Wrong error type")
                return
            }
            XCTAssertEqual(nanIndices, [1])
        }
    }
    
    // MARK: - Edge Case Tests
    
    func testEmptyHandling() throws {
        // Test with dimension that might have empty storage
        let vector = Vector<Dim32>.zeros()
        XCTAssertTrue(vector.isFinite)
        XCTAssertFalse(vector.hasNaN)
        
        let handled = try vector.handleNonFinite(options: .strict)
        XCTAssertEqual(handled, vector)
    }
    
    func testLargeVectorHandling() throws {
        // Test with large vector
        var values = Array(repeating: Float(1.0), count: 1536)
        values[500] = .nan
        values[1000] = .infinity
        let vector = Vector<Dim1536>(values)
        
        let check = vector.checkNonFinite()
        XCTAssertEqual(check.nanIndices, [500])
        XCTAssertEqual(check.infinityIndices, [1000])
        
        let handled = try vector.handleNonFinite(options: .replaceAll)
        XCTAssertTrue(handled.isFinite)
    }
    
    // MARK: - Performance Tests
    
    func testCheckPerformance() {
        let vector = Vector<Dim1536>.random(in: -100...100)
        
        measure {
            _ = vector.checkNonFinite()
        }
    }
    
    func testHandlePerformance() throws {
        var values = Array(repeating: Float(1.0), count: 1536)
        for i in stride(from: 0, to: 1536, by: 100) {
            values[i] = [Float.nan, .infinity, -.infinity].randomElement()!
        }
        let vector = Vector<Dim1536>(values)
        
        measure {
            _ = try? vector.handleNonFinite(options: .replaceAll)
        }
    }
    
    func testBatchFilterPerformance() {
        let vectors = (0..<1000).map { i in
            if i % 10 == 0 {
                // Add some vectors with NaN
                var values = Array(repeating: Float(1.0), count: 256)
                values[0] = .nan
                return Vector<Dim256>(values)
            } else {
                return Vector<Dim256>.random(in: -10...10)
            }
        }
        
        measure {
            _ = SyncBatchOperations.filterFinite(vectors)
        }
    }
}