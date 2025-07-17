// VectorCore: Convenience Initializers Tests
//
// Tests for additional initialization patterns
//

import XCTest
@testable import VectorCore

final class ConvenienceInitializersTests: XCTestCase {
    
    // MARK: - Basis Vector Tests
    
    func testBasisVectors() {
        // Test first basis vector
        let e0 = Vector<Dim32>.basis(axis: 0)
        XCTAssertEqual(e0[0], 1.0)
        for i in 1..<32 {
            XCTAssertEqual(e0[i], 0.0)
        }
        
        // Test middle basis vector
        let e15 = Vector<Dim32>.basis(axis: 15)
        for i in 0..<32 {
            XCTAssertEqual(e15[i], i == 15 ? 1.0 : 0.0)
        }
        
        // Test last basis vector
        let e31 = Vector<Dim32>.basis(axis: 31)
        XCTAssertEqual(e31[31], 1.0)
        for i in 0..<31 {
            XCTAssertEqual(e31[i], 0.0)
        }
    }
    
    func testStandardBasisNotation() {
        let e0 = Vector<Dim64>.e(0)
        let e1 = Vector<Dim64>.e(1)
        let e2 = Vector<Dim64>.e(2)
        
        XCTAssertEqual(e0[0], 1.0)
        XCTAssertEqual(e1[1], 1.0)
        XCTAssertEqual(e2[2], 1.0)
        
        // Verify orthogonality
        XCTAssertEqual(e0.dotProduct(e1), 0.0)
        XCTAssertEqual(e0.dotProduct(e2), 0.0)
        XCTAssertEqual(e1.dotProduct(e2), 0.0)
    }
    
    func testUnitVectorAliases() {
        let unitX = Vector<Dim128>.unitX
        let unitY = Vector<Dim128>.unitY
        let unitZ = Vector<Dim128>.unitZ
        
        XCTAssertEqual(unitX[0], 1.0)
        XCTAssertEqual(unitY[1], 1.0)
        XCTAssertEqual(unitZ[2], 1.0)
        
        // Test edge case for low dimensions
        let unitY2D = Vector<Dim32>.unitY
        XCTAssertEqual(unitY2D[1], 1.0)
        
        // For 1D vectors, unitY and unitZ should be zero
        // (Would need a Dim1 type to test this properly)
    }
    
    // MARK: - Sequential Initializer Tests
    
    func testLinspace() {
        let lin = Vector<Dim32>.linspace(from: 0, to: 31)
        
        // Check first and last values
        XCTAssertEqual(lin[0], 0.0)
        XCTAssertEqual(lin[31], 31.0)
        
        // Check linearity
        for i in 0..<32 {
            XCTAssertEqual(lin[i], Float(i), accuracy: 1e-6)
        }
        
        // Test negative range
        let negLin = Vector<Dim32>.linspace(from: -1, to: 1)
        XCTAssertEqual(negLin[0], -1.0)
        XCTAssertEqual(negLin[31], 1.0)
        XCTAssertEqual(negLin[16], 0.03225806, accuracy: 1e-6) // Midpoint approximately
    }
    
    func testRange() {
        let r1 = Vector<Dim64>.range()
        for i in 0..<64 {
            XCTAssertEqual(r1[i], Float(i))
        }
        
        let r2 = Vector<Dim64>.range(from: 10)
        for i in 0..<64 {
            XCTAssertEqual(r2[i], Float(10 + i))
        }
        
        let r3 = Vector<Dim64>.range(from: -5)
        for i in 0..<64 {
            XCTAssertEqual(r3[i], Float(-5 + i))
        }
    }
    
    // MARK: - Function-based Initializer Tests
    
    func testGenerate() {
        // Square function
        let squares = Vector<Dim32>.generate { i in Float(i * i) }
        for i in 0..<32 {
            XCTAssertEqual(squares[i], Float(i * i))
        }
        
        // Sine wave
        let sine = Vector<Dim128>.generate { i in
            sin(Float(i) * 2 * .pi / 128)
        }
        XCTAssertEqual(sine[0], 0.0, accuracy: 1e-6)
        XCTAssertEqual(sine[32], sin(.pi / 2), accuracy: 1e-6)
        XCTAssertEqual(sine[64], 0.0, accuracy: 1e-6)
    }
    
    func testIndexMapInitializer() {
        let v = Vector<Dim32>(indexMap: { i in Float(i).squareRoot() })
        
        for i in 0..<32 {
            XCTAssertEqual(v[i], sqrt(Float(i)), accuracy: 1e-6)
        }
    }
    
    // MARK: - Mathematical Sequence Tests
    
    func testGeometric() {
        let geo = Vector<Dim32>.geometric(initial: 1, ratio: 2)
        
        // Should be powers of 2: [1, 2, 4, 8, ...]
        for i in 0..<10 { // Test first 10 to avoid overflow
            XCTAssertEqual(geo[i], pow(2, Float(i)))
        }
        
        // Test with ratio < 1
        let decay = Vector<Dim32>.geometric(initial: 100, ratio: 0.5)
        XCTAssertEqual(decay[0], 100)
        XCTAssertEqual(decay[1], 50)
        XCTAssertEqual(decay[2], 25)
        XCTAssertEqual(decay[3], 12.5)
    }
    
    func testPowers() {
        let pow3 = Vector<Dim32>.powers(of: 3)
        
        for i in 0..<8 { // Test first 8 to avoid overflow
            XCTAssertEqual(pow3[i], pow(3, Float(i)))
        }
    }
    
    func testAlternating() {
        let alt = Vector<Dim64>.alternating()
        
        for i in 0..<64 {
            XCTAssertEqual(alt[i], i % 2 == 0 ? 1.0 : -1.0)
        }
        
        let alt5 = Vector<Dim64>.alternating(magnitude: 5.0)
        for i in 0..<64 {
            XCTAssertEqual(alt5[i], i % 2 == 0 ? 5.0 : -5.0)
        }
    }
    
    // MARK: - Special Pattern Tests
    
    func testOneHot() {
        for index in [0, 15, 31] {
            let oneHot = Vector<Dim32>.oneHot(at: index)
            
            for i in 0..<32 {
                XCTAssertEqual(oneHot[i], i == index ? 1.0 : 0.0)
            }
        }
    }
    
    func testRepeatingPattern() {
        let pattern = Vector<Dim32>.repeatingPattern([1, 2, 3])
        
        for i in 0..<32 {
            let expected = Float([1, 2, 3][i % 3])
            XCTAssertEqual(pattern[i], expected)
        }
        
        // Test single element pattern
        let single = Vector<Dim32>.repeatingPattern([42])
        for i in 0..<32 {
            XCTAssertEqual(single[i], 42)
        }
    }
    
    func testSparse() {
        let sparse = Vector<Dim128>.sparse(value: 10.0, at: [5, 10, 20, 50])
        
        // Check sparse values
        XCTAssertEqual(sparse[5], 10.0)
        XCTAssertEqual(sparse[10], 10.0)
        XCTAssertEqual(sparse[20], 10.0)
        XCTAssertEqual(sparse[50], 10.0)
        
        // Check zeros
        for i in 0..<128 {
            if ![5, 10, 20, 50].contains(i) {
                XCTAssertEqual(sparse[i], 0.0)
            }
        }
        
        // Test with out-of-bounds indices (should be ignored)
        let sparseOOB = Vector<Dim32>.sparse(value: 1.0, at: [-1, 10, 100])
        XCTAssertEqual(sparseOOB[10], 1.0)
        XCTAssertEqual(sparseOOB.magnitude, 1.0) // Only one non-zero element
    }
    
    // MARK: - DynamicVector Tests
    
    func testDynamicBasis() {
        let e5 = DynamicVector.basis(dimension: 10, axis: 5)
        XCTAssertEqual(e5.dimension, 10)
        XCTAssertEqual(e5[5], 1.0)
        XCTAssertEqual(e5.magnitude, 1.0)
    }
    
    func testDynamicLinspace() {
        let lin = DynamicVector.linspace(dimension: 11, from: 0, to: 10)
        XCTAssertEqual(lin.dimension, 11)
        
        for i in 0..<11 {
            XCTAssertEqual(lin[i], Float(i), accuracy: 1e-6)
        }
    }
    
    func testDynamicRange() {
        let r = DynamicVector.range(dimension: 20, from: -10)
        XCTAssertEqual(r.dimension, 20)
        
        for i in 0..<20 {
            XCTAssertEqual(r[i], Float(-10 + i))
        }
    }
    
    func testDynamicGenerate() {
        let squares = DynamicVector.generate(dimension: 16) { i in
            Float(i * i)
        }
        
        XCTAssertEqual(squares.dimension, 16)
        for i in 0..<16 {
            XCTAssertEqual(squares[i], Float(i * i))
        }
    }
    
    func testDynamicIndexMap() {
        let v = DynamicVector(dimension: 8, indexMap: { i in Float(1 << i) })
        
        XCTAssertEqual(v.dimension, 8)
        for i in 0..<8 {
            XCTAssertEqual(v[i], Float(1 << i))
        }
    }
    
    func testDynamicGeometric() {
        let geo = DynamicVector.geometric(dimension: 10, initial: 1, ratio: 3)
        
        XCTAssertEqual(geo.dimension, 10)
        for i in 0..<10 {
            XCTAssertEqual(geo[i], pow(3, Float(i)))
        }
    }
    
    func testDynamicAlternating() {
        let alt = DynamicVector.alternating(dimension: 7, magnitude: 2.5)
        
        XCTAssertEqual(alt.dimension, 7)
        XCTAssertEqual(alt[0], 2.5)
        XCTAssertEqual(alt[1], -2.5)
        XCTAssertEqual(alt[2], 2.5)
        XCTAssertEqual(alt[3], -2.5)
    }
    
    // MARK: - Edge Case Tests
    
    func testEdgeCases() {
        // Single dimension linspace
        let single = DynamicVector.linspace(dimension: 1, from: 5, to: 10)
        XCTAssertEqual(single.dimension, 1)
        XCTAssertEqual(single[0], 5.0) // Should use 'from' value
        
        // Empty pattern should trap in debug
        // XCTAssertThrowsError(Vector<Dim32>.repeatingPattern([]))
        
        // Out of bounds basis should trap in debug
        // XCTAssertThrowsError(Vector<Dim32>.basis(axis: 32))
    }
    
    // MARK: - Performance Tests
    
    func testBasisPerformance() {
        measure {
            for i in 0..<1000 {
                _ = Vector<Dim256>.basis(axis: i % 256)
            }
        }
    }
    
    func testGeneratePerformance() {
        measure {
            for _ in 0..<100 {
                _ = Vector<Dim1536>.generate { i in
                    sin(Float(i) * 0.01)
                }
            }
        }
    }
    
    func testLinspacePerformance() {
        measure {
            for _ in 0..<1000 {
                _ = Vector<Dim512>.linspace(from: -100, to: 100)
            }
        }
    }
}