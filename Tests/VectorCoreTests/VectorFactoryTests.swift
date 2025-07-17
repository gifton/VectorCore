// VectorCore: Vector Factory Tests
//
// Comprehensive tests for VectorFactory
//

import XCTest
@testable import VectorCore

final class VectorFactoryTests: XCTestCase {
    
    // MARK: - Create Generic Vector Tests
    
    func testCreateGenericVector() throws {
        // Test creating vectors with compile-time dimensions
        let values128 = Array(repeating: Float(1.0), count: 128)
        let vector128 = try VectorFactory.create(Dim128.self, from: values128)
        XCTAssertEqual(vector128.scalarCount, 128)
        XCTAssertEqual(vector128.toArray(), values128)
        
        let values256 = Array(repeating: Float(2.0), count: 256)
        let vector256 = try VectorFactory.create(Dim256.self, from: values256)
        XCTAssertEqual(vector256.scalarCount, 256)
        XCTAssertEqual(vector256.toArray(), values256)
        
        let values512 = (0..<512).map { Float($0) }
        let vector512 = try VectorFactory.create(Dim512.self, from: values512)
        XCTAssertEqual(vector512.scalarCount, 512)
        XCTAssertEqual(vector512.toArray(), values512)
    }
    
    func testCreateGenericVectorDimensionMismatch() {
        // Test dimension mismatch errors
        let wrongSize = Array(repeating: Float(1.0), count: 100)
        
        XCTAssertThrowsError(try VectorFactory.create(Dim128.self, from: wrongSize)) { error in
            guard let vectorError = error as? VectorError else {
                XCTFail("Expected VectorError")
                return
            }
            if case .dimensionMismatch = vectorError {
                // Expected error
            } else {
                XCTFail("Expected dimensionMismatch error")
            }
        }
        
        // Test empty array
        XCTAssertThrowsError(try VectorFactory.create(Dim256.self, from: [])) { error in
            guard let vectorError = error as? VectorError else {
                XCTFail("Expected VectorError")
                return
            }
            if case .dimensionMismatch = vectorError {
                // Expected error
            } else {
                XCTFail("Expected dimensionMismatch error")
            }
        }
    }
    
    // MARK: - Runtime Vector Creation Tests
    
    func testVectorCreationAllDimensions() throws {
        // Test all supported dimensions
        let dimensions = [128, 256, 512, 768, 1536, 3072]
        
        for dim in dimensions {
            let values = Array(repeating: Float(1.0), count: dim)
            let vector = try VectorFactory.vector(of: dim, from: values)
            
            XCTAssertEqual(vector.scalarCount, dim)
            XCTAssertEqual(vector.toArray(), values)
            
            // Verify correct type is returned
            switch dim {
            case 128:
                XCTAssertTrue(vector is Vector<Dim128>)
            case 256:
                XCTAssertTrue(vector is Vector<Dim256>)
            case 512:
                XCTAssertTrue(vector is Vector<Dim512>)
            case 768:
                XCTAssertTrue(vector is Vector<Dim768>)
            case 1536:
                XCTAssertTrue(vector is Vector<Dim1536>)
            case 3072:
                XCTAssertTrue(vector is Vector<Dim3072>)
            default:
                XCTFail("Unexpected dimension")
            }
        }
    }
    
    func testVectorCreationUnsupportedDimension() throws {
        // Test unsupported dimensions fall back to DynamicVector
        let unsupportedDims = [64, 100, 1000, 2048, 4096]
        
        for dim in unsupportedDims {
            let values = Array(repeating: Float(0.5), count: dim)
            let vector = try VectorFactory.vector(of: dim, from: values)
            
            XCTAssertEqual(vector.scalarCount, dim)
            XCTAssertTrue(vector is DynamicVector)
            XCTAssertEqual(vector.toArray(), values)
        }
    }
    
    // MARK: - Random Vector Tests
    
    func testRandomVectorGeneration() {
        // Test default range
        let vector1 = VectorFactory.random(dimension: 256)
        XCTAssertEqual(vector1.scalarCount, 256)
        
        let values1 = vector1.toArray()
        for value in values1 {
            XCTAssertGreaterThanOrEqual(value, -1.0)
            XCTAssertLessThanOrEqual(value, 1.0)
        }
        
        // Test custom range
        let customRange: ClosedRange<Float> = 0...10
        let vector2 = VectorFactory.random(dimension: 512, range: customRange)
        XCTAssertEqual(vector2.scalarCount, 512)
        
        let values2 = vector2.toArray()
        for value in values2 {
            XCTAssertGreaterThanOrEqual(value, 0.0)
            XCTAssertLessThanOrEqual(value, 10.0)
        }
        
        // Test randomness (vectors should be different)
        let vector3 = VectorFactory.random(dimension: 256)
        XCTAssertNotEqual(vector1.toArray(), vector3.toArray())
    }
    
    // MARK: - Zero and Ones Tests
    
    func testZerosCreation() {
        // Test all supported dimensions
        let dimensions = [128, 256, 512, 768, 1536, 3072, 1000]
        
        for dim in dimensions {
            let vector = VectorFactory.zeros(dimension: dim)
            XCTAssertEqual(vector.scalarCount, dim)
            
            let values = vector.toArray()
            for value in values {
                XCTAssertEqual(value, 0.0)
            }
            
            // Verify magnitude is zero
            XCTAssertEqual(vector.magnitude, 0.0)
        }
    }
    
    func testOnesCreation() {
        // Test all supported dimensions
        let dimensions = [128, 256, 512, 768, 1536, 3072, 1000]
        
        for dim in dimensions {
            let vector = VectorFactory.ones(dimension: dim)
            XCTAssertEqual(vector.scalarCount, dim)
            
            let values = vector.toArray()
            for value in values {
                XCTAssertEqual(value, 1.0)
            }
            
            // Verify magnitude
            let expectedMagnitude = sqrt(Float(dim))
            XCTAssertEqual(vector.magnitude, expectedMagnitude, accuracy: 1e-5)
        }
    }
    
    // MARK: - Pattern Creation Tests
    
    func testPatternCreation() {
        // Test simple patterns
        let linearPattern = VectorFactory.withPattern(dimension: 256) { Float($0) }
        XCTAssertEqual(linearPattern.scalarCount, 256)
        
        let linearValues = linearPattern.toArray()
        for (index, value) in linearValues.enumerated() {
            XCTAssertEqual(value, Float(index))
        }
        
        // Test sine wave pattern
        let sinePattern = VectorFactory.withPattern(dimension: 512) { index in
            sin(Float(index) * 0.1)
        }
        XCTAssertEqual(sinePattern.scalarCount, 512)
        
        let sineValues = sinePattern.toArray()
        for (index, value) in sineValues.enumerated() {
            XCTAssertEqual(value, sin(Float(index) * 0.1), accuracy: 1e-6)
        }
        
        // Test alternating pattern
        let alternating = VectorFactory.withPattern(dimension: 128) { $0 % 2 == 0 ? 1.0 : -1.0 }
        let altValues = alternating.toArray()
        for (index, value) in altValues.enumerated() {
            XCTAssertEqual(value, index % 2 == 0 ? 1.0 : -1.0)
        }
    }
    
    // MARK: - Batch Creation Tests
    
    func testBatchCreation() throws {
        // Test valid batch creation
        let flatData = (0..<1024).map { Float($0) }
        let batch = try VectorFactory.batch(dimension: 256, from: flatData)
        
        XCTAssertEqual(batch.count, 4) // 1024 / 256 = 4
        
        for (i, vector) in batch.enumerated() {
            XCTAssertEqual(vector.scalarCount, 256)
            let expected = Array(flatData[i*256..<(i+1)*256])
            XCTAssertEqual(vector.toArray(), expected)
        }
    }
    
    func testBatchCreationInvalidSize() {
        // Test non-multiple of dimension
        let flatData = (0..<1000).map { Float($0) }
        
        XCTAssertThrowsError(try VectorFactory.batch(dimension: 256, from: flatData)) { error in
            guard let vectorError = error as? VectorError else {
                XCTFail("Expected VectorError")
                return
            }
            if case .invalidValues = vectorError {
                // Expected error  
            } else {
                XCTFail("Expected invalidValues error")
            }
        }
        
        // Test empty data
        XCTAssertNoThrow(try VectorFactory.batch(dimension: 256, from: []))
        let emptyBatch = try! VectorFactory.batch(dimension: 256, from: [])
        XCTAssertTrue(emptyBatch.isEmpty)
    }
    
    // MARK: - Helper Methods Tests
    
    func testOptimalDimensionSelection() {
        // Test exact matches
        XCTAssertEqual(VectorFactory.optimalDimension(for: 128), 128)
        XCTAssertEqual(VectorFactory.optimalDimension(for: 256), 256)
        XCTAssertEqual(VectorFactory.optimalDimension(for: 512), 512)
        XCTAssertEqual(VectorFactory.optimalDimension(for: 768), 768)
        XCTAssertEqual(VectorFactory.optimalDimension(for: 1536), 1536)
        XCTAssertEqual(VectorFactory.optimalDimension(for: 3072), 3072)
        
        // Test nearest selection
        XCTAssertEqual(VectorFactory.optimalDimension(for: 100), 128)
        XCTAssertEqual(VectorFactory.optimalDimension(for: 200), 256)
        XCTAssertEqual(VectorFactory.optimalDimension(for: 400), 512)
        XCTAssertEqual(VectorFactory.optimalDimension(for: 1000), 768)
        XCTAssertEqual(VectorFactory.optimalDimension(for: 2000), 1536)
        XCTAssertEqual(VectorFactory.optimalDimension(for: 4000), 3072)
        
        // Test edge cases
        XCTAssertEqual(VectorFactory.optimalDimension(for: 0), 128)
        XCTAssertEqual(VectorFactory.optimalDimension(for: 192), 256) // Exactly between 128 and 256
    }
    
    func testIsSupported() {
        // Test supported dimensions
        XCTAssertTrue(VectorFactory.isSupported(dimension: 128))
        XCTAssertTrue(VectorFactory.isSupported(dimension: 256))
        XCTAssertTrue(VectorFactory.isSupported(dimension: 512))
        XCTAssertTrue(VectorFactory.isSupported(dimension: 768))
        XCTAssertTrue(VectorFactory.isSupported(dimension: 1536))
        XCTAssertTrue(VectorFactory.isSupported(dimension: 3072))
        
        // Test unsupported dimensions
        XCTAssertFalse(VectorFactory.isSupported(dimension: 64))
        XCTAssertFalse(VectorFactory.isSupported(dimension: 100))
        XCTAssertFalse(VectorFactory.isSupported(dimension: 1024))
        XCTAssertFalse(VectorFactory.isSupported(dimension: 2048))
        XCTAssertFalse(VectorFactory.isSupported(dimension: 4096))
    }
    
    // MARK: - Basis Vector Tests
    
    func testBasisVectorCreation() throws {
        // Test supported dimensions
        let dimensions = [128, 256, 512]
        
        for dim in dimensions {
            // Test first basis vector
            let basis0 = try VectorFactory.basis(dimension: dim, index: 0)
            XCTAssertEqual(basis0.scalarCount, dim)
            let values0 = basis0.toArray()
            XCTAssertEqual(values0[0], 1.0)
            for i in 1..<dim {
                XCTAssertEqual(values0[i], 0.0)
            }
            
            // Test last basis vector
            let basisLast = try VectorFactory.basis(dimension: dim, index: dim - 1)
            let valuesLast = basisLast.toArray()
            XCTAssertEqual(valuesLast[dim - 1], 1.0)
            for i in 0..<(dim - 1) {
                XCTAssertEqual(valuesLast[i], 0.0)
            }
            
            // Test middle basis vector
            let mid = dim / 2
            let basisMid = try VectorFactory.basis(dimension: dim, index: mid)
            let valuesMid = basisMid.toArray()
            XCTAssertEqual(valuesMid[mid], 1.0)
            XCTAssertEqual(basisMid.magnitude, 1.0, accuracy: 1e-6)
        }
    }
    
    func testBasisVectorInvalidIndex() {
        // Test negative index
        XCTAssertThrowsError(try VectorFactory.basis(dimension: 256, index: -1)) { error in
            guard let vectorError = error as? VectorError else {
                XCTFail("Expected VectorError")
                return
            }
            if case .indexOutOfBounds = vectorError {
                // Expected error
            } else {
                XCTFail("Expected indexOutOfBounds error")
            }
        }
        
        // Test index >= dimension
        XCTAssertThrowsError(try VectorFactory.basis(dimension: 256, index: 256)) { error in
            guard let vectorError = error as? VectorError else {
                XCTFail("Expected VectorError")
                return
            }
            if case .indexOutOfBounds = vectorError {
                // Expected error
            } else {
                XCTFail("Expected indexOutOfBounds error")
            }
        }
        
        XCTAssertThrowsError(try VectorFactory.basis(dimension: 128, index: 200)) { error in
            guard let vectorError = error as? VectorError else {
                XCTFail("Expected VectorError")
                return
            }
            if case .indexOutOfBounds = vectorError {
                // Expected error
            } else {
                XCTFail("Expected indexOutOfBounds error")
            }
        }
    }
    
    // MARK: - Normalized Random Tests
    
    func testRandomNormalizedVector() {
        // Test multiple dimensions
        let dimensions = [128, 256, 512, 1000]
        
        for dim in dimensions {
            let vector = VectorFactory.randomNormalized(dimension: dim)
            XCTAssertEqual(vector.scalarCount, dim)
            
            // Verify it's normalized
            XCTAssertEqual(vector.magnitude, 1.0, accuracy: 1e-5)
            
            // Verify values are in reasonable range
            let values = vector.toArray()
            for value in values {
                XCTAssertGreaterThan(value, -1.0)
                XCTAssertLessThan(value, 1.0)
            }
        }
        
        // Test randomness
        let v1 = VectorFactory.randomNormalized(dimension: 512)
        let v2 = VectorFactory.randomNormalized(dimension: 512)
        XCTAssertNotEqual(v1.toArray(), v2.toArray())
    }
    
    // MARK: - Performance Tests
    
    func testFactoryPerformance() {
        // Test creation performance
        measure {
            for _ in 0..<1000 {
                _ = VectorFactory.random(dimension: 512)
            }
        }
    }
    
    func testBatchCreationPerformance() throws {
        let largeData = (0..<100_000).map { Float($0) }
        
        measure {
            _ = try! VectorFactory.batch(dimension: 1000, from: largeData)
        }
    }
}