import XCTest
@testable import VectorCore

final class BinarySerializationTests: XCTestCase {
    
    // MARK: - Test Configuration
    
    let testDimensions = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    let accuracy: Float = 1e-6
    
    // MARK: - Round-Trip Tests
    
    func testVectorRoundTripSerialization() throws {
        // Test Vector128
        let v128 = Vector128([Float](repeating: 1.5, count: 128))
        let encoded128 = try v128.encodeBinary()
        let decoded128 = try Vector128.decodeBinary(from: encoded128)
        XCTAssertEqual(v128.toArray(), decoded128.toArray())
        
        // Test Vector256
        let values256 = (0..<256).map { Float($0) * 0.1 }
        let v256 = Vector256(values256)
        let encoded256 = try v256.encodeBinary()
        let decoded256 = try Vector256.decodeBinary(from: encoded256)
        
        let original256 = v256.toArray()
        let decoded256Array = decoded256.toArray()
        for i in 0..<256 {
            XCTAssertEqual(original256[i], decoded256Array[i], accuracy: accuracy)
        }
        
        // Test Vector512
        let v512 = Vector512([Float](repeating: -2.5, count: 512))
        let encoded512 = try v512.encodeBinary()
        let decoded512 = try Vector512.decodeBinary(from: encoded512)
        XCTAssertEqual(v512.toArray(), decoded512.toArray())
    }
    
    func testDynamicVectorRoundTripSerialization() throws {
        for dim in testDimensions {
            let values = (0..<dim).map { Float($0) / Float(dim) }
            let vector = DynamicVector(values)
            
            let encoded = try vector.encodeBinary()
            let decoded = try DynamicVector.decodeBinary(from: encoded)
            
            XCTAssertEqual(vector.dimension, decoded.dimension)
            
            let originalArray = vector.toArray()
            let decodedArray = decoded.toArray()
            for i in 0..<dim {
                XCTAssertEqual(originalArray[i], decodedArray[i], accuracy: accuracy)
            }
        }
    }
    
    // MARK: - Header Validation Tests
    
    func testInvalidMagicNumber() {
        var data = Data()
        
        // Write invalid magic
        BinaryFormat.writeUInt32(0x12345678, to: &data)
        BinaryFormat.writeUInt16(BinaryHeader.version, to: &data)
        BinaryFormat.writeUInt32(64, to: &data)
        BinaryFormat.writeUInt16(0, to: &data)
        
        // Add vector data
        for _ in 0..<64 {
            BinaryFormat.writeFloat(1.0, to: &data)
        }
        
        // Add checksum
        let checksum = data.crc32()
        BinaryFormat.writeUInt32(checksum, to: &data)
        
        // Should fail with invalid magic
        XCTAssertThrowsError(try DynamicVector.decodeBinary(from: data)) { error in
            guard let vectorError = error as? VectorError,
                  vectorError.kind == .invalidData else {
                XCTFail("Expected invalidData error")
                return
            }
        }
    }
    
    func testInvalidVersion() {
        var data = Data()
        
        // Write header with invalid version
        BinaryFormat.writeUInt32(BinaryHeader.magic, to: &data)
        BinaryFormat.writeUInt16(999, to: &data)  // Invalid version
        BinaryFormat.writeUInt32(64, to: &data)
        BinaryFormat.writeUInt16(0, to: &data)
        
        // Add vector data
        for _ in 0..<64 {
            BinaryFormat.writeFloat(1.0, to: &data)
        }
        
        // Add checksum
        let checksum = data.crc32()
        BinaryFormat.writeUInt32(checksum, to: &data)
        
        // Should fail with invalid version
        XCTAssertThrowsError(try DynamicVector.decodeBinary(from: data)) { error in
            guard let vectorError = error as? VectorError,
                  vectorError.kind == .invalidData else {
                XCTFail("Expected invalidData error")
                return
            }
        }
    }
    
    func testDimensionMismatch() throws {
        // Create a vector with dimension 256
        let v256 = Vector256([Float](repeating: 1.0, count: 256))
        let encoded = try v256.encodeBinary()
        
        // Try to decode as Vector128 - should fail
        XCTAssertThrowsError(try Vector128.decodeBinary(from: encoded)) { error in
            guard let vectorError = error as? VectorError,
                  vectorError.kind == .dimensionMismatch else {
                XCTFail("Expected dimensionMismatch error")
                return
            }
            // Expected values are contained in the error itself
        }
    }
    
    // MARK: - CRC32 Validation Tests
    
    func testCorruptedDataDetection() throws {
        let vector = Vector128([Float](repeating: 1.0, count: 128))
        var encoded = try vector.encodeBinary()
        
        // Corrupt a byte in the middle
        encoded[encoded.count / 2] ^= 0xFF
        
        // Should fail CRC32 check
        XCTAssertThrowsError(try Vector128.decodeBinary(from: encoded)) { error in
            guard let vectorError = error as? VectorError,
                  vectorError.kind == .dataCorruption else {
                XCTFail("Expected dataCorruption error")
                return
            }
        }
    }
    
    func testInsufficientData() {
        // Create data that's too short
        let shortData = Data(repeating: 0, count: 10)
        
        XCTAssertThrowsError(try Vector128.decodeBinary(from: shortData)) { error in
            guard let vectorError = error as? VectorError,
                  vectorError.kind == .insufficientData else {
                XCTFail("Expected insufficientData error")
                return
            }
        }
    }
    
    // MARK: - Edge Cases
    
    func testSingleElementVector() throws {
        let vector = DynamicVector([42.0])
        let encoded = try vector.encodeBinary()
        let decoded = try DynamicVector.decodeBinary(from: encoded)
        
        XCTAssertEqual(decoded.dimension, 1)
        XCTAssertEqual(decoded[0], 42.0, accuracy: accuracy)
    }
    
    func testLargeDimensionVector() throws {
        // Test maximum allowed dimension
        let dim = 10_000
        let vector = DynamicVector(dimension: dim, repeating: 0.5)
        
        let encoded = try vector.encodeBinary()
        let decoded = try DynamicVector.decodeBinary(from: encoded)
        
        XCTAssertEqual(decoded.dimension, dim)
        XCTAssertEqual(decoded[0], 0.5, accuracy: accuracy)
        XCTAssertEqual(decoded[dim - 1], 0.5, accuracy: accuracy)
    }
    
    func testTooLargeDimension() {
        // Try to encode a vector that's too large
        XCTAssertThrowsError(try {
            let vector = DynamicVector(dimension: 10_001, repeating: 0.0)
            _ = try vector.encodeBinary()
        }()) { error in
            guard let vectorError = error as? VectorError,
                  vectorError.kind == .invalidDimension else {
                XCTFail("Expected invalidDimension error")
                return
            }
        }
    }
    
    // MARK: - Special Float Values
    
    func testSpecialFloatValues() throws {
        let specialValues: [Float] = [
            0.0,
            -0.0,
            .infinity,
            -.infinity,
            .nan,
            .greatestFiniteMagnitude,
            -.greatestFiniteMagnitude,
            .leastNormalMagnitude,
            .leastNonzeroMagnitude
        ]
        
        for value in specialValues {
            let vector = DynamicVector([value])
            let encoded = try vector.encodeBinary()
            let decoded = try DynamicVector.decodeBinary(from: encoded)
            
            if value.isNaN {
                XCTAssertTrue(decoded[0].isNaN)
            } else {
                XCTAssertEqual(decoded[0], value)
            }
        }
    }
    
    // MARK: - Cross-Platform Tests
    
    func testEndiannessConsistency() throws {
        // Create test data manually with known byte order
        var data = Data()
        
        // Header in little-endian
        data.append(contentsOf: [0x54, 0x43, 0x45, 0x56])  // "VECT" magic
        data.append(contentsOf: [0x01, 0x00])              // version 1
        data.append(contentsOf: [0x02, 0x00, 0x00, 0x00])  // dimension 2
        data.append(contentsOf: [0x00, 0x00])              // flags 0
        
        // Float values in little-endian (1.0 and 2.0)
        data.append(contentsOf: [0x00, 0x00, 0x80, 0x3F])  // 1.0
        data.append(contentsOf: [0x00, 0x00, 0x00, 0x40])  // 2.0
        
        // Calculate and append CRC32
        let checksum = data.crc32()
        BinaryFormat.writeUInt32(checksum, to: &data)
        
        // Decode and verify
        let decoded = try DynamicVector.decodeBinary(from: data)
        XCTAssertEqual(decoded.dimension, 2)
        XCTAssertEqual(decoded[0], 1.0, accuracy: accuracy)
        XCTAssertEqual(decoded[1], 2.0, accuracy: accuracy)
    }
    
    // MARK: - Performance Tests
    
    func testLargeVectorSerializationPerformance() throws {
        let dimension = 1024
        let vector = DynamicVector(dimension: dimension, repeating: 1.0)
        
        // Measure encoding performance
        measure {
            _ = try! vector.encodeBinary()
        }
    }
    
    func testLargeVectorDeserializationPerformance() throws {
        let dimension = 1024
        let vector = DynamicVector(dimension: dimension, repeating: 1.0)
        let encoded = try vector.encodeBinary()
        
        // Measure decoding performance
        measure {
            _ = try! DynamicVector.decodeBinary(from: encoded)
        }
    }
    
    // MARK: - Property-Based Tests
    
    func testSerializationPreservesValues() throws {
        for _ in 0..<100 {
            let dimension = Int.random(in: 1...1000)
            let values = (0..<dimension).map { _ in Float.random(in: -100...100) }
            let vector = DynamicVector(values)
            
            let encoded = try vector.encodeBinary()
            let decoded = try DynamicVector.decodeBinary(from: encoded)
            
            XCTAssertEqual(vector.dimension, decoded.dimension)
            
            let originalArray = vector.toArray()
            let decodedArray = decoded.toArray()
            for i in 0..<dimension {
                XCTAssertEqual(originalArray[i], decodedArray[i], accuracy: accuracy)
            }
        }
    }
    
    func testPredictableFileSize() throws {
        for dim in testDimensions {
            let vector = DynamicVector(dimension: dim, repeating: 0.0)
            let encoded = try vector.encodeBinary()
            
            let expectedSize = BinaryFormat.expectedDataSize(for: dim)
            XCTAssertEqual(encoded.count, expectedSize)
        }
    }
}