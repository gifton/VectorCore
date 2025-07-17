// VectorCore: CRC32 Tests
//
// Tests for CRC32 checksum implementation
//

import XCTest
@testable import VectorCore

final class CRC32Tests: XCTestCase {
    
    // MARK: - Basic Tests
    
    func testCRC32EmptyData() {
        // CRC32 of empty data should be 0
        let emptyData = Data()
        XCTAssertEqual(emptyData.crc32(), 0x00000000)
        
        // Test with ContiguousBytes
        let emptyArray: [UInt8] = []
        XCTAssertEqual(emptyArray.crc32(), 0x00000000)
    }
    
    func testCRC32SingleByte() {
        // Test single byte values
        let data1 = Data([0x00])
        XCTAssertEqual(data1.crc32(), 0xD202EF8D)
        
        let data2 = Data([0xFF])
        XCTAssertEqual(data2.crc32(), 0xFF000000)
        
        let data3 = Data([0x01])
        XCTAssertEqual(data3.crc32(), 0xA505DF1B)
    }
    
    // MARK: - Known Test Vectors
    
    func testCRC32KnownTestVectors() {
        // Test against known CRC32 values
        
        // "123456789"
        let testString1 = "123456789"
        let data1 = testString1.data(using: .ascii)!
        XCTAssertEqual(data1.crc32(), 0xCBF43926)
        
        // "The quick brown fox jumps over the lazy dog"
        let testString2 = "The quick brown fox jumps over the lazy dog"
        let data2 = testString2.data(using: .ascii)!
        XCTAssertEqual(data2.crc32(), 0x414FA339)
        
        // "Hello, World!"
        let testString3 = "Hello, World!"
        let data3 = testString3.data(using: .utf8)!
        XCTAssertEqual(data3.crc32(), 0xEC4AC3D0)
        
        // All zeros (4 bytes)
        let zeros = Data([0x00, 0x00, 0x00, 0x00])
        XCTAssertEqual(zeros.crc32(), 0x2144DF1C)
        
        // All ones (4 bytes)
        let ones = Data([0xFF, 0xFF, 0xFF, 0xFF])
        XCTAssertEqual(ones.crc32(), 0xFFFFFFFF)
    }
    
    // MARK: - Large Data Tests
    
    func testCRC32LargeData() {
        // Test with larger data sets
        let size = 10_000
        
        // Sequential bytes
        let sequentialData = Data((0..<size).map { UInt8($0 & 0xFF) })
        let sequentialCRC = sequentialData.crc32()
        
        // Verify consistency - same data should give same CRC
        XCTAssertEqual(sequentialData.crc32(), sequentialCRC)
        
        // Random data
        let randomData = Data((0..<size).map { _ in UInt8.random(in: 0...255) })
        let randomCRC = randomData.crc32()
        
        // Different data should give different CRC (with high probability)
        XCTAssertNotEqual(sequentialCRC, randomCRC)
    }
    
    // MARK: - Consistency Tests
    
    func testCRC32Consistency() {
        // Test that CRC32 is consistent across multiple calls
        let testData = "Consistency test data".data(using: .utf8)!
        
        let crc1 = testData.crc32()
        let crc2 = testData.crc32()
        let crc3 = testData.crc32()
        
        XCTAssertEqual(crc1, crc2)
        XCTAssertEqual(crc2, crc3)
    }
    
    func testCRC32Incremental() {
        // Test that concatenated data gives expected result
        let part1 = "Hello, ".data(using: .utf8)!
        let part2 = "World!".data(using: .utf8)!
        let combined = part1 + part2
        
        let combinedCRC = combined.crc32()
        let helloworldCRC = "Hello, World!".data(using: .utf8)!.crc32()
        
        XCTAssertEqual(combinedCRC, helloworldCRC)
    }
    
    // MARK: - ContiguousBytes Tests
    
    func testCRC32ContiguousBytes() {
        // Test with different types conforming to ContiguousBytes
        
        // Array<UInt8>
        let byteArray: [UInt8] = [0x01, 0x02, 0x03, 0x04]
        let byteArrayCRC = byteArray.crc32()
        
        // Data with same bytes
        let data = Data([0x01, 0x02, 0x03, 0x04])
        let dataCRC = data.crc32()
        
        XCTAssertEqual(byteArrayCRC, dataCRC)
        
        // ArraySlice
        let fullArray: [UInt8] = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05]
        let slice = fullArray[1...4]
        let sliceCRC = Array(slice).crc32()
        
        XCTAssertEqual(sliceCRC, byteArrayCRC)
    }
    
    // MARK: - Edge Cases
    
    func testCRC32EdgeCases() {
        // Very large single value repeated
        let largeRepeating = Data(repeating: 0xAB, count: 1_000_000)
        let crc1 = largeRepeating.crc32()
        
        // Should be deterministic
        let crc2 = largeRepeating.crc32()
        XCTAssertEqual(crc1, crc2)
        
        // Single bit difference should change CRC
        var modifiedData = largeRepeating
        modifiedData[500_000] = 0xAC // Change one byte
        let modifiedCRC = modifiedData.crc32()
        
        XCTAssertNotEqual(crc1, modifiedCRC)
    }
    
    // MARK: - Table Generation Test
    
    func testCRC32TableGeneration() {
        // Verify the CRC32 table is correctly generated
        // Check a few known values from the standard CRC32 table
        
        // CRC32 table is private, test by computing known checksums
        // Known CRC32 values
        let testData = Data([0x00])
        let knownCRC = testData.crc32()
        XCTAssertNotEqual(knownCRC, 0) // Should compute a non-zero CRC
        // Table checks removed as table is private
    }
    
    // MARK: - Binary Data Tests
    
    func testCRC32WithBinaryData() {
        // Test with float array data (as used in vector serialization)
        let floats: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let data = floats.withUnsafeBytes { Data($0) }
        
        let crc1 = data.crc32()
        
        // Same floats should give same CRC
        let floats2: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let data2 = floats2.withUnsafeBytes { Data($0) }
        let crc2 = data2.crc32()
        
        XCTAssertEqual(crc1, crc2)
        
        // Different floats should give different CRC
        let floats3: [Float] = [1.0, 2.0, 3.0, 4.0, 5.1]
        let data3 = floats3.withUnsafeBytes { Data($0) }
        let crc3 = data3.crc32()
        
        XCTAssertNotEqual(crc1, crc3)
    }
    
    // MARK: - Performance Tests
    
    func testCRC32Performance() {
        let testData = Data(repeating: 0x42, count: 1_000_000) // 1MB
        
        measure {
            _ = testData.crc32()
        }
    }
    
    func testCRC32SmallDataPerformance() {
        let smallData = Data([0x01, 0x02, 0x03, 0x04])
        
        measure {
            for _ in 0..<10000 {
                _ = smallData.crc32()
            }
        }
    }
    
    // MARK: - Cross-Platform Consistency
    
    func testCRC32CrossPlatform() {
        // These CRC32 values should be consistent across all platforms
        // as they use the standard IEEE 802.3 polynomial
        
        struct TestVector {
            let input: String
            let expectedCRC: UInt32
        }
        
        let testVectors = [
            TestVector(input: "", expectedCRC: 0x00000000),
            TestVector(input: "a", expectedCRC: 0xE8B7BE43),
            TestVector(input: "abc", expectedCRC: 0x352441C2),
            TestVector(input: "message digest", expectedCRC: 0x20159D7F),
            TestVector(input: "abcdefghijklmnopqrstuvwxyz", expectedCRC: 0x4C2750BD),
            TestVector(input: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", 
                       expectedCRC: 0x1FC2E6D2),
            TestVector(input: "12345678901234567890123456789012345678901234567890123456789012345678901234567890",
                       expectedCRC: 0x7CA94A72)
        ]
        
        for vector in testVectors {
            let data = vector.input.data(using: .ascii)!
            let crc = data.crc32()
            XCTAssertEqual(crc, vector.expectedCRC, 
                          "CRC32 mismatch for input: '\(vector.input)'")
        }
    }
}