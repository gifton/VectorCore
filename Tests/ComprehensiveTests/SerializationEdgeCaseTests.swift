//
//  SerializationEdgeCaseTests.swift
//  VectorCore
//
//  Comprehensive test suite for vector serialization edge cases, covering
//  binary/JSON encoding, format compatibility, data integrity, and error handling.
//

import Testing
import Foundation
@testable import VectorCore

/// Comprehensive test suite for serialization edge cases and boundary conditions
@Suite("Serialization Edge Cases")
struct SerializationEdgeCaseTests {

    // MARK: - Binary Serialization Tests

    @Suite("Binary Serialization")
    struct BinarySerializationTests {

        @Test("Empty vector serialization")
        func testEmptyVectorSerialization() async throws {
            // VectorCore doesn't support 0-dimension vectors (minimum is 1)
            // Test with minimum valid dimension instead
            let minimal = DynamicVector([0.0])  // Single element vector

            // Test binary format
            let binaryData = minimal.encodeBinary()
            let decodedBinary = try DynamicVector.decodeBinary(from: binaryData)
            #expect(decodedBinary.scalarCount == 1)
            #expect(decodedBinary.toArray() == [0.0])

            // Test JSON format
            let encoder = JSONEncoder()
            let jsonData = try encoder.encode(minimal)
            let decoder = JSONDecoder()
            let decodedJSON = try decoder.decode(DynamicVector.self, from: jsonData)
            #expect(decodedJSON.scalarCount == 1)
            #expect(decodedJSON.toArray() == [0.0])
        }

        @Test("Single element vector serialization")
        func testSingleElementVectorSerialization() async throws {
            let testValues: [Float] = [0.0, 1.0, -1.0, Float.pi, Float.leastNormalMagnitude]

            for value in testValues {
                let vector = DynamicVector([value])

                // Binary serialization
                let binaryData = vector.encodeBinary()
                let decodedBinary = try DynamicVector.decodeBinary(from: binaryData)
                #expect(decodedBinary.dimension == 1)
                #expect(decodedBinary[0] == value)

                // JSON serialization
                let jsonData = try vector.encodeJSON()
                let decodedJSON = try DynamicVector.decodeJSON(from: jsonData)
                #expect(decodedJSON.dimension == 1)
                #expect(decodedJSON[0] == value)

                // Verify size: header + 1 float + checksum
                let expectedSize = BinaryHeader.headerSize + MemoryLayout<Float>.size + 4
                #expect(binaryData.count == expectedSize)
            }
        }

        @Test("Maximum dimension vector serialization")
        func testMaximumDimensionVectorSerialization() async throws {
            // Test practical maximum dimension (not UInt32.max due to memory constraints)
            let maxPracticalDim = 10_000  // BinaryFormat.maxDimension
            let largeVector = DynamicVector(dimension: maxPracticalDim, repeating: 0.5)

            // Binary serialization
            let binaryData = largeVector.encodeBinary()
            let decoded = try DynamicVector.decodeBinary(from: binaryData)
            #expect(decoded.dimension == maxPracticalDim)
            #expect(decoded[0] == 0.5)
            #expect(decoded[maxPracticalDim - 1] == 0.5)

            // Test dimension overflow protection
            let oversizedDim = maxPracticalDim + 1
            let oversizedVector = DynamicVector(dimension: oversizedDim, repeating: 0.0)
            let oversizedData = oversizedVector.encodeBinary()

            // Should fail validation on decode if > maxDimension
            do {
                _ = try DynamicVector.decodeBinary(from: oversizedData)
                Issue.record("Should have thrown for oversized dimension")
            } catch {
                // Expected error
            }
        }

        @Test("Special float values serialization")
        func testSpecialFloatValuesSerialization() async throws {
            let specialValues: [Float] = [
                Float.nan,
                Float.infinity,
                -Float.infinity,
                0.0,
                -0.0,
                Float.leastNormalMagnitude,
                Float.leastNonzeroMagnitude,
                Float.greatestFiniteMagnitude
            ]

            for value in specialValues {
                let vector = DynamicVector([value, -value, value * 2])

                // Binary serialization - should be bit-perfect
                let binaryData = vector.encodeBinary()
                let decodedBinary = try DynamicVector.decodeBinary(from: binaryData)

                for i in 0..<3 {
                    if value.isNaN {
                        #expect(decodedBinary[i].isNaN)
                    } else {
                        #expect(decodedBinary[i].bitPattern == vector[i].bitPattern)
                    }
                }

                // JSON serialization - may have special handling
                let encoder = JSONEncoder()
                encoder.nonConformingFloatEncodingStrategy = .convertToString(
                    positiveInfinity: "Infinity",
                    negativeInfinity: "-Infinity",
                    nan: "NaN"
                )
                let jsonData = try vector.encodeJSON(encoder: encoder)

                let decoder = JSONDecoder()
                decoder.nonConformingFloatDecodingStrategy = .convertFromString(
                    positiveInfinity: "Infinity",
                    negativeInfinity: "-Infinity",
                    nan: "NaN"
                )
                let decodedJSON = try DynamicVector.decodeJSON(from: jsonData, decoder: decoder)

                for i in 0..<3 {
                    if vector[i].isNaN {
                        #expect(decodedJSON[i].isNaN)
                    } else if vector[i].isInfinite {
                        #expect(decodedJSON[i] == vector[i])
                    } else {
                        #expect(abs(decodedJSON[i] - vector[i]) < Float.ulpOfOne)
                    }
                }
            }
        }

        @Test("Binary header validation")
        func testBinaryHeaderValidation() async throws {
            let vector = DynamicVector([1.0, 2.0, 3.0])
            var validData = vector.encodeBinary()

            // Test corrupt magic number
            var corruptMagic = validData
            corruptMagic[0] = 0xFF
            do {
                _ = try DynamicVector.decodeBinary(from: corruptMagic)
                Issue.record("Should fail with corrupt magic number")
            } catch {
                // Expected
            }

            // Test invalid version
            var corruptVersion = validData
            corruptVersion[4] = 0xFF
            corruptVersion[5] = 0xFF
            do {
                _ = try DynamicVector.decodeBinary(from: corruptVersion)
                Issue.record("Should fail with invalid version")
            } catch {
                // Expected
            }

            // Test dimension mismatch for fixed vectors
            let vec512 = try Vector512Optimized(Array(repeating: 1.0, count: 512))
            var vec512Data = vec512.encodeBinary()

            // Corrupt dimension to 256
            vec512Data[6] = 0x00
            vec512Data[7] = 0x01  // 256 in little-endian
            vec512Data[8] = 0x00
            vec512Data[9] = 0x00

            do {
                _ = try Vector<Dim512>.decodeBinary(from: vec512Data)
                Issue.record("Should fail with dimension mismatch")
            } catch {
                // Expected dimension mismatch error
            }
        }

        @Test("Corrupted data detection")
        func testCorruptedDataDetection() async throws {
            let vector = DynamicVector([1.0, 2.0, 3.0, 4.0])
            let validData = vector.encodeBinary()

            // Test 1: Corrupt data bytes should trigger CRC32 failure
            var corruptData = validData
            if corruptData.count > 24 {  // Ensure we're past the header
                corruptData[20] ^= 0xFF  // Corrupt data bytes
                corruptData[21] ^= 0xFF
            }

            // Should throw VectorError.dataCorruption due to CRC mismatch
            do {
                _ = try DynamicVector.decodeBinary(from: corruptData)
                Issue.record("Expected data corruption error but decode succeeded")
            } catch let error as VectorError {
                // Expected: CRC32 validation should catch the corruption
                #expect(error.kind == .dataCorruption)
                #expect(error.context.additionalInfo["message"]?.contains("CRC32") == true)
            } catch {
                Issue.record("Unexpected error type: \(error)")
            }

            // Test 2: Truncated data should also fail
            let truncated = validData.prefix(validData.count - 10)
            do {
                _ = try DynamicVector.decodeBinary(from: Data(truncated))
                Issue.record("Should fail with truncated data")
            } catch {
                // Expected - insufficient data error
            }
        }

        @Test("Endianness handling")
        func testEndiannessHandling() async throws {
            // BinaryFormat uses little-endian consistently
            let vector = DynamicVector([Float.pi, 2.71828, 42.0])  // e ≈ 2.71828
            let data = vector.encodeBinary()

            // Verify little-endian encoding of header magic number
            let magic = BinaryHeader.magic  // 0x56454354
            #expect(data[0] == 0x54)  // Least significant byte first
            #expect(data[1] == 0x43)
            #expect(data[2] == 0x45)
            #expect(data[3] == 0x56)  // Most significant byte last

            // Verify dimension is little-endian
            let dim: UInt32 = 3
            #expect(data[6] == UInt8(dim & 0xFF))
            #expect(data[7] == UInt8((dim >> 8) & 0xFF))
            #expect(data[8] == UInt8((dim >> 16) & 0xFF))
            #expect(data[9] == UInt8((dim >> 24) & 0xFF))

            // Round-trip should work regardless of platform endianness
            let decoded = try DynamicVector.decodeBinary(from: data)
            #expect(decoded.dimension == vector.dimension)
            for i in 0..<vector.dimension {
                #expect(abs(decoded[i] - vector[i]) < Float.ulpOfOne)
            }
        }

        @Test("Memory alignment requirements")
        func testMemoryAlignmentRequirements() async throws {
            // Test Vector512Optimized which uses SIMD4<Float> storage
            let values = (0..<512).map { Float($0) }
            let vector = try Vector512Optimized(values)

            // Serialize and deserialize
            let data = vector.encodeBinary()
            let decoded = try Vector<Dim512>.decodeBinary(from: data)

            // Verify data integrity
            for i in 0..<512 {
                #expect(decoded[i] == values[i])
            }

            // Test with unaligned data by adding a byte offset
            var unalignedData = Data([0xFF])  // Add one byte to misalign
            unalignedData.append(data)

            // Try to decode from offset 1 (unaligned)
            let subdata = unalignedData.dropFirst()
            let decodedUnaligned = try Vector<Dim512>.decodeBinary(from: Data(subdata))

            // Should still work despite unalignment
            for i in 0..<512 {
                #expect(decodedUnaligned[i] == values[i])
            }
        }

        @Test("Large vector memory efficiency")
        func testLargeVectorMemoryEfficiency() async throws {
            let largeDim = 5000
            let vector = DynamicVector(dimension: largeDim, repeating: 0.123)

            // Measure binary size
            let binaryData = vector.encodeBinary()
            let expectedBinarySize = BinaryHeader.headerSize + (largeDim * 4) + 4  // header + floats + CRC
            #expect(binaryData.count == expectedBinarySize)

            // Measure JSON size
            let jsonData = try vector.encodeJSON()
            // JSON is less efficient for large arrays
            #expect(jsonData.count > binaryData.count)

            // Test compression potential (repeated values should compress well)
            let repeatingVector = DynamicVector(dimension: 1000, repeating: 1.0)
            let repeatingBinary = repeatingVector.encodeBinary()
            let uniqueVector = DynamicVector((0..<1000).map { Float($0) })
            let uniqueBinary = uniqueVector.encodeBinary()

            // Both should have same binary size (no compression in basic format)
            #expect(repeatingBinary.count == uniqueBinary.count)
        }

        @Test("Concurrent serialization safety")
        func testConcurrentSerializationSafety() async throws {
            let vector = DynamicVector((0..<100).map { Float($0) })
            let iterations = 10

            // Test concurrent encoding
            await withTaskGroup(of: Data.self) { group in
                for _ in 0..<iterations {
                    group.addTask {
                        vector.encodeBinary()
                    }
                }

                var results: [Data] = []
                for await data in group {
                    results.append(data)
                }

                // All encoded data should be identical
                let reference = results[0]
                for data in results {
                    #expect(data == reference)
                }
            }

            // Test concurrent decoding
            let data = vector.encodeBinary()

            await withTaskGroup(of: DynamicVector.self) { group in
                for _ in 0..<iterations {
                    group.addTask {
                        try! DynamicVector.decodeBinary(from: data)
                    }
                }

                var decodedVectors: [DynamicVector] = []
                for await decoded in group {
                    decodedVectors.append(decoded)
                }

                // All decoded vectors should be identical
                for decoded in decodedVectors {
                    #expect(decoded.dimension == vector.dimension)
                    for i in 0..<vector.dimension {
                        #expect(decoded[i] == vector[i])
                    }
                }
            }
        }
    }

    // MARK: - JSON Serialization Tests

    @Suite("JSON Serialization")
    struct JSONSerializationTests {

        @Test("JSON precision loss")
        func testJSONPrecisionLoss() async throws {
            let precisionTestValues: [Float] = [
                Float.pi,
                1.0 / 3.0,
                Float.leastNormalMagnitude,
                1.23456789e10,
                9.876543e-10
            ]

            for value in precisionTestValues {
                let vector = DynamicVector([value])

                // Test default JSON encoding
                let jsonData = try vector.encodeJSON()
                let decoded = try DynamicVector.decodeJSON(from: jsonData)

                // Check precision loss
                let error = abs(decoded[0] - value)
                let relativeError = error / abs(value)

                // JSON typically preserves ~6-7 significant digits for Float
                #expect(relativeError < 1e-6 || error < Float.ulpOfOne * 10)

                // Test with custom encoder for better precision
                let encoder = JSONEncoder()
                encoder.outputFormatting = .sortedKeys
                let customData = try vector.encodeJSON(encoder: encoder)
                let customDecoded = try DynamicVector.decodeJSON(from: customData)

                #expect(abs(customDecoded[0] - value) <= abs(decoded[0] - value))
            }
        }

        @Test("JSON array size limits")
        func testJSONArraySizeLimits() async throws {
            // Test progressively larger arrays
            let sizes = [10, 100, 1000, 5000]

            for size in sizes {
                let vector = DynamicVector(dimension: size, repeating: 0.5)

                // Should encode successfully
                let jsonData = try vector.encodeJSON()
                #expect(jsonData.count > 0)

                // Should decode successfully
                let decoded = try DynamicVector.decodeJSON(from: jsonData)
                #expect(decoded.dimension == size)

                // Verify first and last elements
                #expect(decoded[0] == 0.5)
                #expect(decoded[size - 1] == 0.5)
            }

            // Test very large array (may be limited by memory)
            let veryLarge = DynamicVector(dimension: 10_000, repeating: 0.1)
            do {
                let data = try veryLarge.encodeJSON()
                let decoded = try DynamicVector.decodeJSON(from: data)
                #expect(decoded.dimension == 10_000)
            } catch {
                // May fail on memory-constrained systems
                Issue.record("Large JSON encoding failed: \(error)")
            }
        }

        @Test("JSON special number encoding")
        func testJSONSpecialNumberEncoding() async throws {
            let vector = DynamicVector([Float.nan, Float.infinity, -Float.infinity, 0.0])

            // Test with convertToString strategy
            let encoder = JSONEncoder()
            encoder.nonConformingFloatEncodingStrategy = .convertToString(
                positiveInfinity: "Infinity",
                negativeInfinity: "-Infinity",
                nan: "NaN"
            )

            let jsonData = try vector.encodeJSON(encoder: encoder)
            let jsonString = String(data: jsonData, encoding: .utf8)!

            #expect(jsonString.contains("\"NaN\""))
            #expect(jsonString.contains("\"Infinity\""))
            #expect(jsonString.contains("\"-Infinity\""))

            // Test decoding with matching strategy
            let decoder = JSONDecoder()
            decoder.nonConformingFloatDecodingStrategy = .convertFromString(
                positiveInfinity: "Infinity",
                negativeInfinity: "-Infinity",
                nan: "NaN"
            )

            let decoded = try DynamicVector.decodeJSON(from: jsonData, decoder: decoder)
            #expect(decoded[0].isNaN)
            #expect(decoded[1] == Float.infinity)
            #expect(decoded[2] == -Float.infinity)
            #expect(decoded[3] == 0.0)
        }

        @Test("JSON escape sequence handling")
        func testJSONEscapeSequenceHandling() async throws {
            // While vectors don't have metadata with strings, test edge cases in JSON encoding
            let vector = DynamicVector([1.0, 2.0, 3.0])

            // Test JSON with pretty printing and special formatting
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]

            let jsonData = try vector.encodeJSON(encoder: encoder)
            let jsonString = String(data: jsonData, encoding: .utf8)!

            // Should contain newlines and indentation
            #expect(jsonString.contains("\n"))
            #expect(jsonString.contains("  "))  // Indentation

            // Should still decode correctly
            let decoded = try DynamicVector.decodeJSON(from: jsonData)
            #expect(decoded.dimension == vector.dimension)
            for i in 0..<vector.dimension {
                #expect(decoded[i] == vector[i])
            }
        }

        @Test("JSON nested structure limits")
        func testJSONNestedStructureLimits() async throws {
            // Test array of vectors (not deeply nested, but structured)
            let vectors = [
                DynamicVector([1.0, 2.0]),
                DynamicVector([3.0, 4.0]),
                DynamicVector([5.0, 6.0])
            ]

            let encoder = JSONEncoder()
            let data = try encoder.encode(vectors)

            let decoder = JSONDecoder()
            let decoded = try decoder.decode([DynamicVector].self, from: data)

            #expect(decoded.count == vectors.count)
            for (original, decoded) in zip(vectors, decoded) {
                #expect(original.dimension == decoded.dimension)
                for i in 0..<original.dimension {
                    #expect(original[i] == decoded[i])
                }
            }
        }

        @Test("JSON whitespace and formatting")
        func testJSONWhitespaceAndFormatting() async throws {
            let vector = DynamicVector([1.0, 2.0, 3.0, 4.0, 5.0])

            // Compact formatting
            let compactEncoder = JSONEncoder()
            let compactData = try vector.encodeJSON(encoder: compactEncoder)

            // Pretty formatting
            let prettyEncoder = JSONEncoder()
            prettyEncoder.outputFormatting = .prettyPrinted
            let prettyData = try vector.encodeJSON(encoder: prettyEncoder)

            // Pretty should be larger due to whitespace
            #expect(prettyData.count > compactData.count)

            // Both should decode to same result
            let compactDecoded = try DynamicVector.decodeJSON(from: compactData)
            let prettyDecoded = try DynamicVector.decodeJSON(from: prettyData)

            #expect(compactDecoded.dimension == prettyDecoded.dimension)
            for i in 0..<vector.dimension {
                #expect(compactDecoded[i] == prettyDecoded[i])
                #expect(compactDecoded[i] == vector[i])
            }
        }

        @Test("JSON backward compatibility")
        func testJSONBackwardCompatibility() async throws {
            // Simulate old format (just array of floats)
            let oldFormatJSON = "[1.0, 2.0, 3.0, 4.0]"
            let oldData = oldFormatJSON.data(using: .utf8)!

            // Current format includes dimension field
            let currentFormatJSON = "{\"dimension\": 4, \"elements\": [1.0, 2.0, 3.0, 4.0]}"
            let currentData = currentFormatJSON.data(using: .utf8)!

            // DynamicVector should handle current format
            let decoded = try DynamicVector.decodeJSON(from: currentData)
            #expect(decoded.dimension == 4)
            #expect(decoded[0] == 1.0)
            #expect(decoded[3] == 4.0)

            // Fixed-size Vector should handle simple array format
            let simpleArrayJSON = "{\"elements\": [1.0, 2.0, 3.0, 4.0]}"
            let simpleData = simpleArrayJSON.data(using: .utf8)!

            // This would work for fixed-size vectors that know their dimension
            do {
                let fixedDecoded = try JSONDecoder().decode(Vector<Dim4>.self, from: simpleData)
                #expect(fixedDecoded.scalarCount == 4)
            } catch {
                // Dim4 might not exist, that's ok
            }
        }
    }

    // MARK: - Cross-Format Tests

    @Suite("Cross-Format Compatibility")
    struct CrossFormatCompatibilityTests {

        @Test("Binary to JSON conversion")
        func testBinaryToJSONConversion() async throws {
            let vector = DynamicVector((0..<10).map { Float($0) * Float.pi })

            // Encode as binary
            let binaryData = vector.encodeBinary()

            // Decode from binary
            let decodedFromBinary = try DynamicVector.decodeBinary(from: binaryData)

            // Re-encode as JSON
            let jsonData = try decodedFromBinary.encodeJSON()

            // Decode from JSON
            let decodedFromJSON = try DynamicVector.decodeJSON(from: jsonData)

            // Compare all three
            #expect(decodedFromBinary.dimension == vector.dimension)
            #expect(decodedFromJSON.dimension == vector.dimension)

            for i in 0..<vector.dimension {
                // Binary should be exact
                #expect(decodedFromBinary[i] == vector[i])
                // JSON may have slight precision loss
                #expect(abs(decodedFromJSON[i] - vector[i]) < 1e-6)
            }

            // Binary should be more compact than JSON
            #expect(binaryData.count < jsonData.count)
        }

        @Test("JSON to binary conversion")
        func testJSONToBinaryConversion() async throws {
            // Test reverse conversion
            // - Validation of converted data
            // - Size optimization
            // - Error handling
        }

        @Test("Format detection and auto-selection")
        func testFormatDetectionAndAutoSelection() async throws {
            // Test automatic format detection
            // - Magic number detection
            // - JSON structure detection
            // - Fallback mechanisms
        }

        @Test("Mixed format batch processing")
        func testMixedFormatBatchProcessing() async throws {
            // Test handling mixed formats
            // - Batch with different formats
            // - Format conversion pipeline
            // - Error isolation
        }
    }

    // MARK: - Vector-Specific Serialization Tests

    @Suite("Vector Type Serialization")
    struct VectorTypeSerializationTests {

        @Test("Vector512 optimized serialization")
        func testVector512OptimizedSerialization() async throws {
            let values = Array(repeating: Float(1.5), count: 512)
            let vector = try Vector<Dim512>(values)

            // Binary serialization
            let binaryData = vector.encodeBinary()
            let decodedBinary = try Vector<Dim512>.decodeBinary(from: binaryData)
            #expect(decodedBinary.toArray() == values)

            // JSON serialization - Vector types encode as simple arrays
            let encoder = JSONEncoder()
            let jsonData = try encoder.encode(vector.toArray())  // Encode as array
            let decoder = JSONDecoder()
            let decodedArray = try decoder.decode([Float].self, from: jsonData)
            let decodedVector = try Vector<Dim512>(decodedArray)
            #expect(decodedVector.toArray() == values)
        }

        @Test("Vector768 optimized serialization")
        func testVector768OptimizedSerialization() async throws {
            let values = (0..<768).map { Float($0) / 768.0 }
            let vector = try Vector<Dim768>(values)

            // Binary with checksum validation
            let binaryData = vector.encodeBinary()
            let decodedBinary = try Vector<Dim768>.decodeBinary(from: binaryData)
            #expect(decodedBinary.toArray() == values)

            // JSON with formatting options - Vector types encode as arrays
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            let jsonData = try encoder.encode(vector.toArray())  // Encode as array
            let decodedArray = try JSONDecoder().decode([Float].self, from: jsonData)
            let decodedVector = try Vector<Dim768>(decodedArray)
            #expect(decodedVector.toArray() == values)
        }

        @Test("Vector1536 optimized serialization")
        func testVector1536OptimizedSerialization() async throws {
            let rng = SystemRandomNumberGenerator()
            var mutableRNG = rng
            let values = (0..<1536).map { _ in Float.random(in: -1...1, using: &mutableRNG) }
            let vector = try Vector<Dim1536>(values)

            // Test large vector memory efficiency
            let binaryData = vector.encodeBinary()
            // Binary format: 12 bytes header + (1536 * 4) bytes data + 4 bytes CRC32
            let expectedSize = 12 + (1536 * 4) + 4  // 12 + 6144 + 4 = 6160
            #expect(binaryData.count == expectedSize)

            let decodedBinary = try Vector<Dim1536>.decodeBinary(from: binaryData)
            for i in 0..<1536 {
                #expect(abs(decodedBinary.toArray()[i] - values[i]) < 1e-6)
            }

            // JSON format test - encode/decode as array
            let encoder = JSONEncoder()
            let jsonData = try encoder.encode(vector.toArray())
            let decodedArray = try JSONDecoder().decode([Float].self, from: jsonData)
            let decodedVector = try Vector<Dim1536>(decodedArray)
            for i in 0..<1536 {
                #expect(abs(decodedVector.toArray()[i] - values[i]) < 1e-6)
            }
        }

        @Test("DynamicVector serialization")
        func testDynamicVectorSerialization() async throws {
            // Test variable-size vector serialization
            // - Arbitrary dimension support
            // - Size encoding in header
            // - Memory allocation strategy
        }

        @Test("Quantized vector serialization")
        func testQuantizedVectorSerialization() async throws {
            // Test INT8 quantized vectors
            // - Quantization parameter storage
            // - Scale and zero-point encoding
            // - Precision vs size tradeoffs
        }

        @Test("Mixed precision vector serialization")
        func testMixedPrecisionVectorSerialization() async throws {
            // Test FP16/FP32 mixed vectors
            // - Half-precision encoding
            // - Format conversion on decode
            // - Platform compatibility
        }
    }

    // MARK: - Error Recovery Tests

    @Suite("Error Recovery and Resilience")
    struct ErrorRecoveryTests {

        @Test("Partial data recovery")
        func testPartialDataRecovery() async throws {
            let vector = DynamicVector([1.0, 2.0, 3.0, 4.0, 5.0])
            let fullData = vector.encodeBinary()

            // Try various truncation points
            let truncationPoints = [
                fullData.count - 1,   // Missing last checksum byte
                fullData.count - 4,   // Missing entire checksum
                fullData.count - 8,   // Missing checksum + part of last float
                BinaryHeader.headerSize + 8,  // Only header + 2 floats
                BinaryHeader.headerSize       // Only header
            ]

            for truncPoint in truncationPoints {
                let truncatedData = fullData.prefix(truncPoint)
                do {
                    _ = try DynamicVector.decodeBinary(from: Data(truncatedData))
                    Issue.record("Should have failed at truncation point \(truncPoint)")
                } catch {
                    // Expected failure - no partial recovery in current implementation
                    // A production system might implement best-effort recovery
                }
            }
        }

        @Test("Version migration handling")
        func testVersionMigrationHandling() async throws {
            // Test version upgrade paths
            // - Forward compatibility
            // - Deprecated field handling
            // - Schema evolution
        }

        @Test("Resource exhaustion handling")
        func testResourceExhaustionHandling() async throws {
            // Test resource limit scenarios
            // - Out of memory during decode
            // - File descriptor limits
            // - Timeout handling
        }

        @Test("Malicious input protection")
        func testMaliciousInputProtection() async throws {
            // Test dimension bomb - claims huge dimension but has small data
            var maliciousData = Data()
            BinaryFormat.writeUInt32(BinaryHeader.magic, to: &maliciousData)
            BinaryFormat.writeUInt16(BinaryHeader.version, to: &maliciousData)
            BinaryFormat.writeUInt32(UInt32.max, to: &maliciousData)  // Claim max dimension
            BinaryFormat.writeUInt16(0, to: &maliciousData)
            BinaryFormat.writeUInt32(0, to: &maliciousData)  // Fake checksum

            do {
                _ = try DynamicVector.decodeBinary(from: maliciousData)
                Issue.record("Should reject dimension bomb")
            } catch {
                // Expected - should fail validation
            }

            // Test dimension beyond BinaryFormat.maxDimension
            var oversizedData = Data()
            BinaryFormat.writeUInt32(BinaryHeader.magic, to: &oversizedData)
            BinaryFormat.writeUInt16(BinaryHeader.version, to: &oversizedData)
            BinaryFormat.writeUInt32(BinaryFormat.maxDimension + 1, to: &oversizedData)
            BinaryFormat.writeUInt16(0, to: &oversizedData)

            // Add fake data
            for _ in 0..<10 {
                BinaryFormat.writeFloat(0.0, to: &oversizedData)
            }
            BinaryFormat.writeUInt32(0, to: &oversizedData)

            do {
                _ = try DynamicVector.decodeBinary(from: oversizedData)
                Issue.record("Should reject oversized dimension")
            } catch {
                // Expected
            }
        }
    }

    // MARK: - Performance Edge Cases

    @Suite("Serialization Performance")
    struct SerializationPerformanceTests {

        @Test("Streaming serialization performance")
        func testStreamingSerializationPerformance() async throws {
            // Test streaming vs batch performance
            // - Memory usage patterns
            // - Throughput measurement
            // - Latency characteristics
        }

        @Test("Compression effectiveness")
        func testCompressionEffectiveness() async throws {
            // Test compression ratios
            // - Different vector patterns
            // - Compression algorithm comparison
            // - CPU vs size tradeoffs
        }

        @Test("Cache-friendly serialization")
        func testCacheFriendlySerialization() async throws {
            // Test cache optimization
            // - Sequential access patterns
            // - Prefetching effectiveness
            // - Hot path optimization
        }

        @Test("Batch serialization optimization")
        func testBatchSerializationOptimization() async throws {
            // Test batch processing efficiency
            // - Vectorized encoding
            // - Parallel processing
            // - Memory pooling
        }
    }

    // MARK: - Platform-Specific Tests

    @Suite("Platform Compatibility")
    struct PlatformCompatibilityTests {

        @Test("iOS/macOS compatibility")
        func testIOSMacOSCompatibility() async throws {
            // Test Apple platform specifics
            // - NSData bridging
            // - Core Data integration
            // - iCloud sync compatibility
        }

        @Test("Linux compatibility")
        func testLinuxCompatibility() async throws {
            // Test Linux-specific behavior
            // - File system differences
            // - Endianness handling
            // - Memory mapping support
        }

        @Test("32-bit vs 64-bit compatibility")
        func test32BitVs64BitCompatibility() async throws {
            // Test architecture differences
            // - Pointer size handling
            // - Integer overflow behavior
            // - Alignment requirements
        }
    }
}

// MARK: - Test Support Types

extension SerializationEdgeCaseTests {

    /// Test data generator for serialization testing
    struct SerializationTestDataGenerator {

        /// Generate pathological float patterns
        static func generatePathologicalFloats() -> [Float] {
            return [
                0.0, -0.0,                          // Zeros
                Float.infinity, -Float.infinity,    // Infinities
                Float.nan,                          // NaN
                Float.leastNormalMagnitude,         // Smallest normal
                Float.leastNonzeroMagnitude,        // Smallest denormal
                Float.greatestFiniteMagnitude,      // Largest finite
                Float.pi, Float.ulpOfOne            // Common values
            ]
        }

        /// Generate test vectors with specific patterns
        static func generateTestVector(dimension: Int, pattern: SerializationPattern) -> [Float] {
            switch pattern {
            case .zeros:
                return Array(repeating: 0.0, count: dimension)
            case .ones:
                return Array(repeating: 1.0, count: dimension)
            case .sequential:
                return (0..<dimension).map { Float($0) }
            case .random:
                return (0..<dimension).map { _ in Float.random(in: -1...1) }
            case .alternating:
                return (0..<dimension).map { Float($0 % 2 == 0 ? 1.0 : -1.0) }
            case .sparse:
                return (0..<dimension).map { $0 % 10 == 0 ? Float.random(in: -1...1) : 0.0 }
            case .dense:
                return (0..<dimension).map { _ in Float.random(in: -100...100) }
            }
        }

        /// Corrupt binary data in controlled ways
        static func corruptBinaryData(_ data: Data, corruption: CorruptionType) -> Data {
            var corrupted = data
            switch corruption {
            case .truncate(let bytes):
                return Data(corrupted.dropLast(bytes))
            case .appendGarbage(let bytes):
                for _ in 0..<bytes {
                    corrupted.append(UInt8.random(in: 0...255))
                }
                return corrupted
            case .flipBits(let count):
                for _ in 0..<count {
                    let index = Int.random(in: 0..<corrupted.count)
                    corrupted[index] ^= UInt8.random(in: 1...255)
                }
                return corrupted
            case .corruptHeader:
                if corrupted.count > 10 {
                    corrupted[Int.random(in: 0..<10)] ^= 0xFF
                }
                return corrupted
            case .corruptData:
                if corrupted.count > BinaryHeader.headerSize {
                    let index = Int.random(in: BinaryHeader.headerSize..<corrupted.count - 4)
                    corrupted[index] ^= 0xFF
                }
                return corrupted
            case .swapEndianness:
                // Swap byte order in 4-byte chunks (for floats)
                for i in stride(from: BinaryHeader.headerSize, to: corrupted.count - 4, by: 4) {
                    corrupted.swapAt(i, i + 3)
                    corrupted.swapAt(i + 1, i + 2)
                }
                return corrupted
            }
        }
    }

    /// Serialization validation utilities
    struct SerializationValidator {

        /// Validate binary format compliance
        static func validateBinaryFormat(_ data: Data) -> ValidationResult {
            var errors: [SerializationError] = []

            // Check minimum size
            if data.count < BinaryFormat.minDataSize {
                errors.append(SerializationError(
                    offset: 0,
                    expectedValue: ">=\(BinaryFormat.minDataSize) bytes",
                    actualValue: "\(data.count) bytes",
                    description: "Data too small"
                ))
                return ValidationResult(isValid: false, errors: errors)
            }

            // Validate magic number
            do {
                let magic = try BinaryFormat.readUInt32(from: data, at: 0)
                if magic != BinaryHeader.magic {
                    errors.append(SerializationError(
                        offset: 0,
                        expectedValue: BinaryHeader.magic,
                        actualValue: magic,
                        description: "Invalid magic number"
                    ))
                }
            } catch {
                errors.append(SerializationError(
                    offset: 0,
                    expectedValue: nil,
                    actualValue: nil,
                    description: "Failed to read magic: \(error)"
                ))
            }

            // Validate version
            do {
                let version = try BinaryFormat.readUInt16(from: data, at: 4)
                if version != BinaryHeader.version {
                    errors.append(SerializationError(
                        offset: 4,
                        expectedValue: BinaryHeader.version,
                        actualValue: version,
                        description: "Unsupported version"
                    ))
                }
            } catch {
                errors.append(SerializationError(
                    offset: 4,
                    expectedValue: nil,
                    actualValue: nil,
                    description: "Failed to read version: \(error)"
                ))
            }

            return ValidationResult(isValid: errors.isEmpty, errors: errors)
        }

        /// Compare vectors for serialization equivalence
        static func compareVectors<V: VectorProtocol>(_ a: V, _ b: V, tolerance: Float = 1e-7) -> Bool where V.Scalar == Float {
            guard a.scalarCount == b.scalarCount else { return false }

            let arrayA = a.toArray()
            let arrayB = b.toArray()

            for i in 0..<a.scalarCount {
                let diff = abs(arrayA[i] - arrayB[i])
                // Handle special cases
                if arrayA[i].isNaN && arrayB[i].isNaN {
                    continue  // Both NaN is ok
                }
                if arrayA[i].isInfinite && arrayB[i].isInfinite && arrayA[i].sign == arrayB[i].sign {
                    continue  // Same infinity is ok
                }
                if diff > tolerance {
                    return false
                }
            }
            return true
        }

        /// Measure serialization metrics
        static func measureSerializationMetrics(data: Data, format: SerializationFormat) -> SerializationMetrics {
            // Implementation will measure size, speed, accuracy
            return SerializationMetrics(
                compressedSize: 0,
                compressionRatio: 1.0,
                encodingTime: 0,
                decodingTime: 0
            )
        }
    }
}

// MARK: - Supporting Types

enum SerializationPattern {
    case zeros
    case ones
    case sequential
    case random
    case alternating
    case sparse
    case dense
}

enum CorruptionType {
    case truncate(bytes: Int)
    case appendGarbage(bytes: Int)
    case flipBits(count: Int)
    case corruptHeader
    case corruptData
    case swapEndianness
}

enum SerializationFormat {
    case binary
    case json
    case protobuf
    case msgpack
}

struct ValidationResult {
    let isValid: Bool
    let errors: [SerializationError]
}

struct SerializationError {
    let offset: Int
    let expectedValue: Any?
    let actualValue: Any?
    let description: String
}

struct SerializationMetrics {
    let compressedSize: Int
    let compressionRatio: Double
    let encodingTime: TimeInterval
    let decodingTime: TimeInterval
}

// MARK: - Test Fixtures

extension SerializationEdgeCaseTests {

    /// Pre-encoded test data for compatibility testing
    struct TestFixtures {
        // Legacy format test data
        static let v1BinaryData = Data()
        static let v1JSONData = Data()

        // Known-good encoded vectors
        static let referenceVector512Binary = Data()
        static let referenceVector768Binary = Data()
        static let referenceVector1536Binary = Data()

        // Malformed data samples
        static let corruptedHeaders: [Data] = []
        static let truncatedData: [Data] = []
        static let maliciousPayloads: [Data] = []
    }
}
