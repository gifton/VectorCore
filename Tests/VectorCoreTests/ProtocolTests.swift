// VectorCoreTests: Protocol Tests
//
// Tests for VectorCore protocols
//

import XCTest
@testable import VectorCore

final class ProtocolTests: XCTestCase {
    
    // MARK: - VectorProtocol Tests
    
    func testVectorProtocolBasicConformance() {
        // Test that init(from:) works correctly
        let array = [Float](repeating: 1.5, count: 256)
        let vector = Vector256(from: array)
        
        for i in 0..<256 {
            XCTAssertEqual(vector[i], 1.5)
        }
        
        // Test toArray()
        let resultArray = vector.toArray()
        XCTAssertEqual(resultArray.count, 256)
        XCTAssertEqual(resultArray, array)
    }
    
    func testVectorProtocolDimensions() {
        // Verify dimensions are correct for all types
        XCTAssertEqual(Vector128.dimensions, 128)
        XCTAssertEqual(Vector256.dimensions, 256)
        XCTAssertEqual(Vector512.dimensions, 512)
        XCTAssertEqual(Vector768.dimensions, 768)
        XCTAssertEqual(Vector1536.dimensions, 1536)
        XCTAssertEqual(SIMD32<Float>.dimensions, 32)
        XCTAssertEqual(SIMD64<Float>.dimensions, 64)
    }
    
    func testVectorProtocolValidation() {
        // Test validation works correctly
        XCTAssertTrue(Vector128.validate([Float](repeating: 0, count: 128)))
        XCTAssertFalse(Vector128.validate([Float](repeating: 0, count: 127)))
        XCTAssertFalse(Vector128.validate([Float](repeating: 0, count: 129)))
        XCTAssertFalse(Vector128.validate([]))
    }
    
    func testVectorProtocolOptionalCreation() {
        // Test create(from:) with valid array
        let validArray = [Float](repeating: 2.0, count: 512)
        if let vector = Vector512.create(from: validArray) {
            XCTAssertEqual(vector[0], 2.0)
            XCTAssertEqual(vector[511], 2.0)
        } else {
            XCTFail("Failed to create vector from valid array")
        }
        
        // Test create(from:) with invalid array
        let invalidArray = [Float](repeating: 2.0, count: 500)
        let invalidVector = Vector512.create(from: invalidArray)
        XCTAssertNil(invalidVector)
    }
    
    func testVectorProtocolRepeatingInitializer() {
        // Test the convenience initializer
        let vector: Vector256 = Vector256(repeating: 3.14)
        for i in 0..<256 {
            XCTAssertEqual(vector[i], 3.14)
        }
    }
    
    // MARK: - VectorSerializable Tests
    
    func testVector128Serialization() {
        let original = Vector128(repeating: 2.5)
        
        // Test serialization
        do {
            let serialized = try original.serialize()
            XCTAssertEqual(serialized.dimensions, 128)
            XCTAssertEqual(serialized.data.count, 128 * MemoryLayout<Float>.size)
            
            // Test deserialization
            let deserialized = try Vector128.deserialize(from: serialized)
            for i in 0..<128 {
                XCTAssertEqual(deserialized[i], original[i])
            }
        } catch {
            XCTFail("Serialization failed: \(error)")
        }
    }
    
    func testVector512SerializationWithData() {
        // Create vector with specific pattern
        let values = (0..<512).map { Float($0) / 512.0 }
        let original = Vector512(values)
        
        do {
            // Serialize to data
            let serialized = try original.serialize()
            
            // Verify data integrity
            XCTAssertEqual(serialized.dimensions, 512)
            
            // Deserialize and verify
            let deserialized = try Vector512.deserialize(from: serialized)
            for i in 0..<512 {
                XCTAssertEqual(deserialized[i], original[i], accuracy: 0.0001)
            }
        } catch {
            XCTFail("Serialization failed: \(error)")
        }
    }
    
    func testVectorSerializationErrors() {
        // Test dimension mismatch error
        let wrongDimensions = VectorSerialization.DataForm(
            dimensions: 256,
            data: Data(count: 128 * MemoryLayout<Float>.size) // Wrong size
        )
        
        XCTAssertThrowsError(try Vector256.deserialize(from: wrongDimensions)) { error in
            if let coreError = error as? VectorCoreError {
                XCTAssertEqual(coreError.code, "INVALID_PARAMETER")
            } else {
                XCTFail("Expected VectorCoreError")
            }
        }
        
        // Test wrong dimension count
        let wrongDimensionCount = VectorSerialization.DataForm(
            dimensions: 128, // Wrong dimension
            data: Data(count: 256 * MemoryLayout<Float>.size)
        )
        
        XCTAssertThrowsError(try Vector256.deserialize(from: wrongDimensionCount)) { error in
            if let coreError = error as? VectorCoreError {
                XCTAssertEqual(coreError.code, "DIMENSION_MISMATCH")
            } else {
                XCTFail("Expected VectorCoreError")
            }
        }
    }
    
    func testSIMDVectorSerialization() {
        // Test SIMD32 serialization
        let values32 = (0..<32).map { Float($0) }
        let simd32 = SIMD32<Float>(from: values32)
        
        do {
            let serialized = try simd32.serialize()
            XCTAssertEqual(serialized.dimensions, 32)
            XCTAssertEqual(serialized.values, values32)
            
            let deserialized = try SIMD32<Float>.deserialize(from: serialized)
            XCTAssertEqual(simd32, deserialized)
        } catch {
            XCTFail("SIMD32 serialization failed: \(error)")
        }
        
        // Test SIMD64 serialization
        let values64 = (0..<64).map { Float($0) * 0.5 }
        let simd64 = SIMD64<Float>(from: values64)
        
        do {
            let serialized = try simd64.serialize()
            XCTAssertEqual(serialized.dimensions, 64)
            XCTAssertEqual(serialized.values, values64)
            
            let deserialized = try SIMD64<Float>.deserialize(from: serialized)
            XCTAssertEqual(simd64, deserialized)
        } catch {
            XCTFail("SIMD64 serialization failed: \(error)")
        }
    }
    
    func testJSONSerialization() {
        let vector = Vector256(repeating: 1.23)
        
        do {
            // Serialize to JSON
            let jsonData = try vector.serializeToJSON()
            XCTAssertGreaterThan(jsonData.count, 0)
            
            // Deserialize from JSON
            let deserialized = try Vector256.deserializeFromJSON(jsonData)
            
            // Verify
            for i in 0..<256 {
                XCTAssertEqual(deserialized[i], 1.23, accuracy: 0.0001)
            }
        } catch {
            XCTFail("JSON serialization failed: \(error)")
        }
    }
    
    // MARK: - DistanceMetric Protocol Tests
    
    func testDistanceMetricProtocol() {
        let metrics: [any DistanceMetric] = [
            EuclideanDistance(),
            CosineDistance(),
            DotProductDistance(),
            ManhattanDistance(),
            HammingDistance(),
            ChebyshevDistance(),
            MinkowskiDistance(p: 3),
            JaccardDistance()
        ]
        
        let v1 = SIMD4<Float>(1, 2, 3, 4)
        let v2 = SIMD4<Float>(5, 6, 7, 8)
        
        for metric in metrics {
            // Test identifier exists
            XCTAssertFalse(metric.identifier.isEmpty)
            
            // Test basic distance computation
            let distance = metric.distance(v1, v2)
            XCTAssertFalse(distance.isNaN)
            XCTAssertFalse(distance.isInfinite)
            
            // Test batch distance
            let batch = metric.batchDistance(query: v1, candidates: [v1, v2])
            XCTAssertEqual(batch.count, 2)
            XCTAssertEqual(batch[0], metric.distance(v1, v1))
            XCTAssertEqual(batch[1], distance)
        }
    }
    
    // MARK: - AccelerationProvider Protocol Tests
    
    // Mock implementation for testing
    struct MockAccelerationProvider: AccelerationProvider {
        typealias Config = [String: Any]
        
        let supportedOperations: Set<AcceleratedOperation>
        
        init(configuration: Config) async throws {
            if let supported = configuration["supported"] as? [String] {
                self.supportedOperations = Set(supported.compactMap { AcceleratedOperation(rawValue: $0) })
            } else {
                self.supportedOperations = []
            }
        }
        
        func isSupported(for operation: AcceleratedOperation) -> Bool {
            return supportedOperations.contains(operation)
        }
        
        func accelerate<T>(_ operation: AcceleratedOperation, input: T) async throws -> T {
            guard isSupported(for: operation) else {
                throw VectorCoreError.notImplemented(feature: "Operation \(operation.rawValue)")
            }
            // Mock implementation just returns input
            return input
        }
    }
    
    func testAccelerationProviderProtocol() async throws {
        let config: [String: Any] = [
            "supported": ["distanceComputation", "matrixMultiplication"]
        ]
        
        let provider = try await MockAccelerationProvider(configuration: config)
        
        // Test supported operations
        XCTAssertTrue(provider.isSupported(for: .distanceComputation))
        XCTAssertTrue(provider.isSupported(for: .matrixMultiplication))
        XCTAssertFalse(provider.isSupported(for: .vectorNormalization))
        XCTAssertFalse(provider.isSupported(for: .batchedOperations))
        
        // Test acceleration
        let input = [1, 2, 3, 4]
        let result = try await provider.accelerate(.distanceComputation, input: input)
        XCTAssertEqual(result, input)
        
        // Test unsupported operation
        do {
            _ = try await provider.accelerate(.vectorNormalization, input: input)
            XCTFail("Expected error for unsupported operation")
        } catch {
            // Expected
        }
    }
    
    // MARK: - Type Safety Tests
    
    func testProtocolTypeSafety() {
        // Ensure protocols maintain type safety
        
        // VectorProtocol requires Scalar == Float
        let floatVector = Vector256(repeating: 1.0)
        XCTAssertTrue(type(of: floatVector).Scalar.self == Float.self)
        
        // Distance metrics work with any SIMD type
        let metric = EuclideanDistance()
        let dist1 = metric.distance(SIMD2<Float>(1, 2), SIMD2<Float>(3, 4))
        let dist2 = metric.distance(SIMD4<Float>(1, 2, 3, 4), SIMD4<Float>(5, 6, 7, 8))
        
        XCTAssertGreaterThan(dist1, 0)
        XCTAssertGreaterThan(dist2, 0)
    }
}