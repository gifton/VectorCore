// VectorCore: Core Protocols Tests
//
// Comprehensive tests for protocol conformance and behavior
//

import XCTest
@testable import VectorCore

// MARK: - Test Types

/// Test type conforming to BaseVectorProtocol
struct TestBaseVector: BaseVectorProtocol {
    typealias Scalar = Float
    static let dimensions = 4
    
    private var storage: [Float]
    
    init(from array: [Float]) {
        precondition(array.count == Self.dimensions)
        self.storage = array
    }
    
    var scalarCount: Int { Self.dimensions }
    
    func toArray() -> [Float] {
        storage
    }
    
    subscript(index: Int) -> Float {
        get { storage[index] }
        set { storage[index] = newValue }
    }
}

/// Test type conforming to ExtendedVectorProtocol
struct TestExtendedVector: ExtendedVectorProtocol {
    typealias Scalar = Float
    static let dimensions = 4
    
    private var storage: [Float]
    
    init(from array: [Float]) {
        precondition(array.count == Self.dimensions)
        self.storage = array
    }
    
    var scalarCount: Int { Self.dimensions }
    
    func toArray() -> [Float] {
        storage
    }
    
    subscript(index: Int) -> Float {
        get { storage[index] }
        set { storage[index] = newValue }
    }
    
    func dotProduct(_ other: Self) -> Float {
        return zip(storage, other.storage).map(*).reduce(0, +)
    }
    
    var magnitude: Float {
        sqrt(dotProduct(self))
    }
    
    func normalized() -> Self {
        let mag = magnitude
        guard mag > 0 else { return self }
        return Self(from: storage.map { $0 / mag })
    }
    
    func distance(to other: Self) -> Float {
        sqrt(zip(storage, other.storage).map { pow($0 - $1, 2) }.reduce(0, +))
    }
    
    func cosineSimilarity(to other: Self) -> Float {
        let dot = dotProduct(other)
        let mag1 = magnitude
        let mag2 = other.magnitude
        guard mag1 > 0 && mag2 > 0 else { return 0 }
        return dot / (mag1 * mag2)
    }
    
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }
}

// Note: BinaryEncodable protocol doesn't exist separately
// Binary encoding is provided through VectorProtocol extension

/// Test type conforming to BinaryDecodable
struct TestBinaryDecodable: BinaryDecodable, ExtendedVectorProtocol {
    typealias Scalar = Float
    static let dimensions = 4  // For testing
    
    let dimension: Int
    let values: [Float]
    
    init(dimension: Int, buffer: UnsafeBufferPointer<Float>) throws {
        guard buffer.count == dimension else {
            throw VectorError.dimensionMismatch(expected: dimension, actual: buffer.count)
        }
        self.dimension = dimension
        self.values = Array(buffer)
    }
    
    static func decodeBinary(from data: Data) throws -> Self {
        // Simple implementation for testing
        let floatCount = data.count / MemoryLayout<Float>.size
        var values = [Float](repeating: 0, count: floatCount)
        _ = values.withUnsafeMutableBytes { buffer in
            data.copyBytes(to: buffer)
        }
        return try values.withUnsafeBufferPointer { buffer in
            try Self(dimension: floatCount, buffer: buffer)
        }
    }
    
    init(from array: [Float]) {
        self.dimension = array.count
        self.values = array
    }
    
    // BaseVectorProtocol conformance
    var scalarCount: Int { dimension }
    func toArray() -> [Float] { values }
    
    subscript(index: Int) -> Float {
        get { values[index] }
    }
    
    // ExtendedVectorProtocol conformance
    func dotProduct(_ other: Self) -> Float {
        zip(values, other.values).map(*).reduce(0, +)
    }
    
    var magnitude: Float {
        sqrt(dotProduct(self))
    }
    
    func normalized() -> Self {
        let mag = magnitude
        guard mag > 0 else { return self }
        return Self(from: values.map { $0 / mag })
    }
    
    func distance(to other: Self) -> Float {
        sqrt(zip(values, other.values).map { pow($0 - $1, 2) }.reduce(0, +))
    }
    
    func cosineSimilarity(to other: Self) -> Float {
        let dot = dotProduct(other)
        let mag1 = magnitude
        let mag2 = other.magnitude
        guard mag1 > 0 && mag2 > 0 else { return 0 }
        return dot / (mag1 * mag2)
    }
    
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try values.withUnsafeBufferPointer(body)
    }
}

// MARK: - Protocol Tests

final class CoreProtocolsTests: XCTestCase {
    
    // MARK: - BaseVectorProtocol Tests
    
    func testBaseVectorProtocolConformance() {
        let values: [Float] = [1, 2, 3, 4]
        let vector = TestBaseVector(from: values)
        
        // Test scalarCount
        XCTAssertEqual(vector.scalarCount, 4)
        
        // Test toArray
        XCTAssertEqual(vector.toArray(), values)
        
        // Test subscript get
        for i in 0..<4 {
            XCTAssertEqual(vector[i], values[i])
        }
        
        // Test subscript set
        var mutableVector = vector
        mutableVector[2] = 10.0
        XCTAssertEqual(mutableVector[2], 10.0)
        XCTAssertEqual(mutableVector.toArray(), [1, 2, 10, 4])
    }
    
    func testBaseVectorProtocolWithBuiltinTypes() {
        // Test with Vector<Dim128>
        let vector128 = Vector<Dim128>(repeating: 1.0)
        XCTAssertEqual(vector128.scalarCount, 128)
        XCTAssertEqual(vector128[0], 1.0)
        
        // Test with DynamicVector
        let dynamicVector = DynamicVector([1, 2, 3, 4, 5])
        XCTAssertEqual(dynamicVector.scalarCount, 5)
        XCTAssertEqual(dynamicVector.toArray(), [1, 2, 3, 4, 5])
    }
    
    // MARK: - ExtendedVectorProtocol Tests
    
    func testExtendedVectorProtocolConformance() {
        let v1 = TestExtendedVector(from: [3, 4, 0, 0])
        let v2 = TestExtendedVector(from: [1, 2, 2, 1])
        
        // Test dotProduct
        let dot = v1.dotProduct(v2)
        XCTAssertEqual(dot, 11.0) // 3*1 + 4*2 + 0*2 + 0*1
        
        // Test magnitude
        XCTAssertEqual(v1.magnitude, 5.0) // sqrt(9 + 16 + 0 + 0)
        
        // Test normalized
        let normalized = v1.normalized()
        XCTAssertEqual(normalized.magnitude, 1.0, accuracy: 1e-6)
        XCTAssertEqual(normalized.toArray(), [0.6, 0.8, 0, 0])
        
        // Test distance
        let distance = v1.distance(to: v2)
        let expected = sqrt(pow(Float(3-1), 2) + pow(Float(4-2), 2) + pow(Float(0-2), 2) + pow(Float(0-1), 2))
        XCTAssertEqual(distance, expected, accuracy: 1e-6)
        
        // Test cosineSimilarity
        let similarity = v1.cosineSimilarity(to: v2)
        let expectedSim = dot / (v1.magnitude * v2.magnitude)
        XCTAssertEqual(similarity, expectedSim, accuracy: 1e-6)
    }
    
    func testExtendedVectorProtocolWithUnsafeBuffer() {
        let vector = TestExtendedVector(from: [1, 2, 3, 4])
        
        let sum = vector.withUnsafeBufferPointer { buffer in
            buffer.reduce(0, +)
        }
        XCTAssertEqual(sum, 10.0)
        
        // Test buffer properties
        vector.withUnsafeBufferPointer { buffer in
            XCTAssertEqual(buffer.count, 4)
            XCTAssertNotNil(buffer.baseAddress)
            for (i, value) in buffer.enumerated() {
                XCTAssertEqual(value, Float(i + 1))
            }
        }
    }
    
    func testExtendedVectorProtocolCrossDimensionOperations() {
        // Test operations between different vector types
        let fixed = Vector<Dim128>((0..<128).map { Float($0) })
        let dynamic = DynamicVector((0..<128).map { Float($0) })
        
        // ExtendedVectorProtocol methods require same type, so test each type separately
        // Test fixed vector operations
        let fixedDot = fixed.dotProduct(fixed)
        let fixedDist = fixed.distance(to: fixed)
        let fixedSim = fixed.cosineSimilarity(to: fixed)
        
        // Test dynamic vector operations
        let dynamicDot = dynamic.dotProduct(dynamic)
        let dynamicDist = dynamic.distance(to: dynamic)
        let dynamicSim = dynamic.cosineSimilarity(to: dynamic)
        
        // Results should be equivalent since vectors have same values
        XCTAssertEqual(fixedDot, dynamicDot, accuracy: 1e-4)
        XCTAssertEqual(fixedDist, dynamicDist, accuracy: 1e-6)
        XCTAssertEqual(fixedSim, dynamicSim, accuracy: 1e-6)
        
        // Self-similarity should be 1.0
        XCTAssertEqual(fixedSim, 1.0, accuracy: 1e-6)
        XCTAssertEqual(dynamicSim, 1.0, accuracy: 1e-6)
    }
    
    // MARK: - VectorType Protocol Tests
    
    func testVectorTypeProtocol() {
        // VectorType combines requirements from BaseVectorProtocol and adds math operations
        let vector = TestExtendedVector(from: [3, 4, 0, 0])
        
        // Test that it satisfies VectorType requirements
        XCTAssertEqual(vector.scalarCount, 4)
        XCTAssertNotNil(vector.toArray())
        XCTAssertNotNil(vector.dotProduct(vector))
        XCTAssertNotNil(vector.magnitude)
        XCTAssertNotNil(vector.normalized())
    }
    
    // MARK: - BinaryEncodable Tests
    
    func testBinaryEncodingThroughVectorProtocol() throws {
        // Test with built-in types that conform to VectorProtocol
        let vector = Vector<Dim128>(repeating: 1.0)
        let vectorData = try vector.encodeBinary()
        XCTAssertGreaterThan(vectorData.count, 0)
        
        // Verify header is present (8 bytes) + data + checksum (4 bytes)
        let expectedSize = 8 + (128 * 4) + 4  // header + floats + checksum
        XCTAssertEqual(vectorData.count, expectedSize)
        
        let dynamic = DynamicVector([1, 2, 3, 4, 5])
        let dynamicData = try dynamic.encodeBinary()
        XCTAssertGreaterThan(dynamicData.count, 0)
        
        // Test with test vector type
        // TestExtendedVector doesn't have encodeBinary since it's not part of ExtendedVectorProtocol
        // let testVector = TestExtendedVector(from: [1, 2, 3, 4])
        // let testData = testVector.encodeBinary()
        // XCTAssertGreaterThan(testData.count, 0)
    }
    
    // MARK: - BinaryDecodable Tests
    
    func testBinaryDecodableProtocol() throws {
        let values: [Float] = [1.0, 2.0, 3.0, 4.0]
        
        let decodable = try values.withUnsafeBufferPointer { buffer in
            try TestBinaryDecodable(dimension: 4, buffer: buffer)
        }
        
        XCTAssertEqual(decodable.dimension, 4)
        XCTAssertEqual(decodable.values, values)
        
        // Test dimension mismatch
        XCTAssertThrowsError(
            try values.withUnsafeBufferPointer { buffer in
                try TestBinaryDecodable(dimension: 5, buffer: buffer)
            }
        ) { error in
            if let vectorError = error as? VectorError,
               vectorError.kind == .dimensionMismatch {
                // Success - got expected error
            } else {
                XCTFail("Expected VectorError.dimensionMismatch, got: \(error)")
            }
        }
    }
    
    // MARK: - Type Alias Tests
    
    func testExtendedVectorProtocolTypes() {
        // Test that common types conform to ExtendedVectorProtocol
        // VectorProtocol is just an alias for ExtendedVectorProtocol
        let _: any ExtendedVectorProtocol = TestExtendedVector(from: [1, 2, 3, 4])
        let _: any ExtendedVectorProtocol = Vector<Dim128>(repeating: 1.0)
        let _: any ExtendedVectorProtocol = DynamicVector([1, 2, 3])
        
        // All should compile without issues
        XCTAssertTrue(true)
    }
    
    // MARK: - Protocol Composition Tests
    
    func testProtocolComposition() throws {
        // Test that types can conform to multiple protocols
        struct MultiProtocolVector: ExtendedVectorProtocol, BinaryDecodable, BinaryEncodable, VectorType {
            typealias Scalar = Float
            static let dimensions = 2
            
            private var x: Float
            private var y: Float
            
            init(from array: [Float]) {
                precondition(array.count == 2)
                self.x = array[0]
                self.y = array[1]
            }
            
            init(dimension: Int, buffer: UnsafeBufferPointer<Float>) throws {
                guard dimension == 2 && buffer.count == 2 else {
                    throw VectorError.dimensionMismatch(expected: 2, actual: buffer.count)
                }
                self.x = buffer[0]
                self.y = buffer[1]
            }
            
            var scalarCount: Int { 2 }
            
            func toArray() -> [Float] { [x, y] }
            
            subscript(index: Int) -> Float {
                get {
                    switch index {
                    case 0: return x
                    case 1: return y
                    default: fatalError("Index out of bounds")
                    }
                }
                set {
                    switch index {
                    case 0: x = newValue
                    case 1: y = newValue
                    default: fatalError("Index out of bounds")
                    }
                }
            }
            
            func dotProduct(_ other: Self) -> Float {
                return x * other.x + y * other.y
            }
            
            var magnitude: Float {
                sqrt(x * x + y * y)
            }
            
            func normalized() -> Self {
                let mag = magnitude
                guard mag > 0 else { return self }
                return Self(from: [x / mag, y / mag])
            }
            
            func distance(to other: Self) -> Float {
                let dx = x - other.x
                let dy = y - other.y
                return sqrt(dx * dx + dy * dy)
            }
            
            func cosineSimilarity(to other: Self) -> Float {
                dotProduct(other) / (magnitude * other.magnitude)
            }
            
            func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
                try [x, y].withUnsafeBufferPointer(body)
            }
            
            static func decodeBinary(from data: Data) throws -> Self {
                guard data.count == 2 * MemoryLayout<Float>.size else {
                    throw VectorError.dimensionMismatch(expected: 2, actual: data.count / MemoryLayout<Float>.size)
                }
                var values = [Float](repeating: 0, count: 2)
                _ = values.withUnsafeMutableBytes { buffer in
                    data.copyBytes(to: buffer)
                }
                return Self(from: values)
            }
            
        }
        
        // Test the multi-protocol type
        let vector = MultiProtocolVector(from: [3, 4])
        XCTAssertEqual(vector.magnitude, 5.0)
        XCTAssertEqual(vector.normalized().magnitude, 1.0, accuracy: 1e-6)
        
        let data = try vector.encodeBinary()
        // Expected: 8 (header) + 8 (2 floats * 4 bytes) + 4 (checksum) = 20 bytes
        XCTAssertEqual(data.count, 20)
        
        // Test decoding
        let decoded = try! [Float](repeating: 0, count: 2).withUnsafeBufferPointer { _ in
            try [Float(3), Float(4)].withUnsafeBufferPointer { buffer in
                try MultiProtocolVector(dimension: 2, buffer: buffer)
            }
        }
        XCTAssertEqual(decoded.toArray(), [3, 4])
    }
    
    // MARK: - Generic Constraint Tests
    
    func testGenericConstraints() {
        // Test function with VectorType constraint
        func computeMagnitude<V: VectorType>(_ vector: V) -> Float {
            vector.magnitude
        }
        
        let v1 = Vector<Dim128>(repeating: 1.0)
        let v2 = DynamicVector([3, 4])
        // TestExtendedVector doesn't conform to VectorType, use a type that does
        
        XCTAssertEqual(computeMagnitude(v1), sqrt(128.0), accuracy: 1e-6)
        XCTAssertEqual(computeMagnitude(v2), 5.0)
        
        // Test function with ExtendedVectorProtocol constraint
        func computeSimilarity<V: ExtendedVectorProtocol>(
            _ v1: V, _ v2: V
        ) -> Float where V.Scalar == Float {
            v1.cosineSimilarity(to: v2)
        }
        
        let sim = computeSimilarity(v2, v2)
        XCTAssertEqual(sim, 1.0, accuracy: 1e-6)
    }
    
    // MARK: - Associated Type Tests
    
    func testAssociatedTypes() {
        // Verify Scalar type is correctly propagated
        XCTAssertTrue(type(of: TestBaseVector.Scalar.self) == Float.Type.self)
        XCTAssertTrue(type(of: TestExtendedVector.Scalar.self) == Float.Type.self)
        XCTAssertTrue(type(of: Vector<Dim128>.Scalar.self) == Float.Type.self)
        XCTAssertTrue(type(of: DynamicVector.Scalar.self) == Float.Type.self)
    }
    
    // MARK: - Protocol Extension Tests
    
    func testProtocolExtensions() {
        // Test that types properly conform to protocols
        let v1 = Vector<Dim256>(repeating: 0)
        XCTAssertEqual(v1.scalarCount, 256)
        
        let v2 = DynamicVector(dimension: 100)
        XCTAssertEqual(v2.scalarCount, 100)
        
        // Both should conform to VectorType
        func acceptsVectorType<V: VectorType>(_ vector: V) -> Int {
            vector.scalarCount
        }
        
        XCTAssertEqual(acceptsVectorType(v1), 256)
        XCTAssertEqual(acceptsVectorType(v2), 100)
    }
}