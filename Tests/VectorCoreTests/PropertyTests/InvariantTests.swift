import XCTest
@testable import VectorCore

/// Tests for invariants that should always hold throughout vector operations
final class InvariantTests: XCTestCase {
    
    // MARK: - Configuration
    
    private let iterations = 100
    private let accuracy: Float = 1e-5
    
    // MARK: - Dimension Preservation Invariants
    
    func testOperationsPreserveDimension() {
        // All operations should preserve the dimension of vectors
        testFixedVectorDimensionPreservation()
        testDynamicVectorDimensionPreservation()
    }
    
    private func testFixedVectorDimensionPreservation() {
        let v1 = Vector<Dim128>.random(in: -10...10)
        let v2 = Vector<Dim128>.random(in: -10...10)
        let scalar: Float = 2.5
        
        // Unary operations
        XCTAssertEqual(v1.scalarCount, 128, "Original vector dimension changed")
        XCTAssertEqual((-v1).scalarCount, 128, "Negation changed dimension")
        XCTAssertEqual(v1.normalized().scalarCount, 128, "Normalization changed dimension")
        
        // Binary operations
        XCTAssertEqual((v1 + v2).scalarCount, 128, "Addition changed dimension")
        XCTAssertEqual((v1 - v2).scalarCount, 128, "Subtraction changed dimension")
        XCTAssertEqual((v1 .* v2).scalarCount, 128, "Element-wise multiplication changed dimension")
        XCTAssertEqual((v1 ./ v2).scalarCount, 128, "Element-wise division changed dimension")
        
        // Scalar operations
        XCTAssertEqual((v1 * scalar).scalarCount, 128, "Scalar multiplication changed dimension")
        XCTAssertEqual((v1 / scalar).scalarCount, 128, "Scalar division changed dimension")
    }
    
    private func testDynamicVectorDimensionPreservation() {
        let dimensions = [32, 64, 128, 256, 512, 768, 1536]
        
        for dim in dimensions {
            let v1 = DynamicVector.random(dimension: dim, in: -10...10)
            let v2 = DynamicVector.random(dimension: dim, in: -10...10)
            let scalar: Float = 2.5
            
            // Unary operations
            XCTAssertEqual(v1.dimension, dim, "Original vector dimension changed")
            XCTAssertEqual((-v1).dimension, dim, "Negation changed dimension for dim \(dim)")
            XCTAssertEqual(v1.normalized().dimension, dim, "Normalization changed dimension for dim \(dim)")
            
            // Binary operations
            XCTAssertEqual((v1 + v2).dimension, dim, "Addition changed dimension for dim \(dim)")
            XCTAssertEqual((v1 - v2).dimension, dim, "Subtraction changed dimension for dim \(dim)")
            XCTAssertEqual((v1 .* v2).dimension, dim, "Element-wise multiplication changed dimension for dim \(dim)")
            XCTAssertEqual((v1 ./ v2).dimension, dim, "Element-wise division changed dimension for dim \(dim)")
            
            // Scalar operations
            XCTAssertEqual((v1 * scalar).dimension, dim, "Scalar multiplication changed dimension for dim \(dim)")
            XCTAssertEqual((v1 / scalar).dimension, dim, "Scalar division changed dimension for dim \(dim)")
        }
    }
    
    // MARK: - NaN/Infinity Invariants
    
    func testNoNaNInNormalOperations() {
        // Normal operations on finite values should never produce NaN
        for _ in 0..<iterations {
            let v1 = Vector<Dim128>.random(in: -100...100)
            let v2 = Vector<Dim128>.random(in: -100...100)
            let scalar = Float.random(in: -10...10)
            
            // Check various operations
            assertNoNaN(v1 + v2, "Addition produced NaN")
            assertNoNaN(v1 - v2, "Subtraction produced NaN")
            assertNoNaN(v1 * scalar, "Scalar multiplication produced NaN")
            if scalar != 0 {
                assertNoNaN(v1 / scalar, "Scalar division produced NaN")
            }
            assertNoNaN(v1 .* v2, "Element-wise multiplication produced NaN")
            
            // Avoid division by zero
            let v2Safe = v2.toArray().map { $0 == 0 ? 1 : $0 }
            let v2NonZero = Vector<Dim128>(v2Safe)
            assertNoNaN(v1 ./ v2NonZero, "Element-wise division produced NaN")
            
            // Check derived values
            XCTAssertFalse(v1.magnitude.isNaN, "Magnitude produced NaN")
            XCTAssertFalse(v1.dotProduct(v2).isNaN, "Dot product produced NaN")
            XCTAssertFalse(v1.distance(to: v2).isNaN, "Distance produced NaN")
            
            if v1.magnitude > 1e-6 && v2.magnitude > 1e-6 {
                XCTAssertFalse(v1.cosineSimilarity(to: v2).isNaN, "Cosine similarity produced NaN")
            }
        }
    }
    
    func testNoInfinityInNormalOperations() {
        // Normal operations on reasonable values should never produce Infinity
        for _ in 0..<iterations {
            let v1 = Vector<Dim64>.random(in: -100...100)
            let v2 = Vector<Dim64>.random(in: -100...100)
            let scalar = Float.random(in: -10...10)
            
            // Check various operations
            assertNoInfinity(v1 + v2, "Addition produced Infinity")
            assertNoInfinity(v1 - v2, "Subtraction produced Infinity")
            assertNoInfinity(v1 * scalar, "Scalar multiplication produced Infinity")
            if abs(scalar) > 1e-6 {
                assertNoInfinity(v1 / scalar, "Scalar division produced Infinity")
            }
            
            // Check derived values
            XCTAssertFalse(v1.magnitude.isInfinite, "Magnitude produced Infinity")
            XCTAssertFalse(v1.dotProduct(v2).isInfinite, "Dot product produced Infinity")
            XCTAssertFalse(v1.distance(to: v2).isInfinite, "Distance produced Infinity")
        }
    }
    
    // MARK: - Storage Optimization Invariants
    
    func testStorageOptimizationTransparency() {
        // Storage optimization should be transparent to the user
        let dimensions = [32, 64, 128, 256, 512]
        
        for dim in dimensions {
            // Create vectors with different storage patterns
            let values = (0..<dim).map { Float($0) }
            
            // Test with Vector types
            switch dim {
            case 32:
                let v = Vector<Dim32>(values)
                verifyStorageTransparency(v, expectedValues: values)
            case 64:
                let v = Vector<Dim64>(values)
                verifyStorageTransparency(v, expectedValues: values)
            case 128:
                let v = Vector<Dim128>(values)
                verifyStorageTransparency(v, expectedValues: values)
            case 256:
                let v = Vector<Dim256>(values)
                verifyStorageTransparency(v, expectedValues: values)
            case 512:
                let v = Vector<Dim512>(values)
                verifyStorageTransparency(v, expectedValues: values)
            default:
                break
            }
            
            // Test with DynamicVector
            let dv = DynamicVector(values)
            verifyDynamicStorageTransparency(dv, expectedValues: values)
        }
    }
    
    private func verifyStorageTransparency<D: Dimension>(_ vector: Vector<D>, expectedValues: [Float]) {
        // Verify all values are accessible correctly
        for i in 0..<vector.scalarCount {
            XCTAssertEqual(vector[i], expectedValues[i], accuracy: 1e-7,
                          "Storage optimization changed value at index \(i)")
        }
        
        // Verify toArray() returns correct values
        let array = vector.toArray()
        XCTAssertEqual(array.count, expectedValues.count, "toArray() returned wrong count")
        for i in 0..<array.count {
            XCTAssertEqual(array[i], expectedValues[i], accuracy: 1e-7,
                          "toArray() returned wrong value at index \(i)")
        }
        
        // Verify operations work correctly
        let doubled = vector * 2.0
        for i in 0..<doubled.scalarCount {
            XCTAssertEqual(doubled[i], expectedValues[i] * 2.0, accuracy: 1e-6,
                          "Operation failed with storage optimization")
        }
    }
    
    private func verifyDynamicStorageTransparency(_ vector: DynamicVector, expectedValues: [Float]) {
        // Similar verification for DynamicVector
        for i in 0..<vector.dimension {
            XCTAssertEqual(vector[i], expectedValues[i], accuracy: 1e-7,
                          "Dynamic storage changed value at index \(i)")
        }
        
        let array = vector.toArray()
        XCTAssertEqual(array.count, expectedValues.count, "toArray() returned wrong count")
        
        let doubled = vector * 2.0
        for i in 0..<doubled.dimension {
            XCTAssertEqual(doubled[i], expectedValues[i] * 2.0, accuracy: 1e-6,
                          "Operation failed with dynamic storage")
        }
    }
    
    // MARK: - Serialization Round-Trip Invariants
    
    func testSerializationRoundTrip() {
        // Test that vectors can be serialized and deserialized without loss
        testFixedVectorSerializationRoundTrip()
        testDynamicVectorSerializationRoundTrip()
    }
    
    private func testFixedVectorSerializationRoundTrip() {
        for _ in 0..<iterations {
            let original = Vector<Dim128>.random(in: -1000...1000)
            
            // Test array round-trip
            let array = original.toArray()
            let reconstructed = Vector<Dim128>(array)
            
            for i in 0..<original.scalarCount {
                XCTAssertEqual(original[i], reconstructed[i], accuracy: 0,
                              "Array serialization round-trip failed at index \(i)")
            }
            
            // Test that operations produce same results
            let scalar: Float = 3.14
            let originalScaled = original * scalar
            let reconstructedScaled = reconstructed * scalar
            
            for i in 0..<originalScaled.scalarCount {
                XCTAssertEqual(originalScaled[i], reconstructedScaled[i], accuracy: 1e-6,
                              "Operations differ after serialization round-trip")
            }
        }
    }
    
    private func testDynamicVectorSerializationRoundTrip() {
        let dimensions = [32, 128, 512, 1024]
        
        for dim in dimensions {
            for _ in 0..<20 {
                let original = DynamicVector.random(dimension: dim, in: -1000...1000)
                
                // Test array round-trip
                let array = original.toArray()
                let reconstructed = DynamicVector(array)
                
                XCTAssertEqual(original.dimension, reconstructed.dimension,
                              "Dimension changed after round-trip")
                
                for i in 0..<original.dimension {
                    XCTAssertEqual(original[i], reconstructed[i], accuracy: 0,
                                  "Dynamic vector serialization failed at index \(i)")
                }
            }
        }
    }
    
    // MARK: - Copy-on-Write Invariants
    
    func testCopyOnWriteSemantics() {
        // Verify that COW semantics work correctly
        let original = Vector<Dim256>.random(in: -10...10)
        var copy = original
        
        // Before mutation, both should be equal
        for i in 0..<original.scalarCount {
            XCTAssertEqual(original[i], copy[i], "COW: Values differ before mutation")
        }
        
        // Mutate the copy
        copy = copy * 2.0
        
        // Original should be unchanged
        let originalArray = original.toArray()
        let copyArray = copy.toArray()
        
        for i in 0..<original.scalarCount {
            XCTAssertEqual(originalArray[i] * 2.0, copyArray[i], accuracy: accuracy,
                          "COW: Mutation affected wrong vector")
        }
    }
    
    // MARK: - Thread Safety Invariants
    
    func testConcurrentReadSafety() {
        // Multiple threads should be able to read the same vector safely
        let vector = Vector<Dim512>.random(in: -100...100)
        let expectedMagnitude = vector.magnitude
        
        let group = DispatchGroup()
        let queue = DispatchQueue(label: "test", attributes: .concurrent)
        var results = [Float](repeating: 0, count: 100)
        
        for i in 0..<100 {
            group.enter()
            queue.async {
                results[i] = vector.magnitude
                group.leave()
            }
        }
        
        group.wait()
        
        // All results should be the same
        for result in results {
            XCTAssertEqual(result, expectedMagnitude, accuracy: accuracy,
                          "Concurrent reads produced different results")
        }
    }
    
    // MARK: - Mathematical Consistency Invariants
    
    func testMathematicalConsistency() {
        // Various mathematical relationships should hold
        for _ in 0..<iterations {
            let v = Vector<Dim128>.random(in: -50...50)
            
            // ||v||² = v·v
            let magnitudeSquared = v.magnitude * v.magnitude
            let dotSelf = v.dotProduct(v)
            XCTAssertEqual(magnitudeSquared, dotSelf, accuracy: accuracy * 10,
                          "||v||² != v·v")
            
            // |v[i]| ≤ ||v||∞ ≤ ||v||₂ * √n
            let lInfNorm = v.lInfinityNorm
            let l2Norm = v.l2Norm
            for i in 0..<v.scalarCount {
                XCTAssertLessThanOrEqual(abs(v[i]), lInfNorm + accuracy,
                                        "|v[i]| > ||v||∞")
            }
            XCTAssertLessThanOrEqual(lInfNorm, l2Norm * sqrt(Float(v.scalarCount)) + accuracy,
                                    "||v||∞ > ||v||₂ * √n")
            
            // ||v||₁ ≥ ||v||₂ ≥ ||v||∞
            let l1Norm = v.l1Norm
            XCTAssertGreaterThanOrEqual(l1Norm, l2Norm - accuracy,
                                       "||v||₁ < ||v||₂")
            XCTAssertGreaterThanOrEqual(l2Norm, lInfNorm - accuracy,
                                       "||v||₂ < ||v||∞")
        }
    }
    
    // MARK: - Helper Methods
    
    private func assertNoNaN<D: Dimension>(_ vector: Vector<D>, _ message: String, 
                                           file: StaticString = #file, line: UInt = #line) {
        for i in 0..<vector.scalarCount {
            XCTAssertFalse(vector[i].isNaN, "\(message) at index \(i)", file: file, line: line)
        }
    }
    
    private func assertNoInfinity<D: Dimension>(_ vector: Vector<D>, _ message: String,
                                               file: StaticString = #file, line: UInt = #line) {
        for i in 0..<vector.scalarCount {
            XCTAssertFalse(vector[i].isInfinite, "\(message) at index \(i)", file: file, line: line)
        }
    }
}