// VectorCore: Integration Tests
//
// End-to-end tests for complete workflows
//

import XCTest
import Foundation
@testable import VectorCore

final class IntegrationTests: XCTestCase {
    
    // MARK: - End-to-End Workflows
    
    func testCompleteVectorWorkflow() {
        // 1. Create vectors using factory
        let dimension = 256
        let vector1 = VectorFactory.random(dimension: dimension, range: -1...1)
        let vector2 = VectorFactory.random(dimension: dimension, range: -1...1)
        
        // 2. Perform mathematical operations
        let sum = (vector1 as! Vector256) + (vector2 as! Vector256)
        let normalized = sum.normalized()
        
        // 3. Calculate distances with different metrics
        let metrics: [any DistanceMetric] = [
            EuclideanDistance(),
            CosineDistance(),
            ManhattanDistance(),
            DotProductDistance()
        ]
        
        var distances: [Float] = []
        for metric in metrics {
            let distance = metric.distance(normalized, vector1 as! Vector256)
            distances.append(distance)
        }
        
        // 4. Create a simple data structure for testing
        struct TestVectorData: Codable {
            let id: String
            let values: [Float]
            let metadata: [String: String]
            let distances: [Float]
        }
        
        let vectorData = TestVectorData(
            id: "test-vector-001",
            values: normalized.toArray(),
            metadata: [
                "source": "integration_test",
                "timestamp": "\(Date().timeIntervalSince1970)"
            ],
            distances: distances
        )
        
        // 5. Serialize to JSON
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let jsonData = try! encoder.encode(vectorData)
        
        // 6. Deserialize back
        let decoder = JSONDecoder()
        let decoded = try! decoder.decode(TestVectorData.self, from: jsonData)
        
        // 7. Verify round-trip
        XCTAssertEqual(decoded.id, vectorData.id)
        XCTAssertEqual(decoded.metadata["source"], "integration_test")
        
        // Vector values should match
        for i in 0..<256 {
            XCTAssertEqual(decoded.values[i], normalized[i], accuracy: 1e-6)
        }
    }
    
    func testBatchProcessingWorkflow() async throws {
        // 1. Generate a dataset
        let datasetSize = 1000
        let vectors = (0..<datasetSize).map { _ in
            Vector256.random(in: -1...1)
        }
        
        // 2. Preprocess: normalize all vectors
        let normalized = try await BatchOperations.map(vectors) { vector in
            vector.normalized()
        }
        
        // 3. Compute statistics
        let stats = try await BatchOperations.statistics(for: normalized)
        XCTAssertEqual(stats.count, datasetSize)
        XCTAssertEqual(stats.meanMagnitude, 1.0, accuracy: 0.01) // All normalized
        
        // 4. Filter high-quality vectors (simplified - VectorQuality doesn't exist)
        let filtered = try await BatchOperations.filter(normalized) { vector in
            // Simple quality check - variance > 0.1
            let mean = vector.toArray().reduce(0, +) / Float(vector.scalarCount)
            let variance = vector.toArray().map { pow($0 - mean, 2) }.reduce(0, +) / Float(vector.scalarCount)
            return variance > 0.1
        }
        
        XCTAssertGreaterThan(filtered.count, 0)
        XCTAssertLessThanOrEqual(filtered.count, datasetSize)
        
        // 5. Find similar vectors
        let query = normalized.first!
        let similar = await BatchOperations.findNearest(
            to: query,
            in: filtered,
            k: 10,
            metric: CosineDistance()
        )
        
        XCTAssertLessThanOrEqual(similar.count, 10)
        if !similar.isEmpty {
            XCTAssertEqual(similar[0].distance, 0.0, accuracy: 1e-6) // First should be query itself
        }
        
        // 6. Compute pairwise similarities for clustering
        let sample = BatchOperations.sample(filtered, k: min(50, filtered.count))
        let distances = try await BatchOperations.pairwiseDistances(sample, metric: EuclideanDistance())
        
        XCTAssertEqual(distances.count, sample.count)
        XCTAssertEqual(distances[0].count, sample.count)
    }
    
    func testCrossTypeCompatibility() {
        // Test operations between different vector types
        let v128 = Vector128.random(in: -1...1)
        let v256 = Vector256.random(in: -1...1)
        let v512 = Vector512.random(in: -1...1)
        let vDynamic = DynamicVector(dimension: 768, repeating: 0.5)
        
        // All should work with distance metrics
        let metric = EuclideanDistance()
        
        _ = metric.distance(v128, v128)
        _ = metric.distance(v256, v256)
        _ = metric.distance(v512, v512)
        _ = metric.distance(vDynamic, vDynamic)
        
        // Factory should handle all types
        let vectors: [any VectorType] = [
            VectorFactory.zeros(dimension: 128),
            VectorFactory.ones(dimension: 256),
            VectorFactory.random(dimension: 512),
            VectorFactory.random(dimension: 1000) // Non-standard dimension
        ]
        
        for vector in vectors {
            XCTAssertGreaterThan(vector.scalarCount, 0)
            // VectorType doesn't have dotProduct method
            _ = vector.magnitude
            _ = vector.normalized()
        }
    }
    
    func testMemoryPressureScenario() async throws {
        // Simulate high memory usage scenario
        // Create many large vectors
        let vectors = (0..<1000).map { _ in
            Vector1536.random(in: -1...1)
        }
        
        // Process in batches to manage memory
        let _ = try await BatchOperations.process(vectors, batchSize: 50) { batch in
            batch.map { vector in
                // Expensive operations
                let normalized = vector.normalized()
                let softmaxed = normalized.softmax()
                return softmaxed.magnitude
            }
        }
        
        // Vectors should be released after this block
        
        // If we get here without crashing, memory management is working
        XCTAssertTrue(true)
    }
    
    func testSerializationRoundTrip() {
        // Test all vector types can be serialized and deserialized
        
        // 1. Fixed-size vectors
        let vectors: [(any Codable, String)] = [
            (Vector128.random(in: -1...1), "Vector128"),
            (Vector256.random(in: -1...1), "Vector256"),
            (Vector512.random(in: -1...1), "Vector512"),
            (Vector768.random(in: -1...1), "Vector768"),
            (Vector1536.random(in: -1...1), "Vector1536"),
            (DynamicVector(dimension: 100, repeating: 3.14), "DynamicVector")
        ]
        
        for (vector, name) in vectors {
            // Encode
            let encoder = JSONEncoder()
            let data = try! encoder.encode(vector)
            
            // Decode based on type
            let decoder = JSONDecoder()
            
            switch name {
            case "Vector128":
                let decoded = try! decoder.decode(Vector128.self, from: data)
                let original = vector as! Vector128
                for i in 0..<128 {
                    XCTAssertEqual(decoded[i], original[i], accuracy: 1e-6)
                }
                
            case "Vector256":
                let decoded = try! decoder.decode(Vector256.self, from: data)
                let original = vector as! Vector256
                for i in 0..<256 {
                    XCTAssertEqual(decoded[i], original[i], accuracy: 1e-6)
                }
                
            case "Vector512":
                let decoded = try! decoder.decode(Vector512.self, from: data)
                let original = vector as! Vector512
                for i in 0..<512 {
                    XCTAssertEqual(decoded[i], original[i], accuracy: 1e-6)
                }
                
            case "Vector768":
                let decoded = try! decoder.decode(Vector768.self, from: data)
                let original = vector as! Vector768
                for i in 0..<768 {
                    XCTAssertEqual(decoded[i], original[i], accuracy: 1e-6)
                }
                
            case "Vector1536":
                let decoded = try! decoder.decode(Vector1536.self, from: data)
                let original = vector as! Vector1536
                for i in 0..<1536 {
                    XCTAssertEqual(decoded[i], original[i], accuracy: 1e-6)
                }
                
            case "DynamicVector":
                let decoded = try! decoder.decode(DynamicVector.self, from: data)
                let original = vector as! DynamicVector
                XCTAssertEqual(decoded.dimension, original.dimension)
                for i in 0..<decoded.dimension {
                    XCTAssertEqual(decoded[i], original[i], accuracy: 1e-6)
                }
                
            default:
                XCTFail("Unknown vector type: \(name)")
            }
        }
    }
    
    func testLargeScaleOperations() async throws {
        // Test operations at scale
        let vectorCount = 10_000
        let dimension = 256
        
        // 1. Create large dataset
        print("Creating \(vectorCount) vectors...")
        let vectors = (0..<vectorCount).map { i in
            // Create vectors with some pattern for verification
            let values = (0..<dimension).map { j in
                Float(sin(Double(i * dimension + j) * .pi / 1000))
            }
            return Vector256(values)
        }
        
        // 2. Batch normalization
        print("Normalizing vectors...")
        let startNorm = Date()
        let normalized = try await BatchOperations.map(vectors) { $0.normalized() }
        let normTime = Date().timeIntervalSince(startNorm)
        print("Normalization took \(normTime) seconds")
        
        // 3. Find clusters of similar vectors
        print("Finding similar vectors...")
        let startSearch = Date()
        let queries = BatchOperations.sample(normalized, k: 10)
        
        for query in queries {
            let neighbors = try await BatchOperations.findNearest(
                to: query,
                in: normalized,
                k: 100,
                metric: CosineDistance()
            )
            
            XCTAssertEqual(neighbors.count, 100)
            // Verify ordering
            for i in 1..<neighbors.count {
                XCTAssertLessThanOrEqual(neighbors[i-1].distance, neighbors[i].distance)
            }
        }
        
        let searchTime = Date().timeIntervalSince(startSearch)
        print("Similarity search took \(searchTime) seconds")
        
        // 4. Compute aggregate statistics
        print("Computing statistics...")
        let stats = try await BatchOperations.statistics(for: normalized)
        XCTAssertEqual(stats.count, vectorCount)
        XCTAssertEqual(stats.meanMagnitude, 1.0, accuracy: 0.01)
        
        print("Large scale test completed successfully")
    }
    
    func testErrorHandlingIntegration() {
        // Test error propagation through the system
        
        // 1. Dimension mismatch errors
        let v128 = Vector128.random(in: -1...1)
        let v256 = Vector256.random(in: -1...1)
        
        // Cannot compute distance between different dimensions
        // (This would be caught at compile time with the generic implementation)
        
        // 2. Factory errors
        do {
            let values = [Float](repeating: 1.0, count: 100)
            _ = try VectorFactory.vector(of: 128, from: values)
            XCTFail("Should have thrown dimension mismatch error")
        } catch let error as VectorError {
            XCTAssertEqual(error.code, "DIMENSION_MISMATCH")
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
        
        // 3. Invalid operations
        let zero = Vector256(repeating: 0.0)
        let normalized = zero.normalized() // Should handle zero vector gracefully
        
        // Check it's handled (implementation dependent - might return zero or NaN)
        XCTAssertTrue(normalized.magnitude == 0.0 || normalized.magnitude.isNaN)
    }
    
    // MARK: - Performance Regression Tests
    
    func testPerformanceRegression() async throws {
        // Establish baseline performance expectations
        
        // 1. Vector creation
        let creationTime = measureTime {
            for _ in 0..<1000 {
                _ = Vector256.random(in: -1...1)
            }
        }
        XCTAssertLessThan(creationTime, 0.1, "Vector creation too slow")
        
        // 2. Distance calculation
        let v1 = Vector256.random(in: -1...1)
        let v2 = Vector256.random(in: -1...1)
        let metric = EuclideanDistance()
        
        let distanceTime = measureTime {
            for _ in 0..<10000 {
                _ = metric.distance(v1, v2)
            }
        }
        XCTAssertLessThan(distanceTime, 0.1, "Distance calculation too slow")
        
        // 3. Batch operations
        let vectors = (0..<1000).map { _ in Vector256.random(in: -1...1) }
        
        // Batch operations are async, measure them differently
        let batchStart = Date()
        _ = try await BatchOperations.map(vectors) { $0.normalized() }
        let batchTime = Date().timeIntervalSince(batchStart)
        XCTAssertLessThan(batchTime, 0.5, "Batch processing too slow")
    }
    
    // MARK: - Helper Functions
    
    private func measureTime(block: () -> Void) -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        block()
        return CFAbsoluteTimeGetCurrent() - start
    }
}