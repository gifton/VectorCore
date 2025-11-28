import Testing
import Foundation
@testable import VectorCore

@Suite("VectorProducer Protocol")
struct VectorProducerSuite {

    // MARK: - Mock VectorProducer

    struct MockProducer: VectorProducer {
        let dimensions: Int
        let producesNormalizedVectors: Bool

        init(dimensions: Int = 128, normalized: Bool = true) {
            self.dimensions = dimensions
            self.producesNormalizedVectors = normalized
        }

        func produce(_ texts: [String]) async throws -> [[Float]] {
            // Simple mock: return zero vectors of correct dimension
            texts.map { _ in Array(repeating: Float(0.0), count: dimensions) }
        }
    }

    // MARK: - Basic Protocol Tests

    @Test
    func testVectorProducer_DimensionsProperty() {
        let producer = MockProducer(dimensions: 256)
        #expect(producer.dimensions == 256)
    }

    @Test
    func testVectorProducer_NormalizationProperty() {
        let normalized = MockProducer(normalized: true)
        #expect(normalized.producesNormalizedVectors == true)

        let unnormalized = MockProducer(normalized: false)
        #expect(unnormalized.producesNormalizedVectors == false)
    }

    @Test
    func testVectorProducer_BatchProduction() async throws {
        let producer = MockProducer(dimensions: 128)
        let texts = ["text1", "text2", "text3"]

        let embeddings = try await producer.produce(texts)

        #expect(embeddings.count == 3)
        #expect(embeddings[0].count == 128)
        #expect(embeddings[1].count == 128)
        #expect(embeddings[2].count == 128)
    }

    @Test
    func testVectorProducer_SingleTextDefault() async throws {
        let producer = MockProducer(dimensions: 64)

        // Should use default implementation that delegates to batch
        let embedding = try await producer.produce("single text")

        #expect(embedding.count == 64)
    }

    @Test
    func testVectorProducer_EmptyBatchHandling() async throws {
        let producer = MockProducer(dimensions: 128)

        let embeddings = try await producer.produce([])

        #expect(embeddings.isEmpty)
    }

    // MARK: - Custom Single Implementation

    struct CustomSingleProducer: VectorProducer {
        let dimensions: Int = 128
        let producesNormalizedVectors: Bool = true
        var singleCallCount = 0

        func produce(_ texts: [String]) async throws -> [[Float]] {
            texts.map { _ in Array(repeating: Float(1.0), count: dimensions) }
        }

        // Custom single implementation
        func produce(_ text: String) async throws -> [Float] {
            // Returns different values to prove custom implementation is used
            Array(repeating: Float(2.0), count: dimensions)
        }
    }

    @Test
    func testVectorProducer_CustomSingleImplementation() async throws {
        let producer = CustomSingleProducer()

        let single = try await producer.produce("test")
        let batch = try await producer.produce(["test"])

        // Custom implementation returns 2.0, batch returns 1.0
        #expect(single[0] == 2.0)
        #expect(batch[0][0] == 1.0)
    }

    // MARK: - VectorProducerHints Tests

    @Test
    func testVectorProducerHints_DefaultValues() {
        let hints = VectorProducerHints(dimensions: 256)

        #expect(hints.dimensions == 256)
        #expect(hints.isNormalized == true)
        #expect(hints.optimalBatchSize == 32)
        #expect(hints.maxBatchSize == 128)
    }

    @Test
    func testVectorProducerHints_CustomValues() {
        let hints = VectorProducerHints(
            dimensions: 512,
            isNormalized: false,
            optimalBatchSize: 64,
            maxBatchSize: 256
        )

        #expect(hints.dimensions == 512)
        #expect(hints.isNormalized == false)
        #expect(hints.optimalBatchSize == 64)
        #expect(hints.maxBatchSize == 256)
    }

    @Test
    func testVectorProducerHints_Equatable() {
        let hints1 = VectorProducerHints(dimensions: 128)
        let hints2 = VectorProducerHints(dimensions: 128)
        let hints3 = VectorProducerHints(dimensions: 256)

        #expect(hints1 == hints2)
        #expect(hints1 != hints3)
    }

    @Test
    func testVectorProducerHints_Sendable() {
        // Compile-time check: VectorProducerHints is Sendable
        let hints = VectorProducerHints(dimensions: 128)

        Task {
            // Should compile without warnings
            let _ = hints.dimensions
        }
    }

    // MARK: - Error Handling Tests

    struct FailingProducer: VectorProducer {
        let dimensions: Int = 128
        let producesNormalizedVectors: Bool = true

        func produce(_ texts: [String]) async throws -> [[Float]] {
            throw VectorError(.operationFailed, message: "Mock failure")
        }
    }

    @Test
    func testVectorProducer_ErrorPropagation() async throws {
        let producer = FailingProducer()

        do {
            _ = try await producer.produce(["test"])
            Issue.record("Expected error to be thrown")
        } catch let error as VectorError {
            #expect(error.kind == .operationFailed)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test
    func testVectorProducer_DefaultSingleErrorHandling() async throws {
        let producer = FailingProducer()

        do {
            // Default single implementation should propagate batch errors
            _ = try await producer.produce("test")
            Issue.record("Expected error to be thrown")
        } catch let error as VectorError {
            #expect(error.kind == .operationFailed)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    // MARK: - Empty Result Handling

    struct EmptyResultProducer: VectorProducer {
        let dimensions: Int = 128
        let producesNormalizedVectors: Bool = true

        func produce(_ texts: [String]) async throws -> [[Float]] {
            []  // Returns empty array even for non-empty input
        }
    }

    @Test
    func testVectorProducer_EmptyResultError() async throws {
        let producer = EmptyResultProducer()

        do {
            // Default single implementation should throw when batch returns empty
            _ = try await producer.produce("test")
            Issue.record("Expected invalidOperation error")
        } catch let error as VectorError {
            #expect(error.kind == .invalidOperation)
            #expect(error.context.additionalInfo["message"]?.contains("empty results") == true)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    // MARK: - Dimension Validation

    @Test
    func testVectorProducer_DimensionConsistency() async throws {
        let producer = MockProducer(dimensions: 384)
        let embeddings = try await producer.produce(["a", "b", "c"])

        // All embeddings should have consistent dimensions
        for embedding in embeddings {
            #expect(embedding.count == producer.dimensions)
        }
    }

    // MARK: - Concurrent Access

    @Test
    func testVectorProducer_ConcurrentAccess() async throws {
        let producer = MockProducer(dimensions: 128)

        // Test concurrent calls
        await withTaskGroup(of: [[Float]].self) { group in
            for i in 0..<5 {
                group.addTask {
                    try! await producer.produce(["text\(i)"])
                }
            }

            var results: [[[Float]]] = []
            for await result in group {
                results.append(result)
            }

            #expect(results.count == 5)
            for result in results {
                #expect(result[0].count == 128)
            }
        }
    }
}
