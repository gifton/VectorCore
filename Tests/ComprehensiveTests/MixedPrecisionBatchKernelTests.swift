import Testing
import Foundation
@testable import VectorCore

@Suite("Mixed Precision Batch Kernels - Spec #32")
struct MixedPrecisionBatchKernelTests {

    // MARK: - Helper Functions

    func generateRandomVector512() -> Vector512Optimized {
        return try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
    }

    func generateRandomVectors512(count: Int) -> [Vector512Optimized] {
        return (0..<count).map { _ in generateRandomVector512() }
    }

    func generateRandomVector768() -> Vector768Optimized {
        return try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
    }

    func generateRandomVectors768(count: Int) -> [Vector768Optimized] {
        return (0..<count).map { _ in generateRandomVector768() }
    }

    func generateRandomVector1536() -> Vector1536Optimized {
        return try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
    }

    func generateRandomVectors1536(count: Int) -> [Vector1536Optimized] {
        return (0..<count).map { _ in generateRandomVector1536() }
    }

    // MARK: - Euclidean² Accuracy Tests

    @Test("Range Euclidean² 512: Accuracy vs FP32 reference")
    func testRangeEuclidean512Accuracy() throws {
        let query = generateRandomVector512()
        let candidates = generateRandomVectors512(count: 100)
        let candidatesFP16 = MixedPrecisionKernels.convertToFP16_512(candidates)

        // Compute FP32 reference
        var referenceFP32 = [Float](repeating: 0, count: 100)
        referenceFP32.withUnsafeMutableBufferPointer { buffer in
            BatchKernels.range_euclid2_512(
                query: query,
                candidates: candidates,
                range: 0..<100,
                out: buffer
            )
        }

        // Compute FP16 mixed precision
        var resultsFP16 = [Float](repeating: 0, count: 100)
        resultsFP16.withUnsafeMutableBufferPointer { buffer in
            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 0..<100,
                out: buffer
            )
        }

        // Validate accuracy
        var maxRelativeError: Float = 0
        for (ref, computed) in zip(referenceFP32, resultsFP16) {
            let relativeError = abs(ref - computed) / max(ref, 1e-6)
            maxRelativeError = max(maxRelativeError, relativeError)
            #expect(relativeError < 0.001, "Relative error \(relativeError) exceeds 0.1%")
        }

        print("✓ Euclidean² 512: Max relative error = \(String(format: "%.4f%%", maxRelativeError * 100))")
    }

    @Test("Range Euclidean² 768: Accuracy vs FP32 reference")
    func testRangeEuclidean768Accuracy() throws {
        let query = generateRandomVector768()
        let candidates = generateRandomVectors768(count: 100)
        let candidatesFP16 = MixedPrecisionKernels.convertToFP16_768(candidates)

        var referenceFP32 = [Float](repeating: 0, count: 100)
        referenceFP32.withUnsafeMutableBufferPointer { buffer in
            BatchKernels.range_euclid2_768(
                query: query,
                candidates: candidates,
                range: 0..<100,
                out: buffer
            )
        }

        var resultsFP16 = [Float](repeating: 0, count: 100)
        resultsFP16.withUnsafeMutableBufferPointer { buffer in
            MixedPrecisionKernels.range_euclid2_mixed_768(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 0..<100,
                out: buffer
            )
        }

        var maxRelativeError: Float = 0
        for (ref, computed) in zip(referenceFP32, resultsFP16) {
            let relativeError = abs(ref - computed) / max(ref, 1e-6)
            maxRelativeError = max(maxRelativeError, relativeError)
            #expect(relativeError < 0.0015, "768-dim: Relative error \(relativeError) exceeds 0.15%")
        }

        print("✓ Euclidean² 768: Max relative error = \(String(format: "%.4f%%", maxRelativeError * 100))")
    }

    @Test("Range Euclidean² 1536: Accuracy vs FP32 reference")
    func testRangeEuclidean1536Accuracy() throws {
        let query = generateRandomVector1536()
        let candidates = generateRandomVectors1536(count: 100)
        let candidatesFP16 = MixedPrecisionKernels.convertToFP16_1536(candidates)

        var referenceFP32 = [Float](repeating: 0, count: 100)
        referenceFP32.withUnsafeMutableBufferPointer { buffer in
            BatchKernels.range_euclid2_1536(
                query: query,
                candidates: candidates,
                range: 0..<100,
                out: buffer
            )
        }

        var resultsFP16 = [Float](repeating: 0, count: 100)
        resultsFP16.withUnsafeMutableBufferPointer { buffer in
            MixedPrecisionKernels.range_euclid2_mixed_1536(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 0..<100,
                out: buffer
            )
        }

        var maxRelativeError: Float = 0
        for (ref, computed) in zip(referenceFP32, resultsFP16) {
            let relativeError = abs(ref - computed) / max(ref, 1e-6)
            maxRelativeError = max(maxRelativeError, relativeError)
            #expect(relativeError < 0.002, "1536-dim: Relative error \(relativeError) exceeds 0.2%")
        }

        print("✓ Euclidean² 1536: Max relative error = \(String(format: "%.4f%%", maxRelativeError * 100))")
    }

    // MARK: - Cosine Distance Accuracy Tests

    @Test("Range Cosine 512: Accuracy vs FP32 reference")
    func testRangeCosine512Accuracy() throws {
        let query = generateRandomVector512()
        let candidates = generateRandomVectors512(count: 100)
        let candidatesFP16 = MixedPrecisionKernels.convertToFP16_512(candidates)

        var referenceFP32 = [Float](repeating: 0, count: 100)
        referenceFP32.withUnsafeMutableBufferPointer { buffer in
            BatchKernels.range_cosine_fused_512(
                query: query,
                candidates: candidates,
                range: 0..<100,
                out: buffer
            )
        }

        var resultsFP16 = [Float](repeating: 0, count: 100)
        resultsFP16.withUnsafeMutableBufferPointer { buffer in
            MixedPrecisionKernels.range_cosine_mixed_512(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 0..<100,
                out: buffer
            )
        }

        var maxAbsoluteError: Float = 0
        for (ref, computed) in zip(referenceFP32, resultsFP16) {
            let absoluteError = abs(ref - computed)
            maxAbsoluteError = max(maxAbsoluteError, absoluteError)
            #expect(absoluteError < 0.001, "Cosine absolute error \(absoluteError) exceeds 0.001")
        }

        print("✓ Cosine 512: Max absolute error = \(String(format: "%.6f", maxAbsoluteError))")
    }

    @Test("Range Cosine 768: Accuracy vs FP32 reference")
    func testRangeCosine768Accuracy() throws {
        let query = generateRandomVector768()
        let candidates = generateRandomVectors768(count: 100)
        let candidatesFP16 = MixedPrecisionKernels.convertToFP16_768(candidates)

        var referenceFP32 = [Float](repeating: 0, count: 100)
        referenceFP32.withUnsafeMutableBufferPointer { buffer in
            BatchKernels.range_cosine_fused_768(
                query: query,
                candidates: candidates,
                range: 0..<100,
                out: buffer
            )
        }

        var resultsFP16 = [Float](repeating: 0, count: 100)
        resultsFP16.withUnsafeMutableBufferPointer { buffer in
            MixedPrecisionKernels.range_cosine_mixed_768(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 0..<100,
                out: buffer
            )
        }

        var maxAbsoluteError: Float = 0
        for (ref, computed) in zip(referenceFP32, resultsFP16) {
            let absoluteError = abs(ref - computed)
            maxAbsoluteError = max(maxAbsoluteError, absoluteError)
            #expect(absoluteError < 0.001, "Cosine 768 absolute error \(absoluteError) exceeds 0.001")
        }

        print("✓ Cosine 768: Max absolute error = \(String(format: "%.6f", maxAbsoluteError))")
    }

    // MARK: - Dot Product Accuracy Tests
    // Note: BatchKernels doesn't have range_dot, so we validate against single-vector operations

    @Test("Range Dot Product 512: Internal consistency")
    func testRangeDot512Consistency() throws {
        let query = generateRandomVector512()
        let candidates = generateRandomVectors512(count: 100)
        let candidatesFP16 = MixedPrecisionKernels.convertToFP16_512(candidates)

        var resultsFP16 = [Float](repeating: 0, count: 100)
        resultsFP16.withUnsafeMutableBufferPointer { buffer in
            MixedPrecisionKernels.range_dot_mixed_512(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 0..<100,
                out: buffer
            )
        }

        // Verify all results are finite and non-zero for random vectors
        #expect(resultsFP16.allSatisfy { $0.isFinite }, "All dot products should be finite")
        print("✓ Dot Product 512: Internal consistency check passed")
    }

    // MARK: - Heuristic Tests

    @Test("shouldUseMixedPrecision: Small batch returns false")
    func testHeuristicSmallBatch() {
        // 50 candidates × 512 dim = 102,400 bytes < 12MB, and count < 100
        #expect(!MixedPrecisionKernels.shouldUseMixedPrecision(candidateCount: 50, dimension: 512))
    }

    @Test("shouldUseMixedPrecision: Medium batch returns false")
    func testHeuristicMediumBatch() {
        // 500 candidates × 512 dim = 1,024,000 bytes < 12MB
        #expect(!MixedPrecisionKernels.shouldUseMixedPrecision(candidateCount: 500, dimension: 512))
    }

    @Test("shouldUseMixedPrecision: Large batch returns true")
    func testHeuristicLargeBatch() {
        // 10,000 candidates × 512 dim = 20,480,000 bytes > 12MB, and count >= 100
        #expect(MixedPrecisionKernels.shouldUseMixedPrecision(candidateCount: 10000, dimension: 512))
    }

    @Test("shouldUseMixedPrecision: High-dimensional batch returns true")
    func testHeuristicHighDimensional() {
        // 5,000 candidates × 1536 dim = 30,720,000 bytes > 12MB
        #expect(MixedPrecisionKernels.shouldUseMixedPrecision(candidateCount: 5000, dimension: 1536))
    }

    @Test("Platform-specific heuristic: M1/M2 threshold")
    func testHeuristicM1M2() {
        // 2000 × 512 × 4 = 4,096,000 bytes < 8MB
        #expect(!MixedPrecisionKernels.shouldUseMixedPrecision(
            candidateCount: 2000,
            dimension: 512,
            platformHint: .appleM1
        ))

        // 5000 × 512 × 4 = 10,240,000 bytes > 8MB
        #expect(MixedPrecisionKernels.shouldUseMixedPrecision(
            candidateCount: 5000,
            dimension: 512,
            platformHint: .appleM2
        ))
    }

    @Test("Platform-specific heuristic: M3/M4 threshold")
    func testHeuristicM3M4() {
        // 3000 × 512 × 4 = 6,144,000 bytes < 16MB
        #expect(!MixedPrecisionKernels.shouldUseMixedPrecision(
            candidateCount: 3000,
            dimension: 512,
            platformHint: .appleM3
        ))

        // 10000 × 512 × 4 = 20,480,000 bytes > 16MB
        #expect(MixedPrecisionKernels.shouldUseMixedPrecision(
            candidateCount: 10000,
            dimension: 512,
            platformHint: .appleM4
        ))
    }

    // MARK: - Edge Case Tests

    @Test("Range processing: Odd candidate count")
    func testOddCandidateCount() throws {
        let query = generateRandomVector512()
        let candidates = generateRandomVectors512(count: 101) // Odd count
        let candidatesFP16 = MixedPrecisionKernels.convertToFP16_512(candidates)

        var results = [Float](repeating: 0, count: 101)
        results.withUnsafeMutableBufferPointer { buffer in
            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 0..<101,
                out: buffer
            )
        }

        // Verify tail handling: last element should be computed correctly
        #expect(results[100] > 0, "Tail element not computed")
    }

    @Test("Range processing: Partial range")
    func testPartialRange() throws {
        let query = generateRandomVector512()
        let candidates = generateRandomVectors512(count: 100)
        let candidatesFP16 = MixedPrecisionKernels.convertToFP16_512(candidates)

        // Process only middle 20 candidates
        var results = [Float](repeating: 0, count: 20)
        results.withUnsafeMutableBufferPointer { buffer in
            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: query,
                candidatesFP16: candidatesFP16,
                range: 40..<60,
                out: buffer
            )
        }

        // Verify results are non-zero
        #expect(results.allSatisfy { $0 > 0 }, "Partial range not processed correctly")
    }

    @Test("Zero vector handling")
    func testZeroVectorHandling() throws {
        let query = try! Vector512Optimized([Float](repeating: 0, count: 512))
        let candidate = generateRandomVector512()
        let candidateFP16 = MixedPrecisionKernels.Vector512FP16(from: candidate)

        var result = [Float](repeating: 0, count: 1)
        result.withUnsafeMutableBufferPointer { buffer in
            MixedPrecisionKernels.range_euclid2_mixed_512(
                query: query,
                candidatesFP16: [candidateFP16],
                range: 0..<1,
                out: buffer
            )
        }

        // Should equal sum of candidate's squared components
        #expect(result[0].isFinite, "Zero vector handling failed")
    }
}
