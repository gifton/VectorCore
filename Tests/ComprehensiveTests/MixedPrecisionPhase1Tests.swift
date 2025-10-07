import Testing
import Foundation
@testable import VectorCore

@Suite("Mixed Precision Phase 1 - Performance Enhancements")
struct MixedPrecisionPhase1Tests {

    // MARK: - Helper Functions

    func generateRandomVector512() -> Vector512Optimized {
        return try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
    }

    func generateRandomVectors512(count: Int) -> [Vector512Optimized] {
        return (0..<count).map { _ in generateRandomVector512() }
    }

    // MARK: - Register-Blocked Batch Tests

    @Test("batchEuclideanBlocked512: FP16 query × FP16 SoA - Correctness")
    func testBlockedBatchFP16x16_Accuracy() throws {
        let query = generateRandomVector512()
        let candidates = generateRandomVectors512(count: 24)  // 3 blocks of 8

        let queryFP16 = MixedPrecisionKernels.Vector512FP16(from: query)
        let candidatesSoAFP16 = MixedPrecisionKernels.createSoA512FP16(from: candidates)

        // Compute reference with standard batch kernel
        var resultsStandard = [Float](repeating: 0, count: 24)
        resultsStandard.withUnsafeMutableBufferPointer { buffer in
            MixedPrecisionKernels.batchEuclidean512(
                query: queryFP16,
                candidates: candidatesSoAFP16,
                results: buffer
            )
        }

        // Compute with blocked kernel
        var resultsBlocked = [Float](repeating: 0, count: 24)
        resultsBlocked.withUnsafeMutableBufferPointer { buffer in
            MixedPrecisionKernels.batchEuclideanBlocked512(
                query: queryFP16,
                candidates: candidatesSoAFP16,
                results: buffer
            )
        }

        // Validate results match
        for (idx, (standard, blocked)) in zip(resultsStandard, resultsBlocked).enumerated() {
            let relativeError = abs(standard - blocked) / max(standard, 1e-6)
            #expect(relativeError < 1e-5, "Index \(idx): blocked result \(blocked) differs from standard \(standard) by \(relativeError)")
        }

        print("✓ Blocked kernel matches standard kernel for FP16×FP16 (24 candidates)")
    }

    @Test("batchEuclideanBlocked512: FP32 query × FP16 SoA - Correctness")
    func testBlockedBatchFP32x16_Accuracy() throws {
        let query = generateRandomVector512()
        let candidates = generateRandomVectors512(count: 16)  // 2 blocks of 8

        let candidatesSoAFP16 = MixedPrecisionKernels.createSoA512FP16(from: candidates)

        // Compute reference with standard batch kernel
        var resultsStandard = [Float](repeating: 0, count: 16)
        resultsStandard.withUnsafeMutableBufferPointer { buffer in
            MixedPrecisionKernels.batchEuclidean512(
                query: query,
                candidates: candidatesSoAFP16,
                results: buffer
            )
        }

        // Compute with blocked kernel
        var resultsBlocked = [Float](repeating: 0, count: 16)
        resultsBlocked.withUnsafeMutableBufferPointer { buffer in
            MixedPrecisionKernels.batchEuclideanBlocked512(
                query: query,
                candidates: candidatesSoAFP16,
                results: buffer
            )
        }

        // Validate results match
        for (idx, (standard, blocked)) in zip(resultsStandard, resultsBlocked).enumerated() {
            let relativeError = abs(standard - blocked) / max(standard, 1e-6)
            #expect(relativeError < 1e-5, "Index \(idx): blocked result \(blocked) differs from standard \(standard) by \(relativeError)")
        }

        print("✓ Blocked kernel matches standard kernel for FP32×FP16 (16 candidates)")
    }

    @Test("batchEuclideanBlocked512: Edge case - Odd candidate counts")
    func testBlockedBatch_OddCounts() throws {
        let query = generateRandomVector512()

        // Test various odd sizes
        for candidateCount in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23] {
            let candidates = generateRandomVectors512(count: candidateCount)
            let candidatesSoAFP16 = MixedPrecisionKernels.createSoA512FP16(from: candidates)

            var resultsStandard = [Float](repeating: 0, count: candidateCount)
            resultsStandard.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchEuclidean512(
                    query: query,
                    candidates: candidatesSoAFP16,
                    results: buffer
                )
            }

            var resultsBlocked = [Float](repeating: 0, count: candidateCount)
            resultsBlocked.withUnsafeMutableBufferPointer { buffer in
                MixedPrecisionKernels.batchEuclideanBlocked512(
                    query: query,
                    candidates: candidatesSoAFP16,
                    results: buffer
                )
            }

            for (idx, (standard, blocked)) in zip(resultsStandard, resultsBlocked).enumerated() {
                let relativeError = abs(standard - blocked) / max(standard, 1e-6)
                #expect(relativeError < 1e-5, "N=\(candidateCount), Index \(idx): mismatch")
            }
        }

        print("✓ Blocked kernel handles odd candidate counts correctly")
    }

    @Test("Enhanced batchEuclidean512: SIMD accumulator performance")
    func testEnhancedBatch_SIMDAccumulators() throws {
        let query = generateRandomVector512()
        let candidates = generateRandomVectors512(count: 100)

        let candidatesSoAFP16 = MixedPrecisionKernels.createSoA512FP16(from: candidates)

        // Compute distances
        var results = [Float](repeating: 0, count: 100)
        results.withUnsafeMutableBufferPointer { buffer in
            MixedPrecisionKernels.batchEuclidean512(
                query: query,
                candidates: candidatesSoAFP16,
                results: buffer
            )
        }

        // Validate all results are finite and reasonable
        #expect(results.allSatisfy { $0.isFinite }, "All results should be finite")
        #expect(results.allSatisfy { $0 >= 0 }, "All distances should be non-negative")

        // Verify against FP32 reference
        let candidatesFP32 = candidates
        var referenceFP32 = [Float](repeating: 0, count: 100)
        referenceFP32.withUnsafeMutableBufferPointer { buffer in
            BatchKernels.range_euclid2_512(
                query: query,
                candidates: candidatesFP32,
                range: 0..<100,
                out: buffer
            )
        }

        // Take sqrt of FP32 results for comparison
        let referenceFP32Sqrt = referenceFP32.map { sqrt($0) }

        var maxError: Float = 0
        for (computed, reference) in zip(results, referenceFP32Sqrt) {
            let error = abs(computed - reference) / max(reference, 1e-6)
            maxError = max(maxError, error)
        }

        #expect(maxError < 0.001, "Max relative error \(maxError) should be < 0.1%")
        print("✓ Enhanced batch kernel: max error = \(String(format: "%.4f%%", maxError * 100))")
    }

    @Test("SoA creation: Efficient transposition")
    func testSoACreation_Efficiency() throws {
        let vectors = generateRandomVectors512(count: 100)

        // Create SoA
        let soa = MixedPrecisionKernels.createSoA512FP16(from: vectors)

        // Verify properties
        #expect(soa.vectorCount == 100, "Should store all 100 vectors")
        #expect(soa.dimension == 512, "Should have 512 dimensions")
        #expect(soa.groupCount == 25, "Should have 25 groups (100/4 rounded up)")

        // Verify storage size: 25 groups × 512 dims × 4 candidates = 51,200 UInt16
        #expect(soa.storage.count == 51200, "Storage size should be 51,200 UInt16 values")

        print("✓ SoA creation validated: \(soa.vectorCount) vectors in \(soa.groupCount) groups")
    }
}
