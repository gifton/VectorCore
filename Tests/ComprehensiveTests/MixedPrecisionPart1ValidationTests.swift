import Testing
import Foundation
@testable import VectorCore

@Suite("Mixed Precision Part 1 - Validation")
struct MixedPrecisionPart1ValidationTests {

    // MARK: - Helper Functions

    func generateRandomVector512() -> Vector512Optimized {
        var values: [Float] = []
        for _ in 0..<512 {
            values.append(Float.random(in: -1...1))
        }
        return try! Vector512Optimized(values)
    }

    // MARK: - Euclidean Distance Tests

    @Test("Euclidean 512: FP16×FP16 accuracy")
    func testEuclidean512_FP16x16_Accuracy() throws {
        let query = generateRandomVector512()
        let candidate = generateRandomVector512()

        let queryFP16 = MixedPrecisionKernels.Vector512FP16(from: query)
        let candidateFP16 = MixedPrecisionKernels.Vector512FP16(from: candidate)

        let refFP32 = EuclideanKernels.distance512(query, candidate)
        let resultFP16 = MixedPrecisionKernels.euclidean512(queryFP16, candidateFP16)

        let relativeError = abs(resultFP16 - refFP32) / max(refFP32, 1e-6)

        #expect(relativeError < 0.01, "Relative error \(String(format: "%.4f%%", relativeError * 100)) exceeds 1%")
    }

    @Test("Euclidean 512: FP32×FP16 accuracy")
    func testEuclidean512_FP32x16_Accuracy() throws {
        let query = generateRandomVector512()
        let candidate = generateRandomVector512()
        let candidateFP16 = MixedPrecisionKernels.Vector512FP16(from: candidate)

        let refFP32 = EuclideanKernels.distance512(query, candidate)
        let resultMixed = MixedPrecisionKernels.euclidean512(query: query, candidate: candidateFP16)

        let relativeError = abs(resultMixed - refFP32) / max(refFP32, 1e-6)

        #expect(relativeError < 0.01, "Relative error \(String(format: "%.4f%%", relativeError * 100)) exceeds 1%")
    }

    @Test("Euclidean 512: FP16×FP32 accuracy")
    func testEuclidean512_FP16x32_Accuracy() throws {
        let query = generateRandomVector512()
        let candidate = generateRandomVector512()
        let queryFP16 = MixedPrecisionKernels.Vector512FP16(from: query)

        let refFP32 = EuclideanKernels.distance512(query, candidate)
        let resultMixed = MixedPrecisionKernels.euclidean512(query: queryFP16, candidate: candidate)

        let relativeError = abs(resultMixed - refFP32) / max(refFP32, 1e-6)

        #expect(relativeError < 0.01, "Relative error \(String(format: "%.4f%%", relativeError * 100)) exceeds 1%")
    }

    @Test("Euclidean 768: FP32×FP16 accuracy")
    func testEuclidean768_Mixed_Accuracy() throws {
        var values: [Float] = []
        for _ in 0..<768 {
            values.append(Float.random(in: -1...1))
        }
        let query = try Vector768Optimized(values)
        let candidate = try Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        let candidateFP16 = MixedPrecisionKernels.Vector768FP16(from: candidate)

        let refFP32 = EuclideanKernels.distance768(query, candidate)
        let resultMixed = MixedPrecisionKernels.euclidean768(query: query, candidate: candidateFP16)

        let relativeError = abs(resultMixed - refFP32) / max(refFP32, 1e-6)

        #expect(relativeError < 0.01, "768-dim: Relative error \(String(format: "%.4f%%", relativeError * 100)) exceeds 1%")
    }

    @Test("Euclidean 1536: FP32×FP16 accuracy")
    func testEuclidean1536_Mixed_Accuracy() throws {
        let query = try Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        let candidate = try Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        let candidateFP16 = MixedPrecisionKernels.Vector1536FP16(from: candidate)

        let refFP32 = EuclideanKernels.distance1536(query, candidate)
        let resultMixed = MixedPrecisionKernels.euclidean1536(query: query, candidate: candidateFP16)

        let relativeError = abs(resultMixed - refFP32) / max(refFP32, 1e-6)

        #expect(relativeError < 0.01, "1536-dim: Relative error \(String(format: "%.4f%%", relativeError * 100)) exceeds 1%")
    }

    // MARK: - Cosine Distance Tests

    @Test("Cosine 512: FP16×FP16 accuracy")
    func testCosine512_FP16x16_Accuracy() throws {
        let query = generateRandomVector512()
        let candidate = generateRandomVector512()
        let queryFP16 = MixedPrecisionKernels.Vector512FP16(from: query)
        let candidateFP16 = MixedPrecisionKernels.Vector512FP16(from: candidate)

        let refFP32 = CosineKernels.distance512_fused(query, candidate)
        let resultFP16 = MixedPrecisionKernels.cosine512(queryFP16, candidateFP16)

        let relativeError = abs(resultFP16 - refFP32) / max(abs(refFP32), 1e-6)

        #expect(relativeError < 0.01, "Cosine: Relative error \(String(format: "%.4f%%", relativeError * 100)) exceeds 1%")
    }

    @Test("Cosine 512: FP32×FP16 accuracy")
    func testCosine512_FP32x16_Accuracy() throws {
        let query = generateRandomVector512()
        let candidate = generateRandomVector512()
        let candidateFP16 = MixedPrecisionKernels.Vector512FP16(from: candidate)

        let refFP32 = CosineKernels.distance512_fused(query, candidate)
        let resultMixed = MixedPrecisionKernels.cosine512(query: query, candidate: candidateFP16)

        let relativeError = abs(resultMixed - refFP32) / max(abs(refFP32), 1e-6)

        #expect(relativeError < 0.01, "Cosine FP32×FP16: Relative error \(String(format: "%.4f%%", relativeError * 100)) exceeds 1%")
    }

    @Test("Cosine 768: FP32×FP16 accuracy")
    func testCosine768_Mixed_Accuracy() throws {
        let query = try Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        let candidate = try Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        let candidateFP16 = MixedPrecisionKernels.Vector768FP16(from: candidate)

        let refFP32 = CosineKernels.distance768_fused(query, candidate)
        let resultMixed = MixedPrecisionKernels.cosine768(query: query, candidate: candidateFP16)

        let relativeError = abs(resultMixed - refFP32) / max(abs(refFP32), 1e-6)

        #expect(relativeError < 0.01, "Cosine 768: Relative error \(String(format: "%.4f%%", relativeError * 100)) exceeds 1%")
    }

    // MARK: - Memory Footprint Tests

    @Test("Memory savings validation")
    func testMemoryFootprint() throws {
        let fp32Size = MemoryLayout<Vector512Optimized>.stride
        let fp16Size = MemoryLayout<MixedPrecisionKernels.Vector512FP16>.stride

        // FP16 should be roughly 50% of FP32 size
        let ratio = Double(fp16Size) / Double(fp32Size)

        #expect(ratio < 0.6 && ratio > 0.4, "Memory ratio \(String(format: "%.2f", ratio)) not close to 50%")
    }

    // MARK: - Edge Case Tests

    @Test("Zero vectors handling")
    func testZeroVectors() throws {
        let zero = try Vector512Optimized(Array(repeating: 0.0, count: 512))
        let zeroFP16 = MixedPrecisionKernels.Vector512FP16(from: zero)
        let nonZero = generateRandomVector512()
        let nonZeroFP16 = MixedPrecisionKernels.Vector512FP16(from: nonZero)

        // Euclidean with zeros
        let eucZero = MixedPrecisionKernels.euclidean512(zeroFP16, zeroFP16)
        #expect(eucZero == 0.0, "Zero vector distance should be 0")

        // Cosine with zero should handle gracefully (returns 0 or 1 based on implementation)
        let cosZero = MixedPrecisionKernels.cosine512(zeroFP16, nonZeroFP16)
        #expect(cosZero.isFinite, "Cosine with zero vector should return finite value")
    }

    @Test("Identical vectors")
    func testIdenticalVectors() throws {
        let vec = generateRandomVector512()
        let vecFP16 = MixedPrecisionKernels.Vector512FP16(from: vec)

        let eucDist = MixedPrecisionKernels.euclidean512(vecFP16, vecFP16)
        #expect(eucDist < 0.001, "Identical vector Euclidean distance should be ~0")

        let cosDist = MixedPrecisionKernels.cosine512(vecFP16, vecFP16)
        #expect(cosDist < 0.001, "Identical vector Cosine distance should be ~0")
    }
}
