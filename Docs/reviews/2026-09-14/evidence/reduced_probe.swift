import Foundation
@testable import VectorCore

@main struct Probe {
    static func main() async throws {
        switch CommandLine.arguments.dropFirst().first ?? "normal" {
        case "quantize-trap":
            let p = LinearQuantizationParams(minValue: -1, maxValue: 1)
            print(p.quantize(1e20))
        case "asymmetric-trap":
            let p = LinearQuantizationParams(minValue: 1e20, maxValue: 1e20, symmetric: false)
            print(p)
        case "zero-cache":
            let cache = SoAFP16Cache512(maxSize: 0)
            _ = await cache.getOrCreate(for: [Vector512Optimized(repeating: 1)])
        default:
            let cache = SoAFP16Cache512()
            let a = Vector512Optimized(repeating: 0)
            var b = a
            b[100] = 1
            _ = await cache.getOrCreate(for: [a])
            let stale = await cache.getOrCreate(for: [b])
            print("cache hashes equal=\(a.hashValue == b.hashValue), input[100]=\(b[100]), cached[100]=\(try stale.getVector(at: 0)[100])")
            for values: [Float] in [[1,2], [-10,-1], [1.5,1.5]] {
                let p = QuantizationSchemes.computeQuantizationParams(values: values, strategy: .perVector)
                let code: [Int8] = values.map { Int8(max(-128, min(127, ($0 / p.scales[0] + p.offsets[0]).rounded()))) }
                let restored = code.map { (Float($0) - p.offsets[0]) * p.scales[0] }
                print("schemes input=\(values), scale=\(p.scales), offset=\(p.offsets), roundtrip=\(restored)")
            }
            let near = Vector512Optimized(repeating: 1.0001)
            let one = Vector512Optimized(repeating: 1)
            print("adaptive strict zero=\(MixedPrecisionKernels.adaptiveEuclideanDistance(query: one, candidate: near, threshold: 0)), fp32=\(one.euclideanDistance(to: near))")
            let big = Vector512Optimized(repeating: 70000)
            print("adaptive candidate overflow=\(MixedPrecisionKernels.adaptiveEuclideanDistance(query: a, candidate: big)), fp32=\(a.euclideanDistance(to: big))")
            let diag = MixedPrecisionDiagnostics.shared
            diag.enable()
            _ = MixedPrecisionKernels.Vector512FP16(from: Vector512Optimized(repeating: 1e-12))
            print("diagnostics subnormal=\(diag.getStatistics().subnormalValues), underflows=\(diag.getStatistics().underflowToZero)")
            diag.disable()
            let tuner = QuantizedAutoTuner.shared
            tuner.clearCache()
            let first = tuner.selectOptimalQuantization(for: "dot", dimension: 512, candidateCount: 2000, accuracyRequirement: 0.02)
            let cached = tuner.selectOptimalQuantization(for: "dot", dimension: 512, candidateCount: 2000, accuracyRequirement: 0.2)
            tuner.clearCache()
            let fresh = tuner.selectOptimalQuantization(for: "dot", dimension: 512, candidateCount: 2000, accuracyRequirement: 0.2)
            print("quantized tuner first=\(first), cached=\(cached), fresh=\(fresh)")
            let q = Vector512INT8(repeating: 1e10)
            print("quantized cosine self 1e10=\(q.cosineDistance(to: q))")
            let nanVector = Vector512Optimized(repeating: .nan)
            print("precision equality nan zero=\(PrecisionValidator.validateVectorEquality512(nanVector, a, tolerance: 0.001))")
        }
    }
}
