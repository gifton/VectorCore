import Testing
import Foundation
@testable import VectorCore

@Suite("Mixed Precision - Performance Benchmarks")
struct MixedPrecisionPerformanceBenchmark {

    // MARK: - Benchmark Utilities

    /// Precise timing using mach_absolute_time
    struct PrecisionTimer {
        private static let timebaseInfo: mach_timebase_info_data_t = {
            var info = mach_timebase_info_data_t()
            mach_timebase_info(&info)
            return info
        }()

        static func measure(iterations: Int = 10000, _ block: () -> Void) -> (median: Double, min: Double, max: Double) {
            var samples: [Double] = []

            // Warmup
            for _ in 0..<100 {
                block()
            }

            // Measure
            for _ in 0..<5 {
                let start = mach_absolute_time()
                for _ in 0..<iterations {
                    block()
                }
                let end = mach_absolute_time()

                let elapsed = end - start
                let nanos = Double(elapsed * UInt64(timebaseInfo.numer)) / Double(timebaseInfo.denom)
                let nsPerOp = nanos / Double(iterations)
                samples.append(nsPerOp)
            }

            samples.sort()
            return (
                median: samples[samples.count / 2],
                min: samples.first!,
                max: samples.last!
            )
        }
    }

    /// Generate random test vectors
    func generateVectors512(count: Int) -> ([Vector512Optimized], [MixedPrecisionKernels.Vector512FP16]) {
        let fp32 = (0..<count).map { _ in
            try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1) })
        }
        let fp16 = fp32.map { MixedPrecisionKernels.Vector512FP16(from: $0) }
        return (fp32, fp16)
    }

    // MARK: - Single-Pair Latency Benchmarks

    @Test("Latency: Euclidean 512-dim (all precision combinations)")
    func benchmarkEuclideanLatency512() throws {
        let (fp32Vecs, fp16Vecs) = generateVectors512(count: 2)
        let q32 = fp32Vecs[0]
        let c32 = fp32Vecs[1]
        let q16 = fp16Vecs[0]
        let c16 = fp16Vecs[1]

        print("\n=== Euclidean Distance Latency (512-dim) ===")

        // FP32×FP32 baseline
        let fp32x32 = PrecisionTimer.measure {
            _ = EuclideanKernels.distance512(q32, c32)
        }
        print("FP32×FP32:  \(String(format: "%.2f", fp32x32.median)) ns (min: \(String(format: "%.2f", fp32x32.min)) ns, max: \(String(format: "%.2f", fp32x32.max)) ns)")

        // FP16×FP16
        let fp16x16 = PrecisionTimer.measure {
            _ = MixedPrecisionKernels.euclidean512(q16, c16)
        }
        let speedup16x16 = fp32x32.median / fp16x16.median
        print("FP16×FP16:  \(String(format: "%.2f", fp16x16.median)) ns (speedup: \(String(format: "%.2fx", speedup16x16)))")

        // FP32×FP16
        let fp32x16 = PrecisionTimer.measure {
            _ = MixedPrecisionKernels.euclidean512(query: q32, candidate: c16)
        }
        let speedup32x16 = fp32x32.median / fp32x16.median
        print("FP32×FP16:  \(String(format: "%.2f", fp32x16.median)) ns (speedup: \(String(format: "%.2fx", speedup32x16)))")

        // FP16×FP32
        let fp16x32 = PrecisionTimer.measure {
            _ = MixedPrecisionKernels.euclidean512(query: q16, candidate: c32)
        }
        let speedup16x32 = fp32x32.median / fp16x32.median
        print("FP16×FP32:  \(String(format: "%.2f", fp16x32.median)) ns (speedup: \(String(format: "%.2fx", speedup16x32)))")

        // Validate speedup expectations
        #expect(speedup16x16 > 1.0, "FP16×FP16 should be faster than FP32×FP32")
        #expect(speedup32x16 > 1.0, "FP32×FP16 should show some speedup")
    }

    @Test("Latency: Cosine 512-dim (all precision combinations)")
    func benchmarkCosineLatency512() throws {
        let (fp32Vecs, fp16Vecs) = generateVectors512(count: 2)
        let q32 = fp32Vecs[0]
        let c32 = fp32Vecs[1]
        let q16 = fp16Vecs[0]
        let c16 = fp16Vecs[1]

        print("\n=== Cosine Distance Latency (512-dim) ===")

        // FP32×FP32 baseline
        let fp32x32 = PrecisionTimer.measure {
            _ = CosineKernels.distance512_fused(q32, c32)
        }
        print("FP32×FP32:  \(String(format: "%.2f", fp32x32.median)) ns")

        // FP16×FP16
        let fp16x16 = PrecisionTimer.measure {
            _ = MixedPrecisionKernels.cosine512(q16, c16)
        }
        let speedup16x16 = fp32x32.median / fp16x16.median
        print("FP16×FP16:  \(String(format: "%.2f", fp16x16.median)) ns (speedup: \(String(format: "%.2fx", speedup16x16)))")

        // FP32×FP16
        let fp32x16 = PrecisionTimer.measure {
            _ = MixedPrecisionKernels.cosine512(query: q32, candidate: c16)
        }
        let speedup32x16 = fp32x32.median / fp32x16.median
        print("FP32×FP16:  \(String(format: "%.2f", fp32x16.median)) ns (speedup: \(String(format: "%.2fx", speedup32x16)))")

        #expect(speedup16x16 > 1.0, "FP16×FP16 should be faster")
    }

    // MARK: - Batch Throughput Benchmarks

    @Test("Throughput: Euclidean batch scaling (N = 10, 100, 1000)")
    func benchmarkEuclideanBatchThroughput() throws {
        let batchSizes = [10, 100, 1000]

        print("\n=== Euclidean Batch Throughput (512-dim) ===")
        print(String(format: "%-10s | %-15s | %-15s | %-10s | %-15s", "Batch Size", "FP32 (M/sec)", "FP16 (M/sec)", "Speedup", "Memory Saved"))
        print(String(repeating: "-", count: 80))

        for n in batchSizes {
            let (fp32Cands, fp16Cands) = generateVectors512(count: n)
            let query32 = fp32Cands[0]
            let query16 = fp16Cands[0]

            // FP32 baseline
            let fp32Time = PrecisionTimer.measure(iterations: 1000) {
                for cand in fp32Cands {
                    _ = EuclideanKernels.distance512(query32, cand)
                }
            }
            let fp32Throughput = (Double(n) / fp32Time.median) * 1000.0  // M/sec

            // FP16
            let fp16Time = PrecisionTimer.measure(iterations: 1000) {
                for cand in fp16Cands {
                    _ = MixedPrecisionKernels.euclidean512(query16, cand)
                }
            }
            let fp16Throughput = (Double(n) / fp16Time.median) * 1000.0  // M/sec

            let speedup = fp16Throughput / fp32Throughput
            let memorySaved = n * 1024  // bytes

            print(String(format: "%-10d | %-15.2f | %-15.2f | %-10.2fx | %-15s",
                n, fp32Throughput, fp16Throughput, speedup, "\(memorySaved / 1024) KB"))
        }
    }

    // MARK: - Dimension Scaling Benchmarks

    @Test("Dimension scaling: Euclidean performance across dimensions")
    func benchmarkDimensionScaling() throws {
        print("\n=== Dimension Scaling (Euclidean FP16×FP16) ===")
        print(String(format: "%-10s | %-15s | %-15s | %-10s", "Dimension", "FP32 (ns)", "FP16 (ns)", "Speedup"))
        print(String(repeating: "-", count: 60))

        // 512-dim
        let (fp32_512, fp16_512) = generateVectors512(count: 2)
        let fp32Time512 = PrecisionTimer.measure {
            _ = EuclideanKernels.distance512(fp32_512[0], fp32_512[1])
        }
        let fp16Time512 = PrecisionTimer.measure {
            _ = MixedPrecisionKernels.euclidean512(fp16_512[0], fp16_512[1])
        }
        print(String(format: "%-10d | %-15.2f | %-15.2f | %-10.2fx",
            512, fp32Time512.median, fp16Time512.median, fp32Time512.median / fp16Time512.median))

        // 768-dim
        let fp32_768 = try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        let cand32_768 = try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1) })
        let fp16_768 = MixedPrecisionKernels.Vector768FP16(from: fp32_768)
        let cand16_768 = MixedPrecisionKernels.Vector768FP16(from: cand32_768)

        let fp32Time768 = PrecisionTimer.measure {
            _ = EuclideanKernels.distance768(fp32_768, cand32_768)
        }
        let fp16Time768 = PrecisionTimer.measure {
            _ = MixedPrecisionKernels.euclidean768(fp16_768, cand16_768)
        }
        print(String(format: "%-10d | %-15.2f | %-15.2f | %-10.2fx",
            768, fp32Time768.median, fp16Time768.median, fp32Time768.median / fp16Time768.median))

        // 1536-dim
        let fp32_1536 = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        let cand32_1536 = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1) })
        let fp16_1536 = MixedPrecisionKernels.Vector1536FP16(from: fp32_1536)
        let cand16_1536 = MixedPrecisionKernels.Vector1536FP16(from: cand32_1536)

        let fp32Time1536 = PrecisionTimer.measure {
            _ = EuclideanKernels.distance1536(fp32_1536, cand32_1536)
        }
        let fp16Time1536 = PrecisionTimer.measure {
            _ = MixedPrecisionKernels.euclidean1536(fp16_1536, cand16_1536)
        }
        print(String(format: "%-10d | %-15.2f | %-15.2f | %-10.2fx",
            1536, fp32Time1536.median, fp16Time1536.median, fp32Time1536.median / fp16Time1536.median))

        // Higher dimensions should show higher speedup
        #expect(fp32Time1536.median / fp16Time1536.median >= fp32Time512.median / fp16Time512.median,
                "Higher dimensions should show equal or better FP16 speedup")
    }

    // MARK: - Accuracy Measurements

    @Test("Accuracy: Relative error measurements across operations")
    func measureAccuracyLoss() throws {
        print("\n=== Accuracy Measurements ===")
        print(String(format: "%-15s | %-20s | %-20s", "Operation", "FP16×FP16 Error %", "FP32×FP16 Error %"))
        print(String(repeating: "-", count: 60))

        let trials = 100
        var euclideanErrors16x16: [Float] = []
        var euclideanErrors32x16: [Float] = []
        var cosineErrors16x16: [Float] = []
        var cosineErrors32x16: [Float] = []

        for _ in 0..<trials {
            let (fp32Vecs, fp16Vecs) = generateVectors512(count: 2)
            let q32 = fp32Vecs[0]
            let c32 = fp32Vecs[1]
            let q16 = fp16Vecs[0]
            let c16 = fp16Vecs[1]

            // Euclidean
            let eucRef = EuclideanKernels.distance512(q32, c32)
            let euc16x16 = MixedPrecisionKernels.euclidean512(q16, c16)
            let euc32x16 = MixedPrecisionKernels.euclidean512(query: q32, candidate: c16)

            euclideanErrors16x16.append(abs(euc16x16 - eucRef) / max(eucRef, 1e-6))
            euclideanErrors32x16.append(abs(euc32x16 - eucRef) / max(eucRef, 1e-6))

            // Cosine
            let cosRef = CosineKernels.distance512_fused(q32, c32)
            let cos16x16 = MixedPrecisionKernels.cosine512(q16, c16)
            let cos32x16 = MixedPrecisionKernels.cosine512(query: q32, candidate: c16)

            cosineErrors16x16.append(abs(cos16x16 - cosRef) / max(abs(cosRef), 1e-6))
            cosineErrors32x16.append(abs(cos32x16 - cosRef) / max(abs(cosRef), 1e-6))
        }

        let avgEucError16x16 = euclideanErrors16x16.reduce(0, +) / Float(trials)
        let avgEucError32x16 = euclideanErrors32x16.reduce(0, +) / Float(trials)
        let avgCosError16x16 = cosineErrors16x16.reduce(0, +) / Float(trials)
        let avgCosError32x16 = cosineErrors32x16.reduce(0, +) / Float(trials)

        print(String(format: "%-15s | %-20.4f | %-20.4f", "Euclidean",
            avgEucError16x16 * 100, avgEucError32x16 * 100))
        print(String(format: "%-15s | %-20.4f | %-20.4f", "Cosine",
            avgCosError16x16 * 100, avgCosError32x16 * 100))

        // Validate errors are acceptable
        #expect(avgEucError16x16 < 0.01, "Euclidean FP16×FP16 error should be < 1%")
        #expect(avgEucError32x16 < 0.01, "Euclidean FP32×FP16 error should be < 1%")
        #expect(avgCosError16x16 < 0.01, "Cosine FP16×FP16 error should be < 1%")
        #expect(avgCosError32x16 < 0.01, "Cosine FP32×FP16 error should be < 1%")
    }

    // MARK: - Memory Footprint Analysis

    @Test("Memory footprint comparison")
    func analyzeMemoryFootprint() throws {
        print("\n=== Memory Footprint Analysis ===")

        let dimensions = [512, 768, 1536]
        print(String(format: "%-10s | %-15s | %-15s | %-15s", "Dimension", "FP32 (bytes)", "FP16 (bytes)", "Savings %"))
        print(String(repeating: "-", count: 60))

        for dim in dimensions {
            let fp32Size: Int
            let fp16Size: Int

            switch dim {
            case 512:
                fp32Size = MemoryLayout<Vector512Optimized>.stride
                fp16Size = MemoryLayout<MixedPrecisionKernels.Vector512FP16>.stride
            case 768:
                fp32Size = MemoryLayout<Vector768Optimized>.stride
                fp16Size = MemoryLayout<MixedPrecisionKernels.Vector768FP16>.stride
            case 1536:
                fp32Size = MemoryLayout<Vector1536Optimized>.stride
                fp16Size = MemoryLayout<MixedPrecisionKernels.Vector1536FP16>.stride
            default:
                continue
            }

            let savings = ((fp32Size - fp16Size) * 100) / fp32Size

            print(String(format: "%-10d | %-15d | %-15d | %-15d%%",
                dim, fp32Size, fp16Size, savings))
        }

        print("\n--- Large-Scale Example ---")
        let numVectors = 100_000
        let fp32Total = numVectors * MemoryLayout<Vector512Optimized>.stride
        let fp16Total = numVectors * MemoryLayout<MixedPrecisionKernels.Vector512FP16>.stride

        print("100,000 vectors (512-dim):")
        print("  FP32: \(fp32Total / (1024 * 1024)) MB")
        print("  FP16: \(fp16Total / (1024 * 1024)) MB")
        print("  Saved: \((fp32Total - fp16Total) / (1024 * 1024)) MB (\((fp32Total - fp16Total) * 100 / fp32Total)%)")
    }

    // MARK: - Heuristic Validation

    @Test("Validate shouldUseMixedPrecision heuristic")
    func validateHeuristic() throws {
        print("\n=== Heuristic Validation ===")
        print(String(format: "%-15s | %-10s | %-20s", "Batch Size", "Dimension", "Recommendation"))
        print(String(repeating: "-", count: 50))

        let testCases: [(Int, Int)] = [
            (10, 512),
            (50, 512),
            (100, 512),
            (500, 512),
            (100, 768),
            (100, 1536),
        ]

        for (batchSize, dimension) in testCases {
            let shouldUse = MixedPrecisionKernels.shouldUseMixedPrecision(
                candidateCount: batchSize,
                dimension: dimension
            )

            print(String(format: "%-15d | %-10d | %-20s",
                batchSize, dimension, shouldUse ? "✅ Use FP16" : "❌ Use FP32"))
        }

        // Validate heuristic logic
        #expect(MixedPrecisionKernels.shouldUseMixedPrecision(candidateCount: 10, dimension: 512) == false,
                "Small batches should use FP32")
        #expect(MixedPrecisionKernels.shouldUseMixedPrecision(candidateCount: 500, dimension: 512) == true,
                "Large batches should use FP16")
    }
}
