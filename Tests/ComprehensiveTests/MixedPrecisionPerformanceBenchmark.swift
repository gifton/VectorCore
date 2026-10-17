import Testing
import Foundation
@preconcurrency import Darwin
@testable import VectorCore

@Suite("Mixed Precision - Performance Benchmarks")
struct MixedPrecisionPerformanceBenchmark {

    // MARK: - Benchmark Utilities

    /// Timing with mach_absolute_time (host timebase ticks). The timed bodies discard their results (`_ = ...`), so the
    /// printed figures are observations, not performance evidence (claim ledger recordedSites, Q17).
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
    func generateVectors512(count: Int, seed: UInt64) -> ([Vector512Optimized], [MixedPrecisionKernels.Vector512FP16]) {
        var rng = SeededGenerator(seed: seed)
        let fp32 = (0..<count).map { _ in
            try! Vector512Optimized((0..<512).map { _ in Float.random(in: -1...1, using: &rng) })
        }
        let fp16 = fp32.map { MixedPrecisionKernels.Vector512FP16(from: $0) }
        return (fp32, fp16)
    }

    // MARK: - Single-Pair Latency Benchmarks

    @Test("Latency: Euclidean 512-dim (all precision combinations)", .enabled(if: ProcessInfo.processInfo.environment["VECTORCORE_TEST_EXTENDED"] == "1"))
    func benchmarkEuclideanLatency512() throws {
        let (fp32Vecs, fp16Vecs) = generateVectors512(count: 2, seed: 0xBE070001)
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
        expectPerformance("MixedPrecisionPerformanceBenchmark.benchmarkEuclideanLatency512.1", speedup16x16 > 1.0, "FP16×FP16 should be faster than FP32×FP32")
        expectPerformance("MixedPrecisionPerformanceBenchmark.benchmarkEuclideanLatency512.2", speedup32x16 > 1.0, "FP32×FP16 should show some speedup")
    }

    @Test("Latency: Cosine 512-dim (all precision combinations)", .enabled(if: ProcessInfo.processInfo.environment["VECTORCORE_TEST_EXTENDED"] == "1"))
    func benchmarkCosineLatency512() throws {
        let (fp32Vecs, fp16Vecs) = generateVectors512(count: 2, seed: 0xBE070002)
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

        expectPerformance("MixedPrecisionPerformanceBenchmark.benchmarkCosineLatency512.3", speedup16x16 > 1.0, "FP16×FP16 should be faster")
    }

    // MARK: - Batch Throughput Benchmarks

    @Test("Throughput: Euclidean batch scaling (N = 10, 100, 1000)")
    func benchmarkEuclideanBatchThroughput() throws {
        let batchSizes = [10, 100, 1000]

        print("\n=== Euclidean Batch Throughput (512-dim) ===")
        print("Batch Size | FP32 (M/sec)    | FP16 (M/sec)    | Speedup    | Logical payload difference (model)")
        print(String(repeating: "-", count: 80))

        for n in batchSizes {
            let (fp32Cands, fp16Cands) = generateVectors512(count: n, seed: 0xBE070003 &+ UInt64(n))
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
            // Logical payload model: n x 512 x (4 - 2) bytes of scalars; not a measured saving.
            let payloadDifference = n * 1024  // bytes

            let memStr = "\(payloadDifference / 1024) KB"
            print(String(format: "%-10d | %-15.2f | %-15.2f | %-10.2fx | ", n, fp32Throughput, fp16Throughput, speedup) + memStr.padding(toLength: 15, withPad: " ", startingAt: 0))
        }
    }

    // MARK: - Dimension Scaling Benchmarks

    @Test("Dimension scaling: Euclidean performance across dimensions", .enabled(if: ProcessInfo.processInfo.environment["VECTORCORE_TEST_EXTENDED"] == "1"))
    func benchmarkDimensionScaling() throws {
        var rng = SeededGenerator(seed: 0xBE070004)
        print("\n=== Dimension Scaling (Euclidean FP16×FP16) ===")
        print("Dimension  | FP32 (ns)       | FP16 (ns)       | Speedup   ")
        print(String(repeating: "-", count: 60))

        // 512-dim
        let (fp32_512, fp16_512) = generateVectors512(count: 2, seed: 0xBE070005)
        let fp32Time512 = PrecisionTimer.measure {
            _ = EuclideanKernels.distance512(fp32_512[0], fp32_512[1])
        }
        let fp16Time512 = PrecisionTimer.measure {
            _ = MixedPrecisionKernels.euclidean512(fp16_512[0], fp16_512[1])
        }
        print(String(format: "%-10d | %-15.2f | %-15.2f | %-10.2fx",
            512, fp32Time512.median, fp16Time512.median, fp32Time512.median / fp16Time512.median))

        // 768-dim
        let fp32_768 = try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1, using: &rng) })
        let cand32_768 = try! Vector768Optimized((0..<768).map { _ in Float.random(in: -1...1, using: &rng) })
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
        let fp32_1536 = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1, using: &rng) })
        let cand32_1536 = try! Vector1536Optimized((0..<1536).map { _ in Float.random(in: -1...1, using: &rng) })
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
        expectPerformance("MixedPrecisionPerformanceBenchmark.benchmarkDimensionScaling.4", fp32Time1536.median / fp16Time1536.median >= fp32Time512.median / fp16Time512.median,
                "Higher dimensions should show equal or better FP16 speedup")
    }

    // MARK: - Accuracy Measurements

    @Test("Accuracy: Relative error measurements across operations")
    func measureAccuracyLoss() throws {
        print("\n=== Accuracy Measurements ===")
        print("Operation       | FP16×FP16 Error %    | FP32×FP16 Error %   ")
        print(String(repeating: "-", count: 60))

        let trials = 100
        var euclideanErrors16x16: [Float] = []
        var euclideanErrors32x16: [Float] = []
        var cosineErrors16x16: [Float] = []
        var cosineErrors32x16: [Float] = []

        for trial in 0..<trials {
            let (fp32Vecs, fp16Vecs) = generateVectors512(count: 2, seed: 0xBE070006 &+ UInt64(trial))
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

        print("Euclidean       | " + String(format: "%-20.4f | %-20.4f",
            avgEucError16x16 * 100, avgEucError32x16 * 100))
        print("Cosine          | " + String(format: "%-20.4f | %-20.4f",
            avgCosError16x16 * 100, avgCosError32x16 * 100))

        // Validate errors are acceptable
        #expect(avgEucError16x16 < 0.01, "Euclidean FP16×FP16 error should be < 1%")
        #expect(avgEucError32x16 < 0.01, "Euclidean FP32×FP16 error should be < 1%")
        #expect(avgCosError16x16 < 0.01, "Cosine FP16×FP16 error should be < 1%")
        #expect(avgCosError32x16 < 0.01, "Cosine FP32×FP16 error should be < 1%")
    }

    // MARK: - Representation Memory Model

    /// Logical scalar payload bytes for `dimension` scalars of `scalarStride` bytes each:
    /// $P = D \cdot s$ with $s$ = `MemoryLayout<Scalar>.stride`. Checked arithmetic; `nil` on overflow.
    ///
    /// This is a model of the scalar payload only. The 2:1 FP32:FP16 scalar-width ratio excludes
    /// array headers, spare capacity, allocation rounding, conversion temporaries and sharing
    /// (the stride itself includes per-element padding).
    static func logicalScalarPayloadBytes(dimension: Int, scalarStride: Int) -> Int? {
        let (product, overflow) = dimension.multipliedReportingOverflow(by: scalarStride)
        return overflow ? nil : product
    }

    /// Stored scalar payload bytes of REAL vector storage: `storage.count * MemoryLayout<E>.stride`.
    /// `E` is inferred from the storage itself, so the result is the real storage's count x element
    /// stride and changes whenever that product changes (a change that preserves it, such as a
    /// same-stride element type, does not); that is what makes assertions built on it capable of
    /// failing. Per-element padding is included in the stride. Not allocation size: excludes the array
    /// header, spare capacity and allocation rounding. Plain `*` traps on overflow.
    static func storedScalarPayloadBytes<E>(_ storage: ContiguousArray<E>) -> Int {
        storage.count * MemoryLayout<E>.stride
    }

    @Test("Representation memory model: independent payload constants and overflow")
    func representationPayloadConstants() throws {
        // Literal expected values, deliberately not recomputed with the production formula.
        let expected: [(dimension: Int, fp32: Int, fp16: Int)] = [
            (512, 2048, 1024),
            (768, 3072, 1536),
            (1536, 6144, 3072)
        ]
        for row in expected {
            #expect(Self.logicalScalarPayloadBytes(dimension: row.dimension, scalarStride: MemoryLayout<Float>.stride) == row.fp32)
            #expect(Self.logicalScalarPayloadBytes(dimension: row.dimension, scalarStride: MemoryLayout<UInt16>.stride) == row.fp16)
        }
        // Overflow is reported, not wrapped.
        #expect(Self.logicalScalarPayloadBytes(dimension: Int.max, scalarStride: 4) == nil)
        #expect(Self.logicalScalarPayloadBytes(dimension: Int.max / 2 + 1, scalarStride: 2) == nil)
        #expect(Self.logicalScalarPayloadBytes(dimension: Int.max / 2, scalarStride: 2) == Int.max - 1)
    }

    @Test("Representation memory model: handle stride vs logical payload")
    func analyzeRepresentationMemoryModel() throws {
        print("\n=== Representation Memory Model: handle stride vs logical payload ===")
        print("Handle stride is the size of the owning handle struct, not the memory it owns.")
        print("Logical scalar payload is D x scalar stride; it excludes array headers, spare capacity,")
        print("allocation rounding, conversion temporaries and sharing.")

        let columns = [
            "Dimension",
            "Handle stride FP32 (MemoryLayout<VectorDOptimized>.stride)",
            "Handle stride FP16",
            "Logical scalar payload FP32 (D·MemoryLayout<Float>.stride)",
            "Logical scalar payload FP16 (D·MemoryLayout<UInt16>.stride)",
            "Measured records: not measured here"
        ]
        print(columns.joined(separator: " | "))
        print(String(repeating: "-", count: 80))

        let dimensions = [512, 768, 1536]
        for dim in dimensions {
            let fp32Handle: Int
            let fp16Handle: Int

            switch dim {
            case 512:
                fp32Handle = MemoryLayout<Vector512Optimized>.stride
                fp16Handle = MemoryLayout<MixedPrecisionKernels.Vector512FP16>.stride
            case 768:
                fp32Handle = MemoryLayout<Vector768Optimized>.stride
                fp16Handle = MemoryLayout<MixedPrecisionKernels.Vector768FP16>.stride
            case 1536:
                fp32Handle = MemoryLayout<Vector1536Optimized>.stride
                fp16Handle = MemoryLayout<MixedPrecisionKernels.Vector1536FP16>.stride
            default:
                continue
            }

            let fp32Payload = try #require(Self.logicalScalarPayloadBytes(dimension: dim, scalarStride: MemoryLayout<Float>.stride))
            let fp16Payload = try #require(Self.logicalScalarPayloadBytes(dimension: dim, scalarStride: MemoryLayout<UInt16>.stride))

            print(String(format: "%-10d | %-12d | %-12d | %-12d | %-12d | not measured",
                dim, fp32Handle, fp16Handle, fp32Payload, fp16Payload))
        }

        print("\n--- 100,000-vector example (512-dim), two labelled models ---")
        let numVectors = 100_000
        let fp32Stride = MemoryLayout<Float>.stride
        let fp16Stride = MemoryLayout<UInt16>.stride
        let perVectorFP32 = try #require(Self.logicalScalarPayloadBytes(dimension: 512, scalarStride: fp32Stride))
        let perVectorFP16 = try #require(Self.logicalScalarPayloadBytes(dimension: 512, scalarStride: fp16Stride))
        let (payloadFP32, o1) = numVectors.multipliedReportingOverflow(by: perVectorFP32)
        let (payloadFP16, o2) = numVectors.multipliedReportingOverflow(by: perVectorFP16)
        #expect(!o1 && !o2)
        let (handlesFP32, o3) = numVectors.multipliedReportingOverflow(by: MemoryLayout<Vector512Optimized>.stride)
        let (handlesFP16, o4) = numVectors.multipliedReportingOverflow(by: MemoryLayout<MixedPrecisionKernels.Vector512FP16>.stride)
        #expect(!o3 && !o4)

        print("Logical payload model (numVectors x D x scalar stride; scalars only):")
        print("  FP32 scalars: \(payloadFP32 / (1024 * 1024)) MiB")
        print("  FP16 scalars: \(payloadFP16 / (1024 * 1024)) MiB")
        print("Handle-array model (numVectors x handle stride; handles only, excludes owned storage):")
        print("  FP32 handles: \(handlesFP32 / 1024) KiB")
        print("  FP16 handles: \(handlesFP16 / 1024) KiB")
        print("Neither model is a total footprint or a measured saving.")
    }

    @Test("Representation memory model: columns equal stored payload of real vectors")
    func representationModelMatchesRealStorage() throws {
        // Zero-filled vectors; only the storage shape matters. Expected values are the model's
        // columns (independent literals live in representationPayloadConstants).
        let v512 = try Vector512Optimized(Array(repeating: 0, count: 512))
        let v768 = try Vector768Optimized(Array(repeating: 0, count: 768))
        let v1536 = try Vector1536Optimized(Array(repeating: 0, count: 1536))
        let real: [(dimension: Int, fp32: Int, fp16: Int)] = [
            (512, Self.storedScalarPayloadBytes(v512.storage),
             Self.storedScalarPayloadBytes(MixedPrecisionKernels.Vector512FP16(from: v512).storage)),
            (768, Self.storedScalarPayloadBytes(v768.storage),
             Self.storedScalarPayloadBytes(MixedPrecisionKernels.Vector768FP16(from: v768).storage)),
            (1536, Self.storedScalarPayloadBytes(v1536.storage),
             Self.storedScalarPayloadBytes(MixedPrecisionKernels.Vector1536FP16(from: v1536).storage))
        ]
        for row in real {
            let model32 = try #require(Self.logicalScalarPayloadBytes(dimension: row.dimension, scalarStride: MemoryLayout<Float>.stride))
            let model16 = try #require(Self.logicalScalarPayloadBytes(dimension: row.dimension, scalarStride: MemoryLayout<UInt16>.stride))
            #expect(row.fp32 == model32, "FP32 stored payload vs model at D=\(row.dimension)")
            #expect(row.fp16 == model16, "FP16 stored payload vs model at D=\(row.dimension)")
        }
    }

    // MARK: - Heuristic Validation

    @Test("Validate shouldUseMixedPrecision heuristic")
    func validateHeuristic() throws {
        print("\n=== Heuristic Validation ===")
        print("Batch Size      | Dimension  | Recommendation      ")
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

            let rec = shouldUse ? "Use FP16" : "Use FP32"
            print(String(format: "%-15d | %-10d | ", batchSize, dimension) + rec.padding(toLength: 20, withPad: " ", startingAt: 0))
        }

        // Validate heuristic logic
        #expect(MixedPrecisionKernels.shouldUseMixedPrecision(candidateCount: 10, dimension: 512) == false,
                "Small batches should use FP32")
        #expect(MixedPrecisionKernels.shouldUseMixedPrecision(candidateCount: 500, dimension: 512) == false,
                "500 x 512 should stay below the FP16 heuristic threshold (heuristic output only; no cache effect is measured)")
    }
}
