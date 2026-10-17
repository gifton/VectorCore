import Foundation
import Testing
@testable import VectorCore

@Suite("R03 matrix metric and logical output contracts")
struct MatrixMetricContractTests {
    @Test func matrixDiagonalAndSimilarityBehavioralRegression() async throws {
        let rows = [DynamicVector([3, 4]), DynamicVector([0, 0])]
        let batchDot = await BatchOperations.pairwiseDistances(rows, metric: DotProductDistance())
        let executionDot = try await ExecutionOperations.distanceMatrix(rows, metric: DotProductDistance())
        #expect(batchDot[0][0] == -25)
        #expect(executionDot[0][0] == -25)
        let batchCosine = await BatchOperations.pairwiseDistances(rows, metric: CosineDistance())
        let executionCosine = try await ExecutionOperations.distanceMatrix(rows, metric: CosineDistance())
        #expect(batchCosine[1][1] == 1)
        #expect(executionCosine[1][1] == 1)
        let a = try Vector<Dim768>([3, 4] + Array(repeating: 0, count: 766))
        let similarity = try await [a, a].cosineSimilarityMatrix()
        #expect(similarity[0][0].isFinite && abs(similarity[0][0] - 1) < 1e-6)
        #expect(similarity[0][1].isFinite && abs(similarity[0][1] - 1) < 1e-6)
    }

    @Test func zeroQuerySoASuffixBehavioralRegression() throws {
        let zero = try Vector512Optimized(Array(repeating: 0, count: 512))
        let soa = SoA512.build(from: [zero])
        let sentinel = Float(bitPattern: 0x42F68001)
        for route in 0..<3 {
            var values = [sentinel, sentinel, sentinel, sentinel]
            // The values are read through the buffer inside the closure and asserted on the returned copy. Reading
            // `values[i]` after the closure instead is miscompiled by Swift 6.4.0 at -O: SIL RedundantLoadElimination
            // forwards the literal's sentinel across the out-of-line closure call, so the kernel's store is not seen
            // (swiftlang/swift#92562; PR #92580 and its 6.4.x backports are believed to fix this mechanism (unverified for this
            // form; see T15/lost-store-investigation); workaround W2).
            // Scripts/ci/tests/test_lost_store_reads.py rejects that shape.
            let observed: [Float] = values.withUnsafeMutableBufferPointer { all in
                let out = UnsafeMutableBufferPointer(rebasing: all[1..<3])
                switch route {
                case 0: BatchKernels_SoA.cosine_fused_blocked_2way(query: zero, queryNorm: 0, soa: soa, out: out)
                case 1: BatchKernels_SoA.cosine_fused_blocked_4way(query: zero, queryNorm: 0, soa: soa, out: out)
                default: BatchKernels_SoA.cosine_fused_tiled(query: zero, queryNorm: 0, soa: soa, out: out)
                }
                return Array(all)
            }
            #expect(observed[1] == 1)
            #expect(observed[0].bitPattern == sentinel.bitPattern)
            #expect(observed[2].bitPattern == sentinel.bitPattern, "route \(route) wrote logical suffix")
            #expect(observed[3].bitPattern == sentinel.bitPattern)
        }
    }
}

private let r03Patterns: [[Float]] = [[0, 0, 0], [3, 4, 0], [-3, -4, 0], [4, -3, 0], [1, 1, 1], [1, -2, 2]]
private func r03Values(_ index: Int, dimension: Int) -> [Float] {
    r03Patterns[index % r03Patterns.count] + Array(repeating: 0, count: dimension - 3)
}
private final class R03Calls: @unchecked Sendable {
    private let lock = NSLock()
    private var calls = 0
    func increment() { lock.withLock { calls += 1 } }
    var count: Int { lock.withLock { calls } }
}
private struct R03SymmetricMetric: DistanceMetric {
    let offset: Float
    let calls: R03Calls?
    init(offset: Float = 17, calls: R03Calls? = nil) { self.offset = offset; self.calls = calls }
    var name: String { "r03-symmetric-configured" }
    func distance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float {
        calls?.increment()
        return offset + 2 * abs(a[0] - b[0]) + 3 * abs(a[1] - b[1]) + 5 * abs(a[2] - b[2])
    }
}
private func r03Scalar<M: DistanceMetric>(_ metric: M, _ a: [Float], _ b: [Float]) -> Double where M.Scalar == Float {
    let x = a.map(Double.init), y = b.map(Double.init)
    let differences = zip(x, y).map { abs($0 - $1) }
    if metric is EuclideanDistance { return sqrt(differences.reduce(0) { $0 + $1 * $1 }) }
    if metric is CosineDistance {
        let normA = sqrt(x.reduce(0) { $0 + $1 * $1 }), normB = sqrt(y.reduce(0) { $0 + $1 * $1 })
        if normA == 0 || normB == 0 { return 1 }
        return 1 - max(-1, min(1, zip(x, y).reduce(0) { $0 + $1.0 * $1.1 } / (normA * normB)))
    }
    if metric is ManhattanDistance { return differences.reduce(0, +) }
    if metric is DotProductDistance { return -zip(x, y).reduce(0) { $0 + $1.0 * $1.1 } }
    if metric is ChebyshevDistance { return differences.max() ?? 0 }
    if let hamming = metric as? HammingDistance { return Double(zip(a, b).filter { ($0 > hamming.threshold) != ($1 > hamming.threshold) }.count) }
    if let minkowski = metric as? MinkowskiDistance {
        if minkowski.p == .infinity { return differences.max() ?? 0 }
        let p = Double(minkowski.p)
        return pow(differences.reduce(0) { $0 + pow($1, p) }, 1 / p)
    }
    let custom = metric as! R03SymmetricMetric
    return Double(custom.offset) + 2 * differences[0] + 3 * differences[1] + 5 * differences[2]
}
private func r03CheckMatrix<V: VectorProtocol, M: DistanceMetric>(_ matrix: [[Float]], rows: [V], metric: M) where V.Scalar == Float, M.Scalar == Float {
    #expect(matrix.count == rows.count)
    guard matrix.count == rows.count else { return }
    for i in rows.indices {
        #expect(matrix[i].count == rows.count)
        guard matrix[i].count == rows.count else { continue }
        for j in rows.indices {
            let actual = matrix[i][j]
            let direct = metric.distance(rows[i], rows[j])
            let scalar = r03Scalar(metric, r03Patterns[i % 6], r03Patterns[j % 6])
            #expect(actual.isFinite && direct.isFinite && scalar.isFinite)
            #expect(abs(Double(actual) - scalar) <= 2e-5 * max(1, abs(scalar)), "cell \(i),\(j) metric \(metric.name)")
            #expect(abs(actual - direct) <= 2e-5 * max(1, abs(direct)))
            if i == j { #expect(actual.bitPattern == direct.bitPattern, "diagonal must use supplied metric directly") }
        }
    }
}

extension MatrixMetricContractTests {
    @Test(arguments: [0, 1, 2, 99, 100, 101])
    func pairwiseAllBuiltInAndConfiguredMetrics(count: Int) async {
        let rows = (0..<count).map { DynamicVector(r03Values($0, dimension: 3)) }
        let metrics: [any DistanceMetric<Float>] = [EuclideanDistance(), CosineDistance(), ManhattanDistance(), DotProductDistance(), ChebyshevDistance(), HammingDistance(),
            HammingDistance(threshold: 2), MinkowskiDistance(p: 1), MinkowskiDistance(p: 2), MinkowskiDistance(p: 3), MinkowskiDistance(p: .infinity), R03SymmetricMetric()]
        for metric in metrics {
            let matrix = await BatchOperations.pairwiseDistances(rows, metric: metric)
            r03CheckMatrix(matrix, rows: rows, metric: metric)
        }
    }

    @Test(arguments: [0, 1, 2, 31, 32])
    func executionMatrixSerialParallelMetricParity(count: Int) async throws {
        let rows = (0..<count).map { DynamicVector(r03Values($0, dimension: 3)) }
        let metrics: [any DistanceMetric<Float>] = [EuclideanDistance(), CosineDistance(), ManhattanDistance(), DotProductDistance(), ChebyshevDistance(),
            HammingDistance(threshold: 2), MinkowskiDistance(p: 3), R03SymmetricMetric()]
        for metric in metrics {
            let matrix = try await ExecutionOperations.distanceMatrix(rows, metric: metric)
            r03CheckMatrix(matrix, rows: rows, metric: metric)
        }
    }

    @Test(arguments: [0, 1, 2, 31, 32])
    func convenienceReturnsSimilarityIncludingZeroRows(count: Int) async throws {
        let rows = try (0..<count).map { try Vector<Dim768>(r03Values($0, dimension: 768)) }
        let matrix = try await rows.cosineSimilarityMatrix()
        #expect(matrix.count == count)
        for i in 0..<count {
            #expect(matrix[i].count == count)
            for j in 0..<count {
                let expected = 1 - r03Scalar(CosineDistance(), r03Patterns[i % 6], r03Patterns[j % 6])
                let value = matrix[i][j]
                #expect(value.isFinite && abs(Double(value) - expected) <= 2e-6)
                #expect(value >= -1 && value <= 1)
            }
        }
    }

    @Test func similarityDoesNotClampNonfiniteScoresIntoFiniteRange() async throws {
        for value: Float in [.nan, .infinity, -.infinity] {
            var values = Array(repeating: Float(0), count: 768); values[0] = value
            let vector = try Vector<Dim768>(values)
            let matrix = try await [vector, vector].cosineSimilarityMatrix()
            #expect(matrix.allSatisfy { $0.allSatisfy { !$0.isFinite } })
        }
    }
}

private func r03CheckSoA<V: SoACompatible & VectorProtocol>(
    dimension: Int, make: ([Float]) throws -> V,
    publicRoute: (V, SoA<V>, UnsafeMutableBufferPointer<Float>) throws -> Void,
    safeRoute: ((V, SoA<V>, UnsafeMutableBufferPointer<Float>) throws -> Void)? = nil
) throws where V.Scalar == Float {
    let patterns = try (0..<6).map { try make(r03Values($0, dimension: dimension)) }
    let gateFloor = BatchKernels_SoA.tilingByteThreshold / (V.lanes * MemoryLayout<SIMD4<Float>>.stride)
    let counts = [0, 1, 3, 4, 5, 255, 256, 257, gateFloor - 1, gateFloor, gateFloor + 1]
    let sentinel = Float(bitPattern: 0x42F68001)
    for count in counts {
        let soa = SoA<V>.build(from: (0..<count).map { patterns[$0 % 6] }, pageAligned: true)
        let logicalBytes = count * V.lanes * MemoryLayout<SIMD4<Float>>.stride
        #expect(soa.layoutDescriptor.logicalByteCount == logicalBytes)
        #expect(soa.layoutDescriptor.allocatedByteCount >= logicalBytes)
        #expect(BatchKernels_SoA.shouldTile(count: count, lanes: V.lanes) == (logicalBytes > 8 * 1024 * 1024))
        for queryIndex in [0, 1] {
            let query = patterns[queryIndex]
            let norm: Float = queryIndex == 0 ? 0 : 5
            let expected = (0..<count).map { r03Scalar(CosineDistance(), r03Patterns[queryIndex], r03Patterns[$0 % 6]) }
            for route in 0..<(safeRoute == nil ? 4 : 5) {
                for (offset, extra) in [(0, 0), (0, 2), (1, 2)] {
                    var values = Array(repeating: sentinel, count: count + extra + offset * 2)
                    func invoke(_ out: UnsafeMutableBufferPointer<Float>) throws {
                        switch route {
                        case 0: try publicRoute(query, soa, out)
                        case 1: BatchKernels_SoA.cosine_fused_blocked_2way(query: query, queryNorm: norm, soa: soa, out: out)
                        case 2: BatchKernels_SoA.cosine_fused_blocked_4way(query: query, queryNorm: norm, soa: soa, out: out)
                        case 3: BatchKernels_SoA.cosine_fused_tiled(query: query, queryNorm: norm, soa: soa, out: out)
                        default: try safeRoute!(query, soa, out)
                        }
                    }
                    if values.isEmpty { try invoke(UnsafeMutableBufferPointer(start: nil, count: 0)) }
                    else {
                        try values.withUnsafeMutableBufferPointer { all in
                            try invoke(UnsafeMutableBufferPointer(rebasing: all[offset..<(offset + count + extra)]))
                        }
                    }
                    for index in values.indices {
                        if index >= offset && index < offset + count {
                            let value = values[index], reference = expected[index - offset]
                            #expect(value.isFinite && abs(Double(value) - reference) <= 3e-6,
                                "dimension \(dimension), N \(count), query \(queryIndex), route \(route), index \(index)")
                        } else {
                            #expect(values[index].bitPattern == sentinel.bitPattern,
                                "outside logical prefix: dimension \(dimension), N \(count), route \(route), index \(index)")
                        }
                    }
                }
            }
        }
    }
}

extension MatrixMetricContractTests {
    @Test(arguments: [512, 768, 1536])
    func soaLogicalOutputMatrix(dimension: Int) throws {
        switch dimension {
        case 512:
            try r03CheckSoA(dimension: 512, make: { try Vector512Optimized($0) },
                publicRoute: { BatchKernels_SoA.batchCosineDistance512(query: $0, soa: $1, out: $2) },
                safeRoute: { try BatchKernels_SoA.batchCosineDistance512Safe(query: $0, soa: $1, out: $2) })
        case 768:
            try r03CheckSoA(dimension: 768, make: { try Vector768Optimized($0) },
                publicRoute: { BatchKernels_SoA.batchCosineDistance768(query: $0, soa: $1, out: $2) })
        default:
            try r03CheckSoA(dimension: 1536, make: { try Vector1536Optimized($0) },
                publicRoute: { BatchKernels_SoA.batchCosineDistance1536(query: $0, soa: $1, out: $2) })
        }
    }

    @Test func safeSoAInsufficientCapacityPreservesOutputAndError() throws {
        let vector = try Vector512Optimized(r03Values(1, dimension: 512))
        let soa = SoA512.build(from: [vector, vector])
        var values: [Float] = [123]
        // Values and the outcome are returned from inside the closure (lost-store workaround W2;
        // swiftlang/swift#92562): nothing written in the closure is read after it.
        let (observed, outcome): ([Float], Result<Void, any Error>) = values.withUnsafeMutableBufferPointer { buffer in
            let outcome = Result { try BatchKernels_SoA.batchCosineDistance512Safe(query: vector, soa: soa, out: buffer) }
            return (Array(buffer), outcome)
        }
        switch outcome {
        case .success:
            Issue.record("expected capacity error")
        case .failure(let error):
            guard let kernelError = error as? BatchKernels_SoA.BatchKernelError,
                  case .bufferTooSmall(let required, let provided) = kernelError else { throw error }
            #expect(required == 2 && provided == 1)
        }
        #expect(observed == [123])
    }
}

private func r03RoutedMatrix<V: VectorProtocol & UnifiedVectorBuffer & Sendable>(
    dimension: Int, count: Int, metricName: String, make: ([Float]) throws -> V
) async throws where V.Scalar == Float {
    let patterns = try (0..<6).map { try make(r03Values($0, dimension: dimension)) }
    let rows = (0..<count).map { patterns[$0 % 6] }
    let config = await BatchOperations.configuration()
    let eligible = config.enableMatrixRouting && count >= config.matrixRoutingMinN && metricName != "custom"
    print("R03_ROUTE_ELIGIBILITY dimension=\(dimension) N=\(count) metric=\(metricName) enabled=\(config.enableMatrixRouting) threshold=\(config.matrixRoutingMinN) eligible=\(eligible)")
    if metricName == "custom" {
        let calls = R03Calls()
        let metric = R03SymmetricMetric(offset: 23, calls: calls)
        let matrix = await BatchOperations.pairwiseDistances(rows, metric: metric)
        #expect(calls.count == (count < 100 ? count * (count + 1) / 2 : count * count))
        r03CheckMatrix(matrix, rows: rows, metric: metric)
    } else if metricName == "euclidean" {
        let matrix = await BatchOperations.pairwiseDistances(rows, metric: EuclideanDistance())
        r03CheckMatrix(matrix, rows: rows, metric: EuclideanDistance())
        if eligible {
            // P02 RD11: the reference is the route's geometry, one prepared/into query row per call. On these integer
            // patterns every Euclidean product and sum is exact, so this equality cannot tell the per-row geometry
            // from one N×N multiply (P02 phase D-F review A-3); the cosine branch below and
            // P02PairwiseRowTests.rowsEqualPreparedIntoPerRow observe it. Here it pins the in-place root formatting
            // and the row placement under the configured gates.
            let prepared = MatrixDistance.prepare(rows, normalized: false)
            for i in 0..<count {
                var reference = [Float](repeating: 0, count: count)   // swiftlang/swift#92562: no array literal
                MatrixDistance.euclideanSquaredMatrix(queries: [rows[i]], prepared: prepared, into: &reference)
                for j in 0..<count where i != j {
                    let expected = sqrt(max(0, reference[j]))
                    #expect(expected.isFinite && abs(matrix[i][j] - expected) <= 2e-5 * max(1, expected))
                    #expect(matrix[i][j] == expected, "cell \(i),\(j): routed \(matrix[i][j]) vs per-row \(expected)")
                }
            }
        }
    } else {
        let matrix = await BatchOperations.pairwiseDistances(rows, metric: CosineDistance())
        r03CheckMatrix(matrix, rows: rows, metric: CosineDistance())
        if eligible {
            // P02 RD11: the per-row prepared/into geometry of the route. Cosine normalization rounds on these
            // patterns, so this equality does tell the per-row geometry from one N×N multiply (P02 phase D-F review
            // A-3: with the N×N reshape restored, mutant MD1, only these cosine scenarios failed).
            let prepared = MatrixDistance.prepare(rows, normalized: true)
            for i in 0..<count {
                var reference = [Float](repeating: 0, count: count)   // swiftlang/swift#92562: no array literal
                MatrixDistance.cosineDistanceMatrix(queries: [rows[i]], prepared: prepared, into: &reference)
                for j in 0..<count where i != j {
                    let expected = reference[j]
                    #expect(expected.isFinite && abs(matrix[i][j] - expected) <= 2e-5)
                    #expect(matrix[i][j] == expected, "cell \(i),\(j): routed \(matrix[i][j]) vs per-row \(expected)")
                }
            }
        }
    }
}

extension MatrixMetricContractTests {
    static var configurationScenarios: [String] {
        [512, 768, 1536].flatMap { width in
            ["euclidean", "cosine"].flatMap { metric in
                [255, 256, 257].map { "\(width)-\(metric)-default-\($0)" }
                + ["\(width)-\(metric)-disabled-257"]
                + [7, 8, 9].map { "\(width)-\(metric)-small-\($0)" }
            } + ["\(width)-custom-default-257"]
        }
    }

    @Test func isolatedConfigurationChild() async throws {
        guard let scenario = ProcessInfo.processInfo.environment["R03_CONFIGURATION_SCENARIO"] else { return }
        let fields = scenario.split(separator: "-").map(String.init)
        let dimension = Int(fields[0])!, metric = fields[1], mode = fields[2], count = Int(fields[3])!
        let original = await BatchOperations.configuration()
        #expect(original.enableMatrixRouting && original.matrixRoutingMinN == 256)
        var requested = original
        requested.enableMatrixRouting = mode != "disabled"
        requested.matrixRoutingMinN = mode == "small" ? 8 : 256
        let snapshot = requested
        try await BatchOperations.updateConfiguration { $0 = snapshot }
        let invocation: [String: Any] = ["argv": CommandLine.arguments,
            "DYLD_FRAMEWORK_PATH": ProcessInfo.processInfo.environment["DYLD_FRAMEWORK_PATH"] ?? "",
            "DYLD_LIBRARY_PATH": ProcessInfo.processInfo.environment["DYLD_LIBRARY_PATH"] ?? "",
            "scenario": scenario]
        print("R03_CHILD_INVOCATION " + String(data: try JSONSerialization.data(withJSONObject: invocation, options: [.sortedKeys]), encoding: .utf8)!)
        print("R03_CONFIGURATION_BODY \(scenario)")
        do {
            switch dimension {
            case 512: try await r03RoutedMatrix(dimension: 512, count: count, metricName: metric, make: { try Vector512Optimized($0) })
            case 768: try await r03RoutedMatrix(dimension: 768, count: count, metricName: metric, make: { try Vector768Optimized($0) })
            default: try await r03RoutedMatrix(dimension: 1536, count: count, metricName: metric, make: { try Vector1536Optimized($0) })
            }
        } catch {
            try await BatchOperations.updateConfiguration { $0 = original }
            throw error
        }
        try await BatchOperations.updateConfiguration { $0 = original }
        let restored = await BatchOperations.configuration()
        #expect(restored.parallelThreshold == original.parallelThreshold)
        #expect(restored.oversubscription == original.oversubscription)
        #expect(restored.minimumChunkSize == original.minimumChunkSize)
        #expect(restored.defaultBatchSize == original.defaultBatchSize)
        #expect(restored.enableMatrixRouting == original.enableMatrixRouting)
        #expect(restored.matrixRoutingMinN == original.matrixRoutingMinN)
        print("R03_CONFIGURATION_COMPLETE \(scenario)")
    }

    @Test(.serialized, arguments: configurationScenarios)
    func configurationScenarioRunsInFreshProcess(scenario: String) throws {
        var environment = ProcessInfo.processInfo.environment
        environment["R03_CONFIGURATION_SCENARIO"] = scenario
        let directory = environment["R03_CAPTURE_DIRECTORY"].map { URL(fileURLWithPath: $0) } ?? FileManager.default.temporaryDirectory
        let log = directory.appendingPathComponent("R03-configuration-\(scenario)-\(UUID().uuidString).log")
        _ = FileManager.default.createFile(atPath: log.path, contents: nil)
        let output = try FileHandle(forWritingTo: log)
        let process = try launchChildProbe(.swiftTesting(filter: "MatrixMetricContractTests/isolatedConfigurationChild"),
                                           environment: environment, output: output, deadline: 15,
                                           cleanup: .terminateThenKill(grace: 1, seconds: 1))
        let timedOut = process.timedOut
        try output.close()
        let text = try String(contentsOf: log, encoding: .utf8)
        print("R03_CHILD_CAPTURE \(scenario) \(log.path)")
        if timedOut {
            throw NSError(domain: "R03.childTimeout", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "child deadline \(scenario): \(log.path)"])
        }
        #expect(!process.isRunning)
        #expect(process.terminationStatus == 0, "\(scenario): \(log.path)")
        #expect(text.contains("R03_CONFIGURATION_BODY \(scenario)"))
        #expect(text.contains("R03_CONFIGURATION_COMPLETE \(scenario)"))
        #expect(text.contains("R03_CHILD_INVOCATION "))
        let fields = scenario.split(separator: "-")
        let minimum = fields[2] == "small" ? 8 : 256
        let eligible = fields[2] != "disabled" && fields[1] != "custom" && Int(fields[3])! >= minimum
        #expect(text.contains("eligible=\(eligible)"))
        #expect(text.contains("Test isolatedConfigurationChild()"))
    }
}
