import Foundation
import Testing
@testable import VectorCore

private enum L03ProbeFailure: Error { case timeout(String), wrongChild }

@Suite(.serialized)
struct UMAPRangeContractTests {
    private func graph(_ scale: Float = 1) throws -> KNNGraph {
        try KNNGraph(pointCount: 3, rowOffsets: [0, 2, 2, 2],
                     neighborIndices: [1, 2], distances: [scale, 2 * scale])
    }

    private func curveSSE(a: Float, b: Float, spread: Float, ratio: Double) -> Double {
        guard a.isFinite, b.isFinite, a > 0, b > 0 else { return .infinity }
        let alpha = Double(a) * Foundation.pow(Double(spread), 2 * Double(b))
        var sum = 0.0
        for i in 0..<300 {
            let t = 3 * Double(i) / 299
            let fitted = 1 / (1 + alpha * Foundation.pow(t, 2 * Double(b)))
            let target = t < ratio ? 1 : Foundation.exp(-(t - ratio))
            sum += (fitted - target) * (fitted - target)
        }
        return sum
    }

    @Test func curveIndependentMinimumAndScale() throws {
        for fixture in L03Reference.curveCases {
            let reference = L03Reference.curves[fixture.reference]
            let spread = Float(bitPattern: fixture.spreadWord)
            let minDist = Float(bitPattern: fixture.minDistWord)
            let result = try UMAPLayout.fitABParams(spread: spread, minDist: minDist)
            let ratio = Double(minDist) / Double(spread)
            let sse = curveSSE(a: result.a, b: result.b, spread: spread, ratio: ratio)
            print("L03_CURVE spreadWord=\(fixture.spreadWord) minDistWord=\(fixture.minDistWord) a=\(result.a) b=\(result.b) sse=\(sse) reference=\(reference.sse)")
            #expect(result.a.isFinite && result.a > 0 && result.b.isFinite && result.b > 0)
            #expect(sse.isFinite && abs(sse - reference.sse) < 1e-5)
            #expect(abs(Double(result.b) - reference.beta) < 3e-5)
            let alpha = Double(result.a) * Foundation.pow(Double(spread), 2 * Double(result.b))
            #expect(alpha.isFinite && abs(Foundation.log(alpha / reference.alpha)) < 3e-5)
            for t in [0.25, 0.5, 1, 2, 3] {
                let actual = 1 / (1 + alpha * Foundation.pow(t, 2 * Double(result.b)))
                let expected = 1 / (1 + reference.alpha * Foundation.pow(t, 2 * reference.beta))
                #expect(actual.isFinite && abs(actual - expected) < 2e-5)
            }
        }
    }

    @Test func curveOrdinaryNonDyadicScales() throws {
        for reference in L03Reference.regressionCurves {
            let spread = Float(bitPattern: reference.spreadWord)
            let minDist = Float(bitPattern: reference.minDistWord)
            let result = try UMAPLayout.fitABParams(spread: spread, minDist: minDist)
            let sse = curveSSE(a: result.a, b: result.b, spread: spread, ratio: Double(minDist) / Double(spread))
            #expect(result.a.isFinite && result.a > 0 && result.b.isFinite && result.b > 0)
            #expect(sse.isFinite && abs(sse - reference.sse) < 1e-5)
            let alpha = Double(result.a) * Foundation.pow(Double(spread), 2 * Double(result.b))
            #expect(alpha.isFinite && abs(Foundation.log(alpha / reference.alpha)) < 3e-5)
            #expect(abs(Double(result.b) - reference.beta) < 3e-5)
        }
    }

    @Test func smoothAnalyticScaleAndTopology() throws {
        let expected = Foundation.log2(3.0) - 1
        for word: UInt32 in [0x0d80_0000, 0x3f80_0000, 0x7180_0000] {
            let scale = Float(bitPattern: word)
            let graph = try graph(scale)
            let calibrated = try FuzzySimplicialSet.smoothDistances(graph)
            let edges = try FuzzySimplicialSet.build(graph)
            let index = edges.heads.indices.first { edges.heads[$0] == 0 && edges.tails[$0] == 2 }
            let weight = index.map { Double(edges.weights[$0]) } ?? -1
            let sigma = Double(calibrated.sigma[0])
            let residual = 1 + Foundation.exp(-Double(scale) / sigma) - Foundation.log2(3.0)
            print("L03_SMOOTH scaleWord=\(word) sigma=\(sigma) weight=\(weight) residual=\(residual)")
            #expect(sigma.isFinite && sigma > 0)
            #expect(residual.isFinite && abs(residual) < 2e-5)
            #expect(weight.isFinite && abs(weight - expected) < 2e-5)
            #expect(edges.count == 4)
        }
    }

    @Test func nonfiniteConfigurationPreflight() {
        for value: Float in [.nan, .infinity, -.infinity] {
            for field in 0..<3 {
                var config = UMAPConfig(neighbors: 2, epochs: 2, initialization: .random)
                if field == 0 { config.spread = value }
                if field == 1 { config.minDist = value; if value == .infinity { config.spread = value } }
                if field == 2 { config.learningRate = value }
                do { try config.validate(); Issue.record("Accepted nonfinite field \(field): \(value)") }
                catch let error as VectorError { #expect(error.kind == .invalidOperation) }
                catch { Issue.record("Wrong error family: \(error)") }
            }
        }
    }

    private func expectKind(_ kind: VectorError.ErrorKind, _ body: () throws -> Void) {
        do { try body(); Issue.record("Expected VectorError.\(kind)") }
        catch let error as VectorError { #expect(error.kind == kind) }
        catch { Issue.record("Wrong error family: \(error)") }
    }

    @Test func curveCoefficientPublicationDomain() throws {
        let reference = L03Reference.curves.last!
        for word: UInt32 in [0x2f00_0000, 0x5000_0000] { // Exact 2^-33 and 2^33.
            let spread = Float(bitPattern: word)
            let result = try UMAPLayout.fitABParams(spread: spread, minDist: spread)
            let sse = curveSSE(a: result.a, b: result.b, spread: spread, ratio: 1)
            #expect(result.a.isFinite && result.a > 0 && result.b.isFinite && result.b > 0)
            #expect(sse.isFinite && abs(sse - reference.sse) < 1e-5)
            if word == 0x5000_0000 { #expect(result.a < Float.leastNormalMagnitude) }
        }
        for (spreadWord, minDistWord): (UInt32, UInt32) in [(1, 0), (0x7f7fffff, 0), (0x67800000, 0x65cccccd)] {
            expectKind(.operationFailed) {
                _ = try UMAPLayout.fitABParams(spread: Float(bitPattern: spreadWord), minDist: Float(bitPattern: minDistWord))
            }
        }
    }

    @Test func smoothHighPrecisionReferenceRows() throws {
        for row in L03Reference.smooth {
            let distances = row.words.map(Float.init(bitPattern:))
            let k = distances.count
            let g = try KNNGraph(pointCount: k + 1, rowOffsets: [0] + Array(repeating: k, count: k + 1),
                                 neighborIndices: (1...k).map(Int32.init), distances: distances)
            let result = try FuzzySimplicialSet.smoothDistances(g)
            let floor = 0.001 * distances.reduce(0.0) { $0 + Double($1) } / Double(k)
            #expect(result.rho[0] == row.rho)
            #expect(result.sigma[0].isFinite && result.sigma[0] > 0 && result.sigma[0] >= floor)
            let expectedSigma = max(row.sigma, floor)
            var total = 0.0
            for i in distances.indices {
                let delta = max(0, Double(distances[i]) - row.rho)
                let expected = delta == 0 ? 1 : Foundation.exp(-delta / expectedSigma)
                let actual = try FuzzySimplicialSet.membership(distance: Double(distances[i]), rho: result.rho[0], sigma: result.sigma[0])
                #expect(actual.isFinite && abs(actual - expected) < 3e-5)
                total += actual
            }
            if floor < row.sigma { #expect(total.isFinite && abs(total - row.target) <= 1e-5 * max(1, row.target)) }
            if k == 2 { #expect(abs(result.sigma[0] / row.sigma - 1) < 8e-5) }
        }
    }

    @Test func smoothTiesFloorsDuplicatesAndUnequalDegrees() throws {
        let expectedDuplicate = (Foundation.sqrt(5.0) - 1) / 2
        var baselineHeads: [Int32] = [], baselineTails: [Int32] = [], baselineWeights: [Float] = []
        for word: UInt32 in [0x0d800000, 0x3f800000, 0x71800000] {
            let scale = Float(bitPattern: word)
            let g = try KNNGraph(pointCount: 5, rowOffsets: [0, 3, 4, 4, 6, 6],
                                 neighborIndices: [1, 2, 2, 0, 4, 0], distances: [scale, 3*scale, 2*scale, scale, 0, 0])
            let result = try FuzzySimplicialSet.smoothDistances(g)
            let edges = try FuzzySimplicialSet.build(g)
            #expect(result.rho[0] == Double(scale) && result.rho[1] == Double(scale))
            #expect(result.sigma[2] == 0 && result.sigma[4] == 0)
            #expect(abs(result.sigma[1] / Double(scale) - 0.001) < 1e-15)
            #expect(abs(result.sigma[3] / Double(scale) - 0.001 * 7 / 6) < 1e-15)
            let index = try #require(edges.heads.indices.first { edges.heads[$0] == 0 && edges.tails[$0] == 2 })
            #expect(abs(Double(edges.weights[index]) - expectedDuplicate) < 3e-5)
            #expect(edges.count == 8)
            if baselineHeads.isEmpty { baselineHeads = edges.heads; baselineTails = edges.tails; baselineWeights = edges.weights }
            else {
                #expect(edges.heads == baselineHeads && edges.tails == baselineTails)
                #expect(zip(edges.weights, baselineWeights).allSatisfy { $0.isFinite && $1.isFinite && abs($0-$1) < 3e-5 })
            }
        }
        let tied = try KNNGraph(pointCount: 4, rowOffsets: [0, 3, 3, 3, 3], neighborIndices: [1, 2, 3], distances: [1, 1, 2])
        let tieResult = try FuzzySimplicialSet.smoothDistances(tied)
        #expect(abs(tieResult.sigma[0] - 0.001 * 4 / 3) < 1e-15)
        #expect(try FuzzySimplicialSet.build(tied).count == 4)
        let zeros = try KNNGraph(pointCount: 3, rowOffsets: [0, 2, 2, 2], neighborIndices: [1, 2], distances: [0, 0])
        let zeroResult = try FuzzySimplicialSet.smoothDistances(zeros)
        #expect(zeroResult.rho == [0, 0, 0] && zeroResult.sigma == [0, 0, 0])
        #expect(try FuzzySimplicialSet.build(zeros).weights == [1, 1, 1, 1])
        #expect(try FuzzySimplicialSet.membership(distance: 2, rho: 1, sigma: 0) == 0)
        #expect(try FuzzySimplicialSet.membership(distance: 1, rho: 1, sigma: 0) == 1)
        let large = [UInt32(0x71800000), 0x71800001, 0x71800002].map(Float.init(bitPattern:))
        let floorGraph = try KNNGraph(pointCount: 4, rowOffsets: [0, 3, 3, 3, 3], neighborIndices: [1, 2, 3], distances: large)
        let floorResult = try FuzzySimplicialSet.smoothDistances(floorGraph)
        let floor = 0.001 * large.reduce(0.0) { $0 + Double($1) } / 3
        #expect(floorResult.sigma[0] == floor)
        let total = try large.reduce(0.0) { sum, value in sum + (try FuzzySimplicialSet.membership(distance: Double(value), rho: floorResult.rho[0], sigma: floorResult.sigma[0])) }
        #expect(total.isFinite && total - 2 > 0.9) // The explicit floor overrides the solved target.
    }

    @Test func publicFloatingAndShapeValidationMatrix() throws {
        let g = try graph()
        let vectors = [DynamicVector([0]), DynamicVector([1]), DynamicVector([2])]
        for field in 0..<3 {
            for value: Float in [.nan, .infinity, -.infinity] {
                var config = UMAPConfig(neighbors: 2, epochs: 2, initialization: .random)
                if field == 0 { config.spread = value }
                if field == 1 { config.minDist = value }
                if field == 2 { config.learningRate = value }
                expectKind(.invalidOperation) { _ = try Operations.umap(vectors, dimensions: 1, config: config) }
                expectKind(.invalidOperation) { _ = try Operations.umap(graph: g, dimensions: 1, config: config) }
            }
        }
        var finiteInvalid: [(UMAPConfig, VectorError.ErrorKind)] = []
        for value: Float in [0, -1] {
            finiteInvalid.append((.init(neighbors: 2, spread: value, epochs: 2), .invalidOperation))
            finiteInvalid.append((.init(neighbors: 2, epochs: 2, learningRate: value), .invalidOperation))
        }
        finiteInvalid += [(.init(neighbors: 1, epochs: 2), .invalidDimension),
                          (.init(neighbors: 2, minDist: -1, epochs: 2), .invalidOperation),
                          (.init(neighbors: 2, minDist: 2, epochs: 2), .invalidOperation),
                          (.init(neighbors: 2, epochs: 0), .invalidOperation),
                          (.init(neighbors: 2, epochs: -1), .invalidOperation),
                          (.init(neighbors: 2, epochs: 2, negativeSampleRate: -1), .invalidOperation)]
        for (config, kind) in finiteInvalid {
            expectKind(kind) { _ = try Operations.umap(vectors, dimensions: 1, graph: g, config: config) }
            expectKind(kind) { _ = try Operations.umap(graph: g, dimensions: 1, config: config) }
        }
        for dimension in [-1, 0] {
            expectKind(.invalidDimension) { _ = try Operations.umap(vectors, dimensions: dimension, graph: g) }
            expectKind(.invalidDimension) { _ = try Operations.umap(graph: g, dimensions: dimension) }
        }
        for value: Float in [.nan, .infinity, -.infinity] {
            let coordinates: [[Float]] = [[0], [value], [2]]
            let bits = coordinates.map { $0.map(\.bitPattern) }
            expectKind(.invalidData) { _ = try Operations.umap(graph: g, dimensions: 1, initialCoordinates: coordinates, config: .init(epochs: 2)) }
            #expect(coordinates.map { $0.map(\.bitPattern) } == bits)
        }
        // Every width is checked before values: later wrong width wins over earlier NaN.
        expectKind(.dimensionMismatch) { _ = try Operations.umap(graph: g, dimensions: 1, initialCoordinates: [[.nan], [1], [2, 3]]) }
        expectKind(.dimensionMismatch) { _ = try Operations.umap(graph: g, dimensions: 1, initialCoordinates: [[0]]) }
        let one = try KNNGraph(pointCount: 1, rowOffsets: [0, 0], neighborIndices: [], distances: [])
        expectKind(.invalidDimension) { _ = try Operations.umap(graph: one, dimensions: Int.max) }
        expectKind(.invalidDimension) { _ = try Operations.umap([DynamicVector([1])], dimensions: 1, config: .init(neighbors: 2)) }
    }

    @Test func pureCountAndScheduleBoundaries() throws {
        let limit = UMAPChecked.maximumScheduleCount
        for rate in [0, 1] { try UMAPChecked.schedule(epochs: limit, negativeRate: rate) }
        try UMAPChecked.schedule(epochs: limit / 3, negativeRate: 3)
        expectKind(.invalidOperation) { try UMAPChecked.schedule(epochs: limit / 3 + 1, negativeRate: 3) }
        for pair in [(0, 1), (-1, 1), (1, -1), (Int.max, 1), (1, Int.max), (limit, 2)] {
            expectKind(.invalidOperation) { try UMAPChecked.schedule(epochs: pair.0, negativeRate: pair.1) }
        }
        #expect(try UMAPChecked.dueCount(-1) == 0)
        #expect(try UMAPChecked.dueCount(0.9) == 0)
        #expect(try UMAPChecked.dueCount(1.9) == 1)
        #expect(try UMAPChecked.dueCount(Double(limit)) == limit)
        for value in [Double.nan, .infinity, -.infinity, Double(limit) + 1, Double(Int.max)] {
            expectKind(.operationFailed) { _ = try UMAPChecked.dueCount(value) }
        }
        let floats = Int.max / MemoryLayout<Float>.stride
        #expect(try UMAPChecked.count([floats], stride: MemoryLayout<Float>.stride, label: "test") == floats)
        expectKind(.invalidDimension) { _ = try UMAPChecked.count([floats + 1], stride: MemoryLayout<Float>.stride, label: "test") }
        expectKind(.invalidDimension) { _ = try UMAPChecked.count([2, Int.max], stride: 1, label: "test") }
        expectKind(.invalidDimension) { _ = try UMAPChecked.layoutCount(points: 3, dimensions: Int.max) }
        expectKind(.invalidDimension) { _ = try UMAPChecked.increment(Int.max, label: "test") }
        #expect(try KNNGraph.checkedSizes(points: 2, dimensions: Int(Int32.max), neighbors: 1).edges == 2)
        for (n, d, k) in [(2, 0, 1), (2, -1, 1), (2, Int.max, 1), (Int.max, 1, 1), (Int(Int32.max), Int(Int32.max), 1), (Int(Int32.max), 1, Int(Int32.max)-1)] {
            expectKind(.invalidDimension) { _ = try KNNGraph.checkedSizes(points: n, dimensions: d, neighbors: k) }
        }
    }

    @Test func finiteSGDRangeAndInputImmutability() throws {
        let g = try graph()
        for coordinates: [[Float]] in [[[0], [1e30], [-1e30]], [[0], [0], [0]]] {
            let bits = coordinates.map { $0.map(\.bitPattern) }
            let rate: Float = coordinates[1][0] == 0 ? .greatestFiniteMagnitude : 1
            expectKind(.operationFailed) {
                _ = try Operations.umap(graph: g, dimensions: 1, initialCoordinates: coordinates,
                                        config: .init(neighbors: 2, epochs: 3, learningRate: rate, negativeSampleRate: 5))
            }
            #expect(coordinates.map { $0.map(\.bitPattern) } == bits)
        }
        let edges = UMAPEdgeList(heads: [0], tails: [1], weights: [1])
        for (a, b, values): (Float, Float, [Float]) in [(1, 100, [0, 10]), (.greatestFiniteMagnitude, 1, [0, 1])] {
            var coordinates = values
            expectKind(.operationFailed) {
                try UMAPLayout.optimize(edges: edges, coordinates: &coordinates, pointCount: 2, dimension: 1,
                                        parameters: .init(a: a, b: b, epochs: 2, learningRate: 1, negativeSampleRate: 0, seed: 1))
            }
        }
        var tiny: [Float] = [0, .leastNonzeroMagnitude]
        try UMAPLayout.optimize(edges: edges, coordinates: &tiny, pointCount: 2, dimension: 1,
                                parameters: .init(a: 1, b: 1, epochs: 2, learningRate: 1, negativeSampleRate: 0, seed: 1))
        #expect(tiny.map(\.bitPattern) == [0, 1]) // Ordinary squared-distance underflow keeps the zero policy.
    }

    @Test func tinyPCAInitializationAndProviderIdentity() throws {
        let g = try graph()
        let values = [UInt32(0), 2, 4].map { DynamicVector([Float(bitPattern: $0)]) }
        let before = values.map { $0.toArray().map(\.bitPattern) }
        let config = UMAPConfig(neighbors: 2, epochs: 1, negativeSampleRate: 0, initialization: .pca)
        func check() throws {
            let result = try Operations.umap(values, dimensions: 1, graph: g, config: config)
            #expect(result.coordinates.count == 3 && result.coordinates.allSatisfy(\.isFinite))
            #expect(abs(result.coordinates[0]) > 9 && abs(result.coordinates[2]) > 9)
            #expect(values.map { $0.toArray().map(\.bitPattern) } == before)
        }
        try check()
        try Operations.$linearAlgebraProvider.withValue(SwiftLinearAlgebraProvider()) { try check() }
        let sentinel = L03Sentinel()
        do {
            _ = try Operations.$linearAlgebraProvider.withValue(L03ThrowingProvider(error: sentinel)) {
                try Operations.umap(values, dimensions: 1, graph: g, config: config)
            }
            Issue.record("Lost provider failure")
        } catch let error as L03Sentinel { #expect(error === sentinel) }
        catch { Issue.record("Provider error identity changed: \(error)") }
    }

    @Test func ordinaryDeterminismAndQualifiedGramCharacterization() throws {
        let vectors = [DynamicVector([Float(10000)]), DynamicVector([Float(10002)]), DynamicVector([Float(10001)])]
        let direct = vectors.indices.map { i in
            vectors.indices.filter { $0 != i }.min {
                let a = Double(vectors[i][0]) - Double(vectors[$0][0])
                let b = Double(vectors[i][0]) - Double(vectors[$1][0])
                return (a*a, $0) < (b*b, $1)
            }!
        }
        #expect(direct == [2, 2, 0])
        let computed = try KNNGraph.bruteForce(vectors, neighbors: 1)
        print("L03_GRAM_CHARACTERIZATION observed=\(computed.neighborIndices) distances=\(computed.distances) independentDirect=\(direct)")
        #expect(computed.edgeCount == 3 && computed.distances.allSatisfy(\.isFinite))
        let g = try graph()
        let initial: [[Float]] = [[-1], [0], [2]]
        let config = UMAPConfig(neighbors: 2, epochs: 8, negativeSampleRate: 2, initialization: .random, seed: 17)
        let first = try Operations.umap(graph: g, dimensions: 1, initialCoordinates: initial, config: config)
        let second = try Operations.umap(graph: g, dimensions: 1, initialCoordinates: initial, config: config)
        #expect(first.coordinates.count == 3 && first.coordinates.allSatisfy(\.isFinite))
        #expect(first.coordinates.map(\.bitPattern) == second.coordinates.map(\.bitPattern))
        let ordinary = [DynamicVector([0, 0]), DynamicVector([1, 0]), DynamicVector([0, 2])]
        let result = try Operations.umap(ordinary, dimensions: 1, config: .init(neighbors: 2, epochs: 5))
        #expect(result.coordinates.count == 3 && result.coordinates.allSatisfy(\.isFinite))
    }

    @Test func childProbe() throws {
        guard let name = ProcessInfo.processInfo.environment["L03_PROBE"] else { return }
        print("L03_CHILD_BEGIN \(name)")
        let expected: VectorError.ErrorKind = name.hasPrefix("csr") ? .invalidData : (name.lowercased().contains("dimension") ? .invalidDimension : .invalidOperation)
        do {
            if name == "csr" {
                _ = try KNNGraph(pointCount: 2, rowOffsets: [0, 3, 2], neighborIndices: [1, 1], distances: [1, 1])
            } else if name == "csrMatrix" {
                for a in -1...3 {
                    for b in -1...3 where !(0 <= a && a <= b && b <= 2) {
                        do {
                            _ = try KNNGraph(pointCount: 3, rowOffsets: [0, a, b, 2], neighborIndices: [1, 2], distances: [1, 2])
                            Issue.record("Accepted malformed offsets [0, \(a), \(b), 2]")
                        } catch let error as VectorError { #expect(error.kind == .invalidData) }
                    }
                }
                print("L03_TYPED_REJECTION csrMatrix invalidData")
            } else {
                let g = try graph()
                var config = UMAPConfig(neighbors: 2, epochs: 3, negativeSampleRate: 0, initialization: .random)
                if name.lowercased().contains("negativerate") { config.negativeSampleRate = Int.max }
                if name.lowercased().contains("epochs") { config.epochs = Int.max }
                if name == "product" { config.epochs = 2; config.negativeSampleRate = UMAPChecked.maximumScheduleCount }
                let dimension = name.lowercased().contains("dimension") ? Int.max : 1
                if name.hasPrefix("vector") {
                    _ = try Operations.umap([DynamicVector([0]), DynamicVector([1]), DynamicVector([2])], dimensions: dimension, graph: g, config: config)
                } else {
                    _ = try Operations.umap(graph: g, dimensions: dimension, config: config)
                }
            }
            if name != "csrMatrix" { Issue.record("Expected typed preflight rejection for \(name)") }
        } catch let error as VectorError {
            #expect(error.kind == expected)
            print("L03_TYPED_REJECTION \(name) \(error.kind)")
        }
        print("L03_CHILD_END \(name)")
    }

    @Test(arguments: ["csr", "csrMatrix", "dimensions", "vectorDimensions", "negativeRate", "vectorNegativeRate", "epochs", "vectorEpochs", "product"])
    func boundedPublicRejections(_ name: String) throws {
        if ProcessInfo.processInfo.environment["L03_VALID_ONLY"] == "1" { return }
        let evidenceRoot = FileManager.default.temporaryDirectory.appendingPathComponent("L03-children")
        let directory = evidenceRoot.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let capture = directory.appendingPathComponent(name + ".log")
        _ = FileManager.default.createFile(atPath: capture.path, contents: nil)
        let output = try FileHandle(forWritingTo: capture)
        defer { try? output.close() }
        var environment = ProcessInfo.processInfo.environment
        environment["L03_PROBE"] = name
        environment["SWIFT_BACKTRACE"] = "enable=no"
        var record: [String: Any] = ["executable": CommandLine.arguments[0], "probe": name, "capture": capture.path,
                                     "deadlineSeconds": 15, "cleanupSeconds": 2]
        defer {
            if let data = try? JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys]) {
                try? data.write(to: directory.appendingPathComponent("command.json"))
            }
        }
        let process = try launchChildProbe(.swiftTesting(filter: "UMAPRangeContractTests/childProbe"),
                                           environment: environment, output: output, deadline: 15, cleanup: .kill(seconds: 2))
        record["arguments"] = process.arguments; record["pid"] = process.processIdentifier
        print("L03_CHILD_CAPTURE \(name) \(capture.path)")
        if process.timedOut {
            record["timeout"] = true
            record["cleanupTimeout"] = !process.cleaned
            throw L03ProbeFailure.timeout(capture.path)
        }
        record["status"] = process.terminationStatus
        record["reason"] = process.terminationReason?.rawValue
        let text = try String(contentsOf: capture, encoding: .utf8)
        #expect(process.terminationStatus == 0, "\(name): \(text)")
        #expect(text.contains("L03_TYPED_REJECTION \(name)"), "\(name): \(text)")
        #expect(text.contains("L03_CHILD_END \(name)"), "\(name): \(text)")
    }
}

private final class L03Sentinel: Error, Sendable {}
private struct L03ThrowingProvider: LinearAlgebraProvider {
    let error: L03Sentinel
    func qrThin(_ a: [Float], rows: Int, columns: Int) throws -> QRFactorization { throw error }
    func svdThin(_ a: [Float], rows: Int, columns: Int) throws -> SingularValueDecomposition { throw error }
    func symmetricEigen(_ a: [Float], dimension: Int, computeEigenvectors: Bool) throws -> SymmetricEigenDecomposition { throw error }
}
