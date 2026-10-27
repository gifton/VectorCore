import Foundation
import Testing
@testable import VectorCore

@Suite("Public vector shape contracts", .serialized)
struct VectorShapeContractTests {
    private enum ProbeError: Error { case expected }
    private static let families = ["fp384", "fp512", "fp768", "fp1536", "q512", "q768", "q1536"]

    private func replace<V: VectorProtocol>(_ value: inout V, _ storage: V.Storage) {
        value.storage = storage
    }

    private func edit<S>(_ storage: inout S, replacement: S, throwing: Bool) throws {
        storage = replacement
        if throwing { throw ProbeError.expected }
    }

    // Removing the storage boundary guard makes the parent observe a normal exit with
    // malformed count metadata. The child never indexes or borrows a malformed vector.
    private func probe<V: VectorProtocol, Lane>(
        _ type: V.Type, lanes: Int, delta: Int, route: String, lane: Lane,
        raw: (ContiguousArray<Lane>) -> V
    ) throws where V.Storage == ContiguousArray<Lane> {
        let candidate = ContiguousArray(repeating: lane, count: lanes + delta)
        var value = V()
        switch route {
        case "raw": value = raw(candidate)
        case "direct": value.storage = candidate
        case "generic": replace(&value, candidate)
        case "nested":
            if delta < 0 { value.storage.removeLast() }
            else if delta > 0 { value.storage.append(lane) }
            else { value.storage[0] = lane }
        case "inout", "unwind":
            do { try edit(&value.storage, replacement: candidate, throwing: route == "unwind") }
            catch ProbeError.expected {}
        default: throw ProbeError.expected
        }
        print("S01_ACCEPTED lanes=\(value.storage.count) scalarCount=\(value.scalarCount)")
    }

    @Test func childProbe() throws {
        guard let specification = ProcessInfo.processInfo.environment["S01_SHAPE_PROBE"] else { return }
        let parts = specification.split(separator: ":").map(String.init)
        let delta = try #require(Int(parts[2]))
        switch parts[0] {
        case "fp384": try probe(Vector384Optimized.self, lanes: 96, delta: delta, route: parts[1], lane: SIMD4<Float>(repeating: 1), raw: Vector384Optimized.init(fromSIMDStorage:))
        case "fp512": try probe(Vector512Optimized.self, lanes: 128, delta: delta, route: parts[1], lane: SIMD4<Float>(repeating: 1), raw: Vector512Optimized.init(fromSIMDStorage:))
        case "fp768": try probe(Vector768Optimized.self, lanes: 192, delta: delta, route: parts[1], lane: SIMD4<Float>(repeating: 1), raw: Vector768Optimized.init(fromSIMDStorage:))
        case "fp1536": try probe(Vector1536Optimized.self, lanes: 384, delta: delta, route: parts[1], lane: SIMD4<Float>(repeating: 1), raw: Vector1536Optimized.init(fromSIMDStorage:))
        case "q512": try probe(Vector512INT8.self, lanes: 128, delta: delta, route: parts[1], lane: SIMD4<Int8>(repeating: 1), raw: { Vector512INT8(storage: $0,
            params: .init(minValue: -127, maxValue: 127)) })
        case "q768": try probe(Vector768INT8.self, lanes: 192, delta: delta, route: parts[1], lane: SIMD4<Int8>(repeating: 1), raw: { Vector768INT8(storage: $0,
            params: .init(minValue: -127, maxValue: 127)) })
        case "q1536": try probe(Vector1536INT8.self, lanes: 384, delta: delta, route: parts[1], lane: SIMD4<Int8>(repeating: 1), raw: { Vector1536INT8(storage: $0,
            params: .init(minValue: -127, maxValue: 127)) })
        default: Issue.record("Unknown probe \(specification)")
        }
    }

    @Test func fixedStorageBoundaryMatrix() throws {
        let logs = FileManager.default.temporaryDirectory.appendingPathComponent("S01-children-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        print("S01 child logs: \(logs.path)")
        for family in Self.families {
            let width = Int(family.filter(\.isNumber))!
            for route in ["raw", "direct", "generic", "nested", "inout", "unwind"] {
                for delta in [-1, 0, 1] {
                    let specification = "\(family):\(route):\(delta)"
                    var environment = ProcessInfo.processInfo.environment
                    environment["S01_SHAPE_PROBE"] = specification
                    environment["SWIFT_BACKTRACE"] = "enable=no"
                    let url = logs.appendingPathComponent(specification + ".log")
                    _ = FileManager.default.createFile(atPath: url.path, contents: nil)
                    let output = try FileHandle(forWritingTo: url)
                    defer { try? output.close() }
                    // The launcher keeps SwiftPM's helper/bundle arguments on macOS and selects only the probe.
                    let process = try launchChildProbe(.swiftTesting(filter: "VectorShapeContractTests/childProbe"),
                                                       environment: environment, output: output, deadline: 15,
                                                       cleanup: .kill(seconds: 2))
                    if process.timedOut {
                        let message = "\(specification): child deadline; cleanup=\(process.cleanup); log=\(url.path)"
                        throw NSError(domain: "VectorShapeContractTests.childTimeout", code: 1,
                                      userInfo: [NSLocalizedDescriptionKey: message])
                    }
                    let diagnostic = try String(contentsOf: url, encoding: .utf8)
                    if delta == 0 {
                        #expect(process.terminationStatus == 0, "\(specification): \(diagnostic)")
                        #expect(diagnostic.contains("S01_ACCEPTED lanes=\(width / 4) scalarCount=\(width)"))
                    } else {
                        #expect(process.terminationReason == .uncaughtSignal, "\(specification): status \(process.statusDescription), \(diagnostic)")
                        #expect(diagnostic.contains("Storage must contain exactly \(width / 4) SIMD4 elements"), "\(specification): \(diagnostic)")
                        #expect(!diagnostic.contains("S01_ACCEPTED"))
                    }
                }
            }
        }
    }

    private func counts<V: VectorProtocol>(_ vector: V, expected: Int) throws where V.Scalar == Float {
        var value = vector
        #expect(value.scalarCount == expected)
        #expect(value.endIndex == expected)
        #expect(value.count == expected)
        #expect(value.toArray().count == expected)
        value.withUnsafeBufferPointer { buffer in
            #expect(buffer.count == expected)
            if expected > 0 { #expect(buffer[expected - 1].isFinite) }
        }
        value.withUnsafeMutableBufferPointer { buffer in
            #expect(buffer.count == expected)
            if expected > 0 { #expect(buffer[expected - 1].isFinite) }
        }
        let decoded = try JSONDecoder().decode(V.self, from: JSONEncoder().encode(vector))
        #expect(decoded.scalarCount == expected)
        #expect(decoded.toArray() == vector.toArray())
        #expect(decoded == vector)
        #expect(decoded.hashValue == vector.hashValue)
    }

    @Test func dynamicReplacementOwnsDimension() throws {
        for count in [0, 1, 2, 3, 2048] {
            let expected = (0..<count).map { Float($0 + 1) }
            let storage = HybridStorage<Float>(from: expected)
            var direct = DynamicVector([1, 2])
            let original = direct
            direct.storage = storage
            // Metadata is safe even against the malformed baseline. Avoid its indexing.
            #expect(direct.scalarCount == count)
            #expect(direct.endIndex == count)
            guard direct.scalarCount == count else { continue }
            try counts(direct, expected: count)
            #expect(direct == DynamicVector(expected))
            #expect(direct.hashValue == DynamicVector(expected).hashValue)
            #expect(original.toArray() == [1, 2])
            var generic = DynamicVector([1, 2])
            replace(&generic, storage)
            try counts(generic, expected: count)
            var hinted = NormalizationHint(vector: DynamicVector([1, 2]), isNormalized: true)
            replace(&hinted, storage)
            #expect(!hinted.isNormalized)
            #expect(hinted.cachedMagnitude == nil)
            try counts(hinted.vector, expected: count)
            let json = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(direct)) as? [String: Any])
            #expect(json["dimension"] as? Int == count)
            #expect(json["elements"] as? [Float] == expected)
            // Nonempty binary shape control; empty coverage is B01BinaryContractTests.emptyDynamicRoundTrip.
            if count > 0 { #expect(try DynamicVector.decodeBinary(from: direct.encodeBinary()) == direct) }
        }
        #expect(throws: (any Error).self) { try DynamicVector(dimension: 3, storage: HybridStorage<Float>(from: [1, 2])) }
        #expect(throws: (any Error).self) { try JSONDecoder().decode(DynamicVector.self, from: Data(#"{"dimension":3,"elements":[1,2]}"#.utf8)) }
    }

    private func decodeINT8<V: QuantizedVectorINT8>(_ type: V.Type) throws {
        let params = LinearQuantizationParams(minValue: -127, maxValue: 127)
        let value = V(storage: ContiguousArray(repeating: SIMD4<Int8>(1, 2, 3, 4), count: V.laneCount), params: params)
        let data = try JSONEncoder().encode(value)
        var json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(Set(json.keys) == ["storage", "quantizationParams"])
        let lanes = try #require(json["storage"] as? [[Int]])
        #expect(lanes.count == V.laneCount)
        #expect(lanes[0] == [1, 2, 3, 4])
        let restored = try JSONDecoder().decode(V.self, from: data)
        #expect(restored.storage == value.storage)
        #expect(restored.quantizationParams == params)
        let asymmetric = LinearQuantizationParams(minValue: -20, maxValue: 40, symmetric: false)
        let asymmetricValue = V(storage: value.storage, params: asymmetric)
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let asymmetricData = try encoder.encode(asymmetricValue)
        let asymmetricRestored = try JSONDecoder().decode(V.self, from: asymmetricData)
        #expect(asymmetricRestored.quantizationParams == asymmetric)
        #expect(try encoder.encode(asymmetricRestored) == asymmetricData)
        for delta in [-1, 1] {
            json["storage"] = Array(repeating: [1, 2, 3, 4], count: V.laneCount + delta)
            let malformed = try JSONSerialization.data(withJSONObject: json)
            #expect(throws: (any Error).self) { try JSONDecoder().decode(V.self, from: malformed) }
        }
    }

    @Test func int8DecodingValidatesShapeAndPreservesWire() throws {
        try decodeINT8(Vector512INT8.self)
        try decodeINT8(Vector768INT8.self)
        try decodeINT8(Vector1536INT8.self)
    }

    private func validFixed<V: VectorProtocol, Lane: Equatable>(
        _ type: V.Type, width: Int, lane: Lane,
        raw: (ContiguousArray<Lane>) -> V
    ) throws where V.Scalar == Float, V.Storage == ContiguousArray<Lane> {
        for delta in [-1, 0, 1] {
            let array = Array(repeating: Float(1), count: width + delta)
            if delta == 0 { try counts(V(array), expected: width) }
            else { #expect(throws: (any Error).self) { try V(array) } }
        }
        let candidate = ContiguousArray(repeating: lane, count: width / 4)
        try counts(raw(candidate), expected: width)
        for route in ["direct", "generic", "nested", "inout", "unwind"] {
            var value = raw(V().storage)
            let retained = value
            let before = retained.toArray()
            switch route {
            case "direct": value.storage = candidate
            case "generic": replace(&value, candidate)
            case "nested":
                for index in candidate.indices { value.storage[index] = lane }
            default:
                do { try edit(&value.storage, replacement: candidate, throwing: route == "unwind") }
                catch ProbeError.expected {}
            }
            #expect(value.storage == candidate)
            #expect(value.toArray() == raw(candidate).toArray())
            #expect(retained.toArray() == before)
            try counts(value, expected: width)
        }
    }

    private func validFP32<V: OptimizedVector>(_ type: V.Type, width: Int,
        raw: (ContiguousArray<SIMD4<Float>>) -> V,
        generated: () -> V, scalarWrite: (inout V) -> Void
    ) throws where V.Storage == ContiguousArray<SIMD4<Float>> {
        try validFixed(type, width: width, lane: SIMD4<Float>(1, 2, 3, 4), raw: raw)
        try counts(generated(), expected: width)
        for delta in [-1, 0, 1] {
            let data = try JSONEncoder().encode(Array(repeating: Float(1), count: width + delta))
            if delta == 0 { try counts(JSONDecoder().decode(V.self, from: data), expected: width) }
            else { #expect(throws: (any Error).self) { try JSONDecoder().decode(V.self, from: data) } }
        }
        var value = V(repeating: 1)
        let retained = value
        scalarWrite(&value)
        #expect(value[width - 1] == 2)
        #expect(retained[width - 1] == 1)
        value.withUnsafeMutableBufferPointer { $0[width - 1] = 3 }
        #expect(value[width - 1] == 3)
        #expect(retained[width - 1] == 1)
        try counts(value, expected: width)
    }

    @Test func validFP32ConstructorsMutationAndBuffers() throws {
        try validFP32(Vector384Optimized.self, width: 384, raw: Vector384Optimized.init(fromSIMDStorage:), generated: { Vector384Optimized(generator: { Float($0) }) }, scalarWrite: { $0[383] = 2 })
        try validFP32(Vector512Optimized.self, width: 512, raw: Vector512Optimized.init(fromSIMDStorage:), generated: { Vector512Optimized(generator: { Float($0) }) }, scalarWrite: { $0[511] = 2 })
        try validFP32(Vector768Optimized.self, width: 768, raw: Vector768Optimized.init(fromSIMDStorage:), generated: { Vector768Optimized(generator: { Float($0) }) }, scalarWrite: { $0[767] = 2 })
        try validFP32(Vector1536Optimized.self, width: 1536, raw: Vector1536Optimized.init(fromSIMDStorage:), generated: { Vector1536Optimized(generator: { Float($0) }) },
            scalarWrite: { $0[1535] = 2 })
    }

    private func validINT8<V: QuantizedVectorINT8>(_ type: V.Type, width: Int,
        distance: (V, V) -> Float, dot: (V, V) -> Float, cosine: (V, V) -> Float
    ) throws where V.Scalar == Float, V.Storage == ContiguousArray<SIMD4<Int8>> {
        let params = LinearQuantizationParams(minValue: -127, maxValue: 127)
        try validFixed(type, width: width, lane: SIMD4<Int8>(1, 2, 3, 4), raw: { V(storage: $0, params: params) })
        var value = V(storage: ContiguousArray(repeating: SIMD4<Int8>(repeating: 0), count: width / 4), params: params)
        value.storage = ContiguousArray(repeating: SIMD4<Int8>(1, 2, 3, 4), count: width / 4)
        let expected = (0..<width).map { Float($0 % 4 + 1) }
        #expect(value.dimension == width)
        #expect(value.toFP32().toArray() == expected)
        #expect(value.quantizationParams == params)
        let fp32 = try V.FloatVectorType(expected)
        let quantized = V(from: fp32, params: params)
        #expect(quantized.storage == value.storage)
        #expect(quantized.toFP32().toArray() == expected)
        let zero = V(storage: ContiguousArray(repeating: SIMD4<Int8>(repeating: 0), count: width / 4), params: params)
        let sumSquares = Float(width / 4) * 30
        #expect(abs(distance(value, zero) - sqrt(sumSquares)) < 0.001)
        #expect(abs(dot(value, value) - sumSquares) < 0.001)
        #expect(abs(cosine(value, value)) < 0.00001)
        // Exercise asymmetric and unequal-scale fallback with a scalar reference.
        let alternate = LinearQuantizationParams(minValue: -20, maxValue: 40, symmetric: false)
        let other = V(storage: value.storage, params: alternate)
        let otherExpected = expected.map { alternate.scale * ($0 - Float(alternate.zeroPoint)) }
        let reference = zip(expected, otherExpected).reduce(Float(0)) { $0 + ($1.0 - $1.1) * ($1.0 - $1.1) }.squareRoot()
        #expect(abs(distance(value, other) - reference) < 0.01)
        try counts(value, expected: width)
    }

    @Test func validINT8ConversionBuffersAndKernels() throws {
        try validINT8(Vector512INT8.self, width: 512, distance: { $0.euclideanDistance(to: $1) }, dot: { $0.dotProduct($1) }, cosine: { $0.cosineDistance(to: $1) })
        try validINT8(Vector768INT8.self, width: 768, distance: { $0.euclideanDistance(to: $1) }, dot: { $0.dotProduct($1) }, cosine: { $0.cosineDistance(to: $1) })
        try validINT8(Vector1536INT8.self, width: 1536, distance: { $0.euclideanDistance(to: $1) }, dot: { $0.dotProduct($1) }, cosine: { $0.cosineDistance(to: $1) })
    }

    private func genericControl<D: StaticDimension>(_ type: D.Type) throws {
        for delta in [-1, 0, 1] {
            let array = Array(repeating: Float(1), count: D.value + delta)
            let data = try JSONSerialization.data(withJSONObject: ["elements": array])
            if delta == 0 {
                var value = try Vector<D>(array)
                let retained = value
                value.storage[D.value - 1] = 2
                #expect(value[D.value - 1] == 2)
                #expect(retained[D.value - 1] == 1)
                try counts(value, expected: D.value)
                try counts(JSONDecoder().decode(Vector<D>.self, from: data), expected: D.value)
            } else {
                #expect(throws: (any Error).self) { try Vector<D>(array) }
                #expect(throws: (any Error).self) { try JSONDecoder().decode(Vector<D>.self, from: data) }
            }
        }
    }

    @Test func genericFixedAndImmutableFP16Controls() throws {
        try genericControl(Dim2.self)
        try genericControl(Dim384.self)
        try genericControl(Dim512.self)
        try genericControl(Dim768.self)
        try genericControl(Dim1536.self)
        let half512 = MixedPrecisionKernels.Vector512FP16(from: Vector512Optimized(repeating: 1))
        let half768 = MixedPrecisionKernels.Vector768FP16(from: Vector768Optimized(repeating: 1))
        let half1536 = MixedPrecisionKernels.Vector1536FP16(from: Vector1536Optimized(repeating: 1))
        #expect(half512.storage.count == 512)
        #expect(half768.storage.count == 768)
        #expect(half1536.storage.count == 1536)
        #expect(half512.toFP32().toArray() == Array(repeating: Float(1), count: 512))
        #expect(half768.toFP32().toArray() == Array(repeating: Float(1), count: 768))
        #expect(half1536.toFP32().toArray() == Array(repeating: Float(1), count: 1536))
        try counts(half512.toFP32(), expected: 512)
        try counts(half768.toFP32(), expected: 768)
        try counts(half1536.toFP32(), expected: 1536)
    }

    private func postReplacementKernelCoverage<Q: QuantizedVectorINT8, F: OptimizedVector>(
        _ quantizedType: Q.Type, _ floatType: F.Type, width: Int,
        mixedForward: (Q, F) -> Float, mixedReverse: (F, Q) -> Float,
        mixedBatch: (F, SoAINT8<F>, UnsafeMutableBufferPointer<Float>) -> Void,
        int8Batch: (Q, SoAINT8<F>, UnsafeMutableBufferPointer<Float>) -> Void
    ) throws where Q.FloatVectorType == F, Q.Storage == ContiguousArray<SIMD4<Int8>>,
        F.Storage == ContiguousArray<SIMD4<Float>> {
        let params = LinearQuantizationParams(minValue: -127, maxValue: 127)
        var query = F()
        query.storage = ContiguousArray(repeating: SIMD4<Float>(1, 2, 3, 4), count: width / 4)
        var quantizedQuery = Q(storage: ContiguousArray(repeating: .zero, count: width / 4), params: params)
        quantizedQuery.storage = ContiguousArray(repeating: SIMD4<Int8>(1, 2, 3, 4), count: width / 4)
        let queryReference = (0..<width).map { Double($0 % 4 + 1) }
        var candidates: [F] = []
        var references: [[Double]] = []
        for candidateIndex in 0..<5 {
            let pattern: [Double] = [-127, Double(candidateIndex - 2), Double(2 * candidateIndex + 1), 127]
            let reference = (0..<width).map { pattern[$0 % 4] }
            var candidate = F()
            candidate.storage = ContiguousArray(repeating: SIMD4<Float>(
                -127, Float(candidateIndex - 2), Float(2 * candidateIndex + 1), 127
            ), count: width / 4)
            candidates.append(candidate)
            references.append(reference)
        }
        let expectedDistances = references.map { candidate in
            zip(queryReference, candidate).reduce(0.0) { sum, pair in
                let difference = pair.0 - pair.1
                return sum + difference * difference
            }.squareRoot()
        }
        func checkDistance(_ actual: Float, _ expected: Double) {
            #expect(actual.isFinite)
            // Integer square sums stay below 2^24 in these fixtures; allow two
            // Float ULPs for the final square root/conversion against Double.
            #expect(abs(Double(actual) - expected) <= 2 * Double(Float(expected).ulp))
        }
        for index in candidates.indices {
            checkDistance(mixedForward(quantizedQuery, candidates[index]), expectedDistances[index])
            checkDistance(mixedReverse(candidates[index], quantizedQuery), expectedDistances[index])
        }

        let calibrated = QuantizationCalibrator.calibrate(vectors: candidates)
        let distribution = QuantizationCalibrator.calibrateWithDistribution(vectors: candidates, percentile: 99)
        let soaCalibration = SoAINT8<F>.calibrateParams(vectors: candidates)
        for calibration in [calibrated, distribution, soaCalibration] {
            #expect(calibration == params)
        }
        // Both explicit parameters and automatic calibration must transpose all five
        // candidates, including the partial second SIMD group, after storage replacement.
        for soa in [SoAINT8<F>(from: candidates, params: params), SoAINT8<F>(from: candidates)] {
            #expect(soa.dimension == width)
            #expect(soa.vectorCount == 5)
            #expect(soa.simdGroups == 2)
            #expect(soa.storage.count == width * 2)
            #expect(soa.quantizationParams == params)
            for dimension in 0..<width {
                for candidateIndex in 0..<5 {
                    let actual = soa.storage[dimension * 2 + candidateIndex / 4][candidateIndex % 4]
                    #expect(Int(actual) == Int(references[candidateIndex][dimension]))
                }
                #expect(soa.storage[dimension * 2 + 1].y == 0)
                #expect(soa.storage[dimension * 2 + 1].z == 0)
                #expect(soa.storage[dimension * 2 + 1].w == 0)
            }
            var mixedResults = Array(repeating: Float.nan, count: 5)
            var quantizedResults = Array(repeating: Float.nan, count: 5)
            mixedResults.withUnsafeMutableBufferPointer { mixedBatch(query, soa, $0) }
            quantizedResults.withUnsafeMutableBufferPointer { int8Batch(quantizedQuery, soa, $0) }
            for index in expectedDistances.indices {
                checkDistance(mixedResults[index], expectedDistances[index])
                checkDistance(quantizedResults[index], expectedDistances[index])
            }
        }
    }

    @Test func postReplacementMixedSoAAndCalibrationKernels() throws {
        try postReplacementKernelCoverage(Vector512INT8.self, Vector512Optimized.self, width: 512,
            mixedForward: { QuantizedKernels.euclidean512(query: $0, candidate: $1) },
            mixedReverse: { QuantizedKernels.euclidean512(query: $0, candidate: $1) },
            mixedBatch: { QuantizedKernels.batchEuclidean512(query: $0, candidates: $1, results: $2) },
            int8Batch: { QuantizedKernels.batchEuclidean512(query: $0, candidates: $1, results: $2) })
        try postReplacementKernelCoverage(Vector768INT8.self, Vector768Optimized.self, width: 768,
            mixedForward: { QuantizedKernels.euclidean768(query: $0, candidate: $1) },
            mixedReverse: { QuantizedKernels.euclidean768(query: $0, candidate: $1) },
            mixedBatch: { QuantizedKernels.batchEuclidean_generic_mixed(query: $0, candidates: $1, results: $2) },
            int8Batch: { QuantizedKernels.batchEuclidean_generic_int8(query: $0, candidates: $1, results: $2) })
        try postReplacementKernelCoverage(Vector1536INT8.self, Vector1536Optimized.self, width: 1536,
            mixedForward: { QuantizedKernels.euclidean1536(query: $0, candidate: $1) },
            mixedReverse: { QuantizedKernels.euclidean1536(query: $0, candidate: $1) },
            mixedBatch: { QuantizedKernels.batchEuclidean_generic_mixed(query: $0, candidates: $1, results: $2) },
            int8Batch: { QuantizedKernels.batchEuclidean_generic_int8(query: $0, candidates: $1, results: $2) })
        #if VC_USE_C_KERNELS
        print("S01 configured INT8 path: VC_USE_C_KERNELS")
        #else
        print("S01 configured INT8 path: Swift; VC_USE_C_KERNELS alternative uncompiled")
        #endif
    }
}
