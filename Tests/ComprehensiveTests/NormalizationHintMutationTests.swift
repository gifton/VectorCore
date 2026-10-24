import Foundation
import Testing
@testable import VectorCore

@Suite("Normalization hint mutation and decoding")
struct NormalizationHintMutationTests {
    private enum BorrowFailure: Error { case expected }

    private func replaceStorage<T: VectorProtocol>(_ value: inout T, _ storage: T.Storage) {
        value.storage = storage
    }

    private func mutateStorage<T: VectorProtocol>(
        _ value: inout T, _ edit: (inout T.Storage) -> Void
    ) {
        edit(&value.storage)
    }

    private func exercise<V: IndexableVector>(
        _ type: V.Type, count: Int,
        setFirst: @escaping (inout V, Float) -> Void,
        setStorageFirst: @escaping (inout V.Storage, Float) -> Void
    ) throws {
        let unit = [Float(1)] + Array(repeating: Float(0), count: count - 1)
        let changed = [Float(2)] + Array(repeating: Float(0), count: count - 1)
        let original = try V(unit)
        let replacement = try V(changed)
        let routes: [(String, (inout NormalizationHint<V>) throws -> Void, Bool)] = [
            ("vector replacement", { $0.vector = replacement }, true),
            ("same vector replacement", { $0.vector = original }, false),
            ("nested subscript", { setFirst(&$0.vector, 2) }, true),
            ("nested storage element", { setStorageFirst(&$0.vector.storage, 2) }, true),
            ("nested storage replacement", { $0.vector.storage = replacement.storage }, true),
            ("wrapper storage replacement", { $0.storage = replacement.storage }, true),
            ("wrapper storage element", { setStorageFirst(&$0.storage, 2) }, true),
            ("generic storage replacement", { replaceStorage(&$0, replacement.storage) }, true),
            ("generic storage element", { mutateStorage(&$0) { setStorageFirst(&$0, 2) } }, true),
            ("mutable buffer", { $0.withUnsafeMutableBufferPointer { $0[0] = 2 } }, true),
            ("throwing mutable buffer", {
                try $0.withUnsafeMutableBufferPointer { buffer in
                    buffer[0] = 2
                    throw BorrowFailure.expected
                }
            }, true),
            ("throwing nested mutable buffer", {
                try $0.vector.withUnsafeMutableBufferPointer { buffer in
                    buffer[0] = 2
                    throw BorrowFailure.expected
                }
            }, true),
            ("no-write mutable buffer", { $0.withUnsafeMutableBufferPointer { _ in } }, false),
            ("no-write nested mutable buffer", { $0.vector.withUnsafeMutableBufferPointer { _ in } }, false)
        ]
        for (route, mutate, changesCoordinates) in routes {
            for autoDetected in [false, true] {
                // Both normalized assertions and a false flag with a populated cache must expire.
                let startingVector = autoDetected ? replacement : original
                var hint = autoDetected
                    ? NormalizationHint(vector: startingVector)
                    : NormalizationHint(vector: startingVector, isNormalized: true)
                let retained = hint
                #expect(hint.cachedMagnitude != nil)
                var didThrow = false
                do { try mutate(&hint) }
                catch BorrowFailure.expected { didThrow = true }
                catch { Issue.record("Unexpected error: \(error)") }
                #expect(didThrow == route.hasPrefix("throwing"))
                let expected = changesCoordinates ? changed : (route == "same vector replacement" ? unit : startingVector.toArray())
                #expect(hint.toArray() == expected, "\(V.self): \(route)")
                #expect(!hint.isNormalized, "\(V.self): \(route)")
                #expect(hint.cachedMagnitude == nil, "\(V.self): \(route)")
                let expectedMagnitude: Float = expected[0]
                #expect(hint.magnitude.isFinite)
                #expect(abs(hint.magnitude - expectedMagnitude) < 0.00001, "\(V.self): \(route)")
                #expect(retained.toArray() == startingVector.toArray())
                #expect(retained.isNormalized == !autoDetected)
                #expect(retained.cachedMagnitude == (autoDetected ? 2 : 1))
            }
        }
        let readOnly = NormalizationHint(vector: original, isNormalized: true)
        _ = readOnly.vector
        _ = readOnly.storage
        _ = readOnly[0]
        readOnly.withUnsafeBufferPointer { #expect(abs($0[0] - 1) < 0.00001) }
        #expect(readOnly.isNormalized)
        #expect(readOnly.cachedMagnitude == 1)
    }

    @Test func mutationsDynamicVector() throws {
        try exercise(DynamicVector.self, count: 7, setFirst: { $0[0] = $1 },
                     setStorageFirst: { $0[0] = $1 })
    }

    @Test func mutationsVector384Optimized() throws {
        try exercise(Vector384Optimized.self, count: 384, setFirst: { $0[0] = $1 },
                     setStorageFirst: { $0[0][0] = $1 })
    }

    @Test func mutationsVector512Optimized() throws {
        try exercise(Vector512Optimized.self, count: 512, setFirst: { $0[0] = $1 },
                     setStorageFirst: { $0[0][0] = $1 })
    }

    @Test func mutationsVector768Optimized() throws {
        try exercise(Vector768Optimized.self, count: 768, setFirst: { $0[0] = $1 },
                     setStorageFirst: { $0[0][0] = $1 })
    }

    @Test func mutationsVector1536Optimized() throws {
        try exercise(Vector1536Optimized.self, count: 1536, setFirst: { $0[0] = $1 },
                     setStorageFirst: { $0[0][0] = $1 })
    }

    private func exerciseFixed<D: StaticDimension>(_ dimension: D.Type) throws {
        try exercise(Vector<D>.self, count: D.value, setFirst: { $0[0] = $1 },
                     setStorageFirst: { $0[0] = $1 })
    }

    @Test func mutationsGenericFixedDimensions() throws {
        try exerciseFixed(Dim2.self)
        try exerciseFixed(Dim3.self)
        try exerciseFixed(Dim4.self)
        try exerciseFixed(Dim8.self)
        try exerciseFixed(Dim16.self)
        try exerciseFixed(Dim32.self)
        try exerciseFixed(Dim64.self)
        try exerciseFixed(Dim128.self)
        try exerciseFixed(Dim256.self)
        try exerciseFixed(Dim384.self)
        try exerciseFixed(Dim512.self)
        try exerciseFixed(Dim768.self)
        try exerciseFixed(Dim1024.self)
        try exerciseFixed(Dim1536.self)
        try exerciseFixed(Dim2048.self)
        try exerciseFixed(Dim3072.self)
    }

    private func exerciseDecoding<V: IndexableVector>(_ type: V.Type, count: Int) throws {
        let coordinates = [Float(2)] + Array(repeating: Float(0), count: count - 1)
        let original = NormalizationHint(vector: try V(coordinates), isNormalized: true)
        let encoded = try JSONEncoder().encode(original)
        let payload = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        #expect(payload["vector"] != nil)
        #expect(payload["isNormalized"] as? Bool == true)
        #expect(payload["cachedMagnitude"] as? Double == 1)
        let advisoryVariants: [[String: Any]] = [
            ["isNormalized": true, "cachedMagnitude": 1],
            ["isNormalized": false, "cachedMagnitude": 999],
            [:],
            ["isNormalized": "untrusted", "cachedMagnitude": ["bad"]]
        ]
        for advisory in advisoryVariants {
            var varied = advisory
            varied["vector"] = payload["vector"]
            let data = try JSONSerialization.data(withJSONObject: varied)
            let decoded: NormalizationHint<V>
            do { decoded = try JSONDecoder().decode(NormalizationHint<V>.self, from: data) }
            catch {
                Issue.record("Advisory fields must not prevent decoding: \(error)")
                continue
            }
            #expect(decoded.toArray() == coordinates)
            #expect(decoded.scalarCount == count)
            #expect(!decoded.isNormalized)
            #expect(decoded.cachedMagnitude == nil)
            #expect(abs(decoded.magnitude - 2) < 0.00001)
        }
        let roundTrip = try JSONDecoder().decode(NormalizationHint<V>.self, from: encoded)
        #expect(roundTrip.toArray() == coordinates)
        #expect(!roundTrip.isNormalized)
        #expect(roundTrip.cachedMagnitude == nil)
        #expect(roundTrip != original)
    }

    @Test func decodedMetadataIsAdvisory() throws {
        try exerciseDecoding(DynamicVector.self, count: 7)
        try exerciseDecoding(Vector512Optimized.self, count: 512)
        try exerciseDecoding(Vector<Dim3>.self, count: 3)
    }
}
