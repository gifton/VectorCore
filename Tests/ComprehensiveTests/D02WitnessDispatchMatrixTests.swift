import Foundation
import os
import Testing
import VectorCore

// D02 phase 4 (RD-W1): which VectorProtocol / VectorFactory / IndexableVector members are dynamically dispatched and
// which are statically selected, checked for EVERY member of the three protocols in concrete, generic and
// existential calls through a shadowing conformer. The member list `d02DispatchRows` is the protocols' member set
// in the public symbol graph (Scripts/ci/tests/D02/protocol-members-base.tsv); test_d02_surface_guards.py fails if
// the two lists differ (count guard), and the D02 evidence regenerates the list from the post-change graph.
//
// D02ShadowVector implements every requirement (recording `<path>`) and declares a same-named member for every
// extension member (recording `shadow:<name>`). A dynamic row must record the conformer's implementation in all
// three contexts; a static row must record its shadow for a concrete call and never for a generic or existential
// one. Existential calls use `any IndexableVector` (Scalar == Float) directly where Swift allows it, otherwise
// implicit opening, which is the generic path (labelled "opened").

final class D02DispatchLog: @unchecked Sendable {
    private let lock = OSAllocatedUnfairLock(initialState: [String]())
    func note(_ entry: String) { lock.withLock { $0.append(entry) } }
    var entries: [String] { lock.withLock { $0 } }
}

struct D02ShadowVector: VectorProtocol, VectorFactory, IndexableVector {
    @TaskLocal static var log: D02DispatchLog?
    static func note(_ entry: String) { log?.note(entry) }
    static let width = 4
    private var backing: [Float]
    var storage: [Float] {
        get { Self.note("VectorProtocol.storage"); return backing }
        set { backing = newValue }
    }

    private init(raw: [Float]) { backing = raw }
    private static func shadow(_ name: String) -> Self { note("shadow:\(name)"); return Self(raw: [42, 42, 42, 42]) }

    // MARK: Requirements (dynamic)
    var scalarCount: Int { Self.note("VectorProtocol.scalarCount"); return backing.count }
    init() { Self.note("VectorProtocol.init()"); backing = [0, 0, 0, 0] }
    init(_ array: [Float]) throws {
        Self.note("VectorProtocol.init(_:)")
        guard array.count == Self.width else {
            throw VectorError.dimensionMismatch(expected: Self.width, actual: array.count)
        }
        backing = array
    }
    init(repeating value: Float) { Self.note("VectorProtocol.init(repeating:)"); backing = [value, value, value, value] }
    func toArray() -> [Float] { Self.note("VectorProtocol.toArray()"); return backing }
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        Self.note("VectorProtocol.withUnsafeBufferPointer(_:)")
        return try backing.withUnsafeBufferPointer { try body($0) }
    }
    mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        Self.note("VectorProtocol.withUnsafeMutableBufferPointer(_:)")
        return try backing.withUnsafeMutableBufferPointer { try body($0) }
    }
    var isFinite: Bool { Self.note("VectorProtocol.isFinite"); return backing.allSatisfy(\.isFinite) }
    var isZero: Bool { Self.note("VectorProtocol.isZero"); return backing.allSatisfy { $0 == 0 } }
    static func create(from scalars: [Float]) throws -> Self { note("VectorFactory.create(from:)"); return Self(raw: scalars) }
    var isNormalized: Bool { Self.note("IndexableVector.isNormalized"); return true }
    var cachedMagnitude: Float? { Self.note("IndexableVector.cachedMagnitude"); return 42 }

    // MARK: Collection members VectorProtocol's extension supplies as defaults (dynamic through Collection)
    var startIndex: Int { Self.note("VectorProtocol.startIndex"); return 0 }
    var endIndex: Int { Self.note("VectorProtocol.endIndex"); return backing.count }
    func index(after i: Int) -> Int { Self.note("VectorProtocol.index(after:)"); return i + 1 }
    subscript(index: Int) -> Float { Self.note("VectorProtocol.subscript(_:)"); return backing[index] }

    // MARK: Shadows of extension members (static)
    static func + (lhs: Self, rhs: Self) -> Self { shadow("+") }
    static func - (lhs: Self, rhs: Self) -> Self { shadow("-") }
    static prefix func - (vector: Self) -> Self { shadow("prefix-") }
    static func * (lhs: Self, rhs: Float) -> Self { shadow("*") }
    static func * (lhs: Float, rhs: Self) -> Self { shadow("*") }
    static func / (lhs: Self, rhs: Float) -> Self { shadow("/") }
    static func .* (lhs: Self, rhs: Self) -> Self { shadow(".*") }
    static func ./ (lhs: Self, rhs: Self) -> Self { shadow("./") }
    static func += (lhs: inout Self, rhs: Self) { lhs = shadow("+=") }
    static func -= (lhs: inout Self, rhs: Self) { lhs = shadow("-=") }
    static func *= (lhs: inout Self, rhs: Float) { lhs = shadow("*=") }
    static func /= (lhs: inout Self, rhs: Float) { lhs = shadow("/=") }
    static var zero: Self { shadow("zero") }
    static var ones: Self { shadow("ones") }
    static func random(in range: ClosedRange<Float> = 0...1) -> Self { shadow("random(in:)") }
    func dotProduct(_ other: Self) -> Float { Self.note("shadow:dotProduct(_:)"); return 42 }
    var magnitude: Float { Self.note("shadow:magnitude"); return 42 }
    var magnitudeSquared: Float { Self.note("shadow:magnitudeSquared"); return 42 }
    func normalized() -> Result<Self, VectorError> { .success(Self.shadow("normalized()")) }
    func normalizedUnchecked() -> Self { Self.shadow("normalizedUnchecked()") }
    var l1Norm: Float { Self.note("shadow:l1Norm"); return 42 }
    var lInfinityNorm: Float { Self.note("shadow:lInfinityNorm"); return 42 }
    func euclideanDistance(to other: Self) -> Float { Self.note("shadow:euclideanDistance(to:)"); return 42 }
    func euclideanDistanceSquared(to other: Self) -> Float { Self.note("shadow:euclideanDistanceSquared(to:)"); return 42 }
    func manhattanDistance(to other: Self) -> Float { Self.note("shadow:manhattanDistance(to:)"); return 42 }
    func chebyshevDistance(to other: Self) -> Float { Self.note("shadow:chebyshevDistance(to:)"); return 42 }
    func cosineSimilarity(to other: Self) -> Float { Self.note("shadow:cosineSimilarity(to:)"); return 42 }
    func cosineDistance(to other: Self) -> Float { Self.note("shadow:cosineDistance(to:)"); return 42 }
    func distance(to other: Self, metric: SupportedDistanceMetric = .euclidean) -> Float {
        Self.note("shadow:distance(to:metric:)"); return 42
    }
    var sum: Float { Self.note("shadow:sum"); return 42 }
    var mean: Float { Self.note("shadow:mean"); return 42 }
    var min: Float? { Self.note("shadow:min"); return 42 }
    var max: Float? { Self.note("shadow:max"); return 42 }
    mutating func inPlaceAdd(_ other: Self) { Self.note("shadow:inPlaceAdd(_:)") }
    mutating func inPlaceSubtract(_ other: Self) { Self.note("shadow:inPlaceSubtract(_:)") }
    mutating func inPlaceMultiply(_ scalar: Float) { Self.note("shadow:inPlaceMultiply(_:)") }
    mutating func inPlaceDivide(_ scalar: Float) { Self.note("shadow:inPlaceDivide(_:)") }
    mutating func inPlaceElementwiseMultiply(_ other: Self) { Self.note("shadow:inPlaceElementwiseMultiply(_:)") }
    func mapElements(_ transform: (Float) throws -> Float) rethrows -> Self { Self.shadow("mapElements(_:)") }
    func isApproximatelyEqual(to other: Self, tolerance: Float = 1e-6) -> Bool {
        Self.note("shadow:isApproximatelyEqual(to:tolerance:)"); return false
    }
    func encodeBinary() -> Data { Self.note("shadow:encodeBinary()"); return Data([42]) }
    func encodeJSON(encoder: JSONEncoder = JSONEncoder()) throws -> Data { Self.note("shadow:encodeJSON(encoder:)"); return Data() }
    func encodeJSONString(encoder: JSONEncoder = JSONEncoder()) throws -> String? {
        Self.note("shadow:encodeJSONString(encoder:)"); return nil
    }
    static func createByTransforming<V: VectorProtocol>(_ vector: V, _ transform: (V.Scalar) -> Float) throws -> Self
    where V.Scalar == Float { shadow("createByTransforming(_:_:)") }
    static func createByCombining<V1: VectorProtocol, V2: VectorProtocol>(
        _ v1: V1, _ v2: V2, _ combine: (V1.Scalar, V2.Scalar) -> Float
    ) throws -> Self where V1.Scalar == Float, V2.Scalar == Float { shadow("createByCombining(_:_:_:)") }
    var isApproximatelyNormalized: Bool { Self.note("shadow:isApproximatelyNormalized"); return false }
    func withNormalizationHint() -> NormalizationHint<Self> {
        Self.note("shadow:withNormalizationHint()"); return NormalizationHint(vector: self, isNormalized: false)
    }
    func normalizedWithHint() -> NormalizationHint<Self>? { Self.note("shadow:normalizedWithHint()"); return nil }
    func normalizedUncheckedWithHint() -> NormalizationHint<Self> {
        Self.note("shadow:normalizedUncheckedWithHint()"); return NormalizationHint(vector: self, isNormalized: false)
    }
}

enum D02Dispatch: Sendable { case dynamic, `static` }

/// One row per protocol member path of the public symbol graph (associated types have no call and are listed only
/// for the count guard). `shadow` is the name the conformer's shadow records for a static row.
let d02DispatchRows: [(path: String, dispatch: D02Dispatch?, shadow: String?)] = [
    // VectorProtocol requirements (12, two of them associated types)
    ("VectorProtocol.Scalar", nil, nil), ("VectorProtocol.Storage", nil, nil),
    ("VectorProtocol.storage", .dynamic, nil), ("VectorProtocol.scalarCount", .dynamic, nil),
    ("VectorProtocol.init()", .dynamic, nil), ("VectorProtocol.init(_:)", .dynamic, nil),
    ("VectorProtocol.init(repeating:)", .dynamic, nil), ("VectorProtocol.toArray()", .dynamic, nil),
    ("VectorProtocol.withUnsafeBufferPointer(_:)", .dynamic, nil),
    ("VectorProtocol.withUnsafeMutableBufferPointer(_:)", .dynamic, nil),
    ("VectorProtocol.isFinite", .dynamic, nil), ("VectorProtocol.isZero", .dynamic, nil),
    // VectorProtocol extension: Collection defaults (dynamic through Collection)
    ("VectorProtocol.startIndex", .dynamic, nil), ("VectorProtocol.endIndex", .dynamic, nil),
    ("VectorProtocol.index(after:)", .dynamic, nil), ("VectorProtocol.subscript(_:)", .dynamic, nil),
    // VectorProtocol extension members (static, 42)
    ("VectorProtocol.+(_:_:)", .static, "+"), ("VectorProtocol.-(_:_:)", .static, "-"),
    ("VectorProtocol.-(_:)", .static, "prefix-"), ("VectorProtocol.*(_:_:)", .static, "*"),
    ("VectorProtocol./(_:_:)", .static, "/"), ("VectorProtocol..*(_:_:)", .static, ".*"),
    ("VectorProtocol../(_:_:)", .static, "./"), ("VectorProtocol.+=(_:_:)", .static, "+="),
    ("VectorProtocol.-=(_:_:)", .static, "-="), ("VectorProtocol.*=(_:_:)", .static, "*="),
    ("VectorProtocol./=(_:_:)", .static, "/="), ("VectorProtocol.zero", .static, "zero"),
    ("VectorProtocol.ones", .static, "ones"), ("VectorProtocol.random(in:)", .static, "random(in:)"),
    ("VectorProtocol.dotProduct(_:)", .static, "dotProduct(_:)"), ("VectorProtocol.magnitude", .static, "magnitude"),
    ("VectorProtocol.magnitudeSquared", .static, "magnitudeSquared"),
    ("VectorProtocol.normalized()", .static, "normalized()"),
    ("VectorProtocol.normalizedUnchecked()", .static, "normalizedUnchecked()"),
    ("VectorProtocol.l1Norm", .static, "l1Norm"), ("VectorProtocol.lInfinityNorm", .static, "lInfinityNorm"),
    ("VectorProtocol.euclideanDistance(to:)", .static, "euclideanDistance(to:)"),
    ("VectorProtocol.euclideanDistanceSquared(to:)", .static, "euclideanDistanceSquared(to:)"),
    ("VectorProtocol.manhattanDistance(to:)", .static, "manhattanDistance(to:)"),
    ("VectorProtocol.chebyshevDistance(to:)", .static, "chebyshevDistance(to:)"),
    ("VectorProtocol.cosineSimilarity(to:)", .static, "cosineSimilarity(to:)"),
    ("VectorProtocol.cosineDistance(to:)", .static, "cosineDistance(to:)"),
    ("VectorProtocol.distance(to:metric:)", .static, "distance(to:metric:)"),
    ("VectorProtocol.sum", .static, "sum"), ("VectorProtocol.mean", .static, "mean"),
    ("VectorProtocol.min", .static, "min"), ("VectorProtocol.max", .static, "max"),
    ("VectorProtocol.inPlaceAdd(_:)", .static, "inPlaceAdd(_:)"),
    ("VectorProtocol.inPlaceSubtract(_:)", .static, "inPlaceSubtract(_:)"),
    ("VectorProtocol.inPlaceMultiply(_:)", .static, "inPlaceMultiply(_:)"),
    ("VectorProtocol.inPlaceDivide(_:)", .static, "inPlaceDivide(_:)"),
    ("VectorProtocol.inPlaceElementwiseMultiply(_:)", .static, "inPlaceElementwiseMultiply(_:)"),
    ("VectorProtocol.mapElements(_:)", .static, "mapElements(_:)"),
    ("VectorProtocol.isApproximatelyEqual(to:tolerance:)", .static, "isApproximatelyEqual(to:tolerance:)"),
    ("VectorProtocol.encodeBinary()", .static, "encodeBinary()"),
    ("VectorProtocol.encodeJSON(encoder:)", .static, "encodeJSON(encoder:)"),
    ("VectorProtocol.encodeJSONString(encoder:)", .static, "encodeJSONString(encoder:)"),
    // VectorFactory: one requirement, 12 extension members
    ("VectorFactory.create(from:)", .dynamic, nil),
    ("VectorFactory.+(_:_:)", .static, "+"), ("VectorFactory.-(_:_:)", .static, "-"),
    ("VectorFactory.*(_:_:)", .static, "*"), ("VectorFactory./(_:_:)", .static, "/"),
    ("VectorFactory..*(_:_:)", .static, ".*"), ("VectorFactory../(_:_:)", .static, "./"),
    ("VectorFactory.+=(_:_:)", .static, "+="), ("VectorFactory.-=(_:_:)", .static, "-="),
    ("VectorFactory.*=(_:_:)", .static, "*="), ("VectorFactory./=(_:_:)", .static, "/="),
    ("VectorFactory.createByTransforming(_:_:)", .static, "createByTransforming(_:_:)"),
    ("VectorFactory.createByCombining(_:_:_:)", .static, "createByCombining(_:_:_:)"),
    // IndexableVector: two requirements with defaults, four extension members
    ("IndexableVector.isNormalized", .dynamic, nil), ("IndexableVector.cachedMagnitude", .dynamic, nil),
    ("IndexableVector.isApproximatelyNormalized", .static, "isApproximatelyNormalized"),
    ("IndexableVector.withNormalizationHint()", .static, "withNormalizationHint()"),
    ("IndexableVector.normalizedWithHint()", .static, "normalizedWithHint()"),
    ("IndexableVector.normalizedUncheckedWithHint()", .static, "normalizedUncheckedWithHint()")
]

/// Calls `path` on the concrete type (static selection binds to the shadow here).
private func d02ConcreteCalls() -> [String: (D02ShadowVector, D02ShadowVector) throws -> Void] {
    let table: [String: (D02ShadowVector, D02ShadowVector) throws -> Void] = [
        "VectorProtocol.storage": { a, _ in _ = a.storage },
        "VectorProtocol.scalarCount": { a, _ in _ = a.scalarCount },
        "VectorProtocol.init()": { _, _ in _ = D02ShadowVector() },
        "VectorProtocol.init(_:)": { _, _ in _ = try D02ShadowVector([1, 2, 3, 4]) },
        "VectorProtocol.init(repeating:)": { _, _ in _ = D02ShadowVector(repeating: 1) },
        "VectorProtocol.toArray()": { a, _ in _ = a.toArray() },
        "VectorProtocol.withUnsafeBufferPointer(_:)": { a, _ in _ = a.withUnsafeBufferPointer { $0.count } },
        "VectorProtocol.withUnsafeMutableBufferPointer(_:)": { a, _ in
            var m = a; _ = m.withUnsafeMutableBufferPointer { $0.count }; _ = m
        },
        "VectorProtocol.isFinite": { a, _ in _ = a.isFinite },
        "VectorProtocol.isZero": { a, _ in _ = a.isZero },
        "VectorProtocol.startIndex": { a, _ in _ = a.startIndex },
        "VectorProtocol.endIndex": { a, _ in _ = a.endIndex },
        "VectorProtocol.index(after:)": { a, _ in _ = a.index(after: 0) },
        "VectorProtocol.subscript(_:)": { a, _ in _ = a[1] },
        "VectorProtocol.+(_:_:)": { a, b in _ = a + b },
        "VectorFactory.+(_:_:)": { a, b in _ = a + b },
        "VectorProtocol.-(_:_:)": { a, b in _ = a - b },
        "VectorFactory.-(_:_:)": { a, b in _ = a - b },
        "VectorProtocol.-(_:)": { a, _ in _ = -a },
        "VectorProtocol.*(_:_:)": { a, _ in _ = a * 2; _ = 2 * a },
        "VectorFactory.*(_:_:)": { a, _ in _ = a * 2; _ = 2 * a },
        "VectorProtocol./(_:_:)": { a, _ in _ = a / 2 },
        "VectorFactory./(_:_:)": { a, _ in _ = a / 2 },
        "VectorProtocol..*(_:_:)": { a, b in _ = a .* b },
        "VectorFactory..*(_:_:)": { a, b in _ = a .* b },
        "VectorProtocol../(_:_:)": { a, b in _ = a ./ b },
        "VectorFactory../(_:_:)": { a, b in _ = a ./ b },
        "VectorProtocol.+=(_:_:)": { a, b in var m = a; m += b; _ = m },
        "VectorFactory.+=(_:_:)": { a, b in var m = a; m += b; _ = m },
        "VectorProtocol.-=(_:_:)": { a, b in var m = a; m -= b; _ = m },
        "VectorFactory.-=(_:_:)": { a, b in var m = a; m -= b; _ = m },
        "VectorProtocol.*=(_:_:)": { a, _ in var m = a; m *= 2; _ = m },
        "VectorFactory.*=(_:_:)": { a, _ in var m = a; m *= 2; _ = m },
        "VectorProtocol./=(_:_:)": { a, _ in var m = a; m /= 2; _ = m },
        "VectorFactory./=(_:_:)": { a, _ in var m = a; m /= 2; _ = m },
        "VectorProtocol.zero": { _, _ in _ = D02ShadowVector.zero },
        "VectorProtocol.ones": { _, _ in _ = D02ShadowVector.ones },
        "VectorProtocol.random(in:)": { _, _ in _ = D02ShadowVector.random(in: 0...1) },
        "VectorProtocol.dotProduct(_:)": { a, b in _ = a.dotProduct(b) },
        "VectorProtocol.magnitude": { a, _ in _ = a.magnitude },
        "VectorProtocol.magnitudeSquared": { a, _ in _ = a.magnitudeSquared },
        "VectorProtocol.normalized()": { a, _ in _ = a.normalized() },
        "VectorProtocol.normalizedUnchecked()": { a, _ in _ = a.normalizedUnchecked() },
        "VectorProtocol.l1Norm": { a, _ in _ = a.l1Norm },
        "VectorProtocol.lInfinityNorm": { a, _ in _ = a.lInfinityNorm },
        "VectorProtocol.euclideanDistance(to:)": { a, b in _ = a.euclideanDistance(to: b) },
        "VectorProtocol.euclideanDistanceSquared(to:)": { a, b in _ = a.euclideanDistanceSquared(to: b) },
        "VectorProtocol.manhattanDistance(to:)": { a, b in _ = a.manhattanDistance(to: b) },
        "VectorProtocol.chebyshevDistance(to:)": { a, b in _ = a.chebyshevDistance(to: b) },
        "VectorProtocol.cosineSimilarity(to:)": { a, b in _ = a.cosineSimilarity(to: b) },
        "VectorProtocol.cosineDistance(to:)": { a, b in _ = a.cosineDistance(to: b) },
        "VectorProtocol.distance(to:metric:)": { a, b in _ = a.distance(to: b, metric: .manhattan) },
        "VectorProtocol.sum": { a, _ in _ = a.sum },
        "VectorProtocol.mean": { a, _ in _ = a.mean },
        "VectorProtocol.min": { a, _ in _ = a.min },
        "VectorProtocol.max": { a, _ in _ = a.max },
        "VectorProtocol.inPlaceAdd(_:)": { a, b in var m = a; m.inPlaceAdd(b); _ = m },
        "VectorProtocol.inPlaceSubtract(_:)": { a, b in var m = a; m.inPlaceSubtract(b); _ = m },
        "VectorProtocol.inPlaceMultiply(_:)": { a, _ in var m = a; m.inPlaceMultiply(2); _ = m },
        "VectorProtocol.inPlaceDivide(_:)": { a, _ in var m = a; m.inPlaceDivide(2); _ = m },
        "VectorProtocol.inPlaceElementwiseMultiply(_:)": { a, b in var m = a; m.inPlaceElementwiseMultiply(b); _ = m },
        "VectorProtocol.mapElements(_:)": { a, _ in _ = a.mapElements { $0 + 1 } },
        "VectorProtocol.isApproximatelyEqual(to:tolerance:)": { a, b in _ = a.isApproximatelyEqual(to: b, tolerance: 0.5) },
        "VectorProtocol.encodeBinary()": { a, _ in _ = a.encodeBinary() },
        "VectorProtocol.encodeJSON(encoder:)": { a, _ in _ = try a.encodeJSON() },
        "VectorProtocol.encodeJSONString(encoder:)": { a, _ in _ = try a.encodeJSONString() },
        "VectorFactory.create(from:)": { _, _ in _ = try D02ShadowVector.create(from: [1, 2, 3, 4]) },
        "VectorFactory.createByTransforming(_:_:)": { a, _ in _ = try D02ShadowVector.createByTransforming(a) { $0 * 2 } },
        "VectorFactory.createByCombining(_:_:_:)": { a, b in _ = try D02ShadowVector.createByCombining(a, b) { $0 + $1 } },
        "IndexableVector.isNormalized": { a, _ in _ = a.isNormalized },
        "IndexableVector.cachedMagnitude": { a, _ in _ = a.cachedMagnitude },
        "IndexableVector.isApproximatelyNormalized": { a, _ in _ = a.isApproximatelyNormalized },
        "IndexableVector.withNormalizationHint()": { a, _ in _ = a.withNormalizationHint() },
        "IndexableVector.normalizedWithHint()": { a, _ in _ = a.normalizedWithHint() },
        "IndexableVector.normalizedUncheckedWithHint()": { a, _ in _ = a.normalizedUncheckedWithHint() }
    ]
    return table
}

func d02CallConcrete(_ path: String, _ a: D02ShadowVector, _ b: D02ShadowVector) throws {
    guard let call = d02ConcreteCalls()[path] else {
        Issue.record("no concrete call for \(path)")
        return
    }
    try call(a, b)
}

/// Calls a VectorProtocol member in a context generic over `V: VectorProtocol` (Scalar == Float).
private func d02GenericVPCalls<V: VectorProtocol>(_: V.Type) -> [String: (V, V) throws -> Void] where V.Scalar == Float {
    let table: [String: (V, V) throws -> Void] = [
        "VectorProtocol.storage": { a, _ in _ = a.storage },
        "VectorProtocol.scalarCount": { a, _ in _ = a.scalarCount },
        "VectorProtocol.init()": { _, _ in _ = V() },
        "VectorProtocol.init(_:)": { _, _ in _ = try V([1, 2, 3, 4]) },
        "VectorProtocol.init(repeating:)": { _, _ in _ = V(repeating: 1) },
        "VectorProtocol.toArray()": { a, _ in _ = a.toArray() },
        "VectorProtocol.withUnsafeBufferPointer(_:)": { a, _ in _ = a.withUnsafeBufferPointer { $0.count } },
        "VectorProtocol.withUnsafeMutableBufferPointer(_:)": { a, _ in
            var m = a; _ = m.withUnsafeMutableBufferPointer { $0.count }; _ = m
        },
        "VectorProtocol.isFinite": { a, _ in _ = a.isFinite },
        "VectorProtocol.isZero": { a, _ in _ = a.isZero },
        "VectorProtocol.startIndex": { a, _ in _ = a.startIndex },
        "VectorProtocol.endIndex": { a, _ in _ = a.endIndex },
        "VectorProtocol.index(after:)": { a, _ in _ = a.index(after: 0) },
        "VectorProtocol.subscript(_:)": { a, _ in _ = a[1] },
        "VectorProtocol.+(_:_:)": { a, b in _ = a + b },
        "VectorProtocol.-(_:_:)": { a, b in _ = a - b },
        "VectorProtocol.-(_:)": { a, _ in _ = -a },
        "VectorProtocol.*(_:_:)": { a, _ in _ = a * 2; _ = 2 * a },
        "VectorProtocol./(_:_:)": { a, _ in _ = a / 2 },
        "VectorProtocol..*(_:_:)": { a, b in _ = a .* b },
        "VectorProtocol../(_:_:)": { a, b in _ = a ./ b },
        "VectorProtocol.+=(_:_:)": { a, b in var m = a; m += b; _ = m },
        "VectorProtocol.-=(_:_:)": { a, b in var m = a; m -= b; _ = m },
        "VectorProtocol.*=(_:_:)": { a, _ in var m = a; m *= 2; _ = m },
        "VectorProtocol./=(_:_:)": { a, _ in var m = a; m /= 2; _ = m },
        "VectorProtocol.zero": { _, _ in _ = V.zero },
        "VectorProtocol.ones": { _, _ in _ = V.ones },
        "VectorProtocol.random(in:)": { _, _ in _ = V.random(in: 0...1) },
        "VectorProtocol.dotProduct(_:)": { a, b in _ = a.dotProduct(b) },
        "VectorProtocol.magnitude": { a, _ in _ = a.magnitude },
        "VectorProtocol.magnitudeSquared": { a, _ in _ = a.magnitudeSquared },
        "VectorProtocol.normalized()": { a, _ in _ = a.normalized() },
        "VectorProtocol.normalizedUnchecked()": { a, _ in _ = a.normalizedUnchecked() },
        "VectorProtocol.l1Norm": { a, _ in _ = a.l1Norm },
        "VectorProtocol.lInfinityNorm": { a, _ in _ = a.lInfinityNorm },
        "VectorProtocol.euclideanDistance(to:)": { a, b in _ = a.euclideanDistance(to: b) },
        "VectorProtocol.euclideanDistanceSquared(to:)": { a, b in _ = a.euclideanDistanceSquared(to: b) },
        "VectorProtocol.manhattanDistance(to:)": { a, b in _ = a.manhattanDistance(to: b) },
        "VectorProtocol.chebyshevDistance(to:)": { a, b in _ = a.chebyshevDistance(to: b) },
        "VectorProtocol.cosineSimilarity(to:)": { a, b in _ = a.cosineSimilarity(to: b) },
        "VectorProtocol.cosineDistance(to:)": { a, b in _ = a.cosineDistance(to: b) },
        "VectorProtocol.distance(to:metric:)": { a, b in _ = a.distance(to: b, metric: .manhattan) },
        "VectorProtocol.sum": { a, _ in _ = a.sum },
        "VectorProtocol.mean": { a, _ in _ = a.mean },
        "VectorProtocol.min": { a, _ in _ = a.min },
        "VectorProtocol.max": { a, _ in _ = a.max },
        "VectorProtocol.inPlaceAdd(_:)": { a, b in var m = a; m.inPlaceAdd(b); _ = m },
        "VectorProtocol.inPlaceSubtract(_:)": { a, b in var m = a; m.inPlaceSubtract(b); _ = m },
        "VectorProtocol.inPlaceMultiply(_:)": { a, _ in var m = a; m.inPlaceMultiply(2); _ = m },
        "VectorProtocol.inPlaceDivide(_:)": { a, _ in var m = a; m.inPlaceDivide(2); _ = m },
        "VectorProtocol.inPlaceElementwiseMultiply(_:)": { a, b in var m = a; m.inPlaceElementwiseMultiply(b); _ = m },
        "VectorProtocol.mapElements(_:)": { a, _ in _ = a.mapElements { $0 + 1 } },
        "VectorProtocol.isApproximatelyEqual(to:tolerance:)": { a, b in _ = a.isApproximatelyEqual(to: b, tolerance: 0.5) },
        "VectorProtocol.encodeBinary()": { a, _ in _ = a.encodeBinary() },
        "VectorProtocol.encodeJSON(encoder:)": { a, _ in _ = try a.encodeJSON() },
        "VectorProtocol.encodeJSONString(encoder:)": { a, _ in _ = try a.encodeJSONString() }
    ]
    return table
}

func d02CallGenericVP<V: VectorProtocol>(_ path: String, _ a: V, _ b: V) throws where V.Scalar == Float {
    guard let call = d02GenericVPCalls(V.self)[path] else {
        Issue.record("no generic VectorProtocol call for \(path)")
        return
    }
    try call(a, b)
}

/// Calls a VectorFactory member in a context generic over `V: VectorFactory`.
func d02CallGenericVF<V: VectorFactory>(_ path: String, _ a: V, _ b: V) throws {
    var m = a
    switch path {
    case "VectorFactory.create(from:)": _ = try V.create(from: a.toArray())
    case "VectorFactory.+(_:_:)": _ = a + b
    case "VectorFactory.-(_:_:)": _ = a - b
    case "VectorFactory.*(_:_:)": _ = a * 2; _ = 2 * a
    case "VectorFactory./(_:_:)": _ = a / 2
    case "VectorFactory..*(_:_:)": _ = a .* b
    case "VectorFactory../(_:_:)": _ = a ./ b
    case "VectorFactory.+=(_:_:)": m += b
    case "VectorFactory.-=(_:_:)": m -= b
    case "VectorFactory.*=(_:_:)": m *= 2
    case "VectorFactory./=(_:_:)": m /= 2
    case "VectorFactory.createByTransforming(_:_:)": _ = try V.createByTransforming(a) { $0 }
    case "VectorFactory.createByCombining(_:_:_:)": _ = try V.createByCombining(a, b) { $0 + $1 }
    default: Issue.record("no generic VectorFactory call for \(path)")
    }
}

/// Calls an IndexableVector member in a context generic over `V: IndexableVector`.
func d02CallGenericIV<V: IndexableVector>(_ path: String, _ a: V, _ b: V) throws {
    switch path {
    case "IndexableVector.isNormalized": _ = a.isNormalized
    case "IndexableVector.cachedMagnitude": _ = a.cachedMagnitude
    case "IndexableVector.isApproximatelyNormalized": _ = a.isApproximatelyNormalized
    case "IndexableVector.withNormalizationHint()": _ = a.withNormalizationHint()
    case "IndexableVector.normalizedWithHint()": _ = a.normalizedWithHint()
    case "IndexableVector.normalizedUncheckedWithHint()": _ = a.normalizedUncheckedWithHint()
    default: Issue.record("no generic IndexableVector call for \(path)")
    }
}

private func d02OpenIV<V: IndexableVector>(_ a: V, _ b: any IndexableVector, _ path: String) throws {
    let w = b as! V
    if path.hasPrefix("IndexableVector.") { try d02CallGenericIV(path, a, w) } else { try d02CallGenericVP(path, a, w) }
}

private func d02OpenVF<V: VectorFactory>(_ a: V, _ b: any VectorFactory, _ path: String) throws {
    try d02CallGenericVF(path, a, b as! V)
}

/// Members callable directly on `any IndexableVector` (Scalar == Float); everything else is opened.
private let d02ExistentialDirectCalls: [String: @Sendable (any IndexableVector) throws -> Void] = [
    "VectorProtocol.storage": { e in _ = e.storage },
    "VectorProtocol.scalarCount": { e in _ = e.scalarCount },
    "VectorProtocol.toArray()": { e in _ = e.toArray() },
    "VectorProtocol.withUnsafeBufferPointer(_:)": { e in _ = e.withUnsafeBufferPointer { $0.count } },
    "VectorProtocol.withUnsafeMutableBufferPointer(_:)": { e in var m = e; _ = m.withUnsafeMutableBufferPointer { $0.count }; _ = m },
    "VectorProtocol.isFinite": { e in _ = e.isFinite },
    "VectorProtocol.isZero": { e in _ = e.isZero },
    "VectorProtocol.startIndex": { e in _ = e.startIndex },
    "VectorProtocol.endIndex": { e in _ = e.endIndex },
    "VectorProtocol.magnitude": { e in _ = e.magnitude },
    "VectorProtocol.magnitudeSquared": { e in _ = e.magnitudeSquared },
    "VectorProtocol.l1Norm": { e in _ = e.l1Norm },
    "VectorProtocol.lInfinityNorm": { e in _ = e.lInfinityNorm },
    "VectorProtocol.sum": { e in _ = e.sum },
    "VectorProtocol.mean": { e in _ = e.mean },
    "VectorProtocol.min": { e in _ = e.min },
    "VectorProtocol.max": { e in _ = e.max },
    "IndexableVector.isNormalized": { e in _ = e.isNormalized },
    "IndexableVector.cachedMagnitude": { e in _ = e.cachedMagnitude },
    "IndexableVector.isApproximatelyNormalized": { e in _ = e.isApproximatelyNormalized }
]

/// Calls `path` on an existential: directly where Swift allows it ("direct"), otherwise by implicit opening.
func d02CallExistential(_ path: String, _ a: D02ShadowVector, _ b: D02ShadowVector) throws -> String {
    let e: any IndexableVector = a
    if let call = d02ExistentialDirectCalls[path] {
        try call(e)
        return "direct"
    }
    if path.hasPrefix("VectorFactory.") {
        let f: any VectorFactory = a
        try d02OpenVF(f, b, path)
    } else {
        try d02OpenIV(e, b, path)
    }
    return "opened"
}

@Suite("D02 witness and static dispatch matrix")
struct D02WitnessDispatchMatrixTests {
    private static func logged(_ body: () throws -> Void) rethrows -> [String] {
        let log = D02DispatchLog()
        try D02ShadowVector.$log.withValue(log) { try body() }
        return log.entries
    }

    @Test("Every requirement is dynamic and every extension member static, in concrete, generic and existential calls")
    func everyMemberFollowsItsDispatch() throws {
        let a = try D02ShadowVector([1, 2, 3, 4]), b = try D02ShadowVector([4, 3, 2, 1])
        var existentialForms: [String: String] = [:]
        for row in d02DispatchRows {
            guard let dispatch = row.dispatch else { continue }
            let concrete = try Self.logged { try d02CallConcrete(row.path, a, b) }
            let generic = try Self.logged {
                if row.path.hasPrefix("VectorFactory.") {
                    try d02CallGenericVF(row.path, a, b)
                } else if row.path.hasPrefix("IndexableVector.") {
                    try d02CallGenericIV(row.path, a, b)
                } else {
                    try d02CallGenericVP(row.path, a, b)
                }
            }
            var form = ""
            let existential = try Self.logged { form = try d02CallExistential(row.path, a, b) }
            existentialForms[row.path] = form
            switch dispatch {
            case .dynamic:
                #expect(concrete.contains(row.path), "concrete \(row.path): \(concrete)")
                #expect(generic.contains(row.path), "generic \(row.path): \(generic)")
                #expect(existential.contains(row.path), "existential \(row.path): \(existential)")
            case .static:
                let shadow = "shadow:\(row.shadow ?? "?")"
                #expect(concrete.contains(shadow), "concrete \(row.path): \(concrete)")
                #expect(!generic.contains(shadow), "generic \(row.path): \(generic)")
                #expect(!existential.contains(shadow), "existential \(row.path): \(existential)")
            }
        }
        print("D02_DISPATCH_MATRIX rows=\(d02DispatchRows.count) existential=\(existentialForms.sorted { $0.key < $1.key })")
        #expect(d02DispatchRows.count == 58 + 13 + 6)
        #expect(d02DispatchRows.filter { $0.dispatch == .dynamic }.count == 10 + 4 + 1 + 2)
    }
}

/// The VectorProtocol extension's cosine distance, forced through a generic context (static selection).
private func d02ExtensionCosineDistance<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float {
    a.cosineDistance(to: b)
}

private func d02IndexableIsNormalized<V: IndexableVector>(_ v: V) -> Bool { v.isNormalized }

/// The five members whose binding changed for code generic over the dimension (I3), read generically over `D`.
private func d02ReadingsOverD<D: StaticDimension>(_ v: Vector<D>, _ w: Vector<D>) -> [UInt32] {
    [v.isNormalized ? 1 : 0, v.l1Norm.bitPattern, v.lInfinityNorm.bitPattern,
     v.manhattanDistance(to: w).bitPattern, v.chebyshevDistance(to: w).bitPattern]
}

// Single-member generic-over-D readers (RR-I1): each call site binds the member on its own.
private func d02OverDL1<D: StaticDimension>(_ v: Vector<D>) -> Float { v.l1Norm }
private func d02OverDLInf<D: StaticDimension>(_ v: Vector<D>) -> Float { v.lInfinityNorm }
private func d02OverDManhattan<D: StaticDimension>(_ v: Vector<D>, _ w: Vector<D>) -> Float { v.manhattanDistance(to: w) }
private func d02OverDChebyshev<D: StaticDimension>(_ v: Vector<D>, _ w: Vector<D>) -> Float { v.chebyshevDistance(to: w) }
private func d02OverDIsNormalized<D: StaticDimension>(_ v: Vector<D>) -> Bool { v.isNormalized }

/// The provider calls `body` makes under a fresh recording provider.
private func d02ProviderCalls(_ body: () -> Void) -> [String] {
    let spy = D02SIMDSpy()
    Operations.$simdProvider.withValue(spy) { body() }
    return spy.record.calls
}

private func d02ConcreteReadings2(_ v: Vector<Dim2>, _ w: Vector<Dim2>) -> [UInt32] {
    [v.isNormalized ? 1 : 0, v.l1Norm.bitPattern, v.lInfinityNorm.bitPattern,
     v.manhattanDistance(to: w).bitPattern, v.chebyshevDistance(to: w).bitPattern]
}

private func d02ConcreteReadings128(_ v: Vector<Dim128>, _ w: Vector<Dim128>) -> [UInt32] {
    [v.isNormalized ? 1 : 0, v.l1Norm.bitPattern, v.lInfinityNorm.bitPattern,
     v.manhattanDistance(to: w).bitPattern, v.chebyshevDistance(to: w).bitPattern]
}

private func d02ConcreteReadings512(_ v: Vector<Dim512>, _ w: Vector<Dim512>) -> [UInt32] {
    [v.isNormalized ? 1 : 0, v.l1Norm.bitPattern, v.lInfinityNorm.bitPattern,
     v.manhattanDistance(to: w).bitPattern, v.chebyshevDistance(to: w).bitPattern]
}

@Suite("D02 VectorProtocol.cosineDistance(to:) and IndexableVector witnesses")
struct D02CosineAndWitnessTests {
    /// Values in [-1, 1) on a 2^-20 grid (normal range; no zero class, no overflow).
    private static func values(_ n: Int, seed: UInt32) -> [Float] {
        var state = seed
        return (0..<n).map { _ in
            state = state &* 1_664_525 &+ 1_013_904_223
            return Float(Int32(bitPattern: state) >> 11) / Float(1 << 20)
        }
    }

    /// RD-R6 tolerance. Each implementation computes s = dot / (norm(a) norm(b)) (as dot / (|a| |b|) or as
    /// dot / sqrt(|a|^2 |b|^2)) with n-term Float sums in its own order. By Cauchy-Schwarz the dot product's
    /// rounding error is at most gamma_n relative to |a||b|; the norm product's relative error is at most
    /// gamma_{n+3} (n-term sums of squares, the product, the square root or roots), times |s| <= 1; the division
    /// adds u. So |s_hat - s| <= gamma_n + gamma_{n+4}, the clamp to [-1, 1] does not increase a difference, and
    /// 1 - s adds u. Two implementations differ by at most 2 (gamma_n + gamma_{n+4}) + 4u (u = 2^-24).
    private static func tolerance(_ n: Int) -> Float {
        let u = 0x1p-24, gamma = { (k: Double) in k * u / (1 - k * u) }
        return Float(2 * (gamma(Double(n)) + gamma(Double(n + 4))) + 4 * u)
    }

    @Test func cosineDistanceStaysWithinTheDerivedBoundOfTheWitness() throws {
        func check<V: VectorProtocol>(_ make: ([Float]) throws -> V, _ n: Int) throws where V.Scalar == Float {
            for seed in [UInt32(1), 7, 99] {
                let a = try make(Self.values(n, seed: seed)), b = try make(Self.values(n, seed: seed &+ 1000))
                let mine = d02ExtensionCosineDistance(a, b)
                let witness = CosineDistance().distance(a, b)
                #expect(Swift.abs(mine - witness) <= Self.tolerance(n), "\(V.self) seed \(seed): \(mine) vs \(witness)")
                #expect(d02ExtensionCosineDistance(a, a) <= Self.tolerance(n), "\(V.self): self distance")
            }
            // T15 policy for this member: NaN or infinite similarity gives NaN; a zero vector gives similarity 0.
            var nanValues = Self.values(n, seed: 5); nanValues[0] = .nan
            var infValues = Self.values(n, seed: 6); infValues[1] = .infinity
            let finite = try make(Self.values(n, seed: 8))
            #expect(d02ExtensionCosineDistance(try make(nanValues), finite).isNaN, "\(V.self) NaN")
            #expect(d02ExtensionCosineDistance(finite, try make(infValues)).isNaN, "\(V.self) infinity")
            #expect(d02ExtensionCosineDistance(try make([Float](repeating: 0, count: n)), finite) == 1, "\(V.self) zero")
            // B-M1: both operands zero (similarity 0, distance 1), a negative infinity, NaN in both operands.
            let zero = try make([Float](repeating: 0, count: n))
            var negInfValues = Self.values(n, seed: 9); negInfValues[2] = -.infinity
            var nanOther = Self.values(n, seed: 10); nanOther[n - 1] = .nan
            #expect(d02ExtensionCosineDistance(zero, zero) == 1, "\(V.self) both zero")
            #expect(d02ExtensionCosineDistance(try make(negInfValues), finite).isNaN, "\(V.self) -infinity")
            #expect(d02ExtensionCosineDistance(try make(nanValues), try make(nanOther)).isNaN, "\(V.self) NaN both")
        }
        try check({ try Vector<Dim128>($0) }, 128)
        try check({ try Vector<Dim512>($0) }, 512)
        try check({ DynamicVector($0) }, 100)
        try check({ try Vector384Optimized($0) }, 384)
        try check({ try Vector512Optimized($0) }, 512)
        try check({ try Vector768Optimized($0) }, 768)
        try check({ try Vector1536Optimized($0) }, 1536)
    }

    @Test("IndexableVector.isNormalized: Vector<D> answers with its own check; the other conformers keep the default false")
    func indexableIsNormalizedWitness() throws {
        let unit4 = try Vector<Dim4>([0.5, 0.5, 0.5, 0.5])
        let nonUnit4 = try Vector<Dim4>([1, 2, 3, 4])
        #expect(d02IndexableIsNormalized(unit4) == unit4.isNormalized && unit4.isNormalized)
        #expect(d02IndexableIsNormalized(nonUnit4) == nonUnit4.isNormalized && !nonUnit4.isNormalized)
        let unit512 = try Vector<Dim512>([1] + [Float](repeating: 0, count: 511))
        #expect(d02IndexableIsNormalized(unit512))
        let existential: any IndexableVector = unit512
        #expect(existential.isNormalized)
        #expect(!d02IndexableIsNormalized(DynamicVector([1, 0, 0, 0])))
        #expect(!d02IndexableIsNormalized(try Vector512Optimized([1] + [Float](repeating: 0, count: 511))))
        #expect(d02IndexableIsNormalized(NormalizationHint(vector: unit4, isNormalized: false)) == false)
    }

    @Test("I3: code generic over the dimension reads Vector's own members, bit-identical to concrete reads")
    func genericOverDimensionEqualsConcrete() throws {
        let unit2 = try Vector<Dim2>([0.6, 0.8]), other2 = try Vector<Dim2>([3, -4])
        let unit128 = try Vector<Dim128>([1] + [Float](repeating: 0, count: 127))
        let other128 = try Vector<Dim128>((0..<128).map { Float($0 % 9) - 4.25 })
        let unit512 = try Vector<Dim512>((0..<512).map { _ in Float(1) / Float(512).squareRoot() })
        let other512 = try Vector<Dim512>((0..<512).map { Float($0 % 13) * 0.5 - 3 })
        for (v, w) in [(unit2, other2), (other2, unit2)] {
            #expect(d02ReadingsOverD(v, w) == d02ConcreteReadings2(v, w), "Dim2")
        }
        for (v, w) in [(unit128, other128), (other128, unit128)] {
            #expect(d02ReadingsOverD(v, w) == d02ConcreteReadings128(v, w), "Dim128")
        }
        for (v, w) in [(unit512, other512), (other512, unit512)] {
            #expect(d02ReadingsOverD(v, w) == d02ConcreteReadings512(v, w), "Dim512")
        }
        // The generic-over-D isNormalized read is the real check (before D02 it read IndexableVector's default false).
        #expect(d02ReadingsOverD(unit2, other2)[0] == 1 && d02ReadingsOverD(unit128, other128)[0] == 1)
        #expect(d02ReadingsOverD(other2, unit2)[0] == 0)
        // Under a recording provider the readings still equal the concrete ones.
        for width in [2, 512] {
            let readings: [UInt32] = Operations.$simdProvider.withValue(D02SIMDSpy()) {
                width == 2 ? d02ReadingsOverD(unit2, other2) : d02ReadingsOverD(unit512, other512)
            }
            #expect(readings == (width == 2 ? d02ConcreteReadings2(unit2, other2) : d02ConcreteReadings512(unit512, other512)))
        }
        // Per member (RR-I1): each member read on its own, generically over D, under a fresh recording provider,
        // must make exactly Vector's own provider calls. The VectorProtocol extension fallbacks of the four norms
        // (`VectorProtocol.l1Norm`, `lInfinityNorm`, `manhattanDistance(to:)`, `chebyshevDistance(to:)`) read no
        // provider, and the `-` operator they and `Vector.chebyshevDistance` use is a protocol-extension operator
        // (VectorProtocol or VectorFactory) that reads none either, so a re-gated member records [] here.
        #expect(d02ProviderCalls { _ = d02OverDL1(other512) } == ["abs"])   // Vector.l1Norm: provider abs above 32 lanes
        #expect(d02ProviderCalls { _ = d02OverDLInf(other2) } == ["abs"])   // Vector.lInfinityNorm: provider abs
        // Vector.manhattanDistance(to:): provider subtract then abs above 32 lanes
        #expect(d02ProviderCalls { _ = d02OverDManhattan(other512, unit512) } == ["subtract", "abs"])
        // Vector.chebyshevDistance(to:): `-` (no provider call), then Vector.lInfinityNorm (provider abs)
        #expect(d02ProviderCalls { _ = d02OverDChebyshev(other2, unit2) } == ["abs"])
        // isNormalized reads no provider; only Vector's member answers true for a unit vector (the
        // IndexableVector default it would otherwise bind to is false).
        #expect(d02OverDIsNormalized(unit2) && d02OverDIsNormalized(unit128) && !d02OverDIsNormalized(other2))
    }
}
