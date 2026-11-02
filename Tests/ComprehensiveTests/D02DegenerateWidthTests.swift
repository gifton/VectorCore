import Foundation
import Testing
import VectorCore

// D02 fix round 1 (A2-I1 class closure): every symbol that the removed `D.Storage` gate shielded
// (Scripts/ci/tests/D02/gated-public-base.tsv, 61 symbols) evaluated at the degenerate widths 0 and 1 on value-only
// dimensions, each with its expected outcome: exact bits, NaN, empty, nil, a thrown error kind, or a trap (run in a
// child process through ChildProbeLauncher). `test_d02_surface_guards.py` ties the rows between the
// `BEGIN/END D02 EDGE ROWS` markers to the TSV, so a missing symbol fails.
//
// Documented edges (behaviour unchanged by D02, now stated in the member docs): `sparsity` is NaN at width 0 (0/0),
// `linspace` returns `[from]` at width 1, and `minElement()` / `maxElement()` / `unitX` / `basis(axis: 0)` /
// `e(0)` / `oneHot(at: 0)` trap at width 0. Trap messages are printed by Debug builds only (a Release `precondition`
// failure traps without its message), so the message check runs under `#if DEBUG`.
//
// Inputs: width 1 uses a = [2.5], b = [-1.5] (safeLog uses [1], squareRoot [6.25]); width 0 uses empty vectors.

enum D02EdgeV0: StaticDimension { static let value = 0 }
enum D02EdgeV1: StaticDimension { static let value = 1 }

enum D02EdgeOutcome: Equatable, Sendable, CustomStringConvertible {
    case floats([UInt32])      // exact Float bit patterns; [] is the empty vector
    case scalar(UInt32)        // exact Float bits
    case nan
    case flag(Bool)
    case text(String)          // structured results rendered exactly
    case isNil
    case thrown(String)        // VectorError kind
    case trap(String)          // child-isolated; the message is checked in Debug builds

    var description: String {
        switch self {
        case .floats(let bits): return "floats\(bits.map { Float(bitPattern: $0) })"
        case .scalar(let bits): return "scalar(\(Float(bitPattern: bits)))"
        case .nan: return "nan"
        case .flag(let value): return "flag(\(value))"
        case .text(let value): return "text(\(value))"
        case .isNil: return "nil"
        case .thrown(let kind): return "thrown(\(kind))"
        case .trap(let message): return "trap(\(message))"
        }
    }
}

private func d02Floats(_ values: [Float]) -> D02EdgeOutcome { .floats(values.map(\.bitPattern)) }
private func d02Scalar(_ value: Float) -> D02EdgeOutcome { value.isNaN ? .nan : .scalar(value.bitPattern) }

/// One closure per formerly gated symbol, for dimension `D` (width 0 or 1).
func d02EdgeRows<D: StaticDimension>(_: D.Type) -> [String: () throws -> D02EdgeOutcome] {
    let w = D.value
    let a = try! Vector<D>(w == 0 ? [] : [2.5])
    let b = try! Vector<D>(w == 0 ? [] : [-1.5])
    let one = try! Vector<D>(w == 0 ? [] : [1])
    let square = try! Vector<D>(w == 0 ? [] : [6.25])
    func v(_ vector: Vector<D>) -> D02EdgeOutcome { d02Floats(vector.toArray()) }
    let rows: [String: () throws -> D02EdgeOutcome] = [
        "Vector.range(from:)": { v(Vector<D>.range(from: 1.5)) },
        "Vector.generate(_:)": { v(Vector<D>.generate({ Float($0) + 0.5 })) },
        "Vector.init(indexMap:)": { v(Vector<D>(indexMap: { Float($0) - 1 })) },
        "Vector.geometric(initial:ratio:)": { v(Vector<D>.geometric(initial: 3, ratio: 0.5)) },
        "Vector.powers(of:)": { v(Vector<D>.powers(of: 2)) },
        "Vector.alternating(magnitude:)": { v(Vector<D>.alternating(magnitude: 2)) },
        "Vector.oneHot(at:)": { v(Vector<D>.oneHot(at: 0)) },
        "Vector.basis(axis:)": { v(Vector<D>.basis(axis: 0)) },
        "Vector.oneHotThrowing(at:)": { v(try Vector<D>.oneHotThrowing(at: 0)) },
        "Vector.repeatingPattern(_:)": { v(Vector<D>.repeatingPattern([4, 5])) },
        "Vector.repeatingPatternThrowing(_:)": { v(try Vector<D>.repeatingPatternThrowing([4, 5])) },
        "Vector.repeatingPattern(safe:)": { Vector<D>.repeatingPattern(safe: [4, 5]).map(v) ?? .isNil },
        "Vector.sparse(value:at:)": { v(Vector<D>.sparse(value: 3, at: [0])) },
        "Vector.basisThrowing(axis:)": { v(try Vector<D>.basisThrowing(axis: 0)) },
        "Vector.unitX": { v(Vector<D>.unitX) },
        "Vector.unitY": { v(Vector<D>.unitY) },
        "Vector.unitZ": { v(Vector<D>.unitZ) },
        "Vector.basis(safe:)": { Vector<D>.basis(safe: 0).map(v) ?? .isNil },
        "Vector.e(_:)": { v(Vector<D>.e(0)) },
        "Vector.eThrowing(_:)": { v(try Vector<D>.eThrowing(0)) },
        "Vector.linspace(from:to:)": { v(Vector<D>.linspace(from: -1, to: 3)) },
        "Vector.range()": { v(Vector<D>.range()) },
        "Vector.isFinite": { .flag(a.isFinite) },
        "Vector.hasNaN": { .flag(a.hasNaN) },
        "Vector.hasInfinity": { .flag(a.hasInfinity) },
        "Vector.handleNonFinite(options:)": { v(try a.handleNonFinite(options: .replaceAll)) },
        "Vector.replacingNonFinite(with:)": { v(a.replacingNonFinite(with: 0)) },
        "Vector.finiteValues()": { .text("\(a.finiteValues().values)\(a.finiteValues().indices)") },
        "Vector.safeDivide(by:options:)": { v(try a.safeDivide(by: 2, options: .replaceAll)) },
        "Vector.safeNormalized(options:)": { v(try a.safeNormalized(options: .replaceAll)) },
        "Vector.safeLog(options:)": { v(try one.safeLog(options: .replaceAll)) },
        "Vector.checkNonFinite()": { .flag(a.checkNonFinite().hasNonFinite) },
        "Vector.lerpUnclamped(to:t:)": { v(a.lerpUnclamped(to: b, t: 1.5)) },
        "Vector.smoothstep(to:t:)": { v(a.smoothstep(to: b, t: 0.5)) },
        "Vector.absoluteValue()": { v(b.absoluteValue()) },
        "Vector.squareRoot()": { v(square.squareRoot()) },
        "Vector.min(_:)": { v(a.min(b)) },
        "Vector..<(_:_:)": { v(a .< b) },
        "Vector..>(_:_:)": { v(a .> b) },
        "Vector.max(_:)": { v(a.max(b)) },
        "Vector.minElement()": { .text("\(a.minElement())") },
        "Vector.maxElement()": { .text("\(a.maxElement())") },
        "Vector.clamped(to:)": { v(a.clamped(to: -1...1)) },
        "Vector.clamp(to:)": { var c = a; c.clamp(to: -1...1); return v(c) },
        "Vector.lerp(to:t:)": { v(a.lerp(to: b, t: 0.25)) },
        "Vector.entropyFast": { d02Scalar(a.entropyFast) },   // deprecated spelling of entropy (RD-R4), still covered
        "Vector.l1Norm": { d02Scalar(a.l1Norm) },
        "Vector.l2Norm": { d02Scalar(a.l2Norm) },
        "Vector.lInfinityNorm": { d02Scalar(a.lInfinityNorm) },
        "Vector.manhattanDistance(to:)": { d02Scalar(a.manhattanDistance(to: b)) },
        "Vector.chebyshevDistance(to:)": { d02Scalar(a.chebyshevDistance(to: b)) },
        "VectorMath.nearestNeighbors(query:in:k:using:)": {
            let found = VectorMath.nearestNeighbors(query: a, in: [a, b], k: 2, using: { $0.manhattanDistance(to: $1) })
            return .text("\(found.map { "\($0.index):\($0.distance)" })")
        },
        "VectorMath.pairwiseDistances(_:using:)": {
            .text("\(VectorMath.pairwiseDistances([a, b], using: { $0.chebyshevDistance(to: $1) }))")
        },
        "Vector.sparsity(threshold:)": { d02Scalar(a.sparsity(threshold: 0.01)) },
        "Vector.entropy": { d02Scalar(a.entropy) },
        "Vector.base64Encoded": { v(try Vector<D>.base64Decoded(from: a.base64Encoded)) },
        "Vector.base64Decoded(from:)": { v(try Vector<D>.base64Decoded(from: b.base64Encoded)) },
        "Vector.normalizedFast()": { v(a.normalizedFast()) },
        "Vector.normalizeFast()": { var c = a; c.normalizeFast(); return v(c) },
        "Vector.isNormalized": { .flag(a.isNormalized) },
        "Vector.normalized(tolerance:)": { v(a.normalized(tolerance: 1e-6)) }
    ]
    return rows
}

private func f(_ values: [Float]) -> D02EdgeOutcome { d02Floats(values) }
private func s(_ value: Float) -> D02EdgeOutcome { d02Scalar(value) }
private let axis = D02EdgeOutcome.trap("Axis index out of bounds")
private let empty = D02EdgeOutcome.floats([])
private let outOfBounds = D02EdgeOutcome.thrown("indexOutOfBounds")

// Expected outcome per symbol: (width 0, width 1).
// BEGIN D02 EDGE ROWS
let d02EdgeExpected: [String: (v0: D02EdgeOutcome, v1: D02EdgeOutcome)] = [
    "Vector.range(from:)": (empty, f([1.5])),
    "Vector.generate(_:)": (empty, f([0.5])),
    "Vector.init(indexMap:)": (empty, f([-1])),
    "Vector.geometric(initial:ratio:)": (empty, f([3])),
    "Vector.powers(of:)": (empty, f([1])),
    "Vector.alternating(magnitude:)": (empty, f([2])),
    "Vector.oneHot(at:)": (axis, f([1])),
    "Vector.basis(axis:)": (axis, f([1])),
    "Vector.oneHotThrowing(at:)": (outOfBounds, f([1])),
    "Vector.repeatingPattern(_:)": (empty, f([4])),
    "Vector.repeatingPatternThrowing(_:)": (empty, f([4])),
    "Vector.repeatingPattern(safe:)": (empty, f([4])),
    "Vector.sparse(value:at:)": (empty, f([3])),
    "Vector.basisThrowing(axis:)": (outOfBounds, f([1])),
    "Vector.unitX": (axis, f([1])),
    "Vector.unitY": (empty, f([0])),
    "Vector.unitZ": (empty, f([0])),
    "Vector.basis(safe:)": (.isNil, f([1])),
    "Vector.e(_:)": (axis, f([1])),
    "Vector.eThrowing(_:)": (outOfBounds, f([1])),
    "Vector.linspace(from:to:)": (empty, f([-1])),              // width 1: [from] (documented)
    "Vector.range()": (empty, f([0])),
    "Vector.isFinite": (.flag(true), .flag(true)),
    "Vector.hasNaN": (.flag(false), .flag(false)),
    "Vector.hasInfinity": (.flag(false), .flag(false)),
    "Vector.handleNonFinite(options:)": (empty, f([2.5])),
    "Vector.replacingNonFinite(with:)": (empty, f([2.5])),
    "Vector.finiteValues()": (.text("[][]"), .text("[2.5][0]")),
    "Vector.safeDivide(by:options:)": (empty, f([1.25])),
    "Vector.safeNormalized(options:)": (empty, f([1])),
    "Vector.safeLog(options:)": (empty, f([0])),
    "Vector.checkNonFinite()": (.flag(false), .flag(false)),
    "Vector.lerpUnclamped(to:t:)": (empty, f([-3.5])),
    "Vector.smoothstep(to:t:)": (empty, f([0.5])),
    "Vector.absoluteValue()": (empty, f([1.5])),
    "Vector.squareRoot()": (empty, f([2.5])),
    "Vector.min(_:)": (empty, f([-1.5])),
    "Vector..<(_:_:)": (empty, f([-1.5])),
    "Vector..>(_:_:)": (empty, f([2.5])),
    "Vector.max(_:)": (empty, f([2.5])),
    "Vector.minElement()": (.trap("minElement requires a nonzero dimension"), .text("(value: 2.5, index: 0)")),
    "Vector.maxElement()": (.trap("maxElement requires a nonzero dimension"), .text("(value: 2.5, index: 0)")),
    "Vector.clamped(to:)": (empty, f([1])),
    "Vector.clamp(to:)": (empty, f([1])),
    "Vector.lerp(to:t:)": (empty, f([1.5])),
    "Vector.entropyFast": (s(0), s(0)),
    "Vector.l1Norm": (s(0), s(2.5)),
    "Vector.l2Norm": (s(0), s(2.5)),
    "Vector.lInfinityNorm": (s(0), s(2.5)),
    "Vector.manhattanDistance(to:)": (s(0), s(4)),
    "Vector.chebyshevDistance(to:)": (s(0), s(4)),
    "VectorMath.nearestNeighbors(query:in:k:using:)": (.text(#"["0:0.0", "1:0.0"]"#), .text(#"["0:0.0", "1:4.0"]"#)),
    "VectorMath.pairwiseDistances(_:using:)": (.text("[[0.0, 0.0], [0.0, 0.0]]"), .text("[[0.0, 4.0], [4.0, 0.0]]")),
    "Vector.sparsity(threshold:)": (.nan, s(0)),                // width 0: 0/0 (documented)
    "Vector.entropy": (s(0), s(0)),
    "Vector.base64Encoded": (empty, f([2.5])),
    "Vector.base64Decoded(from:)": (empty, f([-1.5])),
    "Vector.normalizedFast()": (empty, f([1])),
    "Vector.normalizeFast()": (empty, f([1])),
    "Vector.isNormalized": (.flag(false), .flag(false)),
    "Vector.normalized(tolerance:)": (empty, f([1]))
]
// END D02 EDGE ROWS

private func d02Evaluate(_ row: () throws -> D02EdgeOutcome) -> D02EdgeOutcome {
    do { return try row() } catch let error as VectorError { return .thrown(error.kind.rawValue) } catch {
        return .thrown("\(type(of: error))")
    }
}

@Suite("D02 degenerate widths 0 and 1 for every formerly gated symbol")
struct D02DegenerateWidthTests {
    @Test("The table has a row per gated symbol at both widths (61)")
    func tableCoversEverySymbol() {
        #expect(d02EdgeExpected.count == 61)
        #expect(Set(d02EdgeRows(D02EdgeV0.self).keys) == Set(d02EdgeExpected.keys))
        #expect(Set(d02EdgeRows(D02EdgeV1.self).keys) == Set(d02EdgeExpected.keys))
    }

    @Test("Non-trapping outcomes at widths 0 and 1 match the table exactly")
    func nonTrappingOutcomes() {
        let v0 = d02EdgeRows(D02EdgeV0.self), v1 = d02EdgeRows(D02EdgeV1.self)
        for (path, expected) in d02EdgeExpected.sorted(by: { $0.key < $1.key }) {
            for (width, want, rows) in [(0, expected.v0, v0), (1, expected.v1, v1)] {
                if case .trap = want { continue }
                guard let row = rows[path] else { Issue.record("no row \(path)"); continue }
                #expect(d02Evaluate(row) == want, "\(path) at width \(width)")
            }
        }
    }

    @Test func edgeChildProbe() {
        guard let specification = ProcessInfo.processInfo.environment["D02_EDGE_PROBE"] else { return }
        let parts = specification.split(separator: "|").map(String.init)
        let rows = parts.last == "0" ? d02EdgeRows(D02EdgeV0.self) : d02EdgeRows(D02EdgeV1.self)
        if let row = rows[parts[0]] { _ = d02Evaluate(row) }
        print("D02_EDGE_NOT_TAKEN \(specification)")
    }

    @Test("Trapping outcomes trap in a child (message checked in Debug builds)")
    func trappingOutcomes() throws {
        if ProcessInfo.processInfo.environment["D02_SKIP_TRAPS"] == "1" { return }
        var traps: [(String, String, String)] = []
        for (path, expected) in d02EdgeExpected.sorted(by: { $0.key < $1.key }) {
            if case .trap(let message) = expected.v0 { traps.append((path, "0", message)) }
            if case .trap(let message) = expected.v1 { traps.append((path, "1", message)) }
        }
        #expect(traps.count == 6)
        for (path, width, message) in traps {
            let specification = "\(path)|\(width)"
            var environment = ProcessInfo.processInfo.environment
            environment["D02_EDGE_PROBE"] = specification
            environment["SWIFT_BACKTRACE"] = "enable=no"
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("D02-edge-\(UUID().uuidString).log")
            _ = FileManager.default.createFile(atPath: url.path, contents: nil)
            let handle = try FileHandle(forWritingTo: url)
            let process = try launchChildProbe(.swiftTesting(filter: "D02DegenerateWidthTests/edgeChildProbe"),
                                               environment: environment, output: handle, deadline: 15,
                                               cleanup: .terminateThenKill(grace: 1, seconds: 1))
            let timedOut = process.timedOut
            try handle.close()
            let diagnostic = try String(contentsOf: url, encoding: .utf8)
            print("D02_EDGE_CHILD_CAPTURE \(specification) \(url.path)")
            if timedOut {
                throw NSError(domain: "D02.childTimeout", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "Child deadline exceeded: \(url.path)"])
            }
            #expect(process.terminationReason == .uncaughtSignal, "\(specification): \(diagnostic)")
            #expect(!diagnostic.contains("D02_EDGE_NOT_TAKEN"), "\(specification)")
            #if DEBUG
            #expect(diagnostic.contains(message), "\(specification): expected '\(message)' in \(diagnostic)")
            #endif
        }
    }
}
