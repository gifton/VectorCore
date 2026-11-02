import Foundation
import os
import Testing
@testable import VectorCore

// D02 phase 3: after the StaticDimension.Storage requirement and the D.Storage gates were removed, every Vector<D>
// member is reachable for a custom dimension that declares only `value`. The external, non-@testable consumer
// (Scripts/ci/tests/D02/Reachability.swift, run by D02RunConsumers.py) proves compile-time reachability of every
// formerly gated symbol; these tests pin behaviour:
//   R2 a custom dimension with a built-in width gives bit-identical results to the built-in dimension;
//   R3 non-built-in widths against independent references (shape, norms, policies, entropy, base64);
//   R4 value-copy isolation on both storage tiers;
//   R5 ordinary provider dispatch: the formerly gated simdProvider readers call the installed provider;
//   R6 the defined invalid cases (traps child-isolated, throwing and optional forms).

private enum D02C8: StaticDimension { static let value = 8 }
private enum D02C512: StaticDimension { static let value = 512 }
private enum D02V0: StaticDimension { static let value = 0 }
private enum D02V1: StaticDimension { static let value = 1 }
private enum D02V5: StaticDimension { static let value = 5 }
private enum D02V7: StaticDimension { static let value = 7 }
private enum D02V17: StaticDimension { static let value = 17 }
private enum D02V100: StaticDimension { static let value = 100 }

/// Fixture values with the exceptional cases the brief names: finite, signed zero, NaN, +/-infinity, large magnitude.
private func d02Fixture(_ n: Int, exceptional: Bool) -> [Float] {
    var state: UInt32 = 0x1234_5678 &+ UInt32(n)
    var values = (0..<n).map { _ -> Float in
        state = state &* 1_664_525 &+ 1_013_904_223
        return Float(Int32(bitPattern: state) >> 8) / Float(1 << 20)
    }
    if exceptional && n >= 6 {
        values[0] = -0.0
        values[1] = .nan
        values[2] = .infinity
        values[3] = -.infinity
        values[4] = Float.greatestFiniteMagnitude / 4
        values[5] = 0
    }
    return values
}

private func bits(_ values: [Float]) -> [UInt32] { values.map(\.bitPattern) }

/// Every formerly gated member (and the Vector members they build on) evaluated for one dimension, as bit patterns
/// and error descriptions, so two dimensions of equal width can be compared exactly.
private func d02Signature<D: StaticDimension>(_: D.Type, _ values: [Float], _ other: [Float]) -> [String] {
    var out: [String] = []
    func rec(_ name: String, _ v: [Float]) { out.append("\(name)=\(bits(v))") }
    func rec(_ name: String, _ v: Float) { out.append("\(name)=\(v.bitPattern)") }
    func rec<T>(_ name: String, _ body: () throws -> T, _ show: (T) -> String) {
        // Errors are compared by type and kind (VectorError context can carry per-call data such as a timestamp).
        do { out.append("\(name)=\(show(try body()))") } catch let error as VectorError {
            out.append("\(name)!VectorError.\(error.kind)")
        } catch let error as NonFiniteError {
            out.append("\(name)!\(error)")   // cases carry only index lists
        } catch { out.append("\(name)!\(type(of: error))") }
    }
    let a = try! Vector<D>(values)
    let b = try! Vector<D>(other)
    let n = D.value
    rec("range", Vector<D>.range().toArray()); rec("rangeFrom", Vector<D>.range(from: 1.5).toArray())
    rec("linspace", Vector<D>.linspace(from: -1, to: 3).toArray())
    rec("generate", Vector<D>.generate { Float($0) * 0.25 }.toArray())
    rec("indexMap", Vector<D>(indexMap: { Float($0 % 5) - 2 }).toArray())
    rec("geometric", Vector<D>.geometric(initial: 3, ratio: 0.5).toArray())
    rec("powers", Vector<D>.powers(of: 1.5).toArray())
    rec("alternating", Vector<D>.alternating(magnitude: 2).toArray())
    rec("pattern", Vector<D>.repeatingPattern([1, -2, 3]).toArray())
    rec("sparse", Vector<D>.sparse(value: 4, at: [0, n - 1]).toArray())
    rec("basis", Vector<D>.basis(axis: n - 1).toArray()); rec("e", Vector<D>.e(1).toArray())
    rec("oneHot", Vector<D>.oneHot(at: 2).toArray())
    rec("unit", (Vector<D>.unitX.toArray() + Vector<D>.unitY.toArray() + Vector<D>.unitZ.toArray()))
    rec("isFinite", a.isFinite ? 1 : 0); rec("hasNaN", a.hasNaN ? 1 : 0); rec("hasInfinity", a.hasInfinity ? 1 : 0)
    out.append("check=\(a.checkNonFinite())")
    for (label, option) in [("replaceAll", NonFiniteHandling.replaceAll), ("propagate", .propagateNaN),
                            ("strict", .throwOnNonFinite), ("nanZero", .replaceNaNWithZero)] {
        rec("handle.\(label)", { try a.handleNonFinite(options: option).toArray() }, { "\(bits($0))" })
        rec("divide.\(label)", { try a.safeDivide(by: 3, options: option).toArray() }, { "\(bits($0))" })
        rec("normalize.\(label)", { try a.safeNormalized(options: option).toArray() }, { "\(bits($0))" })
        rec("log.\(label)", { try a.safeLog(options: option).toArray() }, { "\(bits($0))" })
    }
    rec("replacing", a.replacingNonFinite(with: 7).toArray())
    out.append("finite=\(bits(a.finiteValues().values))\(a.finiteValues().indices)")
    rec("min", a.min(b).toArray()); rec("max", a.max(b).toArray())
    out.append("minElement=\(a.minElement().value.bitPattern),\(a.minElement().index)")
    out.append("maxElement=\(a.maxElement().value.bitPattern),\(a.maxElement().index)")
    rec("clamped", a.clamped(to: -2...2).toArray())
    var c = a; c.clamp(to: -1...1); rec("clamp", c.toArray())
    rec("lerp", a.lerp(to: b, t: 0.3).toArray()); rec("lerpU", a.lerpUnclamped(to: b, t: 1.7).toArray())
    rec("smooth", a.smoothstep(to: b, t: 0.4).toArray())
    rec("abs", a.absoluteValue().toArray()); rec("sqrt", a.squareRoot().toArray())
    rec("lt", (a .< b).toArray()); rec("gt", (a .> b).toArray())
    rec("l1", a.l1Norm); rec("l2", a.l2Norm); rec("linf", a.lInfinityNorm)
    rec("manhattan", a.manhattanDistance(to: b)); rec("chebyshev", a.chebyshevDistance(to: b))
    let neighbors = VectorMath.nearestNeighbors(query: a, in: [a, b], k: 2) { $0.manhattanDistance(to: $1) }
    out.append("nn=\(neighbors.map { "\($0.index):\($0.distance.bitPattern)" })")
    out.append("pairwise=\(VectorMath.pairwiseDistances([a, b]) { $0.chebyshevDistance(to: $1) }.map(bits))")
    rec("sparsity", a.sparsity(threshold: 0.01)); rec("entropy", a.entropy)
    out.append("base64=\(a.base64Encoded)")
    rec("normalizedFast", a.normalizedFast().toArray())
    var d = a; d.normalizeFast(); rec("normalizeFast", d.toArray())
    rec("isNormalized", a.isNormalized ? 1 : 0); rec("normalizedTol", a.normalized(tolerance: 1e-6).toArray())
    return out
}

@Suite("D02 custom-dimension reachability")
struct D02DimensionReachabilityTests {
    @Test("R2: a custom dimension of a built-in width is bit-identical to the built-in dimension")
    func builtInWidthEquivalence() {
        for exceptional in [false, true] {
            let a8 = d02Fixture(8, exceptional: exceptional), b8 = d02Fixture(8, exceptional: false).reversed()
            #expect(d02Signature(D02C8.self, a8, Array(b8)) == d02Signature(Dim8.self, a8, Array(b8)))
            let a512 = d02Fixture(512, exceptional: exceptional), b512 = Array(d02Fixture(512, exceptional: false).reversed())
            #expect(d02Signature(D02C512.self, a512, b512) == d02Signature(Dim512.self, a512, b512))
        }
    }

    @Test("R3: non-built-in widths against independent references")
    func nonBuiltInWidths() throws {
        try check(D02V5.self); try check(D02V7.self); try check(D02V17.self); try check(D02V100.self)
    }

    private func check<D: StaticDimension>(_: D.Type) throws {
        let n = D.value
        let values = d02Fixture(n, exceptional: false)
        let v = try Vector<D>(values)
        #expect(v.count == n && v.scalarCount == n && Vector<D>.range().count == n)
        // Norms: Double references; gamma_n(2^-24) bounds any Float summation order of n nonnegative terms
        // (n + 1 for the square and the square root of l2).
        let u = 0x1p-24, gamma = { (k: Double) in k * u / (1 - k * u) }
        let l1 = values.reduce(0.0) { $0 + Swift.abs(Double($1)) }
        let l2 = values.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot()
        let linf = values.map { Swift.abs(Double($0)) }.max() ?? 0
        #expect(Swift.abs(Double(v.l1Norm) - l1) <= gamma(Double(n)) * l1)
        #expect(Swift.abs(Double(v.l2Norm) - l2) <= gamma(Double(n + 2)) * l2)
        #expect(Double(v.lInfinityNorm) == linf)
        // safeDivide by 4 is exact for these normal-range fixtures.
        #expect(bits(try v.safeDivide(by: 4).toArray()) == bits(values.map { $0 / 4 }))
        // Policies on a vector with NaN and infinities: N02 replacement and propagation, strict throws.
        var exceptional = values
        exceptional[0] = .nan; exceptional[1] = .infinity
        let e = try Vector<D>(exceptional)
        let replaced = try e.handleNonFinite(options: .replaceAll).toArray()
        #expect(replaced[0] == 0 && replaced[1] == Float.greatestFiniteMagnitude && Array(replaced[2...]) == Array(values[2...]))
        #expect(throws: (any Error).self) { try e.handleNonFinite(options: .throwOnNonFinite) }
        #expect(try e.safeNormalized(options: .propagateNaN).toArray().contains { $0.isNaN })
        #expect(try e.safeLog(options: .replaceNaNWithZero).toArray()[0] == 0)
        // Sparsity counts |x| <= threshold exactly.
        #expect(v.sparsity(threshold: 0.5) == Float(values.filter { Swift.abs($0) <= 0.5 }.count) / Float(n))
        // Entropy: the N04 independent reference and tolerance (EntropyRangeContractTests).
        let reference = entropyContractReference(values)
        #expect(Swift.abs(v.entropy - reference) <= 8 * Float.ulpOfOne * Swift.max(1, Swift.abs(reference)))
        #expect(v.entropy.bitPattern == v.entropySmall().bitPattern)   // the route `entropy` takes for these widths
        // Base64 round trip is bit-exact.
        #expect(bits(try Vector<D>.base64Decoded(from: v.base64Encoded).toArray()) == bits(values))
    }

    @Test("R4: mutating members leave the source copy untouched on both storage tiers")
    func valueCopyIsolation() throws {
        try isolate(D02V7.self); try isolate(D02V17.self); try isolate(D02V100.self)
    }

    private func isolate<D: StaticDimension>(_: D.Type) throws {
        let values = d02Fixture(D.value, exceptional: false)
        let original = try Vector<D>(values)
        var clampedCopy = original
        clampedCopy.clamp(to: -0.25...0.25)
        var normalizedCopy = original
        normalizedCopy.normalizeFast()
        #expect(bits(original.toArray()) == bits(values))
        #expect(bits(clampedCopy.toArray()) != bits(values))
        #expect(bits(normalizedCopy.toArray()) != bits(values))
    }
}

/// Records every provider call; with `distinctive`, array-returning and index-returning calls return values that
/// cannot come from the default provider, so a result shows which provider produced it.
private final class D02RecordingSIMDProvider: ArraySIMDProvider, @unchecked Sendable {
    let calls = OSAllocatedUnfairLock(initialState: [String]())
    let base = DefaultArraySIMDProvider()
    func note(_ name: String) { calls.withLock { $0.append(name) } }
    func add(_ a: [Float], _ b: [Float]) -> [Float] { note("add"); return base.add(a, b) }
    func subtract(_ a: [Float], _ b: [Float]) -> [Float] { note("subtract"); return base.subtract(a, b) }
    func multiply(_ a: [Float], by s: Float) -> [Float] { note("multiply"); return base.multiply(a, by: s) }
    func divide(_ a: [Float], by s: Float) -> [Float] { note("divide"); return base.divide(a, by: s) }
    func dot(_ a: [Float], _ b: [Float]) -> Float { note("dot"); return base.dot(a, b) }
    func sum(_ a: [Float]) -> Float { note("sum"); return base.sum(a) }
    func max(_ a: [Float]) -> Float { note("max"); return base.max(a) }
    func min(_ a: [Float]) -> Float { note("min"); return base.min(a) }
    func mean(_ a: [Float]) -> Float { note("mean"); return base.mean(a) }
    func magnitude(_ a: [Float]) -> Float { note("magnitude"); return base.magnitude(a) }
    func normalize(_ a: [Float]) -> [Float] { note("normalize"); return base.normalize(a) }
    func magnitudeSquared(_ a: [Float]) -> Float { note("magnitudeSquared"); return base.magnitudeSquared(a) }
    func euclideanDistanceSquared(_ a: [Float], _ b: [Float]) -> Float {
        note("euclideanDistanceSquared"); return base.euclideanDistanceSquared(a, b)
    }
    func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float { note("cosineSimilarity"); return base.cosineSimilarity(a, b) }
    func elementWiseMin(_ a: [Float], _ b: [Float]) -> [Float] { note("elementWiseMin"); return Array(repeating: 117, count: a.count) }
    func elementWiseMax(_ a: [Float], _ b: [Float]) -> [Float] { note("elementWiseMax"); return Array(repeating: 219, count: a.count) }
    func elementWiseMultiply(_ a: [Float], _ b: [Float]) -> [Float] { note("elementWiseMultiply"); return base.elementWiseMultiply(a, b) }
    func elementWiseDivide(_ a: [Float], _ b: [Float]) -> [Float] { note("elementWiseDivide"); return base.elementWiseDivide(a, b) }
    func abs(_ a: [Float]) -> [Float] { note("abs"); return Array(repeating: 7, count: a.count) }
    func sqrt(_ a: [Float]) -> [Float] { note("sqrt"); return Array(repeating: 9, count: a.count) }
    func log(_ a: [Float]) -> [Float] { note("log"); return base.log(a) }
    func clip(_ a: [Float], min: Float, max: Float) -> [Float] { note("clip"); return Array(repeating: 321, count: a.count) }
    func minIndex(_ a: [Float]) -> Int { note("minIndex"); return a.count - 1 }
    func maxIndex(_ a: [Float]) -> Int { note("maxIndex"); return a.count - 2 }
}

@Suite("D02 custom-dimension provider dispatch")
struct D02DimensionProviderDispatchTests {
    @Test("R5: formerly gated simdProvider readers call the installed provider for custom dimensions")
    func installedProviderIsCalled() throws {
        try dispatch(D02V7.self)      // hybrid tier; l1Norm/manhattanDistance stay scalar below 33 lanes
        try dispatch(D02V100.self)    // managed tier; l1Norm/manhattanDistance call the provider
    }

    private func dispatch<D: StaticDimension>(_: D.Type) throws {
        let n = D.value
        let a = try Vector<D>(d02Fixture(n, exceptional: false))
        let b = try Vector<D>(d02Fixture(n, exceptional: false).reversed())
        let spy = D02RecordingSIMDProvider()
        // Results are produced and read inside the scope (no value escapes before it is checked).
        let observed: [String: [Float]] = Operations.$simdProvider.withValue(spy) {
            var c = a
            c.clamp(to: 0...1)
            let minElement = a.minElement(), maxElement = a.maxElement()
            return ["min": a.min(b).toArray(), "max": a.max(b).toArray(), "clamped": a.clamped(to: 0...1).toArray(),
                    "clamp": c.toArray(), "abs": a.absoluteValue().toArray(), "sqrt": a.squareRoot().toArray(),
                    "minElement": [minElement.value, Float(minElement.index)],
                    "maxElement": [maxElement.value, Float(maxElement.index)],
                    "linf": [a.lInfinityNorm], "l1": [a.l1Norm], "manhattan": [a.manhattanDistance(to: b)]]
        }
        let calls = spy.calls.withLock { $0 }
        #expect(observed["min"] == Array(repeating: 117, count: n))
        #expect(observed["max"] == Array(repeating: 219, count: n))
        #expect(observed["clamped"] == Array(repeating: 321, count: n))
        #expect(observed["clamp"] == Array(repeating: 321, count: n))
        #expect(observed["abs"] == Array(repeating: 7, count: n))
        #expect(observed["sqrt"] == Array(repeating: 9, count: n))
        #expect(observed["minElement"] == [a[n - 1], Float(n - 1)])
        #expect(observed["maxElement"] == [a[n - 2], Float(n - 2)])
        #expect(observed["linf"] == [7])
        for name in ["elementWiseMin", "elementWiseMax", "clip", "abs", "sqrt", "minIndex", "maxIndex"] {
            #expect(calls.contains(name), "\(name) not called for width \(n)")
        }
        if n > 32 {
            #expect(observed["l1"] == [Float(7 * n)])
            #expect(observed["manhattan"] == [Float(7 * n)])
            #expect(calls.contains("subtract"))
        } else {
            // Scalar loop below 33 lanes (VectorMath.l1Norm / manhattanDistance): the spy's abs is not used.
            #expect(observed["l1"] != [Float(7 * n)])
        }
    }
}

@Suite("D02 custom-dimension defined invalid cases")
struct D02DimensionInvalidCaseTests {
    @Test("R6: out-of-range throwing and optional forms on a value-only dimension")
    func throwingAndOptionalForms() {
        #expect(Vector<D02V0>.basis(safe: 0) == nil)
        #expect(Vector<D02V7>.basis(safe: 7) == nil)
        #expect(Vector<D02V7>.repeatingPattern(safe: []) == nil)
        #expect(throws: VectorError.self) { try Vector<D02V0>.basisThrowing(axis: 0) }
        #expect(throws: VectorError.self) { try Vector<D02V7>.eThrowing(7) }
        #expect(throws: VectorError.self) { try Vector<D02V7>.oneHotThrowing(at: -1) }
        #expect(throws: VectorError.self) { try Vector<D02V7>.repeatingPatternThrowing([]) }
        #expect(Vector<D02V0>.unitY.count == 0 && Vector<D02V0>.unitZ.count == 0)
        #expect(Vector<D02V1>.unitY.toArray() == [0] && Vector<D02V1>.unitZ.toArray() == [0])
    }

    @Test func trapChildProbe() {
        guard let specification = ProcessInfo.processInfo.environment["D02_TRAP_PROBE"] else { return }
        switch specification {
        case "unitX-V0": _ = Vector<D02V0>.unitX
        case "basis-out-of-range": _ = Vector<D02V7>.basis(axis: 7)
        case "e-negative": _ = Vector<D02V7>.e(-1)
        case "oneHot-out-of-range": _ = Vector<D02V17>.oneHot(at: 17)
        case "pattern-empty": _ = Vector<D02V7>.repeatingPattern([])
        default: break
        }
        print("D02_TRAP_NOT_TAKEN \(specification)")
    }

    @Test("R6: the trapping forms trap (child-isolated)")
    func trappingFormsTrap() throws {
        if ProcessInfo.processInfo.environment["D02_SKIP_TRAPS"] == "1" { return }
        for specification in ["unitX-V0", "basis-out-of-range", "e-negative", "oneHot-out-of-range", "pattern-empty"] {
            var environment = ProcessInfo.processInfo.environment
            environment["D02_TRAP_PROBE"] = specification
            environment["SWIFT_BACKTRACE"] = "enable=no"
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("D02-trap-\(specification)-\(UUID().uuidString).log")
            _ = FileManager.default.createFile(atPath: url.path, contents: nil)
            let handle = try FileHandle(forWritingTo: url)
            let process = try launchChildProbe(.swiftTesting(filter: "D02DimensionInvalidCaseTests/trapChildProbe"),
                                               environment: environment, output: handle, deadline: 15,
                                               cleanup: .terminateThenKill(grace: 1, seconds: 1))
            let timedOut = process.timedOut
            try handle.close()
            let diagnostic = try String(contentsOf: url, encoding: .utf8)
            print("D02_CHILD_CAPTURE \(specification) \(url.path)")
            if timedOut {
                throw NSError(domain: "D02.childTimeout", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "Child deadline exceeded: \(url.path)"])
            }
            #expect(process.terminationReason == .uncaughtSignal, "\(specification): \(diagnostic)")
            #expect(!diagnostic.contains("D02_TRAP_NOT_TAKEN"), "\(specification)")
        }
    }
}
