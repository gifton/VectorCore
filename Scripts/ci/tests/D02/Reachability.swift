// D02 R1: an ordinary Swift 6 consumer (`import VectorCore`, no @testable) whose custom dimensions declare ONLY
// `value`. Every member that was gated on `D.Storage: VectorStorageOperations` before D02 is called here through
// a function generic over any `D: StaticDimension`; `main` adds concrete calls for six of them (linspace, l1Norm,
// lInfinityNorm, absoluteValue, squareRoot, range). DimMisleading declares an unrelated `Storage` alias that must not
// constrain anything. Run by Scripts/ci/tests/D02RunConsumers.py, which checks that every name of
// Scripts/ci/tests/D02/gated-public-base.tsv is called below in code (count guard, comments and strings masked) and
// that each call binds to Vector's own member (SILGen function_ref binding check).
// Before D02 this file does not compile (`does not conform to protocol 'StaticDimension'`).
import VectorCore

enum V0: StaticDimension { static let value = 0 }
enum V1: StaticDimension { static let value = 1 }
enum V7: StaticDimension { static let value = 7 }        // hybrid tier, not a multiple of 4
enum V16: StaticDimension { static let value = 16 }      // hybrid tier boundary
enum V17: StaticDimension { static let value = 17 }      // managed tier, not a multiple of 4
enum V100: StaticDimension { static let value = 100 }    // managed tier
enum DimMisleading: StaticDimension {                    // control: an unrelated alias constrains nothing
    static let value = 5
    typealias Storage = String
}

func viaConformance<V: VectorCoreOperations>(_ v: V) -> V { v.absoluteValue().squareRoot() }

func check(_ condition: Bool, _ message: String) {
    if !condition { fatalError("D02 reachability: \(message)") }
}

/// Calls every formerly gated member for one dimension. Six members need an element and stop with a precondition
/// failure at width 0: oneHot(at:), basis(axis:), unitX, e(_:), minElement(), maxElement(). They are called only
/// when `D.value > 0`; their width-0 trap, and every other width-0/width-1 outcome, is pinned row by row by
/// D02DegenerateWidthTests (child-isolated traps), which covers all 61 gated symbols.
@discardableResult
func exercise<D: StaticDimension>(_: D.Type) throws -> Int {
    let n = D.value
    let a = Vector<D>.range(from: 1)
    let b = Vector<D>.generate({ Float($0) - 2 })
    let c = Vector<D>(indexMap: { Float($0 % 3) })
    _ = Vector<D>.geometric(initial: 1, ratio: 0.5)
    _ = Vector<D>.powers(of: 2)
    _ = Vector<D>.alternating(magnitude: 2)
    _ = Vector<D>.repeatingPattern([1, 2])
    _ = try Vector<D>.repeatingPatternThrowing([1, 2])
    _ = Vector<D>.repeatingPattern(safe: [1, 2])
    _ = Vector<D>.sparse(value: 3, at: n > 0 ? [0] : [])
    _ = Vector<D>.unitY
    _ = Vector<D>.unitZ
    _ = Vector<D>.basis(safe: n)                     // out of range: nil
    _ = Vector<D>.linspace(from: 0, to: 1)
    _ = Vector<D>.range()
    if n > 0 {                                        // width 0: these four trap (D02DegenerateWidthTests)
        _ = Vector<D>.oneHot(at: n - 1)
        _ = Vector<D>.basis(axis: 0)
        _ = Vector<D>.unitX
        _ = Vector<D>.e(0)
    }
    // The throwing forms run at every width: index 0 exists for n > 0 and throws indexOutOfBounds at n == 0.
    let throwingForms: [() throws -> Vector<D>] = [
        { try Vector<D>.oneHotThrowing(at: 0) }, { try Vector<D>.basisThrowing(axis: 0) }, { try Vector<D>.eThrowing(0) }
    ]
    for form in throwingForms {
        do {
            _ = try form()
            check(n > 0, "a throwing form returned at width 0")
        } catch {
            check(n == 0, "a throwing form threw at width \(n)")
        }
    }
    check(a.count == n && b.count == n && c.count == n, "shape")
    // NaN/infinity handling.
    _ = a.isFinite
    _ = a.hasNaN
    _ = a.hasInfinity
    _ = a.checkNonFinite()
    _ = try a.handleNonFinite(options: .replaceAll)
    _ = a.replacingNonFinite(with: 0)
    _ = a.finiteValues()
    _ = try a.safeDivide(by: 2, options: .replaceAll)
    _ = try a.safeNormalized(options: .replaceAll)
    _ = try a.safeLog(options: .replaceAll)
    // VectorCoreOperations.
    var m = a.min(b).max(c)
    if n > 0 {                                        // width 0: both trap (D02DegenerateWidthTests)
        _ = m.minElement()
        _ = m.maxElement()
    }
    m = m.clamped(to: -1...1)
    m.clamp(to: 0...1)
    _ = a.lerp(to: b, t: 0.25)
    _ = a.lerpUnclamped(to: b, t: 1.5)
    _ = a.smoothstep(to: b, t: 0.5)
    _ = a.absoluteValue()
    _ = a.squareRoot()
    _ = a .< b
    _ = a .> b
    _ = viaConformance(b)
    // VectorMath.
    _ = a.l1Norm + a.l2Norm + a.lInfinityNorm
    _ = a.manhattanDistance(to: b) + a.chebyshevDistance(to: b)
    _ = VectorMath.nearestNeighbors(query: a, in: [a, b, c], k: 2, using: { $0.manhattanDistance(to: $1) })
    _ = VectorMath.pairwiseDistances([a, b], using: { $0.chebyshevDistance(to: $1) })
    _ = a.sparsity(threshold: 0)
    _ = a.entropy
    _ = a.entropyFast                                // deprecated spelling of `entropy` (D02 RD-R4)
    let encoded = a.base64Encoded
    let decoded = try Vector<D>.base64Decoded(from: encoded)
    check(decoded.toArray() == a.toArray(), "base64 round trip")
    // Normalization.
    var unit = a.normalizedFast()
    unit.normalizeFast()
    _ = unit.isNormalized
    _ = a.normalized(tolerance: 1e-6)
    return n
}

@main
struct D02Reachability {
    static func main() throws {
        let widths = [try exercise(V0.self), try exercise(V1.self), try exercise(V7.self), try exercise(V16.self),
                      try exercise(V17.self), try exercise(V100.self), try exercise(DimMisleading.self)]
        check(widths == [0, 1, 7, 16, 17, 100, 5], "widths \(widths)")
        // Concrete (non-generic) calls bind to the same members.
        let v = Vector<V7>.linspace(from: -3, to: 3)       // -3 ... 3 in exact unit steps
        check(v.toArray() == [-3, -2, -1, 0, 1, 2, 3], "concrete linspace")
        check(v.l1Norm == 12 && v.lInfinityNorm == 3, "concrete norms (integer sums, exact in any order)")
        check(viaConformance(v).toArray() == v.absoluteValue().squareRoot().toArray(), "conformance route")
        let misleading = Vector<DimMisleading>.range()
        check(misleading.toArray() == [0, 1, 2, 3, 4], "misleading-alias dimension")
        print("D02_REACHABILITY_COMPLETE widths=\(widths)")
    }
}
