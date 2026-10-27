import Foundation

// T16 exceptional-table fixtures, oracle and cell checker (Foundation only: the T16 phase 2 RED-prediction probe
// compiles this same file against the release library, so the predicted RED cells and the table use one fixture
// family and one checker).
//
// Contract checked (T16 phase 1 rulings (a), C2):
// - any NaN input element gives a NaN result (`isNaN` only; no payload is pinned);
// - index reductions return the index of the FIRST NaN;
// - element-wise min/max and clip keep a NaN lane NaN;
// - NaN-free results equal the scalar oracle under `==`, so the sign of a zero max/min is unspecified (S1);
// - a zero `maximumMagnitude` is +0 (Z1);
// - an empty pointer-level reduction returns -inf / +inf / 0 (E1); `SIMDStorage.minMax` starts from +-inf (I1).

/// One fixture: `values` plus the index of the first NaN (nil when NaN-free).
struct T16Fixture: Sendable {
    let id: String
    let values: [Float]
    var firstNaN: Int? { values.firstIndex(where: { $0.isNaN }) }
}

enum T16Fixtures {
    /// Every Swift SIMD8/SIMD4 chunk and scalar-tail position, and the vDSP vector-body/tail split (phase 1 (c)).
    static let sizes = [1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 16, 17, 31, 32, 33, 64, 65, 257]
    static let payloadNaN = Float(bitPattern: 0x7FC0_0123)
    static let negativeNaN = Float(bitPattern: 0xFFC0_0000)
    static let signallingNaN = Float(bitPattern: 0x7FA0_0000)

    /// `(+-)(i + 1) * 0.5`, alternating sign: distinct finite values whose max and min move with the length.
    static func base(_ n: Int) -> [Float] { (0..<n).map { Float($0 % 2 == 0 ? 1 : -1) * Float($0 + 1) * 0.5 } }

    /// The element-wise partner operand: -1, 0, 1 repeating (finite; zeros meet the signed-zero fixtures).
    static func partner(_ n: Int) -> [Float] { (0..<n).map { Float($0 % 3) - 1 } }

    static func all() -> [T16Fixture] {
        var out: [T16Fixture] = [T16Fixture(id: "n0/empty", values: [])]
        func edit(_ n: Int, _ edits: [(Int, Float)]) -> [Float] {
            var v = base(n)
            for (i, x) in edits { v[i] = x }
            return v
        }
        for n in sizes {
            let last = n - 1
            out.append(T16Fixture(id: "n\(n)/finite", values: base(n)))
            for p in 0..<n { out.append(T16Fixture(id: "n\(n)/nan@\(p)", values: edit(n, [(p, .nan)]))) }
            out.append(T16Fixture(id: "n\(n)/payloadnan@last", values: edit(n, [(last, payloadNaN)])))
            out.append(T16Fixture(id: "n\(n)/negnan@last", values: edit(n, [(last, negativeNaN)])))
            out.append(T16Fixture(id: "n\(n)/snan@last", values: edit(n, [(last, signallingNaN)])))
            out.append(T16Fixture(id: "n\(n)/allnan", values: Array(repeating: .nan, count: n)))
            if n >= 2 {
                out.append(T16Fixture(id: "n\(n)/nan@0,+inf@last", values: edit(n, [(0, .nan), (last, .infinity)])))
                out.append(T16Fixture(id: "n\(n)/nan@0,-inf@last", values: edit(n, [(0, .nan), (last, -.infinity)])))
                out.append(T16Fixture(id: "n\(n)/+inf@0,nan@last", values: edit(n, [(0, .infinity), (last, .nan)])))
                out.append(T16Fixture(id: "n\(n)/-inf@0,nan@last", values: edit(n, [(0, -.infinity), (last, .nan)])))
                out.append(T16Fixture(id: "n\(n)/+inf@mid", values: edit(n, [(n / 2, .infinity)])))
                out.append(T16Fixture(id: "n\(n)/-inf@mid", values: edit(n, [(n / 2, -.infinity)])))
            }
            out.append(T16Fixture(id: "n\(n)/all+inf", values: Array(repeating: .infinity, count: n)))
            out.append(T16Fixture(id: "n\(n)/all-inf", values: Array(repeating: -.infinity, count: n)))
            out.append(T16Fixture(id: "n\(n)/zeros+first", values: (0..<n).map { $0 % 2 == 0 ? Float(0) : -Float(0) }))
            out.append(T16Fixture(id: "n\(n)/zeros-first", values: (0..<n).map { $0 % 2 == 0 ? -Float(0) : Float(0) }))
            out.append(T16Fixture(id: "n\(n)/all-0", values: Array(repeating: -Float(0), count: n)))
        }
        return out
    }
}

/// Expected outcome of one cell.
indirect enum T16Expect: Sendable, CustomStringConvertible {
    case nan                              // isNaN
    case equal(Double)                    // == (signed zeros compare equal: S1)
    case positiveZeroOrEqual(Double)      // == and, when the value is zero, the sign is plus (Z1)
    case absent                           // nil (VectorProtocol.min/max of an empty vector)
    case index(Int)                       // an index reduction
    case indexAndValue(Int, T16Expect)    // `minElement()`/`maxElement()`
    case lanes([T16Expect])               // element-wise output, lane by lane

    var description: String {
        switch self {
        case .nan: return "NaN"
        case .equal(let x): return "==\(x)"
        case .positiveZeroOrEqual(let x): return x == 0 ? "+0" : "==\(x)"
        case .absent: return "nil"
        case .index(let i): return "index \(i)"
        case .indexAndValue(let i, let v): return "index \(i) value \(v)"
        case .lanes(let l): return "\(l.count) lanes"
        }
    }
}

/// Observed outcome of one cell (Float results are widened to Double exactly).
enum T16Got: Sendable, CustomStringConvertible {
    case value(Double)
    case absent
    case index(Int)
    case indexAndValue(Int, Double)
    case lanes([Double])

    var description: String {
        func show(_ x: Double) -> String { x.isNaN ? "NaN" : (x == 0 ? (x.sign == .minus ? "-0" : "+0") : "\(x)") }
        switch self {
        case .value(let x): return show(x)
        case .absent: return "nil"
        case .index(let i): return "index \(i)"
        case .indexAndValue(let i, let v): return "index \(i) value \(show(v))"
        case .lanes(let l): return "[" + l.prefix(6).map(show).joined(separator: ",") + (l.count > 6 ? ",..." : "") + "]"
        }
    }
}

enum T16Check {
    static func admits(_ want: T16Expect, _ got: Double) -> Bool {
        switch want {
        case .nan: return got.isNaN
        case .equal(let x): return !got.isNaN && got == x
        case .positiveZeroOrEqual(let x): return !got.isNaN && got == x && (got != 0 || got.sign == .plus)
        default: return false
        }
    }

    /// True when `got` satisfies `want`.
    static func admits(_ want: T16Expect, _ got: T16Got) -> Bool {
        switch (want, got) {
        case (.absent, .absent): return true
        case (.index(let i), .index(let j)): return i == j
        case (.indexAndValue(let i, let v), .indexAndValue(let j, let x)): return i == j && admits(v, x)
        case (.lanes(let wants), .lanes(let gots)):
            return wants.count == gots.count && zip(wants, gots).allSatisfy { admits($0, $1) }
        case (_, .value(let x)): return admits(want, x)
        default: return false
        }
    }

    /// Cross-provider agreement under the same equivalence: NaN with NaN; otherwise `==` (S1), and a zero
    /// magnitude must have the same sign on both sides when `signedZero` is set (Z1); indices exactly.
    static func agree(_ a: T16Got, _ b: T16Got, signedZero: Bool = false) -> Bool {
        func same(_ x: Double, _ y: Double) -> Bool {
            if x.isNaN || y.isNaN { return x.isNaN && y.isNaN }
            return x == y && (!signedZero || x != 0 || x.sign == y.sign)
        }
        switch (a, b) {
        case (.value(let x), .value(let y)): return same(x, y)
        case (.absent, .absent): return true
        case (.index(let i), .index(let j)): return i == j
        case (.indexAndValue(let i, let x), .indexAndValue(let j, let y)): return i == j && same(x, y)
        case (.lanes(let x), .lanes(let y)): return x.count == y.count && zip(x, y).allSatisfy { same($0, $1) }
        default: return false
        }
    }
}

/// Scalar oracle for NaN-free input, and the expectation builders used by the rows.
enum T16Oracle {
    static func maximum(_ v: [Float]) -> Float { v.dropFirst().reduce(v[0]) { $1 > $0 ? $1 : $0 } }
    static func minimum(_ v: [Float]) -> Float { v.dropFirst().reduce(v[0]) { $1 < $0 ? $1 : $0 } }
    static func maximumMagnitude(_ v: [Float]) -> Float { maximum(v.map { Swift.abs($0) }) }
    /// First index of the minimum / maximum (a strict scan, as the providers' NaN-free loop).
    static func minIndex(_ v: [Float]) -> Int { v.indices.dropFirst().reduce(0) { v[$1] < v[$0] ? $1 : $0 } }
    static func maxIndex(_ v: [Float]) -> Int { v.indices.dropFirst().reduce(0) { v[$1] > v[$0] ? $1 : $0 } }

    enum Fold { case max, min, maxMagnitude }

    /// A reduction: NaN if any element is NaN; `empty` for an empty input; otherwise the oracle under `==`
    /// (maxMagnitude: +0 for a zero result, Z1).
    static func reduction(_ fold: Fold, _ v: [Float], empty: T16Expect) -> T16Expect {
        if v.isEmpty { return empty }
        if v.contains(where: { $0.isNaN }) { return .nan }
        switch fold {
        case .max: return .equal(Double(maximum(v)))
        case .min: return .equal(Double(minimum(v)))
        case .maxMagnitude: return .positiveZeroOrEqual(Double(maximumMagnitude(v)))
        }
    }

    /// A running maximum seeded with `seed` (profiler, quantization error): NaN if any element is NaN.
    static func seededMax(_ v: [Float], seed: Float) -> T16Expect {
        if v.contains(where: { $0.isNaN }) { return .nan }
        return .equal(Double(v.reduce(seed) { $1 > $0 ? $1 : $0 }))
    }

    /// An index reduction: the first NaN's index; `empty` for an empty input; otherwise the first extremum.
    static func index(_ fold: Fold, _ v: [Float], empty: Int = 0) -> T16Expect {
        if v.isEmpty { return .index(empty) }
        if let first = v.firstIndex(where: { $0.isNaN }) { return .index(first) }
        return .index(fold == .min ? minIndex(v) : maxIndex(v))
    }

    /// `minElement()`/`maxElement()`: the index as above and the value at it (NaN for a NaN fixture).
    static func element(_ fold: Fold, _ v: [Float]) -> T16Expect {
        if let first = v.firstIndex(where: { $0.isNaN }) { return .indexAndValue(first, .nan) }
        let i = fold == .min ? minIndex(v) : maxIndex(v)
        return .indexAndValue(i, .equal(Double(v[i])))
    }

    /// Element-wise min/max: a NaN in either operand gives a NaN lane; otherwise `Swift.min`/`Swift.max`.
    static func elementWise(_ fold: Fold, _ a: [Float], _ b: [Float]) -> T16Expect {
        .lanes(zip(a, b).map { x, y in
            if x.isNaN || y.isNaN { return .nan }
            return .equal(Double(fold == .min ? Swift.min(x, y) : Swift.max(x, y)))
        })
    }

    /// Clip to [low, high]: a NaN lane stays NaN; otherwise the protocol formula `min(max(x, low), high)`.
    static func clip(_ v: [Float], low: Float, high: Float) -> T16Expect {
        .lanes(v.map { $0.isNaN ? .nan : .equal(Double(Swift.min(Swift.max($0, low), high))) })
    }
}
