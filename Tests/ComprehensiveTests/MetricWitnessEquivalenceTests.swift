//
//  MetricWitnessEquivalenceTests.swift
//  VectorCore
//
//  T14: one arithmetic per (metric, optimized type). For Vector512Optimized, Vector768Optimized and
//  Vector1536Optimized, a call with concrete static types, a generic call through the DistanceMetric witness, an
//  existential call, the concrete, generic and existential batch calls and a kernel pin (the vector type's own
//  `euclideanDistance(to:)` and `dotProduct(_:)` for Euclidean and dot product; for cosine the single-candidate
//  `BatchKernels.range_cosine_fused_*` call, which is `CosineKernels`' fused arithmetic) return the same result, and the
//  search APIs return the witness's result for each selected candidate outside the GEMM routes. The Euclidean and cosine
//  GEMM routes are a different arithmetic and are not checked: `Operations.findNearestBatch` with the explicit `.gram`
//  CPU strategy (P02), and the off-diagonal entries of `BatchOperations.pairwiseDistances` when `enableMatrixRouting` is
//  set and the count reaches `matrixRoutingMinN`; the search suite uses the batch default (`.perQuery`) and stays below
//  the pairwise gate. T14 phase 3 emptied the
//  `pendingPhase3` list: cosine is checked like the other metrics, and `MetricCosinePositionTests` checks that a cosine
//  score does not depend on the candidate's position (odd tail or pair) in the search's ranges.
//
//  "Same result" (`t14SameResult`) is equal bit patterns, which includes the sign of zero, or two NaNs. NaN payloads are
//  not compared (T14 phase 1 root ruling: no NaN-payload pins).
//

import Foundation
import Testing
import VectorCore

// MARK: - Cases

enum T14Metric: String, CaseIterable, Sendable {
    case euclidean, cosine, dotProduct, manhattan, chebyshev

    var existential: any DistanceMetric<Float> {
        switch self {
        case .euclidean: return EuclideanDistance()
        case .cosine: return CosineDistance()
        case .dotProduct: return DotProductDistance()
        case .manhattan: return ManhattanDistance()
        case .chebyshev: return ChebyshevDistance()
        }
    }

    /// The witness through a generic `M`, as Operations' generic code calls it.
    func generic<V: VectorProtocol>(_ a: V, _ b: V) -> Float where V.Scalar == Float {
        switch self {
        case .euclidean: return t14GenericDistance(EuclideanDistance(), a, b)
        case .cosine: return t14GenericDistance(CosineDistance(), a, b)
        case .dotProduct: return t14GenericDistance(DotProductDistance(), a, b)
        case .manhattan: return t14GenericDistance(ManhattanDistance(), a, b)
        case .chebyshev: return t14GenericDistance(ChebyshevDistance(), a, b)
        }
    }

    func genericBatch<V: VectorProtocol>(_ query: V, _ candidates: [V]) -> [Float] where V.Scalar == Float {
        switch self {
        case .euclidean: return t14GenericBatch(EuclideanDistance(), query, candidates)
        case .cosine: return t14GenericBatch(CosineDistance(), query, candidates)
        case .dotProduct: return t14GenericBatch(DotProductDistance(), query, candidates)
        case .manhattan: return t14GenericBatch(ManhattanDistance(), query, candidates)
        case .chebyshev: return t14GenericBatch(ChebyshevDistance(), query, candidates)
        }
    }
}

enum T14Width: Int, CaseIterable, Sendable {
    case w512 = 512, w768 = 768, w1536 = 1536
}

/// Fixture classes: 14 finite and 8 exceptional. Each exceptional class except `allNaN` places its value at
/// positions 0, n / 2 + 1 and n - 1 (one pair per position).
enum T14Fixture: String, CaseIterable, Sendable {
    case rand, wide, cancel, identical, zeroBoth, zeroOne, negZero, subnormal, tiny, underflowNeg, large, largeMixed,
         overflow, maxMagnitude
    case nanA, nanB, posInf, negInf, infInf, infNegInf, nanPayloads, allNaN
}

struct T14MetricCase: Sendable, Hashable, CustomStringConvertible, CustomTestStringConvertible {
    let metric: T14Metric
    let width: T14Width
    let fixture: T14Fixture
    var description: String { "\(metric.rawValue).\(width.rawValue).\(fixture.rawValue)" }
    var testDescription: String { description }
}

// MARK: - Dispatch helpers

/// The generic route. `optimize.sil.specialize.generic.never` keeps the optimizer from specializing this function, so
/// even an optimized test build calls the DistanceMetric witness through the witness table.
@inline(never) @_semantics("optimize.sil.specialize.generic.never")
func t14GenericDistance<M: DistanceMetric, V: VectorProtocol>(_ metric: M, _ a: V, _ b: V) -> Float
    where M.Scalar == Float, V.Scalar == Float {
    metric.distance(a, b)
}

@inline(never) @_semantics("optimize.sil.specialize.generic.never")
func t14GenericBatch<M: DistanceMetric, V: VectorProtocol>(_ metric: M, _ query: V, _ candidates: [V]) -> [Float]
    where M.Scalar == Float, V.Scalar == Float {
    metric.batchDistance(query: query, candidates: candidates)
}

@inline(never) @_semantics("optimize.sil.specialize.generic.never")
func t14ExistentialDistance<V: VectorProtocol>(_ metric: any DistanceMetric<Float>, _ a: V, _ b: V) -> Float
    where V.Scalar == Float {
    metric.distance(a, b)
}

@inline(never) @_semantics("optimize.sil.specialize.generic.never")
func t14ExistentialBatch<V: VectorProtocol>(_ metric: any DistanceMetric<Float>, _ query: V, _ candidates: [V]) -> [Float]
    where V.Scalar == Float {
    metric.batchDistance(query: query, candidates: candidates)
}

/// Equal bit patterns (including the sign of zero), or both NaN (payloads are not compared).
func t14SameResult(_ x: Float, _ y: Float) -> Bool { x.bitPattern == y.bitPattern || (x.isNaN && y.isNaN) }

func t14Hex(_ x: Float) -> String { "0x" + String(x.bitPattern, radix: 16) }

/// `kernel` over the single-candidate range `0..<1`: the odd-tail call of a `BatchKernels` range helper. The value is
/// read inside the buffer closure and returned from it (the Swift 6.4 lost-store lesson, T15/lost-store-investigation).
func t14CosineKernelPin<V>(_ a: V, _ b: V, _ kernel: (V, [V], Range<Int>, UnsafeMutableBufferPointer<Float>) -> Void) -> Float {
    var out = [Float](repeating: .nan, count: 1)
    return out.withUnsafeMutableBufferPointer { buffer -> Float in
        kernel(a, [b], 0..<1, buffer)
        return buffer[0]
    }
}

/// Calls with concrete static types for one optimized width. Inside these closures `a` and `b` have the concrete
/// vector type, so `EuclideanDistance().distance(a, b)` binds the most specific declaration visible: a concrete
/// overload when one exists, otherwise the generic witness.
struct T14ConcreteRoutes<V: VectorProtocol & Sendable> where V.Scalar == Float {
    let make: ([Float]) throws -> V
    let distance: (T14Metric, V, V) -> Float
    let batch: (T14Metric, V, [V]) -> [Float]
    /// A pin of the kernel the metric must use, when the type has one: its own `euclideanDistance(to:)` for Euclidean
    /// and `-dotProduct(_:)` for dot product (both delegate to the optimized kernels in the type's source), and for
    /// cosine the single-candidate `BatchKernels.range_cosine_fused_*` call (its odd-tail kernel,
    /// `CosineKernels.distance*_fused`; `t14CosineKernelPin`).
    let typeMethod: (T14Metric, V, V) -> Float?
}

enum T14Routes {
    static func w512() -> T14ConcreteRoutes<Vector512Optimized> {
        T14ConcreteRoutes(
            make: { try Vector512Optimized($0) },
            distance: { metric, a, b in
                switch metric {
                case .euclidean: return EuclideanDistance().distance(a, b)
                case .cosine: return CosineDistance().distance(a, b)
                case .dotProduct: return DotProductDistance().distance(a, b)
                case .manhattan: return ManhattanDistance().distance(a, b)
                case .chebyshev: return ChebyshevDistance().distance(a, b)
                }
            },
            batch: { metric, q, c in
                switch metric {
                case .euclidean: return EuclideanDistance().batchDistance(query: q, candidates: c)
                case .cosine: return CosineDistance().batchDistance(query: q, candidates: c)
                case .dotProduct: return DotProductDistance().batchDistance(query: q, candidates: c)
                case .manhattan: return ManhattanDistance().batchDistance(query: q, candidates: c)
                case .chebyshev: return ChebyshevDistance().batchDistance(query: q, candidates: c)
                }
            },
            typeMethod: { metric, a, b in
                switch metric {
                case .euclidean: return a.euclideanDistance(to: b)
                case .dotProduct: return -a.dotProduct(b)
                case .cosine:
                    return t14CosineKernelPin(a, b) { BatchKernels.range_cosine_fused_512(query: $0, candidates: $1, range: $2, out: $3) }
                case .manhattan, .chebyshev: return nil
                }
            })
    }

    static func w768() -> T14ConcreteRoutes<Vector768Optimized> {
        T14ConcreteRoutes(
            make: { try Vector768Optimized($0) },
            distance: { metric, a, b in
                switch metric {
                case .euclidean: return EuclideanDistance().distance(a, b)
                case .cosine: return CosineDistance().distance(a, b)
                case .dotProduct: return DotProductDistance().distance(a, b)
                case .manhattan: return ManhattanDistance().distance(a, b)
                case .chebyshev: return ChebyshevDistance().distance(a, b)
                }
            },
            batch: { metric, q, c in
                switch metric {
                case .euclidean: return EuclideanDistance().batchDistance(query: q, candidates: c)
                case .cosine: return CosineDistance().batchDistance(query: q, candidates: c)
                case .dotProduct: return DotProductDistance().batchDistance(query: q, candidates: c)
                case .manhattan: return ManhattanDistance().batchDistance(query: q, candidates: c)
                case .chebyshev: return ChebyshevDistance().batchDistance(query: q, candidates: c)
                }
            },
            typeMethod: { metric, a, b in
                switch metric {
                case .euclidean: return a.euclideanDistance(to: b)
                case .dotProduct: return -a.dotProduct(b)
                case .cosine:
                    return t14CosineKernelPin(a, b) { BatchKernels.range_cosine_fused_768(query: $0, candidates: $1, range: $2, out: $3) }
                case .manhattan, .chebyshev: return nil
                }
            })
    }

    static func w1536() -> T14ConcreteRoutes<Vector1536Optimized> {
        T14ConcreteRoutes(
            make: { try Vector1536Optimized($0) },
            distance: { metric, a, b in
                switch metric {
                case .euclidean: return EuclideanDistance().distance(a, b)
                case .cosine: return CosineDistance().distance(a, b)
                case .dotProduct: return DotProductDistance().distance(a, b)
                case .manhattan: return ManhattanDistance().distance(a, b)
                case .chebyshev: return ChebyshevDistance().distance(a, b)
                }
            },
            batch: { metric, q, c in
                switch metric {
                case .euclidean: return EuclideanDistance().batchDistance(query: q, candidates: c)
                case .cosine: return CosineDistance().batchDistance(query: q, candidates: c)
                case .dotProduct: return DotProductDistance().batchDistance(query: q, candidates: c)
                case .manhattan: return ManhattanDistance().batchDistance(query: q, candidates: c)
                case .chebyshev: return ChebyshevDistance().batchDistance(query: q, candidates: c)
                }
            },
            typeMethod: { metric, a, b in
                switch metric {
                case .euclidean: return a.euclideanDistance(to: b)
                case .dotProduct: return -a.dotProduct(b)
                case .cosine:
                    return t14CosineKernelPin(a, b) { BatchKernels.range_cosine_fused_1536(query: $0, candidates: $1, range: $2, out: $3) }
                case .manhattan, .chebyshev: return nil
                }
            })
    }
}

// MARK: - Fixtures

enum T14Fixtures {
    /// Deterministic pairs for one width and class (SplitMix64, `P01OpsRandom`); the same pairs serve every metric.
    static func pairs(width n: Int, fixture: T14Fixture) -> [([Float], [Float])] {
        let ordinal = UInt64(T14Fixture.allCases.firstIndex(of: fixture) ?? 0)
        var random = P01OpsRandom(state: 0x7140_0000 &+ UInt64(n) &* 1_000 &+ ordinal)
        return finitePairs(n, fixture, &random) ?? exceptionalPairs(n, fixture, &random)
    }

    /// The finite classes; nil for an exceptional class.
    private static func finitePairs(_ n: Int, _ fixture: T14Fixture, _ random: inout P01OpsRandom) -> [([Float], [Float])]? {
        func rand(_ scale: Float = 1) -> [Float] { (0..<n).map { _ in random.unit() * scale } }
        func wide() -> [Float] {
            (0..<n).map { _ in random.unit() * Float(sign: .plus, exponent: Int(random.next() % 41) - 20, significand: 1) }
        }
        let zero = [Float](repeating: 0, count: n), negZero = [Float](repeating: -0.0, count: n)
        switch fixture {
        case .rand: return (0..<48).map { _ in (rand(), rand()) }
        case .wide: return (0..<16).map { _ in (wide(), wide()) }
        case .cancel:
            return (0..<8).map { _ in
                ((0..<n).map { ($0 % 2 == 0 ? 1e4 : -1e4) + random.unit() * 1e-2 }, (0..<n).map { _ in 1 + random.unit() * 1e-3 })
            }
        case .identical: return (0..<4).map { _ in let a = rand(); return (a, a) }
        case .zeroBoth: return [(zero, zero)]
        case .zeroOne: return [(zero, rand()), (rand(), zero)]
        case .negZero: return [(negZero, zero), (negZero, negZero), (negZero, rand())]
        case .subnormal:
            let scale = Float(1e-39 as Double)  // a subnormal scale; a Float literal 1e-39 draws an underflow warning
            return [(rand(scale), rand(scale)), (rand(scale), rand())]
        case .tiny: return [(rand(1e-21), rand(1e-21)), (rand(1e-21), rand())]
        case .underflowNeg:
            // Every product is below half the least subnormal and negative.
            return [((0..<n).map { $0 % 2 == 0 ? 1e-10 : 1e-36 }, (0..<n).map { $0 % 2 == 0 ? -1e-36 : -1e-10 })]
        case .large: return [(rand(1e18), rand(1e18)), (rand(1e18), rand(1e18))]
        case .largeMixed: return [(rand(1e19), rand(1e19)), (rand(1e19), rand(1e19))]
        case .overflow: return [(rand(1e20), rand(1e20)), (rand(1e20), rand(1e20))]
        case .maxMagnitude:
            return [([Float](repeating: .greatestFiniteMagnitude, count: n), [Float](repeating: -.greatestFiniteMagnitude, count: n))]
        case .nanA, .nanB, .posInf, .negInf, .infInf, .infNegInf, .nanPayloads, .allNaN: return nil
        }
    }

    /// The exceptional classes: the value at positions 0, n / 2 + 1 and n - 1 of random pairs, or an all-NaN vector.
    private static func exceptionalPairs(_ n: Int, _ fixture: T14Fixture, _ random: inout P01OpsRandom) -> [([Float], [Float])] {
        func rand() -> [Float] { (0..<n).map { _ in random.unit() } }
        func exceptional(_ build: ([Float], [Float], Int) -> ([Float], [Float])) -> [([Float], [Float])] {
            [0, n / 2 + 1, n - 1].map { position in build(rand(), rand(), position) }
        }
        func with(_ base: [Float], _ position: Int, _ value: Float) -> [Float] {
            var copy = base
            copy[position] = value
            return copy
        }
        switch fixture {
        case .nanA: return exceptional { a, b, p in (with(a, p, .nan), b) }
        case .nanB: return exceptional { a, b, p in (a, with(b, p, .nan)) }
        case .posInf: return exceptional { a, b, p in (with(a, p, .infinity), b) }
        case .negInf: return exceptional { a, b, p in (with(a, p, -.infinity), b) }
        case .infInf: return exceptional { a, b, p in (with(a, p, .infinity), with(b, p, .infinity)) }
        case .infNegInf: return exceptional { a, b, p in (with(a, p, .infinity), with(b, p, -.infinity)) }
        case .nanPayloads:
            return exceptional { a, b, p in
                (with(a, p, Float(bitPattern: 0x7FC1_2345)), with(b, p, Float(bitPattern: 0xFFC5_4321)))
            }
        case .allNaN: return [([Float](repeating: .nan, count: n), rand())]
        default: return []
        }
    }

    /// 128 candidates: the class's second operands, then deterministic random rows. More than 100 candidates, the
    /// size above which the removed Euclidean batch overloads took a chunked branch.
    static func bigCandidates(width n: Int, fixture: T14Fixture) -> [[Float]] {
        var random = P01OpsRandom(state: 0xB16 &+ UInt64(n))
        let own = pairs(width: n, fixture: fixture).map(\.1)
        return own + (own.count..<128).map { _ in (0..<n).map { _ in random.unit() } }
    }
}

// MARK: - Bit equality per metric x width x fixture class

@Suite("Metric witness equivalence (T14)")
struct MetricWitnessEquivalenceTests {
    static let allCases: [T14MetricCase] = T14Metric.allCases.flatMap { metric in
        T14Width.allCases.flatMap { width in T14Fixture.allCases.map { T14MetricCase(metric: metric, width: width, fixture: $0) } }
    }

    /// Metrics whose cases were pending T14 phase 3. Phase 3 aligned cosine (the witness, the search kernels and the
    /// removed concrete overloads use `CosineKernels`' fused arithmetic), so the set is empty.
    static let pendingPhase3Metrics: Set<T14Metric> = []

    /// The named list of cases pending T14 phase 3; `caseCountGuard` counts it. Phase 3 emptied it.
    static let pendingPhase3: [T14MetricCase] = allCases.filter { pendingPhase3Metrics.contains($0.metric) }

    static let checkedCases: [T14MetricCase] = allCases.filter { !pendingPhase3Metrics.contains($0.metric) }

    /// Concrete (non-generic) `distance`/`batchDistance` declarations on DistanceMetric types, one key per
    /// declaration: `<Type>.<name>(<labels>) <parameter types>`. Read verbatim from this file by
    /// Scripts/ci/tests/test_metric_overload_registry.py, which fails on a declaration without a key and on a key
    /// without a declaration. `registryKeysMapToCases` maps every key to its cases here.
    /// Empty since T14 phase 3 removed the six cosine overloads (the last ones).
    static let concreteOverloadRegistry: [String: String] = [:]

    @Test func caseCountGuard() {
        #expect(T14Metric.allCases.count == 5)
        #expect(T14Width.allCases.count == 3)
        #expect(T14Fixture.allCases.count == 22)
        #expect(Self.allCases.count == 5 * 3 * 22)
        #expect(Set(Self.allCases).count == Self.allCases.count)
        #expect(Self.pendingPhase3.isEmpty, "T14 phase 3 emptied the pending list: \(Self.pendingPhase3.count) cases")
        #expect(Self.checkedCases.count == 5 * 3 * 22)
        #expect(Self.checkedCases.count + Self.pendingPhase3.count == Self.allCases.count)
        for width in T14Width.allCases {
            for fixture in T14Fixture.allCases {
                #expect(!T14Fixtures.pairs(width: width.rawValue, fixture: fixture).isEmpty, "\(width) \(fixture)")
            }
        }
    }

    @Test func registryKeysMapToCases() throws {
        #expect(Self.concreteOverloadRegistry.isEmpty, "T14 removed every concrete overload")
        for (key, status) in Self.concreteOverloadRegistry {
            let typeName = String(key.prefix { $0 != "." })
            let metric = try #require(T14Metric.allCases.first { $0.existential.name == Self.metricName(typeName) }, "\(key)")
            let widthText = try #require(key.firstMatch(of: /Vector(\d+)Optimized/)?.1, "\(key)")
            let width = try #require(T14Width(rawValue: Int(widthText) ?? 0), "\(key)")
            let cases = Self.allCases.filter { $0.metric == metric && $0.width == width }
            #expect(cases.count == T14Fixture.allCases.count, "\(key)")
            switch status {
            case "pending T14 phase 3": #expect(cases.allSatisfy(Self.pendingPhase3.contains), "\(key)")
            case "covered": #expect(cases.allSatisfy(Self.checkedCases.contains), "\(key)")
            default: Issue.record("unknown registry status \(status) for \(key)")
            }
        }
    }

    private static func metricName(_ typeName: String) -> String {
        switch typeName {
        case "EuclideanDistance": return "euclidean"
        case "CosineDistance": return "cosine"
        case "DotProductDistance": return "dotproduct"
        case "ManhattanDistance": return "manhattan"
        case "ChebyshevDistance": return "chebyshev"
        default: return typeName
        }
    }

    @Test("Concrete, generic, existential and batch calls agree", arguments: Self.checkedCases)
    func routesAgree(_ testCase: T14MetricCase) throws {
        switch testCase.width {
        case .w512: try Self.check(testCase, T14Routes.w512())
        case .w768: try Self.check(testCase, T14Routes.w768())
        case .w1536: try Self.check(testCase, T14Routes.w1536())
        }
    }

    private static func check<V: VectorProtocol & Sendable>(_ testCase: T14MetricCase, _ routes: T14ConcreteRoutes<V>) throws
        where V.Scalar == Float {
        let metric = testCase.metric, existential = metric.existential, n = testCase.width.rawValue
        let pairs = try T14Fixtures.pairs(width: n, fixture: testCase.fixture).map { (try routes.make($0.0), try routes.make($0.1)) }
        for (index, (a, b)) in pairs.enumerated() {
            let generic = metric.generic(a, b)
            let concreteBatch = routes.batch(metric, a, [b])
            let genericBatch = metric.genericBatch(a, [b])
            let existentialBatch = t14ExistentialBatch(existential, a, [b])
            #expect(concreteBatch.count == 1 && genericBatch.count == 1 && existentialBatch.count == 1, "\(testCase) pair \(index)")
            let values: [(String, Float)] = [
                ("concrete", routes.distance(metric, a, b)), ("existential", t14ExistentialDistance(existential, a, b)),
                ("concreteBatch", concreteBatch.first ?? .nan), ("genericBatch", genericBatch.first ?? .nan),
                ("existentialBatch", existentialBatch.first ?? .nan)
            ] + (routes.typeMethod(metric, a, b).map { [("typeMethod", $0)] } ?? [])
            #expect(values.allSatisfy { t14SameResult($0.1, generic) },
                    "\(testCase) pair \(index): generic=\(t14Hex(generic)) \(values.map { "\($0.0)=\(t14Hex($0.1))" })")
        }

        // One 128-candidate batch per case, query = the first pair's first operand.
        let query = try #require(pairs.first?.0)
        let candidates = try T14Fixtures.bigCandidates(width: n, fixture: testCase.fixture).map(routes.make)
        let concreteBatch = routes.batch(metric, query, candidates)
        let genericBatch = metric.genericBatch(query, candidates)
        let existentialBatch = t14ExistentialBatch(existential, query, candidates)
        #expect(concreteBatch.count == 128 && genericBatch.count == 128 && existentialBatch.count == 128, "\(testCase)")
        var mismatches: [String] = []
        for index in candidates.indices where index < concreteBatch.count && index < genericBatch.count
            && index < existentialBatch.count {
            let witness = metric.generic(query, candidates[index])
            let batchValues = [concreteBatch[index], genericBatch[index], existentialBatch[index]]
            if !batchValues.allSatisfy({ t14SameResult($0, witness) }) {
                mismatches.append("\(index): witness=\(t14Hex(witness)) batches=\(batchValues.map(t14Hex))")
            }
        }
        #expect(mismatches.isEmpty, "\(testCase) 128-candidate batch: \(mismatches.count) mismatches, first \(mismatches.prefix(4))")
    }

    /// P01 phase E run E1: dot product on Vector1536Optimized, the 19th row (index 18) of
    /// `P01OperationsRouteTests.customAndParameterizedWitnesses`, replayed with the same generator state.
    @Test func p01Dot1536Row18Replay() async throws {
        var random = P01OpsRandom(state: 23)
        // That test's earlier draws: 64 values, 64 rows x 510 coordinates, 200 x 384, and the 384-coordinate query.
        for _ in 0..<(64 + 64 * 510 + 200 * 384 + 384) { _ = random.next() }
        let rows = (0..<40).map { _ in Vector1536Optimized(generator: { _ in random.unit() }) }
        let query = Vector1536Optimized(generator: { _ in random.unit() })
        let metric = DotProductDistance()
        let found = try await Operations.findNearest(to: query, in: rows, k: rows.count, metric: metric)
        let search = try #require(found.first { $0.index == 18 }).distance
        let values: [(String, Float)] = [
            ("concrete", metric.distance(query, rows[18])), ("generic", t14GenericDistance(metric, query, rows[18])),
            ("existential", t14ExistentialDistance(metric as any DistanceMetric<Float>, query, rows[18])),
            ("Operations.findNearest k=n", search)
        ]
        let history = "historical (P01 phase E run E1): 0xc1ce604c through Operations and the witness, 0xc1ce604e "
            + "through the concrete overload"
        #expect(values.allSatisfy { $0.1.bitPattern == values[0].1.bitPattern },
                "\(history); now \(values.map { "\($0.0)=\(t14Hex($0.1))" })")
    }
}

// MARK: - Search APIs return the witness's result

private struct T14Collection<V: IndexableVector>: VectorCollection {
    let values: [V]
    var count: Int { values.count }
    var dimension: Int { values.first?.scalarCount ?? 0 }
    var allIDs: [Int] { Array(values.indices) }
    func vector(for id: Int) -> V? { values.indices.contains(id) ? values[id] : nil }
}

struct T14SearchCase: Sendable, CustomStringConvertible, CustomTestStringConvertible {
    let metric: T14Metric
    let width: T14Width
    var description: String { "\(metric.rawValue).\(width.rawValue)" }
    var testDescription: String { description }
}

/// Every public search or matrix API that calls `DistanceMetric.distance` or `batchDistance`, or a kernel that
/// replaces it, for the optimized types (the T14 gate-change reachability list), compared with the witness outside the
/// GEMM routes: `Operations.findNearestBatch` gets 2 queries on its default `.perQuery` strategy (its GEMM route runs
/// only with the explicit `.gram` strategy) and `BatchOperations.pairwiseDistances` 48 and 100 rows (below the default
/// `matrixRoutingMinN`, 256, which `pairwise` checks). The GEMM routes and `MatrixDistance` are a different arithmetic and are not compared.
@Suite("Metric search witness consistency (T14)")
struct MetricSearchWitnessConsistencyTests {
    static let cases: [T14SearchCase] = T14Metric.allCases.flatMap { metric in
        T14Width.allCases.map { T14SearchCase(metric: metric, width: $0) }
    }

    @Test func caseCountGuard() {
        #expect(Self.cases.count == 5 * 3)
        #expect(!Self.cases.contains { MetricWitnessEquivalenceTests.pendingPhase3Metrics.contains($0.metric) })
    }

    @Test("Search scores equal the witness at k = n, n - 1 and 3", arguments: Self.cases)
    func searchEqualsWitness(_ testCase: T14SearchCase) async throws {
        switch testCase.width {
        case .w512: try await Self.check(testCase, T14Routes.w512())
        case .w768: try await Self.check(testCase, T14Routes.w768())
        case .w1536: try await Self.check(testCase, T14Routes.w1536())
        }
    }

    @Test("A candidate's search score does not depend on k", arguments: Self.cases)
    func searchScoreIndependentOfK(_ testCase: T14SearchCase) async throws {
        switch testCase.width {
        case .w512: try await Self.checkK(testCase, T14Routes.w512())
        case .w768: try await Self.checkK(testCase, T14Routes.w768())
        case .w1536: try await Self.checkK(testCase, T14Routes.w1536())
        }
    }

    /// 48 rows: random, with an all-NaN row 7 and a zero row 19.
    static func rows<V>(_ testCase: T14SearchCase, _ make: ([Float]) throws -> V, count: Int = 48, seed: UInt64 = 0) rethrows -> [V] {
        let n = testCase.width.rawValue
        var random = P01OpsRandom(state: 0x5EA2_0000 &+ UInt64(n) &+ seed)
        return try (0..<count).map { row in
            if row == 7 { return try make([Float](repeating: .nan, count: n)) }
            if row == 19 { return try make([Float](repeating: 0, count: n)) }
            return try make((0..<n).map { _ in random.unit() })
        }
    }

    static func query<V>(_ testCase: T14SearchCase, _ make: ([Float]) throws -> V, seed: UInt64) rethrows -> V {
        var random = P01OpsRandom(state: 0x0E21_0000 &+ UInt64(testCase.width.rawValue) &+ seed)
        return try make((0..<testCase.width.rawValue).map { _ in random.unit() })
    }

    /// Entries (index, score) compared by index and `t14SameResult`.
    static func agree(_ actual: [(Int, Float)], _ expected: [(Int, Float)]) -> Bool {
        actual.count == expected.count && zip(actual, expected).allSatisfy { $0.0 == $1.0 && t14SameResult($0.1, $1.1) }
    }

    static func describe(_ entries: [(Int, Float)]) -> String {
        entries.prefix(6).map { "\($0.0):\(t14Hex($0.1))" }.joined(separator: " ")
    }

    /// The witness's best `k` by the Top-K policy (numbers before NaN, then smaller index).
    static func oracle(_ scores: [Float], k: Int) -> [(Int, Float)] {
        P01OpsOracle.select(scores, k: k).map { ($0.index, Float(bitPattern: $0.bits)) }
    }

    /// `queries` holds the query and a second query.
    private static func search<V: IndexableVector & VectorType, M: DistanceMetric>(
        _ label: String, _ metric: M, _ queries: [V], _ rows: [V], witness: (V, V) -> Float
    ) async throws -> [String] where M.Scalar == Float {
        let query = queries[0], second = queries[1]
        var failures: [String] = []
        let scores = rows.map { witness(query, $0) }
        let n = rows.count
        for k in [n, n - 1, 3] {
            let expected = oracle(scores, k: k)
            let operations = try await Operations.findNearest(to: query, in: rows, k: k, metric: metric).map { ($0.index, $0.distance) }
            if !agree(operations, expected) {
                failures.append("\(label) Operations.findNearest k=\(k): \(describe(operations)) vs \(describe(expected))")
            }
            let batch = await BatchOperations.findNearest(to: query, in: rows, k: k, metric: metric).map { ($0.index, $0.distance) }
            if !agree(batch, expected) {
                failures.append("\(label) BatchOperations.findNearest k=\(k): \(describe(batch)) vs \(describe(expected))")
            }
            let facade = await rows.findNearest(to: query, k: k, metric: metric).map { ($0.index, $0.distance) }
            if !agree(facade, expected) {
                failures.append("\(label) Array.findNearest k=\(k): \(describe(facade)) vs \(describe(expected))")
            }
            let topK = TopKSelection.nearest(k: k, query: query, candidates: rows, metric: metric)
            let topKEntries = Array(zip(topK.indices, topK.distances))
            if !agree(topKEntries, expected) { failures.append("\(label) TopKSelection.nearest k=\(k): \(describe(topKEntries))") }
            let found = T14Collection(values: rows).search(query: query, k: k, metric: metric)
            let foundEntries = Array(zip(found.ids, found.distances))
            if !agree(foundEntries, expected) { failures.append("\(label) VectorCollection.search k=\(k): \(describe(foundEntries))") }
            // Two queries on findNearestBatch's default `.perQuery` strategy, so every query takes findNearest's routes.
            let batched = try await Operations.findNearestBatch(queries: [query, second], in: rows, k: k, metric: metric)
            let expectedSecond = oracle(rows.map { witness(second, $0) }, k: k)
            if batched.count != 2 || !agree(batched[0].map { ($0.index, $0.distance) }, expected)
                || !agree(batched[1].map { ($0.index, $0.distance) }, expectedSecond) {
                failures.append("\(label) Operations.findNearestBatch k=\(k)")
            }
            let topKBatch = TopKSelection.batchNearest(k: k, queries: [query, second], candidates: rows, metric: metric)
            if topKBatch.count != 2 || !agree(Array(zip(topKBatch[1].indices, topKBatch[1].distances)), expectedSecond) {
                failures.append("\(label) TopKSelection.batchNearest k=\(k)")
            }
        }
        // Matrices: every entry is the witness's score for its pair.
        let matrix = try await Operations.distanceMatrix(between: [query, second], and: rows, metric: metric)
        for (i, a) in [query, second].enumerated() where matrix.count == 2 {
            for (j, b) in rows.enumerated() where !t14SameResult(matrix[i][j], witness(a, b)) {
                failures.append("\(label) Operations.distanceMatrix [\(i)][\(j)]: \(t14Hex(matrix[i][j])) vs \(t14Hex(witness(a, b)))")
            }
        }
        return failures
    }

    /// Pairwise matrices on the per-pair routes: n = 48 (serial) and n = 100 (row workers); both are below the default
    /// 256-row GEMM gate. The rows are finite, because the pairwise contract assumes a symmetric metric.
    private static func pairwise<V: VectorProtocol & VectorType & Sendable, M: DistanceMetric>(
        _ label: String, _ metric: M, _ rows: [V], witness: (V, V) -> Float
    ) async -> [String] where M.Scalar == Float, V.Scalar == Float {
        var failures: [String] = []
        let config = await BatchOperations.configuration()
        if rows.count >= config.matrixRoutingMinN { return ["\(label): \(rows.count) rows reach the GEMM gate"] }
        for matrix in [await BatchOperations.pairwiseDistances(rows, metric: metric), await rows.pairwiseDistances(metric: metric)] {
            var bad = 0
            for i in rows.indices {
                for j in rows.indices where !(matrix.count == rows.count && t14SameResult(matrix[i][j], witness(rows[i], rows[j]))) {
                    bad += 1
                }
            }
            if bad > 0 { failures.append("\(label) pairwiseDistances n=\(rows.count): \(bad) entries differ from the witness") }
        }
        return failures
    }

    private static func check<V: IndexableVector & VectorType>(_ testCase: T14SearchCase, _ routes: T14ConcreteRoutes<V>) async throws {
        let metric = testCase.metric
        let rows = try Self.rows(testCase, routes.make)
        let query = try Self.query(testCase, routes.make, seed: 1), second = try Self.query(testCase, routes.make, seed: 2)
        let witness: (V, V) -> Float = { metric.generic($0, $1) }
        var failures: [String] = []
        switch metric {
        case .euclidean:
            failures += try await search("\(testCase)", EuclideanDistance(), [query, second], rows, witness: witness)
            // MinkowskiDistance with p = 2 delegates to the Euclidean witness.
            failures += rows.indices.filter { !t14SameResult(MinkowskiDistance(p: 2).distance(query, rows[$0]), witness(query, rows[$0])) }
                .map { "\(testCase) Minkowski p=2 row \($0)" }
        case .dotProduct: failures += try await search("\(testCase)", DotProductDistance(), [query, second], rows, witness: witness)
        case .manhattan:
            failures += try await search("\(testCase)", ManhattanDistance(), [query, second], rows, witness: witness)
            failures += rows.indices.filter { !t14SameResult(MinkowskiDistance(p: 1).distance(query, rows[$0]), witness(query, rows[$0])) }
                .map { "\(testCase) Minkowski p=1 row \($0)" }
        case .chebyshev:
            failures += try await search("\(testCase)", ChebyshevDistance(), [query, second], rows, witness: witness)
            failures += rows.indices.filter {
                !t14SameResult(MinkowskiDistance(p: .infinity).distance(query, rows[$0]), witness(query, rows[$0]))
            }.map { "\(testCase) Minkowski p=inf row \($0)" }
        case .cosine: failures += try await search("\(testCase)", CosineDistance(), [query, second], rows, witness: witness)
        }
        // Parallel search routes: 1,000 rows reach BatchOperations' default parallelThreshold (1000).
        let many = try Self.rows(testCase, routes.make, count: 1_000, seed: 9)
        let finite = try Self.rows(testCase, routes.make, count: 100, seed: 5).enumerated().filter { $0.offset != 7 }.map(\.element)
        switch metric {
        case .euclidean:
            failures += try await parallel("\(testCase)", EuclideanDistance(), query, many, witness: witness)
            failures += await pairwise("\(testCase)", EuclideanDistance(), Array(finite.prefix(48)), witness: witness)
            failures += await pairwise("\(testCase)", EuclideanDistance(), finite + [finite[0]], witness: witness)
        case .dotProduct:
            failures += try await parallel("\(testCase)", DotProductDistance(), query, many, witness: witness)
            failures += await pairwise("\(testCase)", DotProductDistance(), Array(finite.prefix(48)), witness: witness)
            failures += await pairwise("\(testCase)", DotProductDistance(), finite + [finite[0]], witness: witness)
        case .cosine:
            failures += try await parallel("\(testCase)", CosineDistance(), query, many, witness: witness)
            failures += await pairwise("\(testCase)", CosineDistance(), Array(finite.prefix(48)), witness: witness)
            failures += await pairwise("\(testCase)", CosineDistance(), finite + [finite[0]], witness: witness)
        case .manhattan: failures += await pairwise("\(testCase)", ManhattanDistance(), Array(finite.prefix(48)), witness: witness)
        case .chebyshev: failures += await pairwise("\(testCase)", ChebyshevDistance(), Array(finite.prefix(48)), witness: witness)
        }
        #expect(failures.isEmpty, "\(failures.count) failures: \(failures.prefix(12))")
    }

    private static func parallel<V: IndexableVector, M: DistanceMetric>(
        _ label: String, _ metric: M, _ query: V, _ rows: [V], witness: (V, V) -> Float
    ) async throws -> [String] where M.Scalar == Float {
        var failures: [String] = []
        let config = await BatchOperations.configuration()
        if rows.count < config.parallelThreshold { return ["\(label): \(rows.count) rows are below parallelThreshold"] }
        let scores = rows.map { witness(query, $0) }
        for k in [rows.count, 3] {
            let expected = oracle(scores, k: k)
            let batch = await BatchOperations.findNearest(to: query, in: rows, k: k, metric: metric).map { ($0.index, $0.distance) }
            if !agree(batch, expected) { failures.append("\(label) BatchOperations.findNearest parallel k=\(k): \(describe(batch))") }
            let operations = try await Operations.findNearest(to: query, in: rows, k: k, metric: metric).map { ($0.index, $0.distance) }
            if !agree(operations, expected) { failures.append("\(label) Operations.findNearest n=1000 k=\(k): \(describe(operations))") }
        }
        return failures
    }

    /// For each k in {n, n - 1, 3}, every returned candidate's score equals that candidate's score at k = n.
    private static func checkK<V: IndexableVector>(_ testCase: T14SearchCase, _ routes: T14ConcreteRoutes<V>) async throws {
        let rows = try Self.rows(testCase, routes.make)
        let query = try Self.query(testCase, routes.make, seed: 1)
        var failures: [String] = []
        func run<M: DistanceMetric>(_ metric: M) async throws where M.Scalar == Float {
            let full = try await Operations.findNearest(to: query, in: rows, k: rows.count, metric: metric)
            let byIndex = Dictionary(uniqueKeysWithValues: full.map { ($0.index, $0.distance) })
            for k in [rows.count - 1, 3] {
                for entry in try await Operations.findNearest(to: query, in: rows, k: k, metric: metric) {
                    if let reference = byIndex[entry.index], !t14SameResult(entry.distance, reference) {
                        failures.append("k=\(k) row \(entry.index): \(t14Hex(entry.distance)) vs k=n \(t14Hex(reference))")
                    }
                }
            }
        }
        switch testCase.metric {
        case .euclidean: try await run(EuclideanDistance())
        case .dotProduct: try await run(DotProductDistance())
        case .manhattan: try await run(ManhattanDistance())
        case .chebyshev: try await run(ChebyshevDistance())
        case .cosine: try await run(CosineDistance())
        }
        #expect(failures.isEmpty, "\(testCase) Operations.findNearest: \(failures.count) scores depend on k: \(failures.prefix(6))")
    }
}

// MARK: - Cosine scores do not depend on a candidate's position

/// Every `BatchKernels` cosine range helper scores candidates in pairs from `range.lowerBound` and an odd last candidate
/// with its single-candidate kernel. Before T14 phase 3 the cosine pair loop used a different arithmetic from that
/// kernel, so a candidate's score depended on whether it was the odd tail of a range, that is on the number of
/// candidates and on the chunking (`BatchKernelsRangeClosureTests` checks the helpers and the chunked parallel helper
/// directly). This suite checks it through the public search APIs on near-duplicate rows, where the old pair and tail
/// arithmetics differed on most rows in the T14 phase 3a probe: the score of row j at k = n over the first j + 1 rows
/// (an odd count for even j, so row j is the tail of the only range) equals its score over all 48 rows (row j in a
/// pair).
@Suite("Cosine score position independence (T14)")
struct MetricCosinePositionTests {
    @Test("A cosine search score does not depend on the number of candidates", arguments: T14Width.allCases)
    func scoreDoesNotDependOnCandidateCount(_ width: T14Width) async throws {
        switch width {
        case .w512: try await Self.check(width, T14Routes.w512())
        case .w768: try await Self.check(width, T14Routes.w768())
        case .w1536: try await Self.check(width, T14Routes.w1536())
        }
    }

    private static func scores(_ entries: [(Int, Float)]) -> [Int: Float] {
        Dictionary(entries, uniquingKeysWith: { first, _ in first })
    }

    private static func check<V: VectorProtocol & Sendable>(_ width: T14Width, _ routes: T14ConcreteRoutes<V>) async throws
        where V.Scalar == Float {
        var random = P01OpsRandom(state: 0xC050_0000 &+ UInt64(width.rawValue))
        let base = (0..<width.rawValue).map { _ in random.unit() }
        let query = try routes.make(base)
        let rows = try (0..<48).map { _ in try routes.make(base.map { $0 + random.unit() * 1e-3 }) }
        let metric = CosineDistance()
        let fullOps = scores(try await Operations.findNearest(to: query, in: rows, k: rows.count, metric: metric)
            .map { ($0.index, $0.distance) })
        let fullBatch = scores(await BatchOperations.findNearest(to: query, in: rows, k: rows.count, metric: metric)
            .map { ($0.index, $0.distance) })
        #expect(fullOps.count == rows.count && fullBatch.count == rows.count, "\(width)")
        var failures: [String] = []
        for j in stride(from: 0, to: rows.count, by: 2) {
            let prefix = Array(rows[0...j])
            let ops = scores(try await Operations.findNearest(to: query, in: prefix, k: prefix.count, metric: metric)
                .map { ($0.index, $0.distance) })
            let batch = scores(await BatchOperations.findNearest(to: query, in: prefix, k: prefix.count, metric: metric)
                .map { ($0.index, $0.distance) })
            for (api, tail, pair) in [("Operations", ops[j], fullOps[j]), ("BatchOperations", batch[j], fullBatch[j])] {
                guard let tail, let pair else {
                    failures.append("\(api) row \(j): missing")
                    continue
                }
                if !t14SameResult(tail, pair) {
                    failures.append("\(api) row \(j): \(t14Hex(tail)) as the odd tail of \(j + 1) rows vs \(t14Hex(pair)) of 48")
                }
            }
        }
        #expect(failures.isEmpty, "\(width): \(failures.count) position-dependent scores: \(failures.prefix(6))")
    }
}

// MARK: - Dot search routes, observed without score bits

/// Since T14 the dot-product witness on the optimized types returns `DotKernels` bits, so score bits no longer tell
/// `Operations.findNearest`'s direct dot route (step 2: `Vector512Optimized` with `k < n`) from its stream (step 3),
/// which scores through the witness. P01's dot512 direct-route parity and its dot1536 "streams with its own
/// protocol-witness scores" check compare bits, so this suite observes the route by cancellation instead, as
/// `findNearest`'s doc lists it: under `P01UncheckedProviderProbe`, a provider that performs no cancellation checks,
/// a cancelled search on the direct route completes, and the stream throws `CancellationError` before its first metric
/// call. With k = 3 and n = 64 the stream's small-k gate (`k < n / 10`) holds, so a 512 search that left the direct
/// route would stream, and a 768 or 1536 search that left the stream for a route without per-candidate checks would
/// complete.
@Suite("Dot search route observation (T14)")
struct MetricDotRouteObservationTests {
    @Test("A cancelled dot search completes on the 512 direct route and throws on the 768 and 1536 stream",
          arguments: T14Width.allCases)
    func cancelledDotSearchShowsTheRoute(_ width: T14Width) async throws {
        switch width {
        case .w512: try await Self.check(width, T14Routes.w512(), direct: true)
        case .w768: try await Self.check(width, T14Routes.w768(), direct: false)
        case .w1536: try await Self.check(width, T14Routes.w1536(), direct: false)
        }
    }

    private static func check<V: VectorProtocol & Sendable>(
        _ width: T14Width, _ routes: T14ConcreteRoutes<V>, direct: Bool
    ) async throws where V.Scalar == Float {
        var random = P01OpsRandom(state: 0xD070_0000 &+ UInt64(width.rawValue))
        let rows = try (0..<64).map { _ in try routes.make((0..<width.rawValue).map { _ in random.unit() }) }
        let k = 3
        let unchecked = P01UncheckedProviderProbe(plan: { [$0] })
        // Fixture control: without cancellation the same search returns k results.
        let plain = try await Operations.$computeProvider.withValue(unchecked) {
            try await Operations.findNearest(to: rows[0], in: rows, k: k, metric: DotProductDistance())
        }
        #expect(plain.count == k, "\(width): \(plain)")
        let cancelled = await Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await Operations.$computeProvider.withValue(unchecked) {
                try await Operations.findNearest(to: rows[0], in: rows, k: k, metric: DotProductDistance())
            }
        }.result
        switch (direct, cancelled) {
        case (true, .success(let result)):
            #expect(result.map(\.index) == plain.map(\.index), "\(width): the cancelled direct search returned \(result)")
        case (false, .failure(let error)):
            #expect(error is CancellationError, "\(width): \(error)")
        default:
            let expected = direct ? "the direct route to complete" : "the stream to throw CancellationError"
            Issue.record("\(width): expected \(expected), got \(cancelled)")
        }
    }
}
