import Foundation
@testable import VectorCore

// T16 reduction registry (class closure, T16 phase 1 rulings (c), (d)).
//
// `T16ReductionRegistry.rows` lists every min/max-family reduction, index reduction, statistic and element-wise
// min/max/clip implementation, once per provider, as a row of the exceptional table
// (T16ReductionExceptionalTableTests). `asyncRows` are the PerformanceProfiler rows, which run in an isolated child
// process (ruling C4).
//
// `T16ReductionRegistry.sites` is the site map that `Scripts/ci/tests/test_reduction_nan_guard.py` parses: one
// string literal per fold the guard discovers in Sources (`path::Type.body::shape:operand`, a multiset), between the
// BEGIN and END markers. A discovered fold that is missing here, or an entry with no fold, fails the guard. Each entry
// names the reviewed NaN decision:
//   .tabled(row)       a row of the exceptional table exercises the site (the guard checks the row is declared;
//                      T16ReductionExceptionalTableTests.registryCoverage checks it is evaluated);
//   .external(test)    a test outside this table pins the site (`path::Suite.test`; T15-owned folds);
//   .exempt(reason)    `<category>: <reason>`, category finite-guarded, by-construction, int or routed:<task>.

enum T16SiteCoverage: Sendable {
    case tabled(String)
    case external(String)
    case exempt(String)
}

enum T16Kind: String, Sendable { case reduction, index, statistic, elementWise, control }

/// Provider tag of a row: the Swift SIMD providers, the Accelerate (vDSP) providers, or scalar library code.
enum T16Tag: String, Sendable, CaseIterable { case swift, accelerate, scalar }

typealias T16Input = @Sendable (T16Fixture) -> [Float]?
typealias T16Expectation = @Sendable ([Float]) -> T16Expect
typealias T16Run = @Sendable ([Float]) -> T16Got

struct T16Row: Sendable {
    let id: String
    let kind: T16Kind
    let tag: T16Tag
    /// The values the implementation reduces over for a fixture, or nil when the fixture does not apply.
    let input: T16Input
    let expect: T16Expectation
    let run: T16Run
}

struct T16AsyncRow: Sendable {
    let id: String
    let kind: T16Kind
    let tag: T16Tag
    let expect: T16Expectation
    let run: @Sendable ([Float]) async -> T16Got
}

enum T16ReductionRegistry {
    static func row(_ id: String, _ kind: T16Kind, _ tag: T16Tag, input: @escaping T16Input = { $0.values },
                    expect: @escaping T16Expectation, run: @escaping T16Run) -> T16Row {
        T16Row(id: id, kind: kind, tag: tag, input: input, expect: expect, run: run)
    }

    static func asyncRow(_ id: String, _ kind: T16Kind, _ tag: T16Tag, expect: @escaping T16Expectation,
                         run: @escaping @Sendable ([Float]) async -> T16Got) -> T16AsyncRow {
        T16AsyncRow(id: id, kind: kind, tag: tag, expect: expect, run: run)
    }

    // MARK: Inputs

    static func nonEmpty(_ f: T16Fixture) -> [Float]? { f.values.isEmpty ? nil : f.values }
    static func blocksOfFour(_ f: T16Fixture) -> [Float]? { f.values.count % 4 == 0 ? f.values : nil }
    static let vectorDimensions: Set<Int> = [2, 3, 4, 8, 16, 32, 64]
    static func vectorSized(_ f: T16Fixture) -> [Float]? { vectorDimensions.contains(f.values.count) ? f.values : nil }
    static func withNaN(_ f: T16Fixture) -> [Float]? { vectorSized(f).flatMap { $0.contains(where: \.isNaN) ? $0 : nil } }
    /// Vector512Optimized: the fixture followed by 0.25 up to 512 elements (the first NaN keeps its index).
    static func padded512(_ f: T16Fixture) -> [Float]? {
        f.values.isEmpty ? nil : f.values + [Float](repeating: 0.25, count: 512 - f.values.count)
    }

    // MARK: Evaluation helpers (results are returned from inside every closure; T15 lost-store rule C5)

    static func pointer(_ v: [Float], _ f: (UnsafePointer<Float>, Int) -> Float) -> T16Got {
        let storage: [Float] = v.isEmpty ? [7] : v   // count 0 still passes a valid pointer
        return .value(Double(storage.withUnsafeBufferPointer { f($0.baseAddress!, v.count) }))
    }

    static func pointerD(_ v: [Float], _ f: (UnsafePointer<Double>, Int) -> Double) -> T16Got {
        let storage: [Double] = v.isEmpty ? [7] : v.map(Double.init)
        return .value(storage.withUnsafeBufferPointer { f($0.baseAddress!, v.count) })
    }

    static let clipLow: Float = -2, clipHigh: Float = 3

    static func clipF(_ v: [Float], _ f: (UnsafePointer<Float>, Float, Float, UnsafeMutablePointer<Float>, Int) -> Void) -> T16Got {
        let out = v.withUnsafeBufferPointer { a in
            [Float](unsafeUninitializedCapacity: v.count) { buffer, initialized in
                f(a.baseAddress!, clipLow, clipHigh, buffer.baseAddress!, v.count)
                initialized = v.count
            }
        }
        return .lanes(out.map(Double.init))
    }

    static func clipD(_ v: [Float], _ f: (UnsafePointer<Double>, Double, Double, UnsafeMutablePointer<Double>, Int) -> Void) -> T16Got {
        let input = v.map(Double.init)
        let out = input.withUnsafeBufferPointer { a in
            [Double](unsafeUninitializedCapacity: v.count) { buffer, initialized in
                f(a.baseAddress!, Double(clipLow), Double(clipHigh), buffer.baseAddress!, v.count)
                initialized = v.count
            }
        }
        return .lanes(out)
    }

    static func lanes(_ v: [Float]) -> T16Got { .lanes(v.map(Double.init)) }

    static func vpMin<V: VectorProtocol>(_ v: V) -> T16Got where V.Scalar == Float { v.min.map { .value(Double($0)) } ?? .absent }
    static func vpMax<V: VectorProtocol>(_ v: V) -> T16Got where V.Scalar == Float { v.max.map { .value(Double($0)) } ?? .absent }

    enum VectorOp { case minElement, maxElement, softmax, clamped, clamp, elementMin, elementMax, vpMin, vpMax }

    static func onVector(_ v: [Float], _ provider: (any ArraySIMDProvider)?, _ op: VectorOp) -> T16Got {
        switch v.count {
        case 2: return apply(try! Vector<Dim2>(v), provider, op)
        case 3: return apply(try! Vector<Dim3>(v), provider, op)
        case 4: return apply(try! Vector<Dim4>(v), provider, op)
        case 8: return apply(try! Vector<Dim8>(v), provider, op)
        case 16: return apply(try! Vector<Dim16>(v), provider, op)
        case 32: return apply(try! Vector<Dim32>(v), provider, op)
        case 64: return apply(try! Vector<Dim64>(v), provider, op)
        default: preconditionFailure("no static dimension for \(v.count)")
        }
    }

    static func apply<D: StaticDimension>(_ x: Vector<D>, _ provider: (any ArraySIMDProvider)?, _ op: VectorOp) -> T16Got {
        let body = { () -> T16Got in
            let other = try! Vector<D>(T16Fixtures.partner(D.value))
            switch op {
            case .minElement: let r = x.minElement(); return .indexAndValue(r.index, Double(r.value))
            case .maxElement: let r = x.maxElement(); return .indexAndValue(r.index, Double(r.value))
            case .softmax: return lanes(x.softmax().toArray())
            case .clamped: return lanes(x.clamped(to: clipLow...clipHigh).toArray())
            case .clamp: var y = x; y.clamp(to: clipLow...clipHigh); return lanes(y.toArray())
            case .elementMin: return lanes(x.min(other).toArray())
            case .elementMax: return lanes(x.max(other).toArray())
            case .vpMin: return vpMin(x)
            case .vpMax: return vpMax(x)
            }
        }
        guard let provider else { return body() }
        return Operations.$simdProvider.withValue(provider) { body() }
    }

    static func minMax(_ v: [Float]) -> (min: Float, max: Float) {
        var storage = ContiguousArray<SIMD4<Float>>()
        for k in stride(from: 0, to: v.count, by: 4) { storage.append(SIMD4(v[k], v[k + 1], v[k + 2], v[k + 3])) }
        return storage.minMax()
    }

    /// `Operations.statistics` over one row per fixture value: `[x, 1]`, so column 0 is the fixture.
    static func statistics(_ v: [Float]) -> VectorStatistics {
        try! Operations.statistics(for: v.map { DynamicVector([$0, 1]) })
    }

    static func magnitudes(_ v: [Float]) -> [Float] { v.map { Operations.simdProvider.magnitude([$0, 1]) } }

    static func quantizationMaxError(_ v: [Float]) -> T16Got {
        let params = QuantizationParams(strategy: .perVector, scales: [1], offsets: [0])
        let stats = QuantizationSchemes.analyzeQuantizationError(original: v, quantized: [Int8](repeating: 0, count: v.count),
                                                                 params: params)
        return .value(Double(stats.maxAbsoluteError))
    }

    // MARK: Rows (every id is a string literal: the guard's R2 reads `row("<id>"` lexically)

    static let maxE: T16Expectation = { T16Oracle.reduction(.max, $0, empty: .equal(-.infinity)) }
    static let minE: T16Expectation = { T16Oracle.reduction(.min, $0, empty: .equal(.infinity)) }
    static let maxMagnitudeE: T16Expectation = { T16Oracle.reduction(.maxMagnitude, $0, empty: .positiveZeroOrEqual(0)) }
    static let minIndexE: T16Expectation = { T16Oracle.index(.min, $0) }
    static let maxIndexE: T16Expectation = { T16Oracle.index(.max, $0) }
    static let minElementE: T16Expectation = { T16Oracle.element(.min, $0) }
    static let maxElementE: T16Expectation = { T16Oracle.element(.max, $0) }
    static let minAB: T16Expectation = { T16Oracle.elementWise(.min, $0, T16Fixtures.partner($0.count)) }
    static let minBA: T16Expectation = { T16Oracle.elementWise(.min, T16Fixtures.partner($0.count), $0) }
    static let maxAB: T16Expectation = { T16Oracle.elementWise(.max, $0, T16Fixtures.partner($0.count)) }
    static let maxBA: T16Expectation = { T16Oracle.elementWise(.max, T16Fixtures.partner($0.count), $0) }
    static let clipE: T16Expectation = { T16Oracle.clip($0, low: clipLow, high: clipHigh) }
    static let softmaxE: T16Expectation = { .lanes([T16Expect](repeating: .nan, count: $0.count)) }

    /// Element-wise min/max of an array provider: the fixture with the partner operand, in either order.
    static func elementWise(_ p: any ArraySIMDProvider, _ fold: T16Oracle.Fold, swapped: Bool) -> T16Run {
        { v in
            let b = T16Fixtures.partner(v.count)
            let (x, y) = swapped ? (b, v) : (v, b)
            return lanes(fold == .min ? p.elementWiseMin(x, y) : p.elementWiseMax(x, y))
        }
    }

    static func arrayMax(_ p: any ArraySIMDProvider) -> T16Run { { .value(Double(p.max($0))) } }
    static func arrayMin(_ p: any ArraySIMDProvider) -> T16Run { { .value(Double(p.min($0))) } }
    static func arrayMinIndex(_ p: any ArraySIMDProvider) -> T16Run { { .index(p.minIndex($0)) } }
    static func arrayMaxIndex(_ p: any ArraySIMDProvider) -> T16Run { { .index(p.maxIndex($0)) } }
    static func arrayClip(_ p: any ArraySIMDProvider) -> T16Run { { lanes(p.clip($0, min: clipLow, max: clipHigh)) } }
    static func vector(_ p: (any ArraySIMDProvider)?, _ op: VectorOp) -> T16Run { { onVector($0, p, op) } }

    static let rows: [T16Row] = {
        var rows: [T16Row] = []
        // Swift SIMD pointer providers.
        rows += [
            row("SwiftFloatSIMDProvider.maximum", .reduction, .swift, expect: maxE,
                run: { pointer($0) { SwiftFloatSIMDProvider.maximum($0, count: $1) } }),
            row("SwiftFloatSIMDProvider.minimum", .reduction, .swift, expect: minE,
                run: { pointer($0) { SwiftFloatSIMDProvider.minimum($0, count: $1) } }),
            row("SwiftFloatSIMDProvider.maximumMagnitude", .reduction, .swift, expect: maxMagnitudeE,
                run: { pointer($0) { SwiftFloatSIMDProvider.maximumMagnitude($0, count: $1) } }),
            row("SwiftDoubleSIMDProvider.maximum", .reduction, .swift, expect: maxE,
                run: { pointerD($0) { SwiftDoubleSIMDProvider.maximum($0, count: $1) } }),
            row("SwiftDoubleSIMDProvider.minimum", .reduction, .swift, expect: minE,
                run: { pointerD($0) { SwiftDoubleSIMDProvider.minimum($0, count: $1) } }),
            row("SwiftDoubleSIMDProvider.maximumMagnitude", .reduction, .swift, expect: maxMagnitudeE,
                run: { pointerD($0) { SwiftDoubleSIMDProvider.maximumMagnitude($0, count: $1) } }),
            row("SwiftFloatSIMDProvider.clip", .elementWise, .swift, input: nonEmpty, expect: clipE,
                run: { clipF($0) { SwiftFloatSIMDProvider.clip($0, low: $1, high: $2, result: $3, count: $4) } }),
            row("SwiftDoubleSIMDProvider.clip", .elementWise, .swift, input: nonEmpty, expect: clipE,
                run: { clipD($0) { SwiftDoubleSIMDProvider.clip($0, low: $1, high: $2, result: $3, count: $4) } })
        ]
        #if canImport(Accelerate)
        // Accelerate (vDSP) pointer providers.
        rows += [
            row("AccelerateFloatProvider.maximum", .reduction, .accelerate, expect: maxE,
                run: { pointer($0) { AccelerateFloatProvider.maximum($0, count: $1) } }),
            row("AccelerateFloatProvider.minimum", .reduction, .accelerate, expect: minE,
                run: { pointer($0) { AccelerateFloatProvider.minimum($0, count: $1) } }),
            row("AccelerateFloatProvider.maximumMagnitude", .reduction, .accelerate, expect: maxMagnitudeE,
                run: { pointer($0) { AccelerateFloatProvider.maximumMagnitude($0, count: $1) } }),
            row("AccelerateDoubleProvider.maximum", .reduction, .accelerate, expect: maxE,
                run: { pointerD($0) { AccelerateDoubleProvider.maximum($0, count: $1) } }),
            row("AccelerateDoubleProvider.minimum", .reduction, .accelerate, expect: minE,
                run: { pointerD($0) { AccelerateDoubleProvider.minimum($0, count: $1) } }),
            row("AccelerateDoubleProvider.maximumMagnitude", .reduction, .accelerate, expect: maxMagnitudeE,
                run: { pointerD($0) { AccelerateDoubleProvider.maximumMagnitude($0, count: $1) } }),
            row("AccelerateFloatProvider.clip", .elementWise, .accelerate, input: nonEmpty, expect: clipE,
                run: { clipF($0) { AccelerateFloatProvider.clip($0, low: $1, high: $2, result: $3, count: $4) } }),
            row("AccelerateDoubleProvider.clip", .elementWise, .accelerate, input: nonEmpty, expect: clipE,
                run: { clipD($0) { AccelerateDoubleProvider.clip($0, low: $1, high: $2, result: $3, count: $4) } })
        ]
        let current: T16Tag = .accelerate
        #else
        let current: T16Tag = .swift
        #endif
        // SIMDOperations.FloatProvider/DoubleProvider: Accelerate on Apple platforms, the Swift providers elsewhere.
        rows += [
            row("SIMDOperations.FloatProvider.maximum", .reduction, current, expect: maxE,
                run: { pointer($0) { SIMDOperations.FloatProvider.maximum($0, count: $1) } }),
            row("SIMDOperations.FloatProvider.minimum", .reduction, current, expect: minE,
                run: { pointer($0) { SIMDOperations.FloatProvider.minimum($0, count: $1) } }),
            row("SIMDOperations.FloatProvider.maximumMagnitude", .reduction, current, expect: maxMagnitudeE,
                run: { pointer($0) { SIMDOperations.FloatProvider.maximumMagnitude($0, count: $1) } }),
            row("SIMDOperations.DoubleProvider.maximum", .reduction, current, expect: maxE,
                run: { pointerD($0) { SIMDOperations.DoubleProvider.maximum($0, count: $1) } }),
            row("SIMDOperations.DoubleProvider.minimum", .reduction, current, expect: minE,
                run: { pointerD($0) { SIMDOperations.DoubleProvider.minimum($0, count: $1) } }),
            row("SIMDOperations.DoubleProvider.maximumMagnitude", .reduction, current, expect: maxMagnitudeE,
                run: { pointerD($0) { SIMDOperations.DoubleProvider.maximumMagnitude($0, count: $1) } })
        ]
        // Array providers: the default, and the legacy wrapper that is Operations.simdProvider's default.
        let defaultArray = DefaultArraySIMDProvider(), legacy = SwiftSIMDProvider()
        rows += [
            row("DefaultArraySIMDProvider.max", .reduction, .swift, expect: maxE, run: arrayMax(defaultArray)),
            row("DefaultArraySIMDProvider.min", .reduction, .swift, expect: minE, run: arrayMin(defaultArray)),
            row("DefaultArraySIMDProvider.minIndex", .index, .swift, expect: minIndexE, run: arrayMinIndex(defaultArray)),
            row("DefaultArraySIMDProvider.maxIndex", .index, .swift, expect: maxIndexE, run: arrayMaxIndex(defaultArray)),
            row("DefaultArraySIMDProvider.elementWiseMin(a,b)", .elementWise, .swift, expect: minAB,
                run: elementWise(defaultArray, .min, swapped: false)),
            row("DefaultArraySIMDProvider.elementWiseMin(b,a)", .elementWise, .swift, expect: minBA,
                run: elementWise(defaultArray, .min, swapped: true)),
            row("DefaultArraySIMDProvider.elementWiseMax(a,b)", .elementWise, .swift, expect: maxAB,
                run: elementWise(defaultArray, .max, swapped: false)),
            row("DefaultArraySIMDProvider.elementWiseMax(b,a)", .elementWise, .swift, expect: maxBA,
                run: elementWise(defaultArray, .max, swapped: true)),
            row("DefaultArraySIMDProvider.clip", .elementWise, .swift, expect: clipE, run: arrayClip(defaultArray))
        ]
        rows += [
            row("SwiftSIMDProvider.max", .reduction, .swift, expect: maxE, run: arrayMax(legacy)),
            row("SwiftSIMDProvider.min", .reduction, .swift, expect: minE, run: arrayMin(legacy)),
            row("SwiftSIMDProvider.minIndex", .index, .swift, expect: minIndexE, run: arrayMinIndex(legacy)),
            row("SwiftSIMDProvider.maxIndex", .index, .swift, expect: maxIndexE, run: arrayMaxIndex(legacy)),
            row("SwiftSIMDProvider.elementWiseMin(a,b)", .elementWise, .swift, expect: minAB,
                run: elementWise(legacy, .min, swapped: false)),
            row("SwiftSIMDProvider.elementWiseMin(b,a)", .elementWise, .swift, expect: minBA,
                run: elementWise(legacy, .min, swapped: true)),
            row("SwiftSIMDProvider.elementWiseMax(a,b)", .elementWise, .swift, expect: maxAB,
                run: elementWise(legacy, .max, swapped: false)),
            row("SwiftSIMDProvider.elementWiseMax(b,a)", .elementWise, .swift, expect: maxBA,
                run: elementWise(legacy, .max, swapped: true)),
            row("SwiftSIMDProvider.clip", .elementWise, .swift, expect: clipE, run: arrayClip(legacy))
        ]
        // Vector<D> through Operations.simdProvider's default (the legacy Swift wrapper).
        rows += [
            row("Vector<D>.minElement[default]", .index, .swift, input: vectorSized, expect: minElementE,
                run: vector(nil, .minElement)),
            row("Vector<D>.maxElement[default]", .index, .swift, input: vectorSized, expect: maxElementE,
                run: vector(nil, .maxElement)),
            row("Vector<D>.min(_:)[default]", .elementWise, .swift, input: vectorSized, expect: minAB, run: vector(nil, .elementMin)),
            row("Vector<D>.max(_:)[default]", .elementWise, .swift, input: vectorSized, expect: maxAB, run: vector(nil, .elementMax)),
            row("Vector<D>.clamped(to:)[default]", .elementWise, .swift, input: vectorSized, expect: clipE, run: vector(nil, .clamped)),
            row("Vector<D>.clamp(to:)[default]", .elementWise, .swift, input: vectorSized, expect: clipE, run: vector(nil, .clamp)),
            // Control (unchanged by T16): softmax of a NaN input is NaN in every lane.
            row("Vector<D>.softmax[default]", .control, .swift, input: withNaN, expect: softmaxE, run: vector(nil, .softmax))
        ]
        #if canImport(Accelerate)
        let accelerateArray = AccelerateArraySIMDProvider()
        rows += [
            row("AccelerateArraySIMDProvider.max", .reduction, .accelerate, expect: maxE, run: arrayMax(accelerateArray)),
            row("AccelerateArraySIMDProvider.min", .reduction, .accelerate, expect: minE, run: arrayMin(accelerateArray)),
            row("AccelerateArraySIMDProvider.minIndex", .index, .accelerate, expect: minIndexE, run: arrayMinIndex(accelerateArray)),
            row("AccelerateArraySIMDProvider.maxIndex", .index, .accelerate, expect: maxIndexE, run: arrayMaxIndex(accelerateArray)),
            row("AccelerateArraySIMDProvider.elementWiseMin(a,b)", .elementWise, .accelerate, expect: minAB,
                run: elementWise(accelerateArray, .min, swapped: false)),
            row("AccelerateArraySIMDProvider.elementWiseMin(b,a)", .elementWise, .accelerate, expect: minBA,
                run: elementWise(accelerateArray, .min, swapped: true)),
            row("AccelerateArraySIMDProvider.elementWiseMax(a,b)", .elementWise, .accelerate, expect: maxAB,
                run: elementWise(accelerateArray, .max, swapped: false)),
            row("AccelerateArraySIMDProvider.elementWiseMax(b,a)", .elementWise, .accelerate, expect: maxBA,
                run: elementWise(accelerateArray, .max, swapped: true)),
            row("AccelerateArraySIMDProvider.clip", .elementWise, .accelerate, expect: clipE, run: arrayClip(accelerateArray))
        ]
        rows += [
            row("Vector<D>.minElement[accelerate]", .index, .accelerate, input: vectorSized, expect: minElementE,
                run: vector(accelerateArray, .minElement)),
            row("Vector<D>.maxElement[accelerate]", .index, .accelerate, input: vectorSized, expect: maxElementE,
                run: vector(accelerateArray, .maxElement)),
            row("Vector<D>.min(_:)[accelerate]", .elementWise, .accelerate, input: vectorSized, expect: minAB,
                run: vector(accelerateArray, .elementMin)),
            row("Vector<D>.max(_:)[accelerate]", .elementWise, .accelerate, input: vectorSized, expect: maxAB,
                run: vector(accelerateArray, .elementMax)),
            row("Vector<D>.clamped(to:)[accelerate]", .elementWise, .accelerate, input: vectorSized, expect: clipE,
                run: vector(accelerateArray, .clamped)),
            row("Vector<D>.clamp(to:)[accelerate]", .elementWise, .accelerate, input: vectorSized, expect: clipE,
                run: vector(accelerateArray, .clamp)),
            row("Vector<D>.softmax[accelerate]", .control, .accelerate, input: withNaN, expect: softmaxE,
                run: vector(accelerateArray, .softmax))
        ]
        #endif
        // VectorProtocol.min/max (scalar fold in the protocol extension); nil for an empty vector.
        rows += [
            row("VectorProtocol.min(DynamicVector)", .reduction, .scalar, expect: { T16Oracle.reduction(.min, $0, empty: .absent) },
                run: { vpMin(DynamicVector($0)) }),
            row("VectorProtocol.max(DynamicVector)", .reduction, .scalar, expect: { T16Oracle.reduction(.max, $0, empty: .absent) },
                run: { vpMax(DynamicVector($0)) }),
            row("VectorProtocol.min(Vector<D>)", .reduction, .scalar, input: vectorSized,
                expect: { T16Oracle.reduction(.min, $0, empty: .absent) }, run: vector(nil, .vpMin)),
            row("VectorProtocol.max(Vector<D>)", .reduction, .scalar, input: vectorSized,
                expect: { T16Oracle.reduction(.max, $0, empty: .absent) }, run: vector(nil, .vpMax)),
            row("VectorProtocol.min(Vector512Optimized)", .reduction, .scalar, input: padded512,
                expect: { T16Oracle.reduction(.min, $0, empty: .absent) }, run: { vpMin(try! Vector512Optimized($0)) }),
            row("VectorProtocol.max(Vector512Optimized)", .reduction, .scalar, input: padded512,
                expect: { T16Oracle.reduction(.max, $0, empty: .absent) }, run: { vpMax(try! Vector512Optimized($0)) })
        ]
        // DynamicVector element-wise and clamp (through Operations.simdProvider's default).
        rows += [
            row("DynamicVector.min(_:)", .elementWise, .swift, expect: minAB,
                run: { lanes(try! DynamicVector($0).min(DynamicVector(T16Fixtures.partner($0.count))).toArray()) }),
            row("DynamicVector.max(_:)", .elementWise, .swift, expect: maxAB,
                run: { lanes(try! DynamicVector($0).max(DynamicVector(T16Fixtures.partner($0.count))).toArray()) }),
            row("DynamicVector.clamped(to:)", .elementWise, .swift, expect: clipE,
                run: { lanes(DynamicVector($0).clamped(to: clipLow...clipHigh).toArray()) })
        ]
        // SIMDStorage.minMax (I1: an empty input gives (+inf, -inf)) and the statistics.
        rows += [
            row("SIMDStorage.minMax.min", .reduction, .scalar, input: blocksOfFour, expect: minE,
                run: { .value(Double(minMax($0).min)) }),
            row("SIMDStorage.minMax.max", .reduction, .scalar, input: blocksOfFour, expect: maxE,
                run: { .value(Double(minMax($0).max)) }),
            row("Operations.statistics.min", .statistic, .scalar, input: nonEmpty, expect: minE,
                run: { .value(Double(statistics($0).min[0])) }),
            row("Operations.statistics.max", .statistic, .scalar, input: nonEmpty, expect: maxE,
                run: { .value(Double(statistics($0).max[0])) }),
            row("Operations.statistics.magnitudes.min", .statistic, .scalar, input: nonEmpty,
                expect: { T16Oracle.reduction(.min, magnitudes($0), empty: .absent) },
                run: { .value(Double(statistics($0).magnitudes.min)) }),
            row("Operations.statistics.magnitudes.max", .statistic, .scalar, input: nonEmpty,
                expect: { T16Oracle.reduction(.max, magnitudes($0), empty: .absent) },
                run: { .value(Double(statistics($0).magnitudes.max)) }),
            row("QuantizationSchemes.analyzeQuantizationError.maxAbsoluteError", .statistic, .scalar,
                expect: { T16Oracle.seededMax($0.map { Swift.abs($0) }, seed: 0) }, run: quantizationMaxError),
            row("PrecisionValidator.AccuracyStatistics.maxAbsoluteError", .statistic, .scalar,
                expect: { T16Oracle.reduction(.max, $0, empty: .equal(0)) },
                run: { .value(Double(PrecisionValidator.AccuracyStatistics(absoluteErrors: $0, relativeErrors: []).maxAbsoluteError)) }),
            row("PrecisionValidator.AccuracyStatistics.maxRelativeError", .statistic, .scalar,
                expect: { T16Oracle.reduction(.max, $0, empty: .equal(0)) },
                run: { .value(Double(PrecisionValidator.AccuracyStatistics(absoluteErrors: [], relativeErrors: $0).maxRelativeError)) })
        ]
        return rows
    }()

    /// PerformanceProfiler running maxima, seeded with 0; each fixture is one recorded sequence (reset before and after).
    static let asyncRows: [T16AsyncRow] = [
        asyncRow("PerformanceProfiler.recordAccuracyLoss", .statistic, .scalar, expect: { T16Oracle.seededMax($0, seed: 0) },
                 run: { values in
                     let profiler = PerformanceProfiler.shared
                     await profiler.reset()
                     for x in values { await profiler.recordAccuracyLoss(x) }
                     let result = await profiler.getMetrics().accuracyLoss
                     await profiler.reset()
                     return .value(Double(result))
                 }),
        asyncRow("PerformanceProfiler.recordMemoryBandwidthUtilization", .statistic, .scalar,
                 expect: { T16Oracle.seededMax($0, seed: 0) },
                 run: { values in
                     let profiler = PerformanceProfiler.shared
                     await profiler.reset()
                     for x in values { await profiler.recordMemoryBandwidthUtilization(Double(x)) }
                     let result = await profiler.getMetrics().memoryBandwidthUtilization
                     await profiler.reset()
                     return .value(result)
                 })
    ]

    // MARK: Site map (parsed by Scripts/ci/tests/test_reduction_nan_guard.py)

    static let sites: [(site: String, coverage: T16SiteCoverage)] = [
        // BEGIN T16 REDUCTION SITES
        // Swift SIMD providers: a NaN flag selects NaN over the unchanged fold (S1, Z1, E1).
        ("Sources/VectorCore/Platform/SwiftFloatSIMDProvider.swift::SwiftFloatSIMDProvider.maximum::F1:result",
         .tabled("SwiftFloatSIMDProvider.maximum")),
        ("Sources/VectorCore/Platform/SwiftFloatSIMDProvider.swift::SwiftFloatSIMDProvider.maximum::F1:result",
         .tabled("SwiftFloatSIMDProvider.maximum")),
        ("Sources/VectorCore/Platform/SwiftFloatSIMDProvider.swift::SwiftFloatSIMDProvider.maximum::F1:result",
         .tabled("SwiftFloatSIMDProvider.maximum")),
        ("Sources/VectorCore/Platform/SwiftFloatSIMDProvider.swift::SwiftFloatSIMDProvider.maximum::F2:va",
         .tabled("SwiftFloatSIMDProvider.maximum")),
        ("Sources/VectorCore/Platform/SwiftFloatSIMDProvider.swift::SwiftFloatSIMDProvider.maximum::F2:va",
         .tabled("SwiftFloatSIMDProvider.maximum")),
        ("Sources/VectorCore/Platform/SwiftFloatSIMDProvider.swift::SwiftFloatSIMDProvider.minimum::F1:result",
         .tabled("SwiftFloatSIMDProvider.minimum")),
        ("Sources/VectorCore/Platform/SwiftFloatSIMDProvider.swift::SwiftFloatSIMDProvider.minimum::F1:result",
         .tabled("SwiftFloatSIMDProvider.minimum")),
        ("Sources/VectorCore/Platform/SwiftFloatSIMDProvider.swift::SwiftFloatSIMDProvider.minimum::F1:result",
         .tabled("SwiftFloatSIMDProvider.minimum")),
        ("Sources/VectorCore/Platform/SwiftFloatSIMDProvider.swift::SwiftFloatSIMDProvider.minimum::F2:va",
         .tabled("SwiftFloatSIMDProvider.minimum")),
        ("Sources/VectorCore/Platform/SwiftFloatSIMDProvider.swift::SwiftFloatSIMDProvider.minimum::F2:va",
         .tabled("SwiftFloatSIMDProvider.minimum")),
        ("Sources/VectorCore/Platform/SwiftFloatSIMDProvider.swift::SwiftFloatSIMDProvider.maximumMagnitude::F1:result",
         .tabled("SwiftFloatSIMDProvider.maximumMagnitude")),
        ("Sources/VectorCore/Platform/SwiftFloatSIMDProvider.swift::SwiftFloatSIMDProvider.maximumMagnitude::F1:result",
         .tabled("SwiftFloatSIMDProvider.maximumMagnitude")),
        ("Sources/VectorCore/Platform/SwiftFloatSIMDProvider.swift::SwiftFloatSIMDProvider.maximumMagnitude::F1:result",
         .tabled("SwiftFloatSIMDProvider.maximumMagnitude")),
        ("Sources/VectorCore/Platform/SwiftFloatSIMDProvider.swift::SwiftFloatSIMDProvider.maximumMagnitude::F2:absVa",
         .tabled("SwiftFloatSIMDProvider.maximumMagnitude")),
        ("Sources/VectorCore/Platform/SwiftFloatSIMDProvider.swift::SwiftFloatSIMDProvider.maximumMagnitude::F2:absVa",
         .tabled("SwiftFloatSIMDProvider.maximumMagnitude")),
        ("Sources/VectorCore/Platform/SwiftDoubleSIMDProvider.swift::SwiftDoubleSIMDProvider.maximum::F1:result",
         .tabled("SwiftDoubleSIMDProvider.maximum")),
        ("Sources/VectorCore/Platform/SwiftDoubleSIMDProvider.swift::SwiftDoubleSIMDProvider.maximum::F1:result",
         .tabled("SwiftDoubleSIMDProvider.maximum")),
        ("Sources/VectorCore/Platform/SwiftDoubleSIMDProvider.swift::SwiftDoubleSIMDProvider.maximum::F2:va",
         .tabled("SwiftDoubleSIMDProvider.maximum")),
        ("Sources/VectorCore/Platform/SwiftDoubleSIMDProvider.swift::SwiftDoubleSIMDProvider.minimum::F1:result",
         .tabled("SwiftDoubleSIMDProvider.minimum")),
        ("Sources/VectorCore/Platform/SwiftDoubleSIMDProvider.swift::SwiftDoubleSIMDProvider.minimum::F1:result",
         .tabled("SwiftDoubleSIMDProvider.minimum")),
        ("Sources/VectorCore/Platform/SwiftDoubleSIMDProvider.swift::SwiftDoubleSIMDProvider.minimum::F2:va",
         .tabled("SwiftDoubleSIMDProvider.minimum")),
        ("Sources/VectorCore/Platform/SwiftDoubleSIMDProvider.swift::SwiftDoubleSIMDProvider.maximumMagnitude::F1:result",
         .tabled("SwiftDoubleSIMDProvider.maximumMagnitude")),
        ("Sources/VectorCore/Platform/SwiftDoubleSIMDProvider.swift::SwiftDoubleSIMDProvider.maximumMagnitude::F1:result",
         .tabled("SwiftDoubleSIMDProvider.maximumMagnitude")),
        ("Sources/VectorCore/Platform/SwiftDoubleSIMDProvider.swift::SwiftDoubleSIMDProvider.maximumMagnitude::F2:absVa",
         .tabled("SwiftDoubleSIMDProvider.maximumMagnitude")),
        // Accelerate providers: the vDSP call, then an in-source NaN scan (vDSP's NaN behaviour is undocumented; C1).
        ("Sources/VectorCore/Platform/AccelerateSIMDProvider.swift::AccelerateFloatProvider.maximum::F5:vDSP_maxv",
         .tabled("AccelerateFloatProvider.maximum")),
        ("Sources/VectorCore/Platform/AccelerateSIMDProvider.swift::AccelerateFloatProvider.minimum::F5:vDSP_minv",
         .tabled("AccelerateFloatProvider.minimum")),
        ("Sources/VectorCore/Platform/AccelerateSIMDProvider.swift::AccelerateFloatProvider.maximumMagnitude::F5:vDSP_maxmgv",
         .tabled("AccelerateFloatProvider.maximumMagnitude")),
        ("Sources/VectorCore/Platform/AccelerateSIMDProvider.swift::AccelerateDoubleProvider.maximum::F5:vDSP_maxvD",
         .tabled("AccelerateDoubleProvider.maximum")),
        ("Sources/VectorCore/Platform/AccelerateSIMDProvider.swift::AccelerateDoubleProvider.minimum::F5:vDSP_minvD",
         .tabled("AccelerateDoubleProvider.minimum")),
        ("Sources/VectorCore/Platform/AccelerateSIMDProvider.swift::AccelerateDoubleProvider.maximumMagnitude::F5:vDSP_maxmgvD",
         .tabled("AccelerateDoubleProvider.maximumMagnitude")),
        // Index reductions: the first NaN's index.
        ("Sources/VectorCore/Platform/ArraySIMDProvider.swift::DefaultArraySIMDProvider.minIndex::F7:minVal",
         .tabled("DefaultArraySIMDProvider.minIndex")),
        ("Sources/VectorCore/Platform/ArraySIMDProvider.swift::DefaultArraySIMDProvider.maxIndex::F7:maxVal",
         .tabled("DefaultArraySIMDProvider.maxIndex")),
        // Element statistics. The statistics component folds, analyzeQuantizationError and the two PerformanceProfiler
        // maxima call ReductionNaN.propagatingMin/Max (not a fold shape, so no entry); they stay table rows.
        ("Sources/VectorCore/Protocols/SIMDStorage.swift::SIMDStorage.minMax::F1:globalMin",
         .tabled("SIMDStorage.minMax.min")),
        ("Sources/VectorCore/Protocols/SIMDStorage.swift::SIMDStorage.minMax::F2:simd",
         .tabled("SIMDStorage.minMax.min")),
        ("Sources/VectorCore/Protocols/SIMDStorage.swift::SIMDStorage.minMax::F1:globalMax",
         .tabled("SIMDStorage.minMax.max")),
        ("Sources/VectorCore/Protocols/SIMDStorage.swift::SIMDStorage.minMax::F2:simd",
         .tabled("SIMDStorage.minMax.max")),
        ("Sources/VectorCore/Protocols/VectorProtocol.swift::VectorProtocol.min::F1:minValue",
         .tabled("VectorProtocol.min(DynamicVector)")),
        ("Sources/VectorCore/Protocols/VectorProtocol.swift::VectorProtocol.max::F1:maxValue",
         .tabled("VectorProtocol.max(DynamicVector)")),
        ("Sources/VectorCore/Operations/Operations.swift::Operations.statistics::F2:magnitudes",
         .tabled("Operations.statistics.magnitudes.min")),
        ("Sources/VectorCore/Operations/Operations.swift::Operations.statistics::F2:magnitudes",
         .tabled("Operations.statistics.magnitudes.max")),
        ("Sources/VectorCore/Operations/Kernels/MixedPrecisionValidation.swift::AccuracyStatistics.maxAbsoluteError::F2:absoluteErrors",
         .tabled("PrecisionValidator.AccuracyStatistics.maxAbsoluteError")),
        ("Sources/VectorCore/Operations/Kernels/MixedPrecisionValidation.swift::AccuracyStatistics.maxRelativeError::F2:relativeErrors",
         .tabled("PrecisionValidator.AccuracyStatistics.maxRelativeError")),
        // T15-owned folds: a NaN check ahead of the fold, pinned by T15's metric table through chebyshevDistance(to:).
        ("Sources/VectorCore/Protocols/VectorProtocol.swift::VectorProtocol.lInfinityNorm::F1:maxValue",
         // swiftlint:disable:next line_length
         .external("Tests/ComprehensiveTests/T15CosineExceptionalTableTests.swift::T15MetricExceptionalTableTests.everyMetricPinsItsExceptionalClass")),
        ("Sources/VectorCore/Operations/VectorMath.swift::Vector.lInfinityNorm::F2:absArray",
         // swiftlint:disable:next line_length
         .external("Tests/ComprehensiveTests/T15CosineExceptionalTableTests.swift::T15MetricExceptionalTableTests.everyMetricPinsItsExceptionalClass")),
        ("Sources/VectorCore/Operations/DistanceMetrics.swift::ChebyshevDistance.distance::F1:result",
         // swiftlint:disable:next line_length
         .external("Tests/ComprehensiveTests/T15CosineExceptionalTableTests.swift::T15MetricExceptionalTableTests.everyMetricPinsItsExceptionalClass")),
        // Safe: the operands are checked finite before the fold, or the NaN is handled explicitly.
        ("Sources/VectorCore/Protocols/VectorProtocol.swift::VectorProtocol.magnitude::F1:maxAbs",
         .exempt("finite-guarded: a NaN element sets hasNaN and is skipped; the property then returns NaN (T15-touched body)")),
        ("Sources/VectorCore/Operations/Kernels/NormalizationCore.swift::NormalizationCore.plan::F1:scale",
         .exempt("finite-guarded: guard x.isFinite returns a failure before the fold reads x")),
        ("Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift::SwiftLinearAlgebraProvider.offDiagonalMaximum::F1:result",
         .exempt("finite-guarded: a non-finite value returns .infinity before the fold reads it")),
        ("Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift::SwiftLinearAlgebraProvider.norm::F7:scale",
         .exempt("finite-guarded: guard value.isFinite throws before the fold reads value (a Double scaled-norm accumulator)")),
        ("Sources/VectorCore/Factory/VectorTypeFactory.swift::VectorTypeFactory.optimalDimension::F2:supportedDimensions",
         .exempt("int: Sequence.min(by:) over a literal [Int] of supported dimensions")),
        ("Sources/VectorCore/LinearAlgebra/PCA.swift::PCAModel.fit::F7:maxAbs",
         .exempt("finite-guarded: requireFiniteInput first; an SVD NaN the comparison skips fails requireFiniteResult(components)")),
        ("Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift::FuzzySimplicialSet.smoothDistances::F1:nearest",
         .exempt("finite-guarded: globalMean.isFinite (checked first) implies finite distances; the fold takes distance > 0 only")),
        ("Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift::FuzzySimplicialSet.smoothDistances::F2:deltas",
         .exempt("finite-guarded: deltas are max(0, d - rho) over distances that globalMean.isFinite shows are finite")),
        ("Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift::FuzzySimplicialSet.smoothDistances::F2:logs",
         .exempt("finite-guarded: the same guard requires logs.allSatisfy(isFinite)")),
        ("Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift::FuzzySimplicialSet.build::F1:forward",
         .exempt("finite-guarded: weights come from membership(), which throws unless the weight is finite and in [0, 1]")),
        ("Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift::FuzzySimplicialSet.build::F1:backward",
         .exempt("finite-guarded: weights come from membership(), which throws unless the weight is finite and in [0, 1]")),
        ("Sources/VectorCore/ManifoldLearning/UMAP.swift::Operations.pcaInitialCoordinates::F1:maxAbs",
         .exempt("finite-guarded: the PCA projection passes requireFiniteResult before the fold")),
        ("Sources/VectorCore/ManifoldLearning/UMAP.swift::UMAPLayout.fitABParams::F1:deviation",
         .exempt("finite-guarded: logistic() of an ell checked finite here minus curve values that evaluateCurve built from finite ells")),
        ("Sources/VectorCore/ManifoldLearning/UMAP.swift::UMAPLayout.optimize::F1:maxWeight",
         .exempt("finite-guarded: guard weight.isFinite, weight >= 0, weight <= 1 throws before the fold")),
        ("Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift::MixedPrecisionKernels.analyzePrecision::F1:minVal",
         .exempt("finite-guarded: the fold runs only inside if v.isFinite")),
        ("Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift::MixedPrecisionKernels.analyzePrecision::F1:maxVal",
         .exempt("finite-guarded: the fold runs only inside if v.isFinite")),
        ("Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift::MixedPrecisionBenchmark.measureAccuracy512::F2:relativeErrors",
         .exempt("by-construction: internal benchmark; SeededRNG coordinates in [-1, 1] give finite 512-wide dot products")),
        ("Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift::MixedPrecisionBenchmark.measureAccuracy512::F2:absoluteErrors",
         .exempt("by-construction: internal benchmark; SeededRNG coordinates in [-1, 1] give finite 512-wide dot products")),
        // swiftlint:disable:next line_length
        ("Sources/VectorCore/Operations/Kernels/MixedPrecisionValidation.swift::PrecisionValidator.validateBatchDistanceAccuracy::F1:maxError",
         .exempt("finite-guarded: both operands are checked finite, and distanceError of two finite Floats in Double is finite")),
        ("Sources/VectorCore/Operations/Kernels/PrecisionTuningSupport.swift::PrecisionTuningSupport.errors::F1:maximum",
         .exempt("finite-guarded: guard result.isFinite && baseline.isFinite and guard error.isFinite return nil before the fold")),
        ("Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift::Vector512INT8.init::F2:array",
         .exempt("finite-guarded: the initializer throws invalidValues for any non-finite coordinate before min()")),
        ("Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift::Vector512INT8.init::F2:array",
         .exempt("finite-guarded: the initializer throws invalidValues for any non-finite coordinate before max()")),
        ("Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift::Vector768INT8.init::F2:array",
         .exempt("finite-guarded: the initializer throws invalidValues for any non-finite coordinate before min()")),
        ("Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift::Vector768INT8.init::F2:array",
         .exempt("finite-guarded: the initializer throws invalidValues for any non-finite coordinate before max()")),
        ("Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift::Vector1536INT8.init::F2:array",
         .exempt("finite-guarded: the initializer throws invalidValues for any non-finite coordinate before min()")),
        ("Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift::Vector1536INT8.init::F2:array",
         .exempt("finite-guarded: the initializer throws invalidValues for any non-finite coordinate before max()")),
        // Calibration min/max capture: re-tagged under T16 (T15 addendum concern 3); fatal on a non-finite value.
        ("Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift::QuantizationCalibrator.calibrate::F7:minimum",
         .exempt("finite-guarded: guard value.isFinite else fatalError precedes the comparison fold (T16 calibration min/max)")),
        ("Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift::QuantizationCalibrator.calibrate::F7:maximum",
         .exempt("finite-guarded: guard value.isFinite else fatalError precedes the comparison fold (T16 calibration min/max)")),
        ("Sources/VectorCore/Quantization/QuantizationSchemes.swift::QuantizationSchemes.computePerVectorParams::F2:values",
         .exempt("finite-guarded: the only caller, computeQuantizationParams, runs AffineQuantizationCore.requireFinite(values) first")),
        ("Sources/VectorCore/Quantization/QuantizationSchemes.swift::QuantizationSchemes.computePerVectorParams::F2:values",
         .exempt("finite-guarded: the only caller, computeQuantizationParams, runs AffineQuantizationCore.requireFinite(values) first"))
        // END T16 REDUCTION SITES
    ]
}
