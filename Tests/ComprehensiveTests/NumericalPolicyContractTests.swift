import Foundation
import Testing
@testable import VectorCore

private struct PolicyDoubleVector: VectorProtocol {
    var storage: [Double]
    var forbidBorrow = false
    subscript(index: Int) -> Double { storage[index] }
    var scalarCount: Int { storage.count }
    init() { storage = [] }
    init(_ values: [Double]) { storage = values }
    init(repeating value: Double) { storage = [value] }
    func toArray() -> [Double] { storage }
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Double>) throws -> R) rethrows -> R {
        #expect(!forbidBorrow)
        return try storage.withUnsafeBufferPointer(body)
    }
    mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Double>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer { try body($0) }
    }
}

private enum PolicyDim0: StaticDimension { static let value = 0 }
private enum PolicyDim1: StaticDimension { static let value = 1 }
private enum PolicyDim5: StaticDimension { static let value = 5 }

@Suite("Numerical policy contracts")
struct NumericalPolicyContractTests {
    private let tolerances: [Float] = [-1, -0.0, 0, 0.001, .greatestFiniteMagnitude, .nan, .infinity, -.infinity]
    private let values: [Float] = [0, -0.0, 1, -1, 1.0005, .greatestFiniteMagnitude, -.greatestFiniteMagnitude, .nan, .infinity, -.infinity]

    private func equalityOracle(_ a: [Float], _ b: [Float], _ tolerance: Float) -> Bool {
        guard tolerance.isFinite, tolerance >= 0, a.count == b.count else { return false }
        return zip(a, b).allSatisfy { $0.isFinite && $1.isFinite && abs($0 - $1) <= tolerance }
    }

    @Test func publicDynamicAndDoubleEquality() {
        for n in [0, 1, 3, 4, 5] {
            for a in values { for b in values { for t in tolerances {
                let lhs = Array(repeating: a, count: n), rhs = Array(repeating: b, count: n)
                #expect(DynamicVector(lhs).isApproximatelyEqual(to: DynamicVector(rhs), tolerance: t) == equalityOracle(lhs, rhs, t))
                let da = lhs.map(Double.init), db = rhs.map(Double.init), dt = Double(t)
                let expected = dt.isFinite && dt >= 0 && zip(da, db).allSatisfy { $0.isFinite && $1.isFinite && abs($0 - $1) <= dt }
                #expect(PolicyDoubleVector(da).isApproximatelyEqual(to: PolicyDoubleVector(db), tolerance: dt) == expected)
            } } }
        }
        var a = PolicyDoubleVector([1]); a.forbidBorrow = true
        var b = PolicyDoubleVector([1, 2]); b.forbidBorrow = true
        #expect(!a.isApproximatelyEqual(to: b))
        #expect(!b.isApproximatelyEqual(to: a))
        #expect(!PolicyDoubleVector([Double.greatestFiniteMagnitude]).isApproximatelyEqual(to: PolicyDoubleVector([-Double.greatestFiniteMagnitude]), tolerance: .greatestFiniteMagnitude))
        #expect(PolicyDoubleVector([0]).isApproximatelyEqual(to: PolicyDoubleVector([-0.0]), tolerance: -0.0))
    }

    private func checkWidth<V: VectorProtocol>(_ type: V.Type, count: Int, validator: ((V, V, Float) -> Bool)? = nil, conversion: ((V, Float) -> Bool)? = nil) throws where V.Scalar == Float {
        for a in values { for b in values { for t in tolerances {
            let lhs = Array(repeating: a, count: count), rhs = Array(repeating: b, count: count)
            let v1 = try V(lhs), v2 = try V(rhs)
            let expected = equalityOracle(lhs, rhs, t)
            #expect(v1.isApproximatelyEqual(to: v2, tolerance: t) == expected)
            if let validator { #expect(validator(v1, v2, t) == expected) }
            // Float16 does not exist on Intel macOS / Mac Catalyst; the finite round-trip oracle is arm64/iOS-family only.
            #if !(arch(x86_64) && (os(macOS) || targetEnvironment(macCatalyst)))
            if let conversion {
                let restored = lhs.map { Float(Float16($0)) }
                #expect(conversion(v1, t) == equalityOracle(lhs, restored, t))
            }
            #endif
        } } }
        for index in Set([0, 1, 2, 3, count / 2, count - 1]).filter({ $0 >= 0 && $0 < count }) {
            for bad: Float in [.nan, .infinity, -.infinity] {
                var lhs = Array(repeating: Float(1), count: count); lhs[index] = bad
                let v1 = try V(lhs), v2 = try V(Array(repeating: 1, count: count))
                #expect(!v1.isApproximatelyEqual(to: v2))
                #expect(!v2.isApproximatelyEqual(to: v1))
                if let validator { #expect(!validator(v1, v2, 1)); #expect(!validator(v2, v1, 1)) }
                if let conversion { #expect(!conversion(v1, 1)) }
                #expect(v1.toArray().map(\.bitPattern) == lhs.map(\.bitPattern))
            }
        }
    }
    @Test func fixedAndOptimizedEquality() throws {
        for count in [0, 1, 3, 4, 5] { try checkWidth(DynamicVector.self, count: count) }
        try checkWidth(Vector<PolicyDim0>.self, count: 0)
        try checkWidth(Vector<PolicyDim1>.self, count: 1)
        try checkWidth(Vector<PolicyDim5>.self, count: 5)
        try checkWidth(Vector<Dim3>.self, count: 3)
        try checkWidth(Vector<Dim4>.self, count: 4)
        try checkWidth(Vector384Optimized.self, count: 384)
        try checkWidth(Vector512Optimized.self, count: 512, validator: PrecisionValidator.validateVectorEquality512,
            conversion: { PrecisionValidator.validateConversion512(original: $0, converted: .init(from: $0), tolerance: $1) })
        try checkWidth(Vector768Optimized.self, count: 768, validator: PrecisionValidator.validateVectorEquality768,
            conversion: { PrecisionValidator.validateConversion768(original: $0, converted: .init(from: $0), tolerance: $1) })
        try checkWidth(Vector1536Optimized.self, count: 1536, validator: PrecisionValidator.validateVectorEquality1536,
            conversion: { PrecisionValidator.validateConversion1536(original: $0, converted: .init(from: $0), tolerance: $1) })
    }

    private func accuracyOracle(_ a: Float, _ b: Float) -> Double {
        abs(Double(a) - Double(b)) / max(abs(Double(a)), Double(Float(1e-8)))
    }
    @Test func distanceAccuracy() {
        for a in values { for b in values { for t in tolerances {
            let validInput = a.isFinite && b.isFinite && t.isFinite && t >= 0
            let expected = validInput && accuracyOracle(a, b) <= Double(t)
            #expect(PrecisionValidator.validateDistanceAccuracy(referenceFP32: a, computedFP16: b, tolerance: t) == expected)
            let batch = PrecisionValidator.validateBatchDistanceAccuracy(referenceFP32: [a], computedFP16: [b], tolerance: t)
            #expect(batch.allValid == expected)
            #expect(batch.maxError == (validInput ? Float(accuracyOracle(a, b)) : .infinity))
            #expect(batch.meanError == (validInput ? Float(accuracyOracle(a, b)) : .infinity))
        } } }
        for t in tolerances {
            let empty = PrecisionValidator.validateBatchDistanceAccuracy(referenceFP32: [], computedFP16: [], tolerance: t)
            #expect(empty.allValid == (t.isFinite && t >= 0))
            #expect(empty.maxError == (empty.allValid ? 0 : .infinity))
            #expect(empty.meanError == (empty.allValid ? 0 : .infinity))
        }
        for i in 0..<5 { for bad: Float in [.nan, .infinity, -.infinity] {
            var a: [Float] = [1, 2, 3, 4, 5]; a[i] = bad
            for flip in [false, true] {
                let r = PrecisionValidator.validateBatchDistanceAccuracy(referenceFP32: flip ? [1, 2, 3, 4, 5] : a, computedFP16: flip ? a : [1, 2, 3, 4, 5])
                #expect(!r.allValid && r.maxError == .infinity && r.meanError == .infinity)
            }
        } }
        let mismatch = PrecisionValidator.validateBatchDistanceAccuracy(referenceFP32: [1], computedFP16: [])
        #expect(!mismatch.allValid && mismatch.maxError == .infinity && mismatch.meanError == .infinity)
        for t: Float in [Float(0.5).nextDown, Float(0.5), Float(0.5).nextUp] {
            #expect(PrecisionValidator.validateDistanceAccuracy(referenceFP32: 2, computedFP16: 3, tolerance: t) == (t >= 0.5))
        }
        let refs: [Float] = [.greatestFiniteMagnitude, .greatestFiniteMagnitude, 0, 2]
        let comps: [Float] = [-.greatestFiniteMagnitude, -.greatestFiniteMagnitude, 1e-8, 3]
        let errors = zip(refs, comps).map(accuracyOracle)
        let r = PrecisionValidator.validateBatchDistanceAccuracy(referenceFP32: refs, computedFP16: comps, tolerance: 2)
        #expect(r.allValid && r.maxError == 2 && r.meanError == Float(errors.reduce(0, +) / 4))
    }
}

extension NumericalPolicyContractTests {
    private enum LogOutcome {
        case values([Float])
        case nonfinite([Int], [Int], [Int])
        case domain(Int)
    }
    // Independent staged table, expressed using raw bit positions rather than production helpers.
    private func logOracle(_ input: [Float], _ flags: Int) -> LogOutcome {
        let nan = input.indices.filter { input[$0].isNaN }
        let pos = input.indices.filter { input[$0] == .infinity }
        let neg = input.indices.filter { input[$0] == -.infinity }
        if flags & 16 != 0 && !(nan + pos + neg).isEmpty { return .nonfinite(nan, pos, neg) }
        var staged = input
        if flags & 8 != 0 && !nan.isEmpty { staged = Array(repeating: .nan, count: input.count) }
        else {
            for i in nan where flags & 1 != 0 { staged[i] = 0 }
            for i in pos where flags & 2 != 0 { staged[i] = .greatestFiniteMagnitude }
            for i in neg where flags & 4 != 0 { staged[i] = -.greatestFiniteMagnitude }
        }
        var result: [Float] = []
        for (i, x) in staged.enumerated() {
            if x <= 0 {
                if flags & 16 != 0 { return .domain(i) }
                if flags & 8 != 0 { return .values(Array(repeating: .nan, count: input.count)) }
                result.append(flags & 1 != 0 ? 0 : -.greatestFiniteMagnitude)
            } else { result.append(Float(Foundation.log(Double(x)))) }
        }
        return .values(result)
    }
    private func checkLog<D: StaticDimension>(_ type: D.Type) throws {
        let scalars: [Float] = [1, 2, .leastNonzeroMagnitude, .greatestFiniteMagnitude, 0, -0.0, -1, .nan, .infinity, -.infinity]
        var inputs = scalars.map { Array(repeating: $0, count: D.value) }
        for index in Set([0, 1, 2, 3, D.value / 2, D.value - 1]).filter({ $0 < D.value }) {
            for scalar in scalars {
                var input = Array(repeating: Float(2), count: D.value); input[index] = scalar
                inputs.append(input)
            }
        }
        var mixed = Array(repeating: Float(2), count: D.value)
        mixed[0] = -1; mixed[D.value - 1] = .nan; inputs.append(mixed)
        mixed[0] = .nan; mixed[1] = .infinity; mixed[D.value - 1] = -.infinity; inputs.append(mixed)
        for input in inputs { for raw in 0..<32 { for unknown in [0, 1 << 12] {
            let vector = try Vector<D>(input)
            defer { #expect(vector.toArray().map(\.bitPattern) == input.map(\.bitPattern)) }
            let expected = logOracle(input, raw)
            do {
                let result = try vector.safeLog(options: .init(rawValue: raw | unknown)).toArray()
                guard case .values(let values) = expected else { Issue.record("Expected staged error for flags \(raw)"); continue }
                for (actual, target) in zip(result, values) {
                    if target.isNaN { #expect(actual.isNaN) }
                    else if target.isInfinite || target == 0 || target == -.greatestFiniteMagnitude { #expect(actual.bitPattern == target.bitPattern) }
                    else { #expect(actual.isFinite); #expect(abs(Double(actual) - Double(target)) <= Double(target.ulp) * 2) }
                }
            } catch let error as NonFiniteError {
                guard case .nonfinite(let n, let p, let m) = expected else { Issue.record("Unexpected nonfinite error"); continue }
                switch error {
                case .nanDetected(let indices): #expect(indices == n && p.isEmpty && m.isEmpty)
                case .infinityDetected(let indices): #expect(indices == p && n.isEmpty && m.isEmpty)
                case .negativeInfinityDetected(let indices): #expect(indices == m && n.isEmpty && p.isEmpty)
                case .mixedNonFiniteValues(let ni, let pi, let mi):
                    #expect(ni == n && pi == p && mi == m)
                    #expect([n, p, m].filter { !$0.isEmpty }.count > 1)
                }
            } catch let error as VectorError {
                guard case .domain(let i) = expected else { Issue.record("Unexpected domain error"); continue }
                #expect(error.kind == .invalidData)
                #expect(error.context.additionalInfo["indices"] == String(i))
                #expect(error.context.additionalInfo["reason"] == "Logarithm of non-positive value")
            } catch { Issue.record("Unexpected error: \(error)") }
            #expect(vector.toArray().map(\.bitPattern) == input.map(\.bitPattern))
        } } }
        let defaultInput = try Vector<D>(Array(repeating: Float.nan, count: D.value))
        #expect(try defaultInput.safeLog().toArray().allSatisfy { $0 == 0 })
    }
    @Test func safeLogFlagTable() throws {
        try checkLog(Dim3.self)
        try checkLog(Dim512.self)
    }
}
