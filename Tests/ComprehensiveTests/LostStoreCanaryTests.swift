import Testing
import VectorCore

// Canary for a Swift 6.4.0 optimizer bug (T15 lost-store investigation). At -O, SIL RedundantLoadElimination
// forwards the value an array literal stored across a `withUnsafeMutableBufferPointer` call whose closure stays out
// of line and receives the buffer `inout`, so an element read after the call returns the literal although memory
// holds the closure's write. Upstream: swiftlang/swift#92562 (rdar://188227399); PR #92580 and its 6.4.x backports
// (#92630, #92631, #92632) are believed to fix this mechanism (unverified for this form; see T15/lost-store-investigation).
//
// The reproducer is the investigation's pure-Swift form (ladder/d_pure.swift): no VectorCore code is involved.
// (The second suite below, AdaptiveReducedResultTests, covers the library site hardened against this bug.)
// Expectations by build:
// - Debug (`DEBUG` defined, -Onone; RedundantLoadElimination does not run): the store is observed, on every toolchain.
// - Release, compiler >= 6.4.1: the store must be observed. A lost store means the toolchain lacks the fix, and the
//   test fails, so a missing backport is signalled.
// - Release, Swift 6.4.0 (`compiler(>=6.4) && !compiler(>=6.4.1)`): the bug is expected. A lost store is recorded as a
//   known issue; `isIntermittent: true` keeps the test passing if this build does not reproduce it.
// - Release, compiler < 6.4 (CI runs 6.0 and 6.2; whether they have the bug is not known): a lost store is recorded as
//   a known issue and never fails the test, and the test never asserts that the bug is present.
// Scripts/ci/tests/test_lost_store_reads.py allowlists the deliberate read below and rejects the shape elsewhere.

@inline(never)
func lostStoreCanaryWriteOne(_ out: inout UnsafeMutableBufferPointer<Float>) {
    out[0] = 1
}

/// The miscompiled shape on purpose: an array literal, an out-of-line closure that writes through the `inout` buffer,
/// then a direct element read after the call.
@inline(never)
func lostStoreCanaryObservedValue() -> Float {
    var values: [Float] = [2]
    values.withUnsafeMutableBufferPointer(lostStoreCanaryWriteOne)
    return values[0]
}

@Suite("Swift 6.4.0 lost-store canary (swiftlang/swift#92562)")
struct LostStoreCanaryTests {
    @Test func literalArrayStoreIsObservedAfterOutOfLineClosure() {
        let observed = lostStoreCanaryObservedValue()
        let detail = "values[0] after the closure: \(observed) (bits 0x\(String(observed.bitPattern, radix: 16))), expected 1"
        #if DEBUG
        #expect(observed == 1, Comment(rawValue: detail))
        #elseif compiler(>=6.4.1)
        #expect(observed == 1, Comment(rawValue: "toolchain without the swiftlang/swift#92562 fix (PR #92580): " + detail))
        #elseif compiler(>=6.4)
        withKnownIssue("Swift 6.4.0 RedundantLoadElimination lost store, swiftlang/swift#92562", isIntermittent: true) {
            #expect(observed == 1, Comment(rawValue: detail))
        }
        #else
        withKnownIssue("lost store on a pre-6.4 toolchain (swiftlang/swift#92562 class)", isIntermittent: true) {
            #expect(observed == 1, Comment(rawValue: detail))
        }
        #endif
    }
}

// CAPTURED-VAR SHAPES (T15 final fix, M5). The shapes below write a captured `var` inside a `withUnsafe...` closure
// and read it after the call: the form Scripts/ci/tests/test_lost_store_reads.py flags as `captured:` and allowlists
// for 74 library and test sites, citing these tests. K1-K5 are the fix-round-1 probe's shapes; K7 (captured
// array-element write, as in KNNGraph.bruteForce) and K8 (mutable-buffer closure with a captured counter, as in
// StorageAllocationContractTests.checkBorrow) are the two shapes the probe did not cover. Each is a separate
// `@inline(never)` function so the closure and the read share one body, as in the real sites. Where a shape kept its
// store on Swift 6.4.0 the correct value is asserted on every toolchain and build. All of K1-K5, K7 and K8 did
// (Release run on 6.4.0, T15/final-fix/01.log, where the canary above lost its store in the same run), so none needs
// a `withKnownIssue`; if one fails on a later toolchain, gate it like the canary and revisit its allowlist entries. These tests are the in-repo
// evidence that justifies the allowlist; they do not prove the form safe for other bodies, since the trigger follows
// inlining.

@inline(never) func lostStoreCanaryWriteInout(_ x: inout Float, _ v: Float) { x = v }
@inline(never) func lostStoreCanaryWritePointer(_ out: inout UnsafeMutablePointer<Float>) { out.pointee = 1 }
@inline(never) func lostStoreCanaryDot(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ n: Int) -> Float {
    var s: Float = 0
    for i in 0..<n { s += a[i] * b[i] }
    return s
}

protocol LostStoreCanaryBufferBacked {
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R
}

struct LostStoreCanaryBacked: LostStoreCanaryBufferBacked {
    var storage: [Float]
    @inline(never) func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }
    @inline(never) mutating func withUnsafeMutableBufferPointer<R>(
        _ body: (inout UnsafeMutableBufferPointer<Float>) throws -> R
    ) rethrows -> R {
        try storage.withUnsafeMutableBufferPointer(body)
    }
}

extension LostStoreCanaryBufferBacked {
    /// K2: the pre-W2 `VectorProtocol.lInfinityNorm` (captured `maxValue` and `hasNaN`).
    var lostStoreCanaryCapturedLInf: Float {
        var maxValue: Float = 0
        var hasNaN = false
        withUnsafeBufferPointer { buffer in
            for element in buffer {
                if element.isNaN { hasNaN = true; break }
                maxValue = Swift.max(maxValue, Swift.abs(element))
            }
        }
        return hasNaN ? .nan : maxValue
    }
    /// K3: a captured var passed `inout` to an out-of-line callee inside the closure.
    var lostStoreCanaryCapturedInout: Float {
        var m: Float = 0
        withUnsafeBufferPointer { b in lostStoreCanaryWriteInout(&m, b[1]) }
        return m
    }
    /// K4: a captured var whose pointer is passed `inout` to an out-of-line writer inside the closure.
    var lostStoreCanaryCapturedPointer: Float {
        var m: Float = 2
        withUnsafeBufferPointer { _ in
            withUnsafeMutablePointer(to: &m) { p in
                var q = p
                lostStoreCanaryWritePointer(&q)
            }
        }
        return m
    }
}

/// K1: a captured var written in a concrete `Array.withUnsafeBufferPointer` closure.
@inline(never) func lostStoreCanaryK1(_ a: [Float]) -> Float {
    var m: Float = 0
    a.withUnsafeBufferPointer { b in for e in b { m = Swift.max(m, Swift.abs(e)) } }
    return m
}
@inline(never) func lostStoreCanaryK2(_ s: LostStoreCanaryBacked) -> Float { s.lostStoreCanaryCapturedLInf }
@inline(never) func lostStoreCanaryK2Generic<T: LostStoreCanaryBufferBacked>(_ s: T) -> Float {
    s.lostStoreCanaryCapturedLInf
}
@inline(never) func lostStoreCanaryK3(_ s: LostStoreCanaryBacked) -> Float { s.lostStoreCanaryCapturedInout }
@inline(never) func lostStoreCanaryK4(_ s: LostStoreCanaryBacked) -> Float { s.lostStoreCanaryCapturedPointer }
/// K5: the DistanceMetrics witness shape: nested closures, a captured var assigned from a by-value-pointer kernel.
@inline(never) func lostStoreCanaryK5(_ a: [Float], _ b: [Float]) -> Float {
    var dot: Float = 0
    a.withUnsafeBufferPointer { ab in
        b.withUnsafeBufferPointer { bb in dot = lostStoreCanaryDot(ab.baseAddress!, bb.baseAddress!, ab.count) }
    }
    return dot
}
/// K7: a captured array whose elements are written in the closure and read after it (KNNGraph.bruteForce).
@inline(never) func lostStoreCanaryK7(_ flat: [Float], _ n: Int) -> Float {
    var norms = [Float](repeating: 0, count: n)
    flat.withUnsafeBufferPointer { fb in
        for i in 0..<n { norms[i] += fb[i] * fb[i] }
    }
    return norms[0] + norms[1]
}
/// K8: a mutable-buffer closure with a captured counter, read after the call (StorageAllocationContractTests.checkBorrow).
@inline(never) func lostStoreCanaryK8(_ s: LostStoreCanaryBacked) -> Float {
    var box = s
    var calls = 0
    let result = box.withUnsafeMutableBufferPointer { buffer in
        calls += 1
        for i in buffer.indices { buffer[i] = Float(i + 11) }
        return 731
    }
    return Float(calls) * 1000 + Float(result)
}

@Suite("Captured-var shapes keep their store (allowlist evidence for test_lost_store_reads.py)")
struct LostStoreCanaryCapturedVarTests {
    private let backed = LostStoreCanaryBacked(storage: [1, -3, 2])

    @Test func k1ConcreteArrayClosure() {
        let observed = lostStoreCanaryK1([1, -3, 2])
        #expect(observed == 3, Comment(rawValue: "K1 observed \(observed), expected 3"))
    }
    @Test func k2ProtocolRequirementOldLInfinityNorm() {
        let observed = lostStoreCanaryK2(backed), generic = lostStoreCanaryK2Generic(backed)
        #expect(observed == 3, Comment(rawValue: "K2 observed \(observed), expected 3"))
        #expect(generic == 3, Comment(rawValue: "K2 generic caller observed \(generic), expected 3"))
    }
    @Test func k3InoutToOutOfLineCallee() {
        let observed = lostStoreCanaryK3(backed)
        #expect(observed == -3, Comment(rawValue: "K3 observed \(observed), expected -3"))
    }
    @Test func k4PointerInoutToOutOfLineWriter() {
        let observed = lostStoreCanaryK4(backed)
        #expect(observed == 1, Comment(rawValue: "K4 observed \(observed), expected 1"))
    }
    @Test func k5NestedClosuresByValuePointerKernel() {
        let observed = lostStoreCanaryK5([1, -3, 2], [2, -3, 1])
        #expect(observed == 13, Comment(rawValue: "K5 observed \(observed), expected 13"))
    }
    @Test func k7CapturedArrayElementWrite() {
        let observed = lostStoreCanaryK7([3, 4], 2)
        #expect(observed == 25, Comment(rawValue: "K7 observed \(observed), expected 25"))
    }
    @Test func k8MutableBufferClosureCapturedCounter() {
        let observed = lostStoreCanaryK8(backed)
        #expect(observed == 1731, Comment(rawValue: "K8 observed \(observed), expected 1731 (calls 1, result 731)"))
    }
}

/// Library hardening (fix round 1, ruling 3): `MixedPrecisionKernels.adaptiveEuclideanDistance` reads its FP16 result
/// inside the `withUnsafeMutableBufferPointer` closure. If that read were stale (the lost-store shape), the reduced
/// value would be the literal 0, the threshold check would fail and the call would silently return the FP32 reference;
/// the contract tests cannot tell (either value is within the bound). This test pins that an accepted reduced result
/// is the one returned: the FP16 query 1.0001 rounds to 1.0, so the reduced and reference distances differ in bits
/// but agree within the 0.001 threshold.
@Suite("adaptiveEuclideanDistance returns its accepted reduced result")
struct AdaptiveReducedResultTests {
    @Test func acceptedReducedResultIsReturned() {
        let query = Vector512Optimized(repeating: 1.0001), candidate = Vector512Optimized(repeating: 0)
        let reference = query.euclideanDistance(to: candidate)
        let queryFP16 = MixedPrecisionKernels.Vector512FP16(from: query)
        let candidates = MixedPrecisionKernels.createSoA512FP16(from: [candidate])
        var output: [Float] = [0]
        let reduced: Float = output.withUnsafeMutableBufferPointer {
            MixedPrecisionKernels.batchEuclidean512(query: queryFP16, candidates: candidates, results: $0)
            return $0[0]
        }
        #expect(reduced.isFinite && reduced.bitPattern != reference.bitPattern)
        #expect(abs(Double(reduced) - Double(reference)) <= 0.001 * Double(reference))
        let actual = MixedPrecisionKernels.adaptiveEuclideanDistance(query: query, candidate: candidate, threshold: 0.001)
        #expect(actual.bitPattern == reduced.bitPattern, "got \(actual), reduced \(reduced), reference \(reference)")
    }
}
