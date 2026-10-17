import Foundation
import Testing
@testable import VectorCore

// P02 phase B: the pure Gram planner (`GramPlan`, Sources/VectorCore/Operations/GramPlan.swift). Counts only: no test
// here builds a collection proportional to the planned sizes. Phase 1 check map rows T1-T3, T5-T7, T9, T10, T19,
// T35-T40, E13-E19 (planner part).
//
// The independent oracle recomputes every quantity in Int128 (no overflow below 2^127) from the brief's formula
// and compares, so it shares no arithmetic path with the planner's checked Int operations.

/// Outcome of a planner call, compared by kind and context.
enum P02PlanOutcome: Equatable, CustomStringConvertible {
    case plan(GramPlan)
    case invalidField(String)
    case notRepresentable(String)
    case exceeds(required: String, allowed: String)
    case otherVectorError(String)
    case other(String)

    var description: String {
        switch self {
        case .plan(let p): return "plan(payload \(p.temporaryPayloadBytes))"
        case .invalidField(let f): return "invalidConfiguration(\(f))"
        case .notRepresentable(let f): return "notRepresentable(\(f))"
        case let .exceeds(required, allowed): return "exceeds(\(required) > \(allowed))"
        case .otherVectorError(let k): return "VectorError(\(k))"
        case .other(let e): return "error(\(e))"
        }
    }

    static func of(_ error: any Error) -> P02PlanOutcome {
        guard let e = error as? VectorError else { return .other(String(describing: error)) }
        let info = e.context.additionalInfo
        switch e.kind {
        case .invalidConfiguration: return .invalidField(info["field"] ?? "?")
        case .resourceUnavailable where info["reason"] == "not representable": return .notRepresentable(info["field"] ?? "?")
        case .resourceUnavailable: return .exceeds(required: info["requiredBytes"] ?? "?", allowed: info["allowedBytes"] ?? "?")
        default: return .otherVectorError(e.kind.rawValue)
        }
    }
}

struct P02PlanInput: CustomStringConvertible {
    var q: Int, n: Int, d: Int, k: Int
    var euclidean: Bool
    var r: Int, c: Int, cap: Int
    var description: String { "Q\(q) N\(n) D\(d) k\(k) \(euclidean ? "euclid" : "cosine") r\(r) c\(c) cap\(cap)" }

    func plan() -> P02PlanOutcome {
        do {
            return .plan(try GramPlan(queryCount: q, candidateCount: n, dimension: d, k: k, euclidean: euclidean,
                                      queryTileRows: r, candidateTileRows: c, maximumTemporaryPayloadBytes: cap))
        } catch {
            return .of(error)
        }
    }
}

private let i32 = Int(Int32.max)

/// Admission stub for `GramPlan.make`: records whether it ran, then succeeds or throws.
final class P02AdmissionSpy: @unchecked Sendable {
    private(set) var calls = 0
    let failure: (any Error)?
    let euclidean: Bool
    init(failure: (any Error)? = nil, euclidean: Bool = true) { self.failure = failure; self.euclidean = euclidean }
    func admit() throws -> (admitted: String, euclidean: Bool) {
        calls += 1
        if let failure { throw failure }
        return ("admitted", euclidean)
    }
}

struct P02AdmissionError: Error, Equatable { let token: Int }

@Suite("P02 Gram planner")
struct P02GramPlanTests {

    // MARK: T1, field domains

    /// Each field alone out of its domain gives `.invalidConfiguration` naming it (context `field`, `reason`,
    /// `value`, and a message containing the field name); `maximumTemporaryPayloadBytes == 0` is in its domain.
    @Test func fieldDomains() throws {
        let valid = (r: 8, c: 256, cap: 1 << 20)
        let cases: [(r: Int, c: Int, cap: Int, field: String)] = [
            (0, valid.c, valid.cap, "queryTileRows"), (-1, valid.c, valid.cap, "queryTileRows"),
            (Int.min, valid.c, valid.cap, "queryTileRows"),
            (valid.r, 0, valid.cap, "candidateTileRows"), (valid.r, -1, valid.cap, "candidateTileRows"),
            (valid.r, Int.min, valid.cap, "candidateTileRows"),
            (valid.r, valid.c, -1, "maximumTemporaryPayloadBytes"), (valid.r, valid.c, Int.min, "maximumTemporaryPayloadBytes")
        ]
        for item in cases {
            var thrown: (any Error)?
            do {
                try GramPlan.validateFields(queryTileRows: item.r, candidateTileRows: item.c,
                                            maximumTemporaryPayloadBytes: item.cap)
            } catch { thrown = error }
            let error = try #require(thrown as? VectorError, "\(item)")
            #expect(error.kind == .invalidConfiguration, "\(item)")
            #expect(error.context.additionalInfo["field"] == item.field, "\(item)")
            #expect(error.context.additionalInfo["value"] == String(item.r <= 0 ? item.r : item.c <= 0 ? item.c : item.cap))
            #expect(error.context.additionalInfo["message"]?.contains(item.field) == true, "\(item)")
            let input = P02PlanInput(q: 8, n: 256, d: 512, k: 10, euclidean: true, r: item.r, c: item.c, cap: item.cap)
            #expect(input.plan() == .invalidField(item.field), "\(input)")
        }
        #expect(throws: Never.self) {
            try GramPlan.validateFields(queryTileRows: 1, candidateTileRows: 1, maximumTemporaryPayloadBytes: 0)
        }
    }

    /// Several invalid fields: the first in declaration order (queryTileRows, candidateTileRows,
    /// maximumTemporaryPayloadBytes) is reported, ahead of every shape and representability problem.
    @Test func fieldDomainOrder() {
        let cases: [(r: Int, c: Int, cap: Int, field: String)] = [
            (0, 0, -1, "queryTileRows"), (0, 1, -1, "queryTileRows"), (0, 0, 0, "queryTileRows"),
            (1, 0, -1, "candidateTileRows"), (1, -5, 0, "candidateTileRows"), (1, 1, -1, "maximumTemporaryPayloadBytes")
        ]
        for item in cases {
            // An unrepresentable shape (D beyond Int32, Q = 0) must not win over a field-domain error.
            for input in [P02PlanInput(q: 8, n: 256, d: 512, k: 10, euclidean: true, r: item.r, c: item.c, cap: item.cap),
                          P02PlanInput(q: 0, n: 256, d: i32 + 1, k: 10, euclidean: false, r: item.r, c: item.c, cap: item.cap)] {
                #expect(input.plan() == .invalidField(item.field), "\(input)")
            }
        }
    }

    // MARK: T2, T3, T5, precedence through `make`

    /// Field domains precede admission: with invalid fields the admission closure never runs, even when it would
    /// reject the request (a DynamicVector-like unsupported case).
    @Test func precedenceFieldsBeforeAdmission() {
        let spy = P02AdmissionSpy(failure: VectorError(kind: .unsupportedOperation, context: ErrorContext()))
        var outcome: P02PlanOutcome?
        do {
            _ = try GramPlan.make(queryCount: 8, candidateCount: 256, dimension: 512, k: 10, queryTileRows: 0,
                                  candidateTileRows: 0, maximumTemporaryPayloadBytes: -1, admit: spy.admit)
        } catch { outcome = .of(error) }
        #expect(outcome == .invalidField("queryTileRows"))
        #expect(spy.calls == 0)
    }

    /// Admission precedes shape, representability and the limit: its error propagates unchanged (identity for a
    /// non-VectorError) even when the plan would also be unrepresentable and over a zero limit.
    @Test func precedenceAdmissionBeforePlan() {
        let unsupported = P02AdmissionSpy(failure: VectorError(kind: .unsupportedOperation, context: ErrorContext()))
        let custom = P02AdmissionSpy(failure: P02AdmissionError(token: 41))
        for (spy, expected) in [(unsupported, P02PlanOutcome.otherVectorError("unsupportedOperation")),
                                (custom, P02PlanOutcome.other(String(describing: P02AdmissionError(token: 41))))] {
            for (q, d) in [(8, 512), (0, i32 + 1)] {
                var outcome: P02PlanOutcome?
                do {
                    _ = try GramPlan.make(queryCount: q, candidateCount: 256, dimension: d, k: 10, queryTileRows: 8,
                                          candidateTileRows: 256, maximumTemporaryPayloadBytes: 0, admit: spy.admit)
                } catch { outcome = .of(error) }
                #expect(outcome == expected, "Q\(q) D\(d)")
            }
            #expect(spy.calls == 2)
        }
        #expect(throws: P02AdmissionError(token: 41)) {
            _ = try GramPlan.make(queryCount: 8, candidateCount: 256, dimension: 512, k: 10, queryTileRows: 8,
                                  candidateTileRows: 256, maximumTemporaryPayloadBytes: 0, admit: custom.admit)
        }
    }

    /// After admission, shape and representability precede the limit: an unrepresentable plan with a zero limit
    /// reports "not representable", never "exceeds". The message says "must be positive" for a nonpositive count and
    /// "does not fit" for an overflowing one (phase B review F3).
    @Test func precedenceRepresentabilityBeforeLimit() throws {
        let spy = P02AdmissionSpy()
        let cases: [(q: Int, n: Int, d: Int, k: Int, field: String)] = [
            (0, 256, 512, 10, "queryCount"), (8, 0, 512, 10, "candidateCount"), (8, 256, 512, 0, "k"),
            (8, 256, 0, 10, "dimension"), (8, 256, -3, 10, "dimension"), (8, 256, i32 + 1, 10, "dimension (BLAS Int32)")
        ]
        for item in cases {
            var outcome: P02PlanOutcome?
            do {
                _ = try GramPlan.make(queryCount: item.q, candidateCount: item.n, dimension: item.d, k: item.k,
                                      queryTileRows: 8, candidateTileRows: 256, maximumTemporaryPayloadBytes: 0,
                                      admit: spy.admit)
            } catch {
                outcome = .of(error)
                let message = (error as? VectorError)?.context.additionalInfo["message"] ?? ""
                let positivity = item.field == "queryCount" || item.field == "candidateCount" || item.field == "k"
                    || item.field == "dimension"
                #expect(message.contains("must be positive") == positivity, "\(item): \(message)")
                #expect(message.contains("does not fit") == !positivity, "\(item): \(message)")
            }
            #expect(outcome == .notRepresentable(item.field), "\(item)")
        }
        #expect(spy.calls == cases.count)
        // A representable plan over a zero limit reports the limit, and `make` returns the admitted value otherwise.
        let made = try GramPlan.make(queryCount: 8, candidateCount: 256, dimension: 512, k: 10, queryTileRows: 8,
                                     candidateTileRows: 256, maximumTemporaryPayloadBytes: Int.max, admit: spy.admit)
        #expect(made.admitted == "admitted")
        #expect(made.plan.temporaryPayloadBytes == 550_880)
    }

    // MARK: T6, T7, clamping

    /// Requested tile rows are clamped to the actual counts before any product, so `Int.max` requests plan exactly
    /// like requests equal to the counts; k is clamped to N the same way.
    @Test func clampsBeforeProducts() {
        for euclidean in [true, false] {
            for (q, n) in [(8, 256), (9, 300), (1, 1), (3, 7)] {
                let exact = P02PlanInput(q: q, n: n, d: 768, k: 10, euclidean: euclidean, r: q, c: n, cap: Int.max).plan()
                let huge = P02PlanInput(q: q, n: n, d: 768, k: 10, euclidean: euclidean, r: Int.max, c: Int.max, cap: Int.max).plan()
                #expect(huge == exact, "Q\(q) N\(n) \(euclidean)")
                if case .plan(let p) = huge {
                    #expect(p.queryTileRows == q && p.candidateTileRows == n && p.queryTileCount == 1 && p.candidateTileCount == 1)
                    #expect(p.resultK == min(10, n))
                } else {
                    Issue.record("Q\(q) N\(n): \(huge)")
                }
            }
        }
        // k = Int.max: K = N heaps and Q * N result entries (the result is outside the limit).
        if case .plan(let p) = P02PlanInput(q: 8, n: 256, d: 512, k: Int.max, euclidean: true, r: 8, c: 256, cap: Int.max).plan() {
            #expect(p.resultK == 256 && p.heapSlots == 8 * 256 && p.resultElements == 8 * 256)
        } else {
            Issue.record("k = Int.max did not plan")
        }
    }

    // MARK: T9, T10, the limit

    /// The limit is inclusive: required - 1 throws `.resourceUnavailable` (exact `requiredBytes`, `allowedBytes`,
    /// `boundary` and a message naming the logical temporary payload and saying it is not a memory cap); required
    /// and required + 1 plan, and the plan keeps the caller's limit. The tiles are never shrunk.
    @Test func capBoundary() throws {
        let geometries: [(q: Int, n: Int, d: Int, k: Int, r: Int, c: Int)] = [
            (1, 1, 512, 1, 1, 1), (3, 7, 768, 2, 2, 3), (8, 256, 512, 10, 8, 256), (8, 256, 1536, 10, Int.max, Int.max),
            (64, 4096, 1536, Int.max, 16, 1024), (5, 9, 768, 9, 1, 4)
        ]
        for g in geometries {
            for euclidean in [true, false] {
                let base = P02PlanInput(q: g.q, n: g.n, d: g.d, k: g.k, euclidean: euclidean, r: g.r, c: g.c, cap: Int.max)
                guard case .plan(let reference) = base.plan() else { Issue.record("\(base)"); continue }
                let required = reference.temporaryPayloadBytes
                var below = base; below.cap = required - 1
                var thrown: (any Error)?
                do { _ = try GramPlan(queryCount: below.q, candidateCount: below.n, dimension: below.d, k: below.k,
                                      euclidean: euclidean, queryTileRows: below.r, candidateTileRows: below.c,
                                      maximumTemporaryPayloadBytes: below.cap) } catch { thrown = error }
                let error = try #require(thrown as? VectorError, "\(below)")
                let info = error.context.additionalInfo
                #expect(error.kind == .resourceUnavailable, "\(below)")
                #expect(info["requiredBytes"] == String(required) && info["allowedBytes"] == String(required - 1), "\(below)")
                #expect(info["boundary"] == "logical temporary payload", "\(below)")
                #expect(info["message"]?.contains("logical temporary payload") == true, "\(below)")
                #expect(info["message"]?.contains("not a memory cap") == true, "\(below)")
                for cap in [required, required + 1] {
                    var input = base; input.cap = cap
                    guard case .plan(let p) = input.plan() else { Issue.record("\(input)"); continue }
                    #expect(p.maximumTemporaryPayloadBytes == cap)
                    #expect(p.queryTileRows == reference.queryTileRows && p.candidateTileRows == reference.candidateTileRows)
                }
                var zero = base; zero.cap = 0
                #expect(zero.plan() == .exceeds(required: String(required), allowed: "0"), "\(zero)")
            }
        }
    }

    // MARK: T19, T20, payload identity against an independent formula

    /// Hand-computed anchors (Q8 N256 D512 k10, r8 c256): Euclidean floats 8*512 + 256*512 + 8*256 + (8 + 256) =
    /// 137_480, times 4 = 549_920, plus heaps 8*10*12 = 960 gives 550_880; cosine drops the 264 norms: 549_824.
    /// These need no Int128 and run on every supported platform.
    @Test func payloadHandAnchors() throws {
        let euclid = P02PlanInput(q: 8, n: 256, d: 512, k: 10, euclidean: true, r: 8, c: 256, cap: Int.max).plan()
        let cosine = P02PlanInput(q: 8, n: 256, d: 512, k: 10, euclidean: false, r: 8, c: 256, cap: Int.max).plan()
        guard case .plan(let e) = euclid, case .plan(let c) = cosine else { Issue.record("\(euclid) \(cosine)"); return }
        #expect(e.temporaryPayloadBytes == 550_880 && e.normElements == 264 && e.heapPayloadBytes == 960)
        #expect(c.temporaryPayloadBytes == 549_824 && c.normElements == 0)
    }

    /// Random and boundary inputs against the Int128 oracle (every field of the plan, the error kind, and the exact
    /// `requiredBytes` / `allowedBytes` when over the limit). The run must cover all three outcomes. Every input the
    /// oracle finds unrepresentable gets a FINITE limit (0, a small value or a random value below 2^40), so a planner
    /// that checks the limit before a later representability check reports "exceeds" and disagrees with the oracle
    /// (phase B review F1, mutant MA). At least 200 of those inputs must have a representable payload above their
    /// limit, the inputs that separate the two orders.
    /// Int128 needs macOS 15 (or the matching OS): below it Swift Testing skips this test and records the
    /// availability reason (phase B review F5); `payloadHandAnchors` and the planner fixtures still run.
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    @Test func payloadMatchesIndependentFormula() throws {
        var random = P02SplitMix(state: 0x6A11_0002)
        var counts = (plan: 0, unrepresentable: 0, exceeds: 0, separating: 0)
        var mismatches: [String] = []
        let boundary = [1, 2, 3, 7, 8, 255, 256, 257, 1024, i32 - 1, i32, i32 + 1, 1 << 40, Int.max / 16, Int.max - 1, Int.max]
        for index in 0..<20_000 {
            var x = P02PlanInput(q: p02Pick(&random, boundary), n: p02Pick(&random, boundary), d: p02PickSmall(&random, boundary),
                                 k: p02Pick(&random, boundary), euclidean: random.next() & 1 == 0,
                                 r: p02Pick(&random, boundary), c: p02Pick(&random, boundary), cap: Int.max)
            if index % 97 == 0 { x.q = [0, -1, Int.min][index % 3] }
            switch p02Oracle(x) {
            case .plan(let v):
                let payload = v["temporaryPayloadBytes"] ?? 0
                switch random.next() % 4 {
                case 0: x.cap = Int(payload) - 1
                case 1: x.cap = Int(payload)
                case 2: x.cap = Int(truncatingIfNeeded: random.next() >> 1) % max(1, Int(payload))
                default: break
                }
            case .unrepresentable:
                switch random.next() % 3 {
                case 0: x.cap = 0
                case 1: x.cap = Int(random.next() % 4_096)
                default: x.cap = Int(random.next() % (1 << 40))
                }
                if let payload = p02OraclePayload(x), payload > Int128(x.cap) { counts.separating += 1 }
            case .exceeds:
                break
            }
            switch p02Oracle(x) {
            case .plan: counts.plan += 1
            case .unrepresentable: counts.unrepresentable += 1
            case .exceeds: counts.exceeds += 1
            }
            if let problem = p02CompareWithOracle(x) { mismatches.append(problem) }
        }
        #expect(mismatches.isEmpty, "\(mismatches.count) mismatches, first \(mismatches.prefix(3))")
        #expect(counts.plan >= 1_000 && counts.unrepresentable >= 1_000 && counts.exceeds >= 1_000, "\(counts)")
        #expect(counts.separating >= 200, "\(counts)")
    }

    // MARK: T35-T39, representability (one fixture per reachable intermediate)

    /// Each fixture fails first at the named intermediate (checked order: shape, Int32 arguments, packs, tile,
    /// norms, the three float sums, float bytes, heap slots, heap bytes, payload, result elements, result bytes,
    /// row-container bytes, result total). Values were derived offline for this order (P02/phaseB fixture script)
    /// and the Int128 oracle confirms each is unrepresentable. Unreachable on LP64 once D, r and c fit Int32, so
    /// no fixture exists for them: r*D, c*D and r*c (at most (2^31-1)^2 < 2^62), r + c, the first float sum
    /// (2 (2^31-1)^2 < 2^63 - 1), the heap slot stride (a constant), and the row-container bytes (Q * 8 <= Q * K * 16
    /// whenever the result bytes fit).
    /// Each fixture also runs with a zero limit and must still name the same intermediate: representability precedes
    /// the limit for every intermediate, the result counts included (phase B review F1: a planner that checks the
    /// limit before `resultElements` returns "exceeds" for the `resultElements`, `resultBytes` and `resultTotalBytes`
    /// fixtures at limit 0).
    @Test func overflowFixturesNameTheFirstFailingIntermediate() throws {
        try #require(MemoryLayout<Float>.stride == 4 && MemoryLayout<Int>.stride == 8, "fixtures assume LP64")
        try #require(MemoryLayout<NearestNeighborResult>.stride == 16 && MemoryLayout<[NearestNeighborResult]>.stride == 8)
        for (input, field) in Self.overflowFixtures {
            #expect(input.plan() == .notRepresentable(field), "\(input)")
            var zero = input
            zero.cap = 0
            #expect(zero.plan() == .notRepresentable(field), "\(zero)")
        }
    }

    /// The Int128 oracle agrees that every overflow fixture is unrepresentable (macOS 15 or later; skipped with the
    /// recorded availability reason below it, phase B review F5).
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    @Test func overflowFixturesAreUnrepresentableForTheInt128Oracle() {
        for (input, _) in Self.overflowFixtures {
            if case .unrepresentable = p02Oracle(input) {} else { Issue.record("oracle disagrees: \(input)") }
        }
    }

    /// One fixture per reachable intermediate, with an unbounded limit (see `overflowFixturesNameTheFirstFailingIntermediate`).
    static var overflowFixtures: [(P02PlanInput, String)] {
        let m = Int.max
        return [
            (P02PlanInput(q: 8, n: 256, d: i32 + 1, k: 10, euclidean: true, r: 8, c: 256, cap: m), "dimension (BLAS Int32)"),
            (P02PlanInput(q: i32 + 1, n: 256, d: 512, k: 10, euclidean: true, r: i32 + 1, c: 256, cap: m), "queryTileRows (BLAS Int32)"),
            (P02PlanInput(q: 8, n: i32 + 1, d: 512, k: 10, euclidean: true, r: 8, c: i32 + 1, cap: m), "candidateTileRows (BLAS Int32)"),
            (P02PlanInput(q: i32, n: i32, d: i32, k: 1, euclidean: true, r: i32, c: i32, cap: m), "packAndTileElements"),
            (P02PlanInput(q: i32, n: i32, d: 1_073_741_825, k: 1, euclidean: true, r: i32, c: i32, cap: m), "floatElements"),
            (P02PlanInput(q: 1 << 30, n: 1 << 30, d: 1 << 30, k: 1, euclidean: true, r: 1 << 30, c: 1 << 30, cap: m), "floatPayloadBytes"),
            (P02PlanInput(q: i32, n: m, d: 1, k: m / 2, euclidean: false, r: i32, c: 1, cap: m), "heapSlots"),
            (P02PlanInput(q: 1, n: m, d: 1, k: m / 2, euclidean: false, r: 1, c: 1, cap: m), "heapPayloadBytes"),
            (P02PlanInput(q: 1 << 29, n: m, d: 1 << 30, k: 650_000_000, euclidean: true, r: 1 << 29, c: 1 << 29, cap: m),
             "temporaryPayloadBytes"),
            (P02PlanInput(q: m, n: m, d: 1, k: 2, euclidean: false, r: 1, c: 1, cap: m), "resultElements"),
            (P02PlanInput(q: m / 2, n: m, d: 1, k: 1, euclidean: false, r: 1, c: 1, cap: m), "resultBytes"),
            (P02PlanInput(q: (1 << 59) - 1, n: m, d: 1, k: 1, euclidean: false, r: 1, c: 1, cap: m), "resultTotalBytes")
        ]
    }

    // MARK: T37, T38, T40, BLAS Int32 arguments and large representable plans

    /// D, r and c (the BLAS dimensions and leading dimensions) must fit Int32; total Q and N need not; there is no
    /// production-size ceiling beyond representability.
    @Test func int32ArgumentsAndLargeRepresentablePlans() {
        let m = Int.max
        func plans(_ x: P02PlanInput) -> GramPlan? { if case .plan(let p) = x.plan() { return p } else { return nil } }
        let atLimit = plans(P02PlanInput(q: 1, n: 1, d: i32, k: 1, euclidean: true, r: 1, c: 1, cap: m))
        #expect(atLimit?.blasDimension == Int32.max)
        #expect(P02PlanInput(q: 1, n: 1, d: i32 + 1, k: 1, euclidean: true, r: 1, c: 1, cap: m).plan()
                == .notRepresentable("dimension (BLAS Int32)"))
        // A request beyond Int32 is fine when the clamped tile fits.
        #expect(plans(P02PlanInput(q: 8, n: 256, d: 512, k: 10, euclidean: true, r: i32 + 1, c: m, cap: m))?.queryTileRows == 8)
        // Total N beyond Int32 with c = 1024 plans (T38), with ceil(N / c) tiles.
        let wide = plans(P02PlanInput(q: 8, n: i32 + 1, d: 512, k: 10, euclidean: true, r: 8, c: 1024, cap: m))
        #expect(wide?.candidateTileCount == (i32 + 1 - 1) / 1024 + 1 && wide?.blasCandidateTileRows == 1024)
        #expect(wide?.temporaryPayloadBytes == 2_151_392)
        // T40: a large representable plan.
        let large = plans(P02PlanInput(q: 4 * i32, n: 4 * i32, d: 1536, k: 10, euclidean: true, r: 64, c: 1024, cap: m))
        #expect(large?.temporaryPayloadBytes == 6_958_848 && large?.resultElements == 4 * i32 * 10)
    }

    // MARK: Tile ranges

    /// Tiles cover 0..<count in increasing order with no gap or overlap; every tile but the last has exactly the
    /// planned rows; the last one ends at the count without overflow at extreme counts.
    @Test func tileRangesCoverInOrderWithoutOverflow() {
        for count in 1...20 {
            for rows in 1...25 {
                let x = P02PlanInput(q: count, n: count, d: 4, k: 1, euclidean: true, r: rows, c: rows, cap: Int.max)
                guard case .plan(let p) = x.plan() else { Issue.record("\(x)"); continue }
                let queryTiles = (0..<p.queryTileCount).map(p.queryTileRange)
                let candidateTiles = (0..<p.candidateTileCount).map(p.candidateTileRange)
                for tiles in [queryTiles, candidateTiles] {
                    #expect(tiles.flatMap { Array($0) } == Array(0..<count), "\(x)")
                    #expect(tiles.dropLast().allSatisfy { $0.count == min(rows, count) }, "\(x)")
                    #expect(tiles.last.map { $0.count >= 1 && $0.count <= min(rows, count) } == true, "\(x)")
                }
            }
        }
        let q = Int.max / 32, n = Int.max, rows = 1 << 20
        let x = P02PlanInput(q: q, n: n, d: 1, k: 1, euclidean: false, r: rows, c: rows, cap: Int.max)
        guard case .plan(let p) = x.plan() else { Issue.record("\(x): \(x.plan())"); return }
        let lastQuery = p.queryTileRange(p.queryTileCount - 1), lastCandidate = p.candidateTileRange(p.candidateTileCount - 1)
        #expect(lastQuery.upperBound == q && lastQuery.count >= 1 && lastQuery.count <= rows)
        #expect(lastCandidate.upperBound == n && lastCandidate.count >= 1 && lastCandidate.count <= rows)
        #expect(p.queryTileRange(0) == 0..<rows && p.candidateTileRange(1) == rows..<(2 * rows))
    }
}

// MARK: - Independent oracle (Int128)

/// A positive value from a mix of small, medium, Int32-boundary, power-of-two and near-Int.max magnitudes.
func p02Pick(_ random: inout P02SplitMix, _ boundary: [Int]) -> Int {
    switch random.next() % 6 {
    case 0: return Int(random.next() % 300) + 1
    case 1: return Int(random.next() % 1_000_000) + 1
    case 2: return boundary[Int(random.next() % UInt64(boundary.count))]
    case 3: return max(1, (1 << Int(random.next() % 63)) + Int(random.next() % 3) - 1)
    case 4: return Int.max - Int(random.next() % 4)
    default: return max(1, Int(truncatingIfNeeded: random.next() >> 1))
    }
}

/// A dimension: mostly small (so many plans are representable), sometimes a boundary value.
func p02PickSmall(_ random: inout P02SplitMix, _ boundary: [Int]) -> Int {
    random.next() % 4 == 0 ? boundary[Int(random.next() % UInt64(boundary.count))] : Int(random.next() % 2048) + 1
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
enum P02OracleOutcome {
    case unrepresentable
    case exceeds(Int128)
    case plan([String: Int128])
}

/// The brief's formula and checks in Int128.
@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
func p02Oracle(_ x: P02PlanInput) -> P02OracleOutcome {
    guard x.q > 0, x.n > 0, x.k > 0, x.d > 0 else { return .unrepresentable }
    let q = Int128(x.q), n = Int128(x.n), d = Int128(x.d), i32 = Int128(Int32.max)
    let r = min(Int128(x.r), q), c = min(Int128(x.c), n), kk = min(Int128(x.k), n)
    guard d <= i32, r <= i32, c <= i32 else { return .unrepresentable }
    let f = Int128(MemoryLayout<Float>.stride), i = Int128(MemoryLayout<Int>.stride)
    let norms: Int128 = x.euclidean ? r + c : 0
    let floats = r * d + c * d + r * c + norms
    let payload = f * floats + r * kk * (f + i)
    let resultElements = q * kk   // at most (2^63)^2 = 2^126, inside Int128
    // Every other value here stays below 2^100 once D, r and c fit Int32; only Q * K * stride can pass 2^127, so the
    // oracle stops before forming it.
    guard resultElements <= Int128(Int.max) else { return .unrepresentable }
    let resultBytes = resultElements * Int128(MemoryLayout<NearestNeighborResult>.stride)
    let rowBytes = q * Int128(MemoryLayout<[NearestNeighborResult]>.stride)
    let values: [String: Int128] = [
        "queryTileRows": r, "candidateTileRows": c, "resultK": kk, "queryPackElements": r * d,
        "candidatePackElements": c * d, "scoreTileElements": r * c, "normElements": norms, "heapSlots": r * kk,
        "floatPayloadBytes": f * floats, "heapPayloadBytes": r * kk * (f + i), "temporaryPayloadBytes": payload,
        "queryTileCount": (q + r - 1) / r, "candidateTileCount": (n + c - 1) / c, "resultElements": resultElements,
        "resultBytes": resultBytes, "resultRowContainerBytes": rowBytes, "resultTotalBytes": resultBytes + rowBytes
    ]
    if values.values.contains(where: { $0 > Int128(Int.max) }) { return .unrepresentable }
    return payload > Int128(x.cap) ? .exceeds(payload) : .plan(values)
}

/// The oracle's payload for `x` when the shape and the BLAS Int32 checks pass (it may exceed Int.max), else nil.
@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
func p02OraclePayload(_ x: P02PlanInput) -> Int128? {
    guard x.q > 0, x.n > 0, x.k > 0, x.d > 0 else { return nil }
    let q = Int128(x.q), n = Int128(x.n), d = Int128(x.d), i32 = Int128(Int32.max)
    let r = min(Int128(x.r), q), c = min(Int128(x.c), n), kk = min(Int128(x.k), n)
    guard d <= i32, r <= i32, c <= i32 else { return nil }
    let f = Int128(MemoryLayout<Float>.stride), i = Int128(MemoryLayout<Int>.stride)
    return f * (r * d + c * d + r * c + (x.euclidean ? r + c : 0)) + r * kk * (f + i)
}

/// nil when the planner agrees with the oracle, else a description of the disagreement.
@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
func p02CompareWithOracle(_ x: P02PlanInput) -> String? {
    let actual = x.plan()
    switch (p02Oracle(x), actual) {
    case (.unrepresentable, .notRepresentable):
        return nil
    case let (.exceeds(payload), .exceeds(required, allowed)):
        return required == String(payload) && allowed == String(x.cap) ? nil : "\(x): \(actual) vs \(payload)"
    case let (.plan(v), .plan(p)):
        let got: [String: Int] = [
            "queryTileRows": p.queryTileRows, "candidateTileRows": p.candidateTileRows, "resultK": p.resultK,
            "queryPackElements": p.queryPackElements, "candidatePackElements": p.candidatePackElements,
            "scoreTileElements": p.scoreTileElements, "normElements": p.normElements, "heapSlots": p.heapSlots,
            "floatPayloadBytes": p.floatPayloadBytes, "heapPayloadBytes": p.heapPayloadBytes,
            "temporaryPayloadBytes": p.temporaryPayloadBytes, "queryTileCount": p.queryTileCount,
            "candidateTileCount": p.candidateTileCount, "resultElements": p.resultElements, "resultBytes": p.resultBytes,
            "resultRowContainerBytes": p.resultRowContainerBytes, "resultTotalBytes": p.resultTotalBytes
        ]
        let wrong = got.filter { Int128($0.value) != v[$0.key] }.map(\.key).sorted()
        let blas = Int(p.blasDimension) == x.d && Int(p.blasQueryTileRows) == p.queryTileRows
            && Int(p.blasCandidateTileRows) == p.candidateTileRows && p.maximumTemporaryPayloadBytes == x.cap
        return wrong.isEmpty && blas ? nil : "\(x): fields \(wrong) blas \(blas)"
    default:
        return "\(x): planner \(actual), oracle \(p02Oracle(x))"
    }
}
