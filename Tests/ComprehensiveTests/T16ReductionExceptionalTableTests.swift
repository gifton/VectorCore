import Foundation
import Testing
@testable import VectorCore

// T16 exceptional table (T16 phase 1 rulings (a), (c), C2, C4): every row of T16ReductionRegistry × every fixture of
// T16Fixtures (18 sizes, NaN at every index, payload/negative/signalling NaN, all-NaN, NaN with +-inf, +-inf, signed
// zeros, empty). Each row prints one `T16CELLS<TAB>row<TAB>count<TAB>cells` line listing its failing cells, so a run
// log is the per-row RED/GREEN record. Synchronous @Test functions only (Xcode 27 drops failures of nonisolated async
// XCTest methods; the profiler rows are async on an actor and run in an isolated child, see below).

enum T16Table {
    static let fixtures = T16Fixtures.all()

    /// Failing cells of one row: `fixture-id want ... got ...`.
    static func failures(_ row: T16Row) -> [String] {
        var out: [String] = []
        for fixture in fixtures {
            guard let input = row.input(fixture) else { continue }
            let want = row.expect(input)
            let got = row.run(input)
            if !T16Check.admits(want, got) { out.append("\(fixture.id) want \(want) got \(got)") }
        }
        return out
    }

    static func report(_ id: String, _ failing: [String]) {
        let cells = failing.map { String($0.split(separator: " ", maxSplits: 1)[0]) }
        print("T16CELLS\t\(id)\t\(failing.count)\t\(cells.joined(separator: ","))")
    }

    static func cells(_ failing: [String]) -> Set<String> {
        Set(failing.map { String($0.split(separator: " ", maxSplits: 1)[0]) })
    }
}

@Suite("T16 reduction exceptional table (registry-driven)")
struct T16ReductionExceptionalTableTests {

    @Test func everyRowFollowsTheContract() {
        for row in T16ReductionRegistry.rows {
            let failing = T16Table.failures(row)
            T16Table.report(row.id, failing)
            #expect(failing.isEmpty, "\(row.id): \(failing.count) failing cells; first: \(failing.prefix(4))")
        }
    }

    /// Ruling (b)/brief 2: the Swift and Accelerate implementations agree on every fixture under the same
    /// equivalence (NaN with NaN; `==` otherwise, so signed zeros are equal (S1), except maximumMagnitude, where a zero
    /// must have the same sign (Z1); indices exactly).
    @Test func providersAgree() {
        #if canImport(Accelerate)
        let pairs: [(String, String, Bool)] = [
            ("SwiftFloatSIMDProvider.maximum", "AccelerateFloatProvider.maximum", false),
            ("SwiftFloatSIMDProvider.minimum", "AccelerateFloatProvider.minimum", false),
            ("SwiftFloatSIMDProvider.maximumMagnitude", "AccelerateFloatProvider.maximumMagnitude", true),
            ("SwiftDoubleSIMDProvider.maximum", "AccelerateDoubleProvider.maximum", false),
            ("SwiftDoubleSIMDProvider.minimum", "AccelerateDoubleProvider.minimum", false),
            ("SwiftDoubleSIMDProvider.maximumMagnitude", "AccelerateDoubleProvider.maximumMagnitude", true),
            ("SwiftFloatSIMDProvider.clip", "AccelerateFloatProvider.clip", false),
            ("SwiftDoubleSIMDProvider.clip", "AccelerateDoubleProvider.clip", false),
            ("DefaultArraySIMDProvider.max", "AccelerateArraySIMDProvider.max", false),
            ("DefaultArraySIMDProvider.min", "AccelerateArraySIMDProvider.min", false),
            ("DefaultArraySIMDProvider.minIndex", "AccelerateArraySIMDProvider.minIndex", false),
            ("DefaultArraySIMDProvider.maxIndex", "AccelerateArraySIMDProvider.maxIndex", false),
            ("DefaultArraySIMDProvider.elementWiseMin(a,b)", "AccelerateArraySIMDProvider.elementWiseMin(a,b)", false),
            ("DefaultArraySIMDProvider.elementWiseMin(b,a)", "AccelerateArraySIMDProvider.elementWiseMin(b,a)", false),
            ("DefaultArraySIMDProvider.elementWiseMax(a,b)", "AccelerateArraySIMDProvider.elementWiseMax(a,b)", false),
            ("DefaultArraySIMDProvider.elementWiseMax(b,a)", "AccelerateArraySIMDProvider.elementWiseMax(b,a)", false),
            ("DefaultArraySIMDProvider.clip", "AccelerateArraySIMDProvider.clip", false),
            ("Vector<D>.minElement[default]", "Vector<D>.minElement[accelerate]", false),
            ("Vector<D>.maxElement[default]", "Vector<D>.maxElement[accelerate]", false),
            ("Vector<D>.min(_:)[default]", "Vector<D>.min(_:)[accelerate]", false),
            ("Vector<D>.max(_:)[default]", "Vector<D>.max(_:)[accelerate]", false),
            ("Vector<D>.clamped(to:)[default]", "Vector<D>.clamped(to:)[accelerate]", false),
            ("Vector<D>.clamp(to:)[default]", "Vector<D>.clamp(to:)[accelerate]", false),
            ("Vector<D>.softmax[default]", "Vector<D>.softmax[accelerate]", false)
        ]
        let byID = Dictionary(uniqueKeysWithValues: T16ReductionRegistry.rows.map { ($0.id, $0) })
        for (swiftID, accelerateID, signedZero) in pairs {
            guard let swift = byID[swiftID], let accelerate = byID[accelerateID] else {
                Issue.record("missing agreement row \(swiftID) or \(accelerateID)")
                continue
            }
            var disagreeing: [String] = []
            for fixture in T16Table.fixtures {
                guard let input = swift.input(fixture), accelerate.input(fixture) != nil else { continue }
                let a = swift.run(input), b = accelerate.run(input)
                if !T16Check.agree(a, b, signedZero: signedZero) { disagreeing.append("\(fixture.id) swift \(a) accelerate \(b)") }
            }
            T16Table.report("agree:\(swiftID)~\(accelerateID)", disagreeing)
            #expect(disagreeing.isEmpty, "\(swiftID) vs \(accelerateID): \(disagreeing.count) cells; first: \(disagreeing.prefix(4))")
        }
        #endif
    }

    /// The registry, the rows and the providers are consistent; no row is vacuous.
    @Test func registryCoverage() {
        let rows = T16ReductionRegistry.rows, asyncRows = T16ReductionRegistry.asyncRows
        let ids = rows.map(\.id) + asyncRows.map(\.id)
        #expect(Set(ids).count == ids.count, "duplicate row ids")
        var tabled = Set<String>()
        for (site, coverage) in T16ReductionRegistry.sites {
            if case .tabled(let id) = coverage {
                tabled.insert(id)
                #expect(ids.contains(id), "\(site) is tabled as \(id), which is not a row")
            }
        }
        #expect(ids.count >= tabled.count)
        var tags = Set(rows.map(\.tag))
        tags.formUnion(asyncRows.map(\.tag))
        #if canImport(Accelerate)
        #expect(tags == Set(T16Tag.allCases), "provider tags with rows: \(tags)")
        #else
        #expect(tags == [.swift, .scalar], "provider tags with rows: \(tags)")
        #endif
        for kind in [T16Kind.reduction, .index, .statistic, .elementWise, .control] {
            #expect(rows.contains { $0.kind == kind }, "no \(kind) row")
        }
        for row in rows {
            let applicable = T16Table.fixtures.filter { row.input($0) != nil }
            #expect(applicable.contains { $0.firstNaN != nil }, "\(row.id) has no NaN fixture")
            if row.kind != .control {
                #expect(applicable.contains { $0.firstNaN == nil }, "\(row.id) has no NaN-free fixture")
            }
        }
        #expect(T16Fixtures.sizes == [1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 16, 17, 31, 32, 33, 64, 65, 257])
        // NaN at every index of every size.
        let nanAt = Set(T16Table.fixtures.map(\.id).filter { $0.contains("/nan@") && !$0.contains(",") })
        #expect(nanAt.count == T16Fixtures.sizes.reduce(0, +))
    }

    /// The checker bites: known NaN-dropping implementations fail at the cells the phase 1 probe showed.
    @Test func mutationControls() {
        func failingCells(_ expect: @escaping T16Expectation, input: @escaping T16Input = { $0.values },
                          _ run: @escaping T16Run) -> Set<String> {
            T16Table.cells(T16Table.failures(T16Row(id: "mutant", kind: .reduction, tag: .scalar, input: input, expect: expect, run: run)))
        }
        // A plain Swift.max fold (VectorProtocol.max before T16) drops a NaN after element 0.
        let dropping = failingCells(T16ReductionRegistry.maxE) { v in
            .value(Double(v.reduce(-Float.infinity) { Swift.max($0, $1) }))
        }
        #expect(dropping.isSuperset(of: ["n9/nan@8", "n3/allnan", "n2/+inf@0,nan@last"]))
        // vDSP-style empty result 0 instead of -inf (E1).
        let emptyZero = failingCells(T16ReductionRegistry.maxE) { v in .value(v.isEmpty ? 0 : Double(T16Oracle.maximum(v))) }
        #expect(emptyZero.contains("n0/empty"))
        // A -0 magnitude (Z1).
        let negativeZero = failingCells(T16ReductionRegistry.maxMagnitudeE) { v in
            if v.isEmpty { return .value(0) }
            if v.contains(where: \.isNaN) { return .value(.nan) }
            let m = T16Oracle.maximumMagnitude(v)
            return .value(m == 0 ? -0.0 : Double(m))
        }
        #expect(negativeZero.isSuperset(of: ["n5/all-0", "n5/zeros+first"]))
        // An index scan that skips NaN.
        let skipping = failingCells(T16ReductionRegistry.minIndexE) { v in
            .index(v.isEmpty ? 0 : v.indices.dropFirst().reduce(0) { v[$1] < v[$0] ? $1 : $0 })
        }
        #expect(skipping.contains("n8/nan@3"))
        // minMax seeded with greatestFiniteMagnitude (I1).
        let seeded = failingCells(T16ReductionRegistry.minE, input: T16ReductionRegistry.blocksOfFour) { v in
            .value(Double(v.reduce(Float.greatestFiniteMagnitude) { Swift.min($0, $1) }))
        }
        #expect(seeded.isSuperset(of: ["n4/all+inf", "n0/empty"]))
        // SIMD clamped maps a NaN lane to the lower bound.
        let clamped = failingCells(T16ReductionRegistry.clipE) { v in
            let low = SIMD2<Float>(repeating: -2), high = SIMD2<Float>(repeating: 3)
            return .lanes(v.map { Double(SIMD2<Float>(repeating: $0).clamped(lowerBound: low, upperBound: high)[0]) })
        }
        #expect(clamped.isSuperset(of: ["n1/nan@0", "n9/nan@4"]))
        // Element-wise Swift.min keeps a NaN only in its first operand.
        let asymmetric = failingCells(T16ReductionRegistry.minBA) { v in
            .lanes(zip(T16Fixtures.partner(v.count), v).map { Double(Swift.min($0, $1)) })
        }
        #expect(asymmetric.contains("n4/nan@2"))
        // Agreement detects a provider that drops NaN.
        #expect(!T16Check.agree(.value(.nan), .value(1.5)))
        #expect(T16Check.agree(.value(0), .value(-0.0)))
        #expect(!T16Check.agree(.value(0), .value(-0.0), signedZero: true))
    }
}

/// The PerformanceProfiler rows mutate `PerformanceProfiler.shared`, which MixedPrecisionPhase3Tests also mutates
/// (ruling C4). They run in a child process launched by ChildProbeLauncher, in which only `profilerRowsChild` runs, so
/// no other test shares that singleton. In the parent process `profilerRowsChild` returns immediately.
@Suite("T16 PerformanceProfiler rows (isolated child, ruling C4)")
struct T16ReductionProfilerTests {
    static let selector = "T16ReductionProfilerTests/profilerRowsChild"

    @Test func profilerRowsRunInAnIsolatedChild() throws {
        var environment = ProcessInfo.processInfo.environment
        environment["SWIFT_BACKTRACE"] = "enable=no"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("T16-profiler-\(UUID().uuidString).log")
        _ = FileManager.default.createFile(atPath: url.path, contents: nil)
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        let process = try launchChildProbe(.swiftTesting(filter: Self.selector), environment: environment, output: handle,
                                           deadline: 120, cleanup: .terminateThenKill(grace: 1, seconds: 1))
        if process.timedOut {
            Issue.record("T16 profiler child deadline; capture=\(url.path) cleanupReaped=\(process.cleanup == .success)")
            return
        }
        try handle.close()
        let output = try String(contentsOf: url, encoding: .utf8)
        let termination = try process.reaped()
        for line in output.split(separator: "\n") where line.hasPrefix("T16CELLS") { print(line) }
        #expect(termination.status == 0, "profiler child failed: \(output.suffix(3000))")
        #expect(output.contains("T16_PROFILER_CHILD_COMPLETE"), "profiler child did not complete: \(output.suffix(3000))")
    }

    @Test func profilerRowsChild() async {
        guard ProcessInfo.processInfo.environment[childProbeMarkerKey] == Self.selector else { return }
        for row in T16ReductionRegistry.asyncRows {
            var failing: [String] = []
            for fixture in T16Table.fixtures {
                let want = row.expect(fixture.values)
                let got = await row.run(fixture.values)
                if !T16Check.admits(want, got) { failing.append("\(fixture.id) want \(want) got \(got)") }
            }
            T16Table.report(row.id, failing)
            #expect(failing.isEmpty, "\(row.id): \(failing.count) failing cells; first: \(failing.prefix(4))")
        }
        print("T16_PROFILER_CHILD_COMPLETE")
    }
}
