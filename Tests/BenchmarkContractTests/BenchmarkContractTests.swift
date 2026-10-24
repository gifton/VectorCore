import CryptoKit
import Foundation
import XCTest
import VectorCore
@testable import VectorCoreBenchSupport
@testable import VectorCoreBenchmarking

final class BenchmarkContractTests: XCTestCase {
    func testInvalidComparisonNeverCountsAsEvidence() throws {
        let invalid: [([Float], [Double])] = [([1], [1,2]), ([1,2], [1]), ([], []), ([.nan], [1]), ([1], [.nan]), ([.infinity], [.infinity]), ([-.infinity], [1])]
        for (a,b) in invalid {
            let stats = Correctness.vector(currents: a, references: b)
            XCTAssertEqual(stats.samples, 0, "invalid input: \(a), \(b)")
            XCTAssertNoThrow(try JSONEncoder().encode(CorrectnessOut(from: stats)))
        }
    }
}

// MARK: - Source-reading guards
//
// These tests read the benchmark suite sources through a `#filePath`-derived root. They live in this file, apart from the
// tests that write artifacts, because the repository's write-location policy (test_swift_test_output_paths.py) tracks
// `#filePath` taint per file by name, and the names these guards use collide with unrelated locals elsewhere. This file
// performs no writes.

final class BenchmarkSinkConfinementContractTests: XCTestCase {
    /// Lexical check, not dataflow: no `blackHole(` may appear between a provider range closure's `range in`
    /// and its closing `} _:`. The sink is an unsynchronized global that only the serial driver may touch.
    static func rangeClosureSinkViolations(in source: String) -> [Int] {
        var violations: [Int] = []
        var cursor = source.startIndex
        while let open = source.range(of: "range in", range: cursor..<source.endIndex) {
            guard let close = source.range(of: "} _:", range: open.upperBound..<source.endIndex) else { break }
            if source[open.upperBound..<close.lowerBound].contains("blackHole(") {
                violations.append(source[..<open.lowerBound].filter { $0 == "\n" }.count + 1)
            }
            cursor = close.upperBound
        }
        return violations
    }

    func testSinkIsConfinedToSerialDriver() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let suites = root.appendingPathComponent("Benchmarks/VectorCoreBench/Suites")
        let files = try FileManager.default.contentsOfDirectory(atPath: suites.path).filter { $0.hasSuffix(".swift") }
        XCTAssertGreaterThanOrEqual(files.count, 7)
        var closures = 0
        for file in files {
            let source = try String(contentsOf: suites.appendingPathComponent(file), encoding: .utf8)
            closures += source.components(separatedBy: "range in").count - 1
            XCTAssertEqual(Self.rangeClosureSinkViolations(in: source), [], file)
        }
        XCTAssertGreaterThan(closures, 20, "the guard must actually see the provider range closures")
        // Negative control: a sink call inside a range closure is flagged.
        let bad = "let v = try await provider.parallelReduce(items: 0..<n, initial: 0, minChunk: 1) { range in\n  blackHole(1)\n  "
            + "return 0\n} _: { $0 + $1 }"
        XCTAssertEqual(Self.rangeClosureSinkViolations(in: bad), [1])
        let good = "let v = try await provider.parallelReduce(items: 0..<n, initial: 0, minChunk: 1) { range in\n  return 0\n} _: { "
            + "$0 + $1 }\nblackHole(v)"
        XCTAssertEqual(Self.rangeClosureSinkViolations(in: good), [])
    }
}

// MARK: - V04 boundary drift guards (body fingerprints and consumer vocabulary)

final class BenchmarkBoundaryDriftContractTests: XCTestCase {
    private var benchRoot: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Benchmarks/VectorCoreBench")
    }
    private func source(_ relative: String) throws -> String { try String(contentsOf: benchRoot.appendingPathComponent(relative),
        encoding: .utf8) }
    private func sha256(_ text: String) -> String { SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined() }

    func testBodyFingerprintsMatchRecordedHashes() throws {
        XCTAssertGreaterThanOrEqual(Boundaries.bodyFingerprints.count, 18)
        var mismatches: [String] = []
        for fingerprint in Boundaries.bodyFingerprints {
            let text = try source(fingerprint.file)
            // Every listed symbol must exist; a partly missing list must not pass on the strength of the others.
            XCTAssertEqual(BodySource.missingSymbols(fingerprint.symbols, in: text), [],
                "\(fingerprint.family): symbols not found in \(fingerprint.file)")
            let stripped = BodySource.strippedSource(symbols: fingerprint.symbols, in: text)
            XCTAssertFalse(stripped.isEmpty, "\(fingerprint.family)")
            let actual = sha256(stripped)
            print("FINGERPRINT \(fingerprint.family) \(actual)")
            if actual != fingerprint.sha256 { mismatches.append("\(fingerprint.family): recorded \(fingerprint.sha256) actual \(actual)") }
        }
        XCTAssertEqual(mismatches, [],
            "A body changed (or a record was never filled in). Re-read the family's Boundaries record against the new body, then "
                + "update the hash.")
    }

    func testEveryCatalogCaseHasBodyFamiliesWithFingerprints() throws {
        let families = Set(Boundaries.bodyFingerprints.map(\.family))
        XCTAssertEqual(families.count, Boundaries.bodyFingerprints.count, "family ids are unique")
        let value = try ValidatedOptions(parseCLI(arguments: ["--suites", "dot,distance,normalize,batch,matrix,memory,noop,boundary",
            "--mode",
            "quick",
                                                              "--prefer-soa", "--use-mixed-precision"], environment: [:]))
        var checked = 0
        for suite in [DotProductBench.self as any BenchmarkSuite.Type, DistanceBench.self, NormalizationBench.self, BatchBench.self,
                      MatrixDistanceBench.self, MemoryBench.self, NoopBench.self, BoundaryBench.self] {
            for item in suite.catalog(options: value, progress: BenchmarkProgressReporter(format: .none)) {
                let owned = Boundaries.bodyFamilies(forCaseName: item.name)
                XCTAssertFalse(owned.isEmpty, item.name)
                for family in owned { XCTAssertTrue(families.contains(family), "\(item.name): unknown family \(family)") }
                // Every batch case, default or boundary, builds its fixture through BatchFixtures.
                if item.name.hasPrefix("batch.") { XCTAssertTrue(owned.contains("batch.fixtures"), item.name) }
                checked += 1
            }
        }
        XCTAssertGreaterThan(checked, 150)
    }

    func testFingerprintNegativeControls() throws {
        let fingerprint = try XCTUnwrap(Boundaries.bodyFingerprints.first { $0.family == "matrix" })
        let original = try source(fingerprint.file)
        let base = sha256(BodySource.strippedSource(symbols: fingerprint.symbols, in: original))
        // Comments and whitespace-run lengths do not change the hash.
        let reflowed = original.replacingOccurrences(of: "\n", with: " // trailing comment\n")
            .replacingOccurrences(of: "\n    ", with: "\n        ")
            .replacingOccurrences(of: "func consumedValue", with: "func   consumedValue")
            + "\n/* block comment */\n"
        XCTAssertNotEqual(reflowed, original)
        XCTAssertEqual(sha256(BodySource.strippedSource(symbols: fingerprint.symbols, in: reflowed)), base)
        // A real body edit does: restore the pre-V04 weak observation, change a constant, drop the consumer.
        let weak = original.replacingOccurrences(of: "blackHole(consumedValue(output))",
            with: "blackHole(output[0]); blackHole(output[pairs - 1])")
        XCTAssertNotEqual(weak, original)
        XCTAssertNotEqual(sha256(BodySource.strippedSource(symbols: fingerprint.symbols, in: weak)), base)
        let constant = original.replacingOccurrences(of: "unitCount: pairs", with: "unitCount: n")
        XCTAssertNotEqual(sha256(BodySource.strippedSource(symbols: fingerprint.symbols, in: constant)), base)
        // The allocator functions that run inside the mem.alloc timed body are part of the `memory` fingerprint.
        let memory = try XCTUnwrap(Boundaries.bodyFingerprints.first { $0.family == "memory" })
        XCTAssertTrue(memory.symbols.contains("alignedAllocate") && memory.symbols.contains("alignedRelease"))
        let memorySource = try source(memory.file)
        let memoryBase = sha256(BodySource.strippedSource(symbols: memory.symbols, in: memorySource))
        let noDeinit = memorySource.replacingOccurrences(of: "pointer.deinitialize(count: initialized)",
            with: "pointer.deinitialize(count: 0)")
        XCTAssertNotEqual(noDeinit, memorySource)
        XCTAssertNotEqual(sha256(BodySource.strippedSource(symbols: memory.symbols, in: noDeinit)), memoryBase)
        let otherAlignment = memorySource.replacingOccurrences(of: "count: count, alignment: alignment)\n    }",
            with: "count: count, alignment: 1)\n    }")
        XCTAssertNotEqual(otherAlignment, memorySource)
        XCTAssertNotEqual(sha256(BodySource.strippedSource(symbols: memory.symbols, in: otherAlignment)), memoryBase)
        // A partly missing symbol list is detected (not only an entirely missing one).
        XCTAssertEqual(BodySource.missingSymbols(["catalog", "noSuchFunction"], in: memorySource), ["noSuchFunction"])
        XCTAssertEqual(BodySource.missingSymbols(memory.symbols, in: memorySource), [])
        // A computed property is a supported symbol kind.
        XCTAssertFalse(BodySource.declarations(named: "statistics", in: try source("Core/ProviderObservations.swift")).isEmpty)
        XCTAssertEqual(BodySource.declarations(named: "x", in: "struct S { var x: Int { 1 } }").count, 1)
        // A symbol that does not exist yields no source (the test above fails on empty).
        XCTAssertEqual(BodySource.strippedSource(symbols: ["noSuchFunction"], in: original), "")
        // Strings containing comment markers are not truncated.
        let tricky = "func f() { let u = \"http://x\\(1)\" /* c */ ; blackHole(u) }"
        XCTAssertTrue(BodySource.strippedSource(symbols: ["f"], in: tricky).contains("http://x"))
    }

    // Closed vocabulary: sink argument text -> consumer class. Anything else fails the guard.
    private func classify(_ argument: String) -> String? {
        let scalar = try? NSRegularExpression(pattern: "^(\\w+\\.distance\\(\\w+,\\w+\\)|\\w\\.dotProduct\\(\\w\\)|acc)$")
        if scalar?.firstMatch(in: argument, range: NSRange(argument.startIndex..., in: argument)) != nil { return "scalar-sink-v1" }
        switch argument {
        case "observedValue": return "provider-float-sum-observe-sink-v1"
        case "sum", "total": return "float-sum-all-outputs-sink-v1"
        case "consumedValue(output)", "unit.withUnsafeBufferPointer{OutputFold.fold($0)}", "UInt64.zero",
             // The boundary suite's three fold forms, over the full output buffer or array. Nothing else (no rebased or sliced
             // buffer, no element read) is in the vocabulary.
             "OutputFold.fold(UnsafeBufferPointer(out))", "scores.withUnsafeBufferPointer{OutputFold.fold($0)}",
             "precomputed.withUnsafeBufferPointer{OutputFold.fold($0)}":
            return "u64-bitfold-sink-v1"
        case "Float.zero": return "error-outcome-sink-v1"
        case "unit[0]": return "error-outcome-sink-v1/unreachable-first-coordinate"
        case "pointer[0]", "destination[1]", "destination[count-1]", "handle.pointer[0]": return "escaped-pointer-retention-v1"
        default: return nil
        }
    }

    private let allowedClasses: [String: Set<String>] = [
        "scalar-sink-v1": ["scalar-sink-v1"],
        "u64-bitfold-sink-v1": ["u64-bitfold-sink-v1"],
        "error-outcome-sink-v1": ["error-outcome-sink-v1", "error-outcome-sink-v1/unreachable-first-coordinate"],
        "provider-float-sum-observe-sink-v1": ["provider-float-sum-observe-sink-v1"],
        "float-sum-all-outputs-sink-v1": ["float-sum-all-outputs-sink-v1"],
        "escaped-pointer-retention-v1": ["escaped-pointer-retention-v1"]
    ]

    /// Representative case name for a `BenchmarkCase` segment (nil if the segment cannot be mapped, which fails the guard).
    private func representative(file: String, token: String, text: String) -> String? {
        if token.hasPrefix("\"") { return String(token.dropFirst().dropLast()) }
        switch (file, token) {
        case ("Suites/MemoryBench.swift", "allocName"): return "mem.alloc.aligned.512"
        case ("Suites/MemoryBench.swift", "copyName"): return "mem.copy.512"
        case ("Suites/MemoryBench.swift", "fillName"): return "mem.fill.512"
        case ("Suites/MemoryBench.swift", "poolName"): return "mem.pool.acquire.512"
        case ("Suites/MatrixDistanceBench.swift", _): return "matrix.euclidean2.512.N17.optimized"
        case ("Suites/NoopBench.swift", _): return "noop.scalar-add"
        // Every executable companion folds all produced scores (u64-bitfold-sink-v1).
        case ("Suites/BoundaryBench.swift", _): return "batch.euclidean2.512.N257.aos-fp32-reused"
        case ("Suites/BatchBench.swift", _):
            let names: [String: String] = [
                "runBatchGeneric": "batch.euclidean2.512.N4.generic.sequential",
                    "runBatchGenericEuclid": "batch.euclidean.512.N4.generic.sequential",
                "runBatchOptimized": "batch.euclidean2.512.N4.optimized.sequential",
                    "runBatchOptimizedEuclid": "batch.euclidean.512.N4.optimized.sequential",
                "runBatchOptimizedCosineFused": "batch.cosine.512.N4.optimized-fused.sequential",
                "runBatchOptimizedCosinePreNorm": "batch.cosine.512.N4.optimized-preNorm.sequential",
                "runBatchSoA512": "batch.euclidean2.512.N4.optimized-soa", "runBatchSoA768": "batch.euclidean2.768.N4.optimized-soa",
                "runBatchSoA1536": "batch.euclidean2.1536.N4.optimized-soa",
                "runBatchFP16_512": "batch.euclidean2.512.N4.optimized-fp16", "runBatchFP16_768": "batch.euclidean2.768.N4.optimized-fp16",
                "runBatchFP16_1536": "batch.euclidean2.1536.N4.optimized-fp16",
                "runBatchCosineFP16_512": "batch.cosine.512.N4.optimized-fp16",
                    "runBatchCosineFP16_768": "batch.cosine.768.N4.optimized-fp16",
                "runBatchCosineFP16_1536": "batch.cosine.1536.N4.optimized-fp16"
            ]
            return BodySource.helperCalls(in: text).compactMap { names[$0] }.first
        default: return nil
        }
    }

    /// Returns violations for the case segments of one suite file (empty when every sink call is in the vocabulary and
    /// matches the declared consumer of the case the segment implements).
    private func consumerViolations(file: String, text: String) -> (violations: [String], segments: Int) {
        var violations: [String] = []
        let segments = BodySource.caseSegments(in: text)
        for (token, segment) in segments {
            // Unavailable entries are never invoked, so they have no body and no sink; their availability is pinned by the
            // boundary-suite selection tests.
            if segment.contains("availability: .unavailable") { continue }
            var scope = segment
            for helper in BodySource.helperCalls(in: segment) {
                scope += "\n" + BodySource.strippedSource(symbols: [helper], in: text)
            }
            guard let name = representative(file: file, token: token, text: scope),
                  let declared = Boundaries.lookup(name, runSeed: 0)?.consumer, let allowed = allowedClasses[declared] else {
                violations.append("\(file) \(token): no representative case or unknown declared consumer"); continue
            }
            let arguments = BodySource.blackHoleArguments(in: scope)
            if arguments.isEmpty { violations.append("\(file) \(token): no sink call") }
            for argument in arguments {
                guard let kind = classify(argument) else { violations.append("\(file) \(token): sink argument outside the vocabulary: "
                    + "\(argument)"); continue }
                if !allowed.contains(kind) { violations.append("\(file) \(token): \(argument) is \(kind) but \(name) declares "
                    + "\(declared)") }
            }
        }
        return (violations, segments.count)
    }

    func testConsumerTokensMatchDeclaredConsumers() throws {
        var total = 0
        for file in ["Suites/DotProductBench.swift", "Suites/DistanceBench.swift", "Suites/NormalizationBench.swift",
            "Suites/BatchBench.swift",
                     "Suites/MatrixDistanceBench.swift", "Suites/MemoryBench.swift", "Suites/NoopBench.swift",
                     "Suites/BoundaryBench.swift"] {
            let result = consumerViolations(file: file, text: try source(file))
            XCTAssertEqual(result.violations, [], file)
            total += result.segments
        }
        XCTAssertEqual(total, 95 + 9, "the guard must see every BenchmarkCase site")
    }

    /// The default batch suite and the boundary suite share one fixture builder, so "same immutable FP32 fixture" holds by
    /// construction rather than by two expressions that happen to agree.
    /// Violations of the shared-builder pin in a copy of `BatchBench.swift`: exactly 27 builder calls and no private fixture expression.
    private func sharedBuilderViolations(batch: String) -> [String] {
        var found: [String] = []
        let calls = batch.components(separatedBy: "BatchFixtures.makeFixture(").count - 1
        if calls != 27 { found.append("expected 27 BatchFixtures.makeFixture calls, found \(calls)") }
        if batch.contains("InputFactory.randomArray") { found.append("private fixture expression (InputFactory.randomArray)") }
        if batch.contains("InputSeeds.batch") { found.append("private seed expression (InputSeeds.batch)") }
        return found
    }

    func testBatchBenchBuildsItsFixtureThroughTheSharedBuilder() throws {
        let batch = try source("Suites/BatchBench.swift")
        XCTAssertEqual(sharedBuilderViolations(batch: batch), [])
        let boundary = try source("Suites/BoundaryBench.swift")
        XCTAssertEqual(boundary.components(separatedBy: "BatchFixtures.makeFixture(").count - 1, 7)
        XCTAssertFalse(boundary.contains("InputFactory.randomArray(count: dim, seed: InputSeeds.batch"))
        // Negative controls on real copies of the source: a site that rebuilds its own fixture, a stray private seed, a lost call.
        let site = "let fixture = try BatchFixtures.makeFixture(dim: 512, count: n, runSeed: options.runSeed) { try "
            + "Vector512Optimized($0) }"
        XCTAssertTrue(batch.contains(site))
        let rebuilt = batch.replacingOccurrences(of: site,
            with: "let fixture = (query: try Vector512Optimized(InputFactory.randomArray(count: 512, seed: 1)), "
                                                  + "candidates: [Vector512Optimized]())")
        XCTAssertFalse(sharedBuilderViolations(batch: rebuilt).isEmpty)
        let stray = batch + "\nlet strayFixtureSeed = InputSeeds.batch(dim: 512, n: 1, runSeed: 0)\n"
        XCTAssertFalse(sharedBuilderViolations(batch: stray).isEmpty)
        let lost = batch.replacingOccurrences(of: site, with: "let fixture = try BatchFixtures.makeFixtureElsewhere(dim: 512)",
            options: [], range: batch.range(of: site))
        XCTAssertFalse(sharedBuilderViolations(batch: lost).isEmpty)
    }

    func testConsumerGuardNegativeControls() throws {
        // Matrix: the pre-V04 two-diagonal observation is outside the vocabulary and is rejected.
        let matrix = try source("Suites/MatrixDistanceBench.swift")
        let weak = matrix.replacingOccurrences(of: "blackHole(consumedValue(output))",
            with: "blackHole(output[0]); blackHole(output[pairs - 1])")
        XCTAssertNotEqual(weak, matrix)
        XCTAssertFalse(consumerViolations(file: "Suites/MatrixDistanceBench.swift", text: weak).violations.isEmpty)
        // Normalize: folding replaced by the single-coordinate observation under a success label is rejected.
        let normalize = try source("Suites/NormalizationBench.swift")
        let partial = normalize.replacingOccurrences(of: "blackHole(unit.withUnsafeBufferPointer { OutputFold.fold($0) })",
            with: "blackHole(unit[0])")
        XCTAssertNotEqual(partial, normalize)
        XCTAssertFalse(consumerViolations(file: "Suites/NormalizationBench.swift", text: partial).violations.isEmpty)
        // Batch: a SoA body that sinks the handle instead of the sum is rejected.
        let batch = try source("Suites/BatchBench.swift")
        let handle = batch.replacingOccurrences(of: "blackHole(total)", with: "blackHole(resultsPtr)")
        XCTAssertNotEqual(handle, batch)
        XCTAssertFalse(consumerViolations(file: "Suites/BatchBench.swift", text: handle).violations.isEmpty)
        // Boundary suite: partial folds and single-element reads are outside the vocabulary.
        let boundary = try source("Suites/BoundaryBench.swift")
        for replacement in ["OutputFold.fold(UnsafeBufferPointer(rebasing: out[0..<1]))",
            "OutputFold.fold(UnsafeBufferPointer(rebasing: out[0..<2]))",
                            "out[0]"] {
            let partialFold = boundary.replacingOccurrences(of: "OutputFold.fold(UnsafeBufferPointer(out))", with: replacement)
            XCTAssertNotEqual(partialFold, boundary)
            XCTAssertFalse(consumerViolations(file: "Suites/BoundaryBench.swift", text: partialFold).violations.isEmpty, replacement)
        }
        let partialScores = boundary.replacingOccurrences(of: "scores.withUnsafeBufferPointer { OutputFold.fold($0) }",
                                                          with: "scores.prefix(1).withUnsafeBufferPointer { OutputFold.fold($0) }")
        XCTAssertNotEqual(partialScores, boundary)
        XCTAssertFalse(consumerViolations(file: "Suites/BoundaryBench.swift", text: partialScores).violations.isEmpty)
        XCTAssertEqual(consumerViolations(file: "Suites/BoundaryBench.swift", text: boundary).violations, [])
        // A case with no sink call at all is rejected.
        let noop = try source("Suites/NoopBench.swift").replacingOccurrences(of: "blackHole(acc)", with: "_ = acc")
        XCTAssertFalse(consumerViolations(file: "Suites/NoopBench.swift", text: noop).violations.isEmpty)
        // Fingerprint controls for the boundary-suite families: a kernel route edit and a fixture edit each change a hash.
        let boundarySuite = try XCTUnwrap(Boundaries.bodyFingerprints.first { $0.family == "boundary" })
        let boundaryText = try source(boundarySuite.file)
        let boundaryBase = sha256(BodySource.strippedSource(symbols: boundarySuite.symbols, in: boundaryText))
        for (old, new) in [("BatchKernels.range_euclid2_768(query", "BatchKernels.range_euclid2_512(query"),
                           ("r.soa(f.query, soa, out)", "r.soa(f.query, soa, out); r.soa(f.query, soa, out)"),
                           ("blackHole(OutputFold.fold(UnsafeBufferPointer(out)))", "blackHole(0)")] {
            let mutated = boundaryText.replacingOccurrences(of: old, with: new)
            XCTAssertNotEqual(mutated, boundaryText, old)
            XCTAssertNotEqual(sha256(BodySource.strippedSource(symbols: boundarySuite.symbols, in: mutated)), boundaryBase, old)
        }
        let fixtures = try XCTUnwrap(Boundaries.bodyFingerprints.first { $0.family == "batch.fixtures" })
        let fixturesText = try source(fixtures.file)
        let fixturesBase = sha256(BodySource.strippedSource(symbols: fixtures.symbols, in: fixturesText))
        let shiftedSeed = fixturesText.replacingOccurrences(of: "seeds.baseCandidates &+ UInt64($0)",
            with: "seeds.baseCandidates &+ UInt64($0 + 1)")
        XCTAssertNotEqual(shiftedSeed, fixturesText)
        XCTAssertNotEqual(sha256(BodySource.strippedSource(symbols: fixtures.symbols, in: shiftedSeed)), fixturesBase)
        // Unmodified sources pass (control for the controls).
        XCTAssertEqual(consumerViolations(file: "Suites/MatrixDistanceBench.swift", text: matrix).violations, [])
    }

    func testAllocationFlagsDoNotOverclaim() throws {
        func boundary(_ name: String) throws -> MeasurementBoundary { try XCTUnwrap(Boundaries.lookup(name, runSeed: 0), name) }
        // An FP16 note matches when it names exactly one T12 record set (the expected one, with its record path and date), no
        // V04 record, and contains every literal, each anchored on a digit/comma boundary (`containsCountLiteral`).
        func fp16NoteMatches(_ note: String, set: String, literals: [String]) -> Bool {
            let named = ["fp16-boundary", "fp16-default"].filter { note.contains("T12/phase2/D4/final/\($0)/") }
            return named == [set] && note.contains("T12/phase2/D4/final/\(set)/manifest.json") && note.contains("2026-10-04")
                && !note.contains("V04/phaseD/D4/") && literals.allSatisfy { Self.containsCountLiteral(note, $0) }
        }
        let reusedLiterals = ["1 event per invocation", "of 32,800 bytes", "no other event"]
        let setupLiterals = ["4,099 events per invocation", "4,096 of 1,056 bytes at D = 512 (1,568 at D = 768, 3,104 at D = 1536)",
                             "two of 32,800 bytes and one of 16,416 bytes"]
        let defaultLiterals = ["1 allocation event per invocation", "65,568 bytes at D = 512 and 768 with N = 8192",
                               "32,800 bytes at D = 1536 with N = 4096", "one 1,016-byte event per timed window"]
        for dim in [512, 768, 1536] {
            // The range_*_mixed kernels build a temporary array of candidate handles on every call.
            XCTAssertEqual(try boundary("batch.euclidean2.\(dim).N4.optimized-fp16").inclusions.routeInternalAllocation, .included)
            XCTAssertEqual(try boundary("batch.cosine.\(dim).N4.optimized-fp16").inclusions.routeInternalAllocation, .included)
            // N = 4 was not probed at any width.
            XCTAssertEqual(try boundary("batch.euclidean2.\(dim).N4.optimized-soa").inclusions.routeInternalAllocation, .unverified)
            // The V04 phase D probe resolved these at D = 512 only.
            XCTAssertEqual(try boundary("normalize.zeroFail.\(dim).optimized").inclusions.routeInternalAllocation,
                           dim == 512 ? .included : .unverified)
            XCTAssertEqual(try boundary("normalize.success.\(dim).generic").inclusions.routeInternalAllocation,
                           dim == 512 ? .included : .unverified)
            XCTAssertEqual(try boundary("dot.\(dim).optimized").inclusions.routeInternalAllocation,
                           dim == 512 ? .notApplicable : .unverified)
            XCTAssertEqual(try boundary("matrix.euclidean2.\(dim).N17.optimized").inclusions.routeInternalAllocation, .included)
            // The FP16 companions name the T12 re-probe counts at every width.
            let half = try boundary("batch.euclidean2.\(dim).N4.aos-fp16-reused")
            let halfSetup = try boundary("batch.euclidean2.\(dim).N4.aos-fp16-setup")
            XCTAssertEqual(half.inclusions.routeInternalAllocation, .included)
            XCTAssertEqual(halfSetup.inclusions.routeInternalAllocation, .included)
            XCTAssertTrue(half.notes.contains(Boundaries.AllocationProbe.fp16Reused))
            XCTAssertTrue(halfSetup.notes.contains(Boundaries.AllocationProbe.fp16Setup))
            // Independent literals written by hand from the T12 phase 2 D4 counts (fp16-boundary set, every width: reused 16
            // events at 16 invocations and 32 at 32, all of 32,800 bytes; setup 65,584 and 131,168, of which 65,536 and 131,072
            // of 1,056, 1,568 or 3,104 bytes), so a wrong count or record in the shared notes fails here.
            XCTAssertEqual(half.notes.filter { fp16NoteMatches($0, set: "fp16-boundary", literals: reusedLiterals) }.count, 1,
                           "\(dim) fp16Reused")
            XCTAssertEqual(halfSetup.notes.filter { fp16NoteMatches($0, set: "fp16-boundary", literals: setupLiterals) }.count, 1,
                           "\(dim) fp16Setup")
            // The default optimized-fp16 rows, both metrics: literals by hand from the fp16-default set (17 events at 16
            // invocations and 33 at 32 at every probed width: one per invocation plus one 1,016-byte event per window).
            for metric in ["euclidean2", "cosine"] {
                let row = try boundary("batch.\(metric).\(dim).N4.optimized-fp16")
                XCTAssertEqual(row.notes.filter { fp16NoteMatches($0, set: "fp16-default", literals: defaultLiterals) }.count, 1,
                               "\(dim) \(metric) fp16DefaultRow")
            }
        }
        // Negative controls: the FP16 matcher rejects a note that names the other set, also names a V04 record, or states
        // another count.
        let reused = Boundaries.AllocationProbe.fp16Reused
        XCTAssertTrue(fp16NoteMatches(reused, set: "fp16-boundary", literals: reusedLiterals))
        for mutated in [reused.replacingOccurrences(of: "D4/final/fp16-boundary/", with: "D4/final/fp16-default/"),
                        reused + " See also V04/phaseD/D4/primary/manifest.json.",
                        reused.replacingOccurrences(of: "1 event per invocation", with: "2 events per invocation")] {
            XCTAssertNotEqual(mutated, reused)
            XCTAssertFalse(fp16NoteMatches(mutated, set: "fp16-boundary", literals: reusedLiterals), mutated)
        }
        // Count-boundary controls for each FP16 note (`checkFP16CountBoundaries`).
        checkFP16CountBoundaries(fp16NoteMatches, reused: reusedLiterals, setup: setupLiterals, defaultRow: defaultLiterals)
        // V04 phase D (D4) resolved exactly these case IDs: value, record set and observation of each resolved note against
        // an independent table, with negative controls (`checkResolvedAllocationNotes`).
        try checkResolvedAllocationNotes(boundary)
        // Unprobed neighbours keep `unverified`: other widths, N on both sides of the SoA tiling threshold, other memory sizes,
        // the automatic provider, and both sides of the generic pairs whose squared side the probe measured.
        for name in ["batch.euclidean2.768.N4096.aos-fp32-reused", "batch.euclidean2.512.N257.aos-fp32-setup",
                     "batch.euclidean2.512.N4095.soa-fp32-reused", "batch.euclidean2.512.N4097.optimized-soa",
                     "batch.euclidean2.1536.N8192.soa-fp32-reused", "dist.cosine.768.optimized", "normalize.zeroFail.1536.generic",
                     "mem.copy.511", "mem.pool.acquire.513", "batch.euclidean2.512.N4096.generic.automatic",
                     "batch.euclidean2.512.N4096.generic.sequential", "batch.euclidean.512.N4096.generic.parallel"] {
            XCTAssertEqual(try boundary(name).inclusions.routeInternalAllocation, .unverified, name)
        }
        for name in ["batch.euclidean2.512.N257.soa-fp32-reused", "batch.euclidean2.512.N4097.optimized-soa",
                     "batch.euclidean2.768.N100.optimized-soa"] {
            XCTAssertTrue(try boundary(name).notes.contains(Boundaries.AllocationProbe.soaUnprobed), name)
        }
        // Independent literals for the unprobed-SoA note, with count-boundary controls (`checkSoAUnprobedLiterals`).
        checkSoAUnprobedLiterals()
        // The generic pairs keep identical notes and inclusions on both sides, so the declared A/B pair still forms.
        for provider in ["sequential", "parallel"] {
            let left = "batch.euclidean2.512.N4096.generic.\(provider)", right = "batch.euclidean.512.N4096.generic.\(provider)"
            let l = try boundary(left), r = try boundary(right)
            XCTAssertEqual(l.notes, r.notes, provider); XCTAssertEqual(l.inclusions, r.inclusions, provider)
            XCTAssertTrue(l.notes.contains { $0.contains("V04/phaseD/D4/unverified-a/manifest.json") }, provider)
            // Independent literals from the D4 counts for the squared side (unverified-a set): 196,642 events at 16 and
            // 393,282 at 32 invocations (12,290 per added invocation) sequentially; 199,356 and 398,720 in parallel.
            let counts = provider == "sequential" ? ["12,290 allocation events per invocation"] : ["199,356", "398,720"]
            func pairNoteMatches(_ note: String) -> Bool {
                counts.allSatisfy { Self.containsCountLiteral(note, $0) }
                    && note.contains("record V04/phaseD/D4/unverified-a/manifest.json, 2026-10-04")
            }
            XCTAssertEqual(l.notes.filter(pairNoteMatches).count, 1, provider)
            // Count-boundary controls: one count gains an extra leading digit, another an extra trailing digit.
            let pairNote = try XCTUnwrap(Boundaries.AllocationProbe.genericPair(dim: 512, n: 4096, provider: provider), provider)
            XCTAssertTrue(pairNoteMatches(pairNote), provider)
            let edits = provider == "sequential"
                ? [("): 12,290 allocation", "): 112,290 allocation"), ("12,290 allocation", "12,2901 allocation")]
                : [("): 199,356 allocation", "): 1199,356 allocation"), ("398,720 at 32", "398,7201 at 32")]
            for (old, new) in edits {
                let mutated = pairNote.replacingOccurrences(of: old, with: new)
                XCTAssertNotEqual(mutated, pairNote, old)
                XCTAssertFalse(pairNoteMatches(mutated), mutated)
            }
            let report = try ABComparisons.report(cases: JSONWriter.toCases([
                BenchResult(name: left, iterations: 1, totalNanoseconds: 100, unitCount: l.unitsPerInvocation, boundary: l),
                BenchResult(name: right, iterations: 1, totalNanoseconds: 100, unitCount: r.unitsPerInvocation, boundary: r)]))
            XCTAssertEqual(report.comparisons.count, 1, provider)
        }
        XCTAssertEqual(try boundary("mem.copy.512").inclusions.callerOutputAllocation, .excluded)
        XCTAssertEqual(try boundary("mem.fill.512").inclusions.callerOutputAllocation, .excluded)
        XCTAssertEqual(try boundary("mem.alloc.aligned.512").inclusions.callerOutputAllocation, .included)
        // Absent validation is notApplicable with a note, uniformly.
        for name in ["dot.512.generic", "normalize.success.512.generic", "mem.copy.512", "noop.scalar-add"] {
            let b = try boundary(name)
            XCTAssertEqual(b.inclusions.correctnessValidation, .notApplicable, name)
            XCTAssertTrue(b.notes.contains { $0.contains("correctnessValidation is notApplicable") }, name)
        }
        // Every `unverified` flag is accompanied by a note that names the flag.
        let value = try ValidatedOptions(parseCLI(arguments: ["--suites", "dot,distance,normalize,batch,matrix,memory,noop,boundary",
            "--mode",
            "quick",
                                                              "--prefer-soa", "--use-mixed-precision"], environment: [:]))
        for suite in [DotProductBench.self as any BenchmarkSuite.Type, DistanceBench.self, NormalizationBench.self, BatchBench.self,
                      MatrixDistanceBench.self, MemoryBench.self, NoopBench.self, BoundaryBench.self] {
            for item in suite.catalog(options: value, progress: BenchmarkProgressReporter(format: .none)) {
                let flags = item.boundary.flattened().filter { $0.key.hasPrefix("inclusions.") && $0.value == "\"unverified\"" }.keys
                for flag in flags {
                    let field = flag.replacingOccurrences(of: "inclusions.", with: "")
                    XCTAssertTrue(item.boundary.notes.contains { $0.contains("\(field) unverified") }, "\(item.name) \(field)")
                }
                // A resolved-allocation note appears exactly on the resolved case IDs.
                let resolvedNotes = item.boundary.notes.filter { $0.hasPrefix("routeInternalAllocation ") }
                XCTAssertEqual(resolvedNotes.isEmpty, Boundaries.AllocationProbe.resolutions[item.name] == nil, item.name)
            }
        }
        // The enum round-trips through JSON.
        let encoded = try JSONEncoder().encode(Inclusion.unverified)
        XCTAssertEqual(try JSONDecoder().decode(Inclusion.self, from: encoded), .unverified)
    }
}

extension BenchmarkBoundaryDriftContractTests {
    /// Count-boundary controls for each FP16 note, called by `testAllocationFlagsDoNotOverclaim` with its FP16 matcher:
    /// in each note one count gains an extra leading digit and another an extra trailing digit, and the matcher must
    /// reject both.
    private func checkFP16CountBoundaries(_ matches: (String, String, [String]) -> Bool, reused: [String], setup: [String],
                                          defaultRow: [String]) {
        let reusedNote = Boundaries.AllocationProbe.fp16Reused, setupNote = Boundaries.AllocationProbe.fp16Setup
        let defaultNote = Boundaries.AllocationProbe.fp16DefaultRow(metric: "cosine")
        XCTAssertTrue(matches(setupNote, "fp16-boundary", setup))
        XCTAssertTrue(matches(defaultNote, "fp16-default", defaultRow))
        for (note, set, literals, old, new) in [
            (reusedNote, "fp16-boundary", reused, "recorded 1 event per invocation", "recorded 11 event per invocation"),
            (reusedNote, "fp16-boundary", reused, "of 32,800 bytes", "of 32,8001 bytes"),
            (setupNote, "fp16-boundary", setup, "recorded 4,099 events", "recorded 14,099 events"),
            (setupNote, "fp16-boundary", setup, "one of 16,416 bytes", "one of 16,4161 bytes"),
            (defaultNote, "fp16-default", defaultRow, "recorded 1 allocation event", "recorded 21 allocation event"),
            (defaultNote, "fp16-default", defaultRow, "with N = 4096 (by size", "with N = 40961 (by size")] {
            let mutated = note.replacingOccurrences(of: old, with: new)
            XCTAssertNotEqual(mutated, note, old)
            XCTAssertFalse(matches(mutated, set, literals), mutated)
        }
    }

    /// Independent literals for the unprobed-SoA note, called by `testAllocationFlagsDoNotOverclaim`: both record paths
    /// and the two probed N, counts anchored on a digit/comma boundary; controls with an extra leading digit and an extra
    /// trailing digit must be rejected.
    private func checkSoAUnprobedLiterals() {
        let note = Boundaries.AllocationProbe.soaUnprobed
        let literals = ["V04/phaseD/D4/primary/manifest.json", "V04/phaseD/D4/unverified-a/manifest.json",
                        "no allocation events in the timed window at N = 4096", "32 events of 4,096 bytes per invocation at N = 8192"]
        func matches(_ text: String) -> Bool { literals.allSatisfy { Self.containsCountLiteral(text, $0) } }
        XCTAssertTrue(matches(note))
        for (old, new) in [("and 32 events of 4,096", "and 132 events of 4,096"), ("at N = 8192, where", "at N = 81920, where")] {
            let mutated = note.replacingOccurrences(of: old, with: new)
            XCTAssertNotEqual(mutated, note, old)
            XCTAssertFalse(matches(mutated), mutated)
        }
    }

    /// Whether `text` contains `literal` at a position where a count at either end of the literal cannot continue: a
    /// literal that starts with a digit must not follow a digit, or a comma that follows a digit ("14,099" does not match
    /// "4,099 events"); a literal that ends with a digit must not precede a digit, or a comma that precedes a digit
    /// ("N = 40961" does not match "N = 4096"). The count-boundary controls in `checkFP16CountBoundaries`,
    /// `checkSoAUnprobedLiterals`, the generic-pair block of `testAllocationFlagsDoNotOverclaim` and
    /// `checkResolvedAllocationNotes` exercise both ends.
    static func containsCountLiteral(_ text: String, _ literal: String) -> Bool {
        let characters = Array(text), wanted = Array(literal)
        guard let first = wanted.first, let last = wanted.last, wanted.count <= characters.count else { return false }
        func isDigit(_ character: Character) -> Bool { character.isASCII && character.isNumber }
        // A digit, or a comma whose far side is a digit (a digit group), continues a count.
        func continuesCount(at index: Int, step: Int) -> Bool {
            guard characters.indices.contains(index) else { return false }
            if isDigit(characters[index]) { return true }
            return characters[index] == "," && characters.indices.contains(index + step) && isDigit(characters[index + step])
        }
        for start in 0...(characters.count - wanted.count)
        where characters[start..<(start + wanted.count)].elementsEqual(wanted) {
            if isDigit(first) && continuesCount(at: start - 1, step: -1) { continue }
            if isDigit(last) && continuesCount(at: start + wanted.count, step: 1) { continue }
            return true
        }
        return false
    }

    /// The 28 resolved case IDs of V04 phase D (D4), checked against a table written by hand from the D4 counts; called
    /// by `testAllocationFlagsDoNotOverclaim`.
    private func checkResolvedAllocationNotes(_ boundary: (String) throws -> MeasurementBoundary) throws {
        // V04 phase D (D4) resolved exactly these case IDs. Independent table: the value, the record set and an observation
        // literal written by hand from the D4 timed-window counts at 16 invocations (D4-summary.json, 2026-10-04): 0 events;
        // 16 of 16,416 B; 512 of 4,096 B; 48 (136, 64, 96 B); 32 (2,048, 40 B); 16 of 2,080 B; 112 (80 of 40 B, 32 of 80 B).
        let none = "no allocation events in the timed window"
        var resolved: [String: (value: Inclusion, record: String, observation: String)] = [
            "batch.euclidean2.512.N4096.aos-fp32-reused": (.notApplicable, "primary", none),
            "batch.euclidean2.512.N4096.soa-fp32-reused": (.notApplicable, "primary", none),
            "batch.euclidean2.512.N4096.aos-fp32-setup": (.notApplicable, "primary", "1 allocation event per invocation, of 16,416 bytes"),
            "batch.euclidean2.512.N4096.optimized-soa": (.notApplicable, "unverified-a", none),
            "mem.copy.512": (.notApplicable, "unverified-a", none),
            "batch.euclidean2.512.N8192.soa-fp32-reused": (.included, "unverified-a", "32 allocation events of 4,096 bytes per invocation"),
            "batch.euclidean2.512.N8192.optimized-soa": (.included, "unverified-a", "32 allocation events of 4,096 bytes per invocation"),
            "normalize.zeroFail.512.generic": (.included, "unverified-a", "3 allocation events per invocation (136, 64 and 96 bytes)"),
            "normalize.zeroFail.512.optimized": (.included, "unverified-a", "3 allocation events per invocation (136, 64 and 96 bytes)"),
            "normalize.success.512.generic": (.included, "unverified-a", "2 allocation events per invocation (2,048 and 40 bytes)"),
            "normalize.success.512.optimized": (.included, "unverified-a", "1 allocation event per invocation (2,080 bytes)"),
            "mem.pool.acquire.512": (.included, "unverified-a",
                                     "7 allocation events per invocation (five of 40 bytes, two of 80 bytes)"),
            "dot.512.generic": (.notApplicable, "unverified-b", none), "dot.512.optimized": (.notApplicable, "unverified-b", none)]
        for metric in ["euclidean", "cosine", "manhattan", "dot", "chebyshev", "hamming", "minkowski"] {
            for variant in ["generic", "optimized"] { resolved["dist.\(metric).512.\(variant)"] = (.notApplicable, "unverified-b", none) }
        }
        XCTAssertEqual(resolved.count, 28)
        XCTAssertEqual(Boundaries.AllocationProbe.resolutions.mapValues(\.value), resolved.mapValues { $0.value })
        // A resolved note matches when it states the value, names exactly one record set (the expected one, with the
        // record path and date) and contains the observation, anchored on a digit/comma boundary (`containsCountLiteral`).
        func noteMatches(_ note: String, value: Inclusion, record: String, observation: String) -> Bool {
            let named = ["primary", "unverified-a", "unverified-b"].filter { note.contains("V04/phaseD/D4/\($0)/") }
            return note.hasPrefix("routeInternalAllocation \(value.rawValue): ") && named == [record]
                && note.contains("record V04/phaseD/D4/\(record)/manifest.json, 2026-10-04")
                && Self.containsCountLiteral(note, observation)
        }
        for (name, expected) in resolved {
            let b = try boundary(name)
            XCTAssertEqual(b.inclusions.routeInternalAllocation, expected.value, name)
            let notes = b.notes.filter { $0.hasPrefix("routeInternalAllocation ") }
            XCTAssertEqual(notes.count, 1, name)
            XCTAssertTrue(notes.allSatisfy { noteMatches($0, value: expected.value, record: expected.record,
                                                         observation: expected.observation) }, "\(name): \(notes)")
            XCTAssertFalse(b.notes.contains { $0.contains("routeInternalAllocation unverified") }, name)
        }
        // Negative controls: the matcher rejects a note that names another set, names a second set, or states another count
        // (including a count with an extra leading digit or an extra trailing digit).
        let primary = try XCTUnwrap(Boundaries.AllocationProbe.resolutions["batch.euclidean2.512.N4096.aos-fp32-setup"]?.note)
        let setupObservation = "1 allocation event per invocation, of 16,416 bytes"
        XCTAssertTrue(noteMatches(primary, value: .notApplicable, record: "primary", observation: setupObservation))
        for mutated in [primary.replacingOccurrences(of: "D4/primary/", with: "D4/unverified-b/"),
                        primary + " See also V04/phaseD/D4/unverified-a/manifest.json.",
                        primary.replacingOccurrences(of: "1 allocation event per invocation", with: "2 allocation events per invocation"),
                        primary.replacingOccurrences(of: ": 1 allocation event", with: ": 11 allocation event"),
                        primary.replacingOccurrences(of: "of 16,416 bytes", with: "of 16,4161 bytes")] {
            XCTAssertNotEqual(mutated, primary)
            XCTAssertFalse(noteMatches(mutated, value: .notApplicable, record: "primary", observation: setupObservation), mutated)
        }
        XCTAssertFalse(noteMatches(primary, value: .included, record: "primary", observation: setupObservation))
    }
}

extension BenchmarkBoundaryDriftContractTests {
    private static let suiteFiles = ["Suites/DotProductBench.swift", "Suites/DistanceBench.swift", "Suites/NormalizationBench.swift",
                                     "Suites/BatchBench.swift", "Suites/MatrixDistanceBench.swift", "Suites/MemoryBench.swift",
                                     "Suites/NoopBench.swift", "Suites/BoundaryBench.swift"]

    /// Differences between a case's warmup and timed body that are intentional, per file, as (pattern, replacement) rewrites
    /// applied to both bodies before comparing. mem.alloc and mem.pool.acquire pass the value written into the allocation
    /// (0 or 1 for warmup, 1 or 2 for timing); the helper they run is the same.
    private static let justifiedRewrites: [String: [(String, String)]] = [
        "Suites/MemoryBench.swift": [("allocateOnce\\(\\d+\\)", "allocateOnce(_)"), ("acquire\\(\\d+\\)", "acquire(_)")]
    ]

    private func normalized(_ body: String, file: String) -> String {
        var text = body
        for (pattern, replacement) in Self.justifiedRewrites[file] ?? [] {
            text = text.replacingOccurrences(of: pattern, with: replacement, options: .regularExpression)
        }
        return text
    }

    /// Units that contain a warmup and a measurement: every `BenchmarkCase` segment, every `run*` helper function and the
    /// matrix `measure` helper. A unit that only delegates to a helper has neither call and is skipped.
    private func divergences(file: String, text: String) -> (violations: [String], compared: Int) {
        var units: [(String, String)] = BodySource.caseSegments(in: text).map { ($0.token, $0.text) }
        let stripped = BodySource.stripComments(text)
        if let regex = try? NSRegularExpression(pattern: "\\bfunc\\s+(run[A-Z]\\w*|measure)\\s*[<(]") {
            let names = Set(regex.matches(in: stripped, range: NSRange(stripped.startIndex..., in: stripped)).map {
                (stripped as NSString).substring(with: $0.range(at: 1))
            })
            for name in names.sorted() {
                for (index, declaration) in BodySource.declarations(named: name, in: text).enumerated() { units.append(("\(name)#\(index)",
                    declaration)) }
            }
        }
        var violations: [String] = [], compared = 0
        for (label, unit) in units {
            let warmups = BodySource.harnessBodies(callee: "warmup", in: unit) + BodySource.harnessBodies(callee: "warmupAsync", in: unit)
            let measures = BodySource.harnessBodies(callee: "measure", in: unit) + BodySource.harnessBodies(callee: "measureAsync",
                in: unit)
            if warmups.isEmpty && measures.isEmpty { continue }
            guard warmups.count == 1, measures.count == 1, !warmups[0].isEmpty, !measures[0].isEmpty else {
                violations.append("\(file) \(label): expected exactly one warmup and one measurement body, found \(warmups.count) and "
                    + "\(measures.count)")
                continue
            }
            compared += 1
            if normalized(warmups[0], file: file) != normalized(measures[0], file: file) {
                violations.append("\(file) \(label): warmup and timed bodies differ")
            }
        }
        return (violations, compared)
    }

    func testWarmupAndTimedBodiesRunTheSameWork() throws {
        var compared = 0
        for file in Self.suiteFiles {
            let result = divergences(file: file, text: try source(file))
            XCTAssertEqual(result.violations, [], file)
            compared += result.compared
        }
        // 6 dot + 42 distance + 12 normalize + 1 noop + 4 memory cases carry their bodies in the catalogs; the 27 batch cases
        // delegate to 23 `run*` helper functions (1+1+3+3+3+3+3+3+3 overloads); plus the matrix `measure` helper and the
        // boundary suite's shared `runMeasured`.
        XCTAssertEqual(compared, 6 + 42 + 12 + 1 + 4 + 23 + 1 + 1, "the guard must compare every body it is meant to")
    }

    func testWarmupTimedGuardNegativeControls() throws {
        // The original M1 defect: a warmup that sinks the handle while the timed body folds the result.
        let normalize = try source("Suites/NormalizationBench.swift")
        let handleWarmup = normalize.replacingOccurrences(of: "try Harness.warmup(body)",
            with: "try Harness.warmup { blackHole(try? vector.normalized().get()) }")
        XCTAssertNotEqual(handleWarmup, normalize)
        XCTAssertFalse(divergences(file: "Suites/NormalizationBench.swift", text: handleWarmup).violations.isEmpty)
        // Matrix: warmup replaced by an empty closure.
        let matrix = try source("Suites/MatrixDistanceBench.swift")
        let emptyWarmup = matrix.replacingOccurrences(of: "try Harness.warmup(once)", with: "try Harness.warmup { }")
        XCTAssertNotEqual(emptyWarmup, matrix)
        XCTAssertFalse(divergences(file: "Suites/MatrixDistanceBench.swift", text: emptyWarmup).violations.isEmpty)
        // Batch: extra work in one SoA warmup closure.
        var batch = try source("Suites/BatchBench.swift")
        let helper = try XCTUnwrap(batch.range(of: "func runBatchSoA512"))
        let warmup = try XCTUnwrap(batch.range(of: "Harness.warmupAsync {", range: helper.upperBound..<batch.endIndex))
        batch.replaceSubrange(warmup, with: "Harness.warmupAsync { blackHole(1);")
        XCTAssertFalse(divergences(file: "Suites/BatchBench.swift", text: batch).violations.isEmpty)
        // Memory: copy warmup no longer shares the timed closure; a plain value change in the justified position is allowed.
        let memory = try source("Suites/MemoryBench.swift")
        let split = memory.replacingOccurrences(of: "try Harness.warmup(copy)", with: "try Harness.warmup { blackHole(destination[1]) }")
        XCTAssertNotEqual(split, memory)
        XCTAssertFalse(divergences(file: "Suites/MemoryBench.swift", text: split).violations.isEmpty)
        let valueChange = memory.replacingOccurrences(of: "try Harness.warmup { try allocateOnce(0) }",
            with: "try Harness.warmup { try allocateOnce(7) }")
        XCTAssertNotEqual(valueChange, memory)
        XCTAssertEqual(divergences(file: "Suites/MemoryBench.swift", text: valueChange).violations, [])
        // A missing warmup or a second measurement is reported rather than skipped.
        let noWarmup = matrix.replacingOccurrences(of: "try Harness.warmup(once)\n", with: "")
        XCTAssertFalse(divergences(file: "Suites/MatrixDistanceBench.swift", text: noWarmup).violations.isEmpty)
        // Control for the controls.
        XCTAssertEqual(divergences(file: "Suites/MatrixDistanceBench.swift", text: matrix).violations, [])
    }
}

// MARK: - T11 correctness gate (derived thresholds, zero-reference floor, validation declarations)
//
// Synchronous tests: they compute the routes' outputs directly, read benchmark sources through the `#filePath` root and
// evaluate the gate on synthetic results. They launch no process and write nothing.

final class BenchmarkCorrectnessPolicyTests: XCTestCase {
    private static let u32 = 0x1p-24
    private static let u64 = 0x1p-53
    private static let margin = 0x1p-40
    /// The widths and candidate counts of the guide's CI recipe (`--suites batch --profile short`).
    private static let ciDims = [512, 768, 1536]
    private static let ciCounts = [100, 1_000]

    /// Restatements of γ_k and of the two threshold forms, written independently of `RoundingBound` and `ErrorModel`.
    private static func gamma(_ k: Int, _ u: Double) -> Double { Double(k) * u / (1 - Double(k) * u) }
    private static func relativeThreshold(k32: Int, k64: Int) -> Double {
        (gamma(k32, u32) + gamma(k64, u64)) / (1 - gamma(k64, u64)) * (1 + margin)
    }
    private static func absoluteThreshold(k32: Int, k64: Int, scale: Double) -> Double {
        (gamma(k32, u32) + gamma(k64, u64)) * scale * (1 + margin)
    }
    private static func relativeError(_ current: Float, _ reference: Double) -> Double { abs(Double(current) - reference) / abs(reference) }

    private func result(_ name: String, _ correctness: CorrectnessStats?) -> BenchResult {
        BenchResult(name: name, iterations: 1, totalNanoseconds: 100, correctness: correctness)
    }
    private func stats(maxRel: Double, maxAbs: Double) -> CorrectnessStats {
        CorrectnessStats(samples: 1, maxAbsError: maxAbs, meanAbsError: maxAbs, maxRelError: maxRel, meanRelError: maxRel)
    }
    private func gate(_ result: BenchResult, rel: Double? = nil, abs: Double? = nil) -> BenchmarkFailure? {
        CorrectnessPolicy.failure(for: result, overrideRel: rel, overrideAbs: abs)
    }

    // MARK: Derived thresholds

    func testDerivedThresholdsFollowTheRoundingCounts() throws {
        struct Expected { let name: String; let route: String; let k32: Int; let k64: Int; let absolute: Bool }
        var checked = 0
        for dim in Self.ciDims {
            for n in [1, 2, 3, 100, 257, 1_000, 10_000] {
                let s = n - 1  // an FP32 sum of the n provider results adds at most n - 1 factors to a term
                var expected: [Expected] = []
                for mode in ["sequential", "parallel", "automatic", "injected"] {
                    expected += [
                        Expected(name: "batch.euclidean2.\(dim).N\(n).generic.\(mode)", route: "generic-euclid2", k32: dim + 9 + s,
                                 k64: dim + 2 + s, absolute: false),
                        Expected(name: "batch.euclidean.\(dim).N\(n).generic.\(mode)", route: "vdsp-euclid", k32: dim + 3 + s,
                                 k64: dim + 3 + s, absolute: false),
                        Expected(name: "batch.euclidean2.\(dim).N\(n).optimized.\(mode)", route: "optimized-euclid2", k32: dim / 16 + 7 + s,
                                 k64: dim + 2 + s, absolute: false),
                        Expected(name: "batch.euclidean.\(dim).N\(n).optimized.\(mode)", route: "optimized-euclid", k32: dim / 16 + 8 + s,
                                 k64: dim + 3 + s, absolute: false)
                    ]
                }
                for variant in ["aos-fp32-reused", "aos-fp32-setup", "consumer-only"] {
                    expected.append(Expected(name: "batch.euclidean2.\(dim).N\(n).\(variant)", route: "optimized-euclid2",
                                             k32: dim / 16 + 7, k64: dim + 2, absolute: false))
                }
                for variant in ["optimized-soa", "soa-fp32-reused", "soa-fp32-setup"] {
                    expected.append(Expected(name: "batch.euclidean2.\(dim).N\(n).\(variant)", route: "soa-euclid2", k32: dim / 4 + 5,
                                             k64: dim + 2, absolute: false))
                }
                for variant in ["optimized", "consumer-only"] {
                    expected.append(Expected(name: "matrix.euclidean2.\(dim).N\(n).\(variant)", route: "matrix-norm-identity", k32: dim + 2,
                                             k64: dim + 2, absolute: true))
                }
                for item in expected {
                    let model = try XCTUnwrap(CorrectnessPolicy.errorModel(forCaseName: item.name), item.name)
                    XCTAssertEqual(model.route, item.route, item.name)
                    XCTAssertEqual(model.k32, item.k32, item.name)
                    XCTAssertEqual(model.k64, item.k64, item.name)
                    let rule = CorrectnessPolicy.rule(forCaseName: item.name)
                    XCTAssertEqual(rule.basis, .derived(model), item.name)
                    if item.absolute {
                        let scale = 4 * Double(dim)  // 4·D·B^2 with the fixture coordinate bound B = 1
                        XCTAssertEqual(model.kind, .absolute(scale: scale), item.name)
                        XCTAssertNil(rule.maxRel, item.name)
                        XCTAssertEqual(rule.maxAbs, Self.absoluteThreshold(k32: item.k32, k64: item.k64, scale: scale), item.name)
                    } else {
                        XCTAssertEqual(model.kind, .relative, item.name)
                        XCTAssertNil(rule.maxAbs, item.name)
                        XCTAssertEqual(rule.maxRel, Self.relativeThreshold(k32: item.k32, k64: item.k64), item.name)
                    }
                    checked += 1
                }
            }
        }
        XCTAssertEqual(checked, 3 * 7 * (16 + 3 + 3 + 2))
        // Widths without a read kernel, and routes whose summation is not derived, keep the empirical rule.
        for name in ["batch.euclidean2.384.N100.generic.sequential", "batch.cosine.512.N100.optimized-fused.sequential",
                     "batch.euclidean2.512.N100.optimized-fp16", "batch.euclidean2.512.N257.aos-fp16-reused", "dist.euclidean.512.generic",
                     "batch.euclidean.512.fixture", "batch.euclidean2.512.N100.generic"] {
            XCTAssertNil(CorrectnessPolicy.errorModel(forCaseName: name), name)
            guard case .empirical = CorrectnessPolicy.rule(forCaseName: name).basis else { XCTFail(name); continue }
        }
        XCTAssertEqual(CorrectnessPolicy.rule(forCaseName: "dist.cosine.512.generic").maxRel, 1e-6)
        XCTAssertEqual(CorrectnessPolicy.rule(forCaseName: "batch.euclidean2.512.N100.optimized-fp16").maxRel, 2e-3)
        XCTAssertEqual(CorrectnessPolicy.rule(forCaseName: "batch.cosine.512.N100.optimized-fp16").maxRel, 3e-3)
        XCTAssertEqual(RoundingBound.gamma(1, unitRoundoff: Self.u32), Self.u32 / (1 - Self.u32))
        XCTAssertEqual(RoundingBound.gamma(0, unitRoundoff: Self.u64), 0)
    }

    /// The routes' actual outputs on the CI recipe's batch fixture (`BatchFixtures`, run seed 0), with Double references.
    private struct RouteOutputs {
        var references2: [Double] = []
        var references1: [Double] = []
        var generic2: [Float] = []
        var vdsp1: [Float] = []
        var optimized2: [Float] = []
        var optimized1: [Float] = []
        var soa2: [Float] = []
        /// The generic squared distance of candidate 0 with its largest squared difference removed (one dropped term).
        var droppedTermGeneric2: Float = 0
    }

    private func routeOutputs(dim: Int, n: Int) throws -> RouteOutputs {
        let raw = try BatchFixtures.makeFixture(dim: dim, count: n, runSeed: 0) { $0 }
        let query = raw.query
        var o = RouteOutputs()
        o.references2 = raw.candidates.map { c in
            query.withUnsafeBufferPointer { q in c.withUnsafeBufferPointer { DoubleRef.euclid2(q, $0) } }
        }
        o.references1 = raw.candidates.map { c in
            query.withUnsafeBufferPointer { q in c.withUnsafeBufferPointer { DoubleRef.euclid(q, $0) } }
        }
        var dropped = raw.candidates[0]
        let largest = try XCTUnwrap((0..<dim).max { abs(query[$0] - dropped[$0]) < abs(query[$1] - dropped[$1]) })
        dropped[largest] = query[largest]
        func generic<Width: StaticDimension>(_: Width.Type) throws {
            let q = try Vector<Width>(query)
            let c = try raw.candidates.map { try Vector<Width>($0) }
            o.generic2 = c.map { q.euclideanDistanceSquared(to: $0) }
            o.vdsp1 = c.map { EuclideanDistance().distance(q, $0) }
            o.droppedTermGeneric2 = q.euclideanDistanceSquared(to: try Vector<Width>(dropped))
        }
        func filled(_ body: (UnsafeMutableBufferPointer<Float>) -> Void) -> [Float] {
            var out = [Float](repeating: 0, count: n)
            out.withUnsafeMutableBufferPointer { body($0) }
            return out
        }
        switch dim {
        case 512:
            try generic(Dim512.self)
            let q = try Vector512Optimized(query), c = try raw.candidates.map { try Vector512Optimized($0) }
            let soa = SoA<Vector512Optimized>.build(from: c)
            o.optimized2 = filled { BatchKernels.range_euclid2_512(query: q, candidates: c, range: 0..<n, out: $0) }
            o.optimized1 = filled { BatchKernels.range_euclid_512(query: q, candidates: c, range: 0..<n, out: $0) }
            o.soa2 = filled { BatchKernels_SoA.euclid2_512(query: q, soa: soa, out: $0) }
        case 768:
            try generic(Dim768.self)
            let q = try Vector768Optimized(query), c = try raw.candidates.map { try Vector768Optimized($0) }
            let soa = SoA<Vector768Optimized>.build(from: c)
            o.optimized2 = filled { BatchKernels.range_euclid2_768(query: q, candidates: c, range: 0..<n, out: $0) }
            o.optimized1 = filled { BatchKernels.range_euclid_768(query: q, candidates: c, range: 0..<n, out: $0) }
            o.soa2 = filled { BatchKernels_SoA.euclid2_768(query: q, soa: soa, out: $0) }
        case 1536:
            try generic(Dim1536.self)
            let q = try Vector1536Optimized(query), c = try raw.candidates.map { try Vector1536Optimized($0) }
            let soa = SoA<Vector1536Optimized>.build(from: c)
            o.optimized2 = filled { BatchKernels.range_euclid2_1536(query: q, candidates: c, range: 0..<n, out: $0) }
            o.optimized1 = filled { BatchKernels.range_euclid_1536(query: q, candidates: c, range: 0..<n, out: $0) }
            o.soa2 = filled { BatchKernels_SoA.euclid2_1536(query: q, soa: soa, out: $0) }
        default:
            XCTFail("no kernels at width \(dim)")
        }
        return o
    }

    /// The generic squared-Euclidean bound holds on the CI fixtures (per distance and for the summed observation), and two
    /// extreme defects still fail the case: candidate 0 with its LARGEST squared difference dropped, and the observation
    /// with its SMALLEST candidate dropped. A dropped small term is not claimed to be caught (it can pass; see
    /// `CorrectnessPolicy`).
    func testGenericSquaredEuclideanBoundHoldsAndStillCatchesDroppedWork() throws {
        for dim in Self.ciDims {
            for n in Self.ciCounts {
                let o = try routeOutputs(dim: dim, n: n)
                let name = "batch.euclidean2.\(dim).N\(n).generic.sequential"
                let caseThreshold = try XCTUnwrap(CorrectnessPolicy.rule(forCaseName: name).maxRel)
                // The per-distance bound is the same route without an observation (N = 1 adds no summation factor).
                let perDistanceName = "batch.euclidean2.\(dim).N1.generic.sequential"
                let perDistance = try XCTUnwrap(CorrectnessPolicy.errorModel(forCaseName: perDistanceName)).threshold
                XCTAssertEqual(perDistance, Self.relativeThreshold(k32: dim + 9, k64: dim + 2))
                let worst = try XCTUnwrap(zip(o.generic2, o.references2).map { Self.relativeError($0, $1) }.max())
                XCTAssertLessThanOrEqual(worst, perDistance, "\(name) per-distance")
                // The observation: the sequential provider's FP32 sum of the N results against the summed references.
                var sum: Float = 0
                for value in o.generic2 { sum += value }
                let reference = o.references2.reduce(0, +)
                let observation = abs(Double(sum) - reference) / reference
                XCTAssertLessThanOrEqual(observation, caseThreshold, "\(name) observation")
                // The whole case through the gate's own statistics: positive control.
                let observed = ProviderObservations(reference: reference)
                observed.observe(sum)
                let distances = Correctness.vector(currents: o.generic2, references: o.references2)
                XCTAssertNil(gate(result(name, observed.combining(distances))), name)
                print(String(format: "T11 generic-euclid2 D=%d N=%d: per-distance max rel %.3e (bound %.3e); observation rel %.3e; case "
                    + "threshold %.3e; pre-T11 fixed 1e-6 exceeded: %@", dim, n, worst, perDistance, observation, caseThreshold,
                    worst > 1e-6 ? "yes" : "no"))

                // Negative control 1: one dropped term (the largest squared difference of candidate 0).
                XCTAssertGreaterThan(Self.relativeError(o.droppedTermGeneric2, o.references2[0]), caseThreshold, "\(name) dropped term")
                var currents = o.generic2
                currents[0] = o.droppedTermGeneric2
                XCTAssertNotNil(gate(result(name, observed.combining(Correctness.vector(currents: currents, references: o.references2)))))
                // Negative control 2: an observation that misses one candidate, the one with the smallest distance.
                let smallest = try XCTUnwrap(o.generic2.indices.min { o.generic2[$0] < o.generic2[$1] })
                var partial: Float = 0
                for (index, value) in o.generic2.enumerated() where index != smallest { partial += value }
                XCTAssertGreaterThan(abs(Double(partial) - reference) / reference, caseThreshold, "\(name) dropped candidate")
                let missing = ProviderObservations(reference: reference)
                missing.observe(partial)
                XCTAssertNotNil(gate(result(name, missing.combining(distances))), "\(name) dropped candidate")
            }
        }
    }

    /// The derived bounds hold on the library kernels' actual outputs. The per-distance part runs the kernels directly, so
    /// it checks the bound itself; the benchmark records a per-distance comparison for every route below EXCEPT
    /// `batch.euclidean.512.N.optimized.*`, whose 512-wide case body records only the provider observation (its
    /// `range_euclid_512` per-distance outputs are checked here but are not gated by the benchmark). The observation part
    /// covers the optimized euclid2, optimized euclid and vDSP euclid provider routes.
    func testOptimizedAndSoABoundsHoldOnFixtures() throws {
        for dim in Self.ciDims {
            // 3 takes the SoA 2-way kernel, odd counts the AoS odd tail; at 1536, N = 1500 takes the SoA tiled kernel
            // (1500 candidates * 384 lanes * 16 bytes > 8 MiB).
            for n in [3, 100, 257, 1_000] + (dim == 1536 ? [1_500] : []) {
                let o = try routeOutputs(dim: dim, n: n)
                func bound(_ name: String) throws -> Double {
                    try XCTUnwrap(CorrectnessPolicy.errorModel(forCaseName: name), name).threshold
                }
                let routes: [(String, [Float], [Double], Double)] = [
                    ("optimized-euclid2", o.optimized2, o.references2, try bound("batch.euclidean2.\(dim).N1.optimized.sequential")),
                    ("optimized-euclid", o.optimized1, o.references1, try bound("batch.euclidean.\(dim).N1.optimized.sequential")),
                    ("vdsp-euclid", o.vdsp1, o.references1, try bound("batch.euclidean.\(dim).N1.generic.sequential")),
                    ("soa-euclid2", o.soa2, o.references2, try bound("batch.euclidean2.\(dim).N\(n).optimized-soa")),
                    ("aos-fp32 and consumer-only", o.optimized2, o.references2, try bound("batch.euclidean2.\(dim).N\(n).consumer-only"))
                ]
                for (route, currents, references, limit) in routes {
                    let worst = try XCTUnwrap(zip(currents, references).map { Self.relativeError($0, $1) }.max())
                    XCTAssertLessThanOrEqual(worst, limit, "\(route) D=\(dim) N=\(n)")
                    print(String(format: "T11 %@ D=%d N=%d: max rel %.3e (bound %.3e)", route, dim, n, worst, limit))
                }
                let observed: [(String, String, [Float], [Double])] = [("euclidean2", "optimized", o.optimized2, o.references2),
                    ("euclidean", "optimized", o.optimized1, o.references1), ("euclidean", "generic", o.vdsp1, o.references1)]
                for (metric, variant, currents, references) in observed {
                    var sum: Float = 0
                    for value in currents { sum += value }
                    let reference = references.reduce(0, +)
                    let providerCase = "batch.\(metric).\(dim).N\(n).\(variant).parallel"
                    let limit = try XCTUnwrap(CorrectnessPolicy.rule(forCaseName: providerCase).maxRel)
                    XCTAssertLessThanOrEqual(abs(Double(sum) - reference) / reference, limit, "\(providerCase) observation")
                }
            }
        }
    }

    // MARK: Zero-reference class

    func testMatrixZeroReferenceIsGatedOnAbsoluteError() throws {
        var sawNonzeroDiagonal = false
        for dim in Self.ciDims {
            for n in [1, 3, 8, 17] {
                let inputs = (0..<n).map { MatrixDistanceBench.inputs(dim: dim, row: $0, runSeed: 0) }
                let output: [Float]
                switch dim {
                case 512:
                    let v = try inputs.map { try Vector512Optimized($0) }
                    output = MatrixDistance.euclideanSquaredMatrix(queries: v, candidates: v)
                case 768:
                    let v = try inputs.map { try Vector768Optimized($0) }
                    output = MatrixDistance.euclideanSquaredMatrix(queries: v, candidates: v)
                default:
                    let v = try inputs.map { try Vector1536Optimized($0) }
                    output = MatrixDistance.euclideanSquaredMatrix(queries: v, candidates: v)
                }
                // Compared as the matrix suite compares: MatrixDistanceBench.sampledPairs, Double references.
                var currents: [Float] = [], references: [Double] = [], diagonal: [Int] = []
                for (i, j) in MatrixDistanceBench.sampledPairs(count: n) {
                    if i == j { diagonal.append(currents.count) }
                    currents.append(output[i * n + j])
                    references.append(inputs[i].withUnsafeBufferPointer { a in
                        inputs[j].withUnsafeBufferPointer { DoubleRef.euclid2(a, $0) }
                    })
                }
                let name = "matrix.euclidean2.\(dim).N\(n).optimized"
                let stats = Correctness.vector(currents: currents, references: references)
                let rule = CorrectnessPolicy.rule(forCaseName: name)
                XCTAssertNil(rule.maxRel, name)
                let limit = try XCTUnwrap(rule.maxAbs)
                XCTAssertEqual(limit, Self.absoluteThreshold(k32: dim + 2, k64: dim + 2, scale: 4 * Double(dim)))
                XCTAssertLessThanOrEqual(stats.maxAbsError, limit, name)
                // The reported relative error never divides by the zero diagonal reference: |ref| is floored.
                let floored = try XCTUnwrap(zip(currents, references).map {
                    abs(Double($0) - $1) / max(abs($1), Correctness.referenceFloor)
                }.max())
                XCTAssertEqual(stats.maxRelError, floored, name)
                for index in diagonal {
                    XCTAssertEqual(references[index], 0, name)
                    if currents[index] != 0 {
                        sawNonzeroDiagonal = true
                        // The pre-T11 metric divided this error by 1e-12 and failed the case on correct output.
                        XCTAssertGreaterThan(Double(currents[index]) / 1e-12, CorrectnessPolicy.fp32EmpiricalRelative, name)
                    }
                }
                // Positive control: both matrix routes pass on the actual output.
                for caseName in [name, "matrix.euclidean2.\(dim).N\(n).consumer-only"] {
                    XCTAssertNil(gate(result(caseName, stats)), caseName)
                }
                print(String(format: "T11 matrix D=%d N=%d: max abs %.3e (bound %.3e); max diagonal output %.3e", dim, n,
                             stats.maxAbsError, limit, diagonal.map { Double(currents[$0]) }.max() ?? 0))
                // Negative controls: a diagonal (zero-reference) or an off-diagonal entry off by twice the bound fails.
                for index in Set([diagonal[0], currents.count - 2 < 0 ? 0 : currents.count - 2]) {
                    var wrong = currents
                    wrong[index] += Float(2 * limit)
                    let perturbed = Correctness.vector(currents: wrong, references: references)
                    XCTAssertNotNil(gate(result(name, perturbed)), "\(name) entry \(index)")
                }
                if n > 1 {
                    // A transposed or mis-indexed read (the diagonal value where an off-diagonal belongs) fails too.
                    let off = try XCTUnwrap(currents.indices.first { !diagonal.contains($0) })
                    var wrong = currents
                    wrong[off] = currents[diagonal[0]]
                    XCTAssertNotNil(gate(result(name, Correctness.vector(currents: wrong, references: references))), "\(name) mis-indexed")
                }
            }
        }
        XCTAssertTrue(sawNonzeroDiagonal, "the sweep must exercise a nonzero output against a zero reference")
    }

    /// Every way `metric(absError, ref)` breaks the floor contract on a sweep that straddles the floor: below it the
    /// denominator must be the floor, at or above it `|ref|`, and the result never exceeds `absError / floor`.
    static func floorContractViolations(_ metric: (Double, Double) -> Double, floor: Double) -> [String] {
        let magnitudes: [Double] = [0, .leastNonzeroMagnitude, 1e-300, 1e-30, 1e-12, 1e-6, 0.5, floor.nextDown, floor, floor.nextUp, 2, 1e6]
        var violations: [String] = []
        for magnitude in magnitudes {
            for reference in [magnitude, -magnitude] {
                for absError in [0, 1e-9, 1e-3, 1, 1e3] {
                    let value = metric(absError, reference)
                    let expected = absError / (abs(reference) < floor ? floor : abs(reference))
                    if value != expected || value > absError / floor {
                        violations.append("ref=\(reference) absError=\(absError): \(value), expected \(expected)")
                    }
                }
            }
        }
        return violations
    }

    func testRelativeErrorNeverDividesByReferenceBelowFloor() {
        let floor = Correctness.referenceFloor
        XCTAssertEqual(floor, 1)
        XCTAssertEqual(Self.floorContractViolations({ Correctness.relErr(absError: $0, ref: $1) }, floor: floor), [])
        // Negative control: the pre-T11 metric divided by any reference down to 1e-12.
        XCTAssertFalse(Self.floorContractViolations({ $0 / max(abs($1), 1e-12) }, floor: floor).isEmpty)
        // Every entry point that reports a relative error goes through the floor.
        XCTAssertEqual(Correctness.scalar(current: 0.5, reference: 0).maxRelError, 0.5 / floor)
        XCTAssertEqual(Correctness.scalar(current: 0.5, reference: -0.0).maxRelError, 0.5 / floor)
        XCTAssertEqual(Correctness.vector(currents: [1, 0.25, 3], references: [0, 1e-30, 2]).maxRelError, 1 / floor)
        let observed = ProviderObservations(reference: 0)
        observed.observe(0.75)
        XCTAssertEqual(observed.statistics.maxRelError, 0.75 / floor)
        // Above the floor the metric is the ordinary relative error.
        XCTAssertEqual(Correctness.scalar(current: 3, reference: 2).maxRelError, 0.5)
    }

    /// A call of `relErr`, whitespace and line breaks allowed before the parenthesis.
    static let relErrCall = try! NSRegularExpression(pattern: "\\brelErr\\s*\\(")
    /// A construction of comparison statistics from error values: `CorrectnessStats(samples:`, `CorrectnessStats.init(samples:`
    /// or an implicit `.init(samples:`, with whitespace and line breaks allowed between every token.
    static let statisticsConstruction = try! NSRegularExpression(
        pattern: "(?:\\bCorrectnessStats\\s*(?:\\.\\s*init)?|\\.\\s*init)\\s*\\(\\s*samples\\s*:")

    /// Benchmark sources, outside the two allowed files, that compute a relative error or build comparison statistics
    /// from their own error values. Raw text, comments included (stricter than a lexer); one entry per match.
    static func relativeErrorSiteViolations(_ files: [(path: String, text: String)]) -> [String] {
        func matches(_ regex: NSRegularExpression, _ text: String) -> Int {
            regex.numberOfMatches(in: text, range: NSRange(text.startIndex..., in: text))
        }
        var violations: [String] = []
        for file in files where file.path != "Benchmarks/VectorCoreBench/Utils/Correctness.swift" {
            for _ in 0..<matches(relErrCall, file.text) { violations.append("\(file.path): relErr( outside Correctness.swift") }
            guard file.path != "Benchmarks/VectorCoreBench/Core/ProviderObservations.swift" else { continue }
            for _ in 0..<matches(statisticsConstruction, file.text) {
                violations.append("\(file.path): builds CorrectnessStats from its own errors")
            }
        }
        return violations
    }

    func testOnlyCorrectnessComputesRelativeErrors() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        var files: [(path: String, text: String)] = []
        for directory in ["Benchmarks/VectorCoreBench", "Sources/VectorCoreBenchmarking"] {
            let base = root.appendingPathComponent(directory)
            let enumerator = try XCTUnwrap(FileManager.default.enumerator(atPath: base.path))
            for case let relative as String in enumerator where relative.hasSuffix(".swift") {
                files.append((directory + "/" + relative, try String(contentsOf: base.appendingPathComponent(relative), encoding: .utf8)))
            }
        }
        XCTAssertGreaterThan(files.count, 30, "the guard must see the benchmark sources")
        XCTAssertEqual(Self.relativeErrorSiteViolations(files), [])
        // ProviderObservations aggregates `Correctness.scalar` comparisons; it computes no relative error itself.
        let observations = try XCTUnwrap(files.first { $0.path == "Benchmarks/VectorCoreBench/Core/ProviderObservations.swift" })
        XCTAssertTrue(observations.text.contains("Correctness.scalar(current: value, reference: reference)"))
        XCTAssertTrue(try XCTUnwrap(files.first { $0.path == "Benchmarks/VectorCoreBench/Utils/Correctness.swift" }).text
            .contains("absError / max(abs(ref), referenceFloor)"))
        // Negative controls: each spelling below is flagged once, including the ones the pre-fix-round literal check
        // (`contains("relErr(")`, `contains("CorrectnessStats(samples:")`) missed.
        let path = "Benchmarks/VectorCoreBench/Suites/NewBench.swift"
        let tail = "1, maxAbsError: e, meanAbsError: e, maxRelError: r, meanRelError: r)"
        let flagged = ["let s = Correctness.relErr(absError: e, ref: r)", "let s = Correctness.relErr (absError: e, ref: r)",
                       "let s = Correctness.relErr\n(absError: e, ref: r)", "let a = CorrectnessStats(samples: " + tail,
                       "let a = CorrectnessStats.init(samples: " + tail, "let a = CorrectnessStats(\n    samples: " + tail,
                       "let a = CorrectnessStats . init ( samples : " + tail, "let a: CorrectnessStats = .init(samples: " + tail]
        var literalMisses = 0
        for text in flagged {
            XCTAssertEqual(Self.relativeErrorSiteViolations([(path, text)]).count, 1, text)
            if !text.contains("relErr(") && !text.contains("CorrectnessStats(samples:") { literalMisses += 1 }
        }
        XCTAssertEqual(literalMisses, 6, "the fixtures must include the spellings the literal check missed")
        // Positive controls: other constructions and look-alike names are not flagged.
        for text in ["let a = CorrectnessStats(invalidReason: \"control\")", "let samples = 3", "var relErrors: [Double] = []",
                     "let p = PerOperationStatistics(sampleNsPerInvocation: [1], unitsPerInvocation: 1)"] {
            XCTAssertEqual(Self.relativeErrorSiteViolations([(path, text)]), [], text)
        }
    }

    // MARK: Validation requirement and the route ratchet

    private func fullCatalog() throws -> [BenchmarkCase] {
        let value = try ValidatedOptions(parseCLI(arguments: ["--suites", "dot,distance,normalize,batch,matrix,memory,noop,boundary",
            "--batch-ns", "3,100,257,1000,10000", "--prefer-soa", "--use-mixed-precision"], environment: [:]))
        var cases: [BenchmarkCase] = []
        for suite in [DotProductBench.self as any BenchmarkSuite.Type, DistanceBench.self, NormalizationBench.self, BatchBench.self,
                      MatrixDistanceBench.self, MemoryBench.self, NoopBench.self, BoundaryBench.self] {
            cases += suite.catalog(options: value, progress: BenchmarkProgressReporter(format: .none))
        }
        return cases.filter { $0.availability == .available }
    }

    /// A case's route: its ID without width, candidate count and provider.
    private static func routeKey(_ name: String) -> String {
        let p = CaseParsing.parse(name: name)
        switch p.kind {
        case "batch", "matrix": return [p.kind, p.metric, p.variant].compactMap { $0 }.joined(separator: ".")
        case "dist": return "dist.\(p.metric ?? "?").\(p.variant ?? "?")"
        case "dot": return "dot.\(p.variant ?? "?")"
        case "normalize": return "normalize.\(p.status ?? "?").\(p.variant ?? "?")"
        case "mem": return "mem.\(p.metric ?? "?")"
        default: return name
        }
    }

    func testValidationRequirementIsDeclaredPerCase() throws {
        let cases = try fullCatalog()
        var notApplicable: Set<String> = [], required: Set<String> = []
        for item in cases {
            switch CorrectnessPolicy.validation(forCaseName: item.name) {
            case .required: required.insert(Self.routeKey(item.name))
            case .notApplicable(let reason):
                XCTAssertFalse(reason.isEmpty, item.name)
                notApplicable.insert(Self.routeKey(item.name))
            }
        }
        XCTAssertEqual(notApplicable, ["noop.scalar-add", "mem.alloc", "mem.pool", "mem.fill"])
        XCTAssertTrue(notApplicable.isDisjoint(with: required))
        // The suites whose sources record no comparison are exactly dot, normalize, memory and noop; their routes with a
        // VectorCore output stay required, so they still fail "Missing validation" (a gap, not a declaration).
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Benchmarks/VectorCoreBench/Suites")
        for file in try FileManager.default.contentsOfDirectory(atPath: root.path) where file.hasSuffix(".swift") {
            let text = try String(contentsOf: root.appendingPathComponent(file), encoding: .utf8)
            let records = Self.recordsComparison(text)
            let withoutComparison = ["DotProductBench.swift", "NormalizationBench.swift", "MemoryBench.swift", "NoopBench.swift"]
            XCTAssertEqual(records, !withoutComparison.contains(file), file)
        }
        XCTAssertTrue(required.isSuperset(of: ["dot.generic", "dot.optimized", "normalize.success.generic", "normalize.success.optimized",
                                               "normalize.zeroFail.generic", "normalize.zeroFail.optimized", "mem.copy"]))
        // The gate: a declared case without a record passes; a required one fails; a present comparison is never ignored.
        for name in ["noop.scalar-add", "mem.alloc.aligned.512", "mem.pool.acquire.768", "mem.fill.1536"] {
            XCTAssertNil(gate(result(name, nil)), name)
            XCTAssertNotNil(gate(result(name, stats(maxRel: 1, maxAbs: 1))), name)
            XCTAssertNotNil(gate(result(name, CorrectnessStats(invalidReason: "control"))), name)
        }
        for name in ["dot.512.generic", "normalize.success.512.optimized", "mem.copy.512", "batch.euclidean2.512.N100.generic.sequential",
                     "matrix.euclidean2.512.N3.optimized"] {
            XCTAssertEqual(gate(result(name, nil))?.reason, "Missing validation", name)
        }
        let mixed = [result("noop.scalar-add", nil), result("mem.copy.768", nil)]
        XCTAssertEqual(evaluateCorrectnessWithPolicy(results: mixed, overrideRel: nil, overrideAbs: nil).map(\.name), ["mem.copy.768"])
    }

    /// Whether a suite source records a comparison: a `Correctness.` call, `withCorrectness(` or a `correctness:` argument,
    /// whitespace allowed.
    static func recordsComparison(_ text: String) -> Bool {
        let regex = try! NSRegularExpression(pattern: "\\bCorrectness\\s*\\.|\\bwithCorrectness\\s*\\(|\\bcorrectness\\s*:")
        return regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil
    }

    /// Fail-closed check for the routes listed as unvalidated: the suite file that catalogs such a route must record no
    /// comparison. Adding validation to one of those suites therefore fails this until its routes are moved to a derived
    /// model or to `empiricalRoutes`. The check is per suite file (lexical): a suite that validates only some of its cases
    /// must split its entries here.
    static func unvalidatedRouteProblems(routesBySuiteFile: [String: Set<String>], sources: [String: String]) -> [String] {
        var problems: [String] = []
        for (file, routes) in routesBySuiteFile.sorted(by: { $0.key < $1.key }) {
            let listed = routes.intersection(unvalidatedRoutes)
            guard !listed.isEmpty else { continue }
            if recordsComparison(sources[file] ?? "") {
                problems.append("\(file) records a comparison but catalogs routes listed as unvalidated: \(listed.sorted())")
            }
        }
        return problems
    }

    func testEveryValidatedCatalogRouteIsDerivedOrListedEmpirical() throws {
        let value = try ValidatedOptions(parseCLI(arguments: ["--suites", "dot,distance,normalize,batch,matrix,memory,noop,boundary",
            "--batch-ns", "3,100,257,1000,10000", "--prefer-soa", "--use-mixed-precision"], environment: [:]))
        let suitesRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Benchmarks/VectorCoreBench/Suites")
        var derived: Set<String> = [], empirical: Set<String> = [], unvalidated: Set<String> = []
        var routesBySuiteFile: [String: Set<String>] = [:], sources: [String: String] = [:]
        for suite in [DotProductBench.self as any BenchmarkSuite.Type, DistanceBench.self, NormalizationBench.self, BatchBench.self,
                      MatrixDistanceBench.self, MemoryBench.self, NoopBench.self, BoundaryBench.self] {
            let file = String(describing: suite) + ".swift"
            sources[file] = try String(contentsOf: suitesRoot.appendingPathComponent(file), encoding: .utf8)
            let catalog = suite.catalog(options: value, progress: BenchmarkProgressReporter(format: .none))
            for item in catalog where item.availability == .available {
                let route = Self.routeKey(item.name)
                routesBySuiteFile[file, default: []].insert(route)
                if Self.unvalidatedRoutes.contains(route) { unvalidated.insert(route); continue }
                switch CorrectnessPolicy.rule(forCaseName: item.name).basis {
                case .derived: derived.insert(route)
                case .empirical: empirical.insert(route)
                }
            }
        }
        print("T11 derived routes: \(derived.sorted())")
        print("T11 empirical routes: \(empirical.sorted())")
        // Every route of every kind is classified; the unvalidated ones are pinned and their suites record no comparison.
        XCTAssertEqual(unvalidated, Self.unvalidatedRoutes)
        XCTAssertEqual(Self.unvalidatedRouteProblems(routesBySuiteFile: routesBySuiteFile, sources: sources), [])
        // Negative control: the dot suite gaining a comparison fails the check until its routes are reclassified.
        var validatedDot = sources
        validatedDot["DotProductBench.swift", default: ""] += "\nlet c = Correctness.scalar(current: a.dotProduct(b), reference: r)\n"
        XCTAssertEqual(Self.unvalidatedRouteProblems(routesBySuiteFile: routesBySuiteFile, sources: validatedDot).count, 1)
        var spaced = sources
        spaced["NoopBench.swift", default: ""] += "\n    correctness :stats\n"
        XCTAssertEqual(Self.unvalidatedRouteProblems(routesBySuiteFile: routesBySuiteFile, sources: spaced).count, 1)
        XCTAssertEqual(derived, ["batch.euclidean2.generic", "batch.euclidean.generic", "batch.euclidean2.optimized",
                                 "batch.euclidean.optimized", "batch.euclidean2.aos-fp32-reused", "batch.euclidean2.aos-fp32-setup",
                                 "batch.euclidean2.consumer-only", "batch.euclidean2.optimized-soa", "batch.euclidean2.soa-fp32-reused",
                                 "batch.euclidean2.soa-fp32-setup", "matrix.euclidean2.optimized", "matrix.euclidean2.consumer-only"])
        // A ratchet: a new validated route fails here until it gets a derived model or is listed with the empirical thresholds.
        XCTAssertEqual(empirical, Self.empiricalRoutes)
        XCTAssertTrue(derived.isDisjoint(with: empirical))
    }

    /// The routes whose suites record no comparison today (each still declared per case by
    /// `CorrectnessPolicy.validation`: required ones fail "Missing validation").
    private static let unvalidatedRoutes: Set<String> = [
        "dot.generic", "dot.optimized", "normalize.success.generic", "normalize.success.optimized", "normalize.zeroFail.generic",
        "normalize.zeroFail.optimized", "mem.alloc", "mem.copy", "mem.fill", "mem.pool", "noop.scalar-add"
    ]

    /// The validated routes that keep the pre-T11 fixed thresholds (no derived model yet; see `CorrectnessPolicy`).
    private static let empiricalRoutes: Set<String> = [
        "batch.cosine.optimized-fused", "batch.cosine.optimized-preNorm", "batch.cosine.optimized-fp16", "batch.euclidean2.optimized-fp16",
        "batch.euclidean2.aos-fp16-reused", "batch.euclidean2.aos-fp16-setup",
        "dist.euclidean.generic", "dist.euclidean.optimized", "dist.cosine.generic", "dist.cosine.optimized", "dist.manhattan.generic",
        "dist.manhattan.optimized", "dist.dot.generic", "dist.dot.optimized", "dist.chebyshev.generic", "dist.chebyshev.optimized",
        "dist.hamming.generic", "dist.hamming.optimized", "dist.minkowski.generic", "dist.minkowski.optimized"
    ]

    // MARK: Explicit threshold overrides

    func testOverrideReplacesOnlyItsOwnKindAndTheReasonSaysSo() throws {
        // A matrix route has only a derived absolute threshold. `--max-rel-error` alone adds a relative threshold and leaves
        // the derived absolute one in force; the reason names the source of each.
        let matrix = "matrix.euclidean2.512.N3.optimized"
        let rule = CorrectnessPolicy.rule(forCaseName: matrix)
        let absLimit = try XCTUnwrap(rule.maxAbs)
        let derivedAbs = String(format: "%.3e", absLimit) + " (derived matrix-norm-identity k32=514 k64=514)"
        let relFailure = try XCTUnwrap(gate(result(matrix, stats(maxRel: 2e-4, maxAbs: 2e-4)), rel: 1e-6))
        XCTAssertTrue(relFailure.reason.contains("thr(rel=1.000e-06 (override) abs=\(derivedAbs))"), relFailure.reason)
        let absFailure = try XCTUnwrap(gate(result(matrix, stats(maxRel: 0, maxAbs: absLimit * 2)), rel: 1e-6))
        XCTAssertTrue(absFailure.reason.contains("thr(rel=1.000e-06 (override) abs=\(derivedAbs))"), absFailure.reason)
        XCTAssertNil(gate(result(matrix, stats(maxRel: 0, maxAbs: absLimit)), rel: 1e-6), "the derived absolute bound is inclusive")
        // Without overrides the relative kind is absent for the matrix route.
        let plain = try XCTUnwrap(gate(result(matrix, stats(maxRel: 0, maxAbs: absLimit * 2))))
        XCTAssertTrue(plain.reason.contains("thr(rel=nil abs=\(derivedAbs))"), plain.reason)
        // `--max-abs-error` alone on a relative route keeps the derived relative threshold.
        let generic = "batch.euclidean2.512.N100.generic.sequential"
        let relLimit = try XCTUnwrap(CorrectnessPolicy.rule(forCaseName: generic).maxRel)
        let derivedRel = String(format: "%.3e", relLimit) + " (derived generic-euclid2 k32=620 k64=613)"
        let absOverride = try XCTUnwrap(gate(result(generic, stats(maxRel: relLimit * 2, maxAbs: 0)), abs: 1e-3))
        XCTAssertTrue(absOverride.reason.contains("thr(rel=\(derivedRel) abs=1.000e-03 (override))"), absOverride.reason)
        // An empirical route names its basis too.
        let cosine = try XCTUnwrap(gate(result("batch.cosine.512.N100.optimized-fused.sequential", stats(maxRel: 1, maxAbs: 1))))
        XCTAssertTrue(cosine.reason.contains("thr(rel=1.000e-06 (empirical) abs=nil)"), cosine.reason)
    }

    // MARK: Fixture bound and the CI recipe's recorded failures

    func testFixtureCoordinatesLieWithinTheStatedBound() throws {
        let bound = Float(CorrectnessPolicy.fixtureCoordinateBound)
        XCTAssertEqual(bound, 1)
        for runSeed: UInt64 in [0, 1, 0x9E37_79B9_7F4A_7C15, .max] {
            for dim in Self.ciDims {
                let batch = try BatchFixtures.makeFixture(dim: dim, count: 8, runSeed: runSeed) { $0 }
                var values = batch.query + batch.candidates.flatMap { $0 }
                values += (0..<4).flatMap { MatrixDistanceBench.inputs(dim: dim, row: $0, runSeed: runSeed) }
                XCTAssertTrue(values.allSatisfy { $0 >= -bound && $0 < bound }, "seed \(runSeed) D=\(dim)")
            }
        }
        // The construction's extremes, t = 0 and t = (2^24 - 1) / 2^24, through `lowerBound + (upperBound - lowerBound) * t`.
        let range: ClosedRange<Float> = -1.0...1.0
        for t in [Float(0), Float(Double((1 << 24) - 1) / Double(1 << 24))] {
            let value = range.lowerBound + (range.upperBound - range.lowerBound) * t
            XCTAssertTrue(value >= -bound && value < bound, "\(value)")
        }
    }

    func testCIRecipeObservationsPassAndPerturbationsFail() throws {
        // maxRel and maxAbs of the 18 generic batch.euclidean2 failures that the guide's CI command recorded on 2240d2e before
        // T11 (an observation outside this repository; every provider mode recorded the same maxRel).
        let recorded: [(dim: Int, n: Int, rel: Double, abs: Double)] = [
            (512, 100, 1.041e-06, 1.794e-03), (512, 1_000, 1.125e-06, 1.542e-01), (768, 100, 1.276e-06, 5.790e-03),
            (768, 1_000, 1.592e-06, 1.986e-01), (1536, 100, 2.018e-06, 8.165e-03), (1536, 1_000, 2.306e-06, 8.676e-01)
        ]
        var checked = 0
        for item in recorded {
            for mode in ["sequential", "parallel", "automatic"] {
                let name = "batch.euclidean2.\(item.dim).N\(item.n).generic.\(mode)"
                XCTAssertGreaterThan(item.rel, CorrectnessPolicy.fp32EmpiricalRelative, "why \(name) failed before T11")
                XCTAssertNil(gate(result(name, stats(maxRel: item.rel, maxAbs: item.abs))), name)
                let threshold = try XCTUnwrap(CorrectnessPolicy.rule(forCaseName: name).maxRel)
                XCTAssertNil(gate(result(name, stats(maxRel: threshold, maxAbs: 0))), "\(name) at the bound")
                XCTAssertNotNil(gate(result(name, stats(maxRel: threshold.nextUp, maxAbs: 0))), "\(name) above the bound")
                // Explicit thresholds still replace the derived one.
                XCTAssertNotNil(gate(result(name, stats(maxRel: item.rel, maxAbs: item.abs)), rel: 1e-6), "\(name) override")
                XCTAssertNotNil(gate(result(name, stats(maxRel: 0, maxAbs: item.abs)), abs: item.abs / 2), "\(name) abs override")
                checked += 1
            }
        }
        XCTAssertEqual(checked, 18)
    }
}
