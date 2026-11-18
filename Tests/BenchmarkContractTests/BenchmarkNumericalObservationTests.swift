import Foundation
import XCTest
import VectorCore
@testable import VectorCoreBenchSupport
import VectorCoreBenchmarking

final class BenchmarkNumericalObservationTests: XCTestCase {
    // Removing an observation from any repaired catalog case must fail this table.
    @MainActor func testEveryRepairedCatalogCaseCarriesStrictObservation() async throws {
        let options = try ValidatedOptions(parseCLI(arguments: ["--suites", "dot,normalize,memory", "--dims", "512,768,1536",
            "--repeats", "1", "--samples", "2", "--no-ab"], environment: [:]))
        let progress = BenchmarkProgressReporter(format: .none)
        let catalog = DotProductBench.catalog(options: options, progress: progress)
            + NormalizationBench.catalog(options: options, progress: progress)
            + MemoryBench.catalog(options: options, progress: progress).filter { $0.name.hasPrefix("mem.copy.") }
        XCTAssertEqual(catalog.count, 21)
        for item in catalog {
            let results = try await item.operation()
            let result = try XCTUnwrap(results.first)
            let observation = try XCTUnwrap(result.correctness, item.name)
            XCTAssertEqual(observation.status, .valid, item.name)
            XCTAssertEqual(observation.samples, item.name.hasPrefix("normalize.success") ? Int(item.name.split(separator: ".")[2])! :
                (item.name.hasPrefix("mem.copy") ? Int(item.name.split(separator: ".")[2])! * 1024 : 1), item.name)
            XCTAssertNil(CorrectnessPolicy.failure(for: result, overrideRel: nil, overrideAbs: nil), item.name)
            XCTAssertEqual(item.boundary.inclusions.correctnessValidation, .excluded, item.name)
        }
    }

    private func failure(_ name: String, _ observation: CorrectnessStats, override: Double? = nil) -> BenchmarkFailure? {
        CorrectnessPolicy.failure(for: BenchResult(name: name, iterations: 1, totalNanoseconds: 1, correctness: observation),
            overrideRel: override, overrideAbs: override)
    }

    // A Float reference would lose the middle term and accept the deliberately wrong zero.
    func testDotDoubleReferenceRejectsFloatCancellationAndDroppedTerms() {
        let a: [Float] = [16_777_216, 1, -16_777_216]
        let b: [Float] = [1, 1, 1]
        XCTAssertNil(failure("dot.512.generic", Correctness.dot(current: 1, a: a, b: b)))
        XCTAssertNotNil(failure("dot.512.generic", Correctness.dot(current: 0, a: a, b: b)))
        XCTAssertNotNil(failure("dot.512.optimized", Correctness.dot(current: 28, a: [2, 3, 4], b: [5, 6, 7])))
        XCTAssertNotNil(failure("dot.512.generic", Correctness.dot(current: .nan, a: a, b: b)))
        XCTAssertNotNil(failure("dot.512.generic", Correctness.dot(current: 0, a: [1], b: [])))
    }

    // Reusing the produced vector as the reference would accept scaled, permuted or negated results.
    func testNormalizationReferenceRejectsWrongCoordinatesAndOutcomes() {
        let name = "normalize.success.512.generic"
        XCTAssertNil(failure(name, Correctness.normalization(current: .success([0.6, 0.8]), input: [3, 4])))
        let wrongOutputs: [[Float]] = [[6, 8], [0.8, 0.6], [-0.6, -0.8], [0.6], [0.6, .nan]]
        for output in wrongOutputs {
            XCTAssertNotNil(failure(name, Correctness.normalization(current: .success(output), input: [3, 4])))
        }
        XCTAssertNotNil(failure(name, Correctness.normalization(current: .failure(.invalidOperation("normalize", reason: "control")), input: [3, 4])))
        let zeroName = "normalize.zeroFail.512.optimized"
        XCTAssertNil(failure(zeroName, Correctness.normalization(current: .failure(.invalidOperation("normalize", reason: "zero")), input: [0, -0.0])))
        XCTAssertNotNil(failure(zeroName, Correctness.normalization(current: .success([0, 0]), input: [0, 0]), override: 1))
        XCTAssertNotNil(failure(zeroName, Correctness.normalization(current: .failure(.invalidData("control")), input: [0, 0]), override: 1))
        XCTAssertNotNil(failure(name, Correctness.normalization(current: .success([]), input: [])))
    }

    func testCopyRejectsTailPermutationULPAndSignedZeroCorruption() {
        let source: [Float] = [1, 2, 3, 4, -0.0, Float(bitPattern: 0x7fc00017)]
        func compare(_ destination: [Float]) -> CorrectnessStats {
            source.withUnsafeBufferPointer { src in destination.withUnsafeBufferPointer {
                Correctness.exactCopy(source: src, destination: $0)
            } }
        }
        XCTAssertNil(failure("mem.copy.1", compare(source)))
        var swapped = source; swapped.swapAt(2, 3)
        var ulp = source; ulp[3] = ulp[3].nextUp
        var zero = source; zero[4] = 0
        var nan = source; nan[5] = Float(bitPattern: 0x7fc00018)
        for wrong in [swapped, ulp, zero, nan, Array(source.dropLast())] {
            XCTAssertNotNil(failure("mem.copy.1", compare(wrong), override: 1))
        }
    }

    func testAttachingObservationPreservesRawMeasurementAndIdentity() throws {
        let boundary = try XCTUnwrap(Boundaries.lookup("dot.512.generic", runSeed: 0))
        let original = BenchResult(name: "dot.512.generic", iterations: 4, totalNanoseconds: 200, unitCount: 1,
            samples: 2, meanNsPerOp: 50, medianNsPerOp: 50, p90NsPerOp: 60, stddevNsPerOp: 10, rsdPercent: 20,
            executionFailure: "retained", boundary: boundary, sampleNsPerInvocation: [40, 60], sampleIterations: [2, 2], warmupIterations: 3)
        let result = original.validated(Correctness.scalar(current: 1, reference: 1))
        XCTAssertEqual(result.name, original.name)
        XCTAssertEqual(result.boundary, original.boundary)
        XCTAssertEqual(result.sampleNsPerInvocation, [40, 60]); XCTAssertEqual(result.sampleIterations, [2, 2])
        XCTAssertEqual(result.iterations, 4); XCTAssertEqual(result.totalNanoseconds, 200); XCTAssertEqual(result.unitCount, 1)
        XCTAssertEqual(result.samples, 2); XCTAssertEqual(result.warmupIterations, 3)
        XCTAssertEqual(result.meanNsPerOp, 50); XCTAssertEqual(result.medianNsPerOp, 50); XCTAssertEqual(result.p90NsPerOp, 60)
        XCTAssertEqual(result.stddevNsPerOp, 10); XCTAssertEqual(result.rsdPercent, 20); XCTAssertEqual(result.executionFailure, "retained")
        XCTAssertEqual(result.correctness?.samples, 1)
    }
    func testDotModelsUseSignedSumAbsoluteBoundsAtEveryWidth() throws {
        let rows: [(Int, Int, Double, Double)] = [
            (512, 37, 0.015625476880828604, 0.0011291529099318428),
            (768, 53, 0.03515785946459697, 0.002426155190759389),
            (1536, 101, 0.1406378760441469, 0.00924688210068572)
        ]
        for (dim, optimizedCount, genericBound, optimizedBound) in rows {
            let seeds = InputSeeds.dot(dim: dim, runSeed: 0)
            let a = InputFactory.randomArray(count: dim, seed: seeds.a)
            let b = InputFactory.randomArray(count: dim, seed: seeds.b)
            let products: [Double] = zip(a, b).map { Double($0) * Double($1) }
            let reference: Double = products.reduce(0, +)
            let largest = try XCTUnwrap(products.max(by: { abs($0) < abs($1) }))
            for variant in ["generic", "optimized"] {
                let name = "dot.\(dim).\(variant)"
                let model = try XCTUnwrap(CorrectnessPolicy.errorModel(forCaseName: name))
                XCTAssertEqual(model.route, variant == "generic" ? "generic-dot" : "optimized-dot")
                XCTAssertEqual(model.k32, variant == "generic" ? dim : optimizedCount)
                XCTAssertEqual(model.k64, dim)
                XCTAssertEqual(model.kind, .absolute(scale: Double(dim)))
                XCTAssertNil(CorrectnessPolicy.domainProblem(forCaseName: name))
                let rule = CorrectnessPolicy.rule(forCaseName: name)
                XCTAssertNil(rule.maxRel)
                let limit = try XCTUnwrap(rule.maxAbs)
                let expected = variant == "generic" ? genericBound : optimizedBound
                XCTAssertEqual(limit, expected, accuracy: expected * 1e-12)
                // An error outside the bound and the largest omitted product are both rejected.
                let perturbed = Correctness.scalar(current: Float(reference + 2 * limit), reference: reference)
                let omitted = Correctness.scalar(current: Float(reference - largest), reference: reference)
                XCTAssertNotNil(failure(name, perturbed))
                XCTAssertNotNil(failure(name, omitted))
                // A relative-only override cannot disable the derived absolute check.
                let result = BenchResult(name: name, iterations: 1, totalNanoseconds: 1, correctness: omitted)
                XCTAssertNotNil(CorrectnessPolicy.failure(for: result, overrideRel: 1, overrideAbs: nil))
            }
        }
        for name in ["dot.384.generic", "dot.512.generic.extra", "dot.512.unknown", "dist.dot.512.generic", "dot.1536.optimized.extra"] {
            XCTAssertFalse(CorrectnessPolicy.hasDerivedModel(forCaseName: name), name)
            XCTAssertNil(CorrectnessPolicy.errorModel(forCaseName: name), name)
        }
    }

    private func actualDot(_ a: [Float], _ b: [Float], dimension: Int, variant: String) throws -> Float {
        switch (dimension, variant) {
        case (512, "generic"): return try Vector<Dim512>(a).dotProduct(Vector<Dim512>(b))
        case (512, "optimized"): return try Vector512Optimized(a).dotProduct(Vector512Optimized(b))
        case (768, "generic"): return try Vector<Dim768>(a).dotProduct(Vector<Dim768>(b))
        case (768, "optimized"): return try Vector768Optimized(a).dotProduct(Vector768Optimized(b))
        case (1536, "generic"): return try Vector<Dim1536>(a).dotProduct(Vector<Dim1536>(b))
        case (1536, "optimized"): return try Vector1536Optimized(a).dotProduct(Vector1536Optimized(b))
        default: throw NSError(domain: "Unsupported test route", code: 1)
        }
    }

    // Exact one-coordinate products close the omitted-index class independently of a random-fixture tolerance.
    func testEveryDotRouteRetainsEachSignedCoordinateExactly() throws {
        for dimension in [512, 768, 1536] {
            var partner = [Float](repeating: 0, count: dimension)
            var input = [Float](repeating: 0, count: dimension)
            for variant in ["generic", "optimized"] {
                for index in input.indices {
                    partner[index] = 1
                    for sign in [Float(1), Float(-1)] {
                        input[index] = sign
                        let current = try actualDot(input, partner, dimension: dimension, variant: variant)
                        XCTAssertEqual(current.bitPattern, sign.bitPattern, "\(dimension).\(variant) coordinate \(index)")
                        input[index] = 0
                        // Deliberately removing this sole contribution must fail the same exact-output assertion.
                        let omitted = try actualDot(input, partner, dimension: dimension, variant: variant)
                        XCTAssertNotEqual(omitted.bitPattern, sign.bitPattern, "omitted \(dimension).\(variant) coordinate \(index)")
                    }
                    partner[index] = 0
                }
            }
        }
    }
}
