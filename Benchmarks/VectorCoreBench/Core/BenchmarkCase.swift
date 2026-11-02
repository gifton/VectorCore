import Foundation
import VectorCoreBenchmarking

enum CaseAvailability: Sendable, Equatable {
    case available
    /// The route does not exist. The entry is selected like any other but is never invoked, timed or validated.
    case unavailable(reason: String)
}

struct BenchmarkCase: Sendable {
    let name: String
    /// The case's measurement boundary. Attached to every result the case produces.
    let boundary: MeasurementBoundary
    let availability: CaseAvailability
    /// True only for the default adapter of injected test fixture suites, whose results are named differently
    /// from the single adapter case. Production catalog cases must return results named exactly like the case.
    let acceptsAnyResultName: Bool
    let operation: @Sendable () async throws -> [BenchResult]
    init(_ name: String, boundary: MeasurementBoundary, availability: CaseAvailability = .available,
         acceptsAnyResultName: Bool = false, operation: @escaping @Sendable () async throws -> [BenchResult]) {
        self.name = name; self.boundary = boundary; self.availability = availability
        self.acceptsAnyResultName = acceptsAnyResultName; self.operation = operation
    }
}

protocol BenchmarkSuite: Sendable {
    static var name: String { get }
    static func catalog(options: ValidatedOptions, progress: BenchmarkProgressReporter) -> [BenchmarkCase]
    static func run(options: ValidatedOptions, progress: BenchmarkProgressReporter) async -> [BenchResult]
}

extension BenchmarkSuite {
    // Only injected V01 fixture suites use this adapter. Production suites define catalogs.
    static func catalog(options: ValidatedOptions, progress: BenchmarkProgressReporter) -> [BenchmarkCase] {
        [BenchmarkCase(name, boundary: Boundaries.fixtureInjected(name: name), acceptsAnyResultName: true) { await run(options: options,
            progress: progress) }]
    }
    static func run(options: ValidatedOptions, progress: BenchmarkProgressReporter) async -> [BenchResult] {
        await CaseExecution.run(catalog(options: options, progress: progress), options: options, progress: progress, suite: name)
    }
}

struct CaseExecutionOutput: Sendable {
    var results: [BenchResult] = []
    var unavailable: [UnavailableCase] = []
}

extension BenchResult {
    /// Copy with execution-time identity and raw measurements attached. Never alters measured values.
    func attaching(name: String? = nil, boundary: MeasurementBoundary?, samples recorded: (nsPerInvocation: [Double], iterations: [Int])?,
                   warmupIterations: Int?, executionFailure: String? = nil) -> BenchResult {
        BenchResult(name: name ?? self.name, iterations: iterations, totalNanoseconds: totalNanoseconds, unitCount: unitCount,
                    samples: samples, meanNsPerOp: meanNsPerOp, medianNsPerOp: medianNsPerOp, p90NsPerOp: p90NsPerOp,
                    stddevNsPerOp: stddevNsPerOp, rsdPercent: rsdPercent, correctness: correctness,
                    executionFailure: executionFailure ?? self.executionFailure, boundary: boundary,
                    sampleNsPerInvocation: sampleNsPerInvocation.isEmpty ? (recorded?.nsPerInvocation ?? []) : sampleNsPerInvocation,
                    sampleIterations: sampleIterations.isEmpty ? (recorded?.iterations ?? []) : sampleIterations,
                    warmupIterations: self.warmupIterations ?? warmupIterations)
    }
}

enum CaseExecution {
    /// Selection happens before setup: an excluded case's operation is never invoked. Unavailable cases are
    /// recorded without invoking their operation.
    static func execute(_ catalog: [BenchmarkCase], options: ValidatedOptions, progress: BenchmarkProgressReporter,
                        suite: String, onCase: (@Sendable (String) -> Void)? = nil) async -> CaseExecutionOutput {
        let selected = catalog.filter { options.caseFilter.matches($0.name) }
        var output = CaseExecutionOutput()
        for (index, item) in selected.enumerated() {
            if case .unavailable(let reason) = item.availability {
                output.unavailable.append(UnavailableCase(name: item.name, boundary: item.boundary, reason: reason))
                continue
            }
            onCase?(item.name)
            progress.caseStarted(suite: suite, name: item.name, index: index, total: selected.count)
            let start = Date()
            let recorder = CaseRecorder(caseName: item.name)
            recorder.begin(.setup)
            let produced: [BenchResult]
            do {
                produced = try await CaseRecorder.$current.withValue(recorder) { try await item.operation() }
            } catch {
                produced = [BenchResult(name: item.name, iterations: 0, totalNanoseconds: 0, executionFailure: String(describing: error))]
            }
            recorder.finish()
            if !item.acceptsAnyResultName, produced.count != 1 {
                // A production case reports exactly one result: zero is a silent drop, several hide duplicates.
                let failure = BenchResult(name: item.name, iterations: 0, totalNanoseconds: 0,
                                          executionFailure: "Case returned \(produced.count) results; exactly one is required")
                output.results.append(failure.attaching(boundary: item.boundary, samples: nil, warmupIterations: nil))
                progress.caseCompleted(suite: suite, name: item.name, index: index, total: selected.count, durationMs: Date().timeIntervalSince(start) * 1000)
                continue
            }
            output.results += produced.map { finish($0, item: item, recorder: recorder, single: produced.count == 1) }
            progress.caseCompleted(suite: suite, name: item.name, index: index, total: selected.count,
                durationMs: Date().timeIntervalSince(start) * 1000)
        }
        return output
    }

    private static func finish(_ result: BenchResult, item: BenchmarkCase, recorder: CaseRecorder, single: Bool) -> BenchResult {
        let recorded = single ? recorder.samples : nil
        let warmups = single ? recorder.warmupIterations : nil
        if item.acceptsAnyResultName {
            // Injected fixture adapter: keep a boundary an inner production catalog already attached.
            return result.attaching(boundary: result.boundary ?? item.boundary, samples: recorded, warmupIterations: warmups)
        }
        // Production catalog: the result must be the case's own, and must report the declared work unit.
        if result.name != item.name {
            return result.attaching(name: item.name, boundary: item.boundary, samples: nil, warmupIterations: nil,
                                    executionFailure: "Result name \(result.name) differs from catalog case \(item.name)")
        }
        if result.executionFailure == nil, result.unitCount != item.boundary.unitsPerInvocation {
            return result.attaching(boundary: item.boundary, samples: recorded, warmupIterations: warmups,
                                    executionFailure: "Reported unitCount \(result.unitCount) differs from declared boundary "
                                        + "unitsPerInvocation \(item.boundary.unitsPerInvocation)")
        }
        return result.attaching(boundary: item.boundary, samples: recorded, warmupIterations: warmups)
    }

    static func run(_ catalog: [BenchmarkCase], options: ValidatedOptions, progress: BenchmarkProgressReporter,
                    suite: String, onCase: (@Sendable (String) -> Void)? = nil) async -> [BenchResult] {
        await execute(catalog, options: options, progress: progress, suite: suite, onCase: onCase).results
    }
}
