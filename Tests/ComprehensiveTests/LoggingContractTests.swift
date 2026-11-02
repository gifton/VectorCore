import Testing
import Foundation
@testable import VectorCore

// D03 contracts: lazy log messages and monotonic cumulative PerformanceTimer.
// Tests use a per-instance configuration and record sink (internal seam), so they do
// not touch the shared Logger.configuration. The public-consumer process is in
// LoggingPublicConsumerTests.

private let levels: [LogLevel] = [.debug, .info, .warning, .error, .critical]

private final class Recorder: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: [LogRecord] = []
    func append(_ record: LogRecord) { lock.lock(); defer { lock.unlock() }; stored.append(record) }
    var records: [LogRecord] { lock.lock(); defer { lock.unlock() }; return stored }
}

private final class Spy: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0
    /// Side-effectful: every call increments the evaluation count.
    func evaluate() -> Int { lock.lock(); defer { lock.unlock() }; value += 1; return value }
    var count: Int { lock.lock(); defer { lock.unlock() }; return value }
}

/// Reports Instants from a fixed list (last one repeats) and counts reads.
private final class SequenceClock: @unchecked Sendable {
    private let lock = NSLock()
    private let instants: [ContinuousClock.Instant]
    private var reads = 0
    init(_ instants: [ContinuousClock.Instant]) { self.instants = instants }
    func read() -> ContinuousClock.Instant {
        lock.lock(); defer { lock.unlock() }
        defer { reads += 1 }
        return instants[min(reads, instants.count - 1)]
    }
    var readCount: Int { lock.lock(); defer { lock.unlock() }; return reads }
}

private func makeLogger(minimum: LogLevel, recorder: Recorder,
                        category: String = "D03") -> (Logger, LogConfiguration) {
    let configuration = LogConfiguration()
    configuration.minimumLevel = minimum
    let logger = Logger(category: category, configuration: configuration,
                        sink: { recorder.append($0) })
    return (logger, configuration)
}

private struct Site {
    let file: String
    let function: String
    let line: Int
}

private func emit(_ logger: Logger, _ level: LogLevel, _ message: @autoclosure () -> String, at site: Site) {
    let (file, function, line) = (site.file, site.function, site.line)
    switch level {
    case .debug: logger.debug(message(), file: file, function: function, line: line)
    case .info: logger.info(message(), file: file, function: function, line: line)
    case .warning: logger.warning(message(), file: file, function: function, line: line)
    case .error: logger.error(message(), file: file, function: function, line: line)
    case .critical: logger.critical(message(), file: file, function: function, line: line)
    }
}

@Suite struct LoggingContractTests {
    // MARK: Message evaluation: 5 levels x 5 valid minima

    @Test(arguments: levels, levels)
    func messageEvaluationMatrix(level: LogLevel, minimum: LogLevel) {
        let recorder = Recorder()
        let (logger, _) = makeLogger(minimum: minimum, recorder: recorder, category: "Matrix")
        let spy = Spy()
        emit(logger, level, "payload \(spy.evaluate())",
             at: Site(file: "Module/Source.swift", function: "fn(arg:)", line: 321))
        let admitted = level >= minimum
        #expect(spy.count == (admitted ? 1 : 0), "\(level)@\(minimum)")
        if admitted {
            #expect(recorder.records == [LogRecord(level: level, message: "payload 1",
                                                   location: "Source.swift:321",
                                                   function: "fn(arg:)", category: "Matrix")])
        } else {
            #expect(recorder.records.isEmpty)
        }
    }

    @Test func defaultSourceLocationIsForwarded() {
        let recorder = Recorder()
        let (logger, _) = makeLogger(minimum: .debug, recorder: recorder)
        let line = #line + 1
        logger.info("where")
        let record = recorder.records.first
        #expect(record?.location == "LoggingContractTests.swift:\(line)")
        #expect(record?.function == "defaultSourceLocationIsForwarded()")
    }

    // MARK: VectorError overload

    @Test func vectorErrorOverloadForwardsAndRespectsGate() {
        let error = VectorError.dimensionMismatch(expected: 2, actual: 3)
        let recorder = Recorder()
        let (logger, configuration) = makeLogger(minimum: .error, recorder: recorder)
        logger.error(error, file: "M/E.swift", function: "f", line: 7)
        logger.error(error, message: "caller text", file: "M/E.swift", function: "f", line: 8)
        #expect(recorder.records.map(\.message) == [error.description, "caller text"])
        #expect(recorder.records.map(\.location) == ["E.swift:7", "E.swift:8"])
        #expect(recorder.records.allSatisfy { $0.level == .error })

        // VectorError.log forwards logger, message and source location.
        error.log(to: logger, message: "via log", file: "M/L.swift", function: "g", line: 9)
        #expect(recorder.records.last == LogRecord(level: .error, message: "via log", location: "L.swift:9",
                                                   function: "g", category: "D03"))
        // Gated calls emit nothing; fallback laziness is vectorErrorFallbackDescriptionIsLazy.
        configuration.minimumLevel = .critical
        let before = recorder.records.count
        logger.error(error)
        error.log(to: logger)
        #expect(recorder.records.count == before)
    }

    private struct SpyError: Error, CustomStringConvertible {
        let spy: Spy
        var description: String { "spy\(spy.evaluate())" }
    }

    /// VectorError.description interpolates `underlying`, so a counting description
    /// observes whether the fallback `error.description` is computed.
    @Test func vectorErrorFallbackDescriptionIsLazy() {
        let spy = Spy()
        let error = VectorError(.invalidData, message: "m", underlying: SpyError(spy: spy))
        let recorder = Recorder()
        let (logger, configuration) = makeLogger(minimum: .critical, recorder: recorder)
        let base = spy.count   // construction must not be counted against the logger

        logger.error(error)
        error.log(to: logger)
        #expect(spy.count == base, "gated calls must not compute error.description")
        #expect(recorder.records.isEmpty)

        for minimum in [LogLevel.error, .warning, .info, .debug] {
            configuration.minimumLevel = minimum
            let before = spy.count
            logger.error(error)
            #expect(spy.count == before + 1, "logger.error admitted at \(minimum)")
            error.log(to: logger)
            #expect(spy.count == before + 2, "error.log admitted at \(minimum)")
        }

        // Caller-supplied message wins; the ?? right-hand side is not evaluated.
        let before = spy.count
        logger.error(error, message: "caller")
        error.log(to: logger, message: "caller")
        #expect(spy.count == before)
        #expect(recorder.records.suffix(2).map(\.message) == ["caller", "caller"])
    }

    // MARK: Actual backend reuse

    @Test func actualBackendEvaluatesAdmittedMessageOnce() {
        let configuration = LogConfiguration()
        let logger = Logger(category: "D03Backend", configuration: configuration, sink: nil)
        for minimum in levels {
            configuration.minimumLevel = minimum
            for level in levels {
                let spy = Spy()
                emit(logger, level, "backend \(spy.evaluate())", at: Site(file: "A/B.swift", function: "f", line: 1))
                #expect(spy.count == (level >= minimum ? 1 : 0), "\(level)@\(minimum)")
            }
        }
    }

    // MARK: Reentrancy (bounded child)

    @Test func reentrantMessageDoesNotDeadlock() throws {
        let result = try d03RunChild(scenario: "reentrancy",
                                     filter: "LoggingContractTests/childReentrancy")
        #expect(result.status == 0, "reentrancy child failed: \(result.output.prefix(4000))")
        #expect(result.output.contains("D03_REENTRANCY_COMPLETE"))
    }

    /// Runs only in the child process: an admitted message reads and updates the
    /// configuration and logs recursively. A held logger lock would hang here and the
    /// parent's 15 s deadline would fail the test.
    @Test func childReentrancy() {
        guard ProcessInfo.processInfo.environment["D03_CHILD"] == "reentrancy" else { return }
        let recorder = Recorder()
        let (logger, configuration) = makeLogger(minimum: .debug, recorder: recorder)
        let a = Spy(), b = Spy(), c = Spy(), d = Spy()
        func messageC() -> String { "C\(c.evaluate())" }
        func messageD() -> String { "D\(d.evaluate())" }
        func messageB() -> String {
            _ = b.evaluate()
            configuration.minimumLevel = .error          // later snapshots see .error
            logger.warning(messageC())                    // gated: not evaluated
            return "B"
        }
        func messageA() -> String {
            _ = a.evaluate()
            _ = configuration.minimumLevel                // read from inside a message
            configuration.minimumLevel = .warning         // update from inside a message
            logger.warning(messageB())                    // nested, admitted
            logger.info(messageD())                       // gated
            return "A"
        }
        logger.info(messageA())
        let counts: [Int] = [a.count, b.count, c.count, d.count]
        #expect(counts == [1, 1, 0, 0])
        // The outer info record was admitted before the level moved; it is not retracted.
        #expect(recorder.records.map(\.message) == ["B", "A"])
        #expect(configuration.minimumLevel == .error)
        print("D03_REENTRANCY_COMPLETE counts=\(counts)")
    }

    // MARK: PerformanceTimer

    private func timerLogger(minimum: LogLevel = .debug) -> (Logger, Recorder) {
        let recorder = Recorder()
        return (makeLogger(minimum: minimum, recorder: recorder, category: "Timer").0, recorder)
    }

    @Test func timerBoundaryReadings() {
        let base = ContinuousClock.now
        // start, then readings: 0, 1ms - 1ns, 1ms, 1ms + 1ns
        let clock = SequenceClock([base, base, base.advanced(by: .nanoseconds(999_999)),
                                   base.advanced(by: .milliseconds(1)),
                                   base.advanced(by: .nanoseconds(1_000_001))])
        let (logger, recorder) = timerLogger()
        let timer = PerformanceTimer(operation: "op", logger: logger, clock: { clock.read() })
        timer.log(threshold: 0.001)   // elapsed 0
        timer.log(threshold: 0.001)   // just below
        #expect(recorder.records.isEmpty)
        timer.log(threshold: 0.001)   // equality emits
        timer.log(threshold: 0.001)   // above emits
        #expect(recorder.records.map(\.message) == ["op took 1.000ms", "op took 1.000ms"])
        #expect(recorder.records.allSatisfy { $0.level == .debug && $0.category == "Timer" })
        #expect(clock.readCount == 5)
    }

    @Test func timerZeroThresholdEmitsImmediateZero() {
        let base = ContinuousClock.now
        let clock = SequenceClock([base, base])
        let (logger, recorder) = timerLogger()
        PerformanceTimer(operation: "z", logger: logger, clock: { clock.read() }).log(threshold: 0)
        #expect(recorder.records.map(\.message) == ["z took 0.000ms"])
    }

    @Test func timerReportsCumulativeNotLapElapsed() {
        let base = ContinuousClock.now
        let clock = SequenceClock([base, base.advanced(by: .milliseconds(2)),
                                   base.advanced(by: .milliseconds(5)),
                                   base.advanced(by: .seconds(3) + .milliseconds(500))])
        let (logger, recorder) = timerLogger()
        let timer = PerformanceTimer(operation: "cum", logger: logger, clock: { clock.read() })
        timer.log(threshold: 0); timer.log(threshold: 0); timer.log(threshold: 0)
        #expect(recorder.records.map(\.message) ==
                ["cum took 2.000ms", "cum took 5.000ms", "cum took 3500.000ms"])
        #expect(clock.readCount == 4)
    }

    @Test(arguments: [-1.0, -Double.leastNonzeroMagnitude, Double.nan, Double.infinity, -Double.infinity])
    func timerInvalidThresholdSuppressesButStillReadsClock(threshold: Double) {
        let base = ContinuousClock.now
        let clock = SequenceClock([base, base.advanced(by: .seconds(10)), base.advanced(by: .seconds(20))])
        let (logger, recorder) = timerLogger()
        let timer = PerformanceTimer(operation: "bad", logger: logger, clock: { clock.read() })
        timer.log(threshold: threshold)
        timer.log(threshold: threshold)
        #expect(recorder.records.isEmpty)
        #expect(clock.readCount == 3)   // one start read plus one per log call
    }

    @Test func timerHugeFiniteThresholdDoesNotEmitOrTrap() {
        let base = ContinuousClock.now
        let clock = SequenceClock([base, base.advanced(by: .seconds(1_000))])
        let (logger, recorder) = timerLogger()
        let timer = PerformanceTimer(operation: "huge", logger: logger, clock: { clock.read() })
        timer.log(threshold: Double.greatestFiniteMagnitude)
        timer.log(threshold: 1e300)
        #expect(recorder.records.isEmpty)
    }

    @Test func timerNegativeInjectedElapsedSuppresses() {
        let base = ContinuousClock.now
        let clock = SequenceClock([base, base.advanced(by: .milliseconds(-5))])
        let (logger, recorder) = timerLogger()
        PerformanceTimer(operation: "neg", logger: logger, clock: { clock.read() }).log(threshold: 0)
        #expect(recorder.records.isEmpty)
        #expect(clock.readCount == 2)
    }

    @Test func timerSuppressedWhenDebugLevelIsDisabled() {
        let base = ContinuousClock.now
        let clock = SequenceClock([base, base.advanced(by: .seconds(1))])
        let (logger, recorder) = timerLogger(minimum: .info)
        PerformanceTimer(operation: "gated", logger: logger, clock: { clock.read() }).log(threshold: 0)
        #expect(recorder.records.isEmpty)
        #expect(clock.readCount == 2)
    }
}
