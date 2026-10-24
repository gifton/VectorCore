import Foundation
import VectorCoreBenchmarking

// Task-local measurement seams (V04).
//
// Two independent seams, neither of which uses global mutable state:
// - `MeasurementPhases.observer`: lets an out-of-process-boundary probe (for example the allocation-event
//   probe) learn which phase of a case is running without copying the case bodies. It lives only in this
//   benchmark-support module; no VectorCore or VectorCoreBenchmarking product references it.
// - `CaseRecorder.current`: collects the raw per-sample values and warmup iteration counts that the
//   harness produces, so suite bodies need no changes to report them.
//
// Both are read once per harness call, outside every clock interval. A nil observer changes nothing.

package enum MeasurementPhase: String, Sendable {
    case setup, warmup, timed, validation
}

/// Observer contract. `CaseExecution` and the harness signal phases through `PhaseSignals`, which keeps the
/// sequence balanced: a `begin` while another phase is open first ends that phase, every `end` matches the
/// open phase, and the case's last open phase is ended when the case finishes (including after a throw).
/// An observer may still assume nothing about timing between a case's `end` and the next case's `begin`.
package protocol MeasurementPhaseObserver: Sendable {
    func begin(_ phase: MeasurementPhase, caseName: String)
    func end(_ phase: MeasurementPhase, caseName: String)
}

package enum MeasurementPhases {
    @TaskLocal package static var observer: (any MeasurementPhaseObserver)?
}

/// Per-case collector for harness observations. Lock-protected; one instance per executing case.
final class CaseRecorder: @unchecked Sendable {
    @TaskLocal static var current: CaseRecorder?

    let caseName: String
    private let lock = NSLock()
    private var warmups = 0
    private var sawWarmup = false
    private var sampleNs: [Double] = []
    private var sampleIters: [Int] = []

    private var openPhase: MeasurementPhase?
    let observer: (any MeasurementPhaseObserver)?

    init(caseName: String) {
        self.caseName = caseName
        self.observer = MeasurementPhases.observer
    }

    func begin(_ phase: MeasurementPhase) {
        let closing: MeasurementPhase? = lock.withLock {
            let previous = openPhase
            openPhase = phase
            return previous
        }
        if let closing { observer?.end(closing, caseName: caseName) }
        observer?.begin(phase, caseName: caseName)
    }

    func end(_ phase: MeasurementPhase) {
        let matched: Bool = lock.withLock {
            guard openPhase == phase else { return false }
            openPhase = nil
            return true
        }
        if matched { observer?.end(phase, caseName: caseName) }
    }

    /// Ends whatever phase is still open. Called once when the case returns or throws.
    func finish() {
        let closing: MeasurementPhase? = lock.withLock {
            let previous = openPhase
            openPhase = nil
            return previous
        }
        if let closing { observer?.end(closing, caseName: caseName) }
    }

    func recordWarmup(iterations: Int) {
        lock.withLock { warmups += iterations; sawWarmup = true }
    }

    /// Replaces the sample record. Only the outermost measurement call records, so a case with one
    /// measurement reports exactly its samples.
    func recordSamples(nsPerInvocation: [Double], iterations: [Int]) {
        lock.withLock { sampleNs = nsPerInvocation; sampleIters = iterations }
    }

    var warmupIterations: Int? { lock.withLock { sawWarmup ? warmups : nil } }
    var samples: (nsPerInvocation: [Double], iterations: [Int]) { lock.withLock { (sampleNs, sampleIters) } }
}

/// Phase signalling used by the harness. With a recorder (inside `CaseExecution`) the sequence is balanced;
/// without one (a harness call outside a case, for example a unit test) it forwards to the observer directly.
struct PhaseSignals {
    private let recorder: CaseRecorder?
    private let observer: (any MeasurementPhaseObserver)?
    private let name: String

    init(fallbackName: String) {
        recorder = CaseRecorder.current
        observer = MeasurementPhases.observer
        name = recorder?.caseName ?? fallbackName
    }

    func begin(_ phase: MeasurementPhase) {
        if let recorder { recorder.begin(phase) } else { observer?.begin(phase, caseName: name) }
    }

    func end(_ phase: MeasurementPhase) {
        if let recorder { recorder.end(phase) } else { observer?.end(phase, caseName: name) }
    }
}
