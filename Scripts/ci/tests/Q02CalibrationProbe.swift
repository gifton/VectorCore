import Foundation
import VectorCore
@_silgen_name("q02_start") func start(_ threshold: UInt)
@_silgen_name("q02_stop") func stop()
@_silgen_name("q02_read") func read(_ field: Int32) -> UInt64
@_silgen_name("q02_calibrate_filter") func calibrateFilter()
@_silgen_name("q02_calibrate_c") func calibrateC()
@inline(never) func arrayControl(_ count: Int) -> [Float] { Array(repeating: 3, count: count) }
@main struct Q02CalibrationProbe {
    static func main() async throws {
        let mode = CommandLine.arguments[1], width = Int(CommandLine.arguments[2])!, count = Int(CommandLine.arguments[3])!
        print("Q02_BODY_BEGIN mode=\(mode)"); fflush(nil)
        let tuner = MixedPrecisionAutoTuner()
        if mode == "work" {
            MixedPrecisionDiagnostics.shared.enable()
            let metrics = try await tuner.calibrate(candidateCount: count, dimension: width)
            let elements = MixedPrecisionDiagnostics.shared.getStatistics().totalConversions
            MixedPrecisionDiagnostics.shared.disable()
            precondition(metrics.count == 5 && elements == 25 * (4 * count + 2) * width)
            print("CALIBRATION_WORK width=\(width) count=\(count) convertedElements=\(elements) strategies=\(metrics.count)")
        } else {
            calibrateFilter(); calibrateC()
            _ = arrayControl(4096).count
            start(0); stop(); precondition(read(0) == 0 && read(1) == 0)
            print("CALIBRATION noop events=\(read(0)) bytes=\(read(1))")
            start(0); let array = arrayControl(4096); stop()
            precondition(read(0) > 0 && read(1) >= 16384 && array.reduce(0,+) == 12288)
            print("CALIBRATION array events=\(read(0)) bytes=\(read(1))")
            let clock = ContinuousClock(), before = clock.now
            start(0)
            let metrics = try await tuner.calibrate(candidateCount: count, dimension: width)
            stop()
            let duration = before.duration(to: clock.now)
            let elapsed = Double(duration.components.seconds) + Double(duration.components.attoseconds) * 1e-18
            precondition(metrics.count == 5)
            print("EXPLICIT_CALIBRATION width=\(width) count=\(count) events=\(read(0)) bytes=\(read(1)) elapsedSeconds=\(elapsed) strategies=\(metrics.count)")
        }
        print("Q02_RESULT \(mode)")
        print("Q02_BODY_END mode=\(mode)")
    }
}
