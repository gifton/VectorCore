import VectorCore

// Public supplied values avoid changing the existing type-construction API.
func nonthrowingConsumer(probes: CalibrationProbes, tuning: Tuning) -> Tuning {
    AutoTuning.set(dim: 12, kind: .dot, tuning: tuning)
    _ = AutoTuning.get(dim: 12, kind: .dot)
    return AutoTuning.calibrateIfNeeded(dim: 12, kind: .dot, providerCores: 2) { probes }
}
func throwingConsumer(callback: () throws -> CalibrationProbes) rethrows -> Tuning {
    try AutoTuning.calibrateIfNeeded(dim: 13, kind: .dot, providerCores: 2, calibrator: callback)
}
@MainActor func mainActorConsumer(callback: () throws -> CalibrationProbes) rethrows -> Tuning {
    try AutoTuning.calibrateIfNeeded(dim: 14, kind: .dot, providerCores: 2, calibrator: callback)
}
actor CalibrationConsumer {
    func call(probes: CalibrationProbes) -> Tuning {
        AutoTuning.calibrateIfNeeded(dim: 15, kind: .dot, providerCores: 2) { probes }
    }
}
