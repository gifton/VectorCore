import Foundation
import VectorCore
func q02InitializerAdmission(_ url: URL) {
    _ = KernelAutoTuner(persistenceURL: url)
    _ = MixedPrecisionAutoTuner()
    _ = QuantizedAutoTuner()
}
