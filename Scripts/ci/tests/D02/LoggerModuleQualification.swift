// Module qualification after removal of the VectorCore namespace enum.
// The external consumer checks both Logger types with both modules imported.
import os
import VectorCore

@main
struct D02LoggerModuleQualification {
    static func main() {
        let core: VectorCore.Logger = VectorCore.Logger(category: "d02")
        let system: os.Logger = os.Logger(subsystem: "d02", category: "d02")
        let names = (String(reflecting: type(of: core)), String(reflecting: type(of: system)))
        precondition(names == ("VectorCore.Logger", "os.Logger"))
        print("D02_LOGGER_MODULE_QUALIFICATION_COMPLETE")
    }
}
