// D02 RD-R5 (positive): a file that imports both `os` and VectorCore names VectorCore's logger through the scoped
// import Logger.swift documents, and Apple's as `os.Logger`. D02 changes only documentation here, so the program
// runs the same before and after D02.
import os
import VectorCore
import struct VectorCore.Logger

@main
struct D02LoggerQualification {
    static func main() {
        let core: Logger = Logger(category: "d02")
        let system: os.Logger = os.Logger(subsystem: "d02", category: "d02")
        let names = (String(reflecting: type(of: core)), String(reflecting: type(of: system)))
        guard names == ("VectorCore.Logger", "os.Logger") else {
            fatalError("D02 logger qualification resolved \(names)")
        }
        print("D02_LOGGER_QUALIFICATION_COMPLETE \(names.0) \(names.1)")
    }
}
