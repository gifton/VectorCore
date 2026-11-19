import VectorCore

private enum UMAPInitialization {
    case consumer
}

@main
struct ModuleQualification {
    static func main() {
        let local: UMAPInitialization = .consumer
        let core: VectorCore.UMAPInitialization = .random
        let vector = VectorCore.DynamicVector([1.25, -2.5, 4])
        let logger = VectorCore.Logger(category: "module-qualification")

        guard case .consumer = local, case .random = core else {
            fatalError("Unexpected initialization strategy")
        }
        let elements: [Float] = vector.toArray()
        precondition(elements.map(\.bitPattern) == [0x3fa00000, 0xc0200000, 0x40800000])
        logger.debug("Module qualification fixture")
        print("MODULE_QUALIFICATION_OK")
    }
}
