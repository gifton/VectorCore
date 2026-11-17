// D02 D1-D3 (positive): the descriptive labels, createProvider() and VectorCore's provider support, as an ordinary
// consumer sees them. Before D02, isSupportedByCore and coreSupportedDevices do not exist.
import VectorCore

@main
struct D02DeviceSupport {
    static func main() {
        let labels: [ComputeDevice] = [.cpu, .neural] + [Int.min, -1, 0, 1, Int.max].map { .gpu(index: $0) }
        for device in labels {
            let provided = device.createProvider() != nil
            guard device.isSupportedByCore == provided,
                  provided == ComputeDevice.coreSupportedDevices.contains(device) else {
                fatalError("D02 device support disagrees for \(device)")
            }
        }
        guard ComputeDevice.coreSupportedDevices == [.cpu], ComputeDevice.cpu.createProvider()?.device == .cpu else {
            fatalError("D02 Core-supported list")
        }
        print("D02_DEVICE_SUPPORT_COMPLETE \(labels.map(\.description))")
    }
}
