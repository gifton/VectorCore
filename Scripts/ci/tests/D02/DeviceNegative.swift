// D02 D5 (negative): each line uses one removed ComputeDevice discovery member or the removed DeviceCapabilities
// type, so after D02 each line is one compile error. Before D02 the file compiles.
import VectorCore

func isAvailable() -> Bool { ComputeDevice.cpu.isAvailable }
func isAccelerated() -> Bool { ComputeDevice.cpu.isAccelerated }
func requiredAlignment() -> Int { ComputeDevice.cpu.requiredAlignment }
func minimumEfficientSize() -> Int { ComputeDevice.cpu.minimumEfficientSize }
func queryCapabilities() -> Any? { ComputeDevice.cpu.queryCapabilities() }
func availableDevices() -> [ComputeDevice] { ComputeDevice.availableDevices }
func mostCapable() -> ComputeDevice { ComputeDevice.mostCapable }
func precision() -> Any { DeviceCapabilities.Precision.float32 }
