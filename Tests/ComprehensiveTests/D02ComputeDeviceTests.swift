import Foundation
import Testing
import VectorCore

// D02 phase 4: ComputeDevice is a descriptive label; VectorCore supplies a provider for `.cpu` only.
// `isSupportedByCore` and `coreSupportedDevices` describe VectorCore's own providers, never hardware presence.
// The removed discovery surface (isAvailable, isAccelerated, requiredAlignment, minimumEfficientSize,
// queryCapabilities, availableDevices, mostCapable, DeviceCapabilities) is pinned absent by the external
// negative consumer Scripts/ci/tests/D02/DeviceNegative.swift; the absence of hardware probes in
// ComputeDevice.swift is a lexical guard in Scripts/ci/tests/test_d02_surface_guards.py.

@Suite("D02 ComputeDevice Core provider support")
struct D02ComputeDeviceTests {
    private static let labels: [ComputeDevice] =
        [.cpu, .neural] + [Int.min, -1, 0, 1, Int.max].map { ComputeDevice.gpu(index: $0) }

    @Test("Support, the supported list and createProvider agree for cpu, neural and gpu indices Int.min/-1/0/1/Int.max")
    func supportListAndProviderAgree() {
        #expect(ComputeDevice.coreSupportedDevices == [.cpu])
        for device in Self.labels {
            let provider = device.createProvider()
            #expect(device.isSupportedByCore == (provider != nil), "\(device)")
            #expect(device.isSupportedByCore == ComputeDevice.coreSupportedDevices.contains(device), "\(device)")
            switch device {
            case .cpu:
                #expect(device.isSupportedByCore)
                #expect(provider is CPUComputeProvider)
                #expect(provider?.device == .cpu)
            case .gpu, .neural:
                #expect(!device.isSupportedByCore, "\(device)")
                #expect(provider == nil, "\(device)")
            }
        }
        for device in ComputeDevice.coreSupportedDevices {
            #expect(device.isSupportedByCore && device.createProvider() != nil)
        }
    }

    @Test("The cpu provider is CPUComputeProvider.automatic")
    func cpuProviderIsAutomatic() throws {
        let provider = try #require(ComputeDevice.cpu.createProvider() as? CPUComputeProvider)
        let automatic = CPUComputeProvider.automatic
        #expect(provider.device == automatic.device)
        #expect(provider.maxConcurrency == automatic.maxConcurrency)
    }

    @Test("Descriptions are preserved")
    func descriptions() {
        #expect(ComputeDevice.cpu.description == "CPU")
        #expect(ComputeDevice.gpu().description == "GPU")
        #expect(ComputeDevice.gpu(index: 0).description == "GPU")
        #expect(ComputeDevice.gpu(index: 1).description == "GPU[1]")
        #expect(ComputeDevice.gpu(index: -1).description == "GPU[-1]")
        #expect(ComputeDevice.gpu(index: Int.min).description == "GPU[-9223372036854775808]")
        #expect(ComputeDevice.gpu(index: Int.max).description == "GPU[9223372036854775807]")
        #expect(ComputeDevice.neural.description == "Neural Engine")
    }

    @Test("Hashing and equality are preserved")
    func hashingAndEquality() {
        #expect(ComputeDevice.gpu(index: 1) != ComputeDevice.gpu(index: 2))
        #expect(ComputeDevice.gpu() == ComputeDevice.gpu(index: 0))
        #expect(ComputeDevice.cpu != ComputeDevice.neural && ComputeDevice.cpu != ComputeDevice.gpu())
        let set: Set<ComputeDevice> = [.cpu, .cpu, .gpu(), .gpu(index: 0), .gpu(index: 1), .neural, .neural]
        #expect(set.count == 4)
        #expect(set.contains(.gpu(index: 0)) && set.contains(.gpu(index: 1)))
        var hasher = Hasher(), other = Hasher()
        ComputeDevice.gpu().hash(into: &hasher)
        ComputeDevice.gpu(index: 0).hash(into: &other)
        #expect(hasher.finalize() == other.finalize())
    }
}
