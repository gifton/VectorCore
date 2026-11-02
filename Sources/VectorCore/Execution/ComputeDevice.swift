// VectorCore: ComputeDevice Enumeration
//
// Descriptive device labels, and which of them VectorCore itself supplies a compute provider for.
//

import Foundation

/// A descriptive label for the kind of device a compute provider runs on.
///
/// `ComputeDevice` names a requested or reported execution target; it does not discover hardware.
/// VectorCore itself supplies a provider only for `.cpu` (`CPUComputeProvider.automatic`). `.gpu(index:)`
/// and `.neural` are labels that another package's `ComputeProvider` may report as its `device`.
/// To learn whether a physical GPU or Neural Engine is present, use that backend's own discovery API;
/// to run on one, install that package's provider with `Operations.$computeProvider.withValue(_:)`.
public enum ComputeDevice: Sendable, Hashable, CustomStringConvertible {
    /// CPU execution on the processor cores, the label of VectorCore's `CPUComputeProvider`.
    case cpu

    /// A GPU label. `index` distinguishes GPUs for a provider that drives several (default 0);
    /// VectorCore gives it no further meaning and supplies no GPU provider.
    case gpu(index: Int = 0)

    /// A Neural Engine label. VectorCore supplies no Neural Engine provider.
    case neural

    /// Human-readable description of the device
    public var description: String {
        switch self {
        case .cpu:
            return "CPU"
        case .gpu(let index):
            return index == 0 ? "GPU" : "GPU[\(index)]"
        case .neural:
            return "Neural Engine"
        }
    }
}

// MARK: - VectorCore provider support

public extension ComputeDevice {
    /// Whether VectorCore itself supplies a compute provider for this label: `true` for `.cpu`, `false` for
    /// every `.gpu(index:)` and for `.neural`. Always equal to `createProvider() != nil`
    /// (D02ComputeDeviceTests).
    ///
    /// This describes VectorCore's own providers only. It says nothing about whether the device's
    /// hardware is present on this machine, and nothing about providers that other packages supply.
    var isSupportedByCore: Bool {
        switch self {
        case .cpu:
            return true
        case .gpu, .neural:
            return false
        }
    }

    /// The labels for which VectorCore itself supplies a compute provider: `[.cpu]`.
    ///
    /// Like `isSupportedByCore`, this lists VectorCore's own providers only. It is not a list of the
    /// hardware present on this machine and does not include providers that other packages supply.
    static var coreSupportedDevices: [ComputeDevice] { [.cpu] }

    /// The compute provider VectorCore supplies for this label: `CPUComputeProvider.automatic` for `.cpu`,
    /// and `nil` for every `.gpu(index:)` and for `.neural`, for which VectorCore has no provider.
    func createProvider() -> (any ComputeProvider)? {
        switch self {
        case .cpu:
            return CPUComputeProvider.automatic
        case .gpu, .neural:
            return nil
        }
    }
}
