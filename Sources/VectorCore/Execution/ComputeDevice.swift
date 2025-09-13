// VectorCore: ComputeDevice Enumeration
//
// Defines available compute devices for vector operations
//

import Foundation

/// Enumeration of available compute devices for vector operations
///
/// ComputeDevice represents the different processing units available
/// for executing vector operations, from traditional CPUs to specialized
/// accelerators like GPUs and Neural Engines.
public enum ComputeDevice: Sendable, Hashable, CustomStringConvertible {
    /// CPU execution using standard processor cores
    ///
    /// - Supports automatic parallelization via GCD/Swift Concurrency
    /// - Optimized for general-purpose computation
    /// - Best for small to medium datasets or complex branching logic
    case cpu

    /// GPU execution using Metal (when available)
    ///
    /// - Parameter index: GPU index for multi-GPU systems (default: 0)
    /// - Optimized for highly parallel, regular computations
    /// - Best for large datasets with simple per-element operations
    case gpu(index: Int = 0)

    /// Neural Engine execution (Apple Silicon)
    ///
    /// - Optimized for machine learning inference
    /// - Best for neural network operations
    /// - Currently experimental/future support
    case neural

    /// Whether this device provides hardware acceleration
    public var isAccelerated: Bool {
        switch self {
        case .cpu:
            return false
        case .gpu, .neural:
            return true
        }
    }

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

    /// Check if this device type is available on the current system
    public var isAvailable: Bool {
        switch self {
        case .cpu:
            // CPU is always available
            return true

        case .gpu:
            // Check for Metal support
            #if canImport(Metal) && !targetEnvironment(simulator)
            return true
            #else
            return false
            #endif

        case .neural:
            // Neural Engine requires Apple Silicon
            #if os(macOS) || os(iOS)
            if #available(macOS 11.0, iOS 14.0, *) {
                // Check for Apple Silicon (simplified check)
                return ProcessInfo.processInfo.processorCount > 0
            }
            #endif
            return false
        }
    }

    /// Memory alignment requirements for this device (in bytes)
    public var requiredAlignment: Int {
        switch self {
        case .cpu:
            // SIMD alignment requirements
            #if arch(arm64)
            return 16  // ARM NEON alignment
            #else
            return 32  // AVX alignment
            #endif

        case .gpu:
            // Metal buffer alignment
            return 256

        case .neural:
            // Neural Engine alignment (conservative)
            return 64
        }
    }

    /// Recommended minimum data size for efficient execution
    ///
    /// Operations smaller than this may have overhead that exceeds benefits
    public var minimumEfficientSize: Int {
        switch self {
        case .cpu:
            return 1000      // ~4KB of Float data
        case .gpu:
            return 10000     // ~40KB of Float data
        case .neural:
            return 1000      // Similar to CPU for now
        }
    }
}

// MARK: - Device Capabilities

/// Capabilities and characteristics of a compute device
public struct DeviceCapabilities: Sendable {
    /// Maximum number of parallel execution units
    public let maxParallelism: Int

    /// Available memory in bytes (nil if unlimited/unknown)
    public let availableMemory: Int?

    /// Supported precision modes
    public let supportedPrecisions: Set<Precision>

    /// Whether the device supports unified memory
    public let hasUnifiedMemory: Bool

    /// Precision modes for computation
    public enum Precision: String, Sendable {
        case float16 = "f16"
        case float32 = "f32"
        case float64 = "f64"
        case int8 = "i8"
        case int16 = "i16"
        case int32 = "i32"
    }
}

// MARK: - Device Query

public extension ComputeDevice {
    /// Query capabilities of this device
    ///
    /// - Returns: Device capabilities if available, nil if device not present
    func queryCapabilities() -> DeviceCapabilities? {
        guard isAvailable else { return nil }

        switch self {
        case .cpu:
            return DeviceCapabilities(
                maxParallelism: ProcessInfo.processInfo.activeProcessorCount,
                availableMemory: nil,  // No fixed limit
                supportedPrecisions: [.float32, .float64, .int32],
                hasUnifiedMemory: true
            )

        case .gpu:
            // Placeholder for Metal device query
            return DeviceCapabilities(
                maxParallelism: 1024,  // Typical GPU thread group size
                availableMemory: nil,  // Would query from MTLDevice
                supportedPrecisions: [.float16, .float32],
                hasUnifiedMemory: true  // True on Apple Silicon
            )

        case .neural:
            // Placeholder for Neural Engine
            return DeviceCapabilities(
                maxParallelism: 16,    // Hypothetical
                availableMemory: nil,
                supportedPrecisions: [.float16, .int8],
                hasUnifiedMemory: true
            )
        }
    }

    /// Get all available devices on this system
    static var availableDevices: [ComputeDevice] {
        var devices: [ComputeDevice] = [.cpu]

        if ComputeDevice.gpu().isAvailable {
            devices.append(.gpu())
        }

        if ComputeDevice.neural.isAvailable {
            devices.append(.neural)
        }

        return devices
    }

    /// Create appropriate compute provider for this device
    ///
    /// - Returns: ComputeProvider for the device, or nil if unavailable
    func createProvider() -> (any ComputeProvider)? {
        switch self {
        case .cpu:
            return CPUComputeProvider.automatic

        case .gpu:
            // GPU providers now live in VectorAccelerate package
            return nil

        case .neural:
            // Neural providers now live in VectorAI package
            return nil
        }
    }

    /// Get the most capable available device
    ///
    /// Priority order: Neural Engine > GPU > CPU
    static var mostCapable: ComputeDevice {
        let available = availableDevices

        if available.contains(.neural) {
            return .neural
        } else if available.contains(where: { if case .gpu = $0 { return true } else { return false } }) {
            return .gpu()
        } else {
            return .cpu
        }
    }
}
