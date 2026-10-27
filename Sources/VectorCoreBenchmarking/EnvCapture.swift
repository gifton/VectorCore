import Foundation
#if canImport(Darwin)
import Darwin
import MachO
#endif
import VectorCore

/// Utilities for capturing benchmark environment metadata
///
/// Collects system information including:
/// - OS version, architecture, CPU core count
/// - Device model, and `swiftc -version` resolved through PATH at run time (not the compiler that built the binary)
/// - git HEAD of the working directory (not the build's source), and binary file sizes
/// - Runtime configuration flags
public enum EnvCapture {
    /// Collect complete metadata for a benchmark run
    ///
    /// - Parameters:
    ///   - suites: List of benchmark suites being executed
    ///   - dims: Dimensions being tested
    ///   - buildConfig: Build configuration (debug/release)
    ///   - runSeed: Random seed for reproducibility
    ///   - runLabel: Optional label for this run
    ///   - flags: Runtime feature flags
    ///   - thresholds: Optional correctness thresholds
    ///   - filters: Optional case filters
    /// - Returns: Complete benchmark metadata
    public static func collectMetadata(
        suites: [String],
        dims: [Int],
        buildConfig: String,
        runSeed: UInt64,
        runLabel: String?,
        flags: RunFlags,
        thresholds: RunThresholds? = nil,
        filters: RunFilters? = nil,
        selectedCaseIDs: [String] = [], samples: Int = 1, repeats: Int? = nil, minTimeSeconds: Double = 0,
        batchNs: [Int] = [], profile: String = "custom", mode: String? = nil,
        unavailableCaseIDs: [String] = [], thermalStateStart: RuntimeObservation? = nil, harnessCalibration: String = "unknown"
    ) -> BenchMetadata {
        let package = "VectorCore"
        let packageVersion = VectorCoreVersion.versionString
        let checkout = observe(command: ["/usr/bin/env", "git", "rev-parse", "HEAD"])
        let gitSHA = checkout.value
        let date = ISO8601DateFormatter().string(from: Date())
        let os = osVersionString()
        let arch = currentArch()
        let cpuCores = ProcessInfo.processInfo.activeProcessorCount
        let deviceModel = sysctlString(["machdep.cpu.brand_string", "hw.model", "hw.machine"]) // first non-nil
        let toolchain = observe(command: ["/usr/bin/env", "swiftc", "-version"])
        let swiftVersion = toolchain.value
        let deviceTag = computeDeviceTag(arch: arch)
        let sizes = binarySizes()
        return BenchMetadata(
            package: package,
            packageVersion: packageVersion,
            gitSHA: gitSHA,
            date: date,
            os: os,
            arch: arch,
            cpuCores: cpuCores,
            deviceModel: deviceModel,
            swiftVersion: swiftVersion,
            buildConfiguration: buildConfig,
            deviceTag: deviceTag,
            binarySizes: sizes,
            suites: suites,
            dims: dims,
            runSeed: runSeed,
            runLabel: runLabel,
            flags: flags,
            thresholds: thresholds,
            filters: filters, coreConfiguration: VectorCoreCompiledFeatures.configuration, benchmarkConfiguration: buildConfig,
            selectedCaseIDs: selectedCaseIDs, samples: samples, repeats: repeats, minTimeSeconds: minTimeSeconds,
            batchNs: batchNs, profile: profile, mode: mode,
            runtimeCheckout: checkout, runtimeToolchain: toolchain,
            unavailableCaseIDs: unavailableCaseIDs,
            thermalStateStart: thermalStateStart ?? .unavailable("Start-of-run thermal state was not captured"),
            thermalStateEnd: thermalState(), lowPowerMode: lowPowerMode(),
            harnessCalibration: harnessCalibration
        )
    }

    /// Current `ProcessInfo.thermalState` as an explicit observation. This is a coarse OS state, not a temperature.
    public static func thermalState() -> RuntimeObservation {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return RuntimeObservation(value: "nominal")
        case .fair: return RuntimeObservation(value: "fair")
        case .serious: return RuntimeObservation(value: "serious")
        case .critical: return RuntimeObservation(value: "critical")
        @unknown default: return .unavailable("Unrecognized ProcessInfo.thermalState value")
        }
    }

    public static func lowPowerMode() -> RuntimeObservation {
        RuntimeObservation(value: ProcessInfo.processInfo.isLowPowerModeEnabled ? "enabled" : "disabled")
    }

    private static func currentArch() -> String {
        #if arch(arm64)
        return "arm64"
        #elseif arch(x86_64)
        return "x86_64"
        #else
        return "unknown"
        #endif
    }

    private static func sysctlString(_ keys: [String]) -> String? {
        for key in keys {
            var size: size_t = 0
            if sysctlbyname(key, nil, &size, nil, 0) == 0 && size > 0 {
                var buf = [CChar](repeating: 0, count: Int(size))
                if sysctlbyname(key, &buf, &size, nil, 0) == 0 {
                    let count = Int(size)
                    if count > 0 {
                        // sysctl strings are NUL-terminated; drop the trailing NUL if present
                        let u8: [UInt8] = buf.prefix(count).dropLast().map { UInt8(bitPattern: $0) }
                        let str = String(decoding: u8, as: UTF8.self)
                        if !str.isEmpty { return str }
                    }
                }
            }
        }
        return nil
    }

    private static func osVersionString() -> String {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        let version = "os\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
        // macOS keeps the historical bare form; other families are named so a simulator or
        // Catalyst run is not mistaken for a macOS run (the host CPU model is still recorded).
        #if os(macOS)
        return version
        #elseif targetEnvironment(macCatalyst)
        return "maccatalyst-" + version
        #elseif os(iOS)
        return "ios-" + version
        #elseif os(tvOS)
        return "tvos-" + version
        #elseif os(watchOS)
        return "watchos-" + version
        #elseif os(visionOS)
        return "visionos-" + version
        #else
        return "unknown-" + version
        #endif
    }

    private static func computeDeviceTag(arch: String) -> String {
        // Prefer explicit tag from env; otherwise derive stable shorthand
        if let tag = ProcessInfo.processInfo.environment["VC_DEVICE_TAG"], !tag.isEmpty {
            return tag
        }
        let os = osVersionString()
        // Try hostname for local disambiguation
        #if os(macOS)
        let host = (Host.current().localizedName ?? "local").replacingOccurrences(of: " ", with: "-")
        #else
        // Foundation's `Host` exists only on macOS; record that instead of inventing a name.
        let host = "host-unavailable"
        #endif
        return "\(arch)-\(os)-\(host)"
    }

    private static func binarySizes() -> BinarySizes {
        let exePath = CommandLine.arguments.first ?? ""
        var exeBytes: UInt64 = 0
        if !exePath.isEmpty {
            if let attrs = try? FileManager.default.attributesOfItem(atPath: exePath),
               let size = attrs[.size] as? NSNumber {
                exeBytes = size.uint64Value
            }
        }

        // A loaded VectorCore dynamic library or framework binary; nil when VectorCore is linked statically.
        var vcBytes: UInt64? = nil
        for path in loadedImagePaths() where isVectorCoreLibraryImage(path) {
            if let attrs = try? FileManager.default.attributesOfItem(atPath: path),
               let size = attrs[.size] as? NSNumber {
                vcBytes = size.uint64Value
                break
            }
        }

        return BinarySizes(benchExecutableBytes: exeBytes, vectorCoreLibraryBytes: vcBytes)
    }

    /// Paths of the images loaded in this process, in dyld order (empty off Darwin).
    static func loadedImagePaths() -> [String] {
        #if canImport(Darwin)
        return (0..<_dyld_image_count()).compactMap { index in
            _dyld_get_image_name(index).flatMap { String(validatingCString: $0) }
        }
        #else
        return []
        #endif
    }

    /// Whether `path` names a VectorCore library binary by its file name: `libVectorCore.dylib`, or a file named
    /// `VectorCore` inside a `VectorCore.framework` directory. A path that only contains "VectorCore" in a directory or in
    /// another file name (an executable built inside a VectorCore checkout, `libVectorCoreBenchmarking.dylib`) does not
    /// match (`testVectorCoreLibraryImageMatchesFileNameOnly`).
    static func isVectorCoreLibraryImage(_ path: String) -> Bool {
        let components = path.split(separator: "/")
        guard let name = components.last else { return false }
        return name == "libVectorCore.dylib" || (name == "VectorCore" && components.dropLast().contains("VectorCore.framework"))
    }

    /// Internal command seam for deterministic timeout/error fixtures; never a CLI setting.
    static func observe(command: [String], timeout: TimeInterval = 2) -> RuntimeObservation {
        guard let executable = command.first, timeout > 0, timeout <= 15 else { return .unavailable("Invalid observation command/deadline") }
        #if os(macOS)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("V02-metadata-" + UUID().uuidString)
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let output = directory.appendingPathComponent("stdout")
            let errors = directory.appendingPathComponent("stderr")
            FileManager.default.createFile(atPath: output.path, contents: nil)
            FileManager.default.createFile(atPath: errors.path, contents: nil)
            let stdout = try FileHandle(forWritingTo: output), stderr = try FileHandle(forWritingTo: errors)
            defer { try? stdout.close(); try? stderr.close() }
            let process = Process()
            process.executableURL = URL(fileURLWithPath: executable)
            process.arguments = Array(command.dropFirst())
            process.standardOutput = stdout; process.standardError = stderr
            let finished = DispatchSemaphore(value: 0)
            process.terminationHandler = { _ in finished.signal() }
            try process.run()
            if finished.wait(timeout: .now() + timeout) == .timedOut {
                kill(process.processIdentifier, SIGKILL)
                let cleaned = finished.wait(timeout: .now() + 2) == .success
                // Never inspect termination properties after an execution timeout.
                return .unavailable("Observation timeout; cleanup completed: \(cleaned)")
            }
            guard process.terminationReason == .exit, process.terminationStatus == 0 else {
                return .unavailable("Observation command failed: \(process.terminationStatus)")
            }
            let text = try String(contentsOf: output, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
            return text.isEmpty ? .unavailable("Observation returned empty output") : RuntimeObservation(value: text)
        } catch { return .unavailable("Observation setup/read failed: \(error)") }
        #else
        // Foundation's `Process` is unavailable on this platform, so the observation is recorded as unavailable.
        _ = executable
        return .unavailable("Subprocess observation is unavailable on this platform (Foundation Process is macOS-only)")
        #endif
    }
}
