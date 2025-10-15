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
/// - Device model, Swift version
/// - Git SHA, binary sizes
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
        buildConfig: String = "release",
        runSeed: UInt64,
        runLabel: String?,
        flags: RunFlags,
        thresholds: RunThresholds? = nil,
        filters: RunFilters? = nil
    ) -> BenchMetadata {
        let package = "VectorCore"
        let packageVersion = VectorCoreVersion.versionString
        let gitSHA = safeGitSHA()
        let date = ISO8601DateFormatter().string(from: Date())
        let os = osVersionString()
        let arch = currentArch()
        let cpuCores = ProcessInfo.processInfo.activeProcessorCount
        let deviceModel = sysctlString(["machdep.cpu.brand_string", "hw.model", "hw.machine"]) // first non-nil
        let swiftVersion = swiftcVersion()
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
            filters: filters
        )
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
        return "os\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
    }

    private static func computeDeviceTag(arch: String) -> String {
        // Prefer explicit tag from env; otherwise derive stable shorthand
        if let tag = ProcessInfo.processInfo.environment["VC_DEVICE_TAG"], !tag.isEmpty {
            return tag
        }
        let os = osVersionString()
        // Try hostname for local disambiguation
        let host = (Host.current().localizedName ?? "local").replacingOccurrences(of: " ", with: "-")
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

        // Attempt to discover a VectorCore dynamic library in loaded images (may be nil if statically linked)
        var vcBytes: UInt64? = nil
        #if canImport(Darwin)
        let count = _dyld_image_count()
        for i in 0..<count {
            if let cstr = _dyld_get_image_name(i), let path = String(validatingCString: cstr) {
                if path.contains("VectorCore") {
                    if let attrs = try? FileManager.default.attributesOfItem(atPath: path),
                       let size = attrs[.size] as? NSNumber {
                        vcBytes = size.uint64Value
                        break
                    }
                }
            }
        }
        #endif

        return BinarySizes(benchExecutableBytes: exeBytes, vectorCoreLibraryBytes: vcBytes)
    }

    private static func safeGitSHA() -> String? {
        let task = Process()
        task.launchPath = "/usr/bin/env"
        task.arguments = ["git", "rev-parse", "HEAD"]

        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = Pipe()
        do {
            try task.run()
        } catch {
            return nil
        }
        task.waitUntilExit()
        guard task.terminationStatus == 0 else { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let sha = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
        return sha?.isEmpty == false ? sha : nil
    }

    private static func swiftcVersion() -> String? {
        let task = Process()
        task.launchPath = "/usr/bin/env"
        task.arguments = ["swiftc", "-version"]
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = pipe
        do {
            try task.run()
        } catch {
            return nil
        }
        task.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
