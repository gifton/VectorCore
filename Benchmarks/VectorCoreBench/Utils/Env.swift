import Foundation
#if canImport(Darwin)
import Darwin
#endif
import VectorCore

public struct BenchMetadata: Codable {
    public let package: String
    public let packageVersion: String
    public let gitSHA: String?
    public let date: String
    public let os: String
    public let arch: String
    public let cpuCores: Int
    public let deviceModel: String?
    public let swiftVersion: String?
    public let buildConfiguration: String
    public let suites: [String]
    public let dims: [Int]
}

enum EnvCapture {
    static func collectMetadata(suites: [String], dims: [Int], buildConfig: String = "release") -> BenchMetadata {
        let package = "VectorCore"
        let packageVersion = VectorCoreVersion.versionString
        let gitSHA = safeGitSHA()
        let date = ISO8601DateFormatter().string(from: Date())
        let os = ProcessInfo.processInfo.operatingSystemVersionString
        let arch = currentArch()
        let cpuCores = ProcessInfo.processInfo.activeProcessorCount
        let deviceModel = sysctlString(["machdep.cpu.brand_string", "hw.model", "hw.machine"]) // first non-nil
        let swiftVersion = swiftcVersion()
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
            suites: suites,
            dims: dims
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
