// Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift

import Foundation
import OSLog

// MARK: - Overflow Tracking & Diagnostics

/// Diagnostic utilities for tracking FP16 conversion overflow and precision issues
public final class MixedPrecisionDiagnostics: @unchecked Sendable {

    // MARK: - Shared Instance

    public static let shared = MixedPrecisionDiagnostics()

    private let lock = NSLock()
    private let logger = Logger(subsystem: "com.vectorcore.mixedprecision", category: "diagnostics")

    // MARK: - Overflow Tracking

    public struct OverflowStatistics: Sendable {
        public var totalConversions: Int = 0
        public var overflowToInfinity: Int = 0
        public var underflowToZero: Int = 0
        public var subnormalValues: Int = 0
        public var nanValues: Int = 0
        public var maxValueSeen: Float = 0
        public var minValueSeen: Float = 0

        public var overflowRate: Double {
            guard totalConversions > 0 else { return 0 }
            return Double(overflowToInfinity) / Double(totalConversions)
        }

        public var underflowRate: Double {
            guard totalConversions > 0 else { return 0 }
            return Double(underflowToZero) / Double(totalConversions)
        }

        public var summary: String {
            """
            FP16 Conversion Statistics:
            ---------------------------
            Total conversions: \(totalConversions)
            Overflows (→∞):    \(overflowToInfinity) (\(String(format: "%.4f%%", overflowRate * 100)))
            Underflows (→0):   \(underflowToZero) (\(String(format: "%.4f%%", underflowRate * 100)))
            Subnormals:        \(subnormalValues)
            NaN values:        \(nanValues)
            Value range:       [\(String(format: "%.2f", minValueSeen)), \(String(format: "%.2f", maxValueSeen))]
            """
        }
    }

    private var statistics = OverflowStatistics()
    private var isEnabled = false

    private init() {}

    // MARK: - Public API

    /// Enable overflow tracking (adds performance overhead, use for debugging only)
    public func enable() {
        lock.lock()
        defer { lock.unlock() }
        isEnabled = true
        statistics = OverflowStatistics()
        logger.info("Mixed precision diagnostics enabled")
    }

    /// Disable overflow tracking
    public func disable() {
        lock.lock()
        defer { lock.unlock() }
        isEnabled = false
        logger.info("Mixed precision diagnostics disabled")
    }

    /// Get current statistics
    public func getStatistics() -> OverflowStatistics {
        lock.lock()
        defer { lock.unlock() }
        return statistics
    }

    /// Reset statistics
    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        statistics = OverflowStatistics()
        logger.info("Mixed precision diagnostics reset")
    }

    /// Record a conversion event
    @inline(__always)
    internal func recordConversion(_ value: Float) {
        guard isEnabled else { return }

        lock.lock()
        defer { lock.unlock() }

        statistics.totalConversions += 1

        // Update min/max
        if value.isFinite {
            statistics.maxValueSeen = max(statistics.maxValueSeen, value)
            statistics.minValueSeen = min(statistics.minValueSeen, value)
        }

        // Check for overflow/underflow
        let fp16Max: Float = 65504.0
        let fp16Min: Float = -65504.0
        let fp16MinNormal: Float = 6.10352e-5

        if value.isNaN {
            statistics.nanValues += 1
        } else if value.isInfinite {
            statistics.overflowToInfinity += 1
            logger.debug("FP16 overflow: value \(value) → infinity")
        } else if value > fp16Max {
            statistics.overflowToInfinity += 1
            logger.warning("FP16 overflow: value \(value) exceeds max (\(fp16Max)) → clamped to infinity")
        } else if value < fp16Min {
            statistics.overflowToInfinity += 1
            logger.warning("FP16 overflow: value \(value) below min (\(fp16Min)) → clamped to -infinity")
        } else if abs(value) < fp16MinNormal && value != 0 {
            statistics.subnormalValues += 1
            if statistics.subnormalValues % 1000 == 0 {  // Log every 1000th to avoid spam
                logger.debug("FP16 subnormal: value \(value) in subnormal range")
            }
        } else if abs(value) < 1e-10 && value != 0 {
            statistics.underflowToZero += 1
            logger.debug("FP16 underflow: value \(value) → flushed to zero")
        }
    }

    /// Record batch conversion events (optimized for performance)
    @inline(__always)
    internal func recordBatchConversion(_ values: [Float]) {
        guard isEnabled else { return }

        lock.lock()
        defer { lock.unlock() }

        let fp16Max: Float = 65504.0
        let fp16Min: Float = -65504.0
        let fp16MinNormal: Float = 6.10352e-5

        var batchOverflows = 0
        var batchUnderflows = 0
        var batchSubnormals = 0
        var batchNaNs = 0

        for value in values {
            statistics.totalConversions += 1

            if value.isFinite {
                statistics.maxValueSeen = max(statistics.maxValueSeen, value)
                statistics.minValueSeen = min(statistics.minValueSeen, value)
            }

            if value.isNaN {
                batchNaNs += 1
            } else if value.isInfinite || value > fp16Max || value < fp16Min {
                batchOverflows += 1
            } else if abs(value) < fp16MinNormal && value != 0 {
                batchSubnormals += 1
            } else if abs(value) < 1e-10 && value != 0 {
                batchUnderflows += 1
            }
        }

        statistics.overflowToInfinity += batchOverflows
        statistics.underflowToZero += batchUnderflows
        statistics.subnormalValues += batchSubnormals
        statistics.nanValues += batchNaNs

        if batchOverflows > 0 {
            logger.warning("Batch conversion: \(batchOverflows) overflows detected in \(values.count) values")
        }
        if batchUnderflows > 0 {
            logger.debug("Batch conversion: \(batchUnderflows) underflows detected in \(values.count) values")
        }
    }

    // MARK: - Hardware Detection

    public struct HardwareCapabilities: Sendable {
        public let processorName: String
        public let processorCoreCount: Int
        public let hasFP16Support: Bool
        public let hasNEON: Bool
        public let hasSIMD128: Bool
        public let cacheLineSize: Int
        public let l1CacheSize: Int  // bytes
        public let l2CacheSize: Int  // bytes

        public var summary: String {
            """
            Hardware Capabilities:
            ----------------------
            Processor: \(processorName) (\(processorCoreCount) cores)
            FP16 Support: \(hasFP16Support ? "Yes" : "No")
            NEON/SIMD: \(hasNEON ? "Yes" : "No")
            SIMD128: \(hasSIMD128 ? "Yes" : "No")
            Cache Line: \(cacheLineSize) bytes
            L1 Cache: \(l1CacheSize / 1024) KB
            L2 Cache: \(l2CacheSize / 1024) KB
            """
        }
    }

    /// Detect hardware capabilities
    public func detectHardwareCapabilities() -> HardwareCapabilities {
        #if arch(arm64)
        let hasFP16 = true
        let hasNEON = true
        let hasSIMD128 = true
        #elseif arch(x86_64)
        let hasFP16 = checkF16CSupport()
        let hasNEON = false
        let hasSIMD128 = true
        #else
        let hasFP16 = false
        let hasNEON = false
        let hasSIMD128 = false
        #endif

        return HardwareCapabilities(
            processorName: getProcessorName(),
            processorCoreCount: ProcessInfo.processInfo.processorCount,
            hasFP16Support: hasFP16,
            hasNEON: hasNEON,
            hasSIMD128: hasSIMD128,
            cacheLineSize: getCacheLineSize(),
            l1CacheSize: getL1CacheSize(),
            l2CacheSize: getL2CacheSize()
        )
    }

    // MARK: - Private Hardware Detection Helpers

    private func getProcessorName() -> String {
        #if os(macOS) || os(iOS)
        var size: size_t = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var brandString = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &brandString, &size, nil, 0)
        // Convert CChar array to UInt8, truncating null terminator
        let utf8Bytes = brandString.prefix(while: { $0 != 0 }).map { UInt8(bitPattern: $0) }
        return String(decoding: utf8Bytes, as: UTF8.self)
        #else
        return "Unknown"
        #endif
    }

    private func getCacheLineSize() -> Int {
        #if os(macOS) || os(iOS)
        var size: Int = 0
        var len: size_t = MemoryLayout<Int>.size
        sysctlbyname("hw.cachelinesize", &size, &len, nil, 0)
        return size > 0 ? size : 64  // Default to 64 bytes
        #else
        return 64
        #endif
    }

    private func getL1CacheSize() -> Int {
        #if os(macOS) || os(iOS)
        var size: Int = 0
        var len: size_t = MemoryLayout<Int>.size
        sysctlbyname("hw.l1dcachesize", &size, &len, nil, 0)
        return size > 0 ? size : 32 * 1024  // Default to 32KB
        #else
        return 32 * 1024
        #endif
    }

    private func getL2CacheSize() -> Int {
        #if os(macOS) || os(iOS)
        var size: Int = 0
        var len: size_t = MemoryLayout<Int>.size
        sysctlbyname("hw.l2cachesize", &size, &len, nil, 0)
        return size > 0 ? size : 256 * 1024  // Default to 256KB
        #else
        return 256 * 1024
        #endif
    }

    private func checkF16CSupport() -> Bool {
        #if arch(x86_64)
        // Check CPUID for F16C support (bit 29 of ECX from CPUID leaf 1)
        // This is a simplified check; full implementation would use assembly
        return false  // Conservative: assume not available
        #else
        return false
        #endif
    }
}

// MARK: - Performance Profiler

/// Detailed performance profiler for mixed precision operations
public final class MixedPrecisionProfiler: @unchecked Sendable {

    public static let shared = MixedPrecisionProfiler()

    private let lock = NSLock()
    private let logger = Logger(subsystem: "com.vectorcore.mixedprecision", category: "profiler")

    public struct ProfilingResult: Sendable {
        public let operationName: String
        public let dimension: Int
        public let iterations: Int
        public let totalTimeNs: Double
        public let meanTimeNs: Double
        public let medianTimeNs: Double
        public let p95TimeNs: Double
        public let p99TimeNs: Double
        public let stdDevNs: Double
        public let minTimeNs: Double
        public let maxTimeNs: Double
        public let throughputOpsPerSec: Double
        public let memoryBandwidthGBps: Double?
        public let cpuUtilization: Double?

        public var summary: String {
            let bw = memoryBandwidthGBps.map { String(format: "%.2f GB/s", $0) } ?? "N/A"
            let cpu = cpuUtilization.map { String(format: "%.1f%%", $0) } ?? "N/A"

            return """
            \(operationName) (\(dimension)D, \(iterations) iterations):
            ────────────────────────────────────────────────
              Mean:       \(String(format: "%10.2f", meanTimeNs)) ns
              Median:     \(String(format: "%10.2f", medianTimeNs)) ns
              P95:        \(String(format: "%10.2f", p95TimeNs)) ns
              P99:        \(String(format: "%10.2f", p99TimeNs)) ns
              Std Dev:    \(String(format: "%10.2f", stdDevNs)) ns
              Min:        \(String(format: "%10.2f", minTimeNs)) ns
              Max:        \(String(format: "%10.2f", maxTimeNs)) ns
              Throughput: \(String(format: "%10.2f", throughputOpsPerSec / 1_000_000)) M ops/sec
              Bandwidth:  \(bw)
              CPU:        \(cpu)
            """
        }
    }

    private init() {}

    /// Profile a mixed precision operation on actual hardware
    public func profileOperation(
        name: String,
        dimension: Int,
        iterations: Int = 1000,
        warmup: Int = 100,
        memoryBytes: Int? = nil,
        operation: () -> Void
    ) -> ProfilingResult {
        // Warmup
        for _ in 0..<warmup {
            operation()
        }

        // Collect timing samples
        var times: [Double] = []
        times.reserveCapacity(iterations)

        for _ in 0..<iterations {
            let start = mach_absolute_time()
            operation()
            let end = mach_absolute_time()
            times.append(machTimeToNanoseconds(end - start))
        }

        // Statistical analysis
        let sorted = times.sorted()
        let mean = times.reduce(0, +) / Double(times.count)
        let median = sorted[sorted.count / 2]
        let p95 = sorted[Int(Double(sorted.count) * 0.95)]
        let p99 = sorted[Int(Double(sorted.count) * 0.99)]
        let variance = times.map { pow($0 - mean, 2) }.reduce(0, +) / Double(times.count)
        let stdDev = sqrt(variance)
        let throughput = 1_000_000_000.0 / mean

        let bandwidth = memoryBytes.map { bytes in
            (Double(bytes) / mean)  // bytes per nanosecond = GB/s
        }

        let result = ProfilingResult(
            operationName: name,
            dimension: dimension,
            iterations: iterations,
            totalTimeNs: times.reduce(0, +),
            meanTimeNs: mean,
            medianTimeNs: median,
            p95TimeNs: p95,
            p99TimeNs: p99,
            stdDevNs: stdDev,
            minTimeNs: sorted.first ?? 0,
            maxTimeNs: sorted.last ?? 0,
            throughputOpsPerSec: throughput,
            memoryBandwidthGBps: bandwidth,
            cpuUtilization: nil  // Would require additional platform-specific APIs
        )

        logger.info("Profiling complete: \(name)")
        return result
    }

    private func machTimeToNanoseconds(_ time: UInt64) -> Double {
        var timebase = mach_timebase_info_data_t()
        mach_timebase_info(&timebase)
        return Double(time) * Double(timebase.numer) / Double(timebase.denom)
    }
}

// Import necessary system APIs
#if canImport(Darwin)
import Darwin.Mach
#endif
