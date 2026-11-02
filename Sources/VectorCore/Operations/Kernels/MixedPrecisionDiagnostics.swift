// Sources/VectorCore/Operations/Kernels/MixedPrecisionDiagnostics.swift

import Foundation
import OSLog

// MARK: - Overflow Tracking & Diagnostics

/// Diagnostics for FP32 inputs and their rounded FP16 representation.
/// Mutable recording state is protected by one lock; snapshots and batch commits
/// share that lock. The unchecked Sendable conformance relies on this discipline.
public final class MixedPrecisionDiagnostics: @unchecked Sendable {

    // MARK: - Shared Instance

    public static let shared = MixedPrecisionDiagnostics()

    private let lock = NSLock()
    private let logger = Logger(subsystem: "com.vectorcore.mixedprecision", category: "diagnostics")

    // MARK: - Overflow Tracking

    /// Accepted events, with each cumulative counter saturating independently at
    /// Int.max. Rates widen counters to Double before arithmetic. Exceptional
    /// categories are mutually exclusive per event before cumulative saturation.
    public struct OverflowStatistics: Sendable {
        public var totalConversions: Int = 0
        /// Finite original inputs whose rounded FP16 bits are signed infinity.
        public var overflowToInfinity: Int = 0
        /// Finite nonzero inputs whose rounded bits are either signed zero.
        public var underflowToZero: Int = 0
        /// Finite inputs whose rounded bits have zero exponent and nonzero fraction.
        public var subnormalValues: Int = 0
        /// Original NaN inputs; original infinities contribute only to the total.
        public var nanValues: Int = 0
        /// Original finite-input upper bound; both bounds remain +0 until a finite
        /// input occurs. A lone signed zero preserves its sign; mixed signs of zero
        /// produce -0 as the minimum and +0 as the maximum.
        public var maxValueSeen: Float = 0
        /// Original finite-input lower bound, with the same empty/signed-zero policy.
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

    private struct State {
        var statistics = OverflowStatistics()
        var isEnabled = false
        var hasFiniteValue = false
    }
    private var state = State()

    /// Internal per-instance seam; the public recorder remains `shared`.
    internal init() {}

    /// Seeds counters only for saturation-boundary tests; uses the normal commit path.
    internal func seedCountersForTesting(_ counters: OverflowStatistics) {
        precondition([counters.totalConversions, counters.overflowToInfinity,
                      counters.underflowToZero, counters.subnormalValues, counters.nanValues].allSatisfy { $0 >= 0 })
        lock.lock()
        state.statistics = counters
        state.statistics.minValueSeen = 0
        state.statistics.maxValueSeen = 0
        state.hasFiniteValue = false
        lock.unlock()
    }

    // MARK: - Public API

    /// Atomically reset statistics and finite-range state, then enable recording.
    /// Repeated enable calls reset as well. Intended for diagnostic use.
    public func enable() {
        lock.lock()
        state = State()
        state.isEnabled = true
        lock.unlock()
        logger.info("Mixed precision diagnostics enabled")
    }

    /// Disable future accepted records while retaining the last snapshot.
    public func disable() {
        lock.lock()
        state.isEnabled = false
        lock.unlock()
        logger.info("Mixed precision diagnostics disabled")
    }

    /// Synchronized hint, not a reservation. A false hint may skip preparation even
    /// if another thread enables recording afterward.
    public var isRecording: Bool {
        lock.lock()
        defer { lock.unlock() }
        return state.isEnabled
    }

    /// Return a coherent value snapshot. Counters independently saturate at Int.max.
    public func getStatistics() -> OverflowStatistics {
        lock.lock()
        defer { lock.unlock() }
        return state.statistics
    }

    /// Atomically reset counters and finite range without changing enabled state.
    public func reset() {
        lock.lock()
        state.statistics = OverflowStatistics()
        state.hasFiniteValue = false
        lock.unlock()
        logger.info("Mixed precision diagnostics reset")
    }

    private static func saturatedAdd(_ lhs: Int, _ rhs: Int) -> Int {
        let (sum, overflow) = lhs.addingReportingOverflow(rhs)
        return overflow ? Int.max : sum
    }

    private static func lower(_ lhs: Float, _ rhs: Float) -> Float {
        if rhs < lhs || (rhs == 0 && lhs == 0 && rhs.sign == .minus) { return rhs }
        return lhs
    }

    private static func upper(_ lhs: Float, _ rhs: Float) -> Float {
        if rhs > lhs || (rhs == 0 && lhs == 0 && rhs.sign == .plus) { return rhs }
        return lhs
    }

    /// Local immutable-input reduction; it never reads shared state.
    private struct Summary {
        var statistics = OverflowStatistics()
        var hasFiniteValue = false

        mutating func add(_ input: Float, rounded bits: UInt16) {
            statistics.totalConversions = saturatedAdd(statistics.totalConversions, 1)
            if input.isFinite {
                if hasFiniteValue {
                    statistics.minValueSeen = lower(statistics.minValueSeen, input)
                    statistics.maxValueSeen = upper(statistics.maxValueSeen, input)
                } else {
                    statistics.minValueSeen = input
                    statistics.maxValueSeen = input
                    hasFiniteValue = true
                }
            }
            let magnitude = bits & 0x7fff
            if input.isNaN {
                statistics.nanValues = saturatedAdd(statistics.nanValues, 1)
            } else if input.isFinite {
                if magnitude == 0x7c00 {
                    statistics.overflowToInfinity = saturatedAdd(statistics.overflowToInfinity, 1)
                } else if magnitude == 0 && input != 0 {
                    statistics.underflowToZero = saturatedAdd(statistics.underflowToZero, 1)
                } else if magnitude > 0 && magnitude < 0x0400 {
                    statistics.subnormalValues = saturatedAdd(statistics.subnormalValues, 1)
                }
            }
        }
    }

    /// Commit a whole summary only if enabled under the lock. Preparation begun
    /// before reset may join the new generation; there is no start-time epoch.
    /// Logging uses captured summary values after unlocking.
    private func commit(_ summary: Summary) {
        lock.lock()
        guard state.isEnabled else { lock.unlock(); return }
        let incoming = summary.statistics
        state.statistics.totalConversions = Self.saturatedAdd(state.statistics.totalConversions, incoming.totalConversions)
        state.statistics.overflowToInfinity = Self.saturatedAdd(state.statistics.overflowToInfinity, incoming.overflowToInfinity)
        state.statistics.underflowToZero = Self.saturatedAdd(state.statistics.underflowToZero, incoming.underflowToZero)
        state.statistics.subnormalValues = Self.saturatedAdd(state.statistics.subnormalValues, incoming.subnormalValues)
        state.statistics.nanValues = Self.saturatedAdd(state.statistics.nanValues, incoming.nanValues)
        if summary.hasFiniteValue {
            if state.hasFiniteValue {
                state.statistics.minValueSeen = Self.lower(state.statistics.minValueSeen, incoming.minValueSeen)
                state.statistics.maxValueSeen = Self.upper(state.statistics.maxValueSeen, incoming.maxValueSeen)
            } else {
                state.statistics.minValueSeen = incoming.minValueSeen
                state.statistics.maxValueSeen = incoming.maxValueSeen
                state.hasFiniteValue = true
            }
        }
        lock.unlock()
        if incoming.overflowToInfinity > 0 || incoming.underflowToZero > 0 {
            logger.debug("FP16 batch: \(incoming.totalConversions) events, \(incoming.overflowToInfinity) rounded overflows, \(incoming.underflowToZero) rounded underflows")
        }
    }

    /// Record using the same scalar round-to-nearest-even contract as conversion.
    internal func recordConversion(_ value: Float) {
        guard isRecording else { return }
        var summary = Summary()
        summary.add(value, rounded: MixedPrecisionKernels.fp32ToFp16_scalar(value))
        commit(summary)
    }

    /// An accepted batch commits all events atomically relative to toggles/reset.
    /// Input infinities are not arithmetic overflows; range includes finite inputs only.
    internal func recordBatchConversion(_ values: [Float]) {
        guard isRecording else { return }
        var summary = Summary()
        for value in values {
            summary.add(value, rounded: MixedPrecisionKernels.fp32ToFp16_scalar(value))
        }
        commit(summary)
    }

    /// Constructor recording classifies already-produced bits without reconversion.
    internal func recordBatchConversion(
        _ values: ContiguousArray<SIMD4<Float>>, rounded: ContiguousArray<UInt16>
    ) {
        guard isRecording else { return }
        precondition(rounded.count == values.count * 4, "Diagnostic input/output shape mismatch")
        var summary = Summary()
        for lane in values.indices {
            for element in 0..<4 {
                summary.add(values[lane][element], rounded: rounded[lane * 4 + element])
            }
        }
        commit(summary)
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
