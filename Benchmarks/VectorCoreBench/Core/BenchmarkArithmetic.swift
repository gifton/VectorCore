import Foundation

enum BenchmarkArithmetic {
    static func nanoseconds(_ seconds: Double) throws -> UInt64 {
        let scaled = seconds * 1_000_000_000
        guard seconds.isFinite, seconds >= 0, scaled.isFinite, scaled < 18_446_744_073_709_551_616.0 else {
            throw BenchmarkConfigurationError("min-time nanoseconds must be finite, nonnegative and below 2^64")
        }
        return UInt64(scaled)
    }
    static func product<T: FixedWidthInteger>(_ a: T, _ b: T) throws -> T {
        let (value, overflow) = a.multipliedReportingOverflow(by: b)
        guard !overflow else { throw BenchmarkConfigurationError("Benchmark product overflow") }
        return value
    }
    static func sum<T: FixedWidthInteger>(_ a: T, _ b: T) throws -> T {
        let (value, overflow) = a.addingReportingOverflow(b)
        guard !overflow else { throw BenchmarkConfigurationError("Benchmark sum overflow") }
        return value
    }
    static func duration(_ start: UInt64, _ end: UInt64, requirePositive: Bool = true) throws -> UInt64 {
        guard end >= start, !requirePositive || end > start else { throw BenchmarkConfigurationError("Unusable benchmark clock interval") }
        return end - start
    }
    static func count(_ value: UInt64) throws -> Int {
        guard let count = Int(exactly: value) else { throw BenchmarkConfigurationError("Benchmark iteration count overflow") }
        return count
    }
}
