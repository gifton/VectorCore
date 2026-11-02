import Foundation
#if canImport(Darwin)
import Darwin
#endif

enum Clock {
    static func now() -> UInt64 { // ns
        #if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
        if #available(macOS 13.0, iOS 16.0, tvOS 16.0, watchOS 9.0, *) {
            return clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        } else {
            let t = CFAbsoluteTimeGetCurrent()
            return UInt64(t * 1_000_000_000)
        }
        #else
        let t = CFAbsoluteTimeGetCurrent()
        return UInt64(t * 1_000_000_000)
        #endif
    }
}

/// Resolution of the benchmark clock, derived (not measured) from the timebase, or from the Double ULP where that is coarser
/// (`testTickFormulaWithInjectedTimebaseValues`, `testAbsoluteTimeTickFromTheDoubleULP`).
///
/// `Clock.now()` is `clock_gettime_nsec_np(CLOCK_UPTIME_RAW)` on macOS, iOS, tvOS and watchOS. Its tick is `mach_timebase_info.numer /
/// denom` ns rounded up to a whole nanosecond (`ticks`); `testTickFormulaWithInjectedTimebaseValues` pins the formula with injected
/// values and checks that the running host's tick equals it.
///
/// That CLOCK_UPTIME_RAW readings are mach absolute time converted to nanoseconds, so that consecutive readings differ by multiples of
/// that tick, and that the Apple Silicon timebase is 125/3 (about 41.7 ns) are cited platform facts, not verified by a test here.
///
/// On the other platforms `Clock.now()` reads `CFAbsoluteTimeGetCurrent()`, returned as `UInt64(t * 1e9)`: Double seconds since 2001
/// are quantized to their ULP (about 8.1e8 s gives 2^-23 s, about 119 ns), and the nanosecond product is quantized a second time to
/// its own ULP (128 ns at 8.1e17, 512 ns at 3e18). There the tick is `max(timebase tick, ceil(ULP of the seconds in ns), ceil(ULP of
/// seconds x 1e9))`, cited and derived rather than measured, computed once at startup (`absoluteTimeTick`; values pinned by
/// `testAbsoluteTimeTickFromTheDoubleULP`); that branch is compiled only off macOS/iOS/tvOS/watchOS and is unexercised on the hosts
/// this repository tests, while the formula is covered by the pure-function test on every platform.
///
/// When `mach_timebase_info` fails, or off Darwin, a conservative 1000 ns (`fallbackNanoseconds`) is used and `source` says so.
/// This is the cited source; it is not a probe of the running host.
enum ClockResolution {
    static let fallbackNanoseconds: UInt64 = 1_000

    /// Ceiling of `numer / denom` nanoseconds, or nil for a degenerate timebase. Pure, so it is testable with injected values.
    static func ticks(numer: UInt32, denom: UInt32) -> UInt64? {
        guard numer != 0, denom != 0 else { return nil }
        return (UInt64(numer) + UInt64(denom) - 1) / UInt64(denom)
    }

    /// Tick of a clock that reports `CFAbsoluteTime` (Double seconds) as `UInt64(seconds * 1e9)`: the coarser of two Double ULP grids,
    /// each rounded up to whole nanoseconds: the ULP of `seconds` (converted to ns) and the ULP of the product `seconds * 1e9`. Never
    /// below `timebaseTick`; a non-finite or implausibly large (>= 1e18 ns) grid gives `max(timebaseTick, fallbackNanoseconds)`. Pure and
    /// platform independent so it can be tested with injected epochs (`testAbsoluteTimeTickFromTheDoubleULP`).
    static func absoluteTimeTick(at seconds: Double, timebaseTick: UInt64) -> UInt64 {
        // Two quantization grids: the Double seconds themselves, and the product `UInt64(seconds * 1e9)` that `Clock.now()` returns,
        // which is rounded a second time to the ULP of the nanosecond-scale Double. The smallest nonzero difference between two
        // readings is the coarser of the two (and never finer than the timebase tick). Values pinned by
        // `testAbsoluteTimeTickFromTheDoubleULP`.
        let secondsGrid = (seconds.ulp * 1e9).rounded(.up)
        let nanosecondsGrid = (seconds * 1e9).ulp.rounded(.up)
        guard secondsGrid.isFinite, nanosecondsGrid.isFinite, secondsGrid >= 0, nanosecondsGrid >= 0,
              secondsGrid < 1e18, nanosecondsGrid < 1e18 else { return max(timebaseTick, fallbackNanoseconds) }
        return max(timebaseTick, UInt64(secondsGrid), UInt64(nanosecondsGrid))
    }

    private static let resolved: (nanoseconds: UInt64, source: String) = {
        #if canImport(Darwin)
        var info = mach_timebase_info_data_t()
        if mach_timebase_info(&info) == KERN_SUCCESS, let tick = ticks(numer: info.numer, denom: info.denom) {
            #if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
            return (tick, "mach_timebase_info \(info.numer)/\(info.denom)")
            #else
            return (absoluteTimeTick(at: CFAbsoluteTimeGetCurrent(), timebaseTick: tick),
                    "CFAbsoluteTime ULP at startup (not below mach_timebase_info \(info.numer)/\(info.denom))")
            #endif
        }
        #endif
        return (fallbackNanoseconds, "fallback \(fallbackNanoseconds) ns (no usable timebase)")
    }()

    static var tickNanoseconds: UInt64 { resolved.nanoseconds }
    static var source: String { resolved.source }
}
