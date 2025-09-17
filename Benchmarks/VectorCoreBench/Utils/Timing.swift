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

