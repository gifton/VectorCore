import Testing
import Foundation
@testable import VectorCore

/// Bounds-safety sweep for the array SIMD providers.
///
/// Exercises every `ArraySIMDProvider` operation (and the low-level Double-precision
/// reductions) across all SIMD-boundary sizes 1...64. The point is NOT to check numeric
/// results but to drive every chunked SIMD loop over every remainder size so that an
/// out-of-bounds read trips AddressSanitizer deterministically. This is a regression guard
/// for the off-by-one class fixed in SwiftFloatSIMDProvider (SIMD8 loop bound assumed a
/// 0-based start while the reductions start at i=1, over-reading a[count] when count % 8 == 0).
@Suite("SIMD Provider Bounds Safety")
struct SIMDProviderBoundsTests {

    private var providers: [any ArraySIMDProvider] {
        [SwiftSIMDProvider(), DefaultArraySIMDProvider()]
    }

    @Test("ArraySIMDProvider ops are in-bounds for sizes 1...64")
    func testArrayProviderBoundsAllSizes() {
        var sink: Float = 0
        for provider in providers {
            for n in 1...64 {
                // Positive, distinct values so sqrt/log are valid and max/min/index are meaningful.
                let a = (0..<n).map { Float($0) * 0.5 + 1.0 }
                let b = (0..<n).map { Float(n - $0) * 0.25 + 1.0 }

                sink += provider.add(a, b).last ?? 0
                sink += provider.subtract(a, b).last ?? 0
                sink += provider.multiply(a, by: 1.5).last ?? 0
                sink += provider.divide(a, by: 2.0).last ?? 0
                sink += provider.dot(a, b)
                sink += provider.sum(a)
                sink += provider.max(a)
                sink += provider.min(a)
                sink += provider.mean(a)
                sink += provider.magnitude(a)
                sink += provider.magnitudeSquared(a)
                sink += provider.normalize(a).last ?? 0
                sink += provider.euclideanDistanceSquared(a, b)
                sink += provider.cosineSimilarity(a, b)
                sink += provider.elementWiseMin(a, b).last ?? 0
                sink += provider.elementWiseMax(a, b).last ?? 0
                sink += provider.elementWiseMultiply(a, b).last ?? 0
                sink += provider.elementWiseDivide(a, b).last ?? 0
                sink += provider.abs(a).last ?? 0
                sink += provider.sqrt(a).last ?? 0
                sink += provider.log(a).last ?? 0
                sink += provider.clip(a, min: 2.0, max: 10.0).last ?? 0
                sink += Float(provider.minIndex(a))
                sink += Float(provider.maxIndex(a))
            }
        }
        #expect(sink.isFinite)
    }

    @Test("SwiftFloatSIMDProvider reductions are in-bounds for sizes 1...64")
    func testFloatReductionsBounds() {
        var sink: Float = 0
        for n in 1...64 {
            let a = (0..<n).map { Float($0) * 0.5 - 8.0 }  // mix of signs for maximumMagnitude
            a.withUnsafeBufferPointer { buf in
                let p = buf.baseAddress!
                sink += SwiftFloatSIMDProvider.maximum(p, count: n)
                sink += SwiftFloatSIMDProvider.minimum(p, count: n)
                sink += SwiftFloatSIMDProvider.maximumMagnitude(p, count: n)
                sink += SwiftFloatSIMDProvider.sum(p, count: n)
            }
        }
        #expect(sink.isFinite)
    }

    @Test("SwiftDoubleSIMDProvider reductions are in-bounds for sizes 1...64")
    func testDoubleReductionsBounds() {
        var sink: Double = 0
        for n in 1...64 {
            let a = (0..<n).map { Double($0) * 0.5 - 8.0 }
            a.withUnsafeBufferPointer { buf in
                let p = buf.baseAddress!
                sink += SwiftDoubleSIMDProvider.maximum(p, count: n)
                sink += SwiftDoubleSIMDProvider.minimum(p, count: n)
                sink += SwiftDoubleSIMDProvider.maximumMagnitude(p, count: n)
                sink += SwiftDoubleSIMDProvider.sum(p, count: n)
            }
        }
        #expect(sink.isFinite)
    }
}
