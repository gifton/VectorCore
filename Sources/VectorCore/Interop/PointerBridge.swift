// PointerBridge.swift — helpers to safely bridge Swift buffers to C pointers
// These utilities centralize alignment checks and memory rebinding for AoS/SoA layouts.

import Foundation

@inline(__always)
func debugAssertAligned(_ ptr: UnsafeRawPointer, _ alignment: Int) {
    assert(Int(bitPattern: ptr) % alignment == 0, "Pointer not aligned to \(alignment) bytes")
}

// Bridge ContiguousArray<SIMD4<Float>> to UnsafePointer<Float> with alignment checks.
@inline(__always)
func withSIMD4FloatPointer<R>(_ array: ContiguousArray<SIMD4<Float>>, _ body: (UnsafePointer<Float>, Int) -> R) -> R {
    return array.withUnsafeBufferPointer { buf in
        guard let base = buf.baseAddress else {
            // Empty array: pass a non-null dummy pointer and zero count; body must ignore pointer when count is zero.
            return body(UnsafePointer<Float>(bitPattern: 0x1)!, 0)
        }
        let raw = UnsafeRawPointer(base)
        #if DEBUG
        debugAssertAligned(raw, MemoryLayout<SIMD4<Float>>.alignment)
        #endif
        // Rebind SIMD4<Float> storage to Float for the duration of the call.
        return base.withMemoryRebound(to: Float.self, capacity: buf.count * 4) { floatPtr in
            body(floatPtr, buf.count * 4)
        }
    }
}

// Bridge ContiguousArray<SIMD4<Int8>> to UnsafePointer<Int8> for C kernel interop.
// SIMD4<Int8> is 4 bytes with stride 4, so contiguous storage is a flat int8 array.
@inline(__always)
func withSIMD4Int8Pointer<R>(_ array: ContiguousArray<SIMD4<Int8>>, _ body: (UnsafePointer<Int8>, Int) -> R) -> R {
    assert(MemoryLayout<SIMD4<Int8>>.stride == 4, "SIMD4<Int8> stride must be 4 for safe rebinding")
    return array.withUnsafeBufferPointer { buf in
        guard let base = buf.baseAddress else {
            return body(UnsafePointer<Int8>(bitPattern: 0x1)!, 0)
        }
        return base.withMemoryRebound(to: Int8.self, capacity: buf.count * 4) { int8Ptr in
            body(int8Ptr, buf.count * 4)
        }
    }
}

// Bridge ContiguousArray<Int8> to UnsafePointer<Int8> (alignment for int8 typically 1; optional 16-byte check for SIMD views).
@inline(__always)
func withInt8Pointer<R>(_ array: ContiguousArray<Int8>, _ body: (UnsafePointer<Int8>, Int) -> R) -> R {
    return array.withUnsafeBufferPointer { buf in
        guard let base = buf.baseAddress else {
            return body(UnsafePointer<Int8>(bitPattern: 0x1)!, 0)
        }
        #if DEBUG
        // If used as SIMD lanes, 16-byte alignment is beneficial but not strictly required.
        // debugAssertAligned(UnsafeRawPointer(base), 16)
        #endif
        return body(base, buf.count)
    }
}
