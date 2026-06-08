// VectorCore: Unified Vector Buffer — the zero-copy contract
//
// beta-evolution-4, DOCUMENT-4 (S5). Provides the CPU-only memory contract that
// lets downstream packages read VectorCore vectors — or hand them to the GPU via
// `MTLDevice.makeBuffer(bytesNoCopy:)` — without copying. VectorCore guarantees
// alignment and contiguity; it imports no GPU framework. The Metal side (and the
// ownership handoff it implies) lives entirely in VectorAccelerate.
//
// See: Docs/Memory_Alignment.md, Docs/beta-evolution-4/DOCUMENT-4_Spec_Ecosystem_Seams.md

import Foundation

// MARK: - Read contract

/// A contiguous, alignment-guaranteed read view over a vector's `Float32` elements.
///
/// This is the zero-copy *contract* between VectorCore and downstream packages
/// (VectorAccelerate, EmbedKit). A conforming value exposes its elements as a
/// contiguous, aligned byte region that a consumer can read without copying.
///
/// The protocol describes only *reading*. The base address yielded by
/// ``withUnsafeContiguousBytes(_:)`` is guaranteed aligned to ``alignment`` bytes,
/// but is **not** necessarily page-aligned — only ``PageAlignedBuffer`` guarantees
/// the page alignment that `makeBuffer(bytesNoCopy:)` requires. Consumers that need
/// the GPU-import path must check ``alignment`` (or use ``PageAlignedBuffer``
/// directly); the rest can read any conforming vector uniformly.
public protocol UnifiedVectorBuffer {
    /// Number of logical `Float32` elements in the vector.
    var elementCount: Int { get }

    /// Guaranteed alignment, in bytes, of the base address yielded by
    /// ``withUnsafeContiguousBytes(_:)``. Always a power of two ≥ `MemoryLayout<Float>.alignment`.
    var alignment: Int { get }

    /// Invoke `body` with a contiguous raw view of the logical element bytes
    /// (`elementCount * MemoryLayout<Float>.stride`).
    ///
    /// - Important: The pointer is valid only for the duration of `body`; do not
    ///   escape it. For ``PageAlignedBuffer`` the base address is page-aligned and
    ///   stable for the object's lifetime, but this scoped accessor is still the
    ///   correct way to *read*.
    func withUnsafeContiguousBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R
}

public extension UnifiedVectorBuffer {
    /// Logical size of the vector's elements, in bytes (`elementCount * stride`).
    ///
    /// For ``PageAlignedBuffer`` this is the *logical* size; the *allocated* size
    /// (rounded up to a page multiple, the length for `bytesNoCopy`) is
    /// ``PageAlignedBuffer/allocatedByteCount``.
    var byteCount: Int { elementCount * MemoryLayout<Float>.stride }
}

// MARK: - Page-aligned allocation (bytesNoCopy-ready)

/// A page-aligned `Float32` buffer suitable for zero-copy GPU import via
/// `MTLDevice.makeBuffer(bytesNoCopy:length:options:deallocator:)`.
///
/// `makeBuffer(bytesNoCopy:)` requires, on macOS, that **both** the base address and
/// the length be multiples of the OS page size. This type satisfies both: it
/// allocates page-aligned memory (via the package's `posix_memalign`-backed
/// allocator) and rounds the byte length up to a whole page, zero-filling the
/// padding so the GPU never reads uninitialized memory.
///
/// ## Ownership and zero-copy handoff
///
/// For a true zero-copy handoff, the `MTLBuffer` outlives this object and frees the
/// memory through Metal's `bytesNoCopy` deallocator. To avoid a double free, call
/// ``consumeAllocation()`` to transfer ownership: after that, this object no longer
/// frees on `deinit`, and the caller (the Metal deallocator) becomes responsible for
/// `free()`. If you instead keep this object alive for the buffer's lifetime, pass a
/// no-op deallocator on the Metal side and let `deinit` free as usual.
///
/// VectorCore performs none of the Metal calls; it only provides the page-aligned
/// memory and the ownership primitive.
public final class PageAlignedBuffer: UnifiedVectorBuffer, @unchecked Sendable {

    /// Number of logical `Float32` elements.
    public let elementCount: Int

    /// The OS page size used for alignment (captured at construction).
    public let pageSize: Int

    /// Total allocated size in bytes, rounded up to a multiple of ``pageSize``.
    /// This is the `length` to pass to `makeBuffer(bytesNoCopy:length:)`.
    public let allocatedByteCount: Int

    /// Whether this object still owns (and will free) the allocation. Becomes
    /// `false` after ``consumeAllocation()`` transfers ownership to the caller.
    public private(set) var ownsAllocation: Bool

    /// Float-bound base pointer. Non-nil while owned; the raw allocation is freed on
    /// `deinit` unless ownership was transferred.
    @usableFromInline internal var fptr: UnsafeMutablePointer<Float>?

    /// `bytesNoCopy` requires page alignment.
    public var alignment: Int { pageSize }

    /// Page-aligned base address of the allocation (the `bytesNoCopy` pointer).
    ///
    /// - Precondition: ownership has not been transferred via ``consumeAllocation()``.
    public var baseAddress: UnsafeMutableRawPointer {
        precondition(ownsAllocation, "PageAlignedBuffer: allocation was consumed")
        return UnsafeMutableRawPointer(fptr!)
    }

    /// Allocate a zero-initialized, page-aligned buffer for `elementCount` floats.
    ///
    /// - Parameter elementCount: Number of `Float32` elements (must be > 0).
    /// - Complexity: O(allocatedByteCount) for the zero-fill.
    public init(elementCount: Int) {
        precondition(elementCount > 0, "PageAlignedBuffer: elementCount must be positive")
        let page = PlatformConfiguration.pageSize
        let logical = elementCount * MemoryLayout<Float>.stride
        // Round the byte length up to a whole page (the makeBuffer(bytesNoCopy:) rule).
        let padded = ((logical + page - 1) / page) * page
        let floatCapacity = padded / MemoryLayout<Float>.stride

        let ptr: UnsafeMutablePointer<Float>
        do {
            // pageSize is a power of two ≥ minimumAlignment, so it is a valid
            // posix_memalign alignment. Reusing AlignedMemory keeps allocation and
            // deallocation (free) consistent with the rest of the package.
            ptr = try AlignedMemory.allocateAligned(type: Float.self, count: floatCapacity, alignment: page)
        } catch {
            fatalError("PageAlignedBuffer: page-aligned allocation of \(padded) bytes failed: \(error)")
        }
        // Zero logical + padding so the GPU never imports uninitialized bytes.
        ptr.initialize(repeating: 0, count: floatCapacity)

        self.elementCount = elementCount
        self.pageSize = page
        self.allocatedByteCount = padded
        self.fptr = ptr
        self.ownsAllocation = true
    }

    /// Allocate a page-aligned buffer and copy `source` into it once.
    ///
    /// The canonical EmbedKit use: write a freshly produced embedding into
    /// page-aligned memory a single time, then hand the pointer downstream.
    public convenience init(copying source: [Float]) {
        self.init(elementCount: source.count)
        source.withUnsafeBufferPointer { src in
            fptr!.update(from: src.baseAddress!, count: src.count)
        }
    }

    deinit {
        if ownsAllocation, let p = fptr {
            AlignedMemory.deallocate(p)
        }
    }

    public func withUnsafeContiguousBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        precondition(ownsAllocation, "PageAlignedBuffer: allocation was consumed")
        // Expose only the logical bytes; the page padding is implementation detail.
        return try body(UnsafeRawBufferPointer(start: fptr!, count: byteCount))
    }

    /// Mutable access to the logical `Float32` elements (e.g. to write an embedding).
    public func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        precondition(ownsAllocation, "PageAlignedBuffer: allocation was consumed")
        return try body(UnsafeMutableBufferPointer(start: fptr!, count: elementCount))
    }

    /// Transfer ownership of the page-aligned allocation to the caller.
    ///
    /// After this call, ``ownsAllocation`` is `false`, this object will **not** free
    /// the memory on `deinit`, and further access to ``baseAddress`` /
    /// `withUnsafe*` traps. The caller — typically a Metal `bytesNoCopy` deallocator —
    /// must eventually `free()` the returned base (e.g. via
    /// ``AlignedMemory/deallocate(_:)-(UnsafeMutableRawPointer)``).
    ///
    /// - Returns: The page-aligned base address and the page-rounded byte length.
    public func consumeAllocation() -> (baseAddress: UnsafeMutableRawPointer, allocatedByteCount: Int) {
        precondition(ownsAllocation, "PageAlignedBuffer: allocation already consumed")
        ownsAllocation = false
        return (UnsafeMutableRawPointer(fptr!), allocatedByteCount)
    }
}

// MARK: - Conformances for VectorCore vector types (read contract)
//
// The optimized types back their storage with `ContiguousArray<SIMD4<Float>>`, whose
// element region is 16-byte aligned — enough for the read contract, but not the page
// alignment `bytesNoCopy` needs. Consumers that require the GPU-import path must copy
// into a `PageAlignedBuffer` (e.g. via `PageAlignedBuffer(copying:)`).

extension Vector512Optimized: UnifiedVectorBuffer {
    public var elementCount: Int { 512 }
    public var alignment: Int { MemoryLayout<SIMD4<Float>>.alignment }
    public func withUnsafeContiguousBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        try storage.withUnsafeBytes(body)
    }
}

extension Vector768Optimized: UnifiedVectorBuffer {
    public var elementCount: Int { 768 }
    public var alignment: Int { MemoryLayout<SIMD4<Float>>.alignment }
    public func withUnsafeContiguousBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        try storage.withUnsafeBytes(body)
    }
}

extension Vector1536Optimized: UnifiedVectorBuffer {
    public var elementCount: Int { 1536 }
    public var alignment: Int { MemoryLayout<SIMD4<Float>>.alignment }
    public func withUnsafeContiguousBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        try storage.withUnsafeBytes(body)
    }
}

extension Vector384Optimized: UnifiedVectorBuffer {
    public var elementCount: Int { 384 }
    public var alignment: Int { MemoryLayout<SIMD4<Float>>.alignment }
    public func withUnsafeContiguousBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        try storage.withUnsafeBytes(body)
    }
}

extension DynamicVector: UnifiedVectorBuffer {
    public var elementCount: Int { scalarCount }
    // HybridStorage may store small vectors inline; only Float alignment is
    // guaranteed across the inline and heap paths.
    public var alignment: Int { MemoryLayout<Float>.alignment }
    public func withUnsafeContiguousBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        try withUnsafeBufferPointer { buffer in
            try body(UnsafeRawBufferPointer(buffer))
        }
    }
}
