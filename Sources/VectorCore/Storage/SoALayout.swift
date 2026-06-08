//
//  SoALayout.swift
//  VectorCore
//
//  The frozen, machine-readable description of the SoA memory layout — the single
//  source of truth that downstream GPU kernels derive their indexing constants from,
//  instead of hardcoding magic numbers that silently break if the layout changes.
//
//  See Docs/SoA_Layout_Contract.md for the full human-readable contract.
//

import Foundation
import simd

/// Frozen description of the ``SoA`` in-memory layout.
///
/// `SoA` stores candidate vectors **lane-major, then candidate index**: the element for
/// (lane `ℓ`, candidate `j`) lives at `buffer[ℓ * count + j]`, where each element is a
/// `SIMD4<Float>` packing four contiguous dimensions. This descriptor exposes that layout
/// as data so a consumer — most importantly a VectorAccelerate Metal shader indexing a
/// zero-copy `MTLBuffer` built from ``SoA/pageAlignedBytes`` — derives its constants from
/// one authoritative place rather than re-deriving (and risking drift from) the formula.
///
/// The layout is **stable as of VectorCore 0.3.0**. Any change to the formula, the element
/// type, or the stride is a breaking change to this contract; see `Docs/SoA_Layout_Contract.md`.
///
/// ### Worked example (512-dim, N candidates)
/// `lanes = 128`, `elementStrideBytes = 16`, `laneStrideBytes = N * 16`,
/// `logicalByteCount = 128 * N * 16`. Candidate `j`'s lane `ℓ` is at element index
/// `ℓ * N + j` — equivalently byte offset `(ℓ * N + j) * 16`.
public struct SoALayout: Sendable, Equatable {

    /// Number of `SIMD4<Float>` lanes per vector, `= dimension / 4`.
    ///
    /// Always exact: every ``SoACompatible`` dimension (384, 512, 768, 1536) is divisible
    /// by 4, so there are **no partial / tail lanes** — every lane is a full `SIMD4<Float>`.
    public let lanes: Int

    /// Number of candidate vectors stored — the true `N`.
    ///
    /// This is **not padded**: it is exactly the stride, in elements, between one lane and
    /// the next. There is no candidate-axis block padding regardless of how the `SoA` was
    /// built. (The page-aligned allocation rounds the *whole buffer's* length up to a page;
    /// see ``allocatedByteCount`` — that rounding never changes this value or the stride.)
    public let count: Int

    /// Total bytes the allocation occupies.
    ///
    /// For a page-aligned `SoA` this is ``logicalByteCount`` rounded **up** to a whole page
    /// (the length to pass to `MTLDevice.makeBuffer(bytesNoCopy:)`); for a non-page-aligned
    /// `SoA` it equals ``logicalByteCount``. Always `>= logicalByteCount`; the trailing
    /// `allocatedByteCount - logicalByteCount` bytes are zero-filled slack.
    ///
    /// - Important: **Do not derive ``count`` from this value.** It is page-rounded and
    ///   larger than the logical data; the valid indexed region is `[0, lanes * count)`
    ///   elements only.
    public let allocatedByteCount: Int

    /// Size of one packed element (`SIMD4<Float>`) in bytes — `16`.
    @inlinable
    public static var elementStrideBytes: Int { MemoryLayout<SIMD4<Float>>.stride }

    /// Stride between consecutive lanes, in bytes: `count * elementStrideBytes`.
    ///
    /// A consumer steps from lane `ℓ` to lane `ℓ + 1` by exactly this many bytes.
    @inlinable
    public var laneStrideBytes: Int { count * Self.elementStrideBytes }

    /// Bytes occupied by the logical data: `lanes * count * elementStrideBytes`.
    ///
    /// The valid indexed region is exactly `[0, lanes * count)` elements / `[0, logicalByteCount)`
    /// bytes. Bytes in `[logicalByteCount, allocatedByteCount)` are zero-filled padding.
    @inlinable
    public var logicalByteCount: Int { lanes * count * Self.elementStrideBytes }

    /// Total number of logical `SIMD4<Float>` elements: `lanes * count`.
    @inlinable
    public var elementCount: Int { lanes * count }

    /// Creates a descriptor from its stored fields. Prefer ``forType(_:count:pageAligned:)``
    /// to derive a layout for a vector type, or ``SoA/layoutDescriptor`` for a live instance.
    @inlinable
    public init(lanes: Int, count: Int, allocatedByteCount: Int) {
        self.lanes = lanes
        self.count = count
        self.allocatedByteCount = allocatedByteCount
    }

    /// The frozen index formula: returns the `SIMD4<Float>`-element offset of the element
    /// for `(lane, candidate)`, i.e. `lane * count + candidate`.
    ///
    /// Multiply by ``elementStrideBytes`` for a byte offset. Bounds-checked in debug builds.
    @inlinable
    public func elementIndex(lane: Int, candidate: Int) -> Int {
        precondition(lane >= 0 && lane < lanes, "lane \(lane) out of range [0, \(lanes))")
        precondition(candidate >= 0 && candidate < count, "candidate \(candidate) out of range [0, \(count))")
        return lane * count + candidate
    }

    /// Derives the layout for a ``SoACompatible`` vector type and candidate count **without
    /// allocating** — for a consumer that needs to size a buffer or precompute shader
    /// constants ahead of receiving the pointer.
    ///
    /// - Parameters:
    ///   - type: The fixed-dimension optimized vector type (e.g. `Vector768Optimized`).
    ///   - count: The number of candidates `N`.
    ///   - pageAligned: Pass `true` to match a `SoA` built with `pageAligned: true` —
    ///     ``allocatedByteCount`` is then page-rounded to the `makeBuffer(bytesNoCopy:)`
    ///     length. Defaults to `false` (the exact logical size), matching the `pageAligned`
    ///     default on `SoA.build`/`init` so page-alignment is **opt-in everywhere**. A GPU
    ///     consumer sizing a zero-copy buffer must pass `true` (just as it calls
    ///     `build(from:pageAligned: true)`).
    /// - Returns: A descriptor whose ``allocatedByteCount`` matches what the corresponding
    ///   `SoA` reports for the same `pageAligned` value.
    public static func forType<V: SoACompatible>(
        _ type: V.Type, count: Int, pageAligned: Bool = false
    ) -> SoALayout {
        let lanes = V.lanes
        let logical = lanes * count * elementStrideBytes
        let allocated = (pageAligned && count > 0)
            ? PlatformConfiguration.roundUpToPage(logical)
            : logical
        return SoALayout(lanes: lanes, count: count, allocatedByteCount: allocated)
    }
}
