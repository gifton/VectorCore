// VectorCore: Batch Kernel Provider Protocol
//
// beta-evolution-4, DOCUMENT-5 (R4). A ComputeProvider that can substitute real
// batch kernels (e.g. Metal) for VectorCore's CPU kernels. `Operations` downcasts
// the installed `computeProvider` to this protocol and, when present, delegates —
// so GPU acceleration becomes transparent through VectorCore's own findNearest.
//
// VectorCore defines the protocol; the GPU-backed conformance lives in VectorAccelerate.

import Foundation

/// A `ComputeProvider` that supplies hardware batch kernels rather than only a
/// scheduling strategy. Conformers compute the kernel themselves (CPU SIMD, Metal,
/// etc.) instead of running an opaque closure.
///
/// Semantics a conformer must honor (see DOCUMENT-4 S4 conformance contract):
/// - `batchDistance` returns one distance per candidate, in candidate order.
/// - `findNearest` returns up to `k` `(index, distance)` pairs sorted ascending by
///   distance, indexing into `candidates`; results match the CPU reference within a
///   documented tolerance.
public protocol BatchKernelProvider: ComputeProvider {
    /// Distance from `query` to each candidate, in candidate order.
    func batchDistance<V: VectorProtocol>(
        query: V, candidates: [V], metric: any DistanceMetric
    ) async throws -> [Float] where V.Scalar == Float

    /// Up to `k` nearest candidates, sorted ascending by distance.
    func findNearest<V: VectorProtocol>(
        query: V, candidates: [V], k: Int, metric: any DistanceMetric
    ) async throws -> [(index: Int, distance: Float)] where V.Scalar == Float
}
