import Foundation
import simd

// MARK: - Top‑K Buffer

/// Fixed-capacity Top‑K heap buffer.
/// Distances (minimize) → use Max-Heap (isMinHeap = false).
/// Similarities (maximize) → use Min-Heap (isMinHeap = true).
///
/// Capacity contract:
/// - `k > 0`: retains the best `k` pushed entries under the direction and tie policy.
/// - `k == 0`: a valid empty buffer. `pushIfBetter` ignores every push (it neither traps nor grows the arrays)
///   and `size` stays 0. Callers may therefore pass `min(k, rangeCount)` for an empty range, or skip the push
///   loop; either is safe. Verified by `TopKBufferContractTests` (zero-capacity tests) and
///   `P01TopKBufferRedTests`.
/// - `k < 0`: traps in `Array(repeating:count:)` before a buffer exists (negative-capacity control in
///   `P01TopKBufferRedTests`). Callers validate `k >= 0` first.
@usableFromInline
internal struct TopKBuffer: Sendable {
    public let k: Int
    public var vals: [Float]
    public var idxs: [Int]
    @usableFromInline var size: Int
    public let isMinHeap: Bool
    /// Policy for resolving equal numeric scores and two-NaN ties.
    @usableFromInline let tieBreaker: TieBreaker

    public init(k: Int, isMinHeap: Bool, tieBreaker: TieBreaker = .smallerIndex) {
        self.k = k
        self.isMinHeap = isMinHeap
        self.tieBreaker = tieBreaker
        self.vals = Array(repeating: 0, count: k)
        self.idxs = Array(repeating: -1, count: k)
        self.size = 0
    }

    @usableFromInline @inline(__always)
    internal func isWorse(v1: Float, i1: Int, v2: Float, i2: Int) -> Bool {
        TopKSelection.orderedBefore(
            (i2, v2), (i1, v1), descending: isMinHeap, tieBreaker: tieBreaker
        )
    }

    @inlinable
    public mutating func pushIfBetter(val: Float, idx: Int) {
        // Zero capacity retains nothing: without this guard the full-buffer branch below reads vals[0] of an
        // empty array and traps.
        guard k > 0 else { return }
        if size < k {
            vals[size] = val
            idxs[size] = idx
            size += 1
            if size == k { buildHeap() }
            return
        }
        if isWorse(v1: vals[0], i1: idxs[0], v2: val, i2: idx) {
            vals[0] = val
            idxs[0] = idx
            heapifyDown(from: 0)
        }
    }

    /// Feeds the live entries of `src` into this buffer with `pushIfBetter`, so this buffer ends up holding the
    /// best `k` of its own entries and `src`'s under its direction and tie policy. Unlike
    /// `TopKSelectionKernels.mergeTopK` it keeps this buffer's existing entries and builds no third buffer.
    ///
    /// Preconditions (traps otherwise; each pinned by a child test in `TopKBufferMergeInTrapTests`:
    /// `directionMismatchTraps`, `tiePolicyMismatchTraps` and `smallerDestinationCapacityTraps`):
    /// - `isMinHeap == src.isMinHeap`
    /// - `tieBreaker == src.tieBreaker`
    /// - `k >= src.k`
    ///
    /// Contract (not checked): the two buffers carry disjoint global indices, as for `mergeTopK`. For an exact
    /// union of two exact partial top-lists the caller sizes the destination at `min(K, a.k + b.k)`, where `K` is
    /// the global k and `a`, `b` are the partials' capacities; a smaller destination keeps only its best `k`.
    ///
    /// Allocation: `src` is a by-value parameter and is never mutated. Mutating `self` copies its arrays first
    /// if they are shared with another value (copy-on-write), so no zero-allocation claim is made here. `src`
    /// stays unchanged and independent of `self` (`TopKBufferContractTests` alias tests). Merging a buffer with a
    /// copy of itself violates the disjointness contract and duplicates indices.
    ///
    /// - Complexity: O(src.size * log k).
    @inlinable
    public mutating func mergeIn(from src: TopKBuffer) {
        precondition(isMinHeap == src.isMinHeap, "Heap types must match")
        precondition(tieBreaker == src.tieBreaker, "Tie policies must match")
        precondition(k >= src.k, "Destination capacity too small")
        for i in 0..<src.size { pushIfBetter(val: src.vals[i], idx: src.idxs[i]) }
    }

    @inlinable
    mutating func buildHeap() {
        guard size > 1 else { return }
        let lastParent = (size / 2) - 1
        if lastParent >= 0 {
            for i in stride(from: lastParent, through: 0, by: -1) { heapifyDown(from: i) }
        }
    }

    @inlinable
    mutating func heapifyDown(from index: Int) {
        var current = index
        while true {
            let left = 2 * current + 1
            let right = 2 * current + 2
            var worst = current
            if left < size && isWorse(v1: vals[left], i1: idxs[left], v2: vals[worst], i2: idxs[worst]) { worst = left }
            if right < size && isWorse(v1: vals[right], i1: idxs[right], v2: vals[worst], i2: idxs[worst]) { worst = right }
            if worst == current { break }
            vals.swapAt(current, worst)
            idxs.swapAt(current, worst)
            current = worst
        }
    }
}

// MARK: - Kernels

@usableFromInline
internal enum TopKSelectionKernels {

    // Contract: partials carry disjoint global indices and use the same policy.
    // Index-policy merge order is exercised by
    // TopKNaNContractTests.disjointMergesKeepGlobalIndicesAndDestinationPolicy.
    public static func mergeTopK(_ a: TopKBuffer, _ b: TopKBuffer, into out: inout TopKBuffer) {
        precondition(a.isMinHeap == b.isMinHeap && a.isMinHeap == out.isMinHeap, "Heap types must match")
        precondition(a.tieBreaker == b.tieBreaker && a.tieBreaker == out.tieBreaker, "Tie policies must match")
        precondition(out.k >= a.k && out.k >= b.k, "Output capacity too small")
        var merged = TopKBuffer(k: out.k, isMinHeap: out.isMinHeap, tieBreaker: out.tieBreaker)
        for i in 0..<a.size { merged.pushIfBetter(val: a.vals[i], idx: a.idxs[i]) }
        for i in 0..<b.size { merged.pushIfBetter(val: b.vals[i], idx: b.idxs[i]) }
        out = merged
    }

    // Generic range template
    @usableFromInline @inline(__always)
    static func range_topk_template<V>(
        query: V,
        candidates: [V],
        range: Range<Int>,
        k: Int,
        localTopK: inout TopKBuffer,
        compute: (V, V) -> Float
    ) {
        precondition(localTopK.k == k)
        for i in range {
            let score = compute(query, candidates[i])
            localTopK.pushIfBetter(val: score, idx: i)
        }
    }

    // Euclid2 (distance squared) — distances (minimize) → Max-Heap
    @inlinable
    public static func range_topk_euclid2_512(query: Vector512Optimized, candidates: [Vector512Optimized], range: Range<Int>, k: Int, localTopK: inout TopKBuffer) {
        precondition(!localTopK.isMinHeap, "Euclid2 uses Max-Heap")
        range_topk_template(query: query, candidates: candidates, range: range, k: k, localTopK: &localTopK) {
            EuclideanKernels.squared512($0, $1)
        }
    }

    @inlinable
    public static func range_topk_euclid2_768(query: Vector768Optimized, candidates: [Vector768Optimized], range: Range<Int>, k: Int, localTopK: inout TopKBuffer) {
        precondition(!localTopK.isMinHeap, "Euclid2 uses Max-Heap")
        range_topk_template(query: query, candidates: candidates, range: range, k: k, localTopK: &localTopK) {
            EuclideanKernels.squared768($0, $1)
        }
    }

    @inlinable
    public static func range_topk_euclid2_1536(query: Vector1536Optimized, candidates: [Vector1536Optimized], range: Range<Int>, k: Int, localTopK: inout TopKBuffer) {
        precondition(!localTopK.isMinHeap, "Euclid2 uses Max-Heap")
        range_topk_template(query: query, candidates: candidates, range: range, k: k, localTopK: &localTopK) {
            EuclideanKernels.squared1536($0, $1)
        }
    }

    // Euclid (sqrt wrapper)
    @inlinable
    public static func range_topk_euclid_512(query: Vector512Optimized, candidates: [Vector512Optimized], range: Range<Int>, k: Int, localTopK: inout TopKBuffer) {
        precondition(!localTopK.isMinHeap, "Euclid uses Max-Heap")
        range_topk_template(query: query, candidates: candidates, range: range, k: k, localTopK: &localTopK) {
            sqrt(EuclideanKernels.squared512($0, $1))
        }
    }

    // Dot (maximize) — similarities → Min-Heap
    @inlinable
    public static func range_topk_dot_512(query: Vector512Optimized, candidates: [Vector512Optimized], range: Range<Int>, k: Int, localTopK: inout TopKBuffer) {
        precondition(localTopK.isMinHeap, "Dot uses Min-Heap")
        range_topk_template(query: query, candidates: candidates, range: range, k: k, localTopK: &localTopK) {
            DotKernels.dot512($0, $1)
        }
    }

    // Cosine pre-normalized (distance = 1 − dot) — distances → Max-Heap
    @inlinable
    public static func range_topk_cosine_preNorm_512(query: Vector512Optimized, candidates: [Vector512Optimized], range: Range<Int>, k: Int, localTopK: inout TopKBuffer) {
        precondition(!localTopK.isMinHeap, "Cosine distance uses Max-Heap")
        range_topk_template(query: query, candidates: candidates, range: range, k: k, localTopK: &localTopK) {
            let sim = DotKernels.dot512($0, $1)
            return sim.isFinite ? 1.0 - max(-1.0, min(1.0, sim)) : .nan
        }
    }

    // Cosine fused — distances → Max-Heap. `CosineKernels.fused`'s arithmetic with the query's squared norm computed once
    // (`CosineKernels.fusedCandidate`), the arithmetic of the `CosineDistance` witness and of the other cosine search
    // routes on this type, so a candidate's score does not depend on k (T14 phase 3; MetricSearchWitnessConsistencyTests).
    @inlinable
    public static func range_topk_cosine_fused_512(query: Vector512Optimized, candidates: [Vector512Optimized], range: Range<Int>, k: Int, localTopK: inout TopKBuffer) {
        precondition(!localTopK.isMinHeap, "Cosine distance uses Max-Heap")
        let q = query.storage
        let sumAA = CosineKernels.fusedSquaredNorm(q, laneCount: 128)
        for i in range {
            let d = CosineKernels.fusedCandidate(query: q, queryNorm: sumAA, candidates[i].storage, laneCount: 128)
            localTopK.pushIfBetter(val: d, idx: i)
        }
    }
}
