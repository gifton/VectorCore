import Foundation
import simd

// MARK: - Top‑K Buffer

/// Fixed-capacity Top‑K heap buffer.
/// Distances (minimize) → use Max-Heap (isMinHeap = false).
/// Similarities (maximize) → use Min-Heap (isMinHeap = true).
internal struct TopKBuffer: Sendable {
    public let k: Int
    public var vals: [Float]
    public var idxs: [Int]
    @usableFromInline var size: Int
    public let isMinHeap: Bool

    public init(k: Int, isMinHeap: Bool) {
        self.k = k
        self.isMinHeap = isMinHeap
        self.vals = Array(repeating: 0, count: k)
        self.idxs = Array(repeating: -1, count: k)
        self.size = 0
    }

    @usableFromInline @inline(__always)
    internal func isWorse(v1: Float, i1: Int, v2: Float, i2: Int) -> Bool {
        if v1 != v2 {
            return isMinHeap ? (v1 < v2) : (v1 > v2)
        }
        // Deterministic tiebreaker: prefer smaller index → larger index is worse
        return i1 > i2
    }

    @inlinable
    public mutating func pushIfBetter(val: Float, idx: Int) {
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

internal enum TopKSelectionKernels {

    // Merge two partial Top‑Ks into out (deterministic)
    public static func mergeTopK(_ a: TopKBuffer, _ b: TopKBuffer, into out: inout TopKBuffer) {
        precondition(a.isMinHeap == b.isMinHeap && a.isMinHeap == out.isMinHeap, "Heap types must match")
        precondition(out.k >= a.k && out.k >= b.k, "Output capacity too small")
        var merged = TopKBuffer(k: out.k, isMinHeap: out.isMinHeap)
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
            return 1.0 - max(-1.0, min(1.0, sim))
        }
    }

    // Cosine fused (one-pass via existing helpers) — distances → Max-Heap
    @inlinable
    public static func range_topk_cosine_fused_512(query: Vector512Optimized, candidates: [Vector512Optimized], range: Range<Int>, k: Int, localTopK: inout TopKBuffer) {
        precondition(!localTopK.isMinHeap, "Cosine distance uses Max-Heap")
        // Precompute sumAA once
        let sumAA = DotKernels.dot512(query, query)
        for i in range {
            let c = candidates[i]
            let dot = DotKernels.dot512(query, c)
            let sumBB = DotKernels.dot512(c, c)
            let d = CosineKernels.calculateCosineDistance(dot: dot, sumAA: sumAA, sumBB: sumBB)
            localTopK.pushIfBetter(val: d, idx: i)
        }
    }
}
