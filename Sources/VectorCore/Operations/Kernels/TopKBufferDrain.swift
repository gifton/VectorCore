//
//  TopKBufferDrain.swift
//  VectorCore
//

extension TopKBuffer {
    /// Removes every live entry, worst first, passing each entry's original index and score to `consume`.
    ///
    /// Heapifies once at entry (a partly filled buffer, `size < k`, is not yet a heap; a full one already is), then
    /// repeatedly hands the root to `consume`, moves the last live entry to the root, decrements `size` and restores
    /// the heap with one sift-down. Entries arrive in the reverse of the buffer's best-first order, under its own
    /// direction and tie policy (`isWorse`), so a consumer that fills a result row from its end to its start builds
    /// the best-first row. Score bits are passed unchanged (NaN payloads and zero signs included).
    ///
    /// Afterwards the buffer is empty (`size == 0`) with its capacity and policy unchanged, and can be refilled with
    /// `pushIfBetter`. An empty buffer makes no call. Mutating `self` copies its arrays first when they are shared
    /// with another value (copy-on-write), so a copy taken before the drain keeps its entries. No other array is
    /// created here.
    ///
    /// Tests: `TopKBufferDrainTests` (full, partial, empty, refill, both directions, every tie policy, exceptional
    /// score bits and copy independence, each against an independent sorted oracle).
    ///
    /// - Complexity: O(size) for the entry heapify plus O(log size) per removed entry.
    mutating func drainWorstFirst(_ consume: (Int, Float) -> Void) {
        buildHeap()
        while size > 0 {
            consume(idxs[0], vals[0])
            size -= 1
            if size > 0 {
                vals[0] = vals[size]
                idxs[0] = idxs[size]
                heapifyDown(from: 0)
            }
        }
    }
}
