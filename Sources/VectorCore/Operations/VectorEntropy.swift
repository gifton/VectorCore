// VectorCore: Truncated entropy
//
// Scale-invariant entropy of the normalized magnitude distribution. Public `entropy`
// (VectorMath.swift, which documents the contract), the deprecated `entropyFast`, and both
// internal branch helpers share ONE implementation so the contract cannot diverge between routes.
//

import Foundation

extension Vector {
    /// Deprecated spelling of `entropy`: the same computation and the same value.
    ///
    /// Both width branches run the same two-pass scalar loop (`entropyTruncatedContract`) that
    /// `entropy` runs; no speed difference or speed figure is measured. D02 deprecated the name
    /// (routed from N04 through V04) because it suggested a separate variant. The contract is
    /// documented on `entropy` (VectorMath.swift).
    @available(*, deprecated, renamed: "entropy")
    @inlinable
    public var entropyFast: Float {
        // Both branches implement the same contract via one shared routine.
        if D.value <= 64 {
            return entropySmall()
        } else {
            return entropyLarge()
        }
    }

    /// Small-width branch; forwards to the shared implementation (kept for symbol compatibility).
    @usableFromInline
    internal func entropySmall() -> Float {
        entropyTruncatedContract()
    }

    /// Large-width branch; forwards to the shared implementation (kept for symbol compatibility).
    @usableFromInline
    internal func entropyLarge() -> Float {
        entropyTruncatedContract()
    }

    /// Shared implementation of the truncated-entropy contract documented on `entropy` (VectorMath.swift).
    ///
    /// Range argument (not an exact-sum or universal error theorem): a finite Float magnitude
    /// times an Int-sized count is below $2^{191}$ and the smallest nonzero Float divided by that
    /// bound is above $2^{-340}$, so the widened sum and the Double ratios stay in Double's
    /// normal range for nonzero ratios.
    internal func entropyTruncatedContract() -> Float {
        let cutoff = Double(Float.ulpOfOne)
        var sum = 0.0
        var compensation = 0.0
        var allFinite = true

        // Pass 1: finiteness check and Neumaier-compensated sum of magnitudes.
        storage.withUnsafeBufferPointer { buffer in
            for i in 0..<D.value {
                let x = buffer[i]
                // `isFinite` is false for NaN and both infinities (no min/max reliance).
                guard x.isFinite else {
                    allFinite = false
                    return
                }
                let m = Swift.abs(Double(x))
                let t = sum + m
                if Swift.abs(sum) >= m {
                    compensation += (sum - t) + m
                } else {
                    compensation += (m - t) + sum
                }
                sum = t
            }
        }

        guard allFinite else { return .nan }
        let total = sum + compensation
        // Only an all-zero input has total == 0 (no cutoff is applied to the input sum).
        guard total > 0 else { return 0 }

        // Pass 2: truncated entropy terms in Double.
        var entropy = 0.0
        storage.withUnsafeBufferPointer { buffer in
            for i in 0..<D.value {
                let p = Swift.abs(Double(buffer[i])) / total
                if p > cutoff {
                    entropy -= p * Foundation.log(p)
                }
            }
        }
        return Float(entropy)
    }
}
