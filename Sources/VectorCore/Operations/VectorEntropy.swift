// VectorCore: Optimized Entropy Operations
//
// Efficient entropy calculation using SwiftSIMDProvider
//

import Foundation

extension Vector where D.Storage: VectorStorageOperations {
    /// Optimized Shannon entropy calculation using SIMD operations
    ///
    /// This implementation uses vectorized operations for better performance
    /// while maintaining numerical accuracy.
    @inlinable
    public var entropyFast: Float {
        // For small vectors, use direct calculation
        if D.value <= 64 {
            return entropySmall()
        } else {
            return entropyLarge()
        }
    }

    /// Direct calculation for small vectors
    @usableFromInline
    internal func entropySmall() -> Float {
        var absSum: Float = 0
        var hasNonFinite = false

        // First pass: check for non-finite and sum absolute values
        storage.withUnsafeBufferPointer { buffer in
            for i in 0..<D.value {
                let value = buffer[i]
                if !value.isFinite {
                    hasNonFinite = true
                    break
                }
                absSum += Swift.abs(value)
            }
        }

        guard !hasNonFinite else { return .nan }
        guard absSum > Float.ulpOfOne else { return 0.0 }

        // Second pass: calculate entropy
        var entropy: Float = 0
        storage.withUnsafeBufferPointer { buffer in
            for i in 0..<D.value {
                let p = Swift.abs(buffer[i]) / absSum
                if p > Float.ulpOfOne {
                    entropy -= p * Foundation.log(p)
                }
            }
        }

        return entropy
    }

    /// Vectorized calculation for large vectors
    @usableFromInline
    internal func entropyLarge() -> Float {
        let provider = SwiftSIMDProvider()
        var result: Float = 0

        storage.withUnsafeBufferPointer { buffer in
            // Convert to array for provider
            let values = Array(buffer)

            // Step 1: Compute absolute values
            let absValues = provider.abs(values)

            // Step 2: Check for non-finite values using min/max
            let minVal = provider.min(values)
            let maxVal = provider.max(values)

            guard minVal.isFinite && maxVal.isFinite else {
                result = .nan
                return
            }

            // Step 3: Sum absolute values
            let absSum = provider.sum(absValues)

            // If the sum is non-finite (e.g., any element was NaN/Inf), return NaN for consistency
            guard absSum.isFinite else {
                result = .nan
                return
            }

            guard absSum > Float.ulpOfOne else {
                result = 0.0
                return
            }

            // Step 4: Normalize to probabilities
            let probabilities = provider.divide(absValues, by: absSum)

            // Step 5: Compute entropy using log
            // We need to compute -p * log(p) for each element
            let logProbs = provider.log(probabilities)
            let entropyTerms = provider.elementWiseMultiply(probabilities, logProbs)

            // Handle near-zero probabilities (set to 0)
            var cleanedTerms = entropyTerms
            for i in 0..<cleanedTerms.count {
                if probabilities[i] <= Float.ulpOfOne {
                    cleanedTerms[i] = 0
                }
            }

            // Sum all terms
            result = -provider.sum(cleanedTerms)
        }

        return result
    }
}
