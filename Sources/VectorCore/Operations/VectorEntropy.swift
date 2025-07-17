// VectorCore: Optimized Entropy Operations
//
// Efficient entropy calculation using vForce
//

import Foundation
import Accelerate

extension Vector where D.Storage: VectorStorageOperations {
    /// Optimized Shannon entropy calculation using vForce
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
                absSum += abs(value)
            }
        }
        
        guard !hasNonFinite else { return .nan }
        guard absSum > Float.ulpOfOne else { return 0.0 }
        
        // Second pass: calculate entropy
        var entropy: Float = 0
        storage.withUnsafeBufferPointer { buffer in
            for i in 0..<D.value {
                let p = abs(buffer[i]) / absSum
                if p > Float.ulpOfOne {
                    entropy -= p * log(p)
                }
            }
        }
        
        return entropy
    }
    
    /// Vectorized calculation for large vectors
    @usableFromInline
    internal func entropyLarge() -> Float {
        var result: Float = 0
        
        storage.withUnsafeBufferPointer { buffer in
            // Allocate temporary buffers
            var absValues = [Float](repeating: 0, count: D.value)
            var probabilities = [Float](repeating: 0, count: D.value)
            
            absValues.withUnsafeMutableBufferPointer { absBuffer in
                probabilities.withUnsafeMutableBufferPointer { probBuffer in
                    // Step 1: Compute absolute values
                    vDSP_vabs(buffer.baseAddress!, 1,
                             absBuffer.baseAddress!, 1,
                             vDSP_Length(D.value))
                    
                    // Step 2: Check for non-finite values using min/max
                    var minVal: Float = 0
                    var maxVal: Float = 0
                    vDSP_minv(buffer.baseAddress!, 1, &minVal, vDSP_Length(D.value))
                    vDSP_maxv(buffer.baseAddress!, 1, &maxVal, vDSP_Length(D.value))
                    
                    guard minVal.isFinite && maxVal.isFinite else {
                        result = .nan
                        return
                    }
                    
                    // Step 3: Sum absolute values
                    var absSum: Float = 0
                    vDSP_sve(absBuffer.baseAddress!, 1, &absSum, vDSP_Length(D.value))
                    
                    guard absSum > Float.ulpOfOne else {
                        result = 0.0
                        return
                    }
                    
                    // Step 4: Normalize to probabilities
                    vDSP_vsdiv(absBuffer.baseAddress!, 1, &absSum,
                              probBuffer.baseAddress!, 1, vDSP_Length(D.value))
                    
                    // Step 5: Compute entropy using vForce
                    // We need to compute -p * log(p) for each element
                    var entropyTerms = [Float](repeating: 0, count: D.value)
                    entropyTerms.withUnsafeMutableBufferPointer { entropyBuffer in
                        // First, compute logarithms using vForce
                        var count = Int32(D.value)
                        vvlogf(entropyBuffer.baseAddress!, probBuffer.baseAddress!, &count)
                        
                        // Multiply probabilities by their logarithms
                        vDSP_vmul(probBuffer.baseAddress!, 1,
                                 entropyBuffer.baseAddress!, 1,
                                 entropyBuffer.baseAddress!, 1,
                                 vDSP_Length(D.value))
                        
                        // Handle near-zero probabilities (set to 0)
                        for i in 0..<D.value {
                            if probBuffer[i] <= Float.ulpOfOne {
                                entropyBuffer[i] = 0
                            }
                        }
                        
                        // Sum all terms
                        vDSP_sve(entropyBuffer.baseAddress!, 1, &result, vDSP_Length(D.value))
                    }
                    
                    result = -result
                }
            }
        }
        
        return result
    }
}