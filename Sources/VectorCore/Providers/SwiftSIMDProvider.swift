// VectorCore: Pure Swift SIMD Provider
//
// Cross-platform SIMD implementation using Swift's built-in types
//

import Foundation

/// Pure Swift implementation of SIMDProvider
///
/// Uses Swift's built-in SIMD types (SIMD2, SIMD4, etc.) for vectorized operations.
/// These types compile to efficient SIMD instructions on all platforms.
///
/// ## Performance Characteristics
/// - Expected: 80-90% of platform-specific implementations
/// - Benefits: Zero dependencies, cross-platform
/// - Optimizations: Processes largest SIMD chunks first
public struct SwiftSIMDProvider: SIMDProvider {
    
    public init() {}
    
    // MARK: - Basic Arithmetic
    
    public func dot(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "Vectors must have same length")
        
        var result: Float = 0
        var i = 0
        let count = a.count
        
        // Process SIMD64 chunks (512 bits)
        while i + 64 <= count {
            let va = SIMD64<Float>(a[i..<i+64])
            let vb = SIMD64<Float>(b[i..<i+64])
            result += (va * vb).sum()
            i += 64
        }
        
        // Process SIMD32 chunks (256 bits)
        while i + 32 <= count {
            let va = SIMD32<Float>(a[i..<i+32])
            let vb = SIMD32<Float>(b[i..<i+32])
            result += (va * vb).sum()
            i += 32
        }
        
        // Process SIMD16 chunks (128 bits)
        while i + 16 <= count {
            let va = SIMD16<Float>(a[i..<i+16])
            let vb = SIMD16<Float>(b[i..<i+16])
            result += (va * vb).sum()
            i += 16
        }
        
        // Process SIMD8 chunks
        while i + 8 <= count {
            let va = SIMD8<Float>(a[i..<i+8])
            let vb = SIMD8<Float>(b[i..<i+8])
            result += (va * vb).sum()
            i += 8
        }
        
        // Process SIMD4 chunks
        while i + 4 <= count {
            let va = SIMD4<Float>(a[i..<i+4])
            let vb = SIMD4<Float>(b[i..<i+4])
            result += (va * vb).sum()
            i += 4
        }
        
        // Scalar remainder
        while i < count {
            result += a[i] * b[i]
            i += 1
        }
        
        return result
    }
    
    public func add(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count, "Vectors must have same length")
        
        var result = [Float](repeating: 0, count: a.count)
        var i = 0
        let count = a.count
        
        // Process SIMD64 chunks
        while i + 64 <= count {
            let va = SIMD64<Float>(a[i..<i+64])
            let vb = SIMD64<Float>(b[i..<i+64])
            let vr = va + vb
            for j in 0..<64 {
                result[i+j] = vr[j]
            }
            i += 64
        }
        
        // Process SIMD16 chunks
        while i + 16 <= count {
            let va = SIMD16<Float>(a[i..<i+16])
            let vb = SIMD16<Float>(b[i..<i+16])
            let vr = va + vb
            for j in 0..<16 {
                result[i+j] = vr[j]
            }
            i += 16
        }
        
        // Scalar remainder
        while i < count {
            result[i] = a[i] + b[i]
            i += 1
        }
        
        return result
    }
    
    public func subtract(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count, "Vectors must have same length")
        
        var result = [Float](repeating: 0, count: a.count)
        var i = 0
        let count = a.count
        
        // Process SIMD64 chunks
        while i + 64 <= count {
            let va = SIMD64<Float>(a[i..<i+64])
            let vb = SIMD64<Float>(b[i..<i+64])
            let vr = va - vb
            for j in 0..<64 {
                result[i+j] = vr[j]
            }
            i += 64
        }
        
        // Process SIMD16 chunks
        while i + 16 <= count {
            let va = SIMD16<Float>(a[i..<i+16])
            let vb = SIMD16<Float>(b[i..<i+16])
            let vr = va - vb
            for j in 0..<16 {
                result[i+j] = vr[j]
            }
            i += 16
        }
        
        // Scalar remainder
        while i < count {
            result[i] = a[i] - b[i]
            i += 1
        }
        
        return result
    }
    
    public func multiply(_ a: [Float], by scalar: Float) -> [Float] {
        var result = [Float](repeating: 0, count: a.count)
        var i = 0
        let count = a.count
        let scalarVec64 = SIMD64<Float>(repeating: scalar)
        let scalarVec16 = SIMD16<Float>(repeating: scalar)
        
        // Process SIMD64 chunks
        while i + 64 <= count {
            let va = SIMD64<Float>(a[i..<i+64])
            let vr = va * scalarVec64
            for j in 0..<64 {
                result[i+j] = vr[j]
            }
            i += 64
        }
        
        // Process SIMD16 chunks
        while i + 16 <= count {
            let va = SIMD16<Float>(a[i..<i+16])
            let vr = va * scalarVec16
            for j in 0..<16 {
                result[i+j] = vr[j]
            }
            i += 16
        }
        
        // Scalar remainder
        while i < count {
            result[i] = a[i] * scalar
            i += 1
        }
        
        return result
    }
    
    // MARK: - Reduction Operations
    
    public func sum(_ a: [Float]) -> Float {
        var result: Float = 0
        var i = 0
        let count = a.count
        
        // Process SIMD64 chunks
        while i + 64 <= count {
            let va = SIMD64<Float>(a[i..<i+64])
            result += va.sum()
            i += 64
        }
        
        // Process SIMD16 chunks
        while i + 16 <= count {
            let va = SIMD16<Float>(a[i..<i+16])
            result += va.sum()
            i += 16
        }
        
        // Scalar remainder
        while i < count {
            result += a[i]
            i += 1
        }
        
        return result
    }
    
    public func max(_ a: [Float]) -> Float {
        guard !a.isEmpty else { return -.infinity }
        
        var result = a[0]
        var i = 0
        let count = a.count
        
        // Process SIMD64 chunks
        while i + 64 <= count {
            let va = SIMD64<Float>(a[i..<i+64])
            result = Swift.max(result, va.max())
            i += 64
        }
        
        // Process SIMD16 chunks
        while i + 16 <= count {
            let va = SIMD16<Float>(a[i..<i+16])
            result = Swift.max(result, va.max())
            i += 16
        }
        
        // Scalar remainder
        while i < count {
            result = Swift.max(result, a[i])
            i += 1
        }
        
        return result
    }
    
    public func min(_ a: [Float]) -> Float {
        guard !a.isEmpty else { return .infinity }
        
        var result = a[0]
        var i = 0
        let count = a.count
        
        // Process SIMD64 chunks
        while i + 64 <= count {
            let va = SIMD64<Float>(a[i..<i+64])
            result = Swift.min(result, va.min())
            i += 64
        }
        
        // Process SIMD16 chunks
        while i + 16 <= count {
            let va = SIMD16<Float>(a[i..<i+16])
            result = Swift.min(result, va.min())
            i += 16
        }
        
        // Scalar remainder
        while i < count {
            result = Swift.min(result, a[i])
            i += 1
        }
        
        return result
    }
    
    // MARK: - Vector Operations
    
    public func magnitudeSquared(_ a: [Float]) -> Float {
        return dot(a, a)
    }
    
    // MARK: - Distance Operations
    
    public func euclideanDistanceSquared(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count, "Vectors must have same length")
        
        var result: Float = 0
        var i = 0
        let count = a.count
        
        // Process SIMD64 chunks
        while i + 64 <= count {
            let va = SIMD64<Float>(a[i..<i+64])
            let vb = SIMD64<Float>(b[i..<i+64])
            let diff = va - vb
            result += (diff * diff).sum()
            i += 64
        }
        
        // Process SIMD16 chunks
        while i + 16 <= count {
            let va = SIMD16<Float>(a[i..<i+16])
            let vb = SIMD16<Float>(b[i..<i+16])
            let diff = va - vb
            result += (diff * diff).sum()
            i += 16
        }
        
        // Scalar remainder
        while i < count {
            let diff = a[i] - b[i]
            result += diff * diff
            i += 1
        }
        
        return result
    }
    
    public func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        let dotProduct = dot(a, b)
        let magA = magnitude(a)
        let magB = magnitude(b)
        
        guard magA > 0 && magB > 0 else { return 0 }
        return dotProduct / (magA * magB)
    }
    
    // MARK: - Additional Operations
    
    public func elementWiseMultiply(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count, "Vectors must have same length")
        
        var result = [Float](repeating: 0, count: a.count)
        var i = 0
        let count = a.count
        
        // Process SIMD64 chunks
        while i + 64 <= count {
            let va = SIMD64<Float>(a[i..<i+64])
            let vb = SIMD64<Float>(b[i..<i+64])
            let vr = va * vb
            for j in 0..<64 {
                result[i+j] = vr[j]
            }
            i += 64
        }
        
        // Process SIMD16 chunks
        while i + 16 <= count {
            let va = SIMD16<Float>(a[i..<i+16])
            let vb = SIMD16<Float>(b[i..<i+16])
            let vr = va * vb
            for j in 0..<16 {
                result[i+j] = vr[j]
            }
            i += 16
        }
        
        // Scalar remainder
        while i < count {
            result[i] = a[i] * b[i]
            i += 1
        }
        
        return result
    }
    
    public func elementWiseDivide(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count, "Vectors must have same length")
        
        var result = [Float](repeating: 0, count: a.count)
        var i = 0
        let count = a.count
        
        // Process SIMD64 chunks
        while i + 64 <= count {
            let va = SIMD64<Float>(a[i..<i+64])
            let vb = SIMD64<Float>(b[i..<i+64])
            let vr = va / vb
            for j in 0..<64 {
                result[i+j] = vr[j]
            }
            i += 64
        }
        
        // Process SIMD16 chunks
        while i + 16 <= count {
            let va = SIMD16<Float>(a[i..<i+16])
            let vb = SIMD16<Float>(b[i..<i+16])
            let vr = va / vb
            for j in 0..<16 {
                result[i+j] = vr[j]
            }
            i += 16
        }
        
        // Scalar remainder
        while i < count {
            result[i] = a[i] / b[i]
            i += 1
        }
        
        return result
    }
    
    public func abs(_ a: [Float]) -> [Float] {
        var result = [Float](repeating: 0, count: a.count)
        var i = 0
        let count = a.count
        
        // Process SIMD64 chunks
        while i + 64 <= count {
            let va = SIMD64<Float>(a[i..<i+64])
            let vr = va.replacing(with: -va, where: va .< 0)
            for j in 0..<64 {
                result[i+j] = vr[j]
            }
            i += 64
        }
        
        // Process SIMD16 chunks
        while i + 16 <= count {
            let va = SIMD16<Float>(a[i..<i+16])
            let vr = va.replacing(with: -va, where: va .< 0)
            for j in 0..<16 {
                result[i+j] = vr[j]
            }
            i += 16
        }
        
        // Scalar remainder
        while i < count {
            result[i] = Swift.abs(a[i])
            i += 1
        }
        
        return result
    }
    
    public func elementWiseMin(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count, "Vectors must have same length")
        
        var result = [Float](repeating: 0, count: a.count)
        var i = 0
        let count = a.count
        
        // Process SIMD64 chunks
        while i + 64 <= count {
            let va = SIMD64<Float>(a[i..<i+64])
            let vb = SIMD64<Float>(b[i..<i+64])
            let vr = pointwiseMin(va, vb)
            for j in 0..<64 {
                result[i+j] = vr[j]
            }
            i += 64
        }
        
        // Process SIMD16 chunks
        while i + 16 <= count {
            let va = SIMD16<Float>(a[i..<i+16])
            let vb = SIMD16<Float>(b[i..<i+16])
            let vr = pointwiseMin(va, vb)
            for j in 0..<16 {
                result[i+j] = vr[j]
            }
            i += 16
        }
        
        // Scalar remainder
        while i < count {
            result[i] = Swift.min(a[i], b[i])
            i += 1
        }
        
        return result
    }
    
    public func elementWiseMax(_ a: [Float], _ b: [Float]) -> [Float] {
        precondition(a.count == b.count, "Vectors must have same length")
        
        var result = [Float](repeating: 0, count: a.count)
        var i = 0
        let count = a.count
        
        // Process SIMD64 chunks
        while i + 64 <= count {
            let va = SIMD64<Float>(a[i..<i+64])
            let vb = SIMD64<Float>(b[i..<i+64])
            let vr = pointwiseMax(va, vb)
            for j in 0..<64 {
                result[i+j] = vr[j]
            }
            i += 64
        }
        
        // Process SIMD16 chunks
        while i + 16 <= count {
            let va = SIMD16<Float>(a[i..<i+16])
            let vb = SIMD16<Float>(b[i..<i+16])
            let vr = pointwiseMax(va, vb)
            for j in 0..<16 {
                result[i+j] = vr[j]
            }
            i += 16
        }
        
        // Scalar remainder
        while i < count {
            result[i] = Swift.max(a[i], b[i])
            i += 1
        }
        
        return result
    }
    
    public func clip(_ a: [Float], min minVal: Float, max maxVal: Float) -> [Float] {
        var result = [Float](repeating: 0, count: a.count)
        var i = 0
        let count = a.count
        
        let minSIMD64 = SIMD64<Float>(repeating: minVal)
        let maxSIMD64 = SIMD64<Float>(repeating: maxVal)
        
        // Process SIMD64 chunks
        while i + 64 <= count {
            let va = SIMD64<Float>(a[i..<i+64])
            let clamped = pointwiseMin(pointwiseMax(va, minSIMD64), maxSIMD64)
            for j in 0..<64 {
                result[i+j] = clamped[j]
            }
            i += 64
        }
        
        let minSIMD16 = SIMD16<Float>(repeating: minVal)
        let maxSIMD16 = SIMD16<Float>(repeating: maxVal)
        
        // Process SIMD16 chunks
        while i + 16 <= count {
            let va = SIMD16<Float>(a[i..<i+16])
            let clamped = pointwiseMin(pointwiseMax(va, minSIMD16), maxSIMD16)
            for j in 0..<16 {
                result[i+j] = clamped[j]
            }
            i += 16
        }
        
        // Scalar remainder
        while i < count {
            result[i] = Swift.min(Swift.max(a[i], minVal), maxVal)
            i += 1
        }
        
        return result
    }
    
    public func sqrt(_ a: [Float]) -> [Float] {
        var result = [Float](repeating: 0, count: a.count)
        var i = 0
        let count = a.count
        
        // Process SIMD64 chunks
        while i + 64 <= count {
            let va = SIMD64<Float>(a[i..<i+64])
            let vr = va.squareRoot()
            for j in 0..<64 {
                result[i+j] = vr[j]
            }
            i += 64
        }
        
        // Process SIMD16 chunks
        while i + 16 <= count {
            let va = SIMD16<Float>(a[i..<i+16])
            let vr = va.squareRoot()
            for j in 0..<16 {
                result[i+j] = vr[j]
            }
            i += 16
        }
        
        // Scalar remainder
        while i < count {
            result[i] = Foundation.sqrt(a[i])
            i += 1
        }
        
        return result
    }
}

// MARK: - SIMD Extensions

private extension SIMD where Scalar == Float {
    /// Sum all elements
    func sum() -> Scalar {
        self.indices.reduce(Scalar.zero) { $0 + self[$1] }
    }
}