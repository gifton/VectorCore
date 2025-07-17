// VectorCore Comprehensive Performance Benchmarks
//
// Detailed performance testing for all VectorCore operations
//

import Foundation
import VectorCore

public struct ComprehensiveBenchmark {
    
    static let iterations = 1000
    
    // MARK: - Benchmark Runner
    
    public static func runBenchmarks() {
        print("Running Comprehensive VectorCore Benchmarks")
        print("==========================================")
        print("Platform: \(getPlatformInfo())")
        print("Iterations per test: \(iterations)")
        print()
        
        // Run all benchmark categories
        benchmarkVectorOperations()
        benchmarkStorageTypes()
        benchmarkDistanceMetrics()
        benchmarkQualityMetrics()
        benchmarkDynamicVectors()
        
        print("\n==========================================")
        print("Benchmarks completed successfully")
    }
    
    // MARK: - Vector Operations
    
    static func benchmarkVectorOperations() {
        print("\n=== Vector Operations Benchmarks ===")
        
        // Test different vector sizes
        print("\nVector size: 32")
        let v32_1 = Vector<Dim32>.random(in: -1...1)
        let v32_2 = Vector<Dim32>.random(in: -1...1)
        benchmarkVector32(v1: v32_1, v2: v32_2)
        
        print("\nVector size: 64")
        let v64_1 = Vector<Dim64>.random(in: -1...1)
        let v64_2 = Vector<Dim64>.random(in: -1...1)
        benchmarkVector64(v1: v64_1, v2: v64_2)
        
        print("\nVector size: 128")
        let v128_1 = Vector128.random(in: -1...1)
        let v128_2 = Vector128.random(in: -1...1)
        benchmarkVector128(v1: v128_1, v2: v128_2)
        
        print("\nVector size: 256")
        let v256_1 = Vector256.random(in: -1...1)
        let v256_2 = Vector256.random(in: -1...1)
        benchmarkVector256(v1: v256_1, v2: v256_2)
        
        print("\nVector size: 512")
        let v512_1 = Vector512.random(in: -1...1)
        let v512_2 = Vector512.random(in: -1...1)
        benchmarkVector512(v1: v512_1, v2: v512_2)
        
        print("\nVector size: 768")
        let v768_1 = Vector768.random(in: -1...1)
        let v768_2 = Vector768.random(in: -1...1)
        benchmarkVector768(v1: v768_1, v2: v768_2)
    }
    
    static func benchmarkVector32(v1: Vector<Dim32>, v2: Vector<Dim32>) {
        let scalar: Float = 2.5
        
        // Addition
        let addTime = measure {
            for _ in 0..<iterations {
                _ = v1 + v2
            }
        }
        print("  Addition: \(formatTime(addTime)) (\(formatOps(iterations, addTime)) ops/sec)")
        
        // Subtraction
        let subTime = measure {
            for _ in 0..<iterations {
                _ = v1 - v2
            }
        }
        print("  Subtraction: \(formatTime(subTime)) (\(formatOps(iterations, subTime)) ops/sec)")
        
        // Scalar multiplication
        let scalarTime = measure {
            for _ in 0..<iterations {
                _ = v1 * scalar
            }
        }
        print("  Scalar multiply: \(formatTime(scalarTime)) (\(formatOps(iterations, scalarTime)) ops/sec)")
        
        // Dot product
        let dotTime = measure {
            for _ in 0..<iterations {
                _ = v1.dotProduct(v2)
            }
        }
        print("  Dot product: \(formatTime(dotTime)) (\(formatOps(iterations, dotTime)) ops/sec)")
        
        // Magnitude
        let magTime = measure {
            for _ in 0..<iterations {
                _ = v1.magnitude
            }
        }
        print("  Magnitude: \(formatTime(magTime)) (\(formatOps(iterations, magTime)) ops/sec)")
        
        // Normalization
        let normTime = measure {
            for _ in 0..<iterations {
                _ = v1.normalized()
            }
        }
        print("  Normalize: \(formatTime(normTime)) (\(formatOps(iterations, normTime)) ops/sec)")
    }
    
    static func benchmarkVector64(v1: Vector<Dim64>, v2: Vector<Dim64>) {
        let scalar: Float = 2.5
        
        let addTime = measure {
            for _ in 0..<iterations {
                _ = v1 + v2
            }
        }
        print("  Addition: \(formatTime(addTime)) (\(formatOps(iterations, addTime)) ops/sec)")
        
        let subTime = measure {
            for _ in 0..<iterations {
                _ = v1 - v2
            }
        }
        print("  Subtraction: \(formatTime(subTime)) (\(formatOps(iterations, subTime)) ops/sec)")
        
        let scalarTime = measure {
            for _ in 0..<iterations {
                _ = v1 * scalar
            }
        }
        print("  Scalar multiply: \(formatTime(scalarTime)) (\(formatOps(iterations, scalarTime)) ops/sec)")
        
        let dotTime = measure {
            for _ in 0..<iterations {
                _ = v1.dotProduct(v2)
            }
        }
        print("  Dot product: \(formatTime(dotTime)) (\(formatOps(iterations, dotTime)) ops/sec)")
        
        let magTime = measure {
            for _ in 0..<iterations {
                _ = v1.magnitude
            }
        }
        print("  Magnitude: \(formatTime(magTime)) (\(formatOps(iterations, magTime)) ops/sec)")
        
        let normTime = measure {
            for _ in 0..<iterations {
                _ = v1.normalized()
            }
        }
        print("  Normalize: \(formatTime(normTime)) (\(formatOps(iterations, normTime)) ops/sec)")
    }
    
    static func benchmarkVector128(v1: Vector128, v2: Vector128) {
        let scalar: Float = 2.5
        
        let addTime = measure {
            for _ in 0..<iterations {
                _ = v1 + v2
            }
        }
        print("  Addition: \(formatTime(addTime)) (\(formatOps(iterations, addTime)) ops/sec)")
        
        let dotTime = measure {
            for _ in 0..<iterations {
                _ = v1.dotProduct(v2)
            }
        }
        print("  Dot product: \(formatTime(dotTime)) (\(formatOps(iterations, dotTime)) ops/sec)")
        
        let magTime = measure {
            for _ in 0..<iterations {
                _ = v1.magnitude
            }
        }
        print("  Magnitude: \(formatTime(magTime)) (\(formatOps(iterations, magTime)) ops/sec)")
    }
    
    static func benchmarkVector256(v1: Vector256, v2: Vector256) {
        let scalar: Float = 2.5
        
        let addTime = measure {
            for _ in 0..<iterations {
                _ = v1 + v2
            }
        }
        print("  Addition: \(formatTime(addTime)) (\(formatOps(iterations, addTime)) ops/sec)")
        
        let dotTime = measure {
            for _ in 0..<iterations {
                _ = v1.dotProduct(v2)
            }
        }
        print("  Dot product: \(formatTime(dotTime)) (\(formatOps(iterations, dotTime)) ops/sec)")
        
        let magTime = measure {
            for _ in 0..<iterations {
                _ = v1.magnitude
            }
        }
        print("  Magnitude: \(formatTime(magTime)) (\(formatOps(iterations, magTime)) ops/sec)")
    }
    
    static func benchmarkVector512(v1: Vector512, v2: Vector512) {
        let scalar: Float = 2.5
        
        let addTime = measure {
            for _ in 0..<iterations {
                _ = v1 + v2
            }
        }
        print("  Addition: \(formatTime(addTime)) (\(formatOps(iterations, addTime)) ops/sec)")
        
        let dotTime = measure {
            for _ in 0..<iterations {
                _ = v1.dotProduct(v2)
            }
        }
        print("  Dot product: \(formatTime(dotTime)) (\(formatOps(iterations, dotTime)) ops/sec)")
        
        let magTime = measure {
            for _ in 0..<iterations {
                _ = v1.magnitude
            }
        }
        print("  Magnitude: \(formatTime(magTime)) (\(formatOps(iterations, magTime)) ops/sec)")
    }
    
    static func benchmarkVector768(v1: Vector768, v2: Vector768) {
        let scalar: Float = 2.5
        
        let addTime = measure {
            for _ in 0..<iterations {
                _ = v1 + v2
            }
        }
        print("  Addition: \(formatTime(addTime)) (\(formatOps(iterations, addTime)) ops/sec)")
        
        let dotTime = measure {
            for _ in 0..<iterations {
                _ = v1.dotProduct(v2)
            }
        }
        print("  Dot product: \(formatTime(dotTime)) (\(formatOps(iterations, dotTime)) ops/sec)")
        
        let magTime = measure {
            for _ in 0..<iterations {
                _ = v1.magnitude
            }
        }
        print("  Magnitude: \(formatTime(magTime)) (\(formatOps(iterations, magTime)) ops/sec)")
    }
    
    // MARK: - Storage Types
    
    static func benchmarkStorageTypes() {
        print("\n=== Storage Type Benchmarks ===")
        
        // Small storage (SIMD64)
        print("\nSmallVectorStorage (1-64 elements):")
        let small32 = SmallVectorStorage(count: 32, repeating: 1.0)
        let small64 = SmallVectorStorage(count: 64, repeating: 1.0)
        benchmarkStorage(small32, small32, "32 elements")
        benchmarkStorage(small64, small64, "64 elements")
        
        // Medium storage (AlignedValueStorage)
        print("\nMediumVectorStorage (65-512 elements):")
        let medium128 = MediumVectorStorage(count: 128, repeating: 1.0)
        let medium256 = MediumVectorStorage(count: 256, repeating: 1.0)
        benchmarkStorage(medium128, medium128, "128 elements")
        benchmarkStorage(medium256, medium256, "256 elements")
        
        // Large storage (COWDynamicStorage)
        print("\nLargeVectorStorage (513+ elements):")
        let large768 = LargeVectorStorage(count: 768, repeating: 1.0)
        let large1536 = LargeVectorStorage(count: 1536, repeating: 1.0)
        benchmarkStorage(large768, large768, "768 elements")
        benchmarkStorage(large1536, large1536, "1536 elements")
    }
    
    static func benchmarkStorage<S: VectorStorage & VectorStorageOperations>(_ s1: S, _ s2: S, _ label: String) {
        print("\n  \(label):")
        
        // Element access
        let accessTime = measure {
            for _ in 0..<iterations {
                for i in 0..<min(s1.count, 100) {
                    _ = s1[i]
                }
            }
        }
        print("    Element access: \(formatTime(accessTime)) (\(formatOps(iterations * min(s1.count, 100), accessTime)) ops/sec)")
        
        // Dot product
        let dotTime = measure {
            for _ in 0..<iterations {
                _ = s1.dotProduct(s2)
            }
        }
        print("    Dot product: \(formatTime(dotTime)) (\(formatOps(iterations, dotTime)) ops/sec)")
        
        // Copy performance (for COW types)
        var mutableCopy = s1
        let copyTime = measure {
            for _ in 0..<iterations {
                let _ = mutableCopy  // Copy
                mutableCopy[0] = 2.0  // Trigger COW
            }
        }
        print("    Copy + COW: \(formatTime(copyTime)) (\(formatOps(iterations, copyTime)) ops/sec)")
    }
    
    // MARK: - Distance Metrics
    
    static func benchmarkDistanceMetrics() {
        print("\n=== Distance Metric Benchmarks ===")
        
        let v1 = Vector256.random(in: -1...1)
        let v2 = Vector256.random(in: -1...1)
        
        // Euclidean distance
        let euclideanTime = measure {
            for _ in 0..<iterations {
                _ = v1.distance(to: v2)
            }
        }
        print("  Euclidean distance: \(formatTime(euclideanTime)) (\(formatOps(iterations, euclideanTime)) ops/sec)")
        
        // Cosine similarity
        let cosineTime = measure {
            for _ in 0..<iterations {
                _ = v1.cosineSimilarity(to: v2)
            }
        }
        print("  Cosine similarity: \(formatTime(cosineTime)) (\(formatOps(iterations, cosineTime)) ops/sec)")
        
        // Manhattan distance
        let manhattanTime = measure {
            for _ in 0..<iterations {
                _ = v1.manhattanDistance(to: v2)
            }
        }
        print("  Manhattan distance: \(formatTime(manhattanTime)) (\(formatOps(iterations, manhattanTime)) ops/sec)")
        
        // Norms
        let l1Time = measure {
            for _ in 0..<iterations {
                _ = v1.l1Norm
            }
        }
        print("  L1 norm: \(formatTime(l1Time)) (\(formatOps(iterations, l1Time)) ops/sec)")
        
        let l2Time = measure {
            for _ in 0..<iterations {
                _ = v1.l2Norm
            }
        }
        print("  L2 norm: \(formatTime(l2Time)) (\(formatOps(iterations, l2Time)) ops/sec)")
    }
    
    // MARK: - Quality Metrics
    
    static func benchmarkQualityMetrics() {
        print("\n=== Quality Metric Benchmarks ===")
        
        let sparse = Vector256(Array(repeating: 0.0, count: 200) + Array(repeating: 1.0, count: 56))
        let dense = Vector256.random(in: -1...1)
        
        print("\nSparse vector (78% zeros):")
        benchmarkQualityVector256(sparse)
        
        print("\nDense vector (random values):")
        benchmarkQualityVector256(dense)
    }
    
    static func benchmarkQualityVector256(_ vector: Vector256) {
        // Sparsity
        let sparsityTime = measure {
            for _ in 0..<iterations {
                _ = vector.sparsity
            }
        }
        print("  Sparsity: \(formatTime(sparsityTime)) (\(formatOps(iterations, sparsityTime)) ops/sec)")
        
        // Entropy
        let entropyTime = measure {
            for _ in 0..<iterations {
                _ = vector.entropy
            }
        }
        print("  Entropy: \(formatTime(entropyTime)) (\(formatOps(iterations, entropyTime)) ops/sec)")
        
        // Quality (composite)
        let qualityTime = measure {
            for _ in 0..<iterations {
                _ = vector.quality
            }
        }
        print("  Quality: \(formatTime(qualityTime)) (\(formatOps(iterations, qualityTime)) ops/sec)")
        
        // Base64 encoding
        let base64EncodeTime = measure {
            for _ in 0..<iterations {
                _ = vector.base64Encoded
            }
        }
        print("  Base64 encode: \(formatTime(base64EncodeTime)) (\(formatOps(iterations, base64EncodeTime)) ops/sec)")
        
        // Base64 decoding
        let base64String = vector.base64Encoded
        let base64DecodeTime = measure {
            for _ in 0..<iterations {
                _ = try? Vector256.base64Decoded(from: base64String)
            }
        }
        print("  Base64 decode: \(formatTime(base64DecodeTime)) (\(formatOps(iterations, base64DecodeTime)) ops/sec)")
    }
    
    // MARK: - Dynamic Vectors
    
    static func benchmarkDynamicVectors() {
        print("\n=== Dynamic Vector Benchmarks ===")
        
        let sizes = [128, 384, 768, 1536]
        
        for size in sizes {
            print("\nDynamic vector size: \(size)")
            
            let v1 = DynamicVector.random(dimension: size, in: -1...1)
            let v2 = DynamicVector.random(dimension: size, in: -1...1)
            
            // Basic operations
            let addTime = measure {
                for _ in 0..<iterations {
                    _ = v1 + v2
                }
            }
            print("  Addition: \(formatTime(addTime)) (\(formatOps(iterations, addTime)) ops/sec)")
            
            // Dot product
            let dotTime = measure {
                for _ in 0..<iterations {
                    _ = v1.dotProduct(v2)
                }
            }
            print("  Dot product: \(formatTime(dotTime)) (\(formatOps(iterations, dotTime)) ops/sec)")
            
            // Distance
            let distTime = measure {
                for _ in 0..<iterations {
                    _ = v1.distance(to: v2)
                }
            }
            print("  Euclidean distance: \(formatTime(distTime)) (\(formatOps(iterations, distTime)) ops/sec)")
            
            // Magnitude
            let magTime = measure {
                for _ in 0..<iterations {
                    _ = v1.magnitude
                }
            }
            print("  Magnitude: \(formatTime(magTime)) (\(formatOps(iterations, magTime)) ops/sec)")
        }
    }
    
    // MARK: - Utility Functions
    
    static func measure(block: () -> Void) -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        block()
        let end = CFAbsoluteTimeGetCurrent()
        return end - start
    }
    
    static func formatTime(_ time: TimeInterval) -> String {
        if time < 0.001 {
            return String(format: "%.3f µs", time * 1_000_000)
        } else if time < 1.0 {
            return String(format: "%.3f ms", time * 1_000)
        } else {
            return String(format: "%.3f s", time)
        }
    }
    
    static func formatOps(_ ops: Int, _ time: TimeInterval) -> String {
        let opsPerSec = Double(ops) / time
        if opsPerSec > 1_000_000 {
            return String(format: "%.2fM", opsPerSec / 1_000_000)
        } else if opsPerSec > 1_000 {
            return String(format: "%.2fK", opsPerSec / 1_000)
        } else {
            return String(format: "%.0f", opsPerSec)
        }
    }
    
    static func getPlatformInfo() -> String {
        #if arch(arm64)
        return "Apple Silicon (ARM64)"
        #elseif arch(x86_64)
        return "Intel (x86_64)"
        #else
        return "Unknown"
        #endif
    }
}