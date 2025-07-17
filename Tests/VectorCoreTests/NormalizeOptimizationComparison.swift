import XCTest
@testable import VectorCore

final class NormalizeOptimizationComparison: XCTestCase {
    
    func testNormalizeOptimizationResults() {
        print("\n=== Normalize Operation Optimization Results ===\n")
        
        let iterations = 20000
        
        // Test already normalized vectors
        print("Already Normalized Vectors (Dim256):")
        let normalizedVectors = (0..<100).map { _ in 
            Vector<Dim256>.random(in: -10...10).normalized()
        }
        
        let optimizedTime = measureTime {
            for _ in 0..<iterations/100 {
                for v in normalizedVectors {
                    _ = v.normalized()
                }
            }
        }
        let optimizedOps = Double(iterations) / optimizedTime / 1_000_000
        print("  Performance: \(optimizedOps.formatted()) M ops/sec")
        
        // Test normalizedFast() method
        let fastTime = measureTime {
            for _ in 0..<iterations/100 {
                for v in normalizedVectors {
                    _ = v.normalizedFast()
                }
            }
        }
        let fastOps = Double(iterations) / fastTime / 1_000_000
        print("  Fast method: \(fastOps.formatted()) M ops/sec")
        
        // Test regular (non-normalized) vectors
        print("\nRegular Vectors (Dim256):")
        let regularVectors = (0..<100).map { _ in 
            Vector<Dim256>.random(in: -10...10)
        }
        
        let regularTime = measureTime {
            for _ in 0..<iterations/100 {
                for v in regularVectors {
                    _ = v.normalized()
                }
            }
        }
        let regularOps = Double(iterations) / regularTime / 1_000_000
        print("  Performance: \(regularOps.formatted()) M ops/sec")
        
        let regularFastTime = measureTime {
            for _ in 0..<iterations/100 {
                for v in regularVectors {
                    _ = v.normalizedFast()
                }
            }
        }
        let regularFastOps = Double(iterations) / regularFastTime / 1_000_000
        print("  Fast method: \(regularFastOps.formatted()) M ops/sec")
        
        print("\nSpeedups:")
        print("  Already normalized improvement: \((optimizedOps / regularOps).formatted())×")
        print("  Fast method vs standard: \((fastOps / optimizedOps).formatted())×")
        print("  Fast method on regular vectors: \((regularFastOps / regularOps).formatted())×")
    }
    
    func testIsNormalizedPerformance() {
        let iterations = 50000
        
        print("\n=== isNormalized Property Performance ===")
        
        let normalizedVec = Vector<Dim256>.random(in: -10...10).normalized()
        let regularVec = Vector<Dim256>.random(in: -10...10)
        
        // Test isNormalized check
        let isNormalizedTime = measureTime {
            for _ in 0..<iterations {
                _ = normalizedVec.isNormalized
            }
        }
        let isNormalizedOps = Double(iterations) / isNormalizedTime / 1_000_000
        
        // Compare with magnitude check
        let magnitudeTime = measureTime {
            for _ in 0..<iterations {
                _ = abs(normalizedVec.magnitude - 1.0) < 1e-6
            }
        }
        let magnitudeOps = Double(iterations) / magnitudeTime / 1_000_000
        
        print("  isNormalized check: \(isNormalizedOps.formatted()) M ops/sec")
        print("  magnitude check:    \(magnitudeOps.formatted()) M ops/sec")
        print("  Speedup: \((isNormalizedOps / magnitudeOps).formatted())×")
    }
    
    private func measureTime(block: () -> Void) -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        block()
        let end = CFAbsoluteTimeGetCurrent()
        return end - start
    }
}