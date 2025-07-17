import XCTest
@testable import VectorCore

final class ManhattanDistanceOptimizationTests: XCTestCase {
    
    func testManhattanDistanceOptimizationResults() {
        print("\n=== Manhattan Distance Optimization Results ===\n")
        
        let iterations = 10000
        
        // Test Dim256
        print("Dimension 256:")
        let vectors256a = (0..<100).map { _ in Vector<Dim256>.random(in: -10...10) }
        let vectors256b = (0..<100).map { _ in Vector<Dim256>.random(in: -10...10) }
        
        // Test optimized implementation
        let optimizedTime = measureTime {
            for _ in 0..<iterations/100 {
                for i in 0..<vectors256a.count {
                    _ = vectors256a[i].manhattanDistance(to: vectors256b[i])
                }
            }
        }
        let optimizedOps = Double(iterations) / optimizedTime / 1_000_000
        
        // Compare with euclidean distance
        let euclideanTime = measureTime {
            for _ in 0..<iterations/100 {
                for i in 0..<vectors256a.count {
                    _ = vectors256a[i].distance(to: vectors256b[i])
                }
            }
        }
        let euclideanOps = Double(iterations) / euclideanTime / 1_000_000
        
        print("  Manhattan (optimized): \(optimizedOps.formatted()) M ops/sec")
        print("  Euclidean distance:    \(euclideanOps.formatted()) M ops/sec")
        print("  Performance ratio:     \((optimizedOps / euclideanOps).formatted())×")
        
        // Test Dim512
        print("\nDimension 512:")
        let vectors512a = (0..<100).map { _ in Vector<Dim512>.random(in: -10...10) }
        let vectors512b = (0..<100).map { _ in Vector<Dim512>.random(in: -10...10) }
        
        let optimizedTime512 = measureTime {
            for _ in 0..<iterations/100 {
                for i in 0..<vectors512a.count {
                    _ = vectors512a[i].manhattanDistance(to: vectors512b[i])
                }
            }
        }
        let optimizedOps512 = Double(iterations) / optimizedTime512 / 1_000_000
        
        let euclideanTime512 = measureTime {
            for _ in 0..<iterations/100 {
                for i in 0..<vectors512a.count {
                    _ = vectors512a[i].distance(to: vectors512b[i])
                }
            }
        }
        let euclideanOps512 = Double(iterations) / euclideanTime512 / 1_000_000
        
        print("  Manhattan (optimized): \(optimizedOps512.formatted()) M ops/sec")
        print("  Euclidean distance:    \(euclideanOps512.formatted()) M ops/sec")
        print("  Performance ratio:     \((optimizedOps512 / euclideanOps512).formatted())×")
    }
    
    func testSmallVectorOptimization() {
        print("\n=== Small Vector Optimization ===")
        
        let iterations = 50000
        
        // Test Dim32 (boundary)
        let vectors32a = (0..<100).map { _ in Vector<Dim32>.random(in: -10...10) }
        let vectors32b = (0..<100).map { _ in Vector<Dim32>.random(in: -10...10) }
        
        let time32 = measureTime {
            for _ in 0..<iterations/100 {
                for i in 0..<vectors32a.count {
                    _ = vectors32a[i].manhattanDistance(to: vectors32b[i])
                }
            }
        }
        let ops32 = Double(iterations) / time32 / 1_000_000
        
        // Test Dim64 
        let vectors64a = (0..<100).map { _ in Vector<Dim64>.random(in: -10...10) }
        let vectors64b = (0..<100).map { _ in Vector<Dim64>.random(in: -10...10) }
        
        let time64 = measureTime {
            for _ in 0..<iterations/100 {
                for i in 0..<vectors64a.count {
                    _ = vectors64a[i].manhattanDistance(to: vectors64b[i])
                }
            }
        }
        let ops64 = Double(iterations) / time64 / 1_000_000
        
        print("  Dim32 (small):  \(ops32.formatted()) M ops/sec")
        print("  Dim64 (large):  \(ops64.formatted()) M ops/sec")
        print("  Speedup ratio:  \((ops32 / ops64).formatted())×")
    }
    
    func testL1NormOptimization() {
        print("\n=== L1 Norm Optimization ===")
        
        let iterations = 20000
        
        // Test small vectors
        let vectors32 = (0..<100).map { _ in Vector<Dim32>.random(in: -10...10) }
        let time32 = measureTime {
            for _ in 0..<iterations/100 {
                for v in vectors32 {
                    _ = v.l1Norm
                }
            }
        }
        let ops32 = Double(iterations) / time32 / 1_000_000
        
        // Test large vectors
        let vectors256 = (0..<100).map { _ in Vector<Dim256>.random(in: -10...10) }
        let time256 = measureTime {
            for _ in 0..<iterations/100 {
                for v in vectors256 {
                    _ = v.l1Norm
                }
            }
        }
        let ops256 = Double(iterations) / time256 / 1_000_000
        
        print("  Dim32:  \(ops32.formatted()) M ops/sec")
        print("  Dim256: \(ops256.formatted()) M ops/sec")
    }
    
    func testCorrectnessWithOptimization() {
        // Test Dim32 (small vector path)
        let v32a = Vector<Dim32>.random(in: -10...10)
        let v32b = Vector<Dim32>.random(in: -10...10)
        let dist32 = v32a.manhattanDistance(to: v32b)
        let manual32 = (0..<32).reduce(Float(0)) { sum, i in
            sum + abs(v32a[i] - v32b[i])
        }
        XCTAssertEqual(dist32, manual32, accuracy: 1e-5)
        
        // Test Dim256 (large vector path)
        let v256a = Vector<Dim256>.random(in: -10...10)
        let v256b = Vector<Dim256>.random(in: -10...10)
        let dist256 = v256a.manhattanDistance(to: v256b)
        let manual256 = (0..<256).reduce(Float(0)) { sum, i in
            sum + abs(v256a[i] - v256b[i])
        }
        XCTAssertEqual(dist256, manual256, accuracy: 1e-3)
        
        // Test edge cases
        let zeroVec = Vector<Dim128>()
        let oneVec = Vector<Dim128>(repeating: 1.0)
        
        // Zero distance
        XCTAssertEqual(v32a.manhattanDistance(to: v32a), 0, accuracy: 1e-6)
        
        // Symmetry
        XCTAssertEqual(v32a.manhattanDistance(to: v32b), 
                      v32b.manhattanDistance(to: v32a), 
                      accuracy: 1e-6)
        
        // Known value
        let knownDist = zeroVec.manhattanDistance(to: oneVec)
        XCTAssertEqual(knownDist, 128.0, accuracy: 1e-5)
    }
    
    private func measureTime(block: () -> Void) -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        block()
        let end = CFAbsoluteTimeGetCurrent()
        return end - start
    }
}