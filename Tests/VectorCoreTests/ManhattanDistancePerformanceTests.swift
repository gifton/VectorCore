import XCTest
@testable import VectorCore

final class ManhattanDistancePerformanceTests: XCTestCase {
    
    func testManhattanDistanceBaseline() {
        let iterations = 10000
        
        print("\n=== Manhattan Distance Performance (Baseline) ===")
        
        // Test Dim256
        print("\nDimension 256:")
        let vectors256a = (0..<100).map { _ in Vector<Dim256>.random(in: -10...10) }
        let vectors256b = (0..<100).map { _ in Vector<Dim256>.random(in: -10...10) }
        
        // Test current implementation
        let manhattanTime = measureTime {
            for _ in 0..<iterations/100 {
                for i in 0..<vectors256a.count {
                    _ = vectors256a[i].manhattanDistance(to: vectors256b[i])
                }
            }
        }
        let manhattanOps = Double(iterations) / manhattanTime / 1_000_000
        
        // Compare with separate operations
        let separateTime = measureTime {
            for _ in 0..<iterations/100 {
                for i in 0..<vectors256a.count {
                    let diff = vectors256a[i] - vectors256b[i]
                    _ = diff.l1Norm
                }
            }
        }
        let separateOps = Double(iterations) / separateTime / 1_000_000
        
        // Compare with euclidean distance for reference
        let euclideanTime = measureTime {
            for _ in 0..<iterations/100 {
                for i in 0..<vectors256a.count {
                    _ = vectors256a[i].distance(to: vectors256b[i])
                }
            }
        }
        let euclideanOps = Double(iterations) / euclideanTime / 1_000_000
        
        print("  Manhattan distance:  \(manhattanOps.formatted()) M ops/sec")
        print("  Separate ops:        \(separateOps.formatted()) M ops/sec")
        print("  Euclidean distance:  \(euclideanOps.formatted()) M ops/sec")
        print("  Ratio (vs euclidean): \((manhattanOps / euclideanOps).formatted())×")
        
        // Test Dim512
        print("\nDimension 512:")
        let vectors512a = (0..<100).map { _ in Vector<Dim512>.random(in: -10...10) }
        let vectors512b = (0..<100).map { _ in Vector<Dim512>.random(in: -10...10) }
        
        let manhattanTime512 = measureTime {
            for _ in 0..<iterations/100 {
                for i in 0..<vectors512a.count {
                    _ = vectors512a[i].manhattanDistance(to: vectors512b[i])
                }
            }
        }
        let manhattanOps512 = Double(iterations) / manhattanTime512 / 1_000_000
        
        let euclideanTime512 = measureTime {
            for _ in 0..<iterations/100 {
                for i in 0..<vectors512a.count {
                    _ = vectors512a[i].distance(to: vectors512b[i])
                }
            }
        }
        let euclideanOps512 = Double(iterations) / euclideanTime512 / 1_000_000
        
        print("  Manhattan distance:  \(manhattanOps512.formatted()) M ops/sec")
        print("  Euclidean distance:  \(euclideanOps512.formatted()) M ops/sec")
        print("  Ratio (vs euclidean): \((manhattanOps512 / euclideanOps512).formatted())×")
    }
    
    func testManhattanDistanceCorrectness() {
        let v1 = Vector<Dim128>([1, 2, 3, 4] + Array(repeating: 0, count: 124))
        let v2 = Vector<Dim128>([5, 6, 7, 8] + Array(repeating: 0, count: 124))
        
        let distance = v1.manhattanDistance(to: v2)
        let expected: Float = 4 + 4 + 4 + 4  // |1-5| + |2-6| + |3-7| + |4-8|
        XCTAssertEqual(distance, expected, accuracy: 1e-6)
        
        // Test zero distance
        let sameDistance = v1.manhattanDistance(to: v1)
        XCTAssertEqual(sameDistance, 0, accuracy: 1e-6)
        
        // Test symmetry
        let distance2 = v2.manhattanDistance(to: v1)
        XCTAssertEqual(distance, distance2, accuracy: 1e-6)
    }
    
    func testL1NormPerformance() {
        let iterations = 20000
        
        print("\n=== L1 Norm Performance ===")
        
        let vectors = (0..<100).map { _ in Vector<Dim256>.random(in: -10...10) }
        
        let l1Time = measureTime {
            for _ in 0..<iterations/100 {
                for v in vectors {
                    _ = v.l1Norm
                }
            }
        }
        let l1Ops = Double(iterations) / l1Time / 1_000_000
        
        // Compare with magnitude (L2 norm)
        let l2Time = measureTime {
            for _ in 0..<iterations/100 {
                for v in vectors {
                    _ = v.magnitude
                }
            }
        }
        let l2Ops = Double(iterations) / l2Time / 1_000_000
        
        print("  L1 norm:  \(l1Ops.formatted()) M ops/sec")
        print("  L2 norm:  \(l2Ops.formatted()) M ops/sec")
        print("  Ratio:    \((l1Ops / l2Ops).formatted())×")
    }
    
    private func measureTime(block: () -> Void) -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        block()
        let end = CFAbsoluteTimeGetCurrent()
        return end - start
    }
}