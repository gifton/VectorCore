import XCTest
@testable import VectorCore

final class NormalizePerformanceTests: XCTestCase {
    
    func testNormalizePerformanceBaseline() {
        let iterations = 10000
        
        print("\n=== Normalize Operation Performance (Baseline) ===")
        
        // Test Dim256
        print("\nDimension 256:")
        let vectors256 = (0..<100).map { _ in Vector<Dim256>.random(in: -10...10) }
        
        // Test normalized() method (non-mutating)
        let normalizedTime = measureTime {
            for _ in 0..<iterations/100 {
                for v in vectors256 {
                    _ = v.normalized()
                }
            }
        }
        let normalizedOps = Double(iterations) / normalizedTime / 1_000_000
        
        // Test normalize() method (mutating)
        var mutableVectors = vectors256
        let normalizeTime = measureTime {
            for _ in 0..<iterations/100 {
                for i in 0..<mutableVectors.count {
                    mutableVectors[i] = vectors256[i]  // Reset
                    mutableVectors[i].normalize()
                }
            }
        }
        let normalizeOps = Double(iterations) / normalizeTime / 1_000_000
        
        // Test component operations for comparison
        let magnitudeTime = measureTime {
            for _ in 0..<iterations/100 {
                for v in vectors256 {
                    _ = v.magnitude
                }
            }
        }
        let magnitudeOps = Double(iterations) / magnitudeTime / 1_000_000
        
        // Test division operation
        let divisionTime = measureTime {
            for _ in 0..<iterations/100 {
                for v in vectors256 {
                    _ = v / 2.5  // Arbitrary scalar
                }
            }
        }
        let divisionOps = Double(iterations) / divisionTime / 1_000_000
        
        print("  normalized():     \(normalizedOps.formatted()) M ops/sec")
        print("  normalize():      \(normalizeOps.formatted()) M ops/sec")
        print("  magnitude only:   \(magnitudeOps.formatted()) M ops/sec")
        print("  division only:    \(divisionOps.formatted()) M ops/sec")
        
        // Calculate overhead
        let expectedOps = 1.0 / (1.0/magnitudeOps + 1.0/divisionOps)
        let overhead = normalizedOps / expectedOps
        print("  Efficiency:       \((overhead * 100).formatted())% (vs separate ops)")
        
        // Test Dim512
        print("\nDimension 512:")
        let vectors512 = (0..<100).map { _ in Vector<Dim512>.random(in: -10...10) }
        
        let normalizedTime512 = measureTime {
            for _ in 0..<iterations/100 {
                for v in vectors512 {
                    _ = v.normalized()
                }
            }
        }
        let normalizedOps512 = Double(iterations) / normalizedTime512 / 1_000_000
        
        let magnitudeTime512 = measureTime {
            for _ in 0..<iterations/100 {
                for v in vectors512 {
                    _ = v.magnitude
                }
            }
        }
        let magnitudeOps512 = Double(iterations) / magnitudeTime512 / 1_000_000
        
        print("  normalized():     \(normalizedOps512.formatted()) M ops/sec")
        print("  magnitude only:   \(magnitudeOps512.formatted()) M ops/sec")
    }
    
    func testNormalizeAlreadyNormalized() {
        // Test performance when vector is already normalized
        let iterations = 20000
        
        print("\n=== Already Normalized Vector Performance ===")
        
        // Create pre-normalized vectors
        let normalizedVectors = (0..<100).map { _ in 
            Vector<Dim256>.random(in: -10...10).normalized()
        }
        
        let time = measureTime {
            for _ in 0..<iterations/100 {
                for v in normalizedVectors {
                    _ = v.normalized()
                }
            }
        }
        
        let ops = Double(iterations) / time / 1_000_000
        print("  Already normalized: \(ops.formatted()) M ops/sec")
        
        // Compare with non-normalized
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
        print("  Regular vectors:    \(regularOps.formatted()) M ops/sec")
        print("  Speedup potential:  \((ops / regularOps).formatted())×")
    }
    
    func testNormalizeEdgeCases() {
        // Test zero vector
        let zeroVec = Vector<Dim128>()
        let normalizedZero = zeroVec.normalized()
        XCTAssertEqual(normalizedZero, zeroVec)
        
        // Test already normalized vector
        let unitVec = Vector<Dim128>([1, 0, 0, 0] + Array(repeating: 0, count: 124))
        let normalizedUnit = unitVec.normalized()
        XCTAssertEqual(normalizedUnit.magnitude, 1.0, accuracy: 1e-6)
        
        // Test very small magnitude
        let smallVec = Vector<Dim128>(repeating: 1e-20)
        let normalizedSmall = smallVec.normalized()
        if normalizedSmall.magnitude > 0 {
            XCTAssertEqual(normalizedSmall.magnitude, 1.0, accuracy: 1e-5)
        }
    }
    
    func testNormalizeCorrectness() {
        let v = Vector<Dim256>.random(in: -10...10)
        
        // Test that normalized vector has magnitude 1
        let normalized = v.normalized()
        XCTAssertEqual(normalized.magnitude, 1.0, accuracy: 1e-6)
        
        // Test that direction is preserved
        let dot = v.dotProduct(normalized)
        let expectedDot = v.magnitude  // Since normalized is in same direction
        XCTAssertEqual(dot, expectedDot, accuracy: 1e-5)
        
        // Test mutating version
        var vMut = v
        vMut.normalize()
        XCTAssertEqual(vMut.magnitude, 1.0, accuracy: 1e-6)
        XCTAssertTrue(vMut.isApproximatelyEqual(to: normalized))
    }
    
    private func measureTime(block: () -> Void) -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        block()
        let end = CFAbsoluteTimeGetCurrent()
        return end - start
    }
}