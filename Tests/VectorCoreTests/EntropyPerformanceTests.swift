import XCTest
@testable import VectorCore

final class EntropyPerformanceTests: XCTestCase {
    
    func testEntropyPerformanceBaseline() {
        let iterations = 5000
        
        print("\n=== Entropy Calculation Performance (Baseline) ===")
        
        // Test Dim256
        print("\nDimension 256:")
        let vectors256 = (0..<100).map { _ in Vector<Dim256>.random(in: -10...10) }
        
        let entropyTime = measureTime {
            for _ in 0..<iterations/100 {
                for v in vectors256 {
                    _ = v.entropy
                }
            }
        }
        let entropyOps = Double(iterations) / entropyTime / 1_000_000
        
        // Compare with other metrics
        let magnitudeTime = measureTime {
            for _ in 0..<iterations/100 {
                for v in vectors256 {
                    _ = v.magnitude
                }
            }
        }
        let magnitudeOps = Double(iterations) / magnitudeTime / 1_000_000
        
        let varianceTime = measureTime {
            for _ in 0..<iterations/100 {
                for v in vectors256 {
                    _ = v.variance
                }
            }
        }
        let varianceOps = Double(iterations) / varianceTime / 1_000_000
        
        print("  Entropy:   \(entropyOps.formatted()) M ops/sec")
        print("  Magnitude: \(magnitudeOps.formatted()) M ops/sec")
        print("  Variance:  \(varianceOps.formatted()) M ops/sec")
        print("  Ratio (vs magnitude): \((entropyOps / magnitudeOps).formatted())×")
        
        // Test Dim512
        print("\nDimension 512:")
        let vectors512 = (0..<100).map { _ in Vector<Dim512>.random(in: -10...10) }
        
        let entropyTime512 = measureTime {
            for _ in 0..<iterations/100 {
                for v in vectors512 {
                    _ = v.entropy
                }
            }
        }
        let entropyOps512 = Double(iterations) / entropyTime512 / 1_000_000
        
        print("  Entropy:   \(entropyOps512.formatted()) M ops/sec")
        
        // Test edge cases
        print("\nEdge Case Performance:")
        
        // Zero vector
        let zeroVectors = (0..<100).map { _ in Vector<Dim256>() }
        let zeroTime = measureTime {
            for _ in 0..<iterations/100 {
                for v in zeroVectors {
                    _ = v.entropy
                }
            }
        }
        let zeroOps = Double(iterations) / zeroTime / 1_000_000
        
        // Sparse vectors
        let sparseVectors = (0..<100).map { _ -> Vector<Dim256> in
            var v = Vector<Dim256>()
            v[0] = 1.0
            v[1] = 0.5
            return v
        }
        let sparseTime = measureTime {
            for _ in 0..<iterations/100 {
                for v in sparseVectors {
                    _ = v.entropy
                }
            }
        }
        let sparseOps = Double(iterations) / sparseTime / 1_000_000
        
        print("  Zero vectors:   \(zeroOps.formatted()) M ops/sec")
        print("  Sparse vectors: \(sparseOps.formatted()) M ops/sec")
    }
    
    func testEntropyCorrectness() {
        // Test uniform distribution
        let uniform = Vector<Dim32>(repeating: 1.0)
        let uniformEntropy = uniform.entropy
        let expectedUniform = log(Float(32))
        XCTAssertEqual(uniformEntropy, expectedUniform, accuracy: 1e-5,
                      "Uniform distribution entropy should be log(n)")
        
        // Test single spike (zero entropy)
        var spike = Vector<Dim128>()
        spike[0] = 1.0
        XCTAssertEqual(spike.entropy, 0.0, accuracy: 1e-6,
                      "Single spike should have zero entropy")
        
        // Test zero vector
        let zero = Vector<Dim64>()
        XCTAssertEqual(zero.entropy, 0.0, accuracy: 1e-6,
                      "Zero vector should have zero entropy")
        
        // Test negative values (should use absolute values)
        let negative = Vector<Dim32>(repeating: -1.0)
        let negativeEntropy = negative.entropy
        XCTAssertEqual(negativeEntropy, expectedUniform, accuracy: 1e-5,
                      "Negative uniform should have same entropy as positive")
        
        // Test mixed values
        let mixed = Vector<Dim32>([1, -1, 2, -2] + Array(repeating: 0, count: 28))
        let mixedEntropy = mixed.entropy
        XCTAssertGreaterThan(mixedEntropy, 0)
        XCTAssertLessThan(mixedEntropy, log(4))  // Max entropy for 4 non-zero elements
    }
    
    func testEntropyEdgeCases() {
        // Test NaN handling
        var nanVector = Vector<Dim64>()
        nanVector[0] = .nan
        XCTAssertTrue(nanVector.entropy.isNaN,
                     "Vector with NaN should return NaN entropy")
        
        // Test infinity handling
        var infVector = Vector<Dim64>()
        infVector[0] = .infinity
        XCTAssertTrue(infVector.entropy.isNaN,
                     "Vector with infinity should return NaN entropy")
        
        // Test very small values
        let tiny = Vector<Dim32>(repeating: Float.ulpOfOne * 2)
        let tinyEntropy = tiny.entropy
        // Should be log(32) for uniform distribution
        XCTAssertEqual(tinyEntropy, log(Float(32)), accuracy: 1e-4,
                      "Vector with tiny uniform values should have max entropy")
    }
    
    func testEntropyDistributions() {
        // Two-value distribution
        var twoValue = Vector<Dim128>()
        for i in 0..<64 {
            twoValue[i] = 1.0
            twoValue[i + 64] = 2.0
        }
        let twoValueEntropy = twoValue.entropy
        
        // Calculate expected entropy
        let p1: Float = 64.0 / (64.0 + 128.0)  // Probability of value 1
        let p2: Float = 128.0 / (64.0 + 128.0) // Probability of value 2
        let expected = -(p1 * log(p1) + p2 * log(p2))
        
        XCTAssertEqual(twoValueEntropy, expected, accuracy: 1e-4,
                      "Two-value distribution entropy calculation")
    }
    
    private func measureTime(block: () -> Void) -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        block()
        let end = CFAbsoluteTimeGetCurrent()
        return end - start
    }
}