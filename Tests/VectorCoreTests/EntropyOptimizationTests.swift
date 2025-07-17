import XCTest
@testable import VectorCore

final class EntropyOptimizationTests: XCTestCase {
    
    func testEntropyOptimizationResults() {
        print("\n=== Entropy Calculation Optimization Results ===\n")
        
        let iterations = 5000
        
        // Test Dim256
        print("Dimension 256:")
        let vectors256 = (0..<100).map { _ in Vector<Dim256>.random(in: -10...10) }
        
        // Test optimized implementation
        let optimizedTime = measureTime {
            for _ in 0..<iterations/100 {
                for v in vectors256 {
                    _ = v.entropy
                }
            }
        }
        let optimizedOps = Double(iterations) / optimizedTime / 1_000_000
        
        // Compare with other operations
        let magnitudeTime = measureTime {
            for _ in 0..<iterations/100 {
                for v in vectors256 {
                    _ = v.magnitude
                }
            }
        }
        let magnitudeOps = Double(iterations) / magnitudeTime / 1_000_000
        
        print("  Entropy (optimized): \(optimizedOps.formatted()) M ops/sec")
        print("  Magnitude:           \(magnitudeOps.formatted()) M ops/sec")
        print("  Performance ratio:   \((optimizedOps / magnitudeOps).formatted())×")
        
        // Test Dim512
        print("\nDimension 512:")
        let vectors512 = (0..<100).map { _ in Vector<Dim512>.random(in: -10...10) }
        
        let optimizedTime512 = measureTime {
            for _ in 0..<iterations/100 {
                for v in vectors512 {
                    _ = v.entropy
                }
            }
        }
        let optimizedOps512 = Double(iterations) / optimizedTime512 / 1_000_000
        
        print("  Entropy (optimized): \(optimizedOps512.formatted()) M ops/sec")
        
        // Test small vectors
        print("\nSmall Vector Performance (Dim32):")
        let vectors32 = (0..<100).map { _ in Vector<Dim32>.random(in: -10...10) }
        
        let time32 = measureTime {
            for _ in 0..<iterations/100 {
                for v in vectors32 {
                    _ = v.entropy
                }
            }
        }
        let ops32 = Double(iterations) / time32 / 1_000_000
        print("  Entropy: \(ops32.formatted()) M ops/sec")
    }
    
    func testEntropyAccuracy() {
        // Test uniform distribution
        let uniform256 = Vector<Dim256>(repeating: 1.0)
        let entropy256 = uniform256.entropy
        let expected256 = log(Float(256))
        XCTAssertEqual(entropy256, expected256, accuracy: 1e-5,
                      "Uniform distribution entropy accuracy")
        
        // Test with entropyFast directly
        let entropyFast256 = uniform256.entropyFast
        XCTAssertEqual(entropyFast256, expected256, accuracy: 1e-5,
                      "Fast entropy accuracy")
        
        // Test mixed distribution
        var mixed = Vector<Dim128>()
        for i in 0..<64 {
            mixed[i] = Float(i + 1)
        }
        
        let mixedEntropy = mixed.entropy
        let mixedEntropyFast = mixed.entropyFast
        XCTAssertEqual(mixedEntropy, mixedEntropyFast, accuracy: 1e-5,
                      "Mixed distribution consistency")
        
        // Test small vector path
        let small = Vector<Dim32>.random(in: -1...1)
        let smallEntropy = small.entropy
        let smallEntropyDirect = small.entropySmall()
        XCTAssertEqual(smallEntropy, smallEntropyDirect, accuracy: 1e-6,
                      "Small vector path accuracy")
    }
    
    func testEntropyEdgeCases() {
        // Test NaN handling
        var nanVec = Vector<Dim128>()
        nanVec[0] = .nan
        XCTAssertTrue(nanVec.entropy.isNaN)
        XCTAssertTrue(nanVec.entropyFast.isNaN)
        
        // Test infinity handling
        var infVec = Vector<Dim128>()
        infVec[0] = .infinity
        XCTAssertTrue(infVec.entropy.isNaN)
        XCTAssertTrue(infVec.entropyFast.isNaN)
        
        // Test zero vector
        let zero = Vector<Dim256>()
        XCTAssertEqual(zero.entropy, 0.0, accuracy: 1e-6)
        XCTAssertEqual(zero.entropyFast, 0.0, accuracy: 1e-6)
        
        // Test single spike
        var spike = Vector<Dim512>()
        spike[100] = 5.0
        XCTAssertEqual(spike.entropy, 0.0, accuracy: 1e-6)
        XCTAssertEqual(spike.entropyFast, 0.0, accuracy: 1e-6)
    }
    
    func testQualityMetricPerformance() {
        print("\n=== Quality Metric Performance ===")
        
        let iterations = 1000
        let vectors = (0..<100).map { _ in Vector<Dim256>.random(in: -10...10) }
        
        let qualityTime = measureTime {
            for _ in 0..<iterations/100 {
                for v in vectors {
                    _ = v.quality
                }
            }
        }
        let qualityOps = Double(iterations) / qualityTime / 1_000_000
        
        print("  Full quality metric: \(qualityOps.formatted()) M ops/sec")
    }
    
    private func measureTime(block: () -> Void) -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        block()
        let end = CFAbsoluteTimeGetCurrent()
        return end - start
    }
}