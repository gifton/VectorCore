import XCTest
@testable import VectorCore

final class ScalarOptimizationBenchmark: XCTestCase {
    
    func testScalarOperationSpeedups() {
        let iterations = 20000
        
        print("\n=== Scalar Operation Optimization Results ===\n")
        
        // Test Dim256
        print("Dimension 256:")
        let v256 = Vector<Dim256>.random(in: -1...1)
        
        // Baseline: arbitrary scalar multiplication
        let baselineTime = measureTime {
            for _ in 0..<iterations {
                _ = v256 * 2.5
            }
        }
        let baselineOps = Double(iterations) / baselineTime / 1_000_000
        
        // Fast path: multiply by 0
        let mult0Time = measureTime {
            for _ in 0..<iterations {
                _ = v256 * 0
            }
        }
        let mult0Ops = Double(iterations) / mult0Time / 1_000_000
        let mult0Speedup = mult0Ops / baselineOps
        
        // Fast path: multiply by 1
        let mult1Time = measureTime {
            for _ in 0..<iterations {
                _ = v256 * 1
            }
        }
        let mult1Ops = Double(iterations) / mult1Time / 1_000_000
        let mult1Speedup = mult1Ops / baselineOps
        
        // Fast path: multiply by -1
        let multNeg1Time = measureTime {
            for _ in 0..<iterations {
                _ = v256 * -1
            }
        }
        let multNeg1Ops = Double(iterations) / multNeg1Time / 1_000_000
        let multNeg1Speedup = multNeg1Ops / baselineOps
        
        // Fast path: divide by 1
        let div1Time = measureTime {
            for _ in 0..<iterations {
                _ = v256 / 1
            }
        }
        let div1Ops = Double(iterations) / div1Time / 1_000_000
        let div1Speedup = div1Ops / baselineOps
        
        print("  Baseline (×2.5):  \(baselineOps.formatted()) M ops/sec")
        print("  Multiply by 0:    \(mult0Ops.formatted()) M ops/sec (\(mult0Speedup.formatted())× speedup)")
        print("  Multiply by 1:    \(mult1Ops.formatted()) M ops/sec (\(mult1Speedup.formatted())× speedup)")
        print("  Multiply by -1:   \(multNeg1Ops.formatted()) M ops/sec (\(multNeg1Speedup.formatted())× speedup)")
        print("  Divide by 1:      \(div1Ops.formatted()) M ops/sec (\(div1Speedup.formatted())× speedup)")
        print()
        
        // Test Dim512
        print("Dimension 512:")
        let v512 = Vector<Dim512>.random(in: -1...1)
        
        let baselineTime512 = measureTime {
            for _ in 0..<iterations {
                _ = v512 * 2.5
            }
        }
        let baselineOps512 = Double(iterations) / baselineTime512 / 1_000_000
        
        let mult1Time512 = measureTime {
            for _ in 0..<iterations {
                _ = v512 * 1
            }
        }
        let mult1Ops512 = Double(iterations) / mult1Time512 / 1_000_000
        let mult1Speedup512 = mult1Ops512 / baselineOps512
        
        print("  Baseline (×2.5):  \(baselineOps512.formatted()) M ops/sec")
        print("  Multiply by 1:    \(mult1Ops512.formatted()) M ops/sec (\(mult1Speedup512.formatted())× speedup)")
    }
    
    func testMutatingOperatorPerformance() {
        let iterations = 20000
        
        print("\n=== Mutating Operator Performance ===\n")
        
        var v = Vector<Dim256>.random(in: -1...1)
        let original = v
        
        // Test *= 1 (should be no-op)
        let mult1Time = measureTime {
            for _ in 0..<iterations {
                v = original
                v *= 1
            }
        }
        
        // Test *= 2.5 (general case)
        let mult2_5Time = measureTime {
            for _ in 0..<iterations {
                v = original
                v *= 2.5
            }
        }
        
        // Test /= 1 (should be no-op)
        let div1Time = measureTime {
            for _ in 0..<iterations {
                v = original
                v /= 1
            }
        }
        
        // Test /= 2.5 (general case)
        let div2_5Time = measureTime {
            for _ in 0..<iterations {
                v = original
                v /= 2.5
            }
        }
        
        print("Mutating operators (Dim256):")
        print("  *= 1:    \((Double(iterations)/mult1Time/1_000_000).formatted()) M ops/sec")
        print("  *= 2.5:  \((Double(iterations)/mult2_5Time/1_000_000).formatted()) M ops/sec")
        print("  /= 1:    \((Double(iterations)/div1Time/1_000_000).formatted()) M ops/sec")
        print("  /= 2.5:  \((Double(iterations)/div2_5Time/1_000_000).formatted()) M ops/sec")
        
        let mult1Speedup = mult2_5Time / mult1Time
        let div1Speedup = div2_5Time / div1Time
        
        print("\nSpeedups:")
        print("  *= 1 vs *= 2.5: \(mult1Speedup.formatted())×")
        print("  /= 1 vs /= 2.5:  \(div1Speedup.formatted())×")
    }
    
    private func measureTime(block: () -> Void) -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        block()
        let end = CFAbsoluteTimeGetCurrent()
        return end - start
    }
}