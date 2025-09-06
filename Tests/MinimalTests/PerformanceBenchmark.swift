import XCTest
import Foundation
@testable import VectorCore

/// Performance benchmark to measure protocol overhead and operation speed
final class PerformanceBenchmark: XCTestCase {
    
    let iterations = 10_000
    let vectorSize = 32
    
    func measureTimeNanoseconds(_ block: () throws -> Void) rethrows -> Double {
        let start = CFAbsoluteTimeGetCurrent()
        try block()
        let end = CFAbsoluteTimeGetCurrent()
        return (end - start) * 1_000_000_000 / Double(iterations) // Convert to nanoseconds per operation
    }
    
    func testBaselinePerformance() throws {
        print("\n=== BASELINE PERFORMANCE (Before Protocol Simplification) ===\n")
        
        // Vector Creation
        let creationTime = try measureTimeNanoseconds {
            for _ in 0..<iterations {
                _ = Vector<Dim32>(repeating: 1.0)
            }
        }
        print("Vector Creation: \(String(format: "%.2f", creationTime)) ns/op")
        
        // Vector Addition
        let v1 = Vector<Dim32>(repeating: 1.0)
        let v2 = Vector<Dim32>(repeating: 2.0)
        let additionTime = try measureTimeNanoseconds {
            for _ in 0..<iterations {
                _ = v1 + v2
            }
        }
        print("Vector Addition: \(String(format: "%.2f", additionTime)) ns/op")
        
        // Vector Subtraction
        let subtractionTime = try measureTimeNanoseconds {
            for _ in 0..<iterations {
                _ = v1 - v2
            }
        }
        print("Vector Subtraction: \(String(format: "%.2f", subtractionTime)) ns/op")
        
        // Scalar Multiplication
        let multiplicationTime = try measureTimeNanoseconds {
            for _ in 0..<iterations {
                _ = v1 * 2.5
            }
        }
        print("Scalar Multiplication: \(String(format: "%.2f", multiplicationTime)) ns/op")
        
        // Dot Product
        let dotProductTime = try measureTimeNanoseconds {
            for _ in 0..<iterations {
                _ = v1.dotProduct(v2)
            }
        }
        print("Dot Product: \(String(format: "%.2f", dotProductTime)) ns/op")
        
        // Magnitude
        let magnitudeTime = try measureTimeNanoseconds {
            for _ in 0..<iterations {
                _ = v1.magnitude
            }
        }
        print("Magnitude: \(String(format: "%.2f", magnitudeTime)) ns/op")
        
        // Element Access
        let accessTime = try measureTimeNanoseconds {
            for _ in 0..<iterations {
                _ = v1[15]
            }
        }
        print("Element Access: \(String(format: "%.2f", accessTime)) ns/op")
        
        // Collection Iteration
        let iterationTime = try measureTimeNanoseconds {
            for _ in 0..<iterations {
                var sum: Float = 0
                for element in v1 {
                    sum += element
                }
                _ = sum
            }
        }
        print("Collection Iteration: \(String(format: "%.2f", iterationTime)) ns/op")
        
        // Protocol Dispatch Test (checking protocol overhead)
        func genericOperation<V: VectorProtocol>(_ vector: V) -> Float where V.Scalar == Float {
            return vector[0] + vector[1]
        }
        
        let protocolTime = try measureTimeNanoseconds {
            for _ in 0..<iterations {
                _ = genericOperation(v1)
            }
        }
        print("Protocol Dispatch: \(String(format: "%.2f", protocolTime)) ns/op")
        
        print("\n=== END BASELINE ===\n")
    }
}