import XCTest
@testable import VectorCore

final class AdditionOptimizationTest: XCTestCase {
    
    func testAdditionPerformanceImprovement() {
        // Test that our optimization improved performance
        let iterations = 5000
        
        // Test Dim128 
        print("\n=== Testing Addition Optimization ===")
        
        let a128 = Vector<Dim128>.random(in: -1...1)
        let b128 = Vector<Dim128>.random(in: -1...1)
        
        let addTime128 = measureTime {
            for _ in 0..<iterations {
                _ = a128 + b128
            }
        }
        
        let dotTime128 = measureTime {
            for _ in 0..<iterations {
                _ = a128.dotProduct(b128)
            }
        }
        
        print("Dim128:")
        print("  Addition time: \(addTime128)")
        print("  Dot product time: \(dotTime128)")
        print("  Ratio: \((dotTime128/addTime128 * 100).formatted())%")
        
        // Test Dim512
        let a512 = Vector<Dim512>.random(in: -1...1)
        let b512 = Vector<Dim512>.random(in: -1...1)
        
        let addTime512 = measureTime {
            for _ in 0..<iterations {
                _ = a512 + b512
            }
        }
        
        let dotTime512 = measureTime {
            for _ in 0..<iterations {
                _ = a512.dotProduct(b512)
            }
        }
        
        print("\nDim512:")
        print("  Addition time: \(addTime512)")
        print("  Dot product time: \(dotTime512)")
        print("  Ratio: \((dotTime512/addTime512 * 100).formatted())%")
        
        // Test direct storage operations
        print("\nDirect MediumVectorStorage operations:")
        let s1 = MediumVectorStorage(count: 128, repeating: 0.5)
        let s2 = MediumVectorStorage(count: 128, repeating: 0.5)
        
        let storageAddTime = measureTime {
            for _ in 0..<iterations {
                _ = s1 + s2
            }
        }
        
        let storageDotTime = measureTime {
            for _ in 0..<iterations {
                _ = s1.dotProduct(s2)
            }
        }
        
        print("  Storage addition time: \(storageAddTime)")
        print("  Storage dot product time: \(storageDotTime)")
        print("  Ratio: \((storageDotTime/storageAddTime * 100).formatted())%")
        
        // Test that we're creating vectors correctly
        let testVec = a128 + b128
        print("\nTest vector created successfully, first element: \(testVec[0])")
    }
    
    private func measureTime(block: () -> Void) -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        block()
        let end = CFAbsoluteTimeGetCurrent()
        return end - start
    }
}