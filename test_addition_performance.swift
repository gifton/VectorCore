import Foundation
import VectorCore
import simd

// Test addition performance to isolate the issue
func measureTime(block: () -> Void) -> TimeInterval {
    let start = CFAbsoluteTimeGetCurrent()
    block()
    let end = CFAbsoluteTimeGetCurrent()
    return end - start
}

// Test different dimensions
let iterations = 10000

// Test Dim128
do {
    let a = Vector<Dim128>.random(in: -1...1)
    let b = Vector<Dim128>.random(in: -1...1)
    
    let addTime = measureTime {
        for _ in 0..<iterations {
            _ = a + b
        }
    }
    
    let dotTime = measureTime {
        for _ in 0..<iterations {
            _ = a.dotProduct(b)
        }
    }
    
    print("Dim128:")
    print("  Addition: \(iterations)/\(addTime) = \((Double(iterations)/addTime/1000000).formatted()) M ops/sec")
    print("  Dot Product: \(iterations)/\(dotTime) = \((Double(iterations)/dotTime/1000000).formatted()) M ops/sec")
    print("  Ratio: \((addTime/dotTime * 100).formatted())% (should be close to 100%)")
}

// Test Dim512
do {
    let a = Vector<Dim512>.random(in: -1...1)
    let b = Vector<Dim512>.random(in: -1...1)
    
    let addTime = measureTime {
        for _ in 0..<iterations {
            _ = a + b
        }
    }
    
    let dotTime = measureTime {
        for _ in 0..<iterations {
            _ = a.dotProduct(b)
        }
    }
    
    print("\nDim512:")
    print("  Addition: \(iterations)/\(addTime) = \((Double(iterations)/addTime/1000000).formatted()) M ops/sec")
    print("  Dot Product: \(iterations)/\(dotTime) = \((Double(iterations)/dotTime/1000000).formatted()) M ops/sec")
    print("  Ratio: \((addTime/dotTime * 100).formatted())% (should be close to 100%)")
}

// Direct test with storage to see if that's faster
do {
    print("\nDirect storage operations (MediumVectorStorage):")
    
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
    
    print("  Storage Addition: \((Double(iterations)/storageAddTime/1000000).formatted()) M ops/sec")
    print("  Storage Dot Product: \((Double(iterations)/storageDotTime/1000000).formatted()) M ops/sec")
    print("  Ratio: \((storageAddTime/storageDotTime * 100).formatted())% (should be close to 100%)")
}