import Foundation

// Lightweight deterministic RNG (LCG) for reproducible inputs
public struct LCG {
    private var state: UInt64
    public init(seed: UInt64) { self.state = seed }

    @inline(__always)
    public mutating func next() -> UInt64 {
        // Numerical Recipes 64-bit LCG
        state &*= 6364136223846793005
        state &+= 1
        return state
    }

    @inline(__always)
    public mutating func nextFloat01() -> Float {
        // Use top 24 bits for mantissa to get a float in [0,1)
        let x = next() >> 40
        let val = Double(x) / Double(1 << 24)
        return Float(val)
    }

    @inline(__always)
    public mutating func nextFloat(in range: ClosedRange<Float>) -> Float {
        let t = nextFloat01()
        return range.lowerBound + (range.upperBound - range.lowerBound) * t
    }
}

public enum InputFactory {
    public static func randomArray(count: Int, seed: UInt64, range: ClosedRange<Float> = -1.0...1.0) -> [Float] {
        var rng = LCG(seed: seed)
        var a = [Float](repeating: 0, count: count)
        for i in 0..<count { a[i] = rng.nextFloat(in: range) }
        return a
    }
}

