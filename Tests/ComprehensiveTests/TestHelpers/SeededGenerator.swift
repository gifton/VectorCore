import Foundation

/// Deterministic SplitMix64 pseudo-random generator for reproducible tests.
///
/// Numerical/accuracy tests that feed `Float.random` into tight tolerances are otherwise
/// flaky seed-to-seed. Use a **fresh instance per test** (a local `var rng`) so results are
/// deterministic AND data-race-free under Swift Testing's parallel execution — never share a
/// single generator across concurrently-running tests.
///
/// Usage:
/// ```swift
/// var rng = SeededGenerator(seed: 0x1234)
/// let values = (0..<512).map { _ in Float.random(in: -1...1, using: &rng) }
/// ```
public struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64

    /// - Parameter seed: any 64-bit seed; the default is a fixed constant for full determinism.
    public init(seed: UInt64 = 0x9E37_79B9_7F4A_7C15) {
        // Avoid a zero state (SplitMix64 tolerates it, but keep a non-trivial start).
        self.state = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed
    }

    public mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
