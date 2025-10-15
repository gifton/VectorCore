import Foundation

/// Centralized, deterministic seeds for benchmark inputs.
///
/// Suites should use these helpers instead of hard-coding seeds.
/// A global `runSeed` (CLI `--run-seed` or env `VC_RUN_SEED`) can offset
/// all base seeds to allow controlled variation across runs.
public enum InputSeeds {
    @inline(__always)
    private static func mix(_ base: UInt64, with runSeed: UInt64) -> UInt64 {
        // Simple, deterministic mixing: add and fold. Avoids zeroing when runSeed == 0.
        return base &+ (runSeed &* 0x9E3779B97F4A7C15)
    }

    public static func dot(dim: Int, runSeed: UInt64) -> (a: UInt64, b: UInt64) {
        let pair: (UInt64, UInt64)
        switch dim {
        case 512: pair = (515_201, 515_202)
        case 768: pair = (768_001, 768_002)
        case 1536: pair = (1_536_001, 1_536_002)
        default:
            // Fallback but stable
            pair = (UInt64(dim) &* 1_000 &+ 1, UInt64(dim) &* 1_000 &+ 2)
        }
        return (mix(pair.0, with: runSeed), mix(pair.1, with: runSeed))
    }

    public static func distance(dim: Int, runSeed: UInt64) -> (a: UInt64, b: UInt64) {
        let pair: (UInt64, UInt64)
        switch dim {
        case 512: pair = (615_201, 615_202)
        case 768: pair = (768_101, 768_102)
        case 1536: pair = (1_536_101, 1_536_102)
        default:
            pair = (UInt64(dim) &* 1_000 &+ 101, UInt64(dim) &* 1_000 &+ 102)
        }
        return (mix(pair.0, with: runSeed), mix(pair.1, with: runSeed))
    }

    public static func batch(dim: Int, n: Int, runSeed: UInt64) -> (query: UInt64, baseCandidates: UInt64) {
        // Query base seeds mirror prior suite constants.
        let qBase: UInt64
        switch dim {
        case 512: qBase = 512_777
        case 768: qBase = 768_777
        case 1536: qBase = 1_536_777
        default: qBase = UInt64(dim) &* 1_000 &+ 777
        }
        // Candidates were previously seeded from (dim-specific + N)
        let cBase: UInt64
        switch dim {
        case 512: cBase = 512_100_000
        case 768: cBase = 768_100_000
        case 1536: cBase = 1_536_100_000
        default: cBase = UInt64(dim) &* 100_000
        }
        let querySeed = mix(qBase, with: runSeed)
        let baseSeed = mix(cBase &+ UInt64(n), with: runSeed)
        return (querySeed, baseSeed)
    }
}

