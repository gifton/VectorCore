import Foundation

// MARK: - Definitions

public enum KernelKind: Hashable, Sendable, CustomStringConvertible {
    case euclid2, euclid, cosineFused, cosinePreNorm, dot, euclid2SoA
    case euclid2Mixed, cosineMixed
    case quantizedINT8, quantizedEuclid, quantizedCosine, quantizedDot

    public var description: String {
        switch self {
        case .euclid2: return "euclid2"
        case .euclid: return "euclid"
        case .cosineFused: return "cosineFused"
        case .cosinePreNorm: return "cosinePreNorm"
        case .dot: return "dot"
        case .euclid2SoA: return "euclid2SoA"
        case .euclid2Mixed: return "euclid2Mixed"
        case .cosineMixed: return "cosineMixed"
        case .quantizedINT8: return "quantizedINT8"
        case .quantizedEuclid: return "quantizedEuclid"
        case .quantizedCosine: return "quantizedCosine"
        case .quantizedDot: return "quantizedDot"
        }
    }
}

public enum ModeBias: Sendable { case favorSequential, neutral, favorParallel }

public struct Tuning: Sendable {
    public let breakEvenN: Int
    public let minChunk: Int
    public let modeBias: ModeBias
}

public struct CalibrationProbes {
    public let nsPerCandidateSeq: Double
    public let parallelOverheadNs: Double
    public let effectiveParallelFactor: Double
}

// MARK: - AutoTuning Implementation

public enum AutoTuning {

    // Thread-safe cache. NSLock is appropriate for low contention.
    private static let lock = NSLock()
    nonisolated(unsafe) private static var cache: [TuningKey: Tuning] = [:]

    // State for cumulative time cap safeguard (protected by lock)
    nonisolated(unsafe) private static var accumulatedCalibrationTimeMs: Double = 0.0

    private struct TuningKey: Hashable { let dim: Int; let kind: KernelKind }

    private struct Config {
        static let isDisabled = ProcessInfo.processInfo.environment["VC_AUTOTUNE"] == "0"
        static let printTuning = ProcessInfo.processInfo.environment["VC_AUTOTUNE_PRINT"] == "1"
        static let envBias: ModeBias = {
            switch ProcessInfo.processInfo.environment["VC_AUTOTUNE_BIAS"] {
            case "seq": return .favorSequential
            case "parallel": return .favorParallel
            default: return .neutral
            }
        }()

        static let MIN_BREAKEVEN_N = 200
        static let MAX_BREAKEVEN_N = 4000
        static let MIN_CHUNK_SIZE = 64
        static let MAX_CHUNK_SIZE = 4096
        static let TARGET_TASKS_PER_CORE = 3.0
        static let MAX_TOTAL_CALIBRATION_TIME_MS = 20.0
        static let INEFFICIENT_THRESHOLD = Int.max / 2
    }

    // MARK: - Public API

    public static func get(dim: Int, kind: KernelKind) -> Tuning? {
        if Config.isDisabled { return nil }
        let key = TuningKey(dim: dim, kind: kind)
        lock.lock(); defer { lock.unlock() }
        return cache[key]
    }

    public static func set(dim: Int, kind: KernelKind, tuning: Tuning) {
        let key = TuningKey(dim: dim, kind: kind)
        lock.lock(); cache[key] = tuning; lock.unlock()
        if Config.printTuning {
            print("[VectorCore.AutoTuning] Manually Set: \(dim)/\(kind.description) -> N*=\(tuning.breakEvenN), Chunk=\(tuning.minChunk)")
        }
    }

    public static func calibrateIfNeeded(
        dim: Int,
        kind: KernelKind,
        providerCores: Int,
        calibrator: () -> CalibrationProbes
    ) -> Tuning {
        if Config.isDisabled { return defaultTuning(dim: dim) }

        if let existing = get(dim: dim, kind: kind) { return existing }

        lock.lock(); defer { lock.unlock() }
        let key = TuningKey(dim: dim, kind: kind)
        if let cached = cache[key] { return cached }

        if accumulatedCalibrationTimeMs >= Config.MAX_TOTAL_CALIBRATION_TIME_MS {
            if Config.printTuning {
                print("[VectorCore.AutoTuning] Calibration skipped for \(dim)/\(kind.description). Cumulative time cap exceeded.")
            }
            let def = defaultTuning(dim: dim)
            cache[key] = def
            return def
        }

        let clock = ContinuousClock()
        let start = clock.now
        let probes = calibrator()
        let tuning = calculateTuning(dim: dim, probes: probes, providerCores: providerCores)
        let duration = start.duration(to: clock.now)
        let durationMs = duration.nanoseconds / 1_000_000.0
        accumulatedCalibrationTimeMs += durationMs
        cache[key] = tuning
        if Config.printTuning { logTuning(dim: dim, kind: kind, tuning: tuning, durationMs: durationMs, probes: probes) }
        return tuning
    }

    // MARK: - Calculation Logic

    private static func calculateTuning(dim: Int, probes: CalibrationProbes, providerCores: Int) -> Tuning {
        let P_eff = probes.effectiveParallelFactor
        let Tp = probes.parallelOverheadNs
        let a = probes.nsPerCandidateSeq

        var N_star: Int
        if P_eff <= 1.01 || a < 1e-9 {
            N_star = Config.INEFFICIENT_THRESHOLD
        } else {
            let denom = a * (P_eff - 1.0)
            if denom < 1e-9 {
                N_star = Config.INEFFICIENT_THRESHOLD
            } else {
                let raw = ceil((Tp * P_eff) / denom)
                N_star = Int(min(raw, Double(Config.INEFFICIENT_THRESHOLD)))
            }
        }

        let bias = Config.envBias
        if N_star < Config.INEFFICIENT_THRESHOLD {
            switch bias {
            case .favorSequential: N_star = Int(Double(N_star) * 1.5)
            case .favorParallel: N_star = Int(Double(N_star) * 0.66)
            case .neutral: break
            }
        }
        if N_star < Config.INEFFICIENT_THRESHOLD {
            N_star = max(Config.MIN_BREAKEVEN_N, min(Config.MAX_BREAKEVEN_N, N_star))
        }

        let referenceN = min(N_star, Config.MAX_BREAKEVEN_N)
        let targetTasks = Config.TARGET_TASKS_PER_CORE * Double(providerCores)
        var minChunk = Int(ceil(Double(referenceN) / targetTasks))
        if dim >= 1536 { minChunk = Int(Double(minChunk) * 2.0) } else if dim >= 768 { minChunk = Int(Double(minChunk) * 1.25) }
        minChunk = alignTo64(minChunk)
        minChunk = max(Config.MIN_CHUNK_SIZE, min(Config.MAX_CHUNK_SIZE, minChunk))

        return Tuning(breakEvenN: N_star, minChunk: minChunk, modeBias: bias)
    }

    // MARK: - Helpers

    private static func alignTo64(_ value: Int) -> Int { (max(0, value) + 63) & ~63 }

    private static func defaultTuning(dim: Int) -> Tuning {
        let breakEvenN = 1000
        let minChunk = dim >= 1024 ? 512 : 256
        return Tuning(breakEvenN: breakEvenN, minChunk: minChunk, modeBias: Config.envBias)
    }

    private static func logTuning(dim: Int, kind: KernelKind, tuning: Tuning, durationMs: Double, probes: CalibrationProbes) {
        let message = String(
            format: "[VectorCore.AutoTuning] %d/%@: N*=%d, Chunk=%d, Bias=%@. (Time: %.2fms, Total: %.2fms | Probes: a=%.1f, Tp=%.0f, P_eff=%.2f)",
            dim, kind.description, tuning.breakEvenN, tuning.minChunk, String(describing: tuning.modeBias), durationMs, accumulatedCalibrationTimeMs,
            probes.nsPerCandidateSeq, probes.parallelOverheadNs, probes.effectiveParallelFactor
        )
        print(message)
    }
}
