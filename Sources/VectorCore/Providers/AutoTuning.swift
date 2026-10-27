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

    private struct TuningKey: Hashable { let dim: Int; let kind: KernelKind }
    private static let state = State(isDisabled: Config.isDisabled, printTuning: Config.printTuning, bias: Config.envBias)

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

    public static func get(dim: Int, kind: KernelKind) -> Tuning? { state.get(dim: dim, kind: kind) }

    public static func set(dim: Int, kind: KernelKind, tuning: Tuning) {
        state.set(dim: dim, kind: kind, tuning: tuning)
    }

    /// Calibrate synchronously outside the state lock, at most once concurrently per key.
    /// Recursive/concurrent same-key calls return an unpublished provisional default.
    /// A manual set wins publication. Throwing callbacks retire their reservation and
    /// propagate the same error without caching a default. Completed callback durations
    /// (including failures and overlapping/nested work) count individually toward the
    /// soft 20 ms admission cap; an admitted callback has no duration limit.
    public static func calibrateIfNeeded(
        dim: Int, kind: KernelKind, providerCores: Int,
        calibrator: () throws -> CalibrationProbes
    ) rethrows -> Tuning {
        try state.calibrateIfNeeded(dim: dim, kind: kind, providerCores: providerCores, calibrator: calibrator)
    }

    /// One lock owns all mutable cache, reservation and completed-duration state.
    /// Immutable configuration/clock are Sendable; callbacks, clock reads and output
    /// execute outside the lock. This invariant provides the unchecked conformance.
    internal final class State: @unchecked Sendable {
        private let lock = NSLock()
        private var cache: [TuningKey: Tuning] = [:]
        private var inFlight: Set<TuningKey> = []
        private var accumulatedCalibrationTimeMs: Double = 0
        private let isDisabled: Bool
        private let printTuning: Bool
        private let bias: ModeBias
        private let nowMilliseconds: @Sendable () -> Double

        /// The injected clock must be finite and monotonic; production uses uptime.
        internal init(isDisabled: Bool = false, printTuning: Bool = false, bias: ModeBias = .neutral,
                      nowMilliseconds: @escaping @Sendable () -> Double = { Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000 }) {
            self.isDisabled = isDisabled
            self.printTuning = printTuning
            self.bias = bias
            self.nowMilliseconds = nowMilliseconds
        }

        func get(dim: Int, kind: KernelKind) -> Tuning? {
            if isDisabled { return nil }
            let key = TuningKey(dim: dim, kind: kind)
            lock.lock(); defer { lock.unlock() }
            return cache[key]
        }

        func set(dim: Int, kind: KernelKind, tuning: Tuning) {
            let key = TuningKey(dim: dim, kind: kind)
            lock.lock(); cache[key] = tuning; lock.unlock()
            // A set deliberately leaves reservation ownership with the callback.
            if printTuning {
                print("[VectorCore.AutoTuning] Manually Set: \(dim)/\(kind.description) -> N*=\(tuning.breakEvenN), Chunk=\(tuning.minChunk)")
            }
        }

        func calibrateIfNeeded(dim: Int, kind: KernelKind, providerCores: Int,
                               calibrator: () throws -> CalibrationProbes) rethrows -> Tuning {
            let fallback = defaultTuning(dim: dim, bias: bias)
            guard dim >= 0, providerCores > 0, !isDisabled else { return fallback }
            let key = TuningKey(dim: dim, kind: kind)
            lock.lock()
            if let published = cache[key] { lock.unlock(); return published }
            if inFlight.contains(key) { lock.unlock(); return fallback }
            if accumulatedCalibrationTimeMs >= Config.MAX_TOTAL_CALIBRATION_TIME_MS {
                cache[key] = fallback
                lock.unlock()
                if printTuning { print("[VectorCore.AutoTuning] Calibration skipped for \(dim)/\(kind.description). Cumulative time cap exceeded.") }
                return fallback
            }
            inFlight.insert(key)
            lock.unlock()

            let start = nowMilliseconds()
            let probes: CalibrationProbes
            do { probes = try calibrator() }
            catch {
                let elapsed = nowMilliseconds() - start
                lock.lock()
                accumulatedCalibrationTimeMs += elapsed
                inFlight.remove(key)
                lock.unlock()
                throw error
            }
            // Stop timing before calculation: only completed callbacks are charged.
            let elapsed = nowMilliseconds() - start
            let measured = calculateTuning(dim: dim, probes: probes, providerCores: providerCores, bias: bias)
            lock.lock()
            let winner = cache[key] ?? measured
            cache[key] = winner
            accumulatedCalibrationTimeMs += elapsed
            let total = accumulatedCalibrationTimeMs
            inFlight.remove(key)
            lock.unlock()
            if printTuning { logTuning(dim: dim, kind: kind, tuning: winner, durationMs: elapsed, totalMs: total, probes: probes) }
            return winner
        }
    }

    // MARK: - Calculation Logic

    // swiftlint:disable identifier_name
    // Justification: Mathematical notation follows published parallel heuristic formulas:
    // P_eff = effective parallelism factor, Tp = parallel overhead, N_star = the batch size the formula selects
    // Using standard notation improves formula readability and matches academic literature
    private static func calculateTuning(dim: Int, probes: CalibrationProbes, providerCores: Int, bias: ModeBias) -> Tuning {
        let fallback = defaultTuning(dim: dim, bias: bias)
        guard probes.nsPerCandidateSeq.isFinite, probes.nsPerCandidateSeq > 0,
              probes.parallelOverheadNs.isFinite, probes.parallelOverheadNs >= 0,
              probes.effectiveParallelFactor.isFinite, probes.effectiveParallelFactor > 0 else { return fallback }
        let P_eff = probes.effectiveParallelFactor
        let Tp = probes.parallelOverheadNs
        let a = probes.nsPerCandidateSeq

        var N_star: Int
        if P_eff <= 1.01 || a < 1e-9 {
            N_star = Config.INEFFICIENT_THRESHOLD
        } else {
            let denom = a * (P_eff - 1.0)
            guard denom.isFinite else { return fallback }
            if denom < 1e-9 {
                N_star = Config.INEFFICIENT_THRESHOLD
            } else {
                let product = Tp * P_eff
                guard product.isFinite else { return fallback }
                let raw = ceil(product / denom)
                guard raw.isFinite else { return fallback }
                if raw >= Double(Config.INEFFICIENT_THRESHOLD) {
                    N_star = Config.INEFFICIENT_THRESHOLD
                } else {
                    guard let converted = nonnegativeInteger(raw) else { return fallback }
                    N_star = converted
                }
            }
        }

        if N_star < Config.INEFFICIENT_THRESHOLD {
            switch bias {
            case .favorSequential:
                guard let scaled = nonnegativeInteger(Double(N_star) * 1.5) else { return fallback }
                N_star = scaled
            case .favorParallel:
                guard let scaled = nonnegativeInteger(Double(N_star) * 0.66) else { return fallback }
                N_star = scaled
            case .neutral: break
            }
        }
        if N_star < Config.INEFFICIENT_THRESHOLD {
            N_star = max(Config.MIN_BREAKEVEN_N, min(Config.MAX_BREAKEVEN_N, N_star))
        }

        let referenceN = min(N_star, Config.MAX_BREAKEVEN_N)
        let targetTasks = Config.TARGET_TASKS_PER_CORE * Double(providerCores)
        guard targetTasks.isFinite, targetTasks > 0,
              var minChunk = nonnegativeInteger(ceil(Double(referenceN) / targetTasks)) else { return fallback }
        let scale = dim >= 1536 ? 2.0 : (dim >= 768 ? 1.25 : 1.0)
        guard let scaledChunk = nonnegativeInteger(Double(minChunk) * scale) else { return fallback }
        let (alignmentInput, overflow) = scaledChunk.addingReportingOverflow(63)
        guard !overflow else { return fallback }
        minChunk = alignmentInput & ~63
        minChunk = max(Config.MIN_CHUNK_SIZE, min(Config.MAX_CHUNK_SIZE, minChunk))

        return Tuning(breakEvenN: N_star, minChunk: minChunk, modeBias: bias)
    }
    // swiftlint:enable identifier_name

    // MARK: - Helpers

    private static func nonnegativeInteger(_ value: Double) -> Int? {
        // Int.max rounds upward in Double; the previous representable Double is safe.
        guard value.isFinite, value >= 0, value <= Double(Int.max).nextDown else { return nil }
        return Int(value)
    }

    private static func defaultTuning(dim: Int, bias: ModeBias) -> Tuning {
        let breakEvenN = 1000
        let minChunk = dim >= 1024 ? 512 : 256
        return Tuning(breakEvenN: breakEvenN, minChunk: minChunk, modeBias: bias)
    }

    private static func logTuning(dim: Int, kind: KernelKind, tuning: Tuning, durationMs: Double, totalMs: Double, probes: CalibrationProbes) {
        // swiftlint:disable line_length
        // Justification: Format string for diagnostic logging - breaking would obscure parameter alignment
        let message = String(
            format: "[VectorCore.AutoTuning] %d/%@: N*=%d, Chunk=%d, Bias=%@. (Time: %.2fms, Total: %.2fms | Probes: a=%.1f, Tp=%.0f, P_eff=%.2f)",
            dim, kind.description, tuning.breakEvenN, tuning.minChunk, String(describing: tuning.modeBias), durationMs, totalMs,
            probes.nsPerCandidateSeq, probes.parallelOverheadNs, probes.effectiveParallelFactor
        )
        // swiftlint:enable line_length
        print(message)
    }
}
