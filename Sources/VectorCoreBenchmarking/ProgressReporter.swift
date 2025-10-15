import Foundation

/// Progress event types for benchmark execution
public enum ProgressEvent: Codable, Sendable {
    /// Suite execution started
    case suiteStart(suite: String, timestamp: Double)

    /// Individual case started
    case caseStart(suite: String, caseName: String, index: Int, total: Int)

    /// Individual case completed
    case caseComplete(suite: String, caseName: String, index: Int, total: Int, durationMs: Double)

    /// Suite execution completed
    case suiteComplete(suite: String, cases: Int, durationMs: Double)

    // Codable conformance
    private enum CodingKeys: String, CodingKey {
        case type, suite, timestamp, caseName = "case", index, total, cases, durationMs = "duration_ms"
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)

        switch type {
        case "suite_start":
            let suite = try container.decode(String.self, forKey: .suite)
            let timestamp = try container.decode(Double.self, forKey: .timestamp)
            self = .suiteStart(suite: suite, timestamp: timestamp)
        case "case_start":
            let suite = try container.decode(String.self, forKey: .suite)
            let caseName = try container.decode(String.self, forKey: .caseName)
            let index = try container.decode(Int.self, forKey: .index)
            let total = try container.decode(Int.self, forKey: .total)
            self = .caseStart(suite: suite, caseName: caseName, index: index, total: total)
        case "case_complete":
            let suite = try container.decode(String.self, forKey: .suite)
            let caseName = try container.decode(String.self, forKey: .caseName)
            let index = try container.decode(Int.self, forKey: .index)
            let total = try container.decode(Int.self, forKey: .total)
            let durationMs = try container.decode(Double.self, forKey: .durationMs)
            self = .caseComplete(suite: suite, caseName: caseName, index: index, total: total, durationMs: durationMs)
        case "suite_complete":
            let suite = try container.decode(String.self, forKey: .suite)
            let cases = try container.decode(Int.self, forKey: .cases)
            let durationMs = try container.decode(Double.self, forKey: .durationMs)
            self = .suiteComplete(suite: suite, cases: cases, durationMs: durationMs)
        default:
            throw DecodingError.dataCorruptedError(forKey: .type, in: container, debugDescription: "Unknown event type: \(type)")
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        switch self {
        case .suiteStart(let suite, let timestamp):
            try container.encode("suite_start", forKey: .type)
            try container.encode(suite, forKey: .suite)
            try container.encode(timestamp, forKey: .timestamp)

        case .caseStart(let suite, let caseName, let index, let total):
            try container.encode("case_start", forKey: .type)
            try container.encode(suite, forKey: .suite)
            try container.encode(caseName, forKey: .caseName)
            try container.encode(index, forKey: .index)
            try container.encode(total, forKey: .total)

        case .caseComplete(let suite, let caseName, let index, let total, let durationMs):
            try container.encode("case_complete", forKey: .type)
            try container.encode(suite, forKey: .suite)
            try container.encode(caseName, forKey: .caseName)
            try container.encode(index, forKey: .index)
            try container.encode(total, forKey: .total)
            try container.encode(durationMs, forKey: .durationMs)

        case .suiteComplete(let suite, let cases, let durationMs):
            try container.encode("suite_complete", forKey: .type)
            try container.encode(suite, forKey: .suite)
            try container.encode(cases, forKey: .cases)
            try container.encode(durationMs, forKey: .durationMs)
        }
    }
}

/// Progress output format
public enum ProgressFormat: String, Sendable {
    /// No progress output
    case none
    /// Plain text progress messages
    case text
    /// JSON-lines format (one event per line)
    case json
}

/// Progress reporter for structured benchmark progress events
///
/// Emits progress events to stderr in JSON-lines format for consumption by UI tools.
/// Does not interfere with stdout results output.
public struct ProgressReporter: Sendable {
    public let format: ProgressFormat

    public init(format: ProgressFormat) {
        self.format = format
    }

    /// Emit a progress event
    public func emit(_ event: ProgressEvent) {
        switch format {
        case .none:
            return
        case .text:
            emitText(event)
        case .json:
            emitJSON(event)
        }
    }

    private func emitText(_ event: ProgressEvent) {
        let message: String
        switch event {
        case .suiteStart(let suite, _):
            message = "[PROGRESS] Starting suite: \(suite)"
        case .caseStart(let suite, let caseName, let index, let total):
            let percent = Double(index) / Double(total) * 100.0
            message = "[PROGRESS] \(suite): \(caseName) (\(index+1)/\(total), \(String(format: "%.0f", percent))%)"
        case .caseComplete(let suite, let caseName, let index, let total, let durationMs):
            let percent = Double(index + 1) / Double(total) * 100.0
            message = "[PROGRESS] \(suite): \(caseName) completed in \(String(format: "%.1f", durationMs))ms (\(index+1)/\(total), \(String(format: "%.0f", percent))%)"
        case .suiteComplete(let suite, let cases, let durationMs):
            message = "[PROGRESS] Completed suite: \(suite) (\(cases) cases in \(String(format: "%.1f", durationMs))ms)"
        }
        fputs("\(message)\n", stderr)
        fflush(stderr)
    }

    private func emitJSON(_ event: ProgressEvent) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [] // Compact JSON on single line

        if let data = try? encoder.encode(event),
           let jsonString = String(data: data, encoding: .utf8) {
            fputs("\(jsonString)\n", stderr)
            fflush(stderr)
        }
    }
}

// MARK: - Convenience Extensions

extension ProgressReporter {
    /// Report suite start
    public func suiteStarted(_ suite: String) {
        emit(.suiteStart(suite: suite, timestamp: Date().timeIntervalSince1970))
    }

    /// Report case start
    public func caseStarted(suite: String, name: String, index: Int, total: Int) {
        emit(.caseStart(suite: suite, caseName: name, index: index, total: total))
    }

    /// Report case completion
    public func caseCompleted(suite: String, name: String, index: Int, total: Int, durationMs: Double) {
        emit(.caseComplete(suite: suite, caseName: name, index: index, total: total, durationMs: durationMs))
    }

    /// Report suite completion
    public func suiteCompleted(suite: String, cases: Int, durationMs: Double) {
        emit(.suiteComplete(suite: suite, cases: cases, durationMs: durationMs))
    }
}
