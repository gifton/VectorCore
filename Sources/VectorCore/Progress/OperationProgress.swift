// VectorCore - OperationProgress
// Shared progress reporting type across VSK packages

import Foundation

/// Represents progress of a long-running operation
public struct OperationProgress: Sendable, Equatable {
    public let current: Int
    public let total: Int
    public let phase: String
    public let message: String?
    public let timestamp: Date
    public let estimatedTimeRemaining: TimeInterval?

    public var fraction: Double {
        guard total > 0 else { return 0.0 }
        return Double(current) / Double(total)
    }

    public var percentage: Int {
        Int(fraction * 100)
    }

    public var isComplete: Bool {
        current >= total
    }

    public init(
        current: Int,
        total: Int,
        phase: String = "Processing",
        message: String? = nil,
        estimatedTimeRemaining: TimeInterval? = nil,
        timestamp: Date = Date()
    ) {
        self.current = max(0, current)
        self.total = max(0, total)
        self.phase = phase
        self.message = message
        self.estimatedTimeRemaining = estimatedTimeRemaining
        self.timestamp = timestamp
    }

    public static func started(total: Int, phase: String = "Starting") -> OperationProgress {
        OperationProgress(current: 0, total: total, phase: phase)
    }

    public static func completed(total: Int, phase: String = "Complete") -> OperationProgress {
        OperationProgress(current: total, total: total, phase: phase)
    }

    public static func indeterminate(current: Int, phase: String) -> OperationProgress {
        OperationProgress(current: current, total: 0, phase: phase)
    }
}

/// AsyncSequence wrapper for progress updates
public struct ProgressStream<Element: Sendable>: AsyncSequence, Sendable {
    public typealias AsyncIterator = Iterator

    private let stream: AsyncThrowingStream<(Element, OperationProgress), any Error>

    public init(_ stream: AsyncThrowingStream<(Element, OperationProgress), any Error>) {
        self.stream = stream
    }

    public func makeAsyncIterator() -> Iterator {
        Iterator(iterator: stream.makeAsyncIterator())
    }

    public struct Iterator: AsyncIteratorProtocol {
        var iterator: AsyncThrowingStream<(Element, OperationProgress), any Error>.AsyncIterator

        public mutating func next() async throws -> (Element, OperationProgress)? {
            try await iterator.next()
        }
    }
}
