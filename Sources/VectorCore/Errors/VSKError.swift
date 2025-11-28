// VectorCore - VSKError Protocol
// Unified error protocol for all VSK packages

import Foundation

/// Protocol for errors across VSK package family
public protocol VSKError: Error, Sendable, CustomStringConvertible {
    var errorCode: Int { get }
    var domain: String { get }
    var isRecoverable: Bool { get }
    var recoverySuggestion: String? { get }
    var underlyingError: (any Error)? { get }
    var context: ErrorContext { get }
}

public extension VSKError {
    var description: String {
        var desc = "[\(domain):\(errorCode)] \(localizedDescription)"
        if let suggestion = recoverySuggestion {
            desc += " Suggestion: \(suggestion)"
        }
        return desc
    }

    var recoverySuggestion: String? { nil }
    var underlyingError: (any Error)? { nil }
}

/// Standard error code ranges for VSK packages
public enum VSKErrorCodeRange {
    public static let vectorCore = 1000..<2000
    public static let embedKit = 2000..<3000
    public static let vectorIndex = 3000..<4000
    public static let vectorAccelerate = 4000..<5000
}
