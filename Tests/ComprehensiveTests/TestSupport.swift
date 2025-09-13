import Foundation
import Testing
@testable import VectorCore

// Shared helpers for the ComprehensiveTests suite can go here.

@inline(__always)
func approxEqual(_ a: Float, _ b: Float, tol: Float = 1e-5) -> Bool {
    return abs(a - b) <= tol
}
