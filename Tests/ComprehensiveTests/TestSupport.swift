import Foundation
import Testing
@testable import VectorCore

// Shared helpers for the ComprehensiveTests suite can go here.

/// Absolute-tolerance comparison: `abs(a - b) <= tol`.
///
/// The default `tol` of 1e-5 is an ABSOLUTE tolerance, appropriate only for |values| <= ~100 or
/// for results that are exact by construction. Float spacing is ~7.6e-6 at 64...128 and ~1.5e-5
/// at 128...256, so above ~100 the default is at or below one ULP and a correct but differently
/// rounded result fails (or only exact-integer cases pass by luck). Callers comparing larger
/// magnitudes must pass an explicit relative or ULP-derived `tol:`.
@inline(__always)
func approxEqual(_ a: Float, _ b: Float, tol: Float = 1e-5) -> Bool {
    return abs(a - b) <= tol
}
