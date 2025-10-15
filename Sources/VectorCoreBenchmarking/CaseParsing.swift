import Foundation

/// Utilities for parsing structured information from benchmark case names
///
/// Case naming conventions:
/// - dot: `dot.<dim>.<variant>`
/// - dist: `dist.<metric>.<dim>.<variant>`
/// - normalize: `normalize.<status>.<dim>.<variant>`
/// - batch: `batch.<metric>.<dim>.N<count>.<variant>[.<provider>]`
/// - memory: `mem.<operation>.<variant>.<dim>`
public enum CaseParsing {
    /// Parse structured parameters from a benchmark case name
    public static func parse(name: String) -> CaseParams {
        var p = CaseParams()
        let parts = name.split(separator: ".").map(String.init)
        guard !parts.isEmpty else { return p }
        p.kind = parts.first

        switch p.kind {
        case "dot":
            if parts.count >= 3 {
                p.dim = Int(parts[1])
                p.variant = parts[2]
            }
        case "dist":
            if parts.count >= 4 {
                p.metric = parts[1]
                p.dim = Int(parts[2])
                p.variant = parts[3]
            }
        case "normalize":
            if parts.count >= 4 {
                p.status = parts[1]
                p.dim = Int(parts[2])
                p.variant = parts[3]
            }
        case "batch":
            // canonical: batch.<metric>.<dim>.N<count>.<variant>[.<mode>]
            if parts.count >= 5 {
                p.metric = parts[1]
                p.dim = Int(parts[2])
                if parts[3].hasPrefix("N") { p.n = Int(parts[3].dropFirst()) }
                p.variant = parts[4]
                if parts.count >= 6 { p.provider = parts[5] }
            }
        case "mem", "memory":
            // e.g., mem.alloc.aligned.<dim> or mem.copy.<dim>
            if parts.count >= 3 {
                p.metric = parts[1]
                // last component is often dim
                if let last = parts.last, let d = Int(last) { p.dim = d }
                // variant is any middle components beyond metric and before dim
                if parts.count > 3 {
                    let mid = parts[2..<(parts.count - 1)]
                    if !mid.isEmpty { p.variant = mid.joined(separator: "-") }
                }
            }
        default:
            break
        }
        return p
    }
}
