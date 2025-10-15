import Foundation

/// Utilities for building A/B performance comparisons between benchmark variants
///
/// Generates structured comparisons for:
/// - Euclidean distance: euclidean2 (squared) vs euclidean (sqrt)
/// - Cosine similarity: preNorm vs fused variants
public enum ABComparisons {
    /// Build all A/B comparisons from benchmark cases
    ///
    /// Groups cases by dimension, N, variant, and provider, then compares:
    /// 1. batch euclidean2 vs euclidean (squared distance optimization)
    /// 2. batch cosine preNorm vs fused (normalization strategy)
    ///
    /// - Parameter cases: Array of benchmark cases to analyze
    /// - Returns: Sorted array of A/B comparisons
    public static func buildComparisons(cases: [BenchCase]) -> [ABComparison] {
        var out: [ABComparison] = []

        // Helper to compute ns/unit from case
        func nsPerUnit(_ c: BenchCase) -> Double { c.nsPerUnit }

        // batch euclidean2 vs euclidean
        struct Key: Hashable { let dim: Int; let n: Int; let variant: String?; let provider: String? }
        var byKey: [Key: [String: BenchCase]] = [:]
        for c in cases {
            let p = c.params
            guard p.kind == "batch", let dim = p.dim, let n = p.n else { continue }
            let key = Key(dim: dim, n: n, variant: p.variant, provider: p.provider)
            if let m = p.metric { byKey[key, default: [:]][m] = c }
        }
        for (key, dict) in byKey {
            if let rEuclid2 = dict["euclidean2"], let rEuclid = dict["euclidean"] {
                // Match console summary semantics: label "euclidean2 vs euclidean",
                // value order = euclid2 (left) vs euclid (right),
                // delta = (euclid - euclid2)/euclid * 100
                let left = nsPerUnit(rEuclid2)
                let right = nsPerUnit(rEuclid)
                let delta = (right - left) / right * 100.0
                out.append(ABComparison(
                    kind: "batch",
                    comparison: "euclidean2_vs_euclidean",
                    dim: key.dim,
                    n: key.n,
                    variant: key.variant,
                    provider: key.provider,
                    leftName: rEuclid2.name,
                    rightName: rEuclid.name,
                    leftNsPerUnit: left,
                    rightNsPerUnit: right,
                    deltaPercent: delta
                ))
            }
        }

        // batch cosine preNorm vs fused (optimized)
        var cosByKey: [Key: [String: BenchCase]] = [:]
        for c in cases {
            let p = c.params
            guard p.kind == "batch", p.metric == "cosine", let dim = p.dim, let n = p.n else { continue }
            let v = (p.variant ?? "").lowercased()
            let key = Key(dim: dim, n: n, variant: p.variant, provider: p.provider)
            if v.contains("fused") { cosByKey[key, default: [:]]["fused"] = c }
            else if v.contains("prenorm") { cosByKey[key, default: [:]]["prenorm"] = c }
        }
        for (key, dict) in cosByKey {
            if let fused = dict["fused"], let prenorm = dict["prenorm"] {
                // Match console: label "preNorm vs fused" but we encode
                // comparison key as "cosine_prenorm_vs_fused".
                // value order = preNorm (left) vs fused (right),
                // delta = (fused - preNorm)/fused * 100
                let left = nsPerUnit(prenorm)
                let right = nsPerUnit(fused)
                let delta = (right - left) / right * 100.0
                out.append(ABComparison(
                    kind: "batch",
                    comparison: "cosine_prenorm_vs_fused",
                    dim: key.dim,
                    n: key.n,
                    variant: key.variant,
                    provider: key.provider,
                    leftName: prenorm.name,
                    rightName: fused.name,
                    leftNsPerUnit: left,
                    rightNsPerUnit: right,
                    deltaPercent: delta
                ))
            }
        }

        return out.sorted(by: { a, b in
            if a.comparison != b.comparison { return a.comparison < b.comparison }
            if a.dim != b.dim { return a.dim < b.dim }
            if a.n != b.n { return a.n < b.n }
            return (a.variant ?? "").lexicographicallyPrecedes(b.variant ?? "")
        })
    }
}
