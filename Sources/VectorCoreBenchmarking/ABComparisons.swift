import Foundation

public struct BenchmarkReportError: Error, CustomStringConvertible {
    public let description: String
    public init(_ description: String) { self.description = description }
}

public struct ABReport: Sendable {
    public let comparisons: [ABComparison]
    public let missingSides: [String]
}

/// One pairing policy serves planning, JSON and pretty output.
///
/// Pairing comes only from catalog-declared `boundary.comparison` membership. Case-name patterns never
/// form pairs. A declared pair must agree on every boundary field except the family's allowed differences
/// (`ComparisonFamilies`), and members with different timing scopes are an integrity error.
public enum ABComparisons {
    private struct Key: Hashable {
        let family: String
        let group: String
    }
    private struct Member { let name: String; let boundary: MeasurementBoundary; let side: String }
    private struct Pair { let key: Key; let left: Member?; let right: Member? }

    private static func pairs(_ entries: [(name: String, boundary: MeasurementBoundary?)]) throws -> [Pair] {
        var seen = Set<String>()
        var groups: [Key: [String: Member]] = [:]
        for (name, boundary) in entries {
            guard seen.insert(name).inserted else { throw BenchmarkReportError("Duplicate case ID: \(name)") }
            guard let boundary, let membership = boundary.comparison else { continue }
            guard ComparisonFamilies.allowedDifferences(family: membership.family) != nil else {
                throw BenchmarkReportError("Undeclared comparison family \(membership.family): \(name)")
            }
            guard membership.side == "left" || membership.side == "right" else {
                throw BenchmarkReportError("Invalid comparison side \(membership.side): \(name)")
            }
            let key = Key(family: membership.family, group: membership.group)
            guard groups[key,
                default: [:]][membership.side] == nil else { throw BenchmarkReportError("Duplicate comparison side: \(name)") }
            groups[key, default: [:]][membership.side] = Member(name: name, boundary: boundary, side: membership.side)
        }
        return groups.map { Pair(key: $0.key, left: $0.value["left"], right: $0.value["right"]) }.sorted {
            if $0.key.family != $1.key.family { return $0.key.family < $1.key.family }
            let a = $0.left?.boundary ?? $0.right!.boundary, b = $1.left?.boundary ?? $1.right!.boundary
            if a.fixture.dimension != b.fixture.dimension { return a.fixture.dimension < b.fixture.dimension }
            if a.fixture.count != b.fixture.count { return a.fixture.count < b.fixture.count }
            return $0.key.group < $1.key.group
        }
    }

    /// Pre-execution planning (`--ab-only`): which declared sides lack a counterpart in the selection.
    public static func missingSides(boundaries: [(name: String, boundary: MeasurementBoundary)]) throws -> [String] {
        try missing(pairs(boundaries.map { ($0.name, Optional($0.boundary)) }))
    }

    private static func missing(_ pairs: [Pair]) -> [String] {
        pairs.compactMap { pair in
            guard pair.left == nil || pair.right == nil else { return nil }
            return "Missing \(pair.left == nil ? "left" : "right") comparison side for \(pair.left?.name ?? pair.right!.name)"
        }
    }

    public static func report(cases: [BenchCase]) throws -> ABReport {
        let pairs = try pairs(cases.map { ($0.name, $0.boundary) })
        let byName = Dictionary(uniqueKeysWithValues: cases.map { ($0.name, $0) })
        var comparisons: [ABComparison] = []
        for pair in pairs {
            guard let leftMember = pair.left, let rightMember = pair.right else { continue }
            let leftName = leftMember.name, rightName = rightMember.name
            let leftBoundary = leftMember.boundary, rightBoundary = rightMember.boundary
            guard leftBoundary.timingScope == rightBoundary.timingScope else {
                throw BenchmarkReportError("Cross-scope comparison rejected (\(leftBoundary.timingScope.rawValue) vs "
                    + "\(rightBoundary.timingScope.rawValue)): \(leftName), \(rightName)")
            }
            let allowed = ComparisonFamilies.allowedDifferences(family: pair.key.family)!.union(["comparison.side"])
            let disallowed = leftBoundary.differences(from: rightBoundary).filter { !allowed.contains($0) }
            guard disallowed.isEmpty else {
                throw BenchmarkReportError("Boundary mismatch in \(pair.key.family): \(leftName), \(rightName): "
                    + "\(disallowed.joined(separator: ","))")
            }
            let left = byName[leftName]!, right = byName[rightName]!
            guard left.unitCount > 0, left.unitCount == right.unitCount,
                  left.nsPerUnit.isFinite, left.nsPerUnit > 0,
                  right.nsPerUnit.isFinite, right.nsPerUnit > 0 else {
                throw BenchmarkReportError("Invalid comparison units/timing: \(leftName), \(rightName)")
            }
            let delta = (right.nsPerUnit - left.nsPerUnit) / right.nsPerUnit * 100
            guard delta.isFinite else { throw BenchmarkReportError("Nonfinite comparison delta: \(leftName)") }
            // Group format is "D|N|variantFamily|provider"; the variant is its third component.
            let components = pair.key.group.split(separator: "|", omittingEmptySubsequences: false).map(String.init)
            let variant = components.count == 4 ? components[2] : nil
            comparisons.append(ABComparison(kind: "batch", comparison: pair.key.family,
                dim: leftBoundary.fixture.dimension, n: leftBoundary.fixture.count, variant: variant, provider: leftBoundary.provider,
                leftName: leftName, rightName: rightName, leftNsPerUnit: left.nsPerUnit, rightNsPerUnit: right.nsPerUnit,
                deltaPercent: delta, leftBoundary: leftBoundary, rightBoundary: rightBoundary))
        }
        return ABReport(comparisons: comparisons, missingSides: missing(pairs))
    }
    public static func buildComparisons(cases: [BenchCase]) throws -> [ABComparison] { try report(cases: cases).comparisons }
}
