import Foundation

// MARK: - Measurement boundary identity (V04)
//
// A boundary is the structured, per-case statement of what one timed invocation does. It is supplied
// by the case catalog that executes the case (never by a second list), is carried in JSON/pretty/CSV
// output, and is part of within-run pairing and cross-run comparison identity
// (`testRepresentativeCaseBoundaryAllFamilies`, `testReportRequiresBoundaryIdentity`,
// `testCrossScopeNeverPairsAndUndeclaredNamesDoNotPair`).
//
// Scope of the claims made here (unverified contract, no benchmark evidence implied):
// - A boundary describes the benchmark body as written. It does not prove what the optimizer emitted.
// - Equal boundaries mean equal declared work, not equal speed.

/// Whether a piece of work happens inside the timed invocation.
///
/// `unverified` means not established for this case ID (no applicable measurement, or a measurement that cannot be
/// applied, e.g. the pair constraint: both sides of a declared A/B pair carry the same inclusions). It is a declared gap,
/// never a claim in either direction. Values are resolved per case ID ("Measured resolutions" in
/// Docs/Benchmark_Measurement_Boundaries.md); `testAllocationFlagsDoNotOverclaim` pins the resolved IDs, the unprobed
/// neighbours and the pair case.
public enum Inclusion: String, Codable, Sendable, CaseIterable {
    case included
    case excluded
    case notApplicable
    case unverified
}

/// Top-level timing scope. Different scopes never form an automatic pair (`testCrossScopeNeverPairsAndUndeclaredNamesDoNotPair`).
public enum TimingScope: String, Codable, Sendable, CaseIterable {
    /// Inputs (and any route-specific preparation) are built before timing and reused by every invocation.
    case reusedInput = "reused-input"
    /// Every invocation also performs the route's input preparation and creates its output.
    case setupInclusive = "setup-inclusive"
    /// Only the output consumer runs, on a precomputed output.
    case consumerOnly = "consumer-only"
    /// Test-injected fixture results that did not come from a production catalog.
    /// `testFixtureInjectedScopeIsRejectedInProductionRegistry` pins `injectedScopeProblem`: a production registry that
    /// contains a case of this scope is rejected, and an injected test registry accepts it. The record carries no comparison
    /// membership, so it never forms a within-run A/B pair. `Scripts/compare_benchmarks.swift` accepts this scope in a report
    /// (read from its scope list; not verified by a test here).
    case fixtureInjected = "fixture-injected"
}

/// Explicit inclusion flags. Each preparation or allocation step is counted under exactly one flag (an authoring rule for
/// the boundary records, not verified by a test).
public struct BoundaryInclusions: Codable, Sendable, Equatable {
    /// Constructing the immutable input fixture.
    public var inputConstruction: Inclusion
    /// Preparation that the benchmark body performs for the route (transposition, FP16 conversion, normalization).
    public var inputPreparation: Inclusion
    /// Packing or transposition performed inside the public API call itself.
    public var routeInternalPacking: Inclusion
    /// Allocation of a caller-owned output buffer.
    public var callerOutputAllocation: Inclusion
    /// Provider, API or body intermediate allocation, recorded separately from the caller output.
    public var routeInternalAllocation: Inclusion
    public var providerScheduling: Inclusion
    public var providerObservation: Inclusion
    public var kernelWork: Inclusion
    public var outputConsumption: Inclusion
    public var correctnessValidation: Inclusion

    public init(inputConstruction: Inclusion, inputPreparation: Inclusion, routeInternalPacking: Inclusion,
                callerOutputAllocation: Inclusion, routeInternalAllocation: Inclusion,
                providerScheduling: Inclusion, providerObservation: Inclusion, kernelWork: Inclusion,
                outputConsumption: Inclusion, correctnessValidation: Inclusion) {
        self.inputConstruction = inputConstruction
        self.inputPreparation = inputPreparation
        self.routeInternalPacking = routeInternalPacking
        self.callerOutputAllocation = callerOutputAllocation
        self.routeInternalAllocation = routeInternalAllocation
        self.providerScheduling = providerScheduling
        self.providerObservation = providerObservation
        self.kernelWork = kernelWork
        self.outputConsumption = outputConsumption
        self.correctnessValidation = correctnessValidation
    }
}

/// Identity of the input fixture. Changes whenever the generated inputs or their preparation change.
public struct FixtureIdentity: Codable, Sendable, Equatable {
    public var id: String
    public var version: String
    public var dimension: Int
    public var count: Int
    /// Decimal UInt64 seeds by role (for example `query`, `candidateBase`, `a`, `b`).
    public var seeds: [String: String]
    public var values: String
    /// Preparation applied to the fixture before timing (`none` when the timed route receives it unchanged).
    public var preparation: String

    public init(id: String, version: String, dimension: Int, count: Int, seeds: [String: String],
                values: String, preparation: String) {
        self.id = id
        self.version = version
        self.dimension = dimension
        self.count = count
        self.seeds = seeds
        self.values = values
        self.preparation = preparation
    }
}

/// Declared membership in an A/B comparison family. Pairing comes only from this declaration.
public struct ComparisonMembership: Codable, Sendable, Equatable {
    public var family: String
    /// `left` or `right`.
    public var side: String
    /// Key shared by exactly one left and one right member (`testIndependentPairingAndIntegrity`).
    public var group: String

    public init(family: String, side: String, group: String) {
        self.family = family
        self.side = side
        self.group = group
    }
}

public struct MeasurementBoundary: Codable, Sendable, Equatable {
    public static let currentVersion = "V04-boundary-1"

    public var version: String
    public var timingScope: TimingScope
    public var operation: String
    public var route: String
    public var provider: String?
    /// `candidate`, `produced-entry`, `invocation` or `element`.
    public var workUnit: String
    public var unitsPerInvocation: Int
    public var inputRepresentation: String
    public var outputRepresentation: String
    public var fixture: FixtureIdentity
    public var inclusions: BoundaryInclusions
    /// Identifier of the output consumer (see Docs/Benchmark_Measurement_Boundaries.md).
    public var consumer: String
    public var notes: [String]
    public var comparison: ComparisonMembership?
    /// Recorded outcome of the existing default-eligibility predicate; present only where it is meaningful.
    public var defaultEligible: Bool?

    public init(version: String = MeasurementBoundary.currentVersion, timingScope: TimingScope, operation: String,
                route: String, provider: String?, workUnit: String, unitsPerInvocation: Int,
                inputRepresentation: String, outputRepresentation: String, fixture: FixtureIdentity,
                inclusions: BoundaryInclusions, consumer: String, notes: [String] = [],
                comparison: ComparisonMembership? = nil, defaultEligible: Bool? = nil) {
        self.version = version
        self.timingScope = timingScope
        self.operation = operation
        self.route = route
        self.provider = provider
        self.workUnit = workUnit
        self.unitsPerInvocation = unitsPerInvocation
        self.inputRepresentation = inputRepresentation
        self.outputRepresentation = outputRepresentation
        self.fixture = fixture
        self.inclusions = inclusions
        self.consumer = consumer
        self.notes = notes
        self.comparison = comparison
        self.defaultEligible = defaultEligible
    }

    /// Stable single-line JSON with sorted keys. This is the identity string used for comparison and CSV.
    /// Encoding a boundary cannot fail for valid values; a failure is a programming error and traps rather than
    /// returning an empty identity that would compare equal to another failed encoding.
    public func canonicalJSON() -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard let data = try? encoder.encode(self) else { preconditionFailure("MeasurementBoundary failed to encode") }
        guard let text = String(bytes: data, encoding: .utf8) else { preconditionFailure("MeasurementBoundary JSON is not UTF-8") }
        return text
    }

    /// Flattens to `path -> canonical JSON of the leaf` so differences can be named by field path.
    public func flattened() -> [String: String] {
        guard let data = try? JSONEncoder().encode(self),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            preconditionFailure("MeasurementBoundary failed to encode")
        }
        return MeasurementBoundary.flatten(object)
    }

    static func flatten(_ object: [String: Any], prefix: String = "") -> [String: String] {
        var out: [String: String] = [:]
        for (key, value) in object {
            if let nested = value as? [String: Any] {
                for (path, leaf) in flatten(nested, prefix: prefix + key + ".") { out[path] = leaf }
            } else if let data = try? JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed, .sortedKeys]) {
                out[prefix + key] = String(bytes: data, encoding: .utf8) ?? ""
            } else {
                preconditionFailure("Boundary leaf \(key) is not representable as JSON")
            }
        }
        return out
    }

    /// Sorted field paths whose values differ (a path present on one side only counts as different).
    public func differences(from other: MeasurementBoundary) -> [String] {
        let a = flattened(), b = other.flattened()
        return Set(a.keys).union(b.keys).filter { a[$0] != b[$0] }.sorted()
    }

    /// Human-readable description of the input preparation, derived from the declared boundary (never hard-coded).
    public var preparationDescription: String {
        switch inclusions.inputPreparation {
        case .excluded: return "\(fixture.preparation) (outside timing)"
        case .included: return "\(fixture.preparation) (inside timing)"
        case .notApplicable: return "no separate input preparation"
        case .unverified: return "input preparation unverified"
        }
    }

    /// Structural validity: a boundary that fails this is malformed and never usable as identity
    /// (`testReportRequiresBoundaryIdentity`).
    public func validationError() -> String? {
        if version.isEmpty { return "empty boundary version" }
        if operation.isEmpty || route.isEmpty || workUnit.isEmpty
            || consumer.isEmpty { return "empty operation, route, workUnit or consumer" }
        if unitsPerInvocation <= 0 { return "unitsPerInvocation must be positive" }
        if inputRepresentation.isEmpty || outputRepresentation.isEmpty { return "empty input or output representation" }
        if fixture.id.isEmpty || fixture.version.isEmpty { return "empty fixture identity" }
        return nil
    }
}

/// A selected case whose route does not exist. Reported, never timed, never counted as tested
/// (`testUnavailableEntriesAreRecordedNeverInvoked`).
public struct UnavailableCase: Codable, Sendable, Equatable {
    public let name: String
    public let boundary: MeasurementBoundary
    public let reason: String

    public init(name: String, boundary: MeasurementBoundary, reason: String) {
        self.name = name
        self.boundary = boundary
        self.reason = reason
    }
}

/// Family-specific allowed boundary differences for the two declared A/B families.
public enum ComparisonFamilies {
    public static let euclideanSquaredVersusEuclidean = "euclidean2_vs_euclidean"
    public static let cosinePreNormVersusFused = "cosine_prenorm_vs_fused"

    /// Field paths that may differ between the two sides (in addition to `comparison.side`).
    public static func allowedDifferences(family: String) -> Set<String>? {
        switch family {
        case euclideanSquaredVersusEuclidean:
            return ["operation", "route"]
        case cosinePreNormVersusFused:
            return ["operation", "route", "fixture.preparation", "inclusions.inputPreparation"]
        default:
            return nil
        }
    }

    /// The declared purpose of a family, returned as runtime text; it is a description, not a measurement. Not verified by
    /// a test here: the returned text for the cosine family describes the preNorm and fused batch routes in
    /// `Suites/BatchBench.swift`.
    public static func purpose(family: String) -> String {
        switch family {
        case euclideanSquaredVersusEuclidean:
            return "Compares inclusion of sqrt. The outputs differ numerically (squared versus unsquared distance); inputs, provider "
                + "and consumer are identical."
        case cosinePreNormVersusFused:
            return "Prepared-input strategy comparison. preNorm normalization is excluded from timing; fused norms are computed "
                + "inside the kernel. This is not an end-to-end strategy-speed claim."
        default:
            return "Undeclared family"
        }
    }
}
