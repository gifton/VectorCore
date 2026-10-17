#!/usr/bin/env swift
import Foundation
import CoreFoundation

struct ComparisonError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}
struct CLI {
    var baseline = "", current = "", metric = "median"
    var threshold = 10.0
    var perUnit = false, showImprovements = false, allowMissing = false, allowIncompatible = false
    var includes: [String] = []
    init(arguments: [String]) throws {
        var iterator = arguments.makeIterator()
        func value(_ key: String) throws -> String {
            guard let value = iterator.next(), !value.hasPrefix("--") else { throw ComparisonError("Missing value: \(key)") }
            return value
        }
        while let key = iterator.next() {
            switch key {
            case "--baseline": baseline = try value(key)
            case "--current": current = try value(key)
            case "--metric": metric = try value(key)
            case "--threshold":
                guard let number = Double(try value(key)), number.isFinite, number >= 0 else { throw ComparisonError("Threshold must be finite and nonnegative") }
                threshold = number
            case "--per-unit": perUnit = true
            case "--include": includes.append(try value(key))
            case "--show-improvements": showImprovements = true
            case "--allow-missing": allowMissing = true
            case "--strict": allowMissing = false
            case "--allow-incompatible": allowIncompatible = true
            default: throw ComparisonError("Unknown argument: \(key)")
            }
        }
        guard !baseline.isEmpty, !current.isEmpty, ["median", "p90", "mean", "nsPerOp"].contains(metric) else { throw ComparisonError("Required baseline/current paths and metric median|p90|mean|nsPerOp") }
    }
}
func number(_ value: Any?, field: String, positive: Bool = true) throws -> Double {
    guard let value = value as? NSNumber, CFGetTypeID(value) != CFBooleanGetTypeID(), value.doubleValue.isFinite,
          positive ? value.doubleValue > 0 : value.doubleValue >= 0 else { throw ComparisonError("Invalid or missing numeric field: \(field)") }
    return value.doubleValue
}
struct Row {
    let name: String
    let units: Double
    let metric: Double
    /// Per-case measurement boundary as a JSON object (nil when the result carries none).
    let boundary: [String: Any]?
    /// Set when a boundary is present but is not a structurally valid object.
    let boundaryProblem: String?
}
let requiredBoundaryKeys = ["version", "timingScope", "operation", "route", "workUnit", "unitsPerInvocation", "inputRepresentation",
                            "outputRepresentation", "fixture", "inclusions", "consumer"]
let requiredFixtureKeys = ["id", "version", "dimension", "count", "seeds", "values", "preparation"]
let inclusionKeys = ["inputConstruction", "inputPreparation", "routeInternalPacking", "callerOutputAllocation", "routeInternalAllocation",
                     "providerScheduling", "providerObservation", "kernelWork", "outputConsumption", "correctnessValidation"]
let inclusionValues: Set<String> = ["included", "excluded", "notApplicable", "unverified"]
let timingScopes: Set<String> = ["reused-input", "setup-inclusive", "consumer-only", "fixture-injected"]
/// Structural validation of a boundary object, including the nested fixture and inclusion records.
func boundaryProblem(_ object: [String: Any]) -> String? {
    let missing = requiredBoundaryKeys.filter { object[$0] == nil }
    if !missing.isEmpty { return "missing keys " + missing.joined(separator: ",") }
    for key in ["version", "operation", "route", "workUnit", "inputRepresentation", "outputRepresentation", "consumer"] {
        guard let text = object[key] as? String, !text.isEmpty else { return "\(key) must be a nonempty string" }
    }
    guard let scope = object["timingScope"] as? String, timingScopes.contains(scope) else { return "unknown timingScope" }
    guard let units = object["unitsPerInvocation"] as? NSNumber, CFGetTypeID(units) != CFBooleanGetTypeID(), units.intValue > 0,
          Double(units.intValue) == units.doubleValue else { return "unitsPerInvocation must be a positive integer" }
    guard let fixture = object["fixture"] as? [String: Any] else { return "fixture is not an object" }
    let fixtureMissing = requiredFixtureKeys.filter { fixture[$0] == nil }
    if !fixtureMissing.isEmpty { return "fixture missing keys " + fixtureMissing.joined(separator: ",") }
    for key in ["id", "version", "values", "preparation"] where !(fixture[key] is String) { return "fixture.\(key) must be a string" }
    for key in ["dimension", "count"] {
        guard let number = fixture[key] as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else { return "fixture.\(key) must be a number" }
    }
    guard let seeds = fixture["seeds"] as? [String: Any], seeds.values.allSatisfy({ $0 is String }) else { return "fixture.seeds must map to strings" }
    guard let inclusions = object["inclusions"] as? [String: Any] else { return "inclusions is not an object" }
    for key in inclusionKeys {
        guard let value = inclusions[key] as? String, inclusionValues.contains(value) else { return "inclusions.\(key) missing or not a known state" }
    }
    return nil
}
func flatten(_ object: [String: Any], prefix: String = "") -> [String: String] {
    var out: [String: String] = [:]
    for (key, value) in object {
        if let nested = value as? [String: Any] { for (path, leaf) in flatten(nested, prefix: prefix + key + ".") { out[path] = leaf } }
        else if let data = try? JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed, .sortedKeys]) {
            out[prefix + key] = String(decoding: data, as: UTF8.self)
        }
    }
    return out
}
struct Run {
    let metadata: [String: Any]
    let rows: [String: Row]
    init(path: String, cli: CLI) throws {
        guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as? [String: Any],
              let results = object["results"] as? [[String: Any]] else { throw ComparisonError("Invalid run: \(path)") }
        metadata = object["metadata"] as? [String: Any] ?? [:]
        if let failures = object["failures"] {
            guard let records = failures as? [Any], records.isEmpty else { throw ComparisonError("Execution/correctness failures in \(path)") }
        }
        var map: [String: Row] = [:]
        let selectedField = ["median": "medianNsPerOp", "p90": "p90NsPerOp", "mean": "meanNsPerOp", "nsPerOp": "nsPerOp"][cli.metric]!
        for result in results {
            guard let name = result["name"] as? String, !name.isEmpty, map[name] == nil else { throw ComparisonError("Duplicate or missing case ID in \(path)") }
            let units = try number(result["unitCount"], field: "\(name).unitCount")
            guard units.rounded(.towardZero) == units else { throw ComparisonError("Noninteger unit count: \(name)") }
            let iterations = try number(result["iterations"], field: "\(name).iterations")
            guard iterations.rounded(.towardZero) == iterations else { throw ComparisonError("Noninteger iterations: \(name)") }
            _ = try number(result["totalNS"], field: "\(name).totalNS")
            for key in ["nsPerOp", "nsPerUnit", "throughputPerSec", "meanNsPerOp", "medianNsPerOp", "p90NsPerOp"] where result[key] != nil {
                _ = try number(result[key], field: "\(name).\(key)")
            }
            if let failure = result["executionFailure"], !(failure is NSNull) { throw ComparisonError("Execution failure: \(name)") }
            if let observation = result["correctness"], !(observation is NSNull) {
                guard let correctness = observation as? [String: Any],
                      let status = correctness["status"] as? String, status == "valid" else { throw ComparisonError("Invalid correctness: \(name)") }
                let samples = try number(correctness["samples"], field: "\(name).correctness.samples")
                guard samples.rounded(.towardZero) == samples else { throw ComparisonError("Noninteger correctness samples: \(name)") }
                for key in ["maxAbsError", "meanAbsError", "maxRelError", "meanRelError"] {
                    _ = try number(correctness[key], field: "\(name).correctness.\(key)", positive: false)
                }
            }
            let metric = try number(result[selectedField], field: "\(name).\(selectedField)")
            var boundary: [String: Any]? = nil
            var problem: String? = nil
            if let value = result["boundary"], !(value is NSNull) {
                if let object = value as? [String: Any] {
                    if let reason = boundaryProblem(object) { problem = reason } else { boundary = object }
                } else { problem = "not an object" }
            }
            map[name] = Row(name: name, units: units, metric: cli.perUnit ? metric / units : metric, boundary: boundary, boundaryProblem: problem)
        }
        rows = map
    }
}
func observation(_ object: [String: Any], _ path: String) -> Any? {
    var current: Any = object
    for key in path.split(separator: ".") {
        guard let dictionary = current as? [String: Any], let next = dictionary[String(key)] else { return nil }
        current = next
    }
    return current
}
func identity(_ value: Any?) -> String? {
    guard let value, !(value is NSNull) else { return nil }
    if let text = value as? String {
        guard !text.isEmpty, !["unknown", "unavailable"].contains(text.lowercased()) else { return nil }
        return text
    }
    guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed, .sortedKeys]) else { return nil }
    return String(decoding: data, as: UTF8.self)
}
func mismatches(_ a: [String: Any], _ b: [String: Any]) -> [String] {
    var keys = ["schemaVersion", "fixtureVersion", "measurementBoundaryVersion", "arch", "deviceModel", "deviceTag", "os", "runtimeToolchain.value", "coreConfiguration", "benchmarkConfiguration", "flags.useCKernels", "flags.preferSoA", "flags.useMixedPrecision", "runSeed", "samples", "measurementMode"]
    let modes = [identity(a["measurementMode"]), identity(b["measurementMode"])]
    keys.append(contentsOf: modes.allSatisfy { $0 == "fixed-repeats" } ? ["repeats"] : ["minTimeSeconds"])
    var errors = keys.compactMap { key -> String? in
        let left = identity(observation(a, key)), right = identity(observation(b, key))
        guard let left, let right else { return "\(key): missing/unknown identity (baseline=\(left ?? "missing"), current=\(right ?? "missing"))" }
        return left == right ? nil : "\(key): baseline=\(left), current=\(right)"
    }
    let known = ["schemaVersion": ["3"], "fixtureVersion": ["V02-inputs-1"], "measurementBoundaryVersion": ["V04-boundaries-1"], "measurementMode": ["fixed-repeats", "estimated-duration"], "coreConfiguration": ["debug", "release"], "benchmarkConfiguration": ["debug", "release"]]
    for (side, metadata) in [("baseline", a), ("current", b)] {
        for (key, values) in known.sorted(by: { $0.key < $1.key }) {
            if let value = identity(metadata[key]), !values.contains(value) { errors.append("\(side).\(key): unsupported identity \(value)") }
        }
    }
    return errors
}
func main() throws -> Int32 {
    let cli = try CLI(arguments: Array(CommandLine.arguments.dropFirst()))
    let baseline = try Run(path: cli.baseline, cli: cli), current = try Run(path: cli.current, cli: cli)
    let incompatible = mismatches(baseline.metadata, current.metadata)
    for mismatch in incompatible { print("Metadata mismatch: \(mismatch)") }
    if !incompatible.isEmpty {
        guard cli.allowIncompatible else { throw ComparisonError("Incompatible benchmark metadata; explicit --allow-incompatible required") }
        print("OVERRIDE --allow-incompatible: metadata compatibility checks waived")
    }
    for field in ["runtimeCheckout.value", "gitSHA", "date", "runLabel"] {
        print("Observation \(field): baseline=\(identity(observation(baseline.metadata, field)) ?? "unavailable"), current=\(identity(observation(current.metadata, field)) ?? "unavailable")")
    }
    func selected(_ name: String) -> Bool { cli.includes.isEmpty || cli.includes.contains(where: name.contains) }
    let left = Set(baseline.rows.keys.filter(selected)), right = Set(current.rows.keys.filter(selected))
    let missingCurrent = left.subtracting(right).sorted(), missingBaseline = right.subtracting(left).sorted()
    print("Missing in current: \(missingCurrent)")
    print("Missing in baseline: \(missingBaseline)")
    let matched = left.intersection(right).sorted()
    guard !matched.isEmpty else { throw ComparisonError("Zero matched cases") }
    // Per-case boundary identity. A boundary that is present on both sides but different is never waivable
    // (different measured work, including a different timing scope). A boundary missing from a legacy
    // (non-schema-3) report is waivable only with the explicit --allow-incompatible override.
    var legacyMissing: [String] = []
    var hardBoundaryErrors: [String] = []
    // "Legacy" means the schemaVersion key is absent or is an integer below 3. A present value that is null, a sentinel
    // string ("", "unknown", "unavailable" in any case), a non-integer, a boolean or an integer above 3 is an integrity
    // error that no override waives. (`identity()` maps sentinels to nil, so it must not be used to detect absence.)
    enum Schema { case legacy, three, unsupported(String) }
    func classify(_ metadata: [String: Any]) -> Schema {
        guard let raw = metadata["schemaVersion"] else { return .legacy }
        let number: Int?
        switch raw {
        case let text as String: number = !text.isEmpty && text.allSatisfy({ $0.isASCII && $0.isNumber }) ? Int(text) : nil
        case let value as NSNumber:
            number = CFGetTypeID(value) == CFBooleanGetTypeID() || value.doubleValue.rounded() != value.doubleValue ? nil : value.intValue
        default: number = nil
        }
        guard let number else { return .unsupported(String(describing: raw)) }
        return number < 3 ? .legacy : number == 3 ? .three : .unsupported(String(number))
    }
    let schemas = (classify(baseline.metadata), classify(current.metadata))
    for (side, schema) in [("baseline", schemas.0), ("current", schemas.1)] {
        if case .unsupported(let text) = schema { hardBoundaryErrors.append("\(side): unsupported schemaVersion \(text)") }
    }
    for name in matched {
        let a = baseline.rows[name]!, b = current.rows[name]!
        for (side, row, schema) in [("baseline", a, schemas.0), ("current", b, schemas.1)] {
            if let problem = row.boundaryProblem { hardBoundaryErrors.append("\(name): \(side) boundary malformed (\(problem))") }
            else if row.boundary == nil {
                if case .three = schema { hardBoundaryErrors.append("\(name): \(side) schema-3 report has no boundary") }
                else { legacyMissing.append("\(name): \(side) has no boundary identity") }
            }
        }
        if let left = a.boundary, let right = b.boundary {
            let x = flatten(left), y = flatten(right)
            let paths = Set(x.keys).union(y.keys).filter { x[$0] != y[$0] }.sorted()
            if !paths.isEmpty { hardBoundaryErrors.append("\(name): fields \(paths.joined(separator: ","))") }
        }
    }
    for error in hardBoundaryErrors { print("Boundary mismatch: \(error)") }
    if !hardBoundaryErrors.isEmpty { throw ComparisonError("Boundary identity mismatch is not waivable (--allow-incompatible does not apply)") }
    for item in legacyMissing { print("Boundary identity missing: \(item)") }
    if !legacyMissing.isEmpty {
        guard cli.allowIncompatible else { throw ComparisonError("Missing boundary identity; explicit --allow-incompatible required") }
        print("OVERRIDE --allow-incompatible: missing legacy boundary identity waived for \(legacyMissing.count) entries")
    }
    var regressions = 0
    for name in matched {
        let a = baseline.rows[name]!, b = current.rows[name]!
        guard a.units == b.units else { throw ComparisonError("Inconsistent unit counts: \(name)") }
        let delta = (b.metric - a.metric) / a.metric * 100
        guard delta.isFinite else { throw ComparisonError("Nonfinite comparison delta: \(name)") }
        if delta > cli.threshold { regressions += 1 }
        if delta > cli.threshold || cli.showImprovements {
            print("\(name) baseline=\(a.metric) current=\(b.metric) delta=\(delta)%")
        }
    }
    print("Compared \(matched.count) cases; metric=\(cli.metric); per-unit=\(cli.perUnit); regressions=\(regressions)")
    if cli.allowMissing { print("OVERRIDE --allow-missing: missing cases permitted") }
    return regressions > 0 || (!cli.allowMissing && (!missingCurrent.isEmpty || !missingBaseline.isEmpty)) ? 1 : 0
}
do { exit(try main()) }
catch { fputs("Invalid comparison: \(error)\n", stderr); exit(2) }
