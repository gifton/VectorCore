import Foundation
import VectorCoreBenchmarking

/// Multi-section CSV: metadata key/value rows, measurements, optional unavailable records, then optional failures.
/// Measurement columns after `suspicious` carry the per-case measurement boundary (V04).
enum CSVWriter {
    static func escape(_ cell: String) -> String {
        if cell.contains(",") || cell.contains("\"") || cell.contains("\n") || cell.contains("\r") {
            return "\"" + cell.replacingOccurrences(of: "\"", with: "\"\"") + "\""
        }
        return cell
    }
    static func row(_ cells: [String]) -> String { cells.map(escape).joined(separator: ",") }
    static let boundaryColumns = [
        "boundary_version", "timing_scope", "operation", "route", "provider_mode", "work_unit", "units_per_invocation",
        "input_representation", "output_representation", "fixture_id", "fixture_preparation",
        "incl_input_construction", "incl_input_preparation", "incl_route_internal_packing", "incl_caller_output_allocation",
        "incl_route_internal_allocation", "incl_provider_scheduling", "incl_provider_observation", "incl_kernel_work",
        "incl_output_consumption", "incl_correctness_validation",
        "consumer", "comparison_family", "comparison_side", "statistic_unit", "sample_ns_per_operation", "median_ns_per_operation",
        "mean_ns_per_operation", "p90_ns_per_operation", "stddev_ns_per_operation", "rsd_percent_per_operation",
        "sample_ns_per_invocation", "sample_iterations", "warmup_iterations", "boundary_json"
    ]

    static func boundaryCells(_ c: BenchCase, statisticUnit: String) -> [String] {
        guard let b = c.boundary else { return Array(repeating: "", count: boundaryColumns.count) }
        let i = b.inclusions
        return [
            b.version, b.timingScope.rawValue, b.operation, b.route, b.provider ?? "", b.workUnit, String(b.unitsPerInvocation),
            b.inputRepresentation, b.outputRepresentation, b.fixture.id, b.fixture.preparation,
            i.inputConstruction.rawValue, i.inputPreparation.rawValue, i.routeInternalPacking.rawValue, i.callerOutputAllocation.rawValue,
            i.routeInternalAllocation.rawValue, i.providerScheduling.rawValue, i.providerObservation.rawValue, i.kernelWork.rawValue,
            i.outputConsumption.rawValue, i.correctnessValidation.rawValue,
            b.consumer, b.comparison?.family ?? "", b.comparison?.side ?? "", statisticUnit,
            (c.sampleNsPerOperation ?? []).map { String($0) }.joined(separator: ";"),
            c.medianNsPerOperation.map { String($0) } ?? "", c.meanNsPerOperation.map { String($0) } ?? "",
            c.p90NsPerOperation.map { String($0) } ?? "", c.stddevNsPerOperation.map { String($0) } ?? "",
            c.rsdPercentPerOperation.map { String($0) } ?? "",
            (c.sampleNsPerInvocation ?? []).map { String($0) }.joined(separator: ";"),
            (c.sampleIterations ?? []).map(String.init).joined(separator: ";"),
            c.warmupIterations.map(String.init) ?? "", b.canonicalJSON()
        ]
    }

    static func write(cases: [BenchCase], metadata: BenchMetadata, failures: [BenchmarkFailure] = [],
                      unavailable: [UnavailableCase] = [], to path: String) throws {
        var rows: [String] = []
        let object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(metadata)) as! [String: Any]
        func appendMetadata(_ object: [String: Any], prefix: String = "") throws {
            for key in object.keys.sorted() {
                let value = object[key]!
                if let dictionary = value as? [String: Any] { try appendMetadata(dictionary, prefix: prefix + key + ".") }
                else if let text = value as? String { rows.append(row([prefix + key, text])) }
                else {
                    let data = try JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed, .sortedKeys])
                    rows.append(row([prefix + key, String(decoding: data, as: UTF8.self)]))
                }
            }
        }
        try appendMetadata(object)
        rows.append("")
        // Data header
        rows.append(row([
            "name","kind","metric","dim","variant","provider","N",
            "iterations","total_ns","ns_per_op","unit_count","ns_per_unit","throughput_per_s","gflops","samples","mean_ns_per_op","median_ns_per_op","p90_ns_per_op",
                "stddev_ns_per_op","rsd_percent","suspicious"
        ] + boundaryColumns))

        for c in cases {
            let p = c.params
            // Built in typed parts: the single literal-plus-concatenation form exceeded the
            // Swift 6.0/6.2 type checker's limits in CI (Xcode 16.2 and 26.3).
            let identity: [String] = [
                c.name, p.kind ?? "", p.metric ?? "",
                p.dim.map(String.init) ?? "",
                p.variant ?? "", p.provider ?? "",
                p.n.map(String.init) ?? ""
            ]
            let totals: [String] = [
                String(c.iterations), String(c.totalNS), String(c.nsPerOp),
                String(c.unitCount), String(c.nsPerUnit),
                String(c.throughputPerSec)
            ]
            let statistics: [String] = [
                c.gflops.map { String($0) } ?? "",
                c.samples.map(String.init) ?? "",
                c.meanNsPerOp.map { String($0) } ?? "",
                c.medianNsPerOp.map { String($0) } ?? "",
                c.p90NsPerOp.map { String($0) } ?? "",
                c.stddevNsPerOp.map { String($0) } ?? "",
                c.rsdPercent.map { String($0) } ?? "",
                c.suspicious ? "true" : "false"
            ]
            var fields: [String] = identity
            fields += totals
            fields += statistics
            fields += boundaryCells(c, statisticUnit: metadata.statisticUnit)
            rows.append(row(fields))
        }

        if !unavailable.isEmpty {
            rows.append("")
            rows.append(row(["unavailable_name", "unavailable_reason", "timing_scope", "route", "boundary_json"]))
            for entry in unavailable {
                rows.append(row([entry.name, entry.reason, entry.boundary.timingScope.rawValue, entry.boundary.route,
                    entry.boundary.canonicalJSON()]))
            }
        }

        if !failures.isEmpty {
            rows.append("")
            rows.append(row(["failure_name", "failure_reason", "failure_boundary_json"]))
            for failure in failures { rows.append(row([failure.name, failure.reason, failure.boundary?.canonicalJSON() ?? ""])) }
        }
        let out = rows.joined(separator: "\n") + "\n"
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try out.write(to: url, atomically: true, encoding: .utf8)
    }
}
