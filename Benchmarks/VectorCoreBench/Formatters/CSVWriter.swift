import Foundation

enum CSVWriter {
    static func write(cases: [BenchCase], metadata: BenchMetadata, to path: String) throws {
        var rows: [String] = []
        // Header rows with limited metadata
        rows.append("package,\(metadata.package)")
        rows.append("packageVersion,\(metadata.packageVersion)")
        rows.append("date,\(metadata.date)")
        if let sha = metadata.gitSHA { rows.append("gitSHA,\(sha)") }
        rows.append("os,\(metadata.os)")
        rows.append("arch,\(metadata.arch)")
        if let model = metadata.deviceModel { rows.append("deviceModel,\(model)") }
        if let swift = metadata.swiftVersion { rows.append("swiftVersion,\(swift.replacingOccurrences(of: ",", with: ";"))") }
        rows.append("buildConfiguration,\(metadata.buildConfiguration)")
        rows.append("suites,\(metadata.suites.joined(separator: ";"))")
        rows.append("dims,\(metadata.dims.map(String.init).joined(separator: ";"))")
        rows.append("")

        // Data header
        rows.append([
            "name","kind","metric","dim","variant","provider","N",
            "iterations","total_ns","ns_per_op","unit_count","ns_per_unit","throughput_per_s","gflops","samples","mean_ns_per_op","median_ns_per_op","p90_ns_per_op","stddev_ns_per_op","rsd_percent","suspicious"
        ].joined(separator: ","))

        let q: (String?) -> String = { s in
            guard let s else { return "" }
            if s.contains(",") || s.contains("\"") || s.contains("\n") {
                return "\"" + s.replacingOccurrences(of: "\"", with: "\"\"") + "\""
            }
            return s
        }

        for c in cases {
            let p = c.params
            let fields: [String] = [
                q(c.name), q(p.kind), q(p.metric),
                p.dim.map(String.init) ?? "",
                q(p.variant), q(p.provider),
                p.n.map(String.init) ?? "",
                String(c.iterations), String(c.totalNS), String(format: "%.4f", c.nsPerOp),
                String(c.unitCount), String(format: "%.4f", c.nsPerUnit),
                String(format: "%.2f", c.throughputPerSec),
                c.gflops.map { String(format: "%.2f", $0) } ?? "",
                c.samples.map(String.init) ?? "",
                c.meanNsPerOp.map { String(format: "%.4f", $0) } ?? "",
                c.medianNsPerOp.map { String(format: "%.4f", $0) } ?? "",
                c.p90NsPerOp.map { String(format: "%.4f", $0) } ?? "",
                c.stddevNsPerOp.map { String(format: "%.4f", $0) } ?? "",
                c.rsdPercent.map { String(format: "%.2f", $0) } ?? "",
                c.suspicious ? "true" : "false"
            ]
            rows.append(fields.joined(separator: ","))
        }

        let out = rows.joined(separator: "\n") + "\n"
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try out.write(to: url, atomically: true, encoding: .utf8)
    }
}
