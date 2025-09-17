#!/usr/bin/env swift
import Foundation

// Simple baseline comparison for VectorCoreBench JSON outputs.
// Compares metric per benchmark case by name and reports regressions.
//
// Usage:
//   swift Scripts/compare_benchmarks.swift \
//     --baseline path/to/baseline.json \
//     --current  path/to/current.json \
//     [--metric median|p90|mean|nsPerOp] [--threshold 10.0] [--per-unit] \
//     [--include substr]... [--show-improvements] [--strict]
//
// Exit codes: 0 = no regressions, 1 = regressions found, 2 = invalid usage

struct CLI {
    var baseline: String = ""
    var current: String = ""
    var metric: String = "median" // median|p90|mean|nsPerOp
    var threshold: Double = 10.0
    var perUnit: Bool = false
    var includes: [String] = []
    var showImprovements: Bool = false
    var strictMissing: Bool = false
}

func parseCLI() -> CLI? {
    var cli = CLI()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let arg = it.next() {
        switch arg {
        case "--baseline": if let v = it.next() { cli.baseline = v }
        case "--current": if let v = it.next() { cli.current = v }
        case "--metric": if let v = it.next() { cli.metric = v }
        case "--threshold": if let v = it.next(), let d = Double(v) { cli.threshold = d }
        case "--per-unit": cli.perUnit = true
        case "--include": if let v = it.next() { cli.includes.append(v) }
        case "--show-improvements": cli.showImprovements = true
        case "--strict": cli.strictMissing = true
        case "--help", "-h": return nil
        default:
            fputs("Unknown arg: \(arg)\n", stderr)
            return nil
        }
    }
    guard !cli.baseline.isEmpty, !cli.current.isEmpty else { return nil }
    return cli
}

struct BenchRun: Decodable { let results: [BenchCase] }
struct BenchCase: Decodable {
    let name: String
    let iterations: Int
    let totalNS: UInt64
    let nsPerOp: Double
    let unitCount: Int
    let nsPerUnit: Double
    let samples: Int?
    let meanNsPerOp: Double?
    let medianNsPerOp: Double?
    let p90NsPerOp: Double?
}

func loadRun(_ path: String) throws -> BenchRun {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let dec = JSONDecoder()
    return try dec.decode(BenchRun.self, from: data)
}

func metricValue(case c: BenchCase, metric: String, perUnit: Bool) -> Double {
    switch metric.lowercased() {
    case "median":
        if let m = c.medianNsPerOp { return perUnit ? m / Double(max(c.unitCount, 1)) : m }
        fallthrough
    case "p90":
        if metric.lowercased() == "p90", let p = c.p90NsPerOp { return perUnit ? p / Double(max(c.unitCount, 1)) : p }
        fallthrough
    case "mean":
        if metric.lowercased() == "mean", let m = c.meanNsPerOp { return perUnit ? m / Double(max(c.unitCount, 1)) : m }
        fallthrough
    default: // nsPerOp
        let v = c.nsPerOp
        return perUnit ? v / Double(max(c.unitCount, 1)) : v
    }
}

struct DiffRow { let name: String; let base: Double; let curr: Double; let deltaPct: Double }

func main() {
    guard let cli = parseCLI() else {
        print("Usage: swift Scripts/compare_benchmarks.swift --baseline BASE.json --current CUR.json [--metric median|p90|mean|nsPerOp] [--threshold 10.0] [--per-unit] [--include substr]... [--show-improvements] [--strict]")
        exit(2)
    }
    do {
        let baseRun = try loadRun(cli.baseline)
        let curRun = try loadRun(cli.current)

        let includeCheck: (String) -> Bool = { name in
            if cli.includes.isEmpty { return true }
            return cli.includes.contains(where: { name.contains($0) })
        }

        var baseMap: [String: Double] = [:]
        for c in baseRun.results where includeCheck(c.name) {
            baseMap[c.name] = metricValue(case: c, metric: cli.metric, perUnit: cli.perUnit)
        }

        var diffs: [DiffRow] = []
        var missingInCurrent: [String] = []
        var missingInBaseline: [String] = []

        var seen = Set<String>()
        for c in curRun.results where includeCheck(c.name) {
            let curr = metricValue(case: c, metric: cli.metric, perUnit: cli.perUnit)
            if let base = baseMap[c.name] {
                let delta = (curr - base) / base * 100.0
                diffs.append(DiffRow(name: c.name, base: base, curr: curr, deltaPct: delta))
            } else {
                missingInBaseline.append(c.name)
            }
            seen.insert(c.name)
        }
        for (name, _) in baseMap where !seen.contains(name) { missingInCurrent.append(name) }

        let regressions = diffs.filter { $0.deltaPct > cli.threshold }
        let improvements = diffs.filter { $0.deltaPct < -cli.threshold }
        let neutral = diffs.count - regressions.count - improvements.count

        func fmt(_ v: Double) -> String { String(format: "%.4f", v) }

        print("\n=== Benchmark Compare (metric=\(cli.metric) perUnit=\(cli.perUnit ? "1" : "0"), threshold=\(cli.threshold)%%) ===")
        if !regressions.isEmpty {
            print("\nRegressions (> +\(cli.threshold)%): \(regressions.count)")
            for r in regressions.sorted(by: { $0.deltaPct > $1.deltaPct }) {
                print("- \(r.name)  base=\(fmt(r.base))  curr=\(fmt(r.curr))  Δ=\(String(format: "%+.2f%%", r.deltaPct)))")
            }
        } else {
            print("\nRegressions: 0")
        }

        if cli.showImprovements, !improvements.isEmpty {
            print("\nImprovements (< -\(cli.threshold)%): \(improvements.count)")
            for r in improvements.sorted(by: { $0.deltaPct < $1.deltaPct }) {
                print("- \(r.name)  base=\(fmt(r.base))  curr=\(fmt(r.curr))  Δ=\(String(format: "%+.2f%%", r.deltaPct)))")
            }
        }

        print("\nSummary: total=\(diffs.count)  regressions=\(regressions.count)  improvements=\(improvements.count)  neutral=\(neutral)")

        if !missingInBaseline.isEmpty { print("Missing in baseline: \(missingInBaseline.count)") }
        if !missingInCurrent.isEmpty { print("Missing in current: \(missingInCurrent.count)") }

        if !regressions.isEmpty || (cli.strictMissing && (!missingInBaseline.isEmpty || !missingInCurrent.isEmpty)) {
            exit(1)
        } else {
            exit(0)
        }
    } catch {
        fputs("Error: \(error)\n", stderr)
        exit(2)
    }
}

main()

