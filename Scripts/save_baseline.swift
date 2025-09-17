#!/usr/bin/env swift
import Foundation

struct CLI {
    var input: String = ""
    var outdir: String = "Benchmarks/baselines"
    var label: String? = nil
}

func parseCLI() -> CLI? {
    var cli = CLI()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let arg = it.next() {
        switch arg {
        case "--input": if let v = it.next() { cli.input = v }
        case "--outdir": if let v = it.next() { cli.outdir = v }
        case "--label": if let v = it.next() { cli.label = v }
        case "--help", "-h": return nil
        default:
            fputs("Unknown arg: \(arg)\n", stderr)
            return nil
        }
    }
    guard !cli.input.isEmpty else { return nil }
    return cli
}

struct BenchRun: Decodable { let metadata: BenchMetadata }
struct BenchMetadata: Decodable {
    let arch: String
    let deviceModel: String?
    let os: String
    let packageVersion: String
    let gitSHA: String?
    let date: String
}

func loadRun(_ path: String) throws -> BenchRun {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    let dec = JSONDecoder()
    return try dec.decode(BenchRun.self, from: data)
}

func slug(_ s: String) -> String {
    let lowered = s.lowercased()
    let allowed = Set("abcdefghijklmnopqrstuvwxyz0123456789-._")
    let mapped = lowered.map { ch -> Character in
        if allowed.contains(ch) { return ch }
        if ch.isWhitespace { return "-" }
        return "-"
    }
    // Collapse runs of '-'
    var out: [Character] = []
    var lastDash = false
    for ch in mapped {
        if ch == "-" {
            if !lastDash { out.append(ch); lastDash = true }
        } else {
            out.append(ch); lastDash = false
        }
    }
    return String(out).trimmingCharacters(in: CharacterSet(charactersIn: "-"))
}

func extractOSVersion(_ s: String) -> String {
    // Try to find a version-like token (e.g., 15.4.1)
    let pattern = #"(\d+\.\d+(?:\.\d+)?)"#
    if let regex = try? NSRegularExpression(pattern: pattern),
       let m = regex.firstMatch(in: s, options: [], range: NSRange(location: 0, length: s.utf16.count)),
       let r = Range(m.range(at: 1), in: s) {
        return String(s[r])
    }
    return slug(s)
}

func main() {
    guard let cli = parseCLI() else {
        print("Usage: swift Scripts/save_baseline.swift --input .bench/results.json [--outdir Benchmarks/baselines] [--label mytag]")
        exit(2)
    }
    do {
        let run = try loadRun(cli.input)
        let meta = run.metadata
        let arch = slug(meta.arch)
        let model = slug(meta.deviceModel ?? "unknown")
        let osVer = extractOSVersion(meta.os)
        let osSlug = "os" + slug(osVer)
        var hostTag = arch
        if model != "unknown" && model != arch { hostTag += "-" + model }
        hostTag += "-" + osSlug
        if let label = cli.label { hostTag += "-" + slug(label) }

        let fm = FileManager.default
        try fm.createDirectory(atPath: cli.outdir, withIntermediateDirectories: true)
        let outPath = cli.outdir + "/" + hostTag + ".json"
        try fm.copyItem(atPath: cli.input, toPath: outPath)
        print("Saved baseline → \(outPath)")
        exit(0)
    } catch {
        fputs("Error: \(error)\n", stderr)
        exit(1)
    }
}

main()

