import Foundation

enum Filters {
    static func shouldRun(name: String, options: ValidatedOptions) -> Bool { options.caseFilter.matches(name) }
}

struct CaseFilter: Sendable {
    enum Mode { case glob, regex }
    let include: [NSRegularExpression]
    let exclude: [NSRegularExpression]

    init(include: [String], exclude: [String], mode: String) throws {
        let m: Mode = mode.lowercased() == "regex" ? .regex : .glob
        self.include = try include.map { try CaseFilter.compile(pattern: $0, mode: m) }
        self.exclude = try exclude.map { try CaseFilter.compile(pattern: $0, mode: m) }
    }

    func matches(_ name: String) -> Bool {
        let ns = NSRange(location: 0, length: name.utf16.count)
        // Include: if none specified, include all by default
        var ok = include.isEmpty || include.contains(where: { $0.firstMatch(in: name, options: [], range: ns) != nil })
        if !ok { return false }
        // Exclude: any match rejects
        if exclude.contains(where: { $0.firstMatch(in: name, options: [], range: ns) != nil }) {
            ok = false
        }
        return ok
    }

    private static func compile(pattern: String, mode: Mode) throws -> NSRegularExpression {
        let pat = pattern.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !pat.isEmpty else { throw BenchmarkConfigurationError("Empty filter") }
        let regex: String
        switch mode {
        case .regex:
            regex = pat
        case .glob:
            regex = globToRegex(pat)
        }
        return try NSRegularExpression(pattern: regex)
    }

    private static func globToRegex(_ glob: String) -> String {
        var rx = "^"
        for ch in glob {
            switch ch {
            case "*": rx += ".*"
            case "?": rx += "."
            case ".", "+", "(", ")", "[", "]", "{", "}", "^", "$", "|", "\\":
                rx += "\\" + String(ch)
            default:
                rx += String(ch)
            }
        }
        rx += "$"
        return rx
    }
}

