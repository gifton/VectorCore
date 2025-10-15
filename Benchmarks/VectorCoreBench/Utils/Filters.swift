import Foundation

enum Filters {
    nonisolated(unsafe) private static var cacheKey: String = ""
    nonisolated(unsafe) private static var cached: CaseFilter? = nil

    static func shouldRun(name: String, options: CLIOptions) -> Bool {
        let key = cacheKeyFor(options: options)
        if key != cacheKey {
            cacheKey = key
            cached = CaseFilter(include: options.filters, exclude: options.excludes, mode: options.filterMode)
        }
        guard let cf = cached else { return true }
        return cf.matches(name)
    }

    private static func cacheKeyFor(options: CLIOptions) -> String {
        ([options.filterMode] + options.filters + ["--"] + options.excludes).joined(separator: "\u{1f}")
    }
}

struct CaseFilter {
    enum Mode { case glob, regex }
    let include: [NSRegularExpression]
    let exclude: [NSRegularExpression]

    init(include: [String], exclude: [String], mode: String) {
        let m: Mode = mode.lowercased() == "regex" ? .regex : .glob
        self.include = include.compactMap { CaseFilter.compile(pattern: $0, mode: m) }
        self.exclude = exclude.compactMap { CaseFilter.compile(pattern: $0, mode: m) }
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

    private static func compile(pattern: String, mode: Mode) -> NSRegularExpression? {
        let pat = pattern.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !pat.isEmpty else { return nil }
        let regex: String
        switch mode {
        case .regex:
            regex = pat
        case .glob:
            regex = globToRegex(pat)
        }
        return try? NSRegularExpression(pattern: regex)
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

