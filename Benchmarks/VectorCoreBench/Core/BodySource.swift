import Foundation

/// Lexical source utilities for the body-drift guards (see `Boundaries.bodyFingerprints`).
///
/// This is a bounded Swift-aware scanner, not a parser: it understands line and block comments, string literals and
/// string interpolation, balanced braces and parentheses. Multi-line string literals (`"""`) are not supported and
/// `stripComments` traps if it meets one. Raw strings (`#"..."#`) are NOT detected: a `//` or quote inside one would
/// be mis-scanned, so they must not appear in fingerprinted sources (none do today).
enum BodySource {
    /// Source with comments removed (string contents are kept verbatim).
    static func stripComments(_ text: String) -> String {
        let u = Array(text.unicodeScalars)
        var out = String.UnicodeScalarView()
        var i = 0
        while i < u.count {
            if u[i] == "/", i + 1 < u.count, u[i + 1] == "/" {
                while i < u.count, u[i] != "\n" { i += 1 }
            } else if u[i] == "/", i + 1 < u.count, u[i + 1] == "*" {
                i += 2
                while i + 1 < u.count, !(u[i] == "*" && u[i + 1] == "/") { i += 1 }
                i = min(i + 2, u.count)
                out.append(" ")
            } else if u[i] == "\"" {
                precondition(!(i + 2 < u.count && u[i + 1] == "\"" && u[i + 2] == "\""),
                    "multi-line string literal is not supported by BodySource")
                let start = i
                skipString(u, &i)
                out.append(contentsOf: u[start..<i])
            } else {
                out.append(u[i])
                i += 1
            }
        }
        return String(out)
    }

    /// Whitespace runs collapsed to one space; the form that is hashed.
    static func collapsed(_ text: String) -> String {
        text.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
    }

    /// Comment-stripped source (whitespace untouched) of every declaration named `symbol`: `func symbol` overloads and
    /// `var symbol: Type { ... }` computed properties, in source order. Empty when the symbol is not declared.
    static func declarations(named symbol: String, in source: String) -> [String] {
        let stripped = stripComments(source)
        let u = Array(stripped.unicodeScalars)
        let escaped = NSRegularExpression.escapedPattern(for: symbol)
        guard let regex = try? NSRegularExpression(pattern: "\\b(?:func\\s+" + escaped + "\\s*[<(]|var\\s+" + escaped
            + "\\s*:)") else { return [] }
        let ns = stripped as NSString
        var pieces: [String] = []
        for match in regex.matches(in: stripped, range: NSRange(location: 0, length: ns.length)) {
            // Convert the UTF-16 offset to a scalar index.
            var i = ns.substring(to: match.range.location).unicodeScalars.count
            let start = i
            var depth = 0
            var opened = false
            while i < u.count, !opened {
                switch u[i] {
                case "(", "[": depth += 1; i += 1
                case ")", "]": depth -= 1; i += 1
                case "\"": skipString(u, &i)
                case "{" where depth == 0: opened = true
                default: i += 1
                }
            }
            guard opened else { continue }
            var braces = 0
            repeat {
                if u[i] == "\"" { skipString(u, &i); continue }
                if u[i] == "{" { braces += 1 } else if u[i] == "}" { braces -= 1 }
                i += 1
            } while braces > 0 && i < u.count
            var text = String.UnicodeScalarView()
            text.append(contentsOf: u[start..<i])
            pieces.append(String(text))
        }
        return pieces
    }

    /// Symbols that have no declaration in `source` (a body fingerprint with any missing symbol must fail).
    static func missingSymbols(_ symbols: [String], in source: String) -> [String] {
        symbols.filter { declarations(named: $0, in: source).isEmpty }
    }

    /// Whitespace-collapsed declarations of every symbol (symbol order, then source order), joined by newlines. A missing
    /// symbol contributes nothing, so callers must also check `missingSymbols`.
    static func strippedSource(symbols: [String], in source: String) -> String {
        symbols.flatMap { declarations(named: $0, in: source) }.map(collapsed).joined(separator: "\n")
    }

    /// For each call of `Harness.<callee>` in `text`: the body it runs, as whitespace-collapsed text. The body is the
    /// trailing closure, or a bare identifier passed as the last positional argument. Returns one entry per call.
    static func harnessBodies(callee: String, in text: String) -> [String] {
        let u = Array(text.unicodeScalars)
        let marker = Array(("Harness." + callee).unicodeScalars)
        var bodies: [String] = []
        var i = 0
        while i + marker.count < u.count {
            guard Array(u[i..<(i + marker.count)]) == marker else { i += 1; continue }
            var j = i + marker.count
            // The callee name must end here (warmup vs warmupAsync).
            if u[j].properties.isAlphabetic || u[j] == "_" { i += 1; continue }
            while j < u.count, u[j].properties.isWhitespace { j += 1 }
            var lastArgument = ""
            if j < u.count, u[j] == "(" {
                let open = j
                var depth = 0
                var argumentStart = open + 1
                var arguments: [String] = []
                repeat {
                    if u[j] == "\"" { skipString(u, &j); continue }
                    if u[j] == "(" || u[j] == "[" || u[j] == "{" { depth += 1 }
                    if u[j] == ")" || u[j] == "]" || u[j] == "}" { depth -= 1 }
                    if depth == 1, u[j] == "," {
                        arguments.append(slice(u, argumentStart, j)); argumentStart = j + 1
                    }
                    j += 1
                } while depth > 0 && j < u.count
                arguments.append(slice(u, argumentStart, j - 1))
                lastArgument = collapsed(arguments.last ?? "")
                while j < u.count, u[j].properties.isWhitespace { j += 1 }
            }
            if j < u.count, u[j] == "{" {
                let start = j
                var braces = 0
                repeat {
                    if u[j] == "\"" { skipString(u, &j); continue }
                    if u[j] == "{" { braces += 1 } else if u[j] == "}" { braces -= 1 }
                    j += 1
                } while braces > 0 && j < u.count
                bodies.append(collapsed(slice(u, start, j)))
            } else {
                bodies.append(lastArgument.contains(":") ? "" : lastArgument)
            }
            i = j
        }
        return bodies
    }

    private static func slice(_ u: [Unicode.Scalar], _ from: Int, _ to: Int) -> String {
        var view = String.UnicodeScalarView()
        view.append(contentsOf: u[from..<to])
        return String(view)
    }

    /// Text of each `BenchmarkCase(...) { ... }` expression: header token (first argument) and the call plus trailing closure.
    static func caseSegments(in source: String) -> [(token: String, text: String)] {
        let stripped = stripComments(source)
        let u = Array(stripped.unicodeScalars)
        var segments: [(String, String)] = []
        let marker = Array("BenchmarkCase(".unicodeScalars)
        var i = 0
        while i + marker.count < u.count {
            guard Array(u[i..<(i + marker.count)]) == marker, i == 0 || !(u[i - 1].properties.isAlphabetic) else { i += 1; continue }
            let start = i
            var j = i + marker.count
            let tokenStart = j
            var depth = 1
            var tokenEnd: Int?
            while j < u.count, depth > 0 {
                if u[j] == "\"" { skipString(u, &j); continue }
                if u[j] == "," && depth == 1 && tokenEnd == nil { tokenEnd = j }
                if u[j] == "(" { depth += 1 } else if u[j] == ")" { depth -= 1 }
                j += 1
            }
            while j < u.count, u[j].properties.isWhitespace { j += 1 }
            if j < u.count, u[j] == "{" {
                var braces = 0
                repeat {
                    if u[j] == "\"" { skipString(u, &j); continue }
                    if u[j] == "{" { braces += 1 } else if u[j] == "}" { braces -= 1 }
                    j += 1
                } while braces > 0 && j < u.count
            }
            var token = String.UnicodeScalarView(); token.append(contentsOf: u[tokenStart..<(tokenEnd ?? j)])
            var text = String.UnicodeScalarView(); text.append(contentsOf: u[start..<j])
            segments.append((String(token).trimmingCharacters(in: .whitespaces), String(text)))
            i = j
        }
        return segments
    }

    /// Arguments of every `blackHole(` call in `text`, with all whitespace removed.
    static func blackHoleArguments(in text: String) -> [String] {
        let u = Array(text.unicodeScalars)
        let marker = Array("blackHole(".unicodeScalars)
        var arguments: [String] = []
        var i = 0
        while i + marker.count <= u.count {
            guard Array(u[i..<(i + marker.count)]) == marker, i == 0 || !(u[i - 1].properties.isAlphabetic) else { i += 1; continue }
            var j = i + marker.count
            let start = j
            var depth = 1
            while j < u.count, depth > 0 {
                if u[j] == "\"" { skipString(u, &j); continue }
                if u[j] == "(" { depth += 1 } else if u[j] == ")" { depth -= 1 }
                j += 1
            }
            var argument = String.UnicodeScalarView(); argument.append(contentsOf: u[start..<(j - 1)])
            arguments.append(String(argument).filter { !$0.isWhitespace })
            i = j
        }
        return arguments
    }

    /// Names of helper functions called inside `text` (`runBatch...` helpers and `measure`).
    static func helperCalls(in text: String) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: "\\b(run[A-Z]\\w*|measure)\\s*(?:<[^>(]*>)?\\(") else { return [] }
        let ns = text as NSString
        var seen: [String] = []
        for match in regex.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            let name = ns.substring(with: match.range(at: 1))
            if !seen.contains(name) { seen.append(name) }
        }
        return seen
    }

    // MARK: Scanning primitives

    /// Advances `i` past a string literal that starts at the opening quote, recursing into interpolations.
    private static func skipString(_ u: [Unicode.Scalar], _ i: inout Int) {
        i += 1
        while i < u.count {
            if u[i] == "\\" {
                if i + 1 < u.count, u[i + 1] == "(" { i += 2; skipInterpolation(u, &i) } else { i += 2 }
            } else if u[i] == "\"" {
                i += 1
                return
            } else {
                i += 1
            }
        }
    }

    private static func skipInterpolation(_ u: [Unicode.Scalar], _ i: inout Int) {
        var depth = 1
        while i < u.count, depth > 0 {
            if u[i] == "\"" { skipString(u, &i); continue }
            if u[i] == "(" { depth += 1 } else if u[i] == ")" { depth -= 1 }
            i += 1
        }
    }
}
