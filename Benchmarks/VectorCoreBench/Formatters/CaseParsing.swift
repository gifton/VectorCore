import Foundation

public struct CaseParams: Codable {
    public var kind: String? = nil         // dot, dist, normalize, batch
    public var metric: String? = nil       // euclidean, cosine, manhattan, dot
    public var dim: Int? = nil
    public var variant: String? = nil      // generic|optimized
    public var provider: String? = nil     // sequential|parallel|automatic
    public var n: Int? = nil               // candidate count for batch
    public var status: String? = nil       // success|zeroFail for normalize
}

enum CaseParsing {
    static func parse(name: String) -> CaseParams {
        var p = CaseParams()
        let parts = name.split(separator: ".").map(String.init)
        guard !parts.isEmpty else { return p }
        p.kind = parts.first

        switch p.kind {
        case "dot":
            if parts.count >= 3 {
                p.dim = Int(parts[1])
                p.variant = parts[2]
            }
        case "dist":
            if parts.count >= 4 {
                p.metric = parts[1]
                p.dim = Int(parts[2])
                p.variant = parts[3]
            }
        case "normalize":
            if parts.count >= 4 {
                p.status = parts[1]
                p.dim = Int(parts[2])
                p.variant = parts[3]
            }
        case "batch":
            // batch.euclidean.<dim>.N<count>.<variant>.<mode>
            if parts.count >= 6 {
                p.metric = parts[1]
                p.dim = Int(parts[2])
                if parts[3].hasPrefix("N") { p.n = Int(parts[3].dropFirst()) }
                p.variant = parts[4]
                p.provider = parts[5]
            }
        default:
            break
        }
        return p
    }
}

