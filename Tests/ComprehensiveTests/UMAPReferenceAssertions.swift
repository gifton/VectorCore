import Foundation
import Testing
import VectorCore

/// Test-only frozen-reference comparisons. Exact receipts follow a throwing word
/// assertion; the Python runner independently checks their words and inventory.
enum UMAPReferenceAssertions {
    static func compare(_ actual: [Float], words: [UInt32], tolerance: Float, caseID: String) throws {
        try #require(actual.count == words.count, "\(caseID): shape")
        for (index, pair) in zip(actual, words).enumerated() {
            let expected = Float(bitPattern: pair.1)
            try #require(pair.0.isFinite && expected.isFinite, "\(caseID): finite coordinate \(index)")
            try #require(abs(pair.0 - expected) <= tolerance, "\(caseID): coordinate \(index)")
        }
        let exact = ProcessInfo.processInfo.environment["VECTORCORE_UMAP_CHECK_BASELINE_BITS"] == "1"
        let actualWords = actual.map(\.bitPattern)
        if exact {
            try #require(actualWords == words, "\(caseID): frozen words")
        }
        guard let encoded = ProcessInfo.processInfo.environment["VECTORCORE_UMAP_REFERENCE_CONTEXT"] else { return }
        let context = try JSONDecoder().decode(Context.self, from: Data(encoded.utf8))
        let receipt = Receipt(runID: context.runID, configuration: context.configuration,
                              profile: context.profile, fixtureSHA256: context.fixtureSHA256,
                              mode: exact ? "exact" : "tolerance", caseID: caseID,
                              provider: String(reflecting: type(of: Operations.linearAlgebraProvider)),
                              wordCount: actualWords.count, words: actualWords)
        // Unique files retain duplicate case emissions for the runner to reject.
        let path = URL(fileURLWithPath: context.directory).appendingPathComponent(UUID().uuidString + ".json")
        try JSONEncoder().encode(receipt).write(to: path, options: .atomic)
    }

    private struct Context: Decodable {
        let runID: String
        let configuration: String
        let profile: String
        let fixtureSHA256: String
        let directory: String
    }

    private struct Receipt: Encodable {
        let runID: String
        let configuration: String
        let profile: String
        let fixtureSHA256: String
        let mode: String
        let caseID: String
        let provider: String
        let wordCount: Int
        let words: [UInt32]
    }
}
