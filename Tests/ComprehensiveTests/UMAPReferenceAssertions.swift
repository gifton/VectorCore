import Foundation
import Testing
import VectorCore

/// Test-only frozen-reference comparisons. Exact receipts follow a throwing word
/// assertion; the Python runner independently checks their words and inventory.
enum UMAPReferenceAssertions {
    static func compare(_ actual: [Float], words: [UInt32], tolerance: Float, caseID: String,
                        exact: Bool = ProcessInfo.processInfo.environment["VECTORCORE_UMAP_CHECK_BASELINE_BITS"] == "1",
                        contextJSON: String? = ProcessInfo.processInfo.environment["VECTORCORE_UMAP_REFERENCE_CONTEXT"]) throws {
        let context = try contextJSON.map { try JSONDecoder().decode(Context.self, from: Data($0.utf8)) }
        let expectedWords: [UInt32]
        if exact, let context {
            expectedWords = try #require(context.oracle[caseID], "\(caseID): missing profile case")
        } else {
            expectedWords = words
        }
        try #require(actual.count == expectedWords.count, "\(caseID): shape")
        for (index, pair) in zip(actual, expectedWords).enumerated() {
            let expected = Float(bitPattern: pair.1)
            try #require(pair.0.isFinite && expected.isFinite, "\(caseID): finite coordinate \(index)")
            try #require(abs(pair.0 - expected) <= tolerance, "\(caseID): coordinate \(index)")
        }
        let actualWords = actual.map(\.bitPattern)
        if exact {
            try #require(actualWords == expectedWords, "\(caseID): frozen words")
        }
        guard let context else { return }
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
        let oracle: [String: [UInt32]]
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
