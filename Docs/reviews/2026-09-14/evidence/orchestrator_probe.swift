import Foundation
import VectorCore

@main
struct ReviewProbe {
    static func main() async throws {
        let mode = CommandLine.arguments.dropFirst().first ?? "safe"
        if mode == "empty-centroid" {
            let result: Vector<Dim512> = Operations.centroid(of: [])
            print(result[0])
            return
        }
        if mode == "ragged-centroid" {
            let a = try DynamicVector([1, 2, 3, 4, 5])
            let b = try DynamicVector([10])
            let result = Operations.centroid(of: [a, b])
            print(result.toArray())
            return
        }
        if mode == "ragged-statistics" {
            let a = try DynamicVector([1, 2, 3, 4, 5])
            let b = try DynamicVector([10])
            print(Operations.statistics(for: [a, b]).mean)
            return
        }
        let v = Vector<Dim768>(repeating: 1)
        let w = Vector<Dim768>(repeating: -1)
        print("cosineSimilarityMatrix", try await [v, v, w].cosineSimilarityMatrix())
        let huge = try DynamicVector([Float.greatestFiniteMagnitude])
        print("finite-centroid", Operations.centroid(of: [huge, huge]).toArray())
        var hint = NormalizationHint(vector: DynamicVector([1, 0]), isNormalized: true)
        hint.vector[0] = 2
        print("direct-mutation-hint", hint.isNormalized, hint.magnitude, hint.vector.magnitude)
        enum Expected: Error { case stop }
        var throwingHint = NormalizationHint(vector: DynamicVector([1, 0]), isNormalized: true)
        do {
            try throwingHint.withUnsafeMutableBufferPointer { buffer in
                buffer[0] = 3
                throw Expected.stop
            }
        } catch {}
        print("throwing-mutation-hint", throwingHint.isNormalized, throwingHint.magnitude, throwingHint.vector.magnitude)
    }
}
