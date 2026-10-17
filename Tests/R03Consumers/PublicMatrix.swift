import VectorCore

@main struct R03PublicMatrixConsumer {
    static func main() async throws {
        let rows = [DynamicVector([3, 4, 0]), DynamicVector([0, 0, 0])]
        let dot = await BatchOperations.pairwiseDistances(rows, metric: DotProductDistance())
        let cosine = await BatchOperations.pairwiseDistances(rows, metric: CosineDistance())
        precondition(dot[0][0] == -25 && cosine[1][1] == 1)
        let a = try Vector<Dim768>([3, 4, 0] + Array(repeating: 0, count: 765))
        let b = try Vector<Dim768>([-3, -4, 0] + Array(repeating: 0, count: 765))
        let zero = try Vector<Dim768>(Array(repeating: 0, count: 768))
        let similarity = try await [a, b, zero].cosineSimilarityMatrix()
        precondition(similarity[0][0] > 0.99999 && similarity[0][1] < -0.99999)
        precondition(similarity[2].allSatisfy { $0 == 0 })
        let query = try Vector512Optimized(Array(repeating: 0, count: 512))
        let soa = SoA512.build(from: [query])
        var values: [Float] = [17, 17, 17]
        // Values are read inside the closure (lost-store workaround W2; swiftlang/swift#92562).
        let observed: [Float] = try values.withUnsafeMutableBufferPointer {
            try BatchKernels_SoA.batchCosineDistance512Safe(query: query, soa: soa, out: $0)
            return Array($0)
        }
        precondition(observed == [1, 17, 17])
        print("R03_PUBLIC_MATRIX_MIGRATION_COMPLETE")
    }
}
