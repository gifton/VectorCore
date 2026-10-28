import Testing
import VectorCore

@Suite struct BatchZeroQueryContractTests {
    @Test func tinySelfDistanceDoesNotDependOnBatchSize() throws {
        for value: Float in [1e-13, -1e-13, 0, -0.0] {
            let query = Vector512Optimized(repeating: value)
            for count in 0...3 {
                let distances = try BatchKernels_SoA.safeBatchEuclidean512(query: query, candidates: Array(repeating: query, count: count))
                #expect(distances == Array(repeating: Float(0), count: count))
            }
        }
    }

    @Test func mixedTinyCandidatesMatchCoordinateDifferences() throws {
        for magnitude: Float in [0, 1e-13, -1e-13, 0.25] {
            let values = (0..<512).map { $0.isMultiple(of: 2) ? magnitude : -magnitude }
            let query = try Vector512Optimized(values)
            let candidates = [query, Vector512Optimized(repeating: 0), try Vector512Optimized(values.map { -$0 })]
            for count in 0...3 {
                let selected = Array(candidates.prefix(count))
                let actual = try BatchKernels_SoA.safeBatchEuclidean512(query: query, candidates: selected)
                #expect(actual.count == count)
                for (index, candidate) in selected.enumerated() {
                    let expected = zip(values, candidate.toArray()).reduce(0.0) { sum, pair in
                        let difference = Double(pair.0) - Double(pair.1)
                        return sum + difference * difference
                    }
                    if expected == 0 {
                        #expect(actual[index] == 0)
                    } else {
                        #expect(abs(Double(actual[index]) - expected) / expected < 2e-5)
                    }
                }
            }
        }
    }
}
