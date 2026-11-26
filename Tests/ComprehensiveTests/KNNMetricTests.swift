import Foundation
import Testing
@testable import VectorCore

enum KNNMetricFixtures {
    // Independent Double scalar oracle: no production normalization, matrix, or selection helpers.
    static func cosine(_ x: [Float], _ y: [Float]) -> Double {
        let a = x.map(Double.init), b = y.map(Double.init)
        let dot = zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
        let nx = a.reduce(0) { $0 + $1 * $1 }.squareRoot()
        let ny = b.reduce(0) { $0 + $1 * $1 }.squareRoot()
        return min(2, max(0, 1 - dot / (nx * ny)))
    }

    static func rows(dimensions: Int) -> [[Float]] {
        var state: UInt32 = 92417
        let random: [[Float]] = (0..<4).map { row in
            (0..<dimensions).map { _ -> Float in
                state = state &* 1664525 &+ 1013904223
                return Float(Int(state % 2001) - 1000) / Float(97 * (row + 1))
            }
        }
        var near = random[0]
        near[dimensions / 2] += 0.0001
        return random + [near, near.map { -$0 }, random[0].map { $0 * 16 }]
    }

    static func check<V: VectorProtocol>(_ vectors: [V]) throws where V.Scalar == Float {
        let rows = vectors.map { $0.toArray() }
        let graph = try KNNGraph.bruteForce(vectors, neighbors: vectors.count - 1, metric: .cosine)
        #expect(graph.rowOffsets == (0...rows.count).map { $0 * (rows.count - 1) })
        for i in rows.indices {
            let range = graph.rowOffsets[i]..<graph.rowOffsets[i + 1]
            #expect(Set(graph.neighborIndices[range]) == Set(rows.indices.filter { $0 != i }.map(Int32.init)))
            for edge in range {
                let j = Int(graph.neighborIndices[edge]), distance = graph.distances[edge]
                #expect(distance.isFinite && distance >= 0 && distance <= 2)
                #expect(abs(Double(distance) - cosine(rows[i], rows[j])) <= 2e-5)
                if edge > range.lowerBound {
                    let previous = graph.distances[edge - 1]
                    #expect(previous <= distance)
                    if previous.bitPattern == distance.bitPattern {
                        #expect(graph.neighborIndices[edge - 1] < graph.neighborIndices[edge])
                    }
                }
            }
        }
        #expect(vectors.map { $0.toArray().map(\.bitPattern) } == rows.map { $0.map(\.bitPattern) })
    }
}

@Suite
struct KNNMetricTests {
    @Test
    func euclideanPreservesReferenceWords() throws {
        // Independently captured from alpha 44c9ffa before metric edits, checked against
        // parent 28ee9b5 in Debug/Release: Xcode 27.2 (27B52), Swift 6.4.2,
        // arm64 macOS 27.0.1 (26A434). Cross-build words are opt-in.
        let values = UMAPProgressFixtures.vectors()
        let expected: [UInt32] = [
            1084675332, 1089900892, 1090391252, 1083689109, 1084368393, 1089875284,
            1083689109, 1088064086, 1088580131, 1087029165, 1087074753, 1090391252,
            1084531663, 1088064086, 1089770575, 1084531663, 1087074753, 1088580131,
            1084675332, 1089453298, 1089770575, 1084368393, 1087029165, 1089699138
        ]
        for (entry, graph) in [try KNNGraph.bruteForce(values, neighbors: 3),
                      try KNNGraph.bruteForce(values, neighbors: 3, metric: .euclidean)].enumerated() {
            #expect(graph.rowOffsets == Array(stride(from: 0, through: 24, by: 3)))
            #expect(graph.neighborIndices == [6, 5, 3, 2, 7, 6, 1, 4, 5, 7, 5, 0, 5, 2, 6, 4, 3, 2, 0, 5, 4, 1, 3, 2])
            try UMAPReferenceAssertions.compare(graph.distances, words: expected, tolerance: 1e-5,
                                                 caseID: "knn-\(entry == 0 ? "legacy" : "explicit")")
        }
    }

    @Test func euclideanDefaultAndExplicitMatchAtBlockBoundary() throws {
        // This entire graph was also checked against the independent alpha module.
        let values = (0..<257).map { DynamicVector([Float($0), 0]) }
        let old = try KNNGraph.bruteForce(values, neighbors: 2)
        let explicit = try KNNGraph.bruteForce(values, neighbors: 2, metric: .euclidean)
        #expect(old.rowOffsets == explicit.rowOffsets)
        #expect(old.neighborIndices == explicit.neighborIndices)
        #expect(old.distances.map(\.bitPattern) == explicit.distances.map(\.bitPattern))
        for i in values.indices {
            let indices: [Int32] = i == 0 ? [1, 2] : (i == 256 ? [255, 254] : [Int32(i - 1), Int32(i + 1)])
            let distances: [Float] = [1, i == 0 || i == 256 ? 2 : 1]
            #expect(Array(old.neighborIndices[(2 * i)..<(2 * i + 2)]) == indices)
            #expect(Array(old.distances[(2 * i)..<(2 * i + 2)]).map(\.bitPattern) == distances.map(\.bitPattern))
        }
    }

    @Test(arguments: [1, 2, 3, 7, 64, 257, 384, 512, 768, 1024, 1536, 4096])
    func cosineMatchesDoubleOracle(dimensions: Int) throws {
        let rows = KNNMetricFixtures.rows(dimensions: dimensions)
        try KNNMetricFixtures.check(rows.map(DynamicVector.init))
        if dimensions == 512 { try KNNMetricFixtures.check(rows.map { try Vector512Optimized($0) }) }
    }

    @Test func cosineAnalyticAndExtremeRows() throws {
        try KNNMetricFixtures.check([[1, 0], [7, 0], [0, 3], [-2, 0], [0, -1]].map(DynamicVector.init))
        let tiny = Float.leastNonzeroMagnitude, normal = Float.leastNormalMagnitude, huge = Float.greatestFiniteMagnitude
        let rows: [[Float]] = [[tiny, -tiny, 0], [tiny, 0, 0], [normal, -tiny, normal],
                               [huge, -huge, tiny], [-huge, huge, normal], [tiny, normal, huge]]
        try KNNMetricFixtures.check(rows.map(DynamicVector.init))
    }

    @Test func unitVectorsAgreeOnNeighbors() throws {
        let rows: [[Float]] = [0.0, 0.2, 0.7, 1.6, 2.8, 4.7].map { angle in
            let x = cos(angle), y = sin(angle), norm = (x * x + y * y).squareRoot()
            return [Float(x / norm), Float(y / norm)]
        }
        var oracleIndices: [Int32] = []
        for i in rows.indices {
            let ordered = rows.indices.filter { $0 != i }.sorted {
                KNNMetricFixtures.cosine(rows[i], rows[$0]) < KNNMetricFixtures.cosine(rows[i], rows[$1])
            }
            let distances = ordered.map { KNNMetricFixtures.cosine(rows[i], rows[$0]) }
            for j in 1..<distances.count { #expect(distances[j] - distances[j - 1] >= 1e-3) }
            oracleIndices += ordered.prefix(3).map(Int32.init)
        }
        let vectors = rows.map(DynamicVector.init)
        let cosine = try KNNGraph.bruteForce(vectors, neighbors: 3, metric: .cosine)
        let euclidean = try KNNGraph.bruteForce(vectors, neighbors: 3)
        #expect(cosine.neighborIndices == oracleIndices)
        #expect(cosine.neighborIndices == euclidean.neighborIndices)
    }

    @Test(arguments: [4, 255, 256, 257, 513])
    func cosineTiesAndBlockBoundaries(count: Int) throws {
        let axes: [[Float]] = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        let rows = (0..<count).map { axes[$0 % 4] }
        for k in count == 4 ? [1, 3] : [1, 7] {
            let graph = try KNNGraph.bruteForce(rows.map(DynamicVector.init), neighbors: k, metric: .cosine)
            #expect(graph.rowOffsets == (0...count).map { $0 * k })
            for i in rows.indices {
                // Unit coordinate axes have analytic distances 0 (same), 2 (opposite), or 1.
                let reference: [Double] = rows.indices.map { j in
                    let delta = abs(i % 4 - j % 4)
                    return delta == 0 ? 0 : (delta == 2 ? 2 : 1)
                }
                let oracle = rows.indices.filter { $0 != i }.sorted {
                    let a = reference[$0], b = reference[$1]
                    return a < b || (a.bitPattern == b.bitPattern && $0 < $1)
                }.prefix(k)
                #expect(Array(graph.neighborIndices[(i * k)..<(i * k + k)]) == oracle.map(Int32.init))
                for (edge, j) in oracle.enumerated() {
                    #expect(abs(Double(graph.distances[i * k + edge]) - reference[j]) <= 2e-5)
                }
            }
        }
    }

    @Test(arguments: [257, 513])
    func cosineBlockRowsMatchOracle(count: Int) throws {
        let values = UMAPProgressFixtures.vectors(count: count)
        let rows = values.map { $0.toArray() }
        let graph = try KNNGraph.bruteForce(values, neighbors: 3, metric: .cosine)
        for i in rows.indices {
            let reference = rows.indices.map { KNNMetricFixtures.cosine(rows[i], rows[$0]) }
            let ordered = rows.indices.filter { $0 != i }.sorted { reference[$0] < reference[$1] }
            let indices = Array(graph.neighborIndices[(i * 3)..<(i * 3 + 3)]).map(Int.init)
            #expect(Set(indices).count == 3 && !indices.contains(i))
            for (edge, j) in indices.enumerated() {
                #expect(abs(Double(graph.distances[i * 3 + edge]) - reference[j]) <= 2e-5)
                #expect(reference[j] <= reference[ordered[2]] + 4e-5)
            }
            // Only compare exact identities where the independent oracle separates the cutoff.
            if reference[ordered[3]] - reference[ordered[2]] > 4e-5 {
                #expect(Set(indices) == Set(ordered.prefix(3)))
            }
        }
    }

    @Test func cosineRejectsInvalidRows() throws {
        let valid: [[Float]] = [[1, 0], [0, 1], [-1, 0]]
        for row in [0, 2] {
            for invalid: [Float] in [[0, -0.0], [.nan, 1], [.infinity, 1], [-.infinity, 1]] {
                var rows = valid
                rows[row] = invalid
                let values = rows.map(DynamicVector.init)
                let words = values.map { $0.toArray().map(\.bitPattern) }
                do {
                    _ = try KNNGraph.bruteForce(values, neighbors: 1, metric: .cosine)
                    Issue.record("Accepted invalid cosine row")
                } catch let error as VectorError {
                    #expect(error.kind == .invalidData)
                    #expect(String(describing: error).contains("row \(row)"))
                    #expect(String(describing: error).contains(invalid.allSatisfy { $0.isZero } ? "zero" : "nonfinite"))
                }
                #expect(values.map { $0.toArray().map(\.bitPattern) } == words)
            }
        }
        do {
            _ = try KNNGraph.bruteForce([[0, 0], [.nan, 1], [1, 1]].map(DynamicVector.init), neighbors: 1, metric: .cosine)
            Issue.record("Accepted zero row before nonfinite row")
        } catch let error as VectorError {
            #expect(error.kind == .invalidData)
            #expect(String(describing: error).contains("row 0"))
            #expect(String(describing: error).contains("zero"))
        }
        do {
            _ = try KNNGraph.bruteForce([[0, 0], [1], [.nan, 1]].map(DynamicVector.init), neighbors: 1, metric: .cosine)
            Issue.record("Accepted ragged vectors")
        } catch let error as VectorError { #expect(error.kind == .dimensionMismatch) }
        #expect(throws: VectorError.self) { try KNNGraph.bruteForce([DynamicVector([0, 0])], neighbors: 1, metric: .cosine) }
        #expect(throws: VectorError.self) { try KNNGraph.bruteForce(valid.map(DynamicVector.init), neighbors: 3, metric: .cosine) }
    }

    @Test func cosineRejectsNonfiniteIntermediates() throws {
        for value: Float in [.nan, .infinity, -.infinity] {
            do {
                _ = try KNNGraph.cosineDistance(dot: value)
                Issue.record("A nonfinite dot was clamped or passed through")
            } catch let error as VectorError { #expect(error.kind == .operationFailed) }
        }
        #expect(try KNNGraph.cosineDistance(dot: 1.01).bitPattern == Float(0).bitPattern)
        #expect(try KNNGraph.cosineDistance(dot: -1.01).bitPattern == Float(2).bitPattern)
        #expect(abs(try KNNGraph.cosineDistance(dot: 0.25) - 0.75) <= Float.ulpOfOne)
    }
}
