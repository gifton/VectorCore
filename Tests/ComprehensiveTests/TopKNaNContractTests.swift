import Testing
@testable import VectorCore

@Suite("Top-K NaN ordering contract")
struct TopKNaNContractTests {
    private let canonical: [Float] = [.nan, 2, -.infinity, 2, .infinity, -0.0, 0.0, .nan, -3]
    private let ascending = [2, 8, 5, 6, 1, 3, 4, 0, 7]
    private let descending = [4, 1, 3, 5, 6, 8, 2, 0, 7]

    private func checkValues(_ values: [Float], indices: [Int], scores: [Float]) {
        #expect(values.count == indices.count)
        for (actual, index) in zip(values, indices) {
            let expected = scores[index]
            if expected.isNaN {
                #expect(actual.isNaN)
            } else {
                #expect(actual.bitPattern == expected.bitPattern)
            }
        }
    }

    private struct ScoredElement {
        let label: Int
        let score: Float
    }

    private func checkPublicPaths(_ scores: [Float], k: Int, expected: [Int], policy: TieBreaker = .smallerIndex) {
        let array = TopKSelection.select(k: k, from: scores, tieBreaker: policy)
        #expect(array.indices == expected)
        checkValues(array.distances, indices: array.indices, scores: scores)
        scores.withUnsafeBufferPointer { buffer in
            let pointer = TopKSelection.select(k: k, from: buffer.baseAddress!, count: buffer.count, tieBreaker: policy)
            #expect(pointer.indices.map(Int.init) == expected)
            checkValues(pointer.distances, indices: pointer.indices.map(Int.init), scores: scores)
        }
        let elements = scores.enumerated().map { ScoredElement(label: 1000 - $0.offset, score: $0.element) }
        let generic = TopKSelection.select(k: k, from: elements, distance: { $0.score }, tieBreaker: policy)
        #expect(generic.map(\.label) == expected.map { 1000 - $0 })
        checkValues(generic.map(\.score), indices: generic.map { 1000 - $0.label }, scores: scores)
    }

    private func extract(_ heap: TopKBuffer) -> TopKResult {
        heap.isMinHeap
            ? TopKSelection.extractSortedResultDescending(from: heap)
            : TopKSelection.extractSortedResult(from: heap, sqrt: false)
    }

    @Test func heapEvictsInitialNaN() {
        for maximize in [false, true] {
            var heap = TopKBuffer(k: 1, isMinHeap: maximize)
            heap.pushIfBetter(val: .nan, idx: 0)
            heap.pushIfBetter(val: 4, idx: 1)
            #expect(heap.size == 1)
            #expect(heap.idxs[0] == 1)
            #expect(heap.vals[0] == 4)
        }
    }

    @Test func publicSelectionKeepsNaNsLastAndPreservesCount() {
        let scores: [Float] = [.nan, 2, -.infinity, 2, .infinity, -0.0, 0.0, .nan, -3]
        let result = TopKSelection.select(k: 99, from: scores)
        #expect(result.indices == [2, 8, 5, 6, 1, 3, 4, 0, 7])
        #expect(result.count == scores.count)
        for position in result.indices.indices {
            let expected = scores[result.indices[position]]
            let actual = result.distances[position]
            if expected.isNaN {
                #expect(actual.isNaN)
            } else {
                #expect(actual.bitPattern == expected.bitPattern)
            }
        }
    }

    @Test func canonicalMembershipAndOrderBothDirections() {
        for maximize in [false, true] {
            for k in [1, 5, 9] {
                let expected = Array((maximize ? descending : ascending).prefix(k))
                for reverse in [false, true] {
                    var heap = TopKBuffer(k: k, isMinHeap: maximize)
                    let indices = reverse ? Array(canonical.indices.reversed()) : Array(canonical.indices)
                    for index in indices { heap.pushIfBetter(val: canonical[index], idx: index) }
                    let result = extract(heap)
                    #expect(result.indices == expected)
                    checkValues(result.distances, indices: result.indices, scores: canonical)
                }
                if !maximize { checkPublicPaths(canonical, k: k, expected: expected) }
            }
        }
    }

    @Test func publicHeapEvictionAndCrossover() {
        var scores = [Float](repeating: 100, count: 100)
        scores.replaceSubrange(0..<7, with: [.nan, .nan, .nan, 2, 1, 1, 1])
        checkPublicPaths(scores, k: 3, expected: [4, 5, 6])
        checkPublicPaths(scores, k: 9, expected: [4, 5, 6, 3, 7, 8, 9, 10, 11])
        checkPublicPaths(scores, k: 10, expected: [4, 5, 6, 3, 7, 8, 9, 10, 11, 12])
    }

    @Test func nanCandidatesFillRemainingPositions() {
        var scores = [Float](repeating: .nan, count: 100)
        checkPublicPaths(scores, k: 3, expected: [0, 1, 2])
        checkPublicPaths(Array(scores.prefix(5)), k: 3, expected: [0, 1, 2])
        scores[99] = 1
        checkPublicPaths(scores, k: 3, expected: [99, 0, 1])
    }

    @Test func numericInfinityOutranksNaN() {
        checkPublicPaths([.nan, .infinity], k: 1, expected: [1])
        var heap = TopKBuffer(k: 1, isMinHeap: true)
        heap.pushIfBetter(val: .nan, idx: 0)
        heap.pushIfBetter(val: -.infinity, idx: 1)
        #expect(extract(heap).indices == [1])
    }

    @Test func exactTiesAndSignedZeroKeepOriginalPositionsAndBits() {
        checkPublicPaths([Float](repeating: 1, count: 100), k: 3, expected: [0, 1, 2])
        checkPublicPaths([0.0, -0.0, 0.0, -0.0], k: 3, expected: [0, 1, 2])
        // Adjacent representable scores must not be treated as approximate ties.
        checkPublicPaths([Float(1).nextUp, 1], k: 1, expected: [1])
        for maximize in [false, true] {
            var heap = TopKBuffer(k: 3, isMinHeap: maximize)
            for index in (0..<100).reversed() { heap.pushIfBetter(val: 1, idx: index) }
            #expect(extract(heap).indices == [0, 1, 2])
        }
    }

    @Test func pointerIDsAreLabelsAfterSelection() {
        for ids: [Int32] in [[90, 10, 50], [90, 90, 10]] {
            for scores: [Float] in [[1, 1, 1], [.nan, .nan, .nan], [.nan, 1, .nan]] {
                for heapPath in [false, true] {
                    let allScores = heapPath ? scores + [Float](repeating: .nan, count: 97) : scores
                    let allIDs = heapPath ? ids + [Int32](repeating: -1, count: 97) : ids
                    let positions = scores[0].isNaN && !scores[1].isNaN ? [1, 0] : [0, 1]
                    allScores.withUnsafeBufferPointer { values in
                        allIDs.withUnsafeBufferPointer { labels in
                            let result = TopKSelection.select(k: 2, from: values.baseAddress!, count: values.count, ids: labels.baseAddress!)
                            #expect(result.indices == positions.map { ids[$0] })
                            checkValues(result.distances, indices: positions, scores: scores)
                        }
                    }
                }
            }
        }
    }

    @Test func explicitTiePoliciesRespectTheirScope() {
        for policy in [TieBreaker.smallerIndex, .insertionOrder] {
            checkPublicPaths(canonical, k: 9, expected: ascending, policy: policy)
            checkPublicPaths([Float](repeating: .nan, count: 100), k: 3, expected: [0, 1, 2], policy: policy)
        }
        #expect(TopKSelection.select(k: 5, from: canonical).indices == TopKSelection.select(k: 5, from: canonical, tieBreaker: .smallerIndex).indices)
        for scores in [canonical, [.nan, 1, 1] + [Float](repeating: .nan, count: 97)] {
            let result = TopKSelection.select(k: 9, from: scores, tieBreaker: .smallerValue)
            #expect(result.count == 9)
            #expect(Set(result.indices).count == 9)
            checkValues(result.distances, indices: result.indices, scores: scores)
            var sawNaN = false
            var previous: Float = -.infinity
            for score in result.distances {
                if score.isNaN { sawNaN = true } else {
                    #expect(!sawNaN)
                    #expect(score >= previous)
                    previous = score
                }
            }
        }
    }

    @Test func disjointMergesKeepGlobalIndicesAndDestinationPolicy() {
        let fixtures: [([Float], [Int], [Int])] = [
            (canonical, ascending, descending),
            ([Float](repeating: .nan, count: 9), Array(0..<9), Array(0..<9)),
            ([Float](repeating: 1, count: 9), Array(0..<9), Array(0..<9))
        ]
        for (scores, minOrder, maxOrder) in fixtures {
            for maximize in [false, true] {
                for policy in [TieBreaker.smallerIndex, .insertionOrder, .smallerValue] {
                    for k in [3, 5, 9] {
                        for chunks in [[[0, 3, 6], [1, 4, 7], [2, 5, 8]], [[0, 1], [2, 3, 4, 5], [6, 7, 8]]] {
                            let partials = chunks.map { indices in
                                var heap = TopKBuffer(k: k, isMinHeap: maximize, tieBreaker: policy)
                                for index in indices.reversed() { heap.pushIfBetter(val: scores[index], idx: index) }
                                return heap
                            }
                            for order in [[0, 1, 2], [2, 1, 0], [1, 0, 2]] {
                                var output = TopKBuffer(k: k, isMinHeap: maximize, tieBreaker: policy)
                                for part in order {
                                    let previous = output
                                    TopKSelectionKernels.mergeTopK(previous, partials[part], into: &output)
                                }
                                #expect(output.tieBreaker == policy)
                                let result = extract(output)
                                #expect(result.count == k)
                                if policy != .smallerValue {
                                    #expect(result.indices == Array((maximize ? maxOrder : minOrder).prefix(k)))
                                } else {
                                    // Equivalent identities are unspecified; score ranks still match the literal oracle.
                                    for (actual, index) in zip(result.distances, (maximize ? maxOrder : minOrder).prefix(k)) {
                                        let expected = scores[index]
                                        #expect((actual.isNaN && expected.isNaN) || actual == expected)
                                    }
                                }
                                checkValues(result.distances, indices: result.indices, scores: scores)
                            }
                        }
                    }
                }
            }
        }
    }

    @Test func publicDotProductMaximizesWithNaNsLast() throws {
        let query = try Vector512Optimized([Float](repeating: 1, count: 512))
        let candidates = try [Float.nan, 2, 2, -1, .nan].map {
            try Vector512Optimized([Float](repeating: $0, count: 512))
        }
        let scores = candidates.map { DotKernels.dot512(query, $0) }
        #expect(scores[0].isNaN && scores[4].isNaN)
        #expect(scores[1] == 1024 && scores[2] == 1024 && scores[3] == -512)
        let result = TopKSelection.nearestDotProduct512(k: 4, query: query, candidates: candidates)
        #expect(result.indices == [1, 2, 3, 0])
        checkValues(result.distances, indices: result.indices, scores: scores)
    }

    @Test func existingPublicSelectionEdges() {
        #expect(TopKSelection.select(k: 3, from: [Float]()).isEmpty)
        #expect(TopKSelection.select(k: 3, from: [ScoredElement](), distance: { $0.score }).isEmpty)
        canonical.withUnsafeBufferPointer { buffer in
            let result = TopKSelection.select(k: 3, from: buffer.baseAddress!, count: 0)
            #expect(result.indices.isEmpty && result.distances.isEmpty)
        }
        checkPublicPaths(canonical, k: 0, expected: [])
        checkPublicPaths(canonical, k: 1, expected: [2])
        checkPublicPaths(canonical, k: 9, expected: ascending)
        checkPublicPaths(canonical, k: 99, expected: ascending)
    }

    @Test func comparatorLaws() {
        let candidates: [(Int, Float)] = canonical.enumerated().map { ($0.offset, $0.element) }
        for maximize in [false, true] {
            for policy in [TieBreaker.smallerIndex, .insertionOrder, .smallerValue] {
                let before: ((Int, Float), (Int, Float)) -> Bool = {
                    TopKSelection.orderedBefore($0, $1, descending: maximize, tieBreaker: policy)
                }
                for a in candidates {
                    #expect(!before(a, a))
                    for b in candidates {
                        if before(a, b) { #expect(!before(b, a)) }
                        if policy != .smallerValue && a.0 != b.0 {
                            #expect(before(a, b) || before(b, a))
                        }
                        for c in candidates {
                            if before(a, b) && before(b, c) { #expect(before(a, c)) }
                            let abEquivalent = !before(a, b) && !before(b, a)
                            let bcEquivalent = !before(b, c) && !before(c, b)
                            if abEquivalent && bcEquivalent {
                                #expect(!before(a, c) && !before(c, a))
                            }
                        }
                    }
                }
            }
        }
    }
}
