import Foundation
import Testing
@testable import VectorCore

/// P01 (API-45) `VectorMath.nearestNeighbors` law: ascending numeric scores before NaNs, exact ties by smaller
/// original index, min(k, N) results, k <= 0 and empty candidates return [] without metric calls, and positive k
/// evaluates the metric exactly once per candidate in index order. Was RED before P01 phase D; phase D lifted
/// `P01RedGate.vectorMath`, so the suite runs by default.
/// The expected orders come from `oracle`, an explicit key comparison that does not call production selection.
@Suite(.serialized, .enabled(if: P01RedGate.vectorMath))
struct VectorMathNearestContractTests {
    static let query = Vector<Dim2>(x: 0, y: 0)

    /// Candidate i carries i in coordinate 0, so the metric can look up its fixture score.
    static func candidates(_ count: Int) -> [Vector<Dim2>] {
        (0..<count).map { Vector<Dim2>(x: Float($0), y: 0) }
    }

    static func run(_ scores: [Float], k: Int, calls: P01Log) -> [(index: Int, distance: Float)] {
        VectorMath.nearestNeighbors(query: query, in: candidates(scores.count), k: k) { _, candidate in
            let index = Int(candidate[0])
            calls.append(index)
            return scores[index]
        }
    }

    /// Independent oracle: numeric before NaN, then ascending value, then smaller original index.
    static func oracle(_ scores: [Float], k: Int) -> [Int] {
        guard k > 0 else { return [] }
        let order = scores.indices.sorted { a, b in
            let aNaN = scores[a].isNaN, bNaN = scores[b].isNaN
            if aNaN != bNaN { return !aNaN }
            if !aNaN && scores[a] != scores[b] { return scores[a] < scores[b] }
            return a < b
        }
        return Array(order.prefix(k))
    }

    private func checkResult(_ result: [(index: Int, distance: Float)], scores: [Float], k: Int, label: String) {
        let expected = Self.oracle(scores, k: k)
        #expect(result.map(\.index) == expected, "\(label) k=\(k): indices \(result.map(\.index)) expected \(expected)")
        for item in result {
            let reference = scores[item.index]
            #expect(reference.isNaN ? item.distance.isNaN : item.distance.bitPattern == reference.bitPattern,
                    "\(label) k=\(k): score bits for index \(item.index)")
        }
    }

    @Test func zeroKEvaluatesNoMetric() {
        let calls = P01Log()
        let result = Self.run([3, 1, 2, 0, 4], k: 0, calls: calls)
        p01Marker("P01_OBSERVED vectormath k=0 calls=\(calls.count) results=\(result.count)")
        #expect(result.isEmpty)
        #expect(calls.count == 0, "k=0 evaluated the metric \(calls.count) time(s); expected none")
    }

    @Test func emptyCandidatesEvaluateNoMetric() {
        let calls = P01Log()
        #expect(Self.run([], k: 3, calls: calls).isEmpty)
        #expect(calls.count == 0)
    }

    @Test func negativeKReturnsEmptyWithoutMetricInChild() throws {
        let child = try p01RunChild("vectormath-negative-k", filter: "VectorMathNearestContractTests/p01Child")
        #expect(child.run.terminationStatus == 0 && child.run.terminationReason == .exit,
                "negative k child status=\(child.run.statusDescription) reason=\(child.run.reasonDescription): \(child.tail)")
        #expect(child.output.contains("P01_CHILD_COMPLETE vectormath-negative-k"), "\(child.tail)")
    }

    @Test func numericScoresPrecedeNaNsWithSmallerIndexTies() {
        let fixtures: [[Float]] = [[.nan, 1, 0], [.nan, 2, 1, .nan, 0], [1, .nan, 0], [.nan, 5, .nan]]
        for scores in fixtures {
            for k in 1...(scores.count + 1) {
                let calls = P01Log()
                let result = Self.run(scores, k: k, calls: calls)
                p01Marker("P01_OBSERVED vectormath scores=\(scores) k=\(k) indices=\(result.map(\.index))")
                #expect(result.count == min(k, scores.count))
                checkResult(result, scores: scores, k: k, label: "\(scores)")
            }
        }
    }

    @Test func positiveKCallsEachCandidateOnceInIndexOrder() {
        for k in [1, 2, 7, 8, Int.max] {
            let calls = P01Log()
            _ = Self.run([4, 2, .nan, 0, -.infinity, .infinity, 2], k: k, calls: calls)
            #expect(calls.snapshot() == Array(0..<7), "k=\(k): calls \(calls.snapshot())")
        }
    }

    @Test func signedZeroTiesKeepSmallerIndexAndBits() {
        let scores: [Float] = [0, -0.0, 0, -0.0]
        let calls = P01Log()
        checkResult(Self.run(scores, k: 4, calls: calls), scores: scores, k: 4, label: "signed zeros")
    }

    /// Small/large strategy neighbours with NaNs first, middle and last, ties and infinities.
    @Test(arguments: [(20, 1), (20, 2), (100, 9), (100, 10), (100, 99), (100, 100), (100, 101)])
    func strategyNeighboursMatchOracle(count: Int, k: Int) {
        let scores: [Float] = (0..<count).map { index in
            if index % 7 == 0 || index == count - 1 { return .nan }
            if index % 11 == 0 { return index % 2 == 0 ? .infinity : -.infinity }
            return Float(index % 5) - 2
        }
        let calls = P01Log()
        let result = Self.run(scores, k: k, calls: calls)
        #expect(calls.snapshot() == Array(0..<count))
        #expect(result.count == min(k, count))
        checkResult(result, scores: scores, k: k, label: "N=\(count)")
    }

    /// Child entry; it returns at once unless the parent selected a scenario.
    @Test func p01Child() throws {
        guard let scenario = ProcessInfo.processInfo.environment["P01_SCENARIO"] else { return }
        p01Marker("P01_CHILD_BEGIN \(scenario)")
        switch scenario {
        case "vectormath-negative-k":
            let calls = P01Log()
            let result = VectorMath.nearestNeighbors(query: Self.query, in: Self.candidates(5), k: -1) { _, candidate in
                let index = Int(candidate[0])
                calls.append(index)
                p01Marker("P01_METRIC_CALL \(index)")
                return Float(index)
            }
            p01Marker("P01_OBSERVED vectormath k=-1 calls=\(calls.count) results=\(result.count)")
            #expect(result.isEmpty)
            #expect(calls.count == 0)
        default:
            Issue.record("Unknown P01 scenario \(scenario)")
        }
        p01Marker("P01_CHILD_COMPLETE \(scenario)")
    }
}
