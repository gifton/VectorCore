import Foundation
import Testing
@testable import VectorCore

/// P01 (OWN-07) optimized-vector hash contract at 384/512/768/1536.
///
/// The lanes hashed before P01 phase B were the first two and last two SIMD4 blocks, lanes [0, 8) and [D-8, D).
/// Every corpus here varies ONLY lanes in [8, D-8), including late lanes D-12...D-9, so the former sampled hash gave
/// one value for the whole corpus. The suite passes since phase B (ungated); each corpus has at least 64 rows.
/// Hash values are process-seeded: nothing here hard-codes a hash or requires all hashes to be unique.
@Suite
struct OptimizedHashContractTests {
    static let widths = [384, 512, 768, 1536]

    /// Interior lanes only: the first interior lane, a stride across the middle and the late lanes.
    /// At least 64 lanes at every width (the stride yields about 100 to 120).
    static func interiorLanes(_ width: Int) -> [Int] {
        var lanes = Set([8, 9, 10, 11, 12, width - 12, width - 11, width - 10, width - 9])
        for lane in stride(from: 16, to: width - 16, by: max(1, (width - 32) / 96)) { lanes.insert(lane) }
        return lanes.filter { $0 >= 8 && $0 < width - 8 }.sorted()
    }

    static let minimumCorpusRows = 64

    static func denseBase(_ width: Int) -> [Float] { (0..<width).map { Float($0 % 13) + 1 } }

    /// Sparse: one interior lane set to 1 on the zero vector. Dense: one interior lane of the dense base changed.
    static func corpora(_ width: Int) -> [(name: String, rows: [[Float]])] {
        let lanes = interiorLanes(width)
        precondition(lanes.count >= minimumCorpusRows, "corpus must have at least \(minimumCorpusRows) rows")
        let sparse = lanes.map { lane -> [Float] in
            var row = [Float](repeating: 0, count: width); row[lane] = 1; return row
        }
        let dense = lanes.enumerated().map { offset, lane -> [Float] in
            var row = denseBase(width); row[lane] = -Float(offset + 1) * 0.5; return row
        }
        return [("sparse", sparse), ("dense", dense)]
    }

    private func make<V: VectorProtocol>(_ type: V.Type, _ rows: [[Float]]) throws -> [V] where V.Scalar == Float {
        try rows.map { try V($0) }
    }

    @Test(arguments: widths)
    func interiorOnlyCorpusDoesNotCollapse(width: Int) throws {
        switch width {
        case 384: try checkInterior(Vector384Optimized.self, width: width)
        case 512: try checkInterior(Vector512Optimized.self, width: width)
        case 768: try checkInterior(Vector768Optimized.self, width: width)
        default: try checkInterior(Vector1536Optimized.self, width: width)
        }
    }

    private func checkInterior<V: VectorProtocol>(_ type: V.Type, width: Int) throws where V.Scalar == Float {
        for (name, rows) in Self.corpora(width) {
            // Fixture self-checks: every row keeps the formerly hashed lanes of its base, and rows are unequal.
            let base = name == "sparse" ? [Float](repeating: 0, count: width) : Self.denseBase(width)
            for row in rows {
                #expect(Array(row[0..<8]) == Array(base[0..<8]) && Array(row[(width - 8)...]) == Array(base[(width - 8)...]))
            }
            #expect(rows.count >= Self.minimumCorpusRows, "\(name) corpus has \(rows.count) rows at width \(width)")
            let vectors = try make(type, rows)
            for i in vectors.indices { for j in vectors.indices where j > i { #expect(vectors[i] != vectors[j]) } }
            let distinct = Set(vectors.map(\.hashValue)).count
            p01Marker("P01_OBSERVED hash width=\(width) corpus=\(name) rows=\(vectors.count) distinctHashes=\(distinct)")
            #expect(distinct > 1, """
                interior-only \(name) corpus of \(vectors.count) unequal vectors collapsed to \(distinct) \
                hash value(s) at width \(width)
                """)
        }
    }

    @Test(arguments: widths)
    func equalValuesHashEquallyAcrossConstructions(width: Int) throws {
        switch width {
        case 384:
            try checkEqual(width: width, array: { try Vector384Optimized($0) },
                           raw: { Vector384Optimized(fromSIMDStorage: $0) }, generated: { Vector384Optimized(generator: $0) })
        case 512:
            try checkEqual(width: width, array: { try Vector512Optimized($0) },
                           raw: { Vector512Optimized(fromSIMDStorage: $0) }, generated: { Vector512Optimized(generator: $0) })
        case 768:
            try checkEqual(width: width, array: { try Vector768Optimized($0) },
                           raw: { Vector768Optimized(fromSIMDStorage: $0) }, generated: { Vector768Optimized(generator: $0) })
        default:
            try checkEqual(width: width, array: { try Vector1536Optimized($0) },
                           raw: { Vector1536Optimized(fromSIMDStorage: $0) }, generated: { Vector1536Optimized(generator: $0) })
        }
    }

    private func checkEqual<V: VectorProtocol>(
        width: Int, array: ([Float]) throws -> V, raw: (ContiguousArray<SIMD4<Float>>) -> V,
        generated: ((Int) -> Float) -> V
    ) throws where V.Scalar == Float {
        let values = Self.denseBase(width)
        let fromArray = try array(values)
        let copy = fromArray
        var blocks = ContiguousArray<SIMD4<Float>>()
        for b in 0..<(width / 4) { blocks.append(SIMD4(values[4 * b], values[4 * b + 1], values[4 * b + 2], values[4 * b + 3])) }
        let fromRaw = raw(blocks)
        let fromGenerator = generated { values[$0] }
        for other in [copy, fromRaw, fromGenerator] {
            #expect(other == fromArray)
            #expect(other.hashValue == fromArray.hashValue)
        }
    }

    @Test(arguments: widths)
    func signedZeroReplacementInEveryBlockKeepsEqualHashes(width: Int) throws {
        switch width {
        case 384: try checkSignedZero(Vector384Optimized.self, width: width)
        case 512: try checkSignedZero(Vector512Optimized.self, width: width)
        case 768: try checkSignedZero(Vector768Optimized.self, width: width)
        default: try checkSignedZero(Vector1536Optimized.self, width: width)
        }
    }

    private func checkSignedZero<V: VectorProtocol>(_ type: V.Type, width: Int) throws where V.Scalar == Float {
        for block in 0..<(width / 4) {
            for lane in 0..<4 {
                var positive = Self.denseBase(width), negative = positive
                positive[4 * block + lane] = 0
                negative[4 * block + lane] = -0.0
                let a = try V(positive), b = try V(negative)
                #expect(a == b, "+0/-0 at block \(block) lane \(lane) must compare equal")
                #expect(a.hashValue == b.hashValue, "+0/-0 at block \(block) lane \(lane) must hash equally")
            }
        }
    }

    @Test(arguments: widths)
    func setAndDictionaryKeepEveryUnequalValue(width: Int) throws {
        switch width {
        case 384: try checkTables(Vector384Optimized.self, width: width)
        case 512: try checkTables(Vector512Optimized.self, width: width)
        case 768: try checkTables(Vector768Optimized.self, width: width)
        default: try checkTables(Vector1536Optimized.self, width: width)
        }
    }

    private func checkTables<V: VectorProtocol>(_ type: V.Type, width: Int) throws where V.Scalar == Float {
        for (name, rows) in Self.corpora(width) {
            #expect(rows.count >= Self.minimumCorpusRows, "\(name) corpus has \(rows.count) rows at width \(width)")
            let vectors = try make(type, rows)
            let set = Set(vectors)
            #expect(set.count == vectors.count)
            var table: [V: Int] = [:]
            for (index, vector) in vectors.enumerated() { table[vector] = index }
            #expect(table.count == vectors.count)
            for (index, row) in rows.enumerated() {
                let rebuilt = try V(row)  // independently constructed equal value
                #expect(set.contains(rebuilt))
                #expect(table[rebuilt] == index)
            }
        }
    }
}
