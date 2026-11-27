import Foundation
import Testing

@Suite
struct UMAPReferenceProfileTests {
    @Test(arguments: [Float(1).nextUp, Float(2)])
    func alternateProfileSuppliesItsOwnWords(_ actual: Float) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let context: [String: Any] = [
            "runID": "alternate-fixture", "configuration": "debug", "profile": "alternate-fixture",
            "fixtureSHA256": "fixture", "directory": directory.path,
            "oracle": ["alternate": [actual.bitPattern]]
        ]
        let encoded = String(decoding: try JSONSerialization.data(withJSONObject: context), as: UTF8.self)
        try UMAPReferenceAssertions.compare([actual], words: [Float(1).bitPattern], tolerance: 1e-6,
                                            caseID: "alternate", exact: true, contextJSON: encoded)
        let receipts = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        try #require(receipts.count == 1)
        let receipt = try JSONSerialization.jsonObject(with: Data(contentsOf: receipts[0])) as? [String: Any]
        #expect(receipt?["mode"] as? String == "exact")
        #expect(receipt?["words"] as? [UInt32] == [actual.bitPattern])
    }

    @Test func ordinaryModeRetainsHistoricalTolerance() throws {
        try UMAPReferenceAssertions.compare([Float(1).nextUp], words: [Float(1).bitPattern], tolerance: 1e-6,
                                            caseID: "ordinary", exact: false, contextJSON: nil)
    }
}
