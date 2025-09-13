import Foundation
import Testing
@testable import VectorCore

@Suite("Serialization")
struct SerializationSuite {

    @Test
    func testBinaryRoundTrip_VectorFixedDim() {
        let vals = Array(0..<16).map { Float($0) }
        let v = try! Vector<Dim16>(vals)
        let data = v.encodeBinary()
        let decoded = try! Vector<Dim16>.decodeBinary(from: data)
        for i in 0..<16 { #expect(approxEqual(decoded[i], vals[i])) }
    }

    @Test
    func testBinaryRoundTrip_DynamicVector() {
        let vals = Array(0..<10).map { Float($0) }
        let v = DynamicVector(vals)
        let data = v.encodeBinary()
        let decoded = try! DynamicVector.decodeBinary(from: data)
        #expect(decoded.scalarCount == v.scalarCount)
        for i in 0..<v.scalarCount { #expect(approxEqual(decoded[i], vals[i])) }
    }

    @Test
    func testBinaryDecode_InvalidMagicThrows() {
        let v = Vector<Dim8>.ones
        var data = v.encodeBinary()
        // Overwrite magic (bytes 0-3)
        let badMagic: UInt32 = 0x00000000
        withUnsafeBytes(of: badMagic.littleEndian) { bytes in
            data.replaceSubrange(0..<4, with: bytes)
        }
        do {
            _ = try Vector<Dim8>.decodeBinary(from: data)
            Issue.record("Expected invalid magic error not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .invalidData)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test
    func testBinaryDecode_UnsupportedVersionThrows() {
        let v = Vector<Dim8>.ones
        var data = v.encodeBinary()
        // Overwrite version (bytes 4-5)
        let badVersion: UInt16 = 2
        withUnsafeBytes(of: badVersion.littleEndian) { bytes in
            data.replaceSubrange(4..<6, with: bytes)
        }
        do {
            _ = try Vector<Dim8>.decodeBinary(from: data)
            Issue.record("Expected unsupported version error not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .invalidData)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test
    func testBinaryDecode_InvalidTotalSizeThrows() {
        let v = Vector<Dim8>.ones
        var data = v.encodeBinary()
        // Truncate one byte from the end
        data.removeLast()
        do {
            _ = try Vector<Dim8>.decodeBinary(from: data)
            Issue.record("Expected invalid data size error not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .insufficientData)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test
    func testBinaryDecode_FixedDimensionMismatchThrows() {
        let v = Vector<Dim16>.ones
        let data = v.encodeBinary()
        do {
            _ = try Vector<Dim8>.decodeBinary(from: data)
            Issue.record("Expected dimension mismatch not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .dimensionMismatch)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test
    func testBinaryDecode_ChecksumMismatchThrows_DataTamper() {
        let v = Vector<Dim8>.ones
        var data = v.encodeBinary()
        // Flip a payload byte (not header, not checksum)
        let payloadIndex = 12 // after 12-byte header
        data[payloadIndex] ^= 0xFF
        do {
            _ = try Vector<Dim8>.decodeBinary(from: data)
            Issue.record("Expected checksum mismatch not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .dataCorruption)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test
    func testBinaryDecode_ChecksumMismatchThrows_ChecksumTamper() {
        let v = Vector<Dim8>.ones
        var data = v.encodeBinary()
        // Corrupt the last 4 bytes (checksum)
        let checksumOffset = data.count - 4
        data[checksumOffset] ^= 0xAA
        do {
            _ = try Vector<Dim8>.decodeBinary(from: data)
            Issue.record("Expected checksum mismatch not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .dataCorruption)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test
    func testBase64RoundTrip_VectorFixedDim() {
        let vals = Array(0..<8).map { _ in Float.random(in: -1...1) }
        let v = try! Vector<Dim8>(vals)
        let b64 = v.base64Encoded
        let decoded = try! Vector<Dim8>.base64Decoded(from: b64)
        for i in 0..<8 { #expect(approxEqual(decoded[i], v[i])) }
    }

    @Test
    func testBase64Decode_InvalidStringThrows() {
        let invalid = "not-base64!!!"
        do {
            _ = try Vector<Dim8>.base64Decoded(from: invalid)
            Issue.record("Expected invalidDataFormat not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .invalidData)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test
    func testJSONRoundTrip_VectorFixedDim() {
        let vals = Array(0..<8).map { Float($0) }
        let v = try! Vector<Dim8>(vals)
        let data = try! v.encodeJSON()
        let decoded = try! Vector<Dim8>.decodeJSON(from: data)
        for i in 0..<8 { #expect(approxEqual(decoded[i], vals[i])) }
    }

    @Test
    func testJSONRoundTrip_DynamicVector() {
        let vals = Array(0..<5).map { Float($0) }
        let v = DynamicVector(vals)
        let data = try! JSONEncoder().encode(v)
        let decoded = try! DynamicVector.decodeJSON(from: data)
        #expect(decoded.scalarCount == 5)
        for i in 0..<5 { #expect(approxEqual(decoded[i], vals[i])) }
    }

    @Test
    func testJSONDecode_InvalidUTF8StringThrows() {
        // Swift strings are always valid UTF-8 when encoded.
        // Instead, verify that non-JSON content throws a decoding error.
        let notJSON = "this is not json"
        do {
            _ = try Vector<Dim8>.decodeJSON(from: notJSON)
            Issue.record("Expected decoding error not thrown")
        } catch {
            // Any error here is acceptable (DecodingError or VectorError)
        }
    }

    @Test
    func testDynamicJSONDecode_MismatchedElementsThrows() {
        // dimension says 4, but provide 3 elements
        let badJSON = "{" +
            "\"dimension\":4," +
            "\"elements\":[1,2,3]" +
            "}"
        do {
            _ = try DynamicVector.decodeJSON(from: badJSON)
            Issue.record("Expected dataCorrupted decoding error not thrown")
        } catch {
            // Expected to throw
        }
    }

    @Test
    func testBinaryDecode_ZeroLengthThrows() {
        let empty = Data()
        do {
            _ = try Vector<Dim8>.decodeBinary(from: empty)
            Issue.record("Expected error for empty data not thrown")
        } catch let e as VectorError {
            #expect(e.kind == .insufficientData)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test
    func testBinaryRoundTrip_RandomizedFuzzSmall() {
        for _ in 0..<10 {
            let vals = Array(0..<8).map { _ in Float.random(in: -10...10) }
            let v = try! Vector<Dim8>(vals)
            let data = v.encodeBinary()
            let decoded = try! Vector<Dim8>.decodeBinary(from: data)
            for i in 0..<8 { #expect(approxEqual(decoded[i], vals[i])) }
        }
    }

}
