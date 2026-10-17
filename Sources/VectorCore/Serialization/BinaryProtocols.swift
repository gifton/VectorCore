// VectorCore: Binary Serialization Protocols
//
// Protocols for binary encoding and decoding of vectors
//

import Foundation

/// Protocol for types that can be encoded to binary format
public protocol BinaryEncodable {
    /// Encode to binary data
    func encodeBinary() throws -> Data
}

/// Protocol for types that can be decoded from binary format
public protocol BinaryDecodable {
    /// Decode from binary data
    static func decodeBinary(from data: Data) throws -> Self
}

/// Combined protocol for types that support both binary encoding and decoding.
///
/// `BinaryCodable` is a convenience type alias that combines `BinaryEncodable`
/// and `BinaryDecodable` protocols, similar to Swift's standard `Codable`.
/// Built-in vector methods do not declare these protocol conformances. Use a
/// consumer-owned adapter with explicit witnesses (B01PublicConsumer exercises
/// concrete, generic and existential calls):
///
/// ```swift
/// struct StoredVector: BinaryCodable {
///     let value: DynamicVector
///     func encodeBinary() throws -> Data { value.encodeBinary() }
///     static func decodeBinary(from data: Data) throws -> StoredVector {
///         StoredVector(value: try DynamicVector.decodeBinary(from: data))
///     }
/// }
/// ```
/// The delegated nonthrowing encoder accepts empty values. This is distinct
/// from the positive-domain checked default on VectorType & BinaryEncodable.
public typealias BinaryCodable = BinaryEncodable & BinaryDecodable

// MARK: - Binary Format Header

/// Header structure for binary encoded vectors.
///
/// `BinaryHeader` provides a standardized header format for binary vector
/// serialization, enabling version compatibility checks and basic validation.
/// The header is designed to be compact (12 bytes) while providing essential
/// metadata for safe deserialization.
///
/// ## Binary Layout
/// - Bytes 0-3: Numeric magic 0x56454354; little-endian bytes 54 43 45 56
/// - Bytes 4-5: Version number
/// - Bytes 6-9: Vector dimension
/// - Bytes 10-11: Reserved flags; encoders emit zero, version1 decoders ignore them
/// - Bytes 12..<(12+4N): N little-endian Float bit-pattern words
/// - Final 4 bytes: little-endian CRC32 of header and payload
///
/// Total size is 16+4N. CRC uses reflected polynomial 0xEDB88320 and initial/final
/// xor 0xffffffff; it detects corruption, not malicious modification. Complete
/// independent frames are exercised by B01BinaryContractTests.
public struct BinaryHeader {
    /// Numeric magic 0x56454354; encoded little-endian as 54 43 45 56.
    public static let magic: UInt32 = 0x56454354

    /// Current binary format version.
    public static let version: UInt16 = 1

    /// Magic number for format validation.
    public let magic: UInt32

    /// Binary format version for compatibility checking.
    public let version: UInt16

    /// Number of elements in the encoded vector.
    public let dimension: UInt32

    /// Reserved flags for future format extensions.
    public let flags: UInt16

    public init(dimension: Int) {
        self.magic = Self.magic
        self.version = Self.version
        self.dimension = UInt32(dimension)
        self.flags = 0
    }

    public static var headerSize: Int {
        return MemoryLayout<UInt32>.size +  // magic
            MemoryLayout<UInt16>.size +  // version
            MemoryLayout<UInt32>.size +  // dimension
            MemoryLayout<UInt16>.size    // flags
    }
}
