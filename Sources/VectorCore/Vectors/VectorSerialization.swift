//
//  VectorSerialization.swift
//  VectorCore
//
//

import Foundation

// MARK: - Codable Conformance

extension Vector: Codable {
    private enum CodingKeys: String, CodingKey {
        case elements
    }
    
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let elements = try container.decode([Scalar].self, forKey: .elements)
        try self.init(elements)
    }
    
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(toArray(), forKey: .elements)
    }
}

extension DynamicVector: Codable {
    private enum CodingKeys: String, CodingKey {
        case dimension
        case elements
    }
    
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let dimension = try container.decode(Int.self, forKey: .dimension)
        let elements = try container.decode([Scalar].self, forKey: .elements)
        
        guard elements.count == dimension else {
            throw DecodingError.dataCorruptedError(
                forKey: .elements,
                in: container,
                debugDescription: "Element count \(elements.count) doesn't match dimension \(dimension)"
            )
        }
        
        self.init(elements)
    }
    
    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(dimension, forKey: .dimension)
        try container.encode(toArray(), forKey: .elements)
    }
}

// MARK: - Binary Serialization (see BinaryFormat + CRC32 in Utilities)

extension VectorProtocol where Scalar == Float {
    /// Encode vector to binary format using BinaryFormat utilities and CRC32 checksum
    public func encodeBinary() -> Data {
        var data = Data()
        data.reserveCapacity(BinaryFormat.expectedDataSize(for: scalarCount))
        // Header
        let header = BinaryHeader(dimension: scalarCount)
        BinaryFormat.writeUInt32(header.magic, to: &data)
        BinaryFormat.writeUInt16(header.version, to: &data)
        BinaryFormat.writeUInt32(header.dimension, to: &data)
        BinaryFormat.writeUInt16(header.flags, to: &data)
        // Payload
        BinaryFormat.writeFloatArray(self.toArray(), to: &data)
        // CRC32
        let checksum = data.crc32()
        BinaryFormat.writeUInt32(checksum, to: &data)
        return data
    }
}

extension Vector {
    /// Decode vector from binary format
    public static func decodeBinary(from data: Data) throws -> Vector {
        // Validate header and checksum using BinaryFormat
        let header = try BinaryFormat.validateHeader(in: data)
        try BinaryFormat.validateChecksum(in: data)
        // Ensure dimension matches D.value
        let dim = Int(header.dimension)
        guard dim == D.value else {
            throw VectorError.dimensionMismatch(expected: D.value, actual: dim)
        }
        // Read payload
        let values = try BinaryFormat.readFloatArray(from: data, at: BinaryHeader.headerSize, count: dim)
        return try Vector(values)
    }
}

extension DynamicVector {
    /// Decode dynamic vector from binary format
    public static func decodeBinary(from data: Data) throws -> DynamicVector {
        let header = try BinaryFormat.validateHeader(in: data)
        try BinaryFormat.validateChecksum(in: data)
        let dim = Int(header.dimension)
        let values = try BinaryFormat.readFloatArray(from: data, at: BinaryHeader.headerSize, count: dim)
        return DynamicVector(values)
    }
}

// MARK: - JSON Convenience

extension VectorProtocol where Self: Encodable {
    /// Encode to JSON data
    public func encodeJSON(encoder: JSONEncoder = JSONEncoder()) throws -> Data {
        try encoder.encode(self)
    }
    
    /// Encode to JSON string
    public func encodeJSONString(encoder: JSONEncoder = JSONEncoder()) throws -> String? {
        let data = try encodeJSON(encoder: encoder)
        return String(data: data, encoding: .utf8)
    }
}

extension Vector {
    /// Decode from JSON data
    public static func decodeJSON(from data: Data, decoder: JSONDecoder = JSONDecoder()) throws -> Vector {
        try decoder.decode(Vector.self, from: data)
    }
    
    /// Decode from JSON string
    public static func decodeJSON(from string: String, decoder: JSONDecoder = JSONDecoder()) throws -> Vector {
        guard let data = string.data(using: .utf8) else {
            throw VectorError.invalidDimension(0, reason: "Invalid UTF-8 string")
        }
        return try decodeJSON(from: data, decoder: decoder)
    }
}

extension DynamicVector {
    /// Decode from JSON data
    public static func decodeJSON(from data: Data, decoder: JSONDecoder = JSONDecoder()) throws -> DynamicVector {
        try decoder.decode(DynamicVector.self, from: data)
    }
    
    /// Decode from JSON string
    public static func decodeJSON(from string: String, decoder: JSONDecoder = JSONDecoder()) throws -> DynamicVector {
        guard let data = string.data(using: .utf8) else {
            throw VectorError.invalidDimension(0, reason: "Invalid UTF-8 string")
        }
        return try decodeJSON(from: data, decoder: decoder)
    }
}
