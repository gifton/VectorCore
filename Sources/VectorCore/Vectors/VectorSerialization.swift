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

// MARK: - Binary Serialization

/// Binary format header for vectors
struct VectorBinaryHeader {
    let magic: UInt32 = 0x56454354  // "VECT" in hex
    let version: UInt16 = 1
    let dimension: UInt32
    let flags: UInt16 = 0
    let checksum: UInt32
}

extension VectorProtocol {
    /// Encode vector to binary format
    public func encodeBinary() -> Data {
        var data = Data()
        data.reserveCapacity(12 + scalarCount * MemoryLayout<Scalar>.size + 4)
        
        // Write header (without checksum)
        withUnsafeBytes(of: UInt32(0x56454354).littleEndian) { data.append(contentsOf: $0) }  // Magic
        withUnsafeBytes(of: UInt16(1).littleEndian) { data.append(contentsOf: $0) }           // Version
        withUnsafeBytes(of: UInt32(scalarCount).littleEndian) { data.append(contentsOf: $0) } // Dimension
        withUnsafeBytes(of: UInt16(0).littleEndian) { data.append(contentsOf: $0) }           // Flags
        
        // Write vector data
        withUnsafeBufferPointer { buffer in
            for element in buffer {
                withUnsafeBytes(of: element) { data.append(contentsOf: $0) }
            }
        }
        
        // Calculate and append checksum
        let checksum = data.reduce(UInt32(0)) { $0 &+ UInt32($1) }
        withUnsafeBytes(of: checksum.littleEndian) { data.append(contentsOf: $0) }
        
        return data
    }
}

extension Vector {
    /// Decode vector from binary format
    public static func decodeBinary(from data: Data) throws -> Vector {
        guard data.count >= 16 else {
            throw VectorError.invalidDimension(0, reason: "Data too small for header")
        }
        
        // Read header
        let magic = data.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt32.self).littleEndian }
        let version = data.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt16.self).littleEndian }
        let dimension = data.withUnsafeBytes { $0.load(fromByteOffset: 6, as: UInt32.self).littleEndian }
        
        // Validate header
        guard magic == 0x56454354 else {
            throw VectorError.invalidDimension(0, reason: "Invalid magic number")
        }
        guard version == 1 else {
            throw VectorError.invalidDimension(0, reason: "Unsupported version")
        }
        guard dimension == D.value else {
            throw VectorError.dimensionMismatch(expected: D.value, actual: Int(dimension))
        }
        
        // Validate data size
        let expectedSize = 16 + Int(dimension) * MemoryLayout<Scalar>.size
        guard data.count == expectedSize else {
            throw VectorError.invalidDimension(data.count, reason: "Invalid data size")
        }
        
        // Read vector elements
        var elements = [Scalar]()
        elements.reserveCapacity(Int(dimension))
        
        let elementData = data.subdata(in: 12..<(data.count - 4))
        elementData.withUnsafeBytes { bytes in
            let buffer = bytes.bindMemory(to: Scalar.self)
            elements.append(contentsOf: buffer)
        }
        
        // Verify checksum
        let checksumData = data.subdata(in: 0..<(data.count - 4))
        let calculatedChecksum = checksumData.reduce(UInt32(0)) { $0 &+ UInt32($1) }
        let storedChecksum = data.withUnsafeBytes { 
            $0.load(fromByteOffset: data.count - 4, as: UInt32.self).littleEndian 
        }
        
        guard calculatedChecksum == storedChecksum else {
            throw VectorError.invalidDimension(0, reason: "Checksum mismatch")
        }
        
        return try Vector(elements)
    }
}

extension DynamicVector {
    /// Decode dynamic vector from binary format
    public static func decodeBinary(from data: Data) throws -> DynamicVector {
        guard data.count >= 16 else {
            throw VectorError.invalidDimension(0, reason: "Data too small for header")
        }
        
        // Read header
        let magic = data.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt32.self).littleEndian }
        let version = data.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt16.self).littleEndian }
        let dimension = data.withUnsafeBytes { $0.load(fromByteOffset: 6, as: UInt32.self).littleEndian }
        
        // Validate header
        guard magic == 0x56454354 else {
            throw VectorError.invalidDimension(0, reason: "Invalid magic number")
        }
        guard version == 1 else {
            throw VectorError.invalidDimension(0, reason: "Unsupported version")
        }
        
        // Validate data size
        let expectedSize = 16 + Int(dimension) * MemoryLayout<Scalar>.size
        guard data.count == expectedSize else {
            throw VectorError.invalidDimension(data.count, reason: "Invalid data size")
        }
        
        // Read vector elements
        var elements = [Scalar]()
        elements.reserveCapacity(Int(dimension))
        
        let elementData = data.subdata(in: 12..<(data.count - 4))
        elementData.withUnsafeBytes { bytes in
            let buffer = bytes.bindMemory(to: Scalar.self)
            elements.append(contentsOf: buffer)
        }
        
        // Verify checksum
        let checksumData = data.subdata(in: 0..<(data.count - 4))
        let calculatedChecksum = checksumData.reduce(UInt32(0)) { $0 &+ UInt32($1) }
        let storedChecksum = data.withUnsafeBytes { 
            $0.load(fromByteOffset: data.count - 4, as: UInt32.self).littleEndian 
        }
        
        guard calculatedChecksum == storedChecksum else {
            throw VectorError.invalidDimension(0, reason: "Checksum mismatch")
        }
        
        return DynamicVector(elements)
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