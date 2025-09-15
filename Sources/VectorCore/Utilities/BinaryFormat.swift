// VectorCore: Binary Format Utilities
//
// Platform-independent binary serialization support
//

import Foundation

/// Binary format utilities for platform-independent serialization
public enum BinaryFormat {

    // MARK: - Constants

    /// Maximum allowed vector dimension for security
    public static let maxDimension: UInt32 = 10_000

    /// Minimum valid data size (header + CRC32)
    public static let minDataSize = 16  // 4 + 2 + 4 + 2 + 4 bytes

    // MARK: - Endianness Conversion

    /// Write UInt32 in little-endian format
    @inlinable
    public static func writeUInt32(_ value: UInt32, to data: inout Data) {
        let littleEndian = value.littleEndian
        withUnsafeBytes(of: littleEndian) { bytes in
            data.append(contentsOf: bytes)
        }
    }

    /// Write UInt16 in little-endian format
    @inlinable
    public static func writeUInt16(_ value: UInt16, to data: inout Data) {
        let littleEndian = value.littleEndian
        withUnsafeBytes(of: littleEndian) { bytes in
            data.append(contentsOf: bytes)
        }
    }

    /// Write Float in little-endian format
    @inlinable
    public static func writeFloat(_ value: Float, to data: inout Data) {
        let bits = value.bitPattern.littleEndian
        withUnsafeBytes(of: bits) { bytes in
            data.append(contentsOf: bytes)
        }
    }

    /// Write Float array in little-endian format
    @inlinable
    public static func writeFloatArray(_ values: [Float], to data: inout Data) {
        // Pre-allocate space for efficiency
        data.reserveCapacity(data.count + values.count * MemoryLayout<Float>.size)

        for value in values {
            writeFloat(value, to: &data)
        }
    }

    /// Read UInt32 from little-endian format
    @inlinable
    public static func readUInt32(from data: Data, at offset: Int) throws -> UInt32 {
        guard offset + MemoryLayout<UInt32>.size <= data.count else {
            throw VectorError.insufficientData(
                expected: offset + MemoryLayout<UInt32>.size,
                actual: data.count
            )
        }

        let value = data.withUnsafeBytes { bytes in
            bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
        }
        return UInt32(littleEndian: value)
    }

    /// Read UInt16 from little-endian format
    @inlinable
    public static func readUInt16(from data: Data, at offset: Int) throws -> UInt16 {
        guard offset + MemoryLayout<UInt16>.size <= data.count else {
            throw VectorError.insufficientData(
                expected: offset + MemoryLayout<UInt16>.size,
                actual: data.count
            )
        }

        let value = data.withUnsafeBytes { bytes in
            bytes.loadUnaligned(fromByteOffset: offset, as: UInt16.self)
        }
        return UInt16(littleEndian: value)
    }

    /// Read Float from little-endian format
    @inlinable
    public static func readFloat(from data: Data, at offset: Int) throws -> Float {
        guard offset + MemoryLayout<Float>.size <= data.count else {
            throw VectorError.insufficientData(
                expected: offset + MemoryLayout<Float>.size,
                actual: data.count
            )
        }

        let bits = data.withUnsafeBytes { bytes in
            bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
        }
        return Float(bitPattern: UInt32(littleEndian: bits))
    }

    /// Read Float array from little-endian format
    @inlinable
    public static func readFloatArray(from data: Data, at offset: Int, count: Int) throws -> [Float] {
        let bytesNeeded = count * MemoryLayout<Float>.size
        guard offset + bytesNeeded <= data.count else {
            throw VectorError.insufficientData(
                expected: offset + bytesNeeded,
                actual: data.count
            )
        }

        var result = [Float]()
        result.reserveCapacity(count)

        for i in 0..<count {
            let value = try readFloat(from: data, at: offset + i * MemoryLayout<Float>.size)
            result.append(value)
        }

        return result
    }

    // MARK: - Header Operations

    /// Calculate expected data size for a vector dimension
    @inlinable
    public static func expectedDataSize(for dimension: Int) -> Int {
        return minDataSize + dimension * MemoryLayout<Float>.size
    }

    /// Validate binary data header
    public static func validateHeader(in data: Data) throws -> (version: UInt16, dimension: UInt32, flags: UInt16) {
        // Check minimum size
        guard data.count >= minDataSize else {
            throw VectorError.insufficientData(expected: minDataSize, actual: data.count)
        }

        // Validate magic number
        let magic = try readUInt32(from: data, at: 0)
        guard magic == BinaryHeader.magic else {
            throw VectorError.invalidDataFormat(
                expected: "VECT magic number",
                actual: "0x\(String(magic, radix: 16))"
            )
        }

        // Read and validate version
        let version = try readUInt16(from: data, at: 4)
        guard version == BinaryHeader.version else {
            throw VectorError.invalidDataFormat(
                expected: "version \(BinaryHeader.version)",
                actual: "version \(version)"
            )
        }

        // Read and validate dimension
        let dimension = try readUInt32(from: data, at: 6)
        guard dimension > 0 && dimension <= maxDimension else {
            throw VectorError.invalidDimension(
                Int(dimension),
                reason: "Must be between 1 and \(maxDimension)"
            )
        }

        // Read flags
        let flags = try readUInt16(from: data, at: 10)

        // Validate total data size
        let expectedSize = expectedDataSize(for: Int(dimension))
        guard data.count == expectedSize else {
            throw VectorError.insufficientData(expected: expectedSize, actual: data.count)
        }

        return (version: version, dimension: dimension, flags: flags)
    }

    /// Calculate and validate CRC32 checksum
    public static func validateChecksum(in data: Data) throws {
        // Read the stored checksum (last 4 bytes)
        let checksumOffset = data.count - MemoryLayout<UInt32>.size
        let storedChecksum = try readUInt32(from: data, at: checksumOffset)

        // Calculate checksum of data (excluding the checksum itself)
        let dataToCheck = data.prefix(checksumOffset)
        let calculatedChecksum = dataToCheck.crc32()

        // Validate
        guard storedChecksum == calculatedChecksum else {
            throw VectorError.dataCorruption(
                reason: "CRC32 mismatch: expected \(storedChecksum), calculated \(calculatedChecksum)"
            )
        }
    }
}

// MARK: - UnsafeRawBufferPointer Extensions

extension UnsafeRawBufferPointer {
    /// Load unaligned value from buffer
    @inlinable
    func loadUnaligned<T>(fromByteOffset offset: Int, as type: T.Type) -> T {
        assert(offset >= 0 && offset + MemoryLayout<T>.size <= count)
        return self.baseAddress!.advanced(by: offset).loadUnaligned(as: type)
    }
}

extension UnsafeRawPointer {
    /// Load unaligned value from pointer
    @inlinable
    func loadUnaligned<T>(as type: T.Type) -> T {
        let buffer = UnsafeMutableRawPointer.allocate(
            byteCount: MemoryLayout<T>.size,
            alignment: MemoryLayout<T>.alignment
        )
        defer { buffer.deallocate() }

        buffer.copyMemory(from: self, byteCount: MemoryLayout<T>.size)
        return buffer.load(as: type)
    }
}
