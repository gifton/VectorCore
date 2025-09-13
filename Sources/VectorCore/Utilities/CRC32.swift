// VectorCore: CRC32 Implementation
//
// CRC32 checksum calculation for data integrity
//

import Foundation

/// CRC32 checksum implementation
public struct CRC32 {
    private static let polynomial: UInt32 = 0xEDB88320
    private static let table: [UInt32] = {
        (0..<256).map { i in
            (0..<8).reduce(UInt32(i)) { crc, _ in
                (crc & 1) == 1 ? (crc >> 1) ^ polynomial : crc >> 1
            }
        }
    }()

    /// Calculate CRC32 checksum for data
    public static func checksum(_ data: Data) -> UInt32 {
        data.reduce(~UInt32(0)) { crc, byte in
            let index = Int((crc ^ UInt32(byte)) & 0xFF)
            return (crc >> 8) ^ table[index]
        } ^ ~UInt32(0)
    }

    /// Calculate CRC32 checksum for byte array
    public static func checksum(_ bytes: [UInt8]) -> UInt32 {
        checksum(Data(bytes))
    }
}

// MARK: - Data Extensions

public extension Data {
    /// Calculate CRC32 checksum
    func crc32() -> UInt32 {
        CRC32.checksum(self)
    }
}

// MARK: - Array Extensions

public extension Array where Element == UInt8 {
    /// Calculate CRC32 checksum
    func crc32() -> UInt32 {
        CRC32.checksum(self)
    }
}
