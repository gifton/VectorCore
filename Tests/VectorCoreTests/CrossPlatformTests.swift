import XCTest
@testable import VectorCore

final class CrossPlatformTests: XCTestCase {
    
    // MARK: - Platform Detection
    
    func testPlatformDetection() {
        #if os(macOS)
        XCTAssertTrue(true, "Running on macOS")
        #elseif os(iOS)
        XCTAssertTrue(true, "Running on iOS")
        #elseif os(tvOS)
        XCTAssertTrue(true, "Running on tvOS")
        #elseif os(watchOS)
        XCTAssertTrue(true, "Running on watchOS")
        #elseif os(Linux)
        XCTAssertTrue(true, "Running on Linux")
        #else
        XCTFail("Unknown platform")
        #endif
    }
    
    // MARK: - Linux Compatibility
    
    func testLinuxCompatibility() {
        // Test operations that might differ on Linux
        
        // Random number generation
        let random = Float.random(in: 0...1)
        XCTAssertGreaterThanOrEqual(random, 0)
        XCTAssertLessThanOrEqual(random, 1)
        
        // File operations for telemetry
        #if os(Linux)
        let documentsPath = FileManager.default.currentDirectoryPath
        #else
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!.path
        #endif
        
        XCTAssertNotNil(documentsPath)
    }
    
    // MARK: - Accelerate Framework Availability
    
    func testAccelerateAvailability() {
        #if canImport(Accelerate)
        // Test Accelerate operations
        import Accelerate
        var result: Float = 0
        let a: [Float] = [1, 2, 3, 4]
        let b: [Float] = [5, 6, 7, 8]
        
        vDSP_dotpr(a, 1, b, 1, &result, 4)
        XCTAssertEqual(result, 70) // 1*5 + 2*6 + 3*7 + 4*8
        #else
        // Fallback implementations should work
        let v1 = Vector128([1, 2, 3, 4] + Array(repeating: 0, count: 124))
        let v2 = Vector128([5, 6, 7, 8] + Array(repeating: 0, count: 124))
        let dot = v1.dotProduct(v2)
        XCTAssertEqual(dot, 70)
        #endif
    }
    
    // MARK: - Endianness Tests
    
    func testEndianness() {
        // Ensure consistent behavior across platforms
        let value: Float = 3.14159
        let data = withUnsafeBytes(of: value) { Data($0) }
        
        XCTAssertEqual(data.count, 4)
        
        // Reconstruct value
        let reconstructed = data.withUnsafeBytes { bytes in
            bytes.load(as: Float.self)
        }
        
        XCTAssertEqual(value, reconstructed, accuracy: 1e-6)
    }
}