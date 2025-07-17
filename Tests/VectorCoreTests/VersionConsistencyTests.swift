// VectorCore: Version Consistency Tests
//
// Ensures version reporting is consistent across the package
//

import XCTest
@testable import VectorCore

final class VersionConsistencyTests: XCTestCase {
    
    func testVersionConsistency() {
        // Test that VectorCore.version matches VectorCoreVersion
        XCTAssertEqual(VectorCore.version, VectorCoreVersion.versionString)
        
        // Test that versionString is properly constructed
        let expectedVersion = "\(VectorCoreVersion.major).\(VectorCoreVersion.minor).\(VectorCoreVersion.patch)"
        XCTAssertEqual(VectorCoreVersion.versionString, expectedVersion)
        
        // Test that version components match the string
        XCTAssertEqual(VectorCoreVersion.version, "0.1.0")
        XCTAssertEqual(VectorCoreVersion.major, 0)
        XCTAssertEqual(VectorCoreVersion.minor, 1)
        XCTAssertEqual(VectorCoreVersion.patch, 0)
    }
    
    func testVersionFormat() {
        // Ensure version follows semantic versioning format
        let versionRegex = #"^\d+\.\d+\.\d+(?:-[\w\.]+)?(?:\+[\w\.]+)?$"#
        let predicate = NSPredicate(format: "SELF MATCHES %@", versionRegex)
        
        XCTAssertTrue(predicate.evaluate(with: VectorCore.version),
                      "Version '\(VectorCore.version)' does not follow semantic versioning format")
    }
    
    func testPrereleaseAndBuildMetadata() {
        // Test that prerelease and build metadata are handled correctly
        XCTAssertNil(VectorCoreVersion.prerelease)
        XCTAssertNil(VectorCoreVersion.buildMetadata)
        
        // Version string should not contain prerelease or build metadata when nil
        XCTAssertFalse(VectorCoreVersion.versionString.contains("-"))
        XCTAssertFalse(VectorCoreVersion.versionString.contains("+"))
    }
}