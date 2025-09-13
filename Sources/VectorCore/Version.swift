// VectorCore: Version Information
//
// Version tracking for the VectorCore package
//

/// Version information for VectorCore.
///
/// `VectorCoreVersion` provides semantic versioning information for the
/// VectorCore package, following the Semantic Versioning 2.0.0 specification.
/// Version numbers convey meaning about the underlying changes and compatibility.
///
/// ## Version Format
/// - **Major**: Incremented for incompatible API changes
/// - **Minor**: Incremented for backwards-compatible functionality additions
/// - **Patch**: Incremented for backwards-compatible bug fixes
/// - **Pre-release**: Optional identifier for pre-release versions (alpha, beta, rc)
/// - **Build**: Optional build metadata (commit hash, build number)
///
/// ## Example Usage
/// ```swift
/// print("VectorCore v\(VectorCoreVersion.versionString)")
/// 
/// if VectorCoreVersion.major >= 1 {
///     // Use stable API features
/// }
/// ```
public struct VectorCoreVersion {
    /// Current version string of VectorCore.
    ///
    /// This matches the version in Package.swift.
    public static let version = "0.1.0"
    
    /// Major version number.
    ///
    /// Incremented when making incompatible API changes.
    public static let major = 0
    
    /// Minor version number.
    ///
    /// Incremented when adding functionality in a backwards-compatible manner.
    public static let minor = 1
    
    /// Patch version number.
    ///
    /// Incremented when making backwards-compatible bug fixes.
    public static let patch = 0
    
    /// Pre-release version identifier.
    ///
    /// Examples: "alpha", "beta.1", "rc.2"
    /// nil for stable releases.
    public static let prerelease: String? = nil
    
    /// Build metadata.
    ///
    /// Examples: "exp.sha.5114f85", "2024.01.15"
    /// Does not affect version precedence.
    public static let buildMetadata: String? = nil
    
    /// Full semantic version string.
    ///
    /// Combines all version components according to semver spec:
    /// `major.minor.patch[-prerelease][+build]`
    ///
    /// ## Examples
    /// - "1.0.0" - Stable release
    /// - "2.1.0-beta.1" - Beta pre-release
    /// - "1.2.3+20240115" - With build metadata
    public static var versionString: String {
        var version = "\(major).\(minor).\(patch)"
        if let pre = prerelease {
            version += "-\(pre)"
        }
        if let build = buildMetadata {
            version += "+\(build)"
        }
        return version
    }
}
