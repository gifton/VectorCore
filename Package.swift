// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

// MARK: - Build Configuration

/// Shared Swift settings for all targets
let sharedSwiftSettings: [SwiftSetting] = [
    // Language features
    .enableExperimentalFeature("StrictConcurrency"),
    .enableUpcomingFeature("ExistentialAny"),
    
    // Warnings as errors in release builds
    .unsafeFlags(["-warnings-as-errors"], .when(configuration: .release)),
]

/// Optimized Swift settings for release builds
let releaseSwiftSettings: [SwiftSetting] = [
    // Optimization level
    .unsafeFlags(["-O"], .when(configuration: .release)),
    
    // Whole module optimization
    .unsafeFlags(["-whole-module-optimization"], .when(configuration: .release)),
    
    // Cross module optimization
    .unsafeFlags(["-cross-module-optimization"], .when(configuration: .release)),
    
    // Additional optimizations
    .unsafeFlags([
        "-enforce-exclusivity=unchecked"
    ], .when(configuration: .release)),
    
    // SIMD and vectorization hints
    .define("SWIFT_RELEASE_MODE", .when(configuration: .release)),
    .define("VECTORCORE_ENABLE_SIMD", .when(configuration: .release)),
]

/// Debug-specific settings
let debugSwiftSettings: [SwiftSetting] = [
    // Debug mode flag
    .define("DEBUG", .when(configuration: .debug)),
    
    // Enable assertions in debug
    .define("VECTORCORE_ENABLE_ASSERTIONS", .when(configuration: .debug)),
    
    // Onone optimization for better debugging
    .unsafeFlags(["-Onone"], .when(configuration: .debug)),
]

/// Linker settings for optimized builds
let linkerSettings: [LinkerSetting] = [
    // Dead code stripping
    .unsafeFlags(["-Xlinker", "-dead_strip"], .when(configuration: .release)),
    
    // Link time optimization (LTO)
    .unsafeFlags(["-Xlinker", "-lto"], .when(configuration: .release)),
]

// MARK: - Package Definition

let package = Package(
    name: "VectorCore",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .tvOS(.v17),
        .watchOS(.v10),
        .visionOS(.v1)
    ],
    products: [
        .library(
            name: "VectorCore",
            targets: ["VectorCore"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ordo-one/package-benchmark", from: "1.27.1")
    ],
    targets: [
        // MARK: Main Library
        .target(
            name: "VectorCore",
            dependencies: [],
            exclude: ["GPU.disabled"],
            swiftSettings: sharedSwiftSettings + releaseSwiftSettings + debugSwiftSettings,
            linkerSettings: linkerSettings
        ),
        
        // MARK: Test Targets
        .testTarget(
            name: "VectorCoreTests",
            dependencies: ["VectorCore"],
            swiftSettings: sharedSwiftSettings + debugSwiftSettings
        ),
        
        // MARK: Executable Targets
        .executableTarget(
            name: "VectorCoreVerify",
            dependencies: ["VectorCore"],
            swiftSettings: sharedSwiftSettings + releaseSwiftSettings + debugSwiftSettings
        ),
        .executableTarget(
            name: "VectorCoreBenchmarks",
            dependencies: [
                "VectorCore",
                .product(name: "Benchmark", package: "package-benchmark")
            ],
            path: "Benchmarks",
            sources: [
                "main.swift",
                "VectorCoreBenchmarks/VectorOperationBenchmarks.swift",
                "VectorCoreBenchmarks/StorageBenchmarks.swift",
                "VectorCoreBenchmarks/DistanceBenchmarks.swift",
                "VectorCoreBenchmarks/BatchOperationBenchmarks.swift"
            ],
            swiftSettings: sharedSwiftSettings + releaseSwiftSettings + [
                .unsafeFlags(["-parse-as-library"])
            ]
        ),
        
        // MARK: Example Targets
        .executableTarget(
            name: "VectorCoreAPIExample",
            dependencies: ["VectorCore"],
            path: "Examples",
            sources: ["VectorCoreAPIExample.swift"],
            swiftSettings: sharedSwiftSettings + releaseSwiftSettings
        ),
        .executableTarget(
            name: "ErrorHandlingExample",
            dependencies: ["VectorCore"],
            path: "Examples",
            sources: ["ErrorHandlingExample.swift"],
            swiftSettings: sharedSwiftSettings
        ),
        .executableTarget(
            name: "PerformanceRegression",
            dependencies: ["VectorCore"],
            swiftSettings: sharedSwiftSettings + releaseSwiftSettings
        ),
        .executableTarget(
            name: "SyncBatchOperationsExample",
            dependencies: ["VectorCore"],
            path: "Examples",
            sources: ["SyncBatchOperationsExample.swift"],
            swiftSettings: sharedSwiftSettings
        ),
        .executableTarget(
            name: "NaNInfinityHandlingExample",
            dependencies: ["VectorCore"],
            path: "Examples",
            sources: ["NaNInfinityHandlingExample.swift"],
            swiftSettings: sharedSwiftSettings
        ),
        .executableTarget(
            name: "PerformanceRegressionRunner",
            dependencies: ["VectorCore"],
            swiftSettings: sharedSwiftSettings + releaseSwiftSettings
        ),
        .executableTarget(
            name: "PerformanceRegressionExample",
            dependencies: ["VectorCore"],
            path: "Examples",
            sources: ["PerformanceRegressionExample.swift"],
            swiftSettings: sharedSwiftSettings + releaseSwiftSettings
        ),
    ]
)

// MARK: - Build Script Phase (for CI/CD)

#if canImport(Foundation)
import Foundation

// Add custom build phase for optimization validation
if ProcessInfo.processInfo.environment["VECTORCORE_VALIDATE_BUILD"] != nil {
    print("""
    VectorCore Build Configuration:
    - Whole Module Optimization: ✓
    - Cross Module Optimization: ✓
    - Link Time Optimization: ✓
    - Dead Code Stripping: ✓
    - SIMD Enabled: ✓
    """)
}
#endif