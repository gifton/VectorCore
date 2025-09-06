// swift-tools-version: 6.0
import PackageDescription

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
            targets: ["VectorCore"]
        ),
    ],
    dependencies: [],
    targets: [
        // Main Library
        .target(
            name: "VectorCore",
            dependencies: [],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
                .enableUpcomingFeature("ExistentialAny")
            ]
        ),
        
        // Test Target - Minimal working tests only
        .testTarget(
            name: "MinimalTests",
            dependencies: ["VectorCore"],
            swiftSettings: [ .enableExperimentalFeature("StrictConcurrency") ]
        ),
        
    ]
)
