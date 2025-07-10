// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

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
            targets: ["VectorCore"]),
    ],
    dependencies: [
        // No dependencies - VectorCore is the zero-dependency foundation
    ],
    targets: [
        .target(
            name: "VectorCore",
            dependencies: [],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
                .enableUpcomingFeature("ExistentialAny")
            ]
        ),
        .testTarget(
            name: "VectorCoreTests",
            dependencies: ["VectorCore"]
        ),
        .executableTarget(
            name: "VectorCoreVerify",
            dependencies: ["VectorCore"]
        ),
        .executableTarget(
            name: "VectorCoreBenchmarks",
            dependencies: ["VectorCore"],
            path: "Benchmarks/Sources/VectorCoreBenchmarks"
        ),
        .executableTarget(
            name: "ModernAPIExample",
            dependencies: ["VectorCore"],
            path: "Examples",
            sources: ["ModernAPIExample.swift"]
        ),
        .executableTarget(
            name: "ErrorHandlingExample",
            dependencies: ["VectorCore"],
            path: "Examples",
            sources: ["ErrorHandlingExample.swift"]
        ),
    ]
)