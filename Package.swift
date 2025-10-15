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
        // Benchmarking library (models and utilities for benchmark tooling)
        .library(
            name: "VectorCoreBenchmarking",
            targets: ["VectorCoreBenchmarking"]
        ),
        // Benchmark executable (lives under Benchmarks/VectorCoreBench)
        .executable(
            name: "vectorcore-bench",
            targets: ["VectorCoreBench"]
        )
    ],
    dependencies: [],
    targets: [
        // Main Library
        .target(
            name: "VectorCore",
            dependencies: ["VectorCoreC"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
                .enableUpcomingFeature("ExistentialAny")
            ]
        ),
        // C kernels target (scaffolded; usage gated in Swift)
        .target(
            name: "VectorCoreC",
            path: "Sources/VectorCoreC",
            publicHeadersPath: "include"
        ),

        // Benchmarking library - models and utilities for benchmark tools
        .target(
            name: "VectorCoreBenchmarking",
            dependencies: ["VectorCore"],
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
        
        // New comprehensive test suite (skeleton only)
        .testTarget(
            name: "ComprehensiveTests",
            dependencies: ["VectorCore"],
            swiftSettings: [ .enableExperimentalFeature("StrictConcurrency") ]
        ),
        // Benchmark executable target (Phase 1 scaffold)
        .executableTarget(
            name: "VectorCoreBench",
            dependencies: ["VectorCore", "VectorCoreBenchmarking"],
            path: "Benchmarks/VectorCoreBench"
        )
    ]
)
