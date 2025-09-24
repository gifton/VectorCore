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
        // Benchmark executable (lives under Benchmarks/VectorCoreBench)
        .executable(
            name: "vectorcore-bench",
            targets: ["VectorCoreBench"]
        ),
        // Graph traversal demo executable
        .executable(
            name: "graph-demo",
            targets: ["GraphTraversalDemo"]
        ),
        // Batch distance demo executable
        .executable(
            name: "batch-distance-demo",
            targets: ["BatchDistanceDemo"]
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
        
        // New comprehensive test suite (skeleton only)
        .testTarget(
            name: "ComprehensiveTests",
            dependencies: ["VectorCore"],
            swiftSettings: [ .enableExperimentalFeature("StrictConcurrency") ]
        ),
        // Benchmark executable target (Phase 1 scaffold)
        .executableTarget(
            name: "VectorCoreBench",
            dependencies: ["VectorCore"],
            path: "Benchmarks/VectorCoreBench"
        ),
        // Graph traversal demo executable
        .executableTarget(
            name: "GraphTraversalDemo",
            dependencies: ["VectorCore"]
        ),
        // Batch distance demo executable
        .executableTarget(
            name: "BatchDistanceDemo",
            dependencies: ["VectorCore"]
        )
    ]
)
