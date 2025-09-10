// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "VectorCorePerformance",
    platforms: [
        .macOS(.v14), .iOS(.v17), .tvOS(.v17), .watchOS(.v10), .visionOS(.v1)
    ],
    products: [
        .library(name: "VectorCorePerformance", targets: ["VectorCorePerformance"]) 
    ],
    dependencies: [
        // Path dependency to the parent VectorCore package
        .package(path: "..")
    ],
    targets: [
        .target(
            name: "VectorCorePerformance",
            dependencies: [
                .product(name: "VectorCore", package: "VectorCore")
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
                .enableUpcomingFeature("ExistentialAny")
            ]
        )
    ]
)

