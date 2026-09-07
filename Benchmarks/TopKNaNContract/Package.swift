// swift-tools-version: 6.0
import PackageDescription
import Foundation

let source = ProcessInfo.processInfo.environment["VECTORCORE_BENCH_SOURCE"] ?? "../.."
let package = Package(
    name: "TopKNaNContractBench",
    platforms: [.macOS(.v14)],
    dependencies: [.package(name: "VectorCore", path: source)],
    targets: [.executableTarget(
        name: "TopKNaNContractBench",
        dependencies: [.product(name: "VectorCore", package: "VectorCore")]
    )]
)
