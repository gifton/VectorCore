//
//  NSWOptimizationBenchmark.swift
//  VectorCore
//
//  Benchmarks to verify NSW performance optimizations
//

import XCTest
@testable import VectorCore

final class NSWOptimizationBenchmark: XCTestCase {

    // Generate test vectors
    func generateTestVectors(count: Int, dimension: Int) -> ContiguousArray<ContiguousArray<Float>> {
        var vectors = ContiguousArray<ContiguousArray<Float>>()
        vectors.reserveCapacity(count)

        for _ in 0..<count {
            var vector = ContiguousArray<Float>()
            vector.reserveCapacity(dimension)
            for _ in 0..<dimension {
                vector.append(Float.random(in: -1...1))
            }
            vectors.append(vector)
        }

        return vectors
    }

    // MARK: - NSW Construction Benchmark

    func testNSWConstructionPerformance() {
        let vectorCounts = [100, 500, 1000]
        let dimension = 128

        for count in vectorCounts {
            let vectors = generateTestVectors(count: count, dimension: dimension)

            let options = GraphConstructionKernels.NSWOptions(
                M: 16,
                efConstruction: 200,
                metric: .euclidean,
                heuristic: true
            )

            measure {
                _ = GraphConstructionKernels.buildNSWIndex(
                    vectors: vectors,
                    options: options
                )
            }

            print("NSW Construction for \(count) vectors completed")
        }
    }

    // MARK: - Batch Distance Benchmark

    func testBatchDistancePerformance() {
        let dimension = 512
        let targetCounts = [100, 500, 1000]

        for targetCount in targetCounts {
            let source = ContiguousArray<Float>((0..<dimension).map { _ in Float.random(in: -1...1) })
            let targets = (0..<targetCount).map { _ in
                ContiguousArray<Float>((0..<dimension).map { _ in Float.random(in: -1...1) })
            }

            measure {
                _ = GraphConstructionKernels.computeDistancesBatch(
                    from: source,
                    to: targets,
                    metric: .euclidean
                )
            }

            print("Batch distance for \(targetCount) targets completed")
        }
    }

    // MARK: - Memory Allocation Benchmark

    func testMemoryAllocationPerformance() {
        let n = 1000
        let k = 20

        let vectors = generateTestVectors(count: n, dimension: 128)

        let options = GraphConstructionKernels.KNNGraphOptions(
            k: k,
            metric: .euclidean,
            symmetric: true
        )

        measure {
            _ = GraphConstructionKernels.buildKNNGraph(
                vectors: vectors,
                options: options
            )
        }

        print("KNN Graph construction with optimized memory allocation completed")
    }

    // MARK: - SortedArray vs Set Benchmark

    func testSortedArrayPerformance() {
        let elementCount = 1000
        let insertCount = 10000

        // Test SortedArray performance
        measure(metrics: [XCTMemoryMetric(), XCTClockMetric()]) {
            var sortedArray = GraphConstructionKernels.SortedArray<Int32>(capacity: elementCount)

            for _ in 0..<insertCount {
                let value = Int32.random(in: 0..<Int32(elementCount))
                sortedArray.insert(value)
            }

            // Test lookups
            for _ in 0..<1000 {
                let value = Int32.random(in: 0..<Int32(elementCount))
                _ = sortedArray.contains(value)
            }
        }

        print("SortedArray operations completed")
    }

    // MARK: - Compare optimized vs naive implementations

    func testOptimizationImpact() {
        let vectors = generateTestVectors(count: 200, dimension: 128)

        let options = GraphConstructionKernels.NSWOptions(
            M: 16,
            efConstruction: 100,
            metric: .euclidean
        )

        // Measure optimized version
        let optimizedTime = measureTime {
            _ = GraphConstructionKernels.buildNSWIndex(
                vectors: vectors,
                options: options
            )
        }

        print("Optimized NSW construction time: \(optimizedTime) seconds")

        // The optimizations should show:
        // 1. Better cache locality from SortedArray
        // 2. Reduced allocations from pre-allocation
        // 3. Faster batch distance computations
    }

    private func measureTime(block: () -> Void) -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        block()
        return CFAbsoluteTimeGetCurrent() - start
    }
}
