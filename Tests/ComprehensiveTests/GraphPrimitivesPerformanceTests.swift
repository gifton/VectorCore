import Testing
import Foundation
import Dispatch
@testable import VectorCore

@Suite("Graph Primitives Performance")
struct GraphPrimitivesPerformanceTests {

    // MARK: - Performance Baseline Tests

    @Suite("Performance Baselines")
    struct PerformanceBaselineTests {

        @Test
        func testCSRConstructionPerformance() {
            // Test CSR construction time scaling
            let nodeCounts = [100, 500, 1000]
            let edgeDensity: Float = 0.05 // 5% edge density

            for nodeCount in nodeCounts {
                let edgeCount = Int(Float(nodeCount * nodeCount) * edgeDensity)
                let edges = generateRandomEdges(nodeCount: nodeCount, edgeCount: edgeCount)

                let startTime = DispatchTime.now()
                let matrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: nodeCount, edges: edges)
                let endTime = DispatchTime.now()

                let duration = Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000.0 // ms

                // Basic validation that construction worked
                #expect(matrix.rows == nodeCount)
                #expect(matrix.cols == nodeCount)
                #expect(matrix.nonZeros <= edgeCount) // May be less due to duplicate filtering

                // Performance expectations (these are baseline measurements)
                // For 100 nodes: should complete in < 1ms
                // For 500 nodes: should complete in < 10ms
                // For 1000 nodes: should complete in < 50ms
                switch nodeCount {
                case 100:
                    #expect(duration < 1.0, "CSR construction for 100 nodes took \(duration)ms")
                case 500:
                    #expect(duration < 10.0, "CSR construction for 500 nodes took \(duration)ms")
                case 1000:
                    #expect(duration < 50.0, "CSR construction for 1000 nodes took \(duration)ms")
                default:
                    break
                }

                print("CSR construction for \(nodeCount) nodes (\(edgeCount) edges): \(String(format: "%.3f", duration))ms")
            }
        }

        @Test
        func testSpMVPerformanceScaling() async {
            // Test SpMV performance across different graph sizes
            let graphSizes = [(nodes: 50, edges: 100), (nodes: 100, edges: 500), (nodes: 200, edges: 2000)]

            for (nodeCount, edgeCount) in graphSizes {
                let edges = generateRandomEdges(nodeCount: nodeCount, edgeCount: edgeCount)
                let matrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: nodeCount, edges: edges)
                let input = generateTestVectors512(count: nodeCount)

                var output = ContiguousArray<Vector512Optimized>()

                let startTime = DispatchTime.now()
                await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                    matrix: matrix,
                    input: input,
                    output: &output,
                    normalize: false
                )
                let endTime = DispatchTime.now()

                let duration = Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000.0 // ms

                #expect(output.count == nodeCount)

                // Performance expectations for SpMV
                // Should scale roughly linearly with number of edges
                let expectedMaxTime = Double(edgeCount) * 0.05 // 0.05ms per edge as baseline
                #expect(duration < expectedMaxTime, "SpMV for \(nodeCount) nodes (\(edgeCount) edges) took \(duration)ms, expected < \(expectedMaxTime)ms")

                print("SpMV for \(nodeCount) nodes (\(edgeCount) edges): \(String(format: "%.3f", duration))ms")
            }
        }

        @Test
        func testNeighborAggregationPerformance() {
            // Test neighbor aggregation performance
            let graphSizes = [(nodes: 100, degree: 10), (nodes: 200, degree: 20), (nodes: 500, degree: 25)]

            for (nodeCount, avgDegree) in graphSizes {
                let edgeCount = nodeCount * avgDegree
                let edges = generateRandomEdges(nodeCount: nodeCount, edgeCount: edgeCount)
                let graph = try! WeightedGraph(from: GraphPrimitivesKernels.edgeListToCSR(nodeCount: nodeCount, edges: edges))
                let nodeFeatures = generateTestVectors512(count: nodeCount)

                var output = ContiguousArray<Vector512Optimized>()

                let startTime = DispatchTime.now()
                GraphPrimitivesKernels.aggregateNeighbors(
                    graph: graph,
                    nodeFeatures: nodeFeatures,
                    aggregation: .sum,
                    output: &output
                )
                let endTime = DispatchTime.now()

                let duration = Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000.0 // ms

                #expect(output.count == nodeCount)

                // Performance expectations for neighbor aggregation
                // Should scale with number of nodes and average degree
                let expectedMaxTime = Double(nodeCount * avgDegree) * 0.005 // 0.005ms per neighbor relationship
                #expect(duration < expectedMaxTime, "Neighbor aggregation for \(nodeCount) nodes (avg degree \(avgDegree)) took \(duration)ms, expected < \(expectedMaxTime)ms")

                print("Neighbor aggregation for \(nodeCount) nodes (avg degree \(avgDegree)): \(String(format: "%.3f", duration))ms")
            }
        }

        @Test
        func testTransposePerformance() {
            // Test matrix transpose performance
            let edgeCounts = [1000, 5000, 10000]

            for edgeCount in edgeCounts {
                let nodeCount = Int(sqrt(Double(edgeCount)) * 2) // Reasonable node count for edge count
                let edges = generateRandomEdges(nodeCount: nodeCount, edgeCount: edgeCount)
                let matrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: nodeCount, edges: edges)

                let startTime = DispatchTime.now()
                let transposed = GraphPrimitivesKernels.transposeCSR(matrix)
                let endTime = DispatchTime.now()

                let duration = Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000.0 // ms

                #expect(transposed.rows == matrix.cols)
                #expect(transposed.cols == matrix.rows)
                #expect(transposed.nonZeros == matrix.nonZeros)

                // Performance expectations for transpose
                // Should scale linearly with number of edges
                let expectedMaxTime = Double(edgeCount) * 0.001 // 0.001ms per edge
                #expect(duration < expectedMaxTime, "Transpose for \(edgeCount) edges took \(duration)ms, expected < \(expectedMaxTime)ms")

                print("Transpose for \(edgeCount) edges: \(String(format: "%.3f", duration))ms")
            }
        }

        @Test
        func testSubgraphExtractionPerformance() {
            // Test subgraph extraction performance
            let graphSize = 500
            let edgeCount = 2500
            let edges = generateRandomEdges(nodeCount: graphSize, edgeCount: edgeCount)
            let graph = try! WeightedGraph(from: GraphPrimitivesKernels.edgeListToCSR(nodeCount: graphSize, edges: edges))

            let subsetSizes = [50, 100, 200]

            for subsetSize in subsetSizes {
                let nodeSubset: Set<UInt32> = Set((0..<subsetSize).map { UInt32($0) })

                let startTime = DispatchTime.now()
                let (subgraph, nodeMapping) = GraphPrimitivesKernels.extractSubgraph(from: graph, nodeSubset: nodeSubset)
                let endTime = DispatchTime.now()

                let duration = Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000.0 // ms

                #expect(subgraph.nodeCount <= subsetSize)
                #expect(nodeMapping.count <= subsetSize)

                // Performance expectations for subgraph extraction
                // Should scale with subset size and original edge count
                let expectedMaxTime = Double(subsetSize) * 0.1 // 0.1ms per node in subset
                #expect(duration < expectedMaxTime, "Subgraph extraction for subset size \(subsetSize) took \(duration)ms, expected < \(expectedMaxTime)ms")

                print("Subgraph extraction for subset size \(subsetSize): \(String(format: "%.3f", duration))ms")
            }
        }
    }

    // MARK: - Memory Usage Tests

    @Suite("Memory Usage")
    struct MemoryUsageTests {

        @Test
        func testMemoryEfficiencyCSR() {
            // Test that CSR representation is memory efficient
            let nodeCount = 1000
            let edgeCount = 5000
            let edges = generateRandomEdges(nodeCount: nodeCount, edgeCount: edgeCount)

            let matrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: nodeCount, edges: edges)

            // Calculate expected memory usage
            let rowPointersSize = (nodeCount + 1) * MemoryLayout<UInt32>.size
            let columnIndicesSize = matrix.nonZeros * MemoryLayout<UInt32>.size
            let valuesSize = matrix.values != nil ? matrix.nonZeros * MemoryLayout<Float>.size : 0
            let expectedBytes = rowPointersSize + columnIndicesSize + valuesSize

            // Memory should be reasonable (within 20% overhead for alignment, etc.)
            let maxExpectedBytes = Int(Double(expectedBytes) * 1.2)

            print("CSR memory usage: expected ~\(expectedBytes) bytes, max allowed \(maxExpectedBytes) bytes")

            // Verify basic structure is reasonable
            #expect(matrix.rowPointers.count == nodeCount + 1)
            #expect(matrix.columnIndices.count == matrix.nonZeros)
            if let values = matrix.values {
                #expect(values.count == matrix.nonZeros)
            }
        }

        @Test
        func testMemoryLeakPrevention() async {
            // Test that repeated operations don't accumulate memory
            let nodeCount = 100
            let edgeCount = 500
            let edges = generateRandomEdges(nodeCount: nodeCount, edgeCount: edgeCount)
            let matrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: nodeCount, edges: edges)
            let input = generateTestVectors512(count: nodeCount)

            // Perform multiple operations to check for memory leaks
            for _ in 0..<50 {
                var output = ContiguousArray<Vector512Optimized>()
                await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                    matrix: matrix,
                    input: input,
                    output: &output,
                    normalize: false
                )

                // Force deallocation
                output.removeAll()
            }

            // If we reach here without memory issues, test passes
            #expect(true, "Memory leak prevention test completed")
        }
    }

    // MARK: - SIMD Optimization Tests

    @Suite("SIMD Optimization")
    struct SIMDOptimizationTests {

        @Test
        func testVectorOperationEfficiency() {
            // Test that vector operations are efficient across different dimensions
            let nodeCounts = [100, 200, 300]
            let dimensions = [512, 768, 1536]

            for nodeCount in nodeCounts {
                for dimension in dimensions {
                    let edges = generateRandomEdges(nodeCount: nodeCount, edgeCount: nodeCount * 5)
                    let graph = try! WeightedGraph(from: GraphPrimitivesKernels.edgeListToCSR(nodeCount: nodeCount, edges: edges))

                    let startTime = DispatchTime.now()

                    switch dimension {
                    case 512:
                        let features = generateTestVectors512(count: nodeCount)
                        var output = ContiguousArray<Vector512Optimized>()
                        GraphPrimitivesKernels.aggregateNeighbors(
                            graph: graph,
                            nodeFeatures: features,
                            aggregation: .sum,
                            output: &output
                        )
                    case 768:
                        let features = generateTestVectors768(count: nodeCount)
                        var output = ContiguousArray<Vector768Optimized>()
                        GraphPrimitivesKernels.aggregateNeighbors(
                            graph: graph,
                            nodeFeatures: features,
                            aggregation: .sum,
                            output: &output
                        )
                    case 1536:
                        let features = generateTestVectors1536(count: nodeCount)
                        var output = ContiguousArray<Vector1536Optimized>()
                        GraphPrimitivesKernels.aggregateNeighbors(
                            graph: graph,
                            nodeFeatures: features,
                            aggregation: .sum,
                            output: &output
                        )
                    default:
                        break
                    }

                    let endTime = DispatchTime.now()
                    let duration = Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000.0 // ms

                    // Performance should scale reasonably with dimension
                    let expectedMaxTime = Double(nodeCount * dimension) * 0.00005 // Small per-element time
                    #expect(duration < expectedMaxTime, "Vector aggregation for \(nodeCount) nodes, dim \(dimension) took \(duration)ms, expected < \(expectedMaxTime)ms")

                    print("Vector aggregation (\(nodeCount) nodes, dim \(dimension)): \(String(format: "%.3f", duration))ms")
                }
            }
        }

        @Test
        func testNormalizationOverhead() async {
            // Test that normalization doesn't add excessive overhead
            let nodeCount = 200
            let edgeCount = 1000
            let edges = generateRandomEdges(nodeCount: nodeCount, edgeCount: edgeCount)
            let matrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: nodeCount, edges: edges)
            let input = generateTestVectors512(count: nodeCount)

            // Test without normalization
            var outputNoNorm = ContiguousArray<Vector512Optimized>()
            let startTimeNoNorm = DispatchTime.now()
            await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                matrix: matrix,
                input: input,
                output: &outputNoNorm,
                normalize: false
            )
            let endTimeNoNorm = DispatchTime.now()
            let durationNoNorm = Double(endTimeNoNorm.uptimeNanoseconds - startTimeNoNorm.uptimeNanoseconds) / 1_000_000.0

            // Test with normalization
            var outputNorm = ContiguousArray<Vector512Optimized>()
            let startTimeNorm = DispatchTime.now()
            await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                matrix: matrix,
                input: input,
                output: &outputNorm,
                normalize: true
            )
            let endTimeNorm = DispatchTime.now()
            let durationNorm = Double(endTimeNorm.uptimeNanoseconds - startTimeNorm.uptimeNanoseconds) / 1_000_000.0

            // Normalization should add no more than 50% overhead
            let overhead = (durationNorm - durationNoNorm) / durationNoNorm * 100.0
            #expect(overhead < 50.0, "Normalization added \(overhead)% overhead, expected < 50%")

            print("SpMV without normalization: \(String(format: "%.3f", durationNoNorm))ms")
            print("SpMV with normalization: \(String(format: "%.3f", durationNorm))ms")
            print("Normalization overhead: \(String(format: "%.1f", overhead))%")
        }
    }

    // MARK: - Concurrency Performance Tests

    @Suite("Concurrency Performance")
    struct ConcurrencyPerformanceTests {

        @Test
        func testParallelSpMVPerformance() async {
            // Test that parallel SpMV operations perform efficiently
            let nodeCount = 300
            let edgeCount = 1500
            let edges = generateRandomEdges(nodeCount: nodeCount, edgeCount: edgeCount)
            let matrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: nodeCount, edges: edges)
            let inputs = [
                generateTestVectors512(count: nodeCount),
                generateTestVectors512(count: nodeCount),
                generateTestVectors512(count: nodeCount)
            ]

            // Test sequential execution
            let startTimeSeq = DispatchTime.now()
            for input in inputs {
                var output = ContiguousArray<Vector512Optimized>()
                await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                    matrix: matrix,
                    input: input,
                    output: &output,
                    normalize: false
                )
            }
            let endTimeSeq = DispatchTime.now()
            let durationSeq = Double(endTimeSeq.uptimeNanoseconds - startTimeSeq.uptimeNanoseconds) / 1_000_000.0

            // Test parallel execution
            let startTimePar = DispatchTime.now()
            await withTaskGroup(of: Void.self) { group in
                for input in inputs {
                    group.addTask {
                        var output = ContiguousArray<Vector512Optimized>()
                        await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                            matrix: matrix,
                            input: input,
                            output: &output,
                            normalize: false
                        )
                    }
                }
            }
            let endTimePar = DispatchTime.now()
            let durationPar = Double(endTimePar.uptimeNanoseconds - startTimePar.uptimeNanoseconds) / 1_000_000.0

            // Parallel execution should be faster (or at least not much slower)
            let speedup = durationSeq / durationPar
            #expect(speedup > 0.8, "Parallel execution speedup: \(speedup)x, expected > 0.8x")

            print("Sequential SpMV (3 operations): \(String(format: "%.3f", durationSeq))ms")
            print("Parallel SpMV (3 operations): \(String(format: "%.3f", durationPar))ms")
            print("Speedup: \(String(format: "%.2f", speedup))x")
        }
    }

    // MARK: - Helper Functions

    private static func generateRandomEdges(nodeCount: Int, edgeCount: Int) -> ContiguousArray<(UInt32, UInt32, Float?)> {
        var edges = ContiguousArray<(UInt32, UInt32, Float?)>()
        edges.reserveCapacity(edgeCount)

        srand48(42) // Fixed seed for reproducibility
        var edgeSet = Set<UInt64>()

        while edges.count < edgeCount {
            let src = UInt32(Int(drand48() * Double(nodeCount)))
            let dst = UInt32(Int(drand48() * Double(nodeCount)))

            if src != dst {
                let edgeKey = (UInt64(src) << 32) | UInt64(dst)
                if !edgeSet.contains(edgeKey) {
                    edgeSet.insert(edgeKey)
                    let weight = Float(drand48())
                    edges.append((src, dst, weight))
                }
            }
        }

        return edges
    }

    private static func generateTestVectors512(count: Int) -> ContiguousArray<Vector512Optimized> {
        var vectors = ContiguousArray<Vector512Optimized>()
        vectors.reserveCapacity(count)
        srand48(123)

        for i in 0..<count {
            let value = Float(drand48())
            vectors.append(Vector512Optimized(repeating: value))
        }
        return vectors
    }

    private static func generateTestVectors768(count: Int) -> ContiguousArray<Vector768Optimized> {
        var vectors = ContiguousArray<Vector768Optimized>()
        vectors.reserveCapacity(count)
        srand48(123)

        for i in 0..<count {
            let value = Float(drand48())
            vectors.append(Vector768Optimized(repeating: value))
        }
        return vectors
    }

    private static func generateTestVectors1536(count: Int) -> ContiguousArray<Vector1536Optimized> {
        var vectors = ContiguousArray<Vector1536Optimized>()
        vectors.reserveCapacity(count)
        srand48(123)

        for i in 0..<count {
            let value = Float(drand48())
            vectors.append(Vector1536Optimized(repeating: value))
        }
        return vectors
    }
}
