import Foundation
import VectorCore

struct GraphPrimitivesBench: BenchmarkSuite {
    static let name = "graph"

    static func run(options: CLIOptions) async -> [BenchResult] {
        var results: [BenchResult] = []
        for dim in options.dims {
            switch dim {
            case 512: results += await bench512(options)
            case 768: results += await bench768(options)
            case 1536: results += await bench1536(options)
            default:
                fputs("[graph] unsupported dimension: \(dim)\n", stderr)
            }
        }
        return results
    }

    private static func bench512(_ options: CLIOptions) async -> [BenchResult] {
        return await benchmarkDimension(512, options)
    }

    private static func bench768(_ options: CLIOptions) async -> [BenchResult] {
        return await benchmarkDimension(768, options)
    }

    private static func bench1536(_ options: CLIOptions) async -> [BenchResult] {
        return await benchmarkDimension(1536, options)
    }

    private static func benchmarkDimension(_ dim: Int, _ options: CLIOptions) async -> [BenchResult] {
        var results: [BenchResult] = []

        // Test different graph sizes
        let graphSizes = [50, 100, 200]
        let sparsityLevels: [Float] = [0.01, 0.02] // 1%, 2% edge density

        for nodeCount in graphSizes {
            for sparsity in sparsityLevels {
                results += await benchmarkGraphSize(dim: dim, nodeCount: nodeCount, sparsity: sparsity, options: options)
            }
        }

        return results
    }

    private static func benchmarkGraphSize(dim: Int, nodeCount: Int, sparsity: Float, options: CLIOptions) async -> [BenchResult] {
        var results: [BenchResult] = []

        // Generate test data
        let edgeCount = Int(Float(nodeCount * nodeCount) * sparsity)
        let edges = generateRandomEdges(nodeCount: nodeCount, edgeCount: edgeCount)

        // Create sparse matrix
        let matrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: nodeCount, edges: edges)
        let graph = try! WeightedGraph(from: matrix)

        // Benchmark CSR Construction
        results.append(benchmarkCSRConstruction(
            nodeCount: nodeCount,
            edges: edges,
            dim: dim,
            sparsity: sparsity,
            options: options
        ))

        // Dispatch to dimension-specific benchmarks
        switch dim {
        case 512:
            let nodeFeatures = generateRandomVectors512(count: nodeCount)
            results += await benchmarkGraphOperations512(
                matrix: matrix,
                graph: graph,
                nodeFeatures: nodeFeatures,
                nodeCount: nodeCount,
                sparsity: sparsity,
                options: options
            )
        case 768:
            let nodeFeatures = generateRandomVectors768(count: nodeCount)
            results += await benchmarkGraphOperations768(
                matrix: matrix,
                graph: graph,
                nodeFeatures: nodeFeatures,
                nodeCount: nodeCount,
                sparsity: sparsity,
                options: options
            )
        case 1536:
            let nodeFeatures = generateRandomVectors1536(count: nodeCount)
            results += await benchmarkGraphOperations1536(
                matrix: matrix,
                graph: graph,
                nodeFeatures: nodeFeatures,
                nodeCount: nodeCount,
                sparsity: sparsity,
                options: options
            )
        default:
            break
        }

        return results
    }

    // MARK: - Dimension-Specific Benchmark Functions

    private static func benchmarkGraphOperations512(
        matrix: SparseMatrix,
        graph: WeightedGraph,
        nodeFeatures: ContiguousArray<Vector512Optimized>,
        nodeCount: Int,
        sparsity: Float,
        options: CLIOptions
    ) async -> [BenchResult] {
        var results: [BenchResult] = []
        results += await benchmarkSpMV512(matrix: matrix, nodeFeatures: nodeFeatures, nodeCount: nodeCount, sparsity: sparsity, options: options)
        results.append(benchmarkNeighborAggregation512(graph: graph, nodeFeatures: nodeFeatures, nodeCount: nodeCount, sparsity: sparsity, options: options))
        results += benchmarkGraphStructure(graph: graph, matrix: matrix, dim: 512, nodeCount: nodeCount, sparsity: sparsity, options: options)
        return results
    }

    private static func benchmarkGraphOperations768(
        matrix: SparseMatrix,
        graph: WeightedGraph,
        nodeFeatures: ContiguousArray<Vector768Optimized>,
        nodeCount: Int,
        sparsity: Float,
        options: CLIOptions
    ) async -> [BenchResult] {
        var results: [BenchResult] = []
        results += await benchmarkSpMV768(matrix: matrix, nodeFeatures: nodeFeatures, nodeCount: nodeCount, sparsity: sparsity, options: options)
        results.append(benchmarkNeighborAggregation768(graph: graph, nodeFeatures: nodeFeatures, nodeCount: nodeCount, sparsity: sparsity, options: options))
        results += benchmarkGraphStructure(graph: graph, matrix: matrix, dim: 768, nodeCount: nodeCount, sparsity: sparsity, options: options)
        return results
    }

    private static func benchmarkGraphOperations1536(
        matrix: SparseMatrix,
        graph: WeightedGraph,
        nodeFeatures: ContiguousArray<Vector1536Optimized>,
        nodeCount: Int,
        sparsity: Float,
        options: CLIOptions
    ) async -> [BenchResult] {
        var results: [BenchResult] = []
        results += await benchmarkSpMV1536(matrix: matrix, nodeFeatures: nodeFeatures, nodeCount: nodeCount, sparsity: sparsity, options: options)
        results.append(benchmarkNeighborAggregation1536(graph: graph, nodeFeatures: nodeFeatures, nodeCount: nodeCount, sparsity: sparsity, options: options))
        results += benchmarkGraphStructure(graph: graph, matrix: matrix, dim: 1536, nodeCount: nodeCount, sparsity: sparsity, options: options)
        return results
    }

    private static func benchmarkCSRConstruction(
        nodeCount: Int,
        edges: ContiguousArray<(UInt32, UInt32, Float?)>,
        dim: Int,
        sparsity: Float,
        options: CLIOptions
    ) -> BenchResult {
        let sparsityStr = String(format: "%.3f", sparsity)
        let name = "graph.csr_construction.\(dim).nodes_\(nodeCount).sparsity_\(sparsityStr)"

        Harness.warmup {
            let _ = GraphPrimitivesKernels.edgeListToCSR(nodeCount: nodeCount, edges: edges)
        }

        return Harness.measure(
            name: name,
            minTimeSeconds: options.minTimeSeconds,
            repeats: options.repeats,
            samples: options.samples
        ) {
            let matrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: nodeCount, edges: edges)
            blackHole(matrix)
        }
    }

    // MARK: - SpMV Benchmarks

    private static func benchmarkSpMV512(
        matrix: SparseMatrix,
        nodeFeatures: ContiguousArray<Vector512Optimized>,
        nodeCount: Int,
        sparsity: Float,
        options: CLIOptions
    ) async -> [BenchResult] {
        var results: [BenchResult] = []
        let sparsityStr = String(format: "%.3f", sparsity)

        for normalize in [false, true] {
            let normalizeStr = normalize ? "normalized" : "raw"
            let name = "graph.spmv.512.nodes_\(nodeCount).sparsity_\(sparsityStr).\(normalizeStr)"

            await Harness.warmupAsync {
                var output = ContiguousArray<Vector512Optimized>()
                await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                    matrix: matrix, input: nodeFeatures, output: &output, normalize: normalize
                )
            }

            let result = await Harness.measureAsync(
                name: name, minTimeSeconds: options.minTimeSeconds,
                repeats: options.repeats, samples: options.samples
            ) {
                var output = ContiguousArray<Vector512Optimized>()
                await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                    matrix: matrix, input: nodeFeatures, output: &output, normalize: normalize
                )
                blackHole(output)
            }
            results.append(result)
        }
        return results
    }

    private static func benchmarkSpMV768(
        matrix: SparseMatrix,
        nodeFeatures: ContiguousArray<Vector768Optimized>,
        nodeCount: Int,
        sparsity: Float,
        options: CLIOptions
    ) async -> [BenchResult] {
        var results: [BenchResult] = []
        let sparsityStr = String(format: "%.3f", sparsity)

        for normalize in [false, true] {
            let normalizeStr = normalize ? "normalized" : "raw"
            let name = "graph.spmv.768.nodes_\(nodeCount).sparsity_\(sparsityStr).\(normalizeStr)"

            await Harness.warmupAsync {
                var output = ContiguousArray<Vector768Optimized>()
                await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                    matrix: matrix, input: nodeFeatures, output: &output, normalize: normalize
                )
            }

            let result = await Harness.measureAsync(
                name: name, minTimeSeconds: options.minTimeSeconds,
                repeats: options.repeats, samples: options.samples
            ) {
                var output = ContiguousArray<Vector768Optimized>()
                await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                    matrix: matrix, input: nodeFeatures, output: &output, normalize: normalize
                )
                blackHole(output)
            }
            results.append(result)
        }
        return results
    }

    private static func benchmarkSpMV1536(
        matrix: SparseMatrix,
        nodeFeatures: ContiguousArray<Vector1536Optimized>,
        nodeCount: Int,
        sparsity: Float,
        options: CLIOptions
    ) async -> [BenchResult] {
        var results: [BenchResult] = []
        let sparsityStr = String(format: "%.3f", sparsity)

        for normalize in [false, true] {
            let normalizeStr = normalize ? "normalized" : "raw"
            let name = "graph.spmv.1536.nodes_\(nodeCount).sparsity_\(sparsityStr).\(normalizeStr)"

            await Harness.warmupAsync {
                var output = ContiguousArray<Vector1536Optimized>()
                await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                    matrix: matrix, input: nodeFeatures, output: &output, normalize: normalize
                )
            }

            let result = await Harness.measureAsync(
                name: name, minTimeSeconds: options.minTimeSeconds,
                repeats: options.repeats, samples: options.samples
            ) {
                var output = ContiguousArray<Vector1536Optimized>()
                await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                    matrix: matrix, input: nodeFeatures, output: &output, normalize: normalize
                )
                blackHole(output)
            }
            results.append(result)
        }
        return results
    }

    // MARK: - Neighbor Aggregation Benchmarks

    private static func benchmarkNeighborAggregation512(
        graph: WeightedGraph,
        nodeFeatures: ContiguousArray<Vector512Optimized>,
        nodeCount: Int,
        sparsity: Float,
        options: CLIOptions
    ) -> BenchResult {
        let sparsityStr = String(format: "%.3f", sparsity)
        let name = "graph.neighbor_aggregation.512.nodes_\(nodeCount).sparsity_\(sparsityStr)"

        Harness.warmup {
            var output = ContiguousArray<Vector512Optimized>()
            GraphPrimitivesKernels.aggregateNeighbors(
                graph: graph, nodeFeatures: nodeFeatures, aggregation: .sum, output: &output
            )
        }

        return Harness.measure(
            name: name, minTimeSeconds: options.minTimeSeconds,
            repeats: options.repeats, samples: options.samples
        ) {
            var output = ContiguousArray<Vector512Optimized>()
            GraphPrimitivesKernels.aggregateNeighbors(
                graph: graph, nodeFeatures: nodeFeatures, aggregation: .sum, output: &output
            )
            blackHole(output)
        }
    }

    private static func benchmarkNeighborAggregation768(
        graph: WeightedGraph,
        nodeFeatures: ContiguousArray<Vector768Optimized>,
        nodeCount: Int,
        sparsity: Float,
        options: CLIOptions
    ) -> BenchResult {
        let sparsityStr = String(format: "%.3f", sparsity)
        let name = "graph.neighbor_aggregation.768.nodes_\(nodeCount).sparsity_\(sparsityStr)"

        Harness.warmup {
            var output = ContiguousArray<Vector768Optimized>()
            GraphPrimitivesKernels.aggregateNeighbors(
                graph: graph, nodeFeatures: nodeFeatures, aggregation: .sum, output: &output
            )
        }

        return Harness.measure(
            name: name, minTimeSeconds: options.minTimeSeconds,
            repeats: options.repeats, samples: options.samples
        ) {
            var output = ContiguousArray<Vector768Optimized>()
            GraphPrimitivesKernels.aggregateNeighbors(
                graph: graph, nodeFeatures: nodeFeatures, aggregation: .sum, output: &output
            )
            blackHole(output)
        }
    }

    private static func benchmarkNeighborAggregation1536(
        graph: WeightedGraph,
        nodeFeatures: ContiguousArray<Vector1536Optimized>,
        nodeCount: Int,
        sparsity: Float,
        options: CLIOptions
    ) -> BenchResult {
        let sparsityStr = String(format: "%.3f", sparsity)
        let name = "graph.neighbor_aggregation.1536.nodes_\(nodeCount).sparsity_\(sparsityStr)"

        Harness.warmup {
            var output = ContiguousArray<Vector1536Optimized>()
            GraphPrimitivesKernels.aggregateNeighbors(
                graph: graph, nodeFeatures: nodeFeatures, aggregation: .sum, output: &output
            )
        }

        return Harness.measure(
            name: name, minTimeSeconds: options.minTimeSeconds,
            repeats: options.repeats, samples: options.samples
        ) {
            var output = ContiguousArray<Vector1536Optimized>()
            GraphPrimitivesKernels.aggregateNeighbors(
                graph: graph, nodeFeatures: nodeFeatures, aggregation: .sum, output: &output
            )
            blackHole(output)
        }
    }

    private static func benchmarkGraphStructure(
        graph: WeightedGraph,
        matrix: SparseMatrix,
        dim: Int,
        nodeCount: Int,
        sparsity: Float,
        options: CLIOptions
    ) -> [BenchResult] {
        var results: [BenchResult] = []
        let sparsityStr = String(format: "%.3f", sparsity)

        // Benchmark matrix transpose
        let transposeName = "graph.transpose.\(dim).nodes_\(nodeCount).sparsity_\(sparsityStr)"
        Harness.warmup {
            let _ = GraphPrimitivesKernels.transposeCSR(matrix)
        }

        let transposeResult = Harness.measure(
            name: transposeName,
            minTimeSeconds: options.minTimeSeconds,
            repeats: options.repeats,
            samples: options.samples
        ) {
            let transposed = GraphPrimitivesKernels.transposeCSR(matrix)
            blackHole(transposed)
        }
        results.append(transposeResult)

        // Benchmark subgraph extraction (if graph is large enough)
        if nodeCount > 100 {
            let subsetSize = min(nodeCount / 2, 100)
            let nodeSubset = Set<UInt32>((0..<subsetSize).map { UInt32($0) })

            let subgraphName = "graph.subgraph.\(dim).nodes_\(nodeCount).subset_\(subsetSize).sparsity_\(sparsityStr)"
            Harness.warmup {
                let _ = GraphPrimitivesKernels.extractSubgraph(from: graph, nodeSubset: nodeSubset)
            }

            let subgraphResult = Harness.measure(
                name: subgraphName,
                minTimeSeconds: options.minTimeSeconds,
                repeats: options.repeats,
                samples: options.samples
            ) {
                let (subgraph, _) = GraphPrimitivesKernels.extractSubgraph(from: graph, nodeSubset: nodeSubset)
                blackHole(subgraph)
            }
            results.append(subgraphResult)
        }

        return results
    }

    // MARK: - Test Data Generation

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

    private static func generateRandomVectors512(count: Int) -> ContiguousArray<Vector512Optimized> {
        var vectors = ContiguousArray<Vector512Optimized>()
        vectors.reserveCapacity(count)
        srand48(123)

        for _ in 0..<count {
            let array = (0..<512).map { _ in Float(drand48() * 2.0 - 1.0) }
            let vector = try! Vector512Optimized(array)
            vectors.append(vector)
        }
        return vectors
    }

    private static func generateRandomVectors768(count: Int) -> ContiguousArray<Vector768Optimized> {
        var vectors = ContiguousArray<Vector768Optimized>()
        vectors.reserveCapacity(count)
        srand48(123)

        for _ in 0..<count {
            let array = (0..<768).map { _ in Float(drand48() * 2.0 - 1.0) }
            let vector = try! Vector768Optimized(array)
            vectors.append(vector)
        }
        return vectors
    }

    private static func generateRandomVectors1536(count: Int) -> ContiguousArray<Vector1536Optimized> {
        var vectors = ContiguousArray<Vector1536Optimized>()
        vectors.reserveCapacity(count)
        srand48(123)

        for _ in 0..<count {
            let array = (0..<1536).map { _ in Float(drand48() * 2.0 - 1.0) }
            let vector = try! Vector1536Optimized(array)
            vectors.append(vector)
        }
        return vectors
    }
}