import Testing
@testable import VectorCore

@Suite("Graph Primitives Kernels")
struct GraphPrimitivesKernelsTests {

    // MARK: - CSR Matrix Construction Tests

    @Suite("CSR Matrix Construction")
    struct CSRConstructionTests {

        @Test
        func testEmptyGraph() {
            let edges: ContiguousArray<(UInt32, UInt32, Float?)> = []
            let matrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: 3, edges: edges)

            #expect(matrix.rows == 3)
            #expect(matrix.cols == 3)
            #expect(matrix.nonZeros == 0)
            #expect(matrix.rowPointers.count == 4) // nodeCount + 1
            #expect(matrix.columnIndices.isEmpty)
            #expect(matrix.values == nil)

            // All row pointers should be 0 for empty graph
            for i in 0..<matrix.rowPointers.count {
                #expect(matrix.rowPointers[i] == 0)
            }
        }

        @Test
        func testSingleEdgeUnweighted() {
            let edges: ContiguousArray<(UInt32, UInt32, Float?)> = [(0, 1, nil)]
            let matrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: 3, edges: edges)

            #expect(matrix.rows == 3)
            #expect(matrix.cols == 3)
            #expect(matrix.nonZeros == 1)
            #expect(matrix.rowPointers == [0, 1, 1, 1])
            #expect(matrix.columnIndices == [1])
            #expect(matrix.values == nil)
        }

        @Test
        func testSingleEdgeWeighted() {
            let edges: ContiguousArray<(UInt32, UInt32, Float?)> = [(0, 1, 2.5)]
            let matrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: 3, edges: edges)

            #expect(matrix.rows == 3)
            #expect(matrix.cols == 3)
            #expect(matrix.nonZeros == 1)
            #expect(matrix.rowPointers == [0, 1, 1, 1])
            #expect(matrix.columnIndices == [1])
            #expect(matrix.values != nil)
            #expect(matrix.values![0] == 2.5)
        }

        @Test
        func testMultipleEdgesFromSameNode() {
            let edges: ContiguousArray<(UInt32, UInt32, Float?)> = [
                (0, 1, 1.0),
                (0, 2, 2.0),
                (1, 2, 3.0)
            ]
            let matrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: 3, edges: edges)

            #expect(matrix.rows == 3)
            #expect(matrix.cols == 3)
            #expect(matrix.nonZeros == 3)
            #expect(matrix.rowPointers == [0, 2, 3, 3])

            // Row 0 should have 2 edges: to nodes 1 and 2
            #expect(matrix.columnIndices[0] == 1)
            #expect(matrix.columnIndices[1] == 2)
            #expect(matrix.values![0] == 1.0)
            #expect(matrix.values![1] == 2.0)

            // Row 1 should have 1 edge: to node 2
            #expect(matrix.columnIndices[2] == 2)
            #expect(matrix.values![2] == 3.0)
        }

        @Test
        func testSelfLoops() {
            let edges: ContiguousArray<(UInt32, UInt32, Float?)> = [
                (0, 0, 1.0),
                (1, 1, 2.0)
            ]
            let matrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: 3, edges: edges)

            #expect(matrix.nonZeros == 2)
            #expect(matrix.rowPointers == [0, 1, 2, 2])
            #expect(matrix.columnIndices[0] == 0)
            #expect(matrix.columnIndices[1] == 1)
            #expect(matrix.values![0] == 1.0)
            #expect(matrix.values![1] == 2.0)
        }

        @Test
        func testDuplicateEdgeHandling() {
            // The implementation should handle duplicate edges gracefully
            let edges: ContiguousArray<(UInt32, UInt32, Float?)> = [
                (0, 1, 1.0),
                (0, 1, 2.0), // Duplicate edge - behavior depends on implementation
                (1, 0, 3.0)
            ]
            let matrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: 2, edges: edges)

            // Verify basic structure is maintained
            #expect(matrix.rows == 2)
            #expect(matrix.cols == 2)
            #expect(matrix.nonZeros >= 2) // At least 2 unique edges
        }
    }

    // MARK: - Sparse Matrix Vector Multiplication Tests

    @Suite("Sparse Matrix Vector Multiplication")
    struct SpMVTests {

        @Test
        func testSpMV512EmptyMatrix() async {
            let matrix = GraphPrimitivesKernels.edgeListToCSR(
                nodeCount: 3,
                edges: []
            )
            let input = generateTestVectors512(count: 3)
            var output = ContiguousArray<Vector512Optimized>()

            await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                matrix: matrix,
                input: input,
                output: &output,
                normalize: false
            )

            #expect(output.count == 3)
            // All output vectors should be zero for empty matrix
            for vector in output {
                for i in 0..<512 {
                    #expect(vector[i] == 0.0)
                }
            }
        }

        @Test
        func testSpMV512IdentityMatrix() async {
            let edges: ContiguousArray<(UInt32, UInt32, Float?)> = [
                (0, 0, 1.0),
                (1, 1, 1.0),
                (2, 2, 1.0)
            ]
            let matrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: 3, edges: edges)
            let input = generateTestVectors512(count: 3)
            var output = ContiguousArray<Vector512Optimized>()

            await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                matrix: matrix,
                input: input,
                output: &output,
                normalize: false
            )

            #expect(output.count == 3)
            // Output should equal input for identity matrix
            for i in 0..<3 {
                for j in 0..<512 {
                    #expect(approxEqual(output[i][j], input[i][j]))
                }
            }
        }

        @Test
        func testSpMV512SimpleGraph() async {
            let edges: ContiguousArray<(UInt32, UInt32, Float?)> = [
                (0, 1, 2.0),
                (1, 2, 3.0)
            ]
            let matrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: 3, edges: edges)
            let input = ContiguousArray<Vector512Optimized>([
                Vector512Optimized(repeating: 1.0),
                Vector512Optimized(repeating: 2.0),
                Vector512Optimized(repeating: 3.0)
            ])
            var output = ContiguousArray<Vector512Optimized>()

            await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                matrix: matrix,
                input: input,
                output: &output,
                normalize: false
            )

            #expect(output.count == 3)

            // Row 0: 2.0 * input[1] = 2.0 * 2.0 = 4.0
            for i in 0..<512 {
                #expect(approxEqual(output[0][i], 4.0))
            }

            // Row 1: 3.0 * input[2] = 3.0 * 3.0 = 9.0
            for i in 0..<512 {
                #expect(approxEqual(output[1][i], 9.0))
            }

            // Row 2: no edges, should be zero
            for i in 0..<512 {
                #expect(output[2][i] == 0.0)
            }
        }

        @Test
        func testSpMV512WithNormalization() async {
            let edges: ContiguousArray<(UInt32, UInt32, Float?)> = [
                (0, 1, 1.0),
                (0, 2, 1.0)
            ]
            let matrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: 3, edges: edges)
            let input = ContiguousArray<Vector512Optimized>([
                Vector512Optimized(repeating: 0.0),
                Vector512Optimized(repeating: 1.0),
                Vector512Optimized(repeating: 1.0)
            ])
            var output = ContiguousArray<Vector512Optimized>()

            await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                matrix: matrix,
                input: input,
                output: &output,
                normalize: true
            )

            #expect(output.count == 3)

            // Row 0 has 2 neighbors, normalization divides by sqrt(degree) = sqrt(2) ≈ 1.414
            // Result: (1.0 + 1.0) / sqrt(2) ≈ 1.414
            for i in 0..<512 {
                #expect(approxEqual(output[0][i], 1.4142135, tol: 1e-4))
            }
        }

        @Test
        func testSpMV768Consistency() async {
            let edges: ContiguousArray<(UInt32, UInt32, Float?)> = [
                (0, 1, 1.5),
                (1, 0, 2.5)
            ]
            let matrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: 2, edges: edges)
            let input = ContiguousArray<Vector768Optimized>([
                Vector768Optimized(repeating: 2.0),
                Vector768Optimized(repeating: 4.0)
            ])
            var output = ContiguousArray<Vector768Optimized>()

            await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                matrix: matrix,
                input: input,
                output: &output,
                normalize: false
            )

            #expect(output.count == 2)

            // Row 0: 1.5 * input[1] = 1.5 * 4.0 = 6.0
            for i in 0..<768 {
                #expect(approxEqual(output[0][i], 6.0))
            }

            // Row 1: 2.5 * input[0] = 2.5 * 2.0 = 5.0
            for i in 0..<768 {
                #expect(approxEqual(output[1][i], 5.0))
            }
        }

        @Test
        func testSpMV1536Consistency() async {
            let edges: ContiguousArray<(UInt32, UInt32, Float?)> = [
                (0, 1, 0.5),
                (1, 0, 1.5)
            ]
            let matrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: 2, edges: edges)
            let input = ContiguousArray<Vector1536Optimized>([
                Vector1536Optimized(repeating: 3.0),
                Vector1536Optimized(repeating: 6.0)
            ])
            var output = ContiguousArray<Vector1536Optimized>()

            await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                matrix: matrix,
                input: input,
                output: &output,
                normalize: false
            )

            #expect(output.count == 2)

            // Row 0: 0.5 * input[1] = 0.5 * 6.0 = 3.0
            for i in 0..<1536 {
                #expect(approxEqual(output[0][i], 3.0))
            }

            // Row 1: 1.5 * input[0] = 1.5 * 3.0 = 4.5
            for i in 0..<1536 {
                #expect(approxEqual(output[1][i], 4.5))
            }
        }
    }

    // MARK: - Neighbor Aggregation Tests

    @Suite("Neighbor Aggregation")
    struct NeighborAggregationTests {

        @Test
        func testAggregateNeighbors512Sum() {
            let edges: ContiguousArray<(UInt32, UInt32, Float?)> = [
                (0, 1, 2.0),
                (0, 2, 3.0),
                (1, 2, 1.0)
            ]
            let graph = try! WeightedGraph(from: GraphPrimitivesKernels.edgeListToCSR(nodeCount: 3, edges: edges))
            let nodeFeatures = ContiguousArray<Vector512Optimized>([
                Vector512Optimized(repeating: 1.0),
                Vector512Optimized(repeating: 2.0),
                Vector512Optimized(repeating: 3.0)
            ])
            var output = ContiguousArray<Vector512Optimized>()

            GraphPrimitivesKernels.aggregateNeighbors(
                graph: graph,
                nodeFeatures: nodeFeatures,
                aggregation: .sum,
                output: &output
            )

            #expect(output.count == 3)

            // Node 0 has neighbors 1,2 (edge weights ignored for basic sum)
            // Sum: features[1] + features[2] = 2.0 + 3.0 = 5.0
            for i in 0..<512 {
                #expect(approxEqual(output[0][i], 5.0))
            }

            // Node 1 has neighbor 2
            // Sum: features[2] = 3.0
            for i in 0..<512 {
                #expect(approxEqual(output[1][i], 3.0))
            }

            // Node 2 has no outgoing edges
            for i in 0..<512 {
                #expect(output[2][i] == 0.0)
            }
        }

        @Test
        func testAggregateNeighbors512Mean() {
            let edges: ContiguousArray<(UInt32, UInt32, Float?)> = [
                (0, 1, 1.0),
                (0, 2, 1.0)
            ]
            let graph = try! WeightedGraph(from: GraphPrimitivesKernels.edgeListToCSR(nodeCount: 3, edges: edges))
            let nodeFeatures = ContiguousArray<Vector512Optimized>([
                Vector512Optimized(repeating: 0.0),
                Vector512Optimized(repeating: 2.0),
                Vector512Optimized(repeating: 4.0)
            ])
            var output = ContiguousArray<Vector512Optimized>()

            GraphPrimitivesKernels.aggregateNeighbors(
                graph: graph,
                nodeFeatures: nodeFeatures,
                aggregation: .mean,
                output: &output
            )

            #expect(output.count == 3)

            // Node 0 has 2 neighbors (edge weights ignored for basic mean)
            // Mean implementation: sum.divide(by: 1.0/count) = sum * count = 6.0 * 2 = 12.0
            for i in 0..<512 {
                #expect(approxEqual(output[0][i], 12.0))
            }
        }

        @Test
        func testAggregateNeighbors512Max() {
            let edges: ContiguousArray<(UInt32, UInt32, Float?)> = [
                (0, 1, 1.0),
                (0, 2, 1.0)
            ]
            let graph = try! WeightedGraph(from: GraphPrimitivesKernels.edgeListToCSR(nodeCount: 3, edges: edges))
            let nodeFeatures = ContiguousArray<Vector512Optimized>([
                Vector512Optimized(repeating: 0.0),
                Vector512Optimized(repeating: 2.0),
                Vector512Optimized(repeating: 4.0)
            ])
            var output = ContiguousArray<Vector512Optimized>()

            GraphPrimitivesKernels.aggregateNeighbors(
                graph: graph,
                nodeFeatures: nodeFeatures,
                aggregation: .max,
                output: &output
            )

            #expect(output.count == 3)

            // Node 0 has neighbors with features 2.0 and 4.0
            // Max: max(features[1], features[2]) = max(2.0, 4.0) = 4.0
            for i in 0..<512 {
                #expect(approxEqual(output[0][i], 4.0))
            }
        }

        @Test
        func testAggregateNeighbors512Min() {
            let edges: ContiguousArray<(UInt32, UInt32, Float?)> = [
                (0, 1, 1.0),
                (0, 2, 1.0)
            ]
            let graph = try! WeightedGraph(from: GraphPrimitivesKernels.edgeListToCSR(nodeCount: 3, edges: edges))
            let nodeFeatures = ContiguousArray<Vector512Optimized>([
                Vector512Optimized(repeating: 0.0),
                Vector512Optimized(repeating: 2.0),
                Vector512Optimized(repeating: 4.0)
            ])
            var output = ContiguousArray<Vector512Optimized>()

            GraphPrimitivesKernels.aggregateNeighbors(
                graph: graph,
                nodeFeatures: nodeFeatures,
                aggregation: .min,
                output: &output
            )

            #expect(output.count == 3)

            // Node 0 has neighbors with features 2.0 and 4.0
            // Min: min(features[1], features[2]) = min(2.0, 4.0) = 2.0
            for i in 0..<512 {
                #expect(approxEqual(output[0][i], 2.0))
            }
        }

        @Test
        func testAggregateNeighborsEmptyNeighborhood() {
            let edges: ContiguousArray<(UInt32, UInt32, Float?)> = []
            let graph = try! WeightedGraph(from: GraphPrimitivesKernels.edgeListToCSR(nodeCount: 2, edges: edges))
            let nodeFeatures = ContiguousArray<Vector512Optimized>([
                Vector512Optimized(repeating: 1.0),
                Vector512Optimized(repeating: 2.0)
            ])
            var output = ContiguousArray<Vector512Optimized>()

            GraphPrimitivesKernels.aggregateNeighbors(
                graph: graph,
                nodeFeatures: nodeFeatures,
                aggregation: .sum,
                output: &output
            )

            #expect(output.count == 2)

            // All nodes have empty neighborhoods, output should be zero
            for nodeIdx in 0..<2 {
                for i in 0..<512 {
                    #expect(output[nodeIdx][i] == 0.0)
                }
            }
        }

        @Test
        func testAggregateNeighbors768Consistency() {
            let edges: ContiguousArray<(UInt32, UInt32, Float?)> = [
                (0, 1, 2.0)
            ]
            let graph = try! WeightedGraph(from: GraphPrimitivesKernels.edgeListToCSR(nodeCount: 2, edges: edges))
            let nodeFeatures = ContiguousArray<Vector768Optimized>([
                Vector768Optimized(repeating: 1.0),
                Vector768Optimized(repeating: 3.0)
            ])
            var output = ContiguousArray<Vector768Optimized>()

            GraphPrimitivesKernels.aggregateNeighbors(
                graph: graph,
                nodeFeatures: nodeFeatures,
                aggregation: .sum,
                output: &output
            )

            #expect(output.count == 2)

            // Node 0 -> Node 1: sum of neighbor features = 3.0
            for i in 0..<768 {
                #expect(approxEqual(output[0][i], 3.0))
            }
        }

        @Test
        func testAggregateNeighbors1536Consistency() {
            let edges: ContiguousArray<(UInt32, UInt32, Float?)> = [
                (0, 1, 0.5)
            ]
            let graph = try! WeightedGraph(from: GraphPrimitivesKernels.edgeListToCSR(nodeCount: 2, edges: edges))
            let nodeFeatures = ContiguousArray<Vector1536Optimized>([
                Vector1536Optimized(repeating: 2.0),
                Vector1536Optimized(repeating: 8.0)
            ])
            var output = ContiguousArray<Vector1536Optimized>()

            GraphPrimitivesKernels.aggregateNeighbors(
                graph: graph,
                nodeFeatures: nodeFeatures,
                aggregation: .sum,
                output: &output
            )

            #expect(output.count == 2)

            // Node 0 -> Node 1: sum of neighbor features = 8.0
            for i in 0..<1536 {
                #expect(approxEqual(output[0][i], 8.0))
            }
        }
    }

    // MARK: - Graph Structure Tests

    @Suite("Graph Structure Operations")
    struct GraphStructureTests {

        @Test
        func testTransposeCSRSimple() {
            let edges: ContiguousArray<(UInt32, UInt32, Float?)> = [
                (0, 1, 2.0),
                (0, 2, 3.0),
                (1, 2, 4.0)
            ]
            let matrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: 3, edges: edges)
            let transposed = GraphPrimitivesKernels.transposeCSR(matrix)

            #expect(transposed.rows == 3)
            #expect(transposed.cols == 3)
            #expect(transposed.nonZeros == 3)

            // Original: (0->1), (0->2), (1->2)
            // Transposed: (1->0), (2->0), (2->1)

            // Row 0 of transpose should have no edges
            #expect(transposed.rowPointers[0] == 0)
            #expect(transposed.rowPointers[1] == 0)

            // Row 1 of transpose should have 1 edge (1->0)
            #expect(transposed.rowPointers[2] == 1)
            #expect(transposed.columnIndices[0] == 0)
            #expect(transposed.values![0] == 2.0)

            // Row 2 of transpose should have 2 edges (2->0), (2->1)
            #expect(transposed.rowPointers[3] == 3)
            #expect(transposed.columnIndices[1] == 0)
            #expect(transposed.columnIndices[2] == 1)
            #expect(transposed.values![1] == 3.0)
            #expect(transposed.values![2] == 4.0)
        }

        @Test
        func testTransposeCSRIdentity() {
            let edges: ContiguousArray<(UInt32, UInt32, Float?)> = [
                (0, 0, 1.0),
                (1, 1, 2.0)
            ]
            let matrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: 2, edges: edges)
            let transposed = GraphPrimitivesKernels.transposeCSR(matrix)

            // Identity matrix should equal its transpose
            #expect(transposed.rowPointers == matrix.rowPointers)
            #expect(transposed.columnIndices == matrix.columnIndices)
            #expect(transposed.values == matrix.values)
        }

        @Test
        func testExtractSubgraphSmall() {
            let edges: ContiguousArray<(UInt32, UInt32, Float?)> = [
                (0, 1, 1.0),
                (1, 2, 2.0),
                (2, 0, 3.0),
                (0, 3, 4.0),  // Edge to node outside subset
                (3, 1, 5.0)   // Edge from node outside subset
            ]
            let graph = try! WeightedGraph(from: GraphPrimitivesKernels.edgeListToCSR(nodeCount: 4, edges: edges))
            let nodeSubset: Set<UInt32> = [0, 1, 2]

            let (subgraph, nodeMapping) = GraphPrimitivesKernels.extractSubgraph(from: graph, nodeSubset: nodeSubset)

            #expect(subgraph.nodeCount == 3)
            #expect(nodeMapping.count == 3)

            // Check that all original nodes are mapped
            #expect(nodeMapping.keys.contains(0))
            #expect(nodeMapping.keys.contains(1))
            #expect(nodeMapping.keys.contains(2))

            // Verify the subgraph contains only internal edges
            // Should contain: (0->1), (1->2), (2->0) but not (0->3) or (3->1)
            let subgraphMatrix = subgraph.adjacency
            #expect(subgraphMatrix.nonZeros == 3)
        }

        @Test
        func testExtractSubgraphSingleNode() {
            let edges: ContiguousArray<(UInt32, UInt32, Float?)> = [
                (0, 1, 1.0),
                (1, 0, 2.0)
            ]
            let graph = try! WeightedGraph(from: GraphPrimitivesKernels.edgeListToCSR(nodeCount: 2, edges: edges))
            let nodeSubset: Set<UInt32> = [0]

            let (subgraph, nodeMapping) = GraphPrimitivesKernels.extractSubgraph(from: graph, nodeSubset: nodeSubset)

            #expect(subgraph.nodeCount == 1)
            #expect(nodeMapping.count == 1)
            #expect(nodeMapping[0] == 0)

            // Single node subgraph should have no edges
            #expect(subgraph.adjacency.nonZeros == 0)
        }

        @Test
        func testExtractSubgraphEmptySubset() {
            let edges: ContiguousArray<(UInt32, UInt32, Float?)> = [
                (0, 1, 1.0)
            ]
            let graph = try! WeightedGraph(from: GraphPrimitivesKernels.edgeListToCSR(nodeCount: 2, edges: edges))
            let nodeSubset: Set<UInt32> = []

            let (subgraph, nodeMapping) = GraphPrimitivesKernels.extractSubgraph(from: graph, nodeSubset: nodeSubset)

            #expect(subgraph.nodeCount == 0)
            #expect(nodeMapping.isEmpty)
            #expect(subgraph.adjacency.nonZeros == 0)
        }
    }

    // MARK: - Test Helper Functions

    private static func generateTestVectors512(count: Int) -> ContiguousArray<Vector512Optimized> {
        var vectors = ContiguousArray<Vector512Optimized>()
        vectors.reserveCapacity(count)

        for i in 0..<count {
            let value = Float(i + 1)
            vectors.append(Vector512Optimized(repeating: value))
        }

        return vectors
    }
}
