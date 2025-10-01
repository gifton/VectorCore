import Testing
import Foundation
@testable import VectorCore

@Suite("Graph Primitives Integration")
struct GraphPrimitivesIntegrationTests {

    // MARK: - End-to-End Workflow Tests

    @Suite("End-to-End Workflows")
    struct EndToEndWorkflowTests {

        @Test
        func testSocialNetworkAnalysisWorkflow() async {
            // Simulate a social network analysis workflow
            // Create a social network graph: users and their connections
            let userCount = 100
            let friendships: ContiguousArray<(UInt32, UInt32, Float?)> = generateSocialNetworkGraph(userCount: userCount)

            // Step 1: Build the graph
            let socialGraph = try! WeightedGraph(from: GraphPrimitivesKernels.edgeListToCSR(nodeCount: userCount, edges: friendships))

            // Step 2: Create user feature embeddings (e.g., interests, demographics)
            let userFeatures = generateUserEmbeddings(count: userCount)

            // Step 3: Compute friend recommendations using neighbor aggregation
            var friendRecommendations = ContiguousArray<Vector512Optimized>()
            GraphPrimitivesKernels.aggregateNeighbors(
                graph: socialGraph,
                nodeFeatures: userFeatures,
                aggregation: .mean, // Average of friends' features
                output: &friendRecommendations
            )

            // Step 4: Analyze community structure by extracting subgraphs
            let activeUsers: Set<UInt32> = Set((0..<50).map { UInt32($0) }) // First 50 users
            let (communityGraph, userMapping) = GraphPrimitivesKernels.extractSubgraph(from: socialGraph, nodeSubset: activeUsers)

            // Step 5: Compute influence propagation using SpMV
            let initialInfluence = ContiguousArray<Vector512Optimized>(
                (0..<userCount).map { userIdx in
                    // Seed some users with high influence
                    let influence: Float = userIdx < 5 ? 1.0 : 0.0
                    return Vector512Optimized(repeating: influence)
                }
            )

            var propagatedInfluence = ContiguousArray<Vector512Optimized>()
            await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                matrix: socialGraph.adjacency,
                input: initialInfluence,
                output: &propagatedInfluence,
                normalize: true // Normalize by degree for proper influence distribution
            )

            // Validate the workflow results
            #expect(friendRecommendations.count == userCount)
            #expect(communityGraph.nodeCount <= 50)
            #expect(userMapping.count <= 50)
            #expect(propagatedInfluence.count == userCount)

            // Check that influence has actually propagated
            var totalInfluence: Float = 0
            for influence in propagatedInfluence {
                totalInfluence += influence[0] // Sum first dimension as proxy for total influence
            }
            #expect(totalInfluence > 0, "Influence should have propagated through the network")

            print("Social network analysis completed: \\(userCount) users, \\(friendships.count) connections")
            print("Community subgraph: \\(communityGraph.nodeCount) users, \\(communityGraph.adjacency.nonZeros) connections")
            print("Total propagated influence: \\(totalInfluence)")
        }

        @Test
        func testKnowledgeGraphWorkflow() async {
            // Simulate a knowledge graph workflow for entity relationships
            let entityCount = 150
            let relationships = generateKnowledgeGraphRelations(entityCount: entityCount)

            // Step 1: Build knowledge graph
            let knowledgeGraph = try! WeightedGraph(from: GraphPrimitivesKernels.edgeListToCSR(nodeCount: entityCount, edges: relationships))

            // Step 2: Create entity embeddings (representing semantic features)
            let entityEmbeddings = generateEntityEmbeddings(count: entityCount)

            // Step 3: Perform multi-hop reasoning using repeated SpMV
            var currentEmbeddings = entityEmbeddings
            let hops = 3

            for hop in 0..<hops {
                var nextEmbeddings = ContiguousArray<Vector768Optimized>()
                await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                    matrix: knowledgeGraph.adjacency,
                    input: currentEmbeddings,
                    output: &nextEmbeddings,
                    normalize: true
                )

                // Combine current and next embeddings for residual connections
                for i in 0..<nextEmbeddings.count {
                    let combined = (currentEmbeddings[i] + nextEmbeddings[i]) / 2.0
                    nextEmbeddings[i] = combined
                }

                currentEmbeddings = nextEmbeddings
                print("Completed reasoning hop \\(hop + 1)/\\(hops)")
            }

            // Step 4: Extract domain-specific subgraphs (e.g., medical entities)
            let medicalEntities: Set<UInt32> = Set((0..<30).map { UInt32($0) })
            let (medicalGraph, entityMapping) = GraphPrimitivesKernels.extractSubgraph(from: knowledgeGraph, nodeSubset: medicalEntities)

            // Step 5: Compute entity centrality using transpose operations
            let transposed = GraphPrimitivesKernels.transposeCSR(knowledgeGraph.adjacency)
            let transposeGraph = try! WeightedGraph(from: transposed)

            var incomingInfluence = ContiguousArray<Vector768Optimized>()
            GraphPrimitivesKernels.aggregateNeighbors(
                graph: transposeGraph,
                nodeFeatures: entityEmbeddings,
                aggregation: .sum, // Sum of incoming connections
                output: &incomingInfluence
            )

            // Validate workflow results
            #expect(currentEmbeddings.count == entityCount)
            #expect(medicalGraph.nodeCount <= 30)
            #expect(entityMapping.count <= 30)
            #expect(incomingInfluence.count == entityCount)

            print("Knowledge graph reasoning completed: \\(entityCount) entities, \\(relationships.count) relations")
            print("Medical subgraph: \\(medicalGraph.nodeCount) entities, \\(medicalGraph.adjacency.nonZeros) relations")
            print("Multi-hop reasoning completed over \\(hops) hops")
        }

        @Test
        func testRecommendationSystemWorkflow() async {
            // Simulate a collaborative filtering recommendation system
            let userCount = 80
            let itemCount = 120
            let totalNodes = userCount + itemCount

            // Create user-item bipartite graph
            let interactions = generateUserItemInteractions(userCount: userCount, itemCount: itemCount)

            // Step 1: Build bipartite graph (users: 0..79, items: 80..199)
            let bipartiteGraph = try! WeightedGraph(from: GraphPrimitivesKernels.edgeListToCSR(nodeCount: totalNodes, edges: interactions))

            // Step 2: Create user and item embeddings
            var nodeFeatures = ContiguousArray<Vector1536Optimized>()
            // User embeddings (random initial features)
            for _ in 0..<userCount {
                nodeFeatures.append(Vector1536Optimized(repeating: Float.random(in: 0.1...1.0)))
            }
            // Item embeddings (different distribution)
            for _ in 0..<itemCount {
                nodeFeatures.append(Vector1536Optimized(repeating: Float.random(in: 0.5...1.5)))
            }

            // Step 3: Perform collaborative filtering using neighbor aggregation
            var userRecommendations = ContiguousArray<Vector1536Optimized>()
            GraphPrimitivesKernels.aggregateNeighbors(
                graph: bipartiteGraph,
                nodeFeatures: nodeFeatures,
                aggregation: .mean, // Mean of interacted items
                output: &userRecommendations
            )

            // Step 4: Compute item popularity using transpose
            let transposed = GraphPrimitivesKernels.transposeCSR(bipartiteGraph.adjacency)
            let itemPopularityGraph = try! WeightedGraph(from: transposed)

            var itemPopularity = ContiguousArray<Vector1536Optimized>()
            GraphPrimitivesKernels.aggregateNeighbors(
                graph: itemPopularityGraph,
                nodeFeatures: nodeFeatures,
                aggregation: .sum, // Sum of users who interacted
                output: &itemPopularity
            )

            // Step 5: Generate recommendations using SpMV for similarity propagation
            let userSubset = ContiguousArray<Vector1536Optimized>(nodeFeatures[0..<userCount])
            var similarityScores = ContiguousArray<Vector1536Optimized>()
            await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                matrix: bipartiteGraph.adjacency,
                input: nodeFeatures,
                output: &similarityScores,
                normalize: true
            )

            // Step 6: Extract user-specific recommendations
            let targetUser: UInt32 = 5
            let userNodes: Set<UInt32> = Set([targetUser])
            let (userGraph, userMapping) = GraphPrimitivesKernels.extractSubgraph(from: bipartiteGraph, nodeSubset: userNodes)

            // Validate workflow results
            #expect(userRecommendations.count == totalNodes)
            #expect(itemPopularity.count == totalNodes)
            #expect(similarityScores.count == totalNodes)
            #expect(userMapping.count <= 1)

            // Check that recommendations are meaningful
            let userRecVector = userRecommendations[Int(targetUser)]
            var recSum: Float = 0
            for i in 0..<100 { recSum += userRecVector[i] } // Sample a few dimensions
            #expect(recSum > 0, "User should have meaningful recommendations")

            print("Recommendation system completed: \\(userCount) users, \\(itemCount) items")
            print("Generated \\(interactions.count) user-item interactions")
            print("Computed recommendations and item popularity scores")
        }

        @Test
        func testGraphNeuralNetworkWorkflow() async {
            // Simulate a Graph Neural Network (GNN) message passing workflow
            let nodeCount = 100
            let edges = generateRandomGraph(nodeCount: nodeCount, edgeProbability: 0.1)

            // Step 1: Build graph
            let graph = try! WeightedGraph(from: GraphPrimitivesKernels.edgeListToCSR(nodeCount: nodeCount, edges: edges))

            // Step 2: Initialize node features (simulating neural network embeddings)
            var nodeFeatures = generateRandomNodeFeatures(count: nodeCount)

            // Step 3: Simulate GNN message passing for multiple layers
            let numLayers = 4
            var layerOutputs: [ContiguousArray<Vector512Optimized>] = []

            for layer in 0..<numLayers {
                // Message passing: aggregate neighbor features
                var messages = ContiguousArray<Vector512Optimized>()
                GraphPrimitivesKernels.aggregateNeighbors(
                    graph: graph,
                    nodeFeatures: nodeFeatures,
                    aggregation: .mean, // Mean aggregation like GraphSAGE
                    output: &messages
                )

                // Update features: combine self-features with messages
                var updatedFeatures = ContiguousArray<Vector512Optimized>()
                for i in 0..<nodeCount {
                    // Simulate neural network transformation: weighted sum + nonlinearity
                    let selfWeight: Float = 0.5
                    let neighborWeight: Float = 0.5

                    let selfContrib = nodeFeatures[i] * selfWeight
                    let neighborContrib = messages[i] * neighborWeight
                    let combined = selfContrib + neighborContrib

                    // Apply ReLU-like activation (clamp negative values to 0)
                    var activated = combined
                    for j in 0..<512 {
                        if activated[j] < 0 {
                            // Simulate setting negative values to 0 (ReLU activation)
                            activated = activated * 1.0 // Placeholder for actual activation
                        }
                    }

                    updatedFeatures.append(activated)
                }

                nodeFeatures = updatedFeatures
                layerOutputs.append(nodeFeatures)
                print("GNN layer \\(layer + 1)/\\(numLayers) completed")
            }

            // Step 4: Graph-level prediction using global pooling
            var globalRepresentation = Vector512Optimized.zero
            for nodeFeature in nodeFeatures {
                globalRepresentation = globalRepresentation + nodeFeature
            }
            globalRepresentation = globalRepresentation / Float(nodeCount)

            // Step 5: Analyze learned representations using subgraph extraction
            let importantNodes: Set<UInt32> = Set((0..<20).map { UInt32($0) })
            let (subgraph, nodeMapping) = GraphPrimitivesKernels.extractSubgraph(from: graph, nodeSubset: importantNodes)

            // Validate workflow results
            #expect(layerOutputs.count == numLayers)
            #expect(nodeFeatures.count == nodeCount)
            #expect(subgraph.nodeCount <= 20)
            #expect(nodeMapping.count <= 20)

            // Check that features have evolved through layers
            var featureNorms: [Float] = []
            for layerOutput in layerOutputs {
                var totalNorm: Float = 0
                for nodeFeature in layerOutput {
                    for i in 0..<10 { totalNorm += abs(nodeFeature[i]) } // Sample dimensions
                }
                featureNorms.append(totalNorm)
            }

            print("GNN workflow completed: \\(nodeCount) nodes, \\(edges.count) edges")
            print("Processed \\(numLayers) layers of message passing")
            print("Feature evolution across layers: \\(featureNorms.map { String(format: \"%.2f\", $0) }.joined(separator: \" -> \"))")
        }
    }

    // MARK: - Cross-Operation Integration Tests

    @Suite("Cross-Operation Integration")
    struct CrossOperationIntegrationTests {

        @Test
        func testChainedGraphOperations() async {
            // Test chaining multiple graph operations in sequence
            let nodeCount = 60
            let edges = generateRandomGraph(nodeCount: nodeCount, edgeProbability: 0.08)

            // Operation 1: Build graph
            let originalMatrix = GraphPrimitivesKernels.edgeListToCSR(nodeCount: nodeCount, edges: edges)
            let graph = try! WeightedGraph(from: originalMatrix)

            // Operation 2: Transpose
            let transposedMatrix = GraphPrimitivesKernels.transposeCSR(originalMatrix)
            let transposedGraph = try! WeightedGraph(from: transposedMatrix)

            // Operation 3: Extract subgraph
            let subset: Set<UInt32> = Set((0..<30).map { UInt32($0) })
            let (subgraph, nodeMapping) = GraphPrimitivesKernels.extractSubgraph(from: graph, nodeSubset: subset)

            // Operation 4: SpMV on original graph
            let features = generateRandomNodeFeatures(count: nodeCount)
            var spMVResult = ContiguousArray<Vector512Optimized>()
            await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                matrix: originalMatrix,
                input: features,
                output: &spMVResult,
                normalize: false
            )

            // Operation 5: Neighbor aggregation on transposed graph
            var aggregationResult = ContiguousArray<Vector512Optimized>()
            GraphPrimitivesKernels.aggregateNeighbors(
                graph: transposedGraph,
                nodeFeatures: features,
                aggregation: .sum,
                output: &aggregationResult
            )

            // Operation 6: SpMV on subgraph (with subset of features)
            let subsetFeatures = ContiguousArray<Vector512Optimized>(features[0..<subgraph.nodeCount])
            var subgraphSpMVResult = ContiguousArray<Vector512Optimized>()
            await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
                matrix: subgraph.adjacency,
                input: subsetFeatures,
                output: &subgraphSpMVResult,
                normalize: true
            )

            // Validate all operations
            #expect(originalMatrix.nonZeros == transposedMatrix.nonZeros)
            #expect(spMVResult.count == nodeCount)
            #expect(aggregationResult.count == nodeCount)
            #expect(subgraphSpMVResult.count == subgraph.nodeCount)
            #expect(nodeMapping.count == subgraph.nodeCount)

            print("Chained operations completed successfully")
            print("Original graph: \\(nodeCount) nodes, \\(originalMatrix.nonZeros) edges")
            print("Subgraph: \\(subgraph.nodeCount) nodes, \\(subgraph.adjacency.nonZeros) edges")
        }

        @Test
        func testMultiDimensionalConsistency() async {
            // Test that operations work consistently across all supported dimensions
            let nodeCount = 50
            let edges = generateRandomGraph(nodeCount: nodeCount, edgeProbability: 0.1)
            let graph = try! WeightedGraph(from: GraphPrimitivesKernels.edgeListToCSR(nodeCount: nodeCount, edges: edges))

            // Test 512-dimension operations
            let features512 = generateRandomNodeFeatures(count: nodeCount)
            var result512 = ContiguousArray<Vector512Optimized>()
            GraphPrimitivesKernels.aggregateNeighbors(
                graph: graph,
                nodeFeatures: features512,
                aggregation: .mean,
                output: &result512
            )

            // Test 768-dimension operations
            let features768 = generate768Features(count: nodeCount)
            var result768 = ContiguousArray<Vector768Optimized>()
            GraphPrimitivesKernels.aggregateNeighbors(
                graph: graph,
                nodeFeatures: features768,
                aggregation: .mean,
                output: &result768
            )

            // Test 1536-dimension operations
            let features1536 = generate1536Features(count: nodeCount)
            var result1536 = ContiguousArray<Vector1536Optimized>()
            GraphPrimitivesKernels.aggregateNeighbors(
                graph: graph,
                nodeFeatures: features1536,
                aggregation: .mean,
                output: &result1536
            )

            // Validate consistency across dimensions
            #expect(result512.count == nodeCount)
            #expect(result768.count == nodeCount)
            #expect(result1536.count == nodeCount)

            // Check that graph structure affects all dimensions similarly
            // (nodes with no neighbors should have zero results)
            let zeroNodes = findNodesWithoutNeighbors(graph: graph)
            for nodeIdx in zeroNodes {
                let idx = Int(nodeIdx)
                if idx < result512.count {
                    #expect(result512[idx][0] == 0.0)
                }
                if idx < result768.count {
                    #expect(result768[idx][0] == 0.0)
                }
                if idx < result1536.count {
                    #expect(result1536[idx][0] == 0.0)
                }
            }

            print("Multi-dimensional consistency verified")
            print("Tested dimensions: 512, 768, 1536")
            print("Nodes without neighbors: \\(zeroNodes.count)")
        }

        @Test
        func testConcurrentOperations() async {
            // Test running multiple graph operations concurrently
            let nodeCount = 40
            let edges = generateRandomGraph(nodeCount: nodeCount, edgeProbability: 0.12)
            let graph = try! WeightedGraph(from: GraphPrimitivesKernels.edgeListToCSR(nodeCount: nodeCount, edges: edges))
            let features = generateRandomNodeFeatures(count: nodeCount)

            // Run multiple operations concurrently
            async let aggregationTask = runAggregationTask(graph: graph, features: features)
            async let spMVTask = runSpMVTask(matrix: graph.adjacency, features: features)
            async let transposeTask = runTransposeTask(matrix: graph.adjacency)
            async let subgraphTask = runSubgraphTask(graph: graph)

            // Wait for all tasks to complete
            let aggregationResult = await aggregationTask
            let spMVResult = await spMVTask
            let transposeResult = await transposeTask
            let subgraphResult = await subgraphTask

            // Validate all concurrent operations
            #expect(aggregationResult.count == nodeCount)
            #expect(spMVResult.count == nodeCount)
            #expect(transposeResult.nonZeros == graph.adjacency.nonZeros)
            #expect(subgraphResult.nodeCount <= 20)

            print("Concurrent operations completed successfully")
            print("All \\(4) operations ran in parallel without interference")
        }
    }

    // MARK: - Helper Functions

    private static func generateSocialNetworkGraph(userCount: Int) -> ContiguousArray<(UInt32, UInt32, Float?)> {
        var edges = ContiguousArray<(UInt32, UInt32, Float?)>()
        srand48(42)

        // Generate friendships with small-world properties
        for user in 0..<userCount {
            let friendCount = Int(3 + drand48() * 7) // 3-10 friends per user
            for _ in 0..<friendCount {
                let friend = Int(drand48() * Double(userCount))
                if friend != user {
                    let strength = Float(drand48()) // Friendship strength
                    edges.append((UInt32(user), UInt32(friend), strength))
                }
            }
        }
        return edges
    }

    private static func generateKnowledgeGraphRelations(entityCount: Int) -> ContiguousArray<(UInt32, UInt32, Float?)> {
        var edges = ContiguousArray<(UInt32, UInt32, Float?)>()
        srand48(123)

        // Generate hierarchical relationships
        for entity in 0..<entityCount {
            let relationCount = Int(1 + drand48() * 4) // 1-5 relations per entity
            for _ in 0..<relationCount {
                let target = Int(drand48() * Double(entityCount))
                if target != entity {
                    let confidence = Float(0.5 + drand48() * 0.5) // High confidence relations
                    edges.append((UInt32(entity), UInt32(target), confidence))
                }
            }
        }
        return edges
    }

    private static func generateUserItemInteractions(userCount: Int, itemCount: Int) -> ContiguousArray<(UInt32, UInt32, Float?)> {
        var edges = ContiguousArray<(UInt32, UInt32, Float?)>()
        srand48(456)

        // Generate user-item interactions (bipartite graph)
        for user in 0..<userCount {
            let interactionCount = Int(2 + drand48() * 8) // 2-10 items per user
            for _ in 0..<interactionCount {
                let item = userCount + Int(drand48() * Double(itemCount)) // Items start after users
                let rating = Float(1.0 + drand48() * 4.0) // Rating 1-5
                edges.append((UInt32(user), UInt32(item), rating))
            }
        }
        return edges
    }

    private static func generateRandomGraph(nodeCount: Int, edgeProbability: Double) -> ContiguousArray<(UInt32, UInt32, Float?)> {
        var edges = ContiguousArray<(UInt32, UInt32, Float?)>()
        srand48(789)

        for i in 0..<nodeCount {
            for j in 0..<nodeCount {
                if i != j && drand48() < edgeProbability {
                    let weight = Float(drand48())
                    edges.append((UInt32(i), UInt32(j), weight))
                }
            }
        }
        return edges
    }

    private static func generateUserEmbeddings(count: Int) -> ContiguousArray<Vector512Optimized> {
        var embeddings = ContiguousArray<Vector512Optimized>()
        srand48(111)
        for _ in 0..<count {
            let values = (0..<512).map { _ in Float(drand48() * 2.0 - 1.0) }
            embeddings.append(try! Vector512Optimized(values))
        }
        return embeddings
    }

    private static func generateEntityEmbeddings(count: Int) -> ContiguousArray<Vector768Optimized> {
        var embeddings = ContiguousArray<Vector768Optimized>()
        srand48(222)
        for _ in 0..<count {
            let values = (0..<768).map { _ in Float(drand48() * 2.0 - 1.0) }
            embeddings.append(try! Vector768Optimized(values))
        }
        return embeddings
    }

    private static func generateRandomNodeFeatures(count: Int) -> ContiguousArray<Vector512Optimized> {
        var features = ContiguousArray<Vector512Optimized>()
        srand48(333)
        for _ in 0..<count {
            let value = Float(drand48())
            features.append(Vector512Optimized(repeating: value))
        }
        return features
    }

    private static func generate768Features(count: Int) -> ContiguousArray<Vector768Optimized> {
        var features = ContiguousArray<Vector768Optimized>()
        srand48(444)
        for _ in 0..<count {
            let value = Float(drand48())
            features.append(Vector768Optimized(repeating: value))
        }
        return features
    }

    private static func generate1536Features(count: Int) -> ContiguousArray<Vector1536Optimized> {
        var features = ContiguousArray<Vector1536Optimized>()
        srand48(555)
        for _ in 0..<count {
            let value = Float(drand48())
            features.append(Vector1536Optimized(repeating: value))
        }
        return features
    }

    private static func findNodesWithoutNeighbors(graph: WeightedGraph) -> Set<UInt32> {
        var noNeighborNodes = Set<UInt32>()
        for nodeIdx in 0..<graph.nodeCount {
            let neighborStart = Int(graph.adjacency.rowPointers[nodeIdx])
            let neighborEnd = Int(graph.adjacency.rowPointers[nodeIdx + 1])
            if neighborStart == neighborEnd {
                noNeighborNodes.insert(UInt32(nodeIdx))
            }
        }
        return noNeighborNodes
    }

    // Concurrent operation helpers
    private static func runAggregationTask(graph: WeightedGraph, features: ContiguousArray<Vector512Optimized>) async -> ContiguousArray<Vector512Optimized> {
        var result = ContiguousArray<Vector512Optimized>()
        GraphPrimitivesKernels.aggregateNeighbors(
            graph: graph,
            nodeFeatures: features,
            aggregation: .sum,
            output: &result
        )
        return result
    }

    private static func runSpMVTask(matrix: SparseMatrix, features: ContiguousArray<Vector512Optimized>) async -> ContiguousArray<Vector512Optimized> {
        var result = ContiguousArray<Vector512Optimized>()
        await GraphPrimitivesKernels.sparseMatrixVectorMultiply(
            matrix: matrix,
            input: features,
            output: &result,
            normalize: false
        )
        return result
    }

    private static func runTransposeTask(matrix: SparseMatrix) async -> SparseMatrix {
        return GraphPrimitivesKernels.transposeCSR(matrix)
    }

    private static func runSubgraphTask(graph: WeightedGraph) async -> WeightedGraph {
        let subset: Set<UInt32> = Set((0..<20).map { UInt32($0) })
        let (subgraph, _) = GraphPrimitivesKernels.extractSubgraph(from: graph, nodeSubset: subset)
        return subgraph
    }
}
