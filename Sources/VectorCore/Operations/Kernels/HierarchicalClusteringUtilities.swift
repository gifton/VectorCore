// HierarchicalClusteringUtilities.swift
// VectorCore
//
// Advanced utility operations for hierarchical clustering trees
// Implements dendrogram manipulation, traversal, and validation

import Foundation
import simd

// MARK: - Hierarchical Tree Utility Operations Extension

extension HierarchicalTree {

    // MARK: - 1. Cut Dendrogram at Height

    /// Cut dendrogram at specified height to obtain flat clustering.
    ///
    /// Traverses the tree top-down (BFS). A node forms a cluster root if it is a leaf,
    /// or if the merge that formed it occurred at a distance less than or equal to the height.
    ///
    /// # Algorithm
    /// - Breadth-first traversal from root
    /// - Cut criterion: `mergeDistance <= height` OR `isLeaf`
    /// - Sequential cluster ID assignment
    ///
    /// # Complexity
    /// - Time: O(N) where N = number of nodes
    /// - Space: O(N) for queue and results
    ///
    /// # Parameters
    /// - `height`: Distance threshold for cutting (negative values treated as 0)
    ///
    /// # Returns
    /// Array of `(vectorIndex, clusterID)` pairs representing flat clustering
    ///
    /// # Example
    /// ```swift
    /// let tree = HierarchicalTree(...)
    /// let clusters = tree.cutAtHeight(0.5)
    /// // clusters = [(0, 0), (1, 0), (2, 1), ...]
    /// ```
    public func cutAtHeight(_ height: Float) -> [(vectorIndex: Int, clusterID: Int)] {
        guard let root = rootNode else { return [] }

        var results: [(vectorIndex: Int, clusterID: Int)] = []
        results.reserveCapacity(leafNodeIds.count)
        var clusterID = 0
        var queue: [ClusterNode] = [root]

        // Clamp negative heights to 0 (distances are non-negative)
        let effectiveHeight = max(0, height)

        while !queue.isEmpty {
            let node = queue.removeFirst()

            // Cluster formation criterion: merge preserved or leaf
            if node.isLeaf || node.mergeDistance <= effectiveHeight {
                // This node forms a cluster
                for vectorIndex in node.vectorIndices {
                    results.append((vectorIndex: vectorIndex, clusterID: clusterID))
                }
                clusterID += 1
            } else {
                // Merge broken - examine children
                queue.append(contentsOf: self.children(of: node.id))
            }
        }
        return results
    }

    // MARK: - 2. Cut Dendrogram to K Clusters

    /// Cut dendrogram to obtain exactly K clusters.
    ///
    /// # Algorithm
    /// Uses the property that in a dendrogram with N leaves, we need N-K merges
    /// to occur to result in K clusters. Finds the height of the (N-K)th merge
    /// and cuts at that height.
    ///
    /// # Complexity
    /// - Time: O(N log N) due to sorting merge distances
    /// - Space: O(N) for merge distance array
    ///
    /// # Parameters
    /// - `k`: Desired number of clusters (must satisfy 1 ≤ k ≤ N)
    ///
    /// # Returns
    /// Array of `(vectorIndex, clusterID)` pairs, or `nil` if k is invalid
    ///
    /// # Example
    /// ```swift
    /// let clusters = tree.cutAtClusterCount(5)  // Get exactly 5 clusters
    /// ```
    public func cutAtClusterCount(_ k: Int) -> [(vectorIndex: Int, clusterID: Int)]? {
        let N = leafNodeIds.count
        guard k >= 1 && k <= N else { return nil }

        // Edge case: single cluster
        if k == 1 {
            let maxHeight = rootNode?.mergeDistance ?? 0
            return cutAtHeight(maxHeight + 1.0)
        }

        // Edge case: each vector is its own cluster
        if k == N {
            return cutAtHeight(0.0)
        }

        // Collect and sort merge distances from internal nodes
        let mergeDistances = nodes.filter { !$0.isLeaf }.map { $0.mergeDistance }
        let sortedDistances = mergeDistances.sorted()

        // We need N-K merges to occur to get K clusters
        let mergesToAllow = N - k

        // The (N-K)th merge is at 0-based index (N-K)-1
        let targetMergeIndex = mergesToAllow - 1

        // Validate bounds (defensive check)
        guard targetMergeIndex >= 0 && targetMergeIndex < sortedDistances.count else {
            return nil
        }

        let cutHeight = sortedDistances[targetMergeIndex]

        // Cut at this height (inclusive) allows this merge and all preceding ones
        return cutAtHeight(cutHeight)
    }

    // MARK: - 3. Depth-First Traversal

    /// Traverse dendrogram in depth-first order (pre-order: root, left, right).
    ///
    /// # Algorithm
    /// Iterative DFS using explicit stack (safer than recursion for deep trees)
    ///
    /// # Complexity
    /// - Time: O(N)
    /// - Space: O(h) where h = tree height (stack depth)
    ///
    /// # Returns
    /// Array of nodes in DFS pre-order
    ///
    /// # Example
    /// ```swift
    /// let dfsOrder = tree.traverseDepthFirst()
    /// for node in dfsOrder {
    ///     print("Node \(node.id): \(node.vectorIndices.count) vectors")
    /// }
    /// ```
    public func traverseDepthFirst() -> [ClusterNode] {
        guard let root = rootNode else { return [] }

        var result: [ClusterNode] = []
        result.reserveCapacity(nodes.count)
        var stack: [ClusterNode] = [root]

        while let node = stack.popLast() {
            result.append(node)

            // Push children in reverse order (Right then Left)
            // so Left is processed first (LIFO)
            let children = self.children(of: node.id)
            for child in children.reversed() {
                stack.append(child)
            }
        }
        return result
    }

    // MARK: - 4. Breadth-First Traversal

    /// Traverse dendrogram in breadth-first order (level-order).
    ///
    /// # Complexity
    /// - Time: O(N)
    /// - Space: O(w) where w = maximum tree width (queue size)
    ///
    /// # Returns
    /// Array of nodes in BFS level-order
    ///
    /// # Example
    /// ```swift
    /// let bfsOrder = tree.traverseBreadthFirst()
    /// // Process nodes level by level
    /// ```
    public func traverseBreadthFirst() -> [ClusterNode] {
        guard let root = rootNode else { return [] }

        var result: [ClusterNode] = []
        result.reserveCapacity(nodes.count)
        var queue: [ClusterNode] = [root]

        while !queue.isEmpty {
            let node = queue.removeFirst()
            result.append(node)
            queue.append(contentsOf: self.children(of: node.id))
        }
        return result
    }

    // MARK: - 5. Get Labeled Clusters

    /// Get cluster labels for all vectors at specified cut height.
    ///
    /// Convenience wrapper around `cutAtHeight` that returns a dictionary
    /// mapping for O(1) vector → cluster lookup.
    ///
    /// # Complexity
    /// - Time: O(N)
    /// - Space: O(N) for dictionary
    ///
    /// # Parameters
    /// - `cutHeight`: Height at which to cut dendrogram
    ///
    /// # Returns
    /// Dictionary mapping `vectorIndex → clusterLabel`
    ///
    /// # Example
    /// ```swift
    /// let labels = tree.getLabeledClusters(cutHeight: 0.5)
    /// print(labels[0])  // Cluster ID for vector 0
    /// ```
    public func getLabeledClusters(cutHeight: Float) -> [Int: Int] {
        let results = cutAtHeight(cutHeight)
        var labels: [Int: Int] = [:]
        labels.reserveCapacity(results.count)

        for (vectorIndex, clusterID) in results {
            labels[vectorIndex] = clusterID
        }
        return labels
    }

    // MARK: - 6. Validate Hierarchy

    /// Validation result for dendrogram structure.
    public struct ValidationResult: Sendable, CustomStringConvertible {
        public let isValid: Bool
        public let errors: [String]

        public var description: String {
            if isValid {
                return "Dendrogram is valid"
            } else {
                return "Dendrogram has \(errors.count) error(s):\n" + errors.joined(separator: "\n")
            }
        }
    }

    /// Validate dendrogram structure integrity.
    ///
    /// # Validation Checks
    /// 1. **Parent-child consistency**: Child's parent pointer matches parent's ID
    /// 2. **Root properties**: Root has no parent
    /// 3. **Leaf properties**: Leaves have no children and height = 0
    /// 4. **Height consistency**: `parent.height == max(children.height) + 1`
    /// 5. **Merge distance monotonicity**: Parent distance ≥ children distances
    /// 6. **Vector indices partitioning**: Disjoint children, correct union in parent
    /// 7. **Cycle detection**: No circular references
    /// 8. **Reachability**: All nodes reachable from root
    /// 9. **Binary tree structure**: Internal nodes have exactly 2 children
    ///
    /// # Complexity
    /// - Time: O(N)
    /// - Space: O(N) for visited set
    ///
    /// # Returns
    /// `ValidationResult` with detailed error messages if invalid
    ///
    /// # Example
    /// ```swift
    /// let result = tree.validateHierarchy()
    /// if !result.isValid {
    ///     print(result.description)
    /// }
    /// ```
    public func validateHierarchy() -> ValidationResult {
        var errors: [String] = []

        guard let root = rootNode else {
            if nodes.isEmpty {
                return ValidationResult(isValid: true, errors: [])
            } else {
                errors.append("Tree has nodes but rootNode is invalid.")
                return ValidationResult(isValid: false, errors: errors)
            }
        }

        // Check: Root has no parent
        if root.parent != nil {
            errors.append("Root node (ID: \(root.id)) has a parent (ID: \(root.parent!)).")
        }

        var visitedVectorIndices = Set<Int>()
        var queue: [ClusterNode] = [root]
        var visitedNodeIds = Set<Int>()

        // BFS traversal for systematic validation
        while !queue.isEmpty {
            let node = queue.removeFirst()

            // Check for cycles
            if visitedNodeIds.contains(node.id) {
                errors.append("Cycle detected or node reused (ID: \(node.id)).")
                continue
            }
            visitedNodeIds.insert(node.id)

            if node.isLeaf {
                // Leaf node checks
                if node.leftChild != nil || node.rightChild != nil {
                    errors.append("Leaf node (ID: \(node.id)) has children.")
                }

                // Leaf height must be 0
                if node.height != 0 {
                    errors.append("Leaf node (ID: \(node.id)) height is \(node.height), expected 0.")
                }

                // Vector indices must be globally unique
                let intersection = visitedVectorIndices.intersection(node.vectorIndices)
                if !intersection.isEmpty {
                    errors.append("Vector indices overlap detected in Leaf Node (ID: \(node.id)). Indices: \(intersection).")
                }
                visitedVectorIndices.formUnion(node.vectorIndices)

            } else {
                // Internal node checks
                let children = self.children(of: node.id)

                // Binary tree structure
                if children.count != 2 {
                    errors.append("Internal node (ID: \(node.id)) must have 2 children. Found: \(children.count).")
                    continue
                }

                // Parent-child consistency
                for child in children {
                    if child.parent != node.id {
                        errors.append("Parent-child mismatch: Child (ID: \(child.id)) parent pointer is \(String(describing: child.parent)), expected \(node.id).")
                    }
                }

                // Height consistency: parent.height = max(children.height) + 1
                let maxChildHeight = children.map { $0.height }.max()!
                if node.height != maxChildHeight + 1 {
                    errors.append("Height inconsistency at Node (ID: \(node.id)). Height: \(node.height), Expected: \(maxChildHeight + 1).")
                }

                // Merge distance monotonicity
                for child in children {
                    if node.mergeDistance < child.mergeDistance {
                        errors.append("Merge distance monotonicity violated: Parent (ID: \(node.id), Dist: \(node.mergeDistance)) < Child (ID: \(child.id), Dist: \(child.mergeDistance)).")
                    }
                }

                // Vector indices partitioning
                let left = children[0]
                let right = children[1]

                if !left.vectorIndices.isDisjoint(with: right.vectorIndices) {
                    errors.append("Vector indices are not disjoint between children of Node (ID: \(node.id)).")
                }

                let expectedIndices = left.vectorIndices.union(right.vectorIndices)
                if node.vectorIndices != expectedIndices {
                    errors.append("Vector indices summation mismatch at Node (ID: \(node.id)).")
                }

                queue.append(contentsOf: children)
            }
        }

        // Check for unreachable nodes
        if visitedNodeIds.count != nodes.count {
            let unreachableCount = nodes.count - visitedNodeIds.count
            errors.append("Found \(unreachableCount) unreachable nodes.")
        }

        return ValidationResult(isValid: errors.isEmpty, errors: errors)
    }

    // MARK: - 7. Extract Subtree

    /// Extract a subtree rooted at specified node.
    ///
    /// Creates a new independent `HierarchicalTree` with remapped node IDs
    /// (sequential from 0). The extracted subtree is structurally isolated
    /// from the original tree.
    ///
    /// # Algorithm
    /// 1. DFS traversal to collect subtree nodes
    /// 2. Remap IDs sequentially (0, 1, 2, ...)
    /// 3. Update all internal pointers (left/right/parent)
    /// 4. Construct new tree with remapped nodes
    ///
    /// # Complexity
    /// - Time: O(S) where S = subtree size
    /// - Space: O(S) for new tree structure
    ///
    /// # Parameters
    /// - `rootId`: ID of node to use as new root
    ///
    /// # Returns
    /// New tree containing only the subtree, or `nil` if node not found
    ///
    /// # Example
    /// ```swift
    /// if let subtree = tree.extractSubtree(rootId: 42) {
    ///     print("Extracted subtree with \(subtree.nodes.count) nodes")
    /// }
    /// ```
    public func extractSubtree(rootId: Int) -> HierarchicalTree? {
        guard let newRoot = self.node(withId: rootId) else { return nil }

        // 1. Collect subtree nodes via DFS
        var subtreeNodes: [ClusterNode] = []
        var stack: [ClusterNode] = [newRoot]

        while let node = stack.popLast() {
            subtreeNodes.append(node)
            stack.append(contentsOf: self.children(of: node.id))
        }

        // 2. Remap IDs sequentially
        var idMapping: [Int: Int] = [:]
        for (newId, node) in subtreeNodes.enumerated() {
            idMapping[node.id] = newId
        }

        // 3. Create new nodes with updated IDs and pointers
        var remappedNodes = ContiguousArray<ClusterNode>()
        remappedNodes.reserveCapacity(subtreeNodes.count)

        for node in subtreeNodes {
            let newId = idMapping[node.id]!
            let newLeftChild = node.leftChild.flatMap { idMapping[$0] }
            let newRightChild = node.rightChild.flatMap { idMapping[$0] }

            // New root's parent is nil
            let newParent = (node.id == rootId) ? nil : node.parent.flatMap { idMapping[$0] }

            let newNode = ClusterNode(
                id: newId,
                vectorIndices: node.vectorIndices,
                centroid: node.centroid,
                radius: node.radius,
                leftChild: newLeftChild,
                rightChild: newRightChild,
                parent: newParent,
                height: node.height,
                mergeDistance: node.mergeDistance
            )
            remappedNodes.append(newNode)
        }

        // Sort by ID for consistency
        remappedNodes.sort { $0.id < $1.id }

        let newRootId = idMapping[rootId]!

        // Collect leaf node IDs from remapped nodes
        let newLeafNodeIds = Set(remappedNodes.filter { $0.isLeaf }.map { $0.id })

        // 4. Construct new tree
        let newTree = HierarchicalTree(
            nodes: remappedNodes,
            rootNodeId: newRootId,
            leafNodeIds: newLeafNodeIds,
            dimension: self.dimension,
            linkageCriterion: self.linkageCriterion,
            distanceMetric: self.distanceMetric
        )

        return newTree
    }

    // MARK: - 8. Merge Subtrees

    /// Merge multiple dendrograms (self + trees) into a single tree.
    ///
    /// Creates a binary merge hierarchy above the input trees using a cascade
    /// merge strategy (pairwise bottom-up merging).
    ///
    /// # Algorithm
    /// 1. Validate dimension compatibility
    /// 2. Remap all node IDs to avoid collisions
    /// 3. Cascade merge: repeatedly merge pairs until single root
    /// 4. Synthetic merge nodes created with weighted centroid averaging
    ///
    /// # Merge Node Properties
    /// - **Centroid**: Weighted average of children centroids
    /// - **Merge distance**: `max(children distances) × 1.1` (heuristic)
    /// - **Height**: `max(children heights) + 1`
    /// - **Radius**: Conservative upper bound from children
    ///
    /// # Complexity
    /// - Time: O(T×N) where T = number of trees, N = total nodes
    /// - Space: O(T×N) for merged structure
    ///
    /// # Parameters
    /// - `trees`: Array of trees to merge with self
    ///
    /// # Returns
    /// New tree with merged structure, or `nil` if incompatible or empty
    ///
    /// # Example
    /// ```swift
    /// let merged = tree1.mergeSubtrees([tree2, tree3, tree4])
    /// ```
    public func mergeSubtrees(_ trees: [HierarchicalTree]) -> HierarchicalTree? {
        var allTrees = [self]
        allTrees.append(contentsOf: trees)

        if allTrees.count <= 1 { return self }

        // 1. Validate dimension compatibility
        let dimension = self.dimension
        if trees.contains(where: { $0.dimension != dimension }) {
            return nil
        }

        var mergedNodes = ContiguousArray<ClusterNode>()
        var currentOffset = 0

        // Track root IDs to merge
        var currentMerges: [Int] = []

        // 2. Remap node IDs and copy nodes
        for tree in allTrees {
            guard let root = tree.rootNode else { continue }

            var idMapping: [Int: Int] = [:]
            for node in tree.nodes {
                idMapping[node.id] = node.id + currentOffset
            }

            currentMerges.append(idMapping[root.id]!)
            currentOffset += tree.nodes.count

            // Copy nodes with remapped IDs
            for node in tree.nodes {
                let newId = idMapping[node.id]!
                let newLeftChild = node.leftChild.flatMap { idMapping[$0] }
                let newRightChild = node.rightChild.flatMap { idMapping[$0] }
                let newParent = node.isRoot ? nil : node.parent.flatMap { idMapping[$0] }

                let newNode = ClusterNode(
                    id: newId, vectorIndices: node.vectorIndices,
                    centroid: node.centroid, radius: node.radius,
                    leftChild: newLeftChild, rightChild: newRightChild,
                    parent: newParent,
                    height: node.height, mergeDistance: node.mergeDistance
                )
                mergedNodes.append(newNode)
            }
        }

        if currentMerges.isEmpty { return nil }

        // 3. Create binary merge hierarchy (Cascade Merge)
        var nextNodeId = currentOffset

        // Build lookup dictionary for efficient node access
        var nodeMap: [Int: ClusterNode] = [:]
        for node in mergedNodes {
            nodeMap[node.id] = node
        }

        while currentMerges.count > 1 {
            var nextMerges: [Int] = []

            // Merge pairs sequentially
            for i in stride(from: 0, to: currentMerges.count, by: 2) {
                let leftId = currentMerges[i]
                let rightId = (i + 1 < currentMerges.count) ? currentMerges[i+1] : nil

                // ✅ FIX: Use dictionary lookup instead of array indexing
                guard let leftNode = nodeMap[leftId] else { continue }
                let rightNode = rightId.flatMap { nodeMap[$0] }

                // Create merge node
                let mergeId = nextNodeId
                nextNodeId += 1
                nextMerges.append(mergeId)

                let (mergedNode, updatedLeft, updatedRight) = createMergeNode(
                    id: mergeId,
                    left: leftNode,
                    right: rightNode
                )

                // ✅ FIX: Update dictionary instead of array
                nodeMap[mergedNode.id] = mergedNode
                nodeMap[leftId] = updatedLeft
                if let updatedR = updatedRight, let rId = rightId {
                    nodeMap[rId] = updatedR
                }
            }
            currentMerges = nextMerges
        }

        let finalRootId = currentMerges.first!

        // 4. Reconstruct sorted array from dictionary
        let finalNodesArray = ContiguousArray(nodeMap.values.sorted { $0.id < $1.id })

        // Collect leaf node IDs
        let finalLeafNodeIds = Set(finalNodesArray.filter { $0.isLeaf }.map { $0.id })

        // 5. Construct final tree
        let finalTree = HierarchicalTree(
            nodes: finalNodesArray,
            rootNodeId: finalRootId,
            leafNodeIds: finalLeafNodeIds,
            dimension: dimension,
            linkageCriterion: self.linkageCriterion,
            distanceMetric: self.distanceMetric
        )

        return finalTree
    }

    // MARK: - Helper: Create Merge Node

    /// Helper to create a binary merge node and update children's parent pointers.
    ///
    /// Handles immutability of `ClusterNode` by returning reconstructed instances.
    ///
    /// # Merge Node Properties
    /// - **Binary merge** (both children):
    ///   - Centroid: Weighted average by cluster size
    ///   - Radius: Conservative upper bound `max(left.radius, right.radius) × 1.1`
    ///   - Distance: `max(left.distance, right.distance) × 1.1`
    ///   - Height: `max(left.height, right.height) + 1`
    ///
    /// - **Unary promotion** (left child only):
    ///   - All properties inherited from left child
    ///   - Used for handling odd number of trees in cascade merge
    ///
    /// # Parameters
    /// - `id`: New merge node ID
    /// - `left`: Left child node
    /// - `right`: Right child node (optional for unary promotion)
    ///
    /// # Returns
    /// Tuple of `(mergedNode, updatedLeft, updatedRight?)`
    private func createMergeNode(
        id: Int,
        left: ClusterNode,
        right: ClusterNode?
    ) -> (merged: ClusterNode, updatedLeft: ClusterNode, updatedRight: ClusterNode?) {

        let vectorIndices: Set<Int>
        let centroid: Vector512Optimized
        let radius: Float
        let mergeDistance: Float
        let height: Int

        if let right = right {
            // Binary merge
            vectorIndices = left.vectorIndices.union(right.vectorIndices)

            // Weighted average centroid
            let totalSize = Float(left.size + right.size)
            let leftWeight = Float(left.size) / totalSize
            let rightWeight = Float(right.size) / totalSize

            centroid = left.centroid.scaled(by: leftWeight)
                .adding(right.centroid.scaled(by: rightWeight))

            // ✅ FIX: Conservative radius calculation
            // Upper bound: max distance from new centroid to any point in either cluster
            let leftCentroidDist = left.centroid.euclideanDistance(to: centroid)
            let rightCentroidDist = right.centroid.euclideanDistance(to: centroid)
            radius = max(
                left.radius + leftCentroidDist,
                right.radius + rightCentroidDist
            ) * 1.1  // 10% safety margin

            // Merge distance heuristic
            mergeDistance = max(left.mergeDistance, right.mergeDistance) * 1.1
            height = max(left.height, right.height) + 1
        } else {
            // Unary promotion (odd tree in cascade)
            vectorIndices = left.vectorIndices
            centroid = left.centroid
            radius = left.radius
            mergeDistance = left.mergeDistance
            height = left.height
        }

        let mergedNode = ClusterNode(
            id: id,
            vectorIndices: vectorIndices,
            centroid: centroid,
            radius: radius,
            leftChild: left.id,
            rightChild: right?.id,
            parent: nil,
            height: height,
            mergeDistance: mergeDistance
        )

        // Update children's parent pointers (reconstruct nodes)
        let updatedLeft = ClusterNode(
            id: left.id, vectorIndices: left.vectorIndices,
            centroid: left.centroid, radius: left.radius,
            leftChild: left.leftChild, rightChild: left.rightChild,
            parent: id,
            height: left.height, mergeDistance: left.mergeDistance
        )

        let updatedRight = right.map { r in
            ClusterNode(
                id: r.id, vectorIndices: r.vectorIndices,
                centroid: r.centroid, radius: r.radius,
                leftChild: r.leftChild, rightChild: r.rightChild,
                parent: id,
                height: r.height, mergeDistance: r.mergeDistance
            )
        }

        return (mergedNode, updatedLeft, updatedRight)
    }
}
