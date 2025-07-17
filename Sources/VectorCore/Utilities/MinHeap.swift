// VectorCore: Min Heap Implementation
//
// Efficient priority queue for k-nearest neighbor algorithms
//

import Foundation

/// Efficient min-heap implementation for k-nearest neighbor selection
///
/// `MinHeap` provides O(log n) insertion and extraction, making it ideal
/// for maintaining the k smallest/largest elements from a stream of values.
///
/// ## Performance Characteristics
/// - Insert: O(log n)
/// - Extract min: O(log n)
/// - Peek min: O(1)
/// - Build heap: O(n)
///
/// ## Example Usage
/// ```swift
/// var heap = MinHeap<DistanceResult>()
/// for result in distances {
///     heap.insert(result)
///     if heap.count > k {
///         heap.extractMin()
///     }
/// }
/// ```
public struct MinHeap<T: Comparable> {
    /// Internal storage
    @usableFromInline
    internal var items: [T] = []
    
    /// Number of elements in heap
    @inlinable
    public var count: Int { items.count }
    
    /// Check if heap is empty
    @inlinable
    public var isEmpty: Bool { items.isEmpty }
    
    /// Peek at minimum element without removing
    @inlinable
    public var min: T? { items.first }
    
    /// Initialize empty heap
    public init() {}
    
    /// Initialize with capacity hint
    public init(capacity: Int) {
        items.reserveCapacity(capacity)
    }
    
    /// Initialize from array (builds heap in O(n))
    public init(_ array: [T]) {
        self.items = array
        buildHeap()
    }
}

// MARK: - Core Operations

extension MinHeap {
    /// Insert element into heap
    ///
    /// - Parameter item: Element to insert
    /// - Complexity: O(log n)
    @inlinable
    public mutating func insert(_ item: T) {
        items.append(item)
        siftUp(from: items.count - 1)
    }
    
    /// Extract minimum element
    ///
    /// - Returns: Minimum element or nil if empty
    /// - Complexity: O(log n)
    @inlinable
    @discardableResult
    public mutating func extractMin() -> T? {
        guard !items.isEmpty else { return nil }
        
        if items.count == 1 {
            return items.removeLast()
        }
        
        let min = items[0]
        items[0] = items.removeLast()
        siftDown(from: 0)
        
        return min
    }
    
    /// Remove all elements
    @inlinable
    public mutating func clear() {
        items.removeAll(keepingCapacity: true)
    }
}

// MARK: - Heap Maintenance

extension MinHeap {
    /// Restore heap property by moving element up
    @usableFromInline
    internal mutating func siftUp(from index: Int) {
        var child = index
        var parent = (child - 1) / 2
        
        while child > 0 && items[child] < items[parent] {
            items.swapAt(child, parent)
            child = parent
            parent = (child - 1) / 2
        }
    }
    
    /// Restore heap property by moving element down
    @usableFromInline
    internal mutating func siftDown(from index: Int) {
        var parent = index
        
        while true {
            let leftChild = 2 * parent + 1
            let rightChild = 2 * parent + 2
            var candidate = parent
            
            if leftChild < items.count && items[leftChild] < items[candidate] {
                candidate = leftChild
            }
            
            if rightChild < items.count && items[rightChild] < items[candidate] {
                candidate = rightChild
            }
            
            if candidate == parent {
                return
            }
            
            items.swapAt(parent, candidate)
            parent = candidate
        }
    }
    
    /// Build heap from unordered array in O(n)
    @usableFromInline
    internal mutating func buildHeap() {
        guard items.count > 1 else { return }
        
        // Start from last non-leaf node
        for i in stride(from: items.count / 2 - 1, through: 0, by: -1) {
            siftDown(from: i)
        }
    }
}

// MARK: - Max Heap Variant

/// Max heap implementation (reverses comparison)
public struct MaxHeap<T: Comparable> {
    private var minHeap: MinHeap<ReverseComparable<T>>
    
    public init() {
        minHeap = MinHeap<ReverseComparable<T>>()
    }
    
    public init(capacity: Int) {
        minHeap = MinHeap<ReverseComparable<T>>(capacity: capacity)
    }
    
    public var count: Int { minHeap.count }
    public var isEmpty: Bool { minHeap.isEmpty }
    public var max: T? { minHeap.min?.value }
    
    public mutating func insert(_ item: T) {
        minHeap.insert(ReverseComparable(value: item))
    }
    
    @discardableResult
    public mutating func extractMax() -> T? {
        minHeap.extractMin()?.value
    }
}

/// Wrapper to reverse comparison for max heap
@usableFromInline
internal struct ReverseComparable<T: Comparable>: Comparable {
    let value: T
    
    @inlinable
    static func < (lhs: ReverseComparable<T>, rhs: ReverseComparable<T>) -> Bool {
        lhs.value > rhs.value  // Reversed comparison
    }
}

// MARK: - K-Nearest Neighbor Support

/// Result type for nearest neighbor searches
public struct NeighborResult: Comparable {
    public let index: Int
    public let distance: Float
    
    @inlinable
    public init(index: Int, distance: Float) {
        self.index = index
        self.distance = distance
    }
    
    @inlinable
    public static func < (lhs: NeighborResult, rhs: NeighborResult) -> Bool {
        lhs.distance < rhs.distance
    }
}

extension MinHeap where T == NeighborResult {
    /// Specialized k-nearest neighbor heap operations
    
    /// Check if a distance would be accepted into k-nearest
    ///
    /// - Parameters:
    ///   - distance: Distance to check
    ///   - k: Maximum heap size
    /// - Returns: true if distance would be kept
    @inlinable
    public func wouldAccept(distance: Float, k: Int) -> Bool {
        count < k || distance < (min?.distance ?? Float.infinity)
    }
    
    /// Try to insert a neighbor, maintaining k elements
    ///
    /// - Parameters:
    ///   - neighbor: Neighbor to potentially insert
    ///   - k: Maximum heap size
    /// - Returns: true if neighbor was inserted
    @inlinable
    @discardableResult
    public mutating func insertIfBetter(_ neighbor: NeighborResult, k: Int) -> Bool {
        if count < k {
            insert(neighbor)
            return true
        } else if let currentMin = min, neighbor.distance < currentMin.distance {
            extractMin()
            insert(neighbor)
            return true
        }
        return false
    }
    
    /// Extract all results sorted by distance
    ///
    /// - Returns: Array of neighbors sorted by increasing distance
    public mutating func extractSorted() -> [NeighborResult] {
        var results: [NeighborResult] = []
        results.reserveCapacity(count)
        
        while let neighbor = extractMin() {
            results.append(neighbor)
        }
        
        return results
    }
}

// MARK: - Sequence Conformance

extension MinHeap: Sequence {
    public func makeIterator() -> AnyIterator<T> {
        var copy = self
        return AnyIterator {
            copy.extractMin()
        }
    }
}

// MARK: - Collection Support

extension MinHeap {
    /// Get sorted array of all elements (doesn't modify heap)
    public var sorted: [T] {
        var copy = self
        var result: [T] = []
        result.reserveCapacity(count)
        
        while let item = copy.extractMin() {
            result.append(item)
        }
        
        return result
    }
    
    /// Check if heap contains element
    public func contains(_ element: T) -> Bool {
        items.contains(element)
    }
}