//
//  MinHeap.swift
//  VectorCore
//
//

import Foundation

// MARK: - Min Heap

/// A heap whose root precedes its other elements under a strict weak ordering.
/// A capacity retains the last elements in that ordering: `<` keeps the largest,
/// `>` keeps the smallest. See `HeapMembershipContractTests.genericMembershipMatrix`.
public struct MinHeap<Element> {
    @usableFromInline
    internal var elements: [Element]

    @usableFromInline
    internal let compare: (Element, Element) -> Bool

    @usableFromInline
    internal let capacity: Int?

    /// Number of elements in the heap
    public var count: Int { elements.count }

    /// Whether the heap is empty
    public var isEmpty: Bool { elements.isEmpty }

    /// The minimum element (root of heap)
    public var min: Element? { elements.first }

    // MARK: - Initialization

    /// Initialize an empty heap
    /// - Parameters:
    ///   - capacity: Nil for unbounded retention, otherwise nonnegative. Zero
    ///     stays empty. A limit does not eagerly allocate that many elements.
    ///   - compare: Comparison function (true if first element should be before second)
    public init(capacity: Int? = nil, compare: @escaping (Element, Element) -> Bool) {
        precondition(capacity.map { $0 >= 0 } ?? true, "Heap capacity must be nonnegative")
        self.elements = []
        self.capacity = capacity
        self.compare = compare

    }

    /// Initialize with the same retained membership as streaming insertion.
    /// Full-retention inputs use heapify; truncated inputs use bounded insertion.
    public init(
        from array: [Element],
        capacity: Int? = nil,
        compare: @escaping (Element, Element) -> Bool
    ) {
        self.init(capacity: capacity, compare: compare)
        guard capacity != 0 else { return }
        if let capacity, array.count > capacity {
            for element in array { insert(element) }
        } else {
            self.elements = array
            if count > 1 {
                for i in stride(from: count / 2 - 1, through: 0, by: -1) {
                    siftDown(from: i)
                }
            }
        }
    }

    // MARK: - Heap Operations

    /// Insert an element into the heap
    @discardableResult
    @inlinable
    public mutating func insert(_ element: Element) -> Bool {
        // If a capacity is set, retain up to that many of the latest-ordered items.
        if let capacity = capacity, count >= capacity {
            guard let root = min else { return false }
            // Root is the current extremum according to `compare` and sits at index 0.
            // Keep `element` only if it comes after the current root in the heap ordering.
            // This condition correctly handles both min-heap and max-heap configurations.
            if !compare(root, element) {
                return false
            }
            // Replace root and restore heap property
            elements[0] = element
            siftDown(from: 0)
            return true
        }

        // Normal insertion when under capacity or no capacity enforced
        elements.append(element)
        siftUp(from: count - 1)
        return true
    }

    /// Remove and return the minimum element
    @discardableResult
    public mutating func removeMin() -> Element? {
        guard !isEmpty else { return nil }

        if count == 1 {
            return elements.removeLast()
        }

        // Swap first and last, remove last, then sift down
        elements.swapAt(0, count - 1)
        let min = elements.removeLast()
        siftDown(from: 0)

        return min
    }

    /// Get sorted array of all elements
    public func sorted() -> [Element] {
        var heap = self
        var result: [Element] = []
        result.reserveCapacity(count)

        while let min = heap.removeMin() {
            result.append(min)
        }

        return result
    }

    // MARK: - Private Helpers

    /// Sift up from given index to maintain heap property
    @usableFromInline
    internal mutating func siftUp(from index: Int) {
        var child = index
        var parent = parentIndex(of: child)

        while child > 0 && compare(elements[child], elements[parent]) {
            elements.swapAt(child, parent)
            child = parent
            parent = parentIndex(of: child)
        }
    }

    /// Sift down from given index to maintain heap property
    @usableFromInline
    internal mutating func siftDown(from index: Int) {
        var parent = index

        while true {
            let left = leftChildIndex(of: parent)
            let right = rightChildIndex(of: parent)
            var candidate = parent

            // Find the smallest among parent, left child, and right child
            if left < count && compare(elements[left], elements[candidate]) {
                candidate = left
            }

            if right < count && compare(elements[right], elements[candidate]) {
                candidate = right
            }

            // If parent is smallest, we're done
            if candidate == parent {
                break
            }

            // Otherwise, swap and continue
            elements.swapAt(parent, candidate)
            parent = candidate
        }
    }

    /// Get parent index
    @inlinable
    internal func parentIndex(of index: Int) -> Int {
        (index - 1) / 2
    }

    /// Get left child index
    @inlinable
    internal func leftChildIndex(of index: Int) -> Int {
        2 * index + 1
    }

    /// Get right child index
    @inlinable
    internal func rightChildIndex(of index: Int) -> Int {
        2 * index + 2
    }
}

// MARK: - K-Nearest Specific Heap

/// Distance heap ordered numeric-before-NaN, lower distance first, then smaller
/// index for numeric and NaN ties. Duplicate entries are allowed.
/// See `HeapMembershipContractTests.nearestPermutationMatrix`.
public struct KNearestHeap {
    /// Internal storage using max-heap (inverted comparison)
    @usableFromInline
    internal var heap: MinHeap<(index: Int, distance: Float)>

    /// Maximum number of neighbors to keep
    public let k: Int

    /// Current number of neighbors
    public var count: Int { heap.count }

    /// Initialize k-nearest heap
    public init(k: Int) {
        precondition(k >= 0, "Heap capacity must be nonnegative")
        self.k = k
        // Reverse the shared best-first order so the worst retained item is root.
        self.heap = MinHeap(capacity: k) {
            TopKSelection.orderedAscending($1, $0, .smallerIndex)
        }
    }

    /// Try to insert a new neighbor
    @discardableResult
    @inlinable
    public mutating func insert(index: Int, distance: Float) -> Bool {
        heap.insert((index, distance))
    }

    /// Get sorted results (nearest first)
    public func getSorted() -> [(index: Int, distance: Float)] {
        // Max-heap returns largest to smallest, so we need to reverse for nearest first
        Array(heap.sorted().reversed())
    }
}

// MARK: - Equatable Conformance

extension MinHeap: Equatable where Element: Equatable {
    public static func == (lhs: MinHeap<Element>, rhs: MinHeap<Element>) -> Bool {
        lhs.elements == rhs.elements
    }
}

// MARK: - Collection Helpers

extension MinHeap: Sequence {
    public func makeIterator() -> Array<Element>.Iterator {
        elements.makeIterator()
    }
}

extension MinHeap: CustomStringConvertible {
    public var description: String {
        "MinHeap(\(elements))"
    }
}
