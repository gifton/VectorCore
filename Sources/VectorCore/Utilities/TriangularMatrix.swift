// VectorCore: Triangular Matrix Storage
//
// Efficient storage for symmetric matrices (like distance matrices)
// storing only the upper triangle to save ~50% memory
//

import Foundation

/// Memory-efficient storage for symmetric matrices
///
/// `TriangularMatrix` stores only the upper triangle of a symmetric matrix,
/// reducing memory usage by approximately 50%. Perfect for distance matrices
/// where d(i,j) = d(j,i) and diagonal elements are zero.
///
/// ## Memory Savings
/// - Full matrix: n² elements
/// - Triangular: n*(n-1)/2 elements
/// - Savings: ~50% for large n
///
/// ## Example Usage
/// ```swift
/// var distances = TriangularMatrix<Float>(size: 1000)
/// distances[10, 20] = 5.0  // Automatically handles symmetry
/// let d1 = distances[10, 20]  // Returns 5.0
/// let d2 = distances[20, 10]  // Also returns 5.0
/// ```
public struct TriangularMatrix<T> {
    /// Internal storage for upper triangle elements
    @usableFromInline
    internal var storage: [T]
    
    /// Matrix dimension (n×n matrix)
    public let size: Int
    
    /// Number of elements actually stored
    public var elementCount: Int {
        size * (size - 1) / 2
    }
    
    /// Initialize with default value
    ///
    /// - Parameters:
    ///   - size: Matrix dimension
    ///   - defaultValue: Initial value for all elements
    public init(size: Int, defaultValue: T) {
        precondition(size >= 0, "Size must be non-negative")
        self.size = size
        let count = size * (size - 1) / 2
        self.storage = Array(repeating: defaultValue, count: count)
    }
    
    /// Initialize with zero (for numeric types)
    public init(size: Int) where T: Numeric {
        self.init(size: size, defaultValue: .zero)
    }
    
    /// Convert indices to storage position
    @inlinable
    internal func storageIndex(row: Int, col: Int) -> Int {
        // Ensure we're accessing upper triangle
        let (i, j) = row < col ? (row, col) : (col, row)
        // Formula for upper triangular: i * size - i * (i + 1) / 2 + j - i - 1
        return i * size - i * (i + 1) / 2 + j - i - 1
    }
}

// MARK: - Subscript Access

extension TriangularMatrix {
    /// Access matrix elements
    ///
    /// Automatically handles symmetry: m[i,j] == m[j,i]
    /// Returns zero for diagonal elements when T is Numeric
    @inlinable
    public subscript(row: Int, col: Int) -> T where T: Numeric {
        get {
            precondition(row >= 0 && row < size, "Row index out of bounds")
            precondition(col >= 0 && col < size, "Column index out of bounds")
            
            // Diagonal is always zero for distance matrices
            guard row != col else { return .zero }
            
            let index = storageIndex(row: row, col: col)
            return storage[index]
        }
        set {
            precondition(row >= 0 && row < size, "Row index out of bounds")
            precondition(col >= 0 && col < size, "Column index out of bounds")
            precondition(row != col, "Cannot set diagonal elements")
            
            let index = storageIndex(row: row, col: col)
            storage[index] = newValue
        }
    }
    
    /// Access matrix elements (non-numeric version)
    @inlinable
    public subscript(row: Int, col: Int) -> T? {
        get {
            precondition(row >= 0 && row < size, "Row index out of bounds")
            precondition(col >= 0 && col < size, "Column index out of bounds")
            
            // Diagonal returns nil for non-numeric types
            guard row != col else { return nil }
            
            let index = storageIndex(row: row, col: col)
            return storage[index]
        }
        set {
            precondition(row >= 0 && row < size, "Row index out of bounds")
            precondition(col >= 0 && col < size, "Column index out of bounds")
            
            if row != col, let value = newValue {
                let index = storageIndex(row: row, col: col)
                storage[index] = value
            }
        }
    }
}

// MARK: - Safe Access

extension TriangularMatrix {
    /// Safely access elements
    ///
    /// - Parameters:
    ///   - row: Row index
    ///   - col: Column index
    /// - Returns: Element value or nil if indices are out of bounds
    @inlinable
    public func at(_ row: Int, _ col: Int) -> T? where T: Numeric {
        guard row >= 0 && row < size else { return nil }
        guard col >= 0 && col < size else { return nil }
        guard row != col else { return .zero }
        
        let index = storageIndex(row: row, col: col)
        return storage[index]
    }
    
    /// Safely set element
    ///
    /// - Parameters:
    ///   - row: Row index
    ///   - col: Column index
    ///   - value: Value to set
    /// - Returns: true if successful, false if indices invalid
    @inlinable
    public mutating func setAt(_ row: Int, _ col: Int, to value: T) -> Bool {
        guard row >= 0 && row < size else { return false }
        guard col >= 0 && col < size else { return false }
        guard row != col else { return false }
        
        let index = storageIndex(row: row, col: col)
        storage[index] = value
        return true
    }
}

// MARK: - Iteration

extension TriangularMatrix: Sequence {
    /// Element iterator that yields (row, col, value) tuples
    public struct Iterator: IteratorProtocol {
        private let matrix: TriangularMatrix<T>
        private var row = 0
        private var col = 1
        
        init(_ matrix: TriangularMatrix<T>) {
            self.matrix = matrix
        }
        
        public mutating func next() -> (row: Int, col: Int, value: T)? {
            guard row < matrix.size - 1 else { return nil }
            
            let value = matrix.storage[matrix.storageIndex(row: row, col: col)]
            let result = (row, col, value)
            
            // Move to next position
            col += 1
            if col >= matrix.size {
                row += 1
                col = row + 1
            }
            
            return result
        }
    }
    
    public func makeIterator() -> Iterator {
        Iterator(self)
    }
}

// MARK: - Conversion

extension TriangularMatrix {
    /// Convert to full matrix representation
    ///
    /// - Returns: Full n×n matrix with symmetric values
    public func toFullMatrix() -> [[T]] where T: Numeric {
        var result = Array(repeating: Array(repeating: T.zero, count: size), count: size)
        
        for (i, j, value) in self {
            result[i][j] = value
            result[j][i] = value  // Symmetry
        }
        
        return result
    }
    
    /// Create from full matrix (takes upper triangle)
    ///
    /// - Parameter matrix: Full matrix (must be square)
    /// - Returns: Triangular matrix with upper triangle values
    public static func fromFullMatrix(_ matrix: [[T]]) -> TriangularMatrix<T>? {
        guard !matrix.isEmpty else { return nil }
        let size = matrix.count
        guard matrix.allSatisfy({ $0.count == size }) else { return nil }
        
        var result = TriangularMatrix(size: size, defaultValue: matrix[0][1])
        
        for i in 0..<size {
            for j in (i+1)..<size {
                result[i, j] = matrix[i][j]
            }
        }
        
        return result
    }
}

// MARK: - Parallel Operations

extension TriangularMatrix where T: Sendable {
    /// Fill matrix using parallel computation
    ///
    /// - Parameter compute: Function to compute element at (i,j)
    @available(macOS 13.0, iOS 16.0, tvOS 16.0, watchOS 9.0, *)
    public static func buildParallel(
        size: Int,
        compute: @Sendable (Int, Int) async -> T
    ) async -> TriangularMatrix<T> where T: Numeric {
        var result = TriangularMatrix<T>(size: size)
        
        await withTaskGroup(of: (Int, Int, T).self) { group in
            // Add tasks for each element
            for i in 0..<size {
                for j in (i+1)..<size {
                    group.addTask {
                        let value = await compute(i, j)
                        return (i, j, value)
                    }
                }
            }
            
            // Collect results
            for await (i, j, value) in group {
                result[i, j] = value
            }
        }
        
        return result
    }
}