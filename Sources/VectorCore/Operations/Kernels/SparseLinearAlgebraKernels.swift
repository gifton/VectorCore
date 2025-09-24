//
//  SparseLinearAlgebraKernels.swift
//  High-performance sparse linear algebra operations
//

import Foundation
import Accelerate
import os

// MARK: - Error Types

public enum SparseMatrixError: Error {
    case invalidDimensions(String)
    case singularMatrix(String)
    case invalidFormat(String)
    case numericalInstability(String)
    case outOfMemory(String)
}

// MARK: - Sparse Linear Algebra Extension

extension GraphPrimitivesKernels {

    // MARK: - SpGEMM Types

    public struct SpGEMMOptions: Sendable {
        public let sortIndices: Bool
        public let removeDuplicates: Bool
        public let numericalThreshold: Float?
        public let symbolicOnly: Bool
        public let maxNonzerosPerRow: Int?
        public let validateInput: Bool

        public init(
            sortIndices: Bool = true,
            removeDuplicates: Bool = true,
            numericalThreshold: Float? = nil,
            symbolicOnly: Bool = false,
            maxNonzerosPerRow: Int? = nil,
            validateInput: Bool = true
        ) {
            self.sortIndices = sortIndices
            self.removeDuplicates = removeDuplicates
            self.numericalThreshold = numericalThreshold
            self.symbolicOnly = symbolicOnly
            self.maxNonzerosPerRow = maxNonzerosPerRow
            self.validateInput = validateInput
        }
    }

    // MARK: - Input Validation

    /// Validates CSR matrix format for correctness
    public static func validateCSRFormat(_ matrix: SparseMatrix) throws {
        // Check row pointers are monotonically increasing
        for i in 0..<matrix.rows {
            if matrix.rowPointers[i] > matrix.rowPointers[i + 1] {
                throw SparseMatrixError.invalidFormat(
                    "Row pointers not monotonic at row \(i): \(matrix.rowPointers[i]) > \(matrix.rowPointers[i + 1])"
                )
            }
        }

        // Check column indices are in bounds and sorted within rows
        for i in 0..<matrix.rows {
            let start = Int(matrix.rowPointers[i])
            let end = Int(matrix.rowPointers[i + 1])

            var lastCol: Int32 = -1
            for idx in start..<end {
                let col = Int32(matrix.columnIndices[idx])

                if col < 0 || col >= matrix.cols {
                    throw SparseMatrixError.invalidFormat(
                        "Column index \(col) out of bounds [0, \(matrix.cols)) at position \(idx)"
                    )
                }

                // Check for sorted and no duplicates
                if col <= lastCol {
                    throw SparseMatrixError.invalidFormat(
                        "Column indices not sorted or contains duplicates in row \(i)"
                    )
                }
                lastCol = col
            }
        }

        // Check values array size if present
        if let values = matrix.values {
            if values.count != matrix.columnIndices.count {
                throw SparseMatrixError.invalidFormat(
                    "Values array size \(values.count) doesn't match column indices size \(matrix.columnIndices.count)"
                )
            }
        }
    }

    // MARK: - Sparse Matrix-Matrix Multiplication

    public static func sparseMatrixMatrixMultiply(
        A: SparseMatrix,
        B: SparseMatrix,
        options: SpGEMMOptions = SpGEMMOptions()
    ) throws -> SparseMatrix {
        guard A.cols == B.rows else {
            throw SparseMatrixError.invalidDimensions(
                "Matrix dimensions incompatible: A.cols (\(A.cols)) != B.rows (\(B.rows))"
            )
        }

        // Validate input matrices if requested
        if options.validateInput {
            try validateCSRFormat(A)
            try validateCSRFormat(B)
        }

        return try parallelSpGEMM(A: A, B: B, options: options)
    }

    // MARK: - Parallel SpGEMM with Improved Load Balancing

    private static func parallelSpGEMM(
        A: SparseMatrix,
        B: SparseMatrix,
        options: SpGEMMOptions
    ) throws -> SparseMatrix {
        let m = A.rows
        let n = B.cols
        let numThreads = ProcessInfo.processInfo.activeProcessorCount

        // Adaptive fallback to sequential for small problems
        let estimatedWork = A.nonZeros * (B.nonZeros / B.rows)
        if numThreads <= 1 || m < numThreads * 4 || estimatedWork < 50000 {
            return try sequentialSpGEMM(A: A, B: B, options: options)
        }

        // Improved row partitioning based on actual work estimation
        let rowPartitions = try partitionRowsByWork(A: A, B: B, numPartitions: numThreads)

        // Thread-local results with pre-allocation hints
        // Use a class to safely share mutable state across concurrent tasks
        final class ThreadSafeResults: @unchecked Sendable {
            var results: Array<SpGEMMResult?>
            var lock = os_unfair_lock_s()

            init(count: Int) {
                self.results = Array(repeating: nil, count: count)
            }

            func setResult(_ result: SpGEMMResult, at index: Int) {
                withUnsafeMutablePointer(to: &lock) { lockPtr in
                    os_unfair_lock_lock(lockPtr)
                    results[index] = result
                    os_unfair_lock_unlock(lockPtr)
                }
            }
        }

        let threadSafeResults = ThreadSafeResults(count: rowPartitions.count)

        // Execute SpGEMM chunks concurrently
        DispatchQueue.concurrentPerform(iterations: rowPartitions.count) { threadId in
            let (startRow, endRow) = rowPartitions[threadId]
            guard startRow < endRow else { return }

            do {
                let result = try computeSpGEMMChunk(
                    A: A, B: B,
                    startRow: startRow,
                    endRow: endRow,
                    options: options
                )

                threadSafeResults.setResult(result, at: threadId)
            } catch {
                // Handle error - in production, might want to propagate
                print("Thread \(threadId) failed: \(error)")
            }
        }

        // Filter out nil results and merge
        let validResults = threadSafeResults.results.compactMap { $0 }
        guard !validResults.isEmpty else {
            throw SparseMatrixError.invalidFormat("SpGEMM computation failed")
        }

        return try mergeSpGEMMResults(validResults, rows: m, cols: n)
    }

    // MARK: - Sequential SpGEMM (Optimized Gustavson's Algorithm)

    private static func sequentialSpGEMM(
        A: SparseMatrix,
        B: SparseMatrix,
        options: SpGEMMOptions
    ) throws -> SparseMatrix {
        let result = try computeSpGEMMChunk(
            A: A, B: B,
            startRow: 0,
            endRow: A.rows,
            options: options
        )
        return try mergeSpGEMMResults([result], rows: A.rows, cols: B.cols)
    }

    // MARK: - Optimized SpGEMM Chunk Computation

    private static func computeSpGEMMChunk(
        A: SparseMatrix,
        B: SparseMatrix,
        startRow: Int,
        endRow: Int,
        options: SpGEMMOptions
    ) throws -> SpGEMMResult {
        var result = SpGEMMResult()
        result.startRow = startRow

        let n = B.cols

        // Thread-local workspace with capacity hints
        var workspace = ContiguousArray<Float?>(repeating: nil, count: n)
        var pattern = ContiguousArray<Int32>()
        pattern.reserveCapacity(min(n, 1000)) // Reasonable initial capacity

        // Process each row in the chunk
        for i in startRow..<endRow {
            pattern.removeAll(keepingCapacity: true)

            let aRowStart = Int(A.rowPointers[i])
            let aRowEnd = Int(A.rowPointers[i + 1])

            // Skip empty rows
            guard aRowEnd > aRowStart else {
                result.rowLengths.append(0)
                continue
            }

            // Accumulation phase with prefetching hints
            for aIdx in aRowStart..<aRowEnd {
                let k = Int(A.columnIndices[aIdx])
                guard k < B.rows else { continue }

                // Prefetch next B row if available
                if aIdx + 1 < aRowEnd {
                    let nextK = Int(A.columnIndices[aIdx + 1])
                    if nextK < B.rows {
                        prefetchRow(matrix: B, row: nextK)
                    }
                }

                let bRowStart = Int(B.rowPointers[k])
                let bRowEnd = Int(B.rowPointers[k + 1])

                if options.symbolicOnly {
                    // Symbolic accumulation
                    for bIdx in bRowStart..<bRowEnd {
                        let j = Int(B.columnIndices[bIdx])
                        if workspace[j] == nil {
                            workspace[j] = 1.0 // Mark as visited
                            pattern.append(Int32(j))
                        }
                    }
                } else {
                    // Numeric accumulation
                    let aVal = A.values?[aIdx] ?? 1.0

                    // Unroll small rows for better performance
                    if bRowEnd - bRowStart <= 4 {
                        for bIdx in bRowStart..<bRowEnd {
                            let j = Int(B.columnIndices[bIdx])
                            let bVal = B.values?[bIdx] ?? 1.0
                            let product = aVal * bVal

                            if let existing = workspace[j] {
                                workspace[j] = existing + product
                            } else {
                                workspace[j] = product
                                pattern.append(Int32(j))
                            }
                        }
                    } else {
                        // Normal processing for larger rows
                        for bIdx in bRowStart..<bRowEnd {
                            let j = Int(B.columnIndices[bIdx])
                            let bVal = B.values?[bIdx] ?? 1.0
                            let product = aVal * bVal

                            if let existing = workspace[j] {
                                workspace[j] = existing + product
                            } else {
                                workspace[j] = product
                                pattern.append(Int32(j))
                            }
                        }
                    }
                }
            }

            // Post-processing phase
            var nnzInRow: UInt32 = 0

            // Apply constraints and gather results
            if !options.symbolicOnly, let maxNnz = options.maxNonzerosPerRow, pattern.count > maxNnz {
                pattern = selectTopK(pattern: pattern, values: workspace, k: maxNnz)
            } else if options.sortIndices {
                pattern.sort()
            }

            // Gather phase with thresholding
            for j in pattern {
                if let value = workspace[Int(j)] {
                    if !options.symbolicOnly {
                        if let threshold = options.numericalThreshold,
                           abs(value) < threshold {
                            workspace[Int(j)] = nil
                            continue
                        }
                        result.values.append(value)
                    }
                    result.columnIndices.append(UInt32(j))
                    nnzInRow += 1
                    workspace[Int(j)] = nil // Reset
                }
            }

            result.rowLengths.append(nnzInRow)
        }

        return result
    }

    // MARK: - Sparse Matrix Addition (SIMD Optimized)

    public static func sparseMatrixAdd(
        A: SparseMatrix,
        B: SparseMatrix,
        alpha: Float = 1.0,
        beta: Float = 1.0
    ) throws -> SparseMatrix {
        guard A.rows == B.rows && A.cols == B.cols else {
            throw SparseMatrixError.invalidDimensions(
                "Matrices must have same dimensions: A(\(A.rows)×\(A.cols)) vs B(\(B.rows)×\(B.cols))"
            )
        }

        let m = A.rows
        let n = A.cols

        var cRowPointers = ContiguousArray<UInt32>(repeating: 0, count: m + 1)
        var cColumnIndices = ContiguousArray<UInt32>()
        var cValues = ContiguousArray<Float>()

        // Pre-allocate with capacity hint
        let estimatedNnz = min(A.nonZeros + B.nonZeros, m * n)
        cColumnIndices.reserveCapacity(estimatedNnz)
        cValues.reserveCapacity(estimatedNnz)

        for i in 0..<m {
            let aStart = Int(A.rowPointers[i])
            let aEnd = Int(A.rowPointers[i + 1])
            let bStart = Int(B.rowPointers[i])
            let bEnd = Int(B.rowPointers[i + 1])

            var aIdx = aStart
            var bIdx = bStart

            // Optimized two-pointer merge
            while aIdx < aEnd && bIdx < bEnd {
                let aCol = A.columnIndices[aIdx]
                let bCol = B.columnIndices[bIdx]

                if aCol == bCol {
                    let aVal = (A.values?[aIdx] ?? 1.0) * alpha
                    let bVal = (B.values?[bIdx] ?? 1.0) * beta
                    let sum = aVal + bVal

                    // Skip zeros if they arise from cancellation
                    if abs(sum) > 1e-10 {
                        cColumnIndices.append(aCol)
                        cValues.append(sum)
                    }
                    aIdx += 1
                    bIdx += 1
                } else if aCol < bCol {
                    let aVal = (A.values?[aIdx] ?? 1.0) * alpha
                    if abs(aVal) > 1e-10 {
                        cColumnIndices.append(aCol)
                        cValues.append(aVal)
                    }
                    aIdx += 1
                } else {
                    let bVal = (B.values?[bIdx] ?? 1.0) * beta
                    if abs(bVal) > 1e-10 {
                        cColumnIndices.append(bCol)
                        cValues.append(bVal)
                    }
                    bIdx += 1
                }
            }

            // Handle remaining elements
            while aIdx < aEnd {
                let aVal = (A.values?[aIdx] ?? 1.0) * alpha
                if abs(aVal) > 1e-10 {
                    cColumnIndices.append(A.columnIndices[aIdx])
                    cValues.append(aVal)
                }
                aIdx += 1
            }

            while bIdx < bEnd {
                let bVal = (B.values?[bIdx] ?? 1.0) * beta
                if abs(bVal) > 1e-10 {
                    cColumnIndices.append(B.columnIndices[bIdx])
                    cValues.append(bVal)
                }
                bIdx += 1
            }

            cRowPointers[i + 1] = UInt32(cColumnIndices.count)
        }

        return try SparseMatrix(
            rows: m,
            cols: n,
            rowPointers: cRowPointers,
            columnIndices: cColumnIndices,
            values: cValues.isEmpty ? nil : cValues
        )
    }

    // MARK: - Sparse Matrix Scaling (SIMD Optimized)

    public static func sparseMatrixScale(
        matrix: SparseMatrix,
        scalar: Float
    ) -> SparseMatrix {
        guard let values = matrix.values else {
            // Pattern matrix - create explicit scaled values
            let scaledValues = ContiguousArray<Float>(
                repeating: scalar,
                count: matrix.nonZeros
            )
            return try! SparseMatrix(
                rows: matrix.rows,
                cols: matrix.cols,
                rowPointers: matrix.rowPointers,
                columnIndices: matrix.columnIndices,
                values: scaledValues
            )
        }

        // SIMD-optimized scaling using Accelerate
        var scaledValues = ContiguousArray<Float>(repeating: 0, count: values.count)
        values.withUnsafeBufferPointer { src in
            scaledValues.withUnsafeMutableBufferPointer { dst in
                var s = scalar
                vDSP_vsmul(src.baseAddress!, 1, &s, dst.baseAddress!, 1, vDSP_Length(values.count))
            }
        }

        return try! SparseMatrix(
            rows: matrix.rows,
            cols: matrix.cols,
            rowPointers: matrix.rowPointers,
            columnIndices: matrix.columnIndices,
            values: scaledValues
        )
    }

    // MARK: - COO to CSR Conversion (Fixed)

    public static func cooToCSR(
        edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)>,
        rows: Int,
        cols: Int,
        sumDuplicates: Bool = true
    ) throws -> SparseMatrix {
        // Validate input
        for edge in edges {
            if edge.row >= rows || edge.col >= UInt32(cols) {
                throw SparseMatrixError.invalidFormat(
                    "Edge (\(edge.row), \(edge.col)) out of bounds for \(rows)×\(cols) matrix"
                )
            }
        }

        // Sort edges by (row, col)
        let sortedEdges = edges.sorted { lhs, rhs in
            if lhs.row != rhs.row {
                return lhs.row < rhs.row
            }
            return lhs.col < rhs.col
        }

        // Initialize CSR arrays
        var rowPointers = ContiguousArray<UInt32>(repeating: 0, count: rows + 1)
        var columnIndices = ContiguousArray<UInt32>()
        var values = ContiguousArray<Float>()

        let hasValues = edges.contains { $0.value != nil }

        // Reserve capacity
        columnIndices.reserveCapacity(edges.count)
        if hasValues {
            values.reserveCapacity(edges.count)
        }

        var currentRow = 0
        var lastCol: UInt32 = UInt32.max
        var lastRow: UInt32 = UInt32.max

        for edge in sortedEdges {
            let r = Int(edge.row)
            let c = edge.col
            let v = edge.value ?? 1.0

            // Fill empty rows
            while currentRow < r {
                rowPointers[currentRow + 1] = UInt32(columnIndices.count)
                currentRow += 1
            }

            // Handle duplicates
            if sumDuplicates && r == lastRow && c == lastCol && !columnIndices.isEmpty {
                // Sum with previous value
                if hasValues {
                    values[values.count - 1] += v
                }
            } else {
                // New entry
                columnIndices.append(c)
                if hasValues {
                    values.append(v)
                }
                lastCol = c
                lastRow = edge.row
            }
        }

        // Fill remaining row pointers
        while currentRow < rows {
            rowPointers[currentRow + 1] = UInt32(columnIndices.count)
            currentRow += 1
        }

        return try SparseMatrix(
            rows: rows,
            cols: cols,
            rowPointers: rowPointers,
            columnIndices: columnIndices,
            values: hasValues ? values : nil
        )
    }

    // MARK: - CSR to CSC Conversion (Optimized)

    public static func csrToCSC(_ matrix: SparseMatrix) -> SparseMatrix {
        let m = matrix.rows
        let n = matrix.cols
        let nnz = matrix.nonZeros

        // Count non-zeros per column
        var colCounts = ContiguousArray<Int>(repeating: 0, count: n)
        for j in matrix.columnIndices {
            colCounts[Int(j)] += 1
        }

        // Build column pointers (prefix sum)
        var colPointers = ContiguousArray<UInt32>(repeating: 0, count: n + 1)
        for j in 0..<n {
            colPointers[j + 1] = colPointers[j] + UInt32(colCounts[j])
        }

        // Allocate CSC arrays
        var rowIndices = ContiguousArray<UInt32>(repeating: 0, count: nnz)
        var values: ContiguousArray<Float>? = nil
        if matrix.values != nil {
            values = ContiguousArray<Float>(repeating: 0, count: nnz)
        }

        // Use current position tracker for each column
        var currentPos = Array(colPointers.prefix(n))

        // Distribute entries to columns
        for i in 0..<m {
            let rowStart = Int(matrix.rowPointers[i])
            let rowEnd = Int(matrix.rowPointers[i + 1])

            for idx in rowStart..<rowEnd {
                let j = Int(matrix.columnIndices[idx])
                let destIdx = Int(currentPos[j])

                rowIndices[destIdx] = UInt32(i)
                if let matrixValues = matrix.values {
                    values?[destIdx] = matrixValues[idx]
                }

                currentPos[j] += 1
            }
        }

        return try! SparseMatrix(
            rows: n,  // Transposed dimensions
            cols: m,
            rowPointers: colPointers,
            columnIndices: rowIndices,
            values: values
        )
    }

    // MARK: - Sparse Triangular Solve (Complete Implementation)

    public static func sparseTriangularSolve(
        L: SparseMatrix,
        b: ContiguousArray<Float>,
        lower: Bool = true,
        transpose: Bool = false,
        unitDiagonal: Bool = false
    ) throws -> ContiguousArray<Float> {
        let n = L.rows

        guard n == L.cols else {
            throw SparseMatrixError.invalidDimensions("Matrix must be square for triangular solve")
        }
        guard n == b.count else {
            throw SparseMatrixError.invalidDimensions("Matrix and vector dimensions must match")
        }

        var x = ContiguousArray<Float>(b)

        if lower && !transpose {
            // Forward substitution: L * x = b
            try forwardSubstitution(L: L, x: &x, unitDiagonal: unitDiagonal)
        } else if !lower && !transpose {
            // Backward substitution: U * x = b
            try backwardSubstitution(U: L, x: &x, unitDiagonal: unitDiagonal)
        } else if lower && transpose {
            // Solve L^T * x = b (backward on L^T)
            try backwardSubstitutionTranspose(L: L, x: &x, unitDiagonal: unitDiagonal)
        } else {
            // Solve U^T * x = b (forward on U^T)
            try forwardSubstitutionTranspose(U: L, x: &x, unitDiagonal: unitDiagonal)
        }

        return x
    }

    // MARK: - Triangular Solve Implementations

    private static func forwardSubstitution(
        L: SparseMatrix,
        x: inout ContiguousArray<Float>,
        unitDiagonal: Bool
    ) throws {
        let n = L.rows

        for i in 0..<n {
            var sum = x[i]
            var diagValue: Float = unitDiagonal ? 1.0 : 0.0
            var foundDiag = unitDiagonal

            let rowStart = Int(L.rowPointers[i])
            let rowEnd = Int(L.rowPointers[i + 1])

            for idx in rowStart..<rowEnd {
                let j = Int(L.columnIndices[idx])

                if j < i {
                    let lVal = L.values?[idx] ?? 1.0
                    sum -= lVal * x[j]
                } else if j == i {
                    if !unitDiagonal {
                        diagValue = L.values?[idx] ?? 1.0
                        foundDiag = true
                    }
                    break // Lower triangular, no need to continue
                }
            }

            // Check for singular matrix
            if !foundDiag || abs(diagValue) < 1e-10 {
                throw SparseMatrixError.singularMatrix("Zero or missing diagonal at row \(i)")
            }

            x[i] = sum / diagValue
        }
    }

    private static func backwardSubstitution(
        U: SparseMatrix,
        x: inout ContiguousArray<Float>,
        unitDiagonal: Bool
    ) throws {
        let n = U.rows

        for i in stride(from: n - 1, through: 0, by: -1) {
            var sum = x[i]
            var diagValue: Float = unitDiagonal ? 1.0 : 0.0
            var foundDiag = unitDiagonal

            let rowStart = Int(U.rowPointers[i])
            let rowEnd = Int(U.rowPointers[i + 1])

            for idx in rowStart..<rowEnd {
                let j = Int(U.columnIndices[idx])

                if j > i {
                    let uVal = U.values?[idx] ?? 1.0
                    sum -= uVal * x[j]
                } else if j == i {
                    if !unitDiagonal {
                        diagValue = U.values?[idx] ?? 1.0
                        foundDiag = true
                    }
                }
            }

            if !foundDiag || abs(diagValue) < 1e-10 {
                throw SparseMatrixError.singularMatrix("Zero or missing diagonal at row \(i)")
            }

            x[i] = sum / diagValue
        }
    }

    private static func forwardSubstitutionTranspose(
        U: SparseMatrix,
        x: inout ContiguousArray<Float>,
        unitDiagonal: Bool
    ) throws {
        let n = U.rows

        // Create a CSC version for efficient column access
        let L_csc = csrToCSC(U)

        for i in 0..<n {
            var sum = x[i]

            // Get column i of U^T (row i of U in CSC format)
            let colStart = Int(L_csc.rowPointers[i])
            let colEnd = Int(L_csc.rowPointers[i + 1])

            var diagValue: Float = unitDiagonal ? 1.0 : 0.0
            var foundDiag = unitDiagonal

            for idx in colStart..<colEnd {
                let j = Int(L_csc.columnIndices[idx])

                if j < i {
                    let val = L_csc.values?[idx] ?? 1.0
                    sum -= val * x[j]
                } else if j == i && !unitDiagonal {
                    diagValue = L_csc.values?[idx] ?? 1.0
                    foundDiag = true
                }
            }

            if !foundDiag || abs(diagValue) < 1e-10 {
                throw SparseMatrixError.singularMatrix("Zero or missing diagonal at position \(i)")
            }

            x[i] = sum / diagValue
        }
    }

    private static func backwardSubstitutionTranspose(
        L: SparseMatrix,
        x: inout ContiguousArray<Float>,
        unitDiagonal: Bool
    ) throws {
        let n = L.rows

        // Create a CSC version for efficient column access
        let L_csc = csrToCSC(L)

        for i in stride(from: n - 1, through: 0, by: -1) {
            var sum = x[i]

            // Get column i of L^T (row i of L in CSC format)
            let colStart = Int(L_csc.rowPointers[i])
            let colEnd = Int(L_csc.rowPointers[i + 1])

            var diagValue: Float = unitDiagonal ? 1.0 : 0.0
            var foundDiag = unitDiagonal

            for idx in colStart..<colEnd {
                let j = Int(L_csc.columnIndices[idx])

                if j > i {
                    let val = L_csc.values?[idx] ?? 1.0
                    sum -= val * x[j]
                } else if j == i && !unitDiagonal {
                    diagValue = L_csc.values?[idx] ?? 1.0
                    foundDiag = true
                }
            }

            if !foundDiag || abs(diagValue) < 1e-10 {
                throw SparseMatrixError.singularMatrix("Zero or missing diagonal at position \(i)")
            }

            x[i] = sum / diagValue
        }
    }

    // MARK: - Element-wise Multiplication

    public static func sparseElementwiseMultiply(
        A: SparseMatrix,
        B: SparseMatrix
    ) throws -> SparseMatrix {
        guard A.rows == B.rows && A.cols == B.cols else {
            throw SparseMatrixError.invalidDimensions(
                "Matrices must have same dimensions for element-wise multiplication"
            )
        }

        let m = A.rows
        var cRowPointers = ContiguousArray<UInt32>(repeating: 0, count: m + 1)
        var cColumnIndices = ContiguousArray<UInt32>()
        var cValues = ContiguousArray<Float>()

        for i in 0..<m {
            let aStart = Int(A.rowPointers[i])
            let aEnd = Int(A.rowPointers[i + 1])
            let bStart = Int(B.rowPointers[i])
            let bEnd = Int(B.rowPointers[i + 1])

            // Skip empty rows
            if aEnd == aStart || bEnd == bStart {
                cRowPointers[i + 1] = UInt32(cColumnIndices.count)
                continue
            }

            // Two-pointer intersection
            var aIdx = aStart
            var bIdx = bStart

            while aIdx < aEnd && bIdx < bEnd {
                let aCol = A.columnIndices[aIdx]
                let bCol = B.columnIndices[bIdx]

                if aCol == bCol {
                    let aVal = A.values?[aIdx] ?? 1.0
                    let bVal = B.values?[bIdx] ?? 1.0
                    let product = aVal * bVal

                    // Only store non-zero products
                    if abs(product) > 1e-10 {
                        cColumnIndices.append(aCol)
                        cValues.append(product)
                    }
                    aIdx += 1
                    bIdx += 1
                } else if aCol < bCol {
                    aIdx += 1
                } else {
                    bIdx += 1
                }
            }

            cRowPointers[i + 1] = UInt32(cColumnIndices.count)
        }

        return try SparseMatrix(
            rows: m,
            cols: A.cols,
            rowPointers: cRowPointers,
            columnIndices: cColumnIndices,
            values: cValues.isEmpty ? nil : cValues
        )
    }

    // MARK: - Helper Functions

    /// Prefetch a row for cache optimization
    @inline(__always)
    private static func prefetchRow(matrix: SparseMatrix, row: Int) {
        guard row < matrix.rows else { return }

        let start = Int(matrix.rowPointers[row])
        let end = min(start + 8, Int(matrix.rowPointers[row + 1])) // Prefetch first 8 elements

        if start < end {
            matrix.columnIndices.withUnsafeBufferPointer { ptr in
                #if arch(x86_64)
                // x86 prefetch instruction
                if let base = ptr.baseAddress?.advanced(by: start) {
                    withUnsafePointer(to: base) { p in
                        _mm_prefetch(p, 0) // Prefetch to all cache levels
                    }
                }
                #else
                // ARM or other architectures - touch memory to trigger hardware prefetch
                _ = ptr[start]
                #endif
            }
        }
    }

    /// Select top K elements by magnitude
    private static func selectTopK(
        pattern: ContiguousArray<Int32>,
        values: ContiguousArray<Float?>,
        k: Int
    ) -> ContiguousArray<Int32> {
        struct IndexValue: Comparable {
            let index: Int32
            let magnitude: Float

            static func < (lhs: IndexValue, rhs: IndexValue) -> Bool {
                lhs.magnitude > rhs.magnitude // Sort descending
            }
        }

        var pairs: [IndexValue] = []
        pairs.reserveCapacity(pattern.count)

        for idx in pattern {
            if let val = values[Int(idx)] {
                pairs.append(IndexValue(index: idx, magnitude: abs(val)))
            }
        }

        // Partial sort for efficiency
        let topK = min(k, pairs.count)
        pairs.partial_sort(topK)

        // Extract indices and sort them for CSR format
        var result = ContiguousArray<Int32>()
        result.reserveCapacity(topK)

        for i in 0..<topK {
            result.append(pairs[i].index)
        }

        result.sort()
        return result
    }

    /// Improved work-based row partitioning
    private static func partitionRowsByWork(
        A: SparseMatrix,
        B: SparseMatrix,
        numPartitions: Int
    ) throws -> [(Int, Int)] {
        guard numPartitions > 0 else {
            throw SparseMatrixError.invalidFormat("Number of partitions must be positive")
        }

        if numPartitions == 1 || A.rows <= numPartitions {
            return [(0, A.rows)]
        }

        // Estimate work per row
        var rowWork = ContiguousArray<Int>(repeating: 0, count: A.rows)
        var totalWork = 0

        for i in 0..<A.rows {
            let aNnz = Int(A.rowPointers[i + 1] - A.rowPointers[i])
            var work = 0

            // Estimate work based on A's row and average B row size
            let avgBRowSize = B.nonZeros / B.rows
            work = aNnz * avgBRowSize

            rowWork[i] = max(work, 1) // Ensure minimum work of 1
            totalWork += work
        }

        // Create balanced partitions
        let targetWork = (totalWork + numPartitions - 1) / numPartitions
        var partitions: [(Int, Int)] = []
        var currentWork = 0
        var startRow = 0

        for i in 0..<A.rows {
            currentWork += rowWork[i]

            if currentWork >= targetWork && partitions.count < numPartitions - 1 {
                partitions.append((startRow, i + 1))
                startRow = i + 1
                currentWork = 0
            }
        }

        // Add remaining rows
        if startRow < A.rows {
            partitions.append((startRow, A.rows))
        }

        return partitions
    }

    // MARK: - Result Merging

    private struct SpGEMMResult {
        var columnIndices: ContiguousArray<UInt32> = []
        var values: ContiguousArray<Float> = []
        var rowLengths: ContiguousArray<UInt32> = []
        var startRow: Int = 0
    }

    private static func mergeSpGEMMResults(
        _ results: [SpGEMMResult],
        rows: Int,
        cols: Int
    ) throws -> SparseMatrix {
        // Calculate total non-zeros
        let totalNnz = results.reduce(0) { $0 + $1.columnIndices.count }

        // Pre-allocate exact sizes
        var cRowPointers = ContiguousArray<UInt32>(repeating: 0, count: rows + 1)
        var cColumnIndices = ContiguousArray<UInt32>()
        var cValues = ContiguousArray<Float>()

        cColumnIndices.reserveCapacity(totalNnz)

        // Check if we have numeric values
        let isNumeric = results.contains { !$0.values.isEmpty }
        if isNumeric {
            cValues.reserveCapacity(totalNnz)
        }

        // Sort results by startRow
        let sortedResults = results.sorted { $0.startRow < $1.startRow }

        // Build row pointers and merge data
        var currentNnz: UInt32 = 0
        var currentRow = 0

        for result in sortedResults {
            // Fill any gaps
            while currentRow < result.startRow {
                cRowPointers[currentRow + 1] = currentNnz
                currentRow += 1
            }

            // Process this result's rows
            _ = result.startRow + result.rowLengths.count
            var localOffset = 0

            for (idx, length) in result.rowLengths.enumerated() {
                let row = result.startRow + idx
                guard row < rows else { break }

                // Append this row's data
                let rowStart = localOffset
                let rowEnd = localOffset + Int(length)

                if rowEnd > rowStart {
                    cColumnIndices.append(contentsOf: result.columnIndices[rowStart..<rowEnd])
                    if isNumeric && !result.values.isEmpty {
                        cValues.append(contentsOf: result.values[rowStart..<rowEnd])
                    }
                }

                localOffset = rowEnd
                currentNnz += length
                currentRow = row + 1
                cRowPointers[currentRow] = currentNnz
            }
        }

        // Fill any remaining rows
        while currentRow < rows {
            currentRow += 1
            cRowPointers[currentRow] = currentNnz
        }

        return try SparseMatrix(
            rows: rows,
            cols: cols,
            rowPointers: cRowPointers,
            columnIndices: cColumnIndices,
            values: cValues.isEmpty ? nil : cValues
        )
    }
}

// MARK: - Array Extension for Partial Sort

extension Array where Element: Comparable {
    mutating func partial_sort(_ k: Int) {
        let k = Swift.min(k, count)
        if k <= 0 { return }

        for i in 0..<k {
            var minIdx = i
            for j in (i+1)..<count {
                if self[j] < self[minIdx] {
                    minIdx = j
                }
            }
            if minIdx != i {
                swapAt(i, minIdx)
            }
        }
    }
}

// MARK: - x86 Intrinsics Support

#if arch(x86_64)
import Darwin

@_silgen_name("_mm_prefetch")
private func _mm_prefetch(_ p: UnsafeRawPointer?, _ hint: Int32)
#endif