//
//  SparseLinearAlgebraTests.swift
//  Comprehensive test suite for sparse linear algebra kernels
//

import XCTest
@testable import VectorCore

final class SparseLinearAlgebraTests: XCTestCase {

    // MARK: - Test Helpers

    /// Creates a simple test matrix in CSR format
    private func createTestMatrix() -> SparseMatrix {
        // Matrix:
        // [2  0  1]
        // [0  3  0]
        // [4  0  5]
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            (0, 0, 2.0), (0, 2, 1.0),
            (1, 1, 3.0),
            (2, 0, 4.0), (2, 2, 5.0)
        ]
        return try! GraphPrimitivesKernels.cooToCSR(edges: edges, rows: 3, cols: 3)
    }

    /// Creates an identity matrix
    private func createIdentityMatrix(size: Int) -> SparseMatrix {
        var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []
        for i in 0..<size {
            edges.append((UInt32(i), UInt32(i), 1.0))
        }
        return try! GraphPrimitivesKernels.cooToCSR(edges: edges, rows: size, cols: size)
    }

    /// Creates a lower triangular matrix for testing triangular solve
    private func createLowerTriangular() -> SparseMatrix {
        // Matrix:
        // [2  0  0]
        // [1  3  0]
        // [4  2  5]
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            (0, 0, 2.0),
            (1, 0, 1.0), (1, 1, 3.0),
            (2, 0, 4.0), (2, 1, 2.0), (2, 2, 5.0)
        ]
        return try! GraphPrimitivesKernels.cooToCSR(edges: edges, rows: 3, cols: 3)
    }

    /// Creates an upper triangular matrix
    private func createUpperTriangular() -> SparseMatrix {
        // Matrix:
        // [2  1  4]
        // [0  3  2]
        // [0  0  5]
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            (0, 0, 2.0), (0, 1, 1.0), (0, 2, 4.0),
            (1, 1, 3.0), (1, 2, 2.0),
            (2, 2, 5.0)
        ]
        return try! GraphPrimitivesKernels.cooToCSR(edges: edges, rows: 3, cols: 3)
    }

    /// Converts sparse matrix to dense for comparison
    private func sparseToDense(_ sparse: SparseMatrix) -> [[Float]] {
        var dense = Array(repeating: Array(repeating: Float(0), count: sparse.cols), count: sparse.rows)

        for i in 0..<sparse.rows {
            let start = Int(sparse.rowPointers[i])
            let end = Int(sparse.rowPointers[i + 1])

            for idx in start..<end {
                let j = Int(sparse.columnIndices[idx])
                dense[i][j] = sparse.values?[idx] ?? 1.0
            }
        }

        return dense
    }

    /// Dense matrix multiplication for verification
    private func denseMatrixMultiply(_ A: [[Float]], _ B: [[Float]]) -> [[Float]] {
        let m = A.count
        let n = B[0].count
        let k = A[0].count

        var C = Array(repeating: Array(repeating: Float(0), count: n), count: m)

        for i in 0..<m {
            for j in 0..<n {
                for p in 0..<k {
                    C[i][j] += A[i][p] * B[p][j]
                }
            }
        }

        return C
    }

    // MARK: - SpGEMM Tests

    func testSpGEMMBasic() throws {
        let A = createTestMatrix()
        let B = createTestMatrix()

        let C = try GraphPrimitivesKernels.sparseMatrixMatrixMultiply(
            A: A, B: B,
            options: .init(validateInput: true)
        )

        // Convert to dense for verification
        let denseA = sparseToDense(A)
        let denseB = sparseToDense(B)
        let expectedC = denseMatrixMultiply(denseA, denseB)
        let actualC = sparseToDense(C)

        // Compare results
        for i in 0..<3 {
            for j in 0..<3 {
                XCTAssertEqual(actualC[i][j], expectedC[i][j], accuracy: 1e-5,
                             "Mismatch at (\(i), \(j))")
            }
        }
    }

    func testSpGEMMIdentity() throws {
        let A = createTestMatrix()
        let I = createIdentityMatrix(size: 3)

        // A * I should equal A
        let AI = try GraphPrimitivesKernels.sparseMatrixMatrixMultiply(A: A, B: I)
        XCTAssertEqual(sparseToDense(AI), sparseToDense(A))

        // I * A should equal A
        let IA = try GraphPrimitivesKernels.sparseMatrixMatrixMultiply(A: I, B: A)
        XCTAssertEqual(sparseToDense(IA), sparseToDense(A))
    }

    func testSpGEMMSymbolic() throws {
        let A = createTestMatrix()
        let B = createTestMatrix()

        let symbolic = try GraphPrimitivesKernels.sparseMatrixMatrixMultiply(
            A: A, B: B,
            options: .init(symbolicOnly: true)
        )

        // Symbolic result should have pattern but no values
        XCTAssertNil(symbolic.values)
        XCTAssertGreaterThan(symbolic.nonZeros, 0)

        // Pattern should match numeric result
        let numeric = try GraphPrimitivesKernels.sparseMatrixMatrixMultiply(A: A, B: B)
        XCTAssertEqual(symbolic.columnIndices, numeric.columnIndices)
    }

    func testSpGEMMWithThresholding() throws {
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            (0, 0, 0.001), (0, 1, 2.0),
            (1, 0, 3.0), (1, 1, 0.0001)
        ]
        let A = try GraphPrimitivesKernels.cooToCSR(edges: edges, rows: 2, cols: 2)

        let C = try GraphPrimitivesKernels.sparseMatrixMatrixMultiply(
            A: A, B: A,
            options: .init(numericalThreshold: 0.01)
        )

        // Small values should be filtered out
        let denseC = sparseToDense(C)
        for i in 0..<2 {
            for j in 0..<2 {
                if abs(denseC[i][j]) < 0.01 {
                    XCTAssertEqual(denseC[i][j], 0.0, "Small value not filtered")
                }
            }
        }
    }

    // MARK: - Matrix Addition Tests

    func testSparseMatrixAddition() throws {
        let A = createTestMatrix()
        let B = createTestMatrix()

        let C = try GraphPrimitivesKernels.sparseMatrixAdd(A: A, B: B, alpha: 2.0, beta: 3.0)

        let denseA = sparseToDense(A)
        let denseB = sparseToDense(B)
        let denseC = sparseToDense(C)

        // Verify C = 2*A + 3*B
        for i in 0..<3 {
            for j in 0..<3 {
                let expected = 2.0 * denseA[i][j] + 3.0 * denseB[i][j]
                XCTAssertEqual(denseC[i][j], expected, accuracy: 1e-5)
            }
        }
    }

    func testSparseMatrixAdditionCancellation() throws {
        let A = createTestMatrix()

        // Add A and -A should give zero matrix
        let zero = try GraphPrimitivesKernels.sparseMatrixAdd(
            A: A, B: A,
            alpha: 1.0, beta: -1.0
        )

        XCTAssertEqual(zero.nonZeros, 0, "Cancellation should result in zero matrix")
    }

    // MARK: - Scaling Tests

    func testSparseMatrixScaling() {
        let A = createTestMatrix()
        let scalar: Float = 2.5

        let scaled = GraphPrimitivesKernels.sparseMatrixScale(matrix: A, scalar: scalar)

        let denseA = sparseToDense(A)
        let denseScaled = sparseToDense(scaled)

        for i in 0..<3 {
            for j in 0..<3 {
                XCTAssertEqual(denseScaled[i][j], denseA[i][j] * scalar, accuracy: 1e-5)
            }
        }
    }

    func testPatternMatrixScaling() {
        // Create pattern matrix (no explicit values)
        let rowPtrs = ContiguousArray<UInt32>([0, 2, 3, 5])
        let colInds = ContiguousArray<UInt32>([0, 2, 1, 0, 2])

        let pattern = try! SparseMatrix(
            rows: 3, cols: 3,
            rowPointers: rowPtrs,
            columnIndices: colInds,
            values: nil
        )

        let scaled = GraphPrimitivesKernels.sparseMatrixScale(matrix: pattern, scalar: 3.0)

        XCTAssertNotNil(scaled.values)
        XCTAssertEqual(scaled.values?.count, pattern.nonZeros)

        // All values should be 3.0 (1.0 * 3.0)
        for val in scaled.values ?? [] {
            XCTAssertEqual(val, 3.0)
        }
    }

    // MARK: - Format Conversion Tests

    func testCOOtoCSR() throws {
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            (2, 1, 5.0),  // Out of order
            (0, 0, 1.0),
            (1, 2, 3.0),
            (0, 2, 2.0),
            (1, 1, 4.0),
            (2, 0, 6.0)
        ]

        let csr = try GraphPrimitivesKernels.cooToCSR(edges: edges, rows: 3, cols: 3)

        // Verify structure
        XCTAssertEqual(csr.rows, 3)
        XCTAssertEqual(csr.cols, 3)
        XCTAssertEqual(csr.nonZeros, 6)

        // Verify row pointers
        XCTAssertEqual(Array(csr.rowPointers), [0, 2, 4, 6])

        // Verify sorted column indices within rows
        XCTAssertEqual(Array(csr.columnIndices[0..<2]), [0, 2]) // Row 0
        XCTAssertEqual(Array(csr.columnIndices[2..<4]), [1, 2]) // Row 1
        XCTAssertEqual(Array(csr.columnIndices[4..<6]), [0, 1]) // Row 2
    }

    func testCOOtoCSRWithDuplicates() throws {
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            (0, 0, 1.0),
            (0, 0, 2.0),  // Duplicate - should sum to 3.0
            (1, 1, 4.0),
            (1, 1, 5.0),  // Duplicate - should sum to 9.0
        ]

        let csr = try GraphPrimitivesKernels.cooToCSR(
            edges: edges, rows: 2, cols: 2,
            sumDuplicates: true
        )

        XCTAssertEqual(csr.nonZeros, 2)
        XCTAssertEqual(csr.values?[0], 3.0)  // 1.0 + 2.0
        XCTAssertEqual(csr.values?[1], 9.0)  // 4.0 + 5.0
    }

    func testCSRtoCSC() {
        let A = createTestMatrix()
        let A_csc = GraphPrimitivesKernels.csrToCSC(A)

        // CSC of A is CSR of A^T
        XCTAssertEqual(A_csc.rows, A.cols)  // Dimensions transposed
        XCTAssertEqual(A_csc.cols, A.rows)
        XCTAssertEqual(A_csc.nonZeros, A.nonZeros)

        // Convert back and compare
        let A_back = GraphPrimitivesKernels.csrToCSC(A_csc)
        XCTAssertEqual(sparseToDense(A), sparseToDense(A_back))
    }

    // MARK: - Triangular Solve Tests

    func testLowerTriangularSolve() throws {
        let L = createLowerTriangular()
        let b = ContiguousArray<Float>([4.0, 8.0, 26.0])

        let x = try GraphPrimitivesKernels.sparseTriangularSolve(
            L: L, b: b,
            lower: true,
            transpose: false,
            unitDiagonal: false
        )

        // Verify L * x = b
        let Lx = try GraphPrimitivesKernels.sparseMatrixMatrixMultiply(
            A: L,
            B: SparseMatrix(
                rows: 3, cols: 1,
                rowPointers: [0, 1, 2, 3],
                columnIndices: [0, 0, 0],
                values: x
            )
        )

        // Extract result and compare
        for i in 0..<3 {
            let computed = Lx.values?[i] ?? 0
            XCTAssertEqual(computed, b[i], accuracy: 1e-5)
        }
    }

    func testUpperTriangularSolve() throws {
        let U = createUpperTriangular()
        let b = ContiguousArray<Float>([20.0, 16.0, 10.0])

        let x = try GraphPrimitivesKernels.sparseTriangularSolve(
            L: U, b: b,
            lower: false,
            transpose: false,
            unitDiagonal: false
        )

        // Expected solution: x = [4, 4, 2]
        XCTAssertEqual(x[0], 4.0, accuracy: 1e-5)
        XCTAssertEqual(x[1], 4.0, accuracy: 1e-5)
        XCTAssertEqual(x[2], 2.0, accuracy: 1e-5)
    }

    func testTriangularSolveTranspose() throws {
        let L = createLowerTriangular()
        let b = ContiguousArray<Float>([10.0, 14.0, 22.0])

        // Solve L^T * x = b
        let x = try GraphPrimitivesKernels.sparseTriangularSolve(
            L: L, b: b,
            lower: true,
            transpose: true,
            unitDiagonal: false
        )

        // Verify by computing L^T * x
        let LT = GraphPrimitivesKernels.csrToCSC(L)
        let LTx = try GraphPrimitivesKernels.sparseMatrixMatrixMultiply(
            A: LT,
            B: SparseMatrix(
                rows: 3, cols: 1,
                rowPointers: [0, 1, 2, 3],
                columnIndices: [0, 0, 0],
                values: x
            )
        )

        for i in 0..<3 {
            let computed = LTx.values?[i] ?? 0
            XCTAssertEqual(computed, b[i], accuracy: 1e-4)
        }
    }

    func testSingularMatrixDetection() {
        // Create singular matrix (zero diagonal)
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            (0, 0, 0.0),  // Zero diagonal
            (1, 0, 1.0), (1, 1, 2.0),
            (2, 0, 3.0), (2, 1, 4.0), (2, 2, 5.0)
        ]
        let L = try! GraphPrimitivesKernels.cooToCSR(edges: edges, rows: 3, cols: 3)
        let b = ContiguousArray<Float>([1.0, 2.0, 3.0])

        XCTAssertThrowsError(
            try GraphPrimitivesKernels.sparseTriangularSolve(L: L, b: b, lower: true)
        ) { error in
            guard case SparseMatrixError.singularMatrix = error else {
                XCTFail("Expected singular matrix error")
                return
            }
        }
    }

    // MARK: - Element-wise Operations

    func testElementwiseMultiplication() throws {
        let A = createTestMatrix()
        let B = createTestMatrix()

        let C = try GraphPrimitivesKernels.sparseElementwiseMultiply(A: A, B: B)

        let denseA = sparseToDense(A)
        let denseB = sparseToDense(B)
        let denseC = sparseToDense(C)

        // Verify element-wise multiplication
        for i in 0..<3 {
            for j in 0..<3 {
                let expected = denseA[i][j] * denseB[i][j]
                XCTAssertEqual(denseC[i][j], expected, accuracy: 1e-5)
            }
        }
    }

    func testElementwiseWithDifferentPatterns() throws {
        let edgesA: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            (0, 0, 2.0), (0, 1, 3.0),
            (1, 1, 4.0)
        ]
        let edgesB: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            (0, 1, 5.0), (0, 2, 6.0),
            (1, 1, 7.0)
        ]

        let A = try GraphPrimitivesKernels.cooToCSR(edges: edgesA, rows: 2, cols: 3)
        let B = try GraphPrimitivesKernels.cooToCSR(edges: edgesB, rows: 2, cols: 3)

        let C = try GraphPrimitivesKernels.sparseElementwiseMultiply(A: A, B: B)

        // Only (0,1) and (1,1) should be non-zero
        XCTAssertEqual(C.nonZeros, 2)

        let denseC = sparseToDense(C)
        XCTAssertEqual(denseC[0][1], 3.0 * 5.0)  // 15.0
        XCTAssertEqual(denseC[1][1], 4.0 * 7.0)  // 28.0
    }

    // MARK: - Validation Tests

    func testInputValidation() {
        // Invalid row pointers - size mismatch
        XCTAssertThrowsError(
            try SparseMatrix(
                rows: 2, cols: 2,
                rowPointers: [0, 2, 1],  // Wrong size for 2 rows
                columnIndices: [0, 1],
                values: [1.0, 2.0],
                validate: true
            )
        ) { error in
            guard case GraphError.invalidCSRFormat = error else {
                XCTFail("Expected invalidCSRFormat error")
                return
            }
        }

        // Mismatch between row pointers and column indices count
        XCTAssertThrowsError(
            try SparseMatrix(
                rows: 2, cols: 2,
                rowPointers: [0, 1, 3],  // Says 3 total entries
                columnIndices: [0, 1],    // But only 2 entries
                values: [1.0, 2.0],
                validate: true
            )
        ) { error in
            guard case GraphError.invalidCSRFormat = error else {
                XCTFail("Expected invalidCSRFormat error")
                return
            }
        }

        // Values count mismatch
        XCTAssertThrowsError(
            try SparseMatrix(
                rows: 2, cols: 2,
                rowPointers: [0, 1, 2],
                columnIndices: [0, 1],
                values: [1.0],  // Only 1 value but 2 entries
                validate: true
            )
        ) { error in
            guard case GraphError.invalidCSRFormat = error else {
                XCTFail("Expected invalidCSRFormat error")
                return
            }
        }
    }

    // MARK: - Performance Tests

    func testLargeSpGEMMPerformance() throws {
        // Create larger sparse matrices for performance testing
        let n = 100
        var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []

        // Create a banded matrix
        for i in 0..<n {
            edges.append((UInt32(i), UInt32(i), Float.random(in: 1...10)))
            if i > 0 {
                edges.append((UInt32(i), UInt32(i-1), Float.random(in: 1...10)))
            }
            if i < n-1 {
                edges.append((UInt32(i), UInt32(i+1), Float.random(in: 1...10)))
            }
        }

        let A = try GraphPrimitivesKernels.cooToCSR(edges: edges, rows: n, cols: n)

        measure {
            _ = try! GraphPrimitivesKernels.sparseMatrixMatrixMultiply(A: A, B: A)
        }
    }

    func testTriangularSolvePerformance() throws {
        // Create larger triangular system
        let n = 100
        var edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = []

        // Create lower triangular matrix
        for i in 0..<n {
            for j in 0...i {
                if Float.random(in: 0...1) > 0.7 || i == j {  // Keep sparse but ensure diagonal
                    edges.append((UInt32(i), UInt32(j), Float.random(in: 1...10)))
                }
            }
        }

        let L = try GraphPrimitivesKernels.cooToCSR(edges: edges, rows: n, cols: n)
        let b = ContiguousArray<Float>((0..<n).map { _ in Float.random(in: 1...10) })

        measure {
            _ = try! GraphPrimitivesKernels.sparseTriangularSolve(L: L, b: b, lower: true)
        }
    }

    // MARK: - Edge Cases

    func testEmptyMatrices() throws {
        let empty = try SparseMatrix(rows: 3, cols: 3, edges: [])
        let A = createTestMatrix()

        // Empty * A = Empty
        let result1 = try GraphPrimitivesKernels.sparseMatrixMatrixMultiply(A: empty, B: A)
        XCTAssertEqual(result1.nonZeros, 0)

        // A * Empty = Empty
        let result2 = try GraphPrimitivesKernels.sparseMatrixMatrixMultiply(A: A, B: empty)
        XCTAssertEqual(result2.nonZeros, 0)
    }

    func testSingleElementMatrix() throws {
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [(0, 0, 5.0)]
        let single = try GraphPrimitivesKernels.cooToCSR(edges: edges, rows: 1, cols: 1)

        let squared = try GraphPrimitivesKernels.sparseMatrixMatrixMultiply(A: single, B: single)

        XCTAssertEqual(squared.nonZeros, 1)
        XCTAssertEqual(squared.values?[0], 25.0)  // 5 * 5
    }
}

// MARK: - Test Registration

extension SparseLinearAlgebraTests {
    static let allTests = [
        // SpGEMM
        ("testSpGEMMBasic", testSpGEMMBasic),
        ("testSpGEMMIdentity", testSpGEMMIdentity),
        ("testSpGEMMSymbolic", testSpGEMMSymbolic),
        ("testSpGEMMWithThresholding", testSpGEMMWithThresholding),

        // Addition
        ("testSparseMatrixAddition", testSparseMatrixAddition),
        ("testSparseMatrixAdditionCancellation", testSparseMatrixAdditionCancellation),

        // Scaling
        ("testSparseMatrixScaling", testSparseMatrixScaling),
        ("testPatternMatrixScaling", testPatternMatrixScaling),

        // Format Conversion
        ("testCOOtoCSR", testCOOtoCSR),
        ("testCOOtoCSRWithDuplicates", testCOOtoCSRWithDuplicates),
        ("testCSRtoCSC", testCSRtoCSC),

        // Triangular Solve
        ("testLowerTriangularSolve", testLowerTriangularSolve),
        ("testUpperTriangularSolve", testUpperTriangularSolve),
        ("testTriangularSolveTranspose", testTriangularSolveTranspose),
        ("testSingularMatrixDetection", testSingularMatrixDetection),

        // Element-wise
        ("testElementwiseMultiplication", testElementwiseMultiplication),
        ("testElementwiseWithDifferentPatterns", testElementwiseWithDifferentPatterns),

        // Validation
        ("testInputValidation", testInputValidation),

        // Performance
        ("testLargeSpGEMMPerformance", testLargeSpGEMMPerformance),
        ("testTriangularSolvePerformance", testTriangularSolvePerformance),

        // Edge Cases
        ("testEmptyMatrices", testEmptyMatrices),
        ("testSingleElementMatrix", testSingleElementMatrix)
    ]
}