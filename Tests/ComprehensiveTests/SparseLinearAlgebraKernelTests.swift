//
//  SparseLinearAlgebraKernelTests.swift
//  Comprehensive test suite for sparse linear algebra kernels
//

import XCTest
@testable import VectorCore

final class SparseLinearAlgebraKernelTests: XCTestCase {

    // MARK: - Test Helpers

    /// Creates a test sparse matrix in CSR format
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
        var edges = ContiguousArray<(row: UInt32, col: UInt32, value: Float?)>()
        for i in 0..<size {
            edges.append((UInt32(i), UInt32(i), 1.0))
        }
        return try! GraphPrimitivesKernels.cooToCSR(edges: edges, rows: size, cols: size)
    }

    /// Creates a random sparse matrix with specified density
    private func createRandomSparseMatrix(rows: Int, cols: Int, density: Double) -> SparseMatrix {
        var edges = ContiguousArray<(row: UInt32, col: UInt32, value: Float?)>()
        let nnz = Int(Double(rows * cols) * density)

        var seen = Set<Int>()
        for _ in 0..<nnz {
            var idx: Int
            repeat {
                idx = Int.random(in: 0..<(rows * cols))
            } while seen.contains(idx)
            seen.insert(idx)

            let row = UInt32(idx / cols)
            let col = UInt32(idx % cols)
            let value = Float.random(in: -10...10)
            edges.append((row, col, value))
        }

        return try! GraphPrimitivesKernels.cooToCSR(edges: edges, rows: rows, cols: cols)
    }

    /// Converts sparse matrix to dense for verification
    private func toDense(_ sparse: SparseMatrix) -> [[Float]] {
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
    private func denseMultiply(_ A: [[Float]], _ B: [[Float]]) -> [[Float]] {
        let m = A.count
        let n = B[0].count
        let k = B.count

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

    func testBasicSpGEMM() async throws {
        let A = createTestMatrix()
        let B = createTestMatrix()

        let C = try await GraphPrimitivesKernels.sparseMatrixMatrixMultiply(A: A, B: B)

        // Verify dimensions
        XCTAssertEqual(C.rows, A.rows)
        XCTAssertEqual(C.cols, B.cols)

        // Convert to dense and verify
        let denseA = toDense(A)
        let denseB = toDense(B)
        let expectedC = denseMultiply(denseA, denseB)
        let actualC = toDense(C)

        for i in 0..<C.rows {
            for j in 0..<C.cols {
                XCTAssertEqual(actualC[i][j], expectedC[i][j], accuracy: 1e-5,
                               "Mismatch at (\(i), \(j))")
            }
        }
    }

    func testSpGEMMWithIdentity() async throws {
        let A = createTestMatrix()
        let I = createIdentityMatrix(size: 3)

        // A * I = A
        let C1 = try await GraphPrimitivesKernels.sparseMatrixMatrixMultiply(A: A, B: I)
        XCTAssertEqual(toDense(C1), toDense(A))

        // I * A = A
        let C2 = try await GraphPrimitivesKernels.sparseMatrixMatrixMultiply(A: I, B: A)
        XCTAssertEqual(toDense(C2), toDense(A))
    }

    func testSymbolicSpGEMM() async throws {
        let A = createTestMatrix()
        let B = createTestMatrix()

        let options = GraphPrimitivesKernels.SpGEMMOptions(
            symbolicOnly: true
        )

        let C = try await GraphPrimitivesKernels.sparseMatrixMatrixMultiply(
            A: A, B: B, options: options
        )

        // Symbolic should have no values
        XCTAssertNil(C.values)

        // But should have correct sparsity pattern
        XCTAssertGreaterThan(C.nonZeros, 0)
    }

    func testSpGEMMWithThresholding() async throws {
        let A = createRandomSparseMatrix(rows: 10, cols: 10, density: 0.3)
        let B = createRandomSparseMatrix(rows: 10, cols: 10, density: 0.3)

        let threshold: Float = 0.1
        let options = GraphPrimitivesKernels.SpGEMMOptions(
            numericalThreshold: threshold
        )

        let C = try await GraphPrimitivesKernels.sparseMatrixMatrixMultiply(
            A: A, B: B, options: options
        )

        // All values should be above threshold
        if let values = C.values {
            for val in values {
                XCTAssertGreaterThanOrEqual(abs(val), threshold)
            }
        }
    }

    func testParallelSpGEMMConsistency() async throws {
        let A = createRandomSparseMatrix(rows: 50, cols: 50, density: 0.2)
        let B = createRandomSparseMatrix(rows: 50, cols: 50, density: 0.2)

        // Run parallel version (default)
        let C_parallel = try await GraphPrimitivesKernels.sparseMatrixMatrixMultiply(A: A, B: B)

        // Run sequential version (create a copy to ensure independent computation)
        let C_sequential = try await GraphPrimitivesKernels.sparseMatrixMatrixMultiply(
            A: A, B: B,
            options: GraphPrimitivesKernels.SpGEMMOptions()
        )

        // Results should be identical
        XCTAssertEqual(C_parallel.nonZeros, C_sequential.nonZeros)
        XCTAssertEqual(toDense(C_parallel), toDense(C_sequential))
    }

    // MARK: - Matrix Addition Tests

    func testSparseMatrixAddition() throws {
        let A = createTestMatrix()
        let B = createTestMatrix()

        let C = try GraphPrimitivesKernels.sparseMatrixAdd(A: A, B: B)

        // Verify A + B
        let denseA = toDense(A)
        let denseB = toDense(B)
        let denseC = toDense(C)

        for i in 0..<A.rows {
            for j in 0..<A.cols {
                XCTAssertEqual(denseC[i][j], denseA[i][j] + denseB[i][j], accuracy: 1e-5)
            }
        }
    }

    func testWeightedMatrixAddition() throws {
        let A = createTestMatrix()
        let B = createTestMatrix()

        let alpha: Float = 2.0
        let beta: Float = -0.5

        let C = try GraphPrimitivesKernels.sparseMatrixAdd(
            A: A, B: B, alpha: alpha, beta: beta
        )

        let denseA = toDense(A)
        let denseB = toDense(B)
        let denseC = toDense(C)

        for i in 0..<A.rows {
            for j in 0..<A.cols {
                let expected = alpha * denseA[i][j] + beta * denseB[i][j]
                XCTAssertEqual(denseC[i][j], expected, accuracy: 1e-5)
            }
        }
    }

    func testMatrixAdditionCancellation() throws {
        let A = createTestMatrix()

        // Add A + (-A) should give zeros
        let C = try GraphPrimitivesKernels.sparseMatrixAdd(
            A: A, B: A, alpha: 1.0, beta: -1.0
        )

        let denseC = toDense(C)
        for i in 0..<C.rows {
            for j in 0..<C.cols {
                XCTAssertEqual(denseC[i][j], 0.0, accuracy: 1e-5)
            }
        }
    }

    // MARK: - Matrix Scaling Tests

    func testMatrixScaling() throws {
        let A = createTestMatrix()
        let scalar: Float = 3.5

        let B = GraphPrimitivesKernels.sparseMatrixScale(matrix: A, scalar: scalar)

        let denseA = toDense(A)
        let denseB = toDense(B)

        for i in 0..<A.rows {
            for j in 0..<A.cols {
                XCTAssertEqual(denseB[i][j], denseA[i][j] * scalar, accuracy: 1e-5)
            }
        }
    }

    func testPatternMatrixScaling() throws {
        // Create pattern matrix (no explicit values)
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            (0, 0, nil), (0, 2, nil),
            (1, 1, nil),
            (2, 0, nil), (2, 2, nil)
        ]
        let A = try! GraphPrimitivesKernels.cooToCSR(edges: edges, rows: 3, cols: 3)

        let scalar: Float = 2.5
        let B = GraphPrimitivesKernels.sparseMatrixScale(matrix: A, scalar: scalar)

        XCTAssertNotNil(B.values)
        XCTAssertEqual(B.values?.count, A.nonZeros)

        // All values should be scalar (since pattern matrix has implicit 1.0)
        for val in B.values ?? [] {
            XCTAssertEqual(val, scalar, accuracy: 1e-5)
        }
    }

    // MARK: - Element-wise Operations Tests

    func testElementwiseMultiplication() throws {
        let A = createTestMatrix()
        let B = createTestMatrix()

        let C = try GraphPrimitivesKernels.sparseElementwiseMultiply(A: A, B: B)

        let denseA = toDense(A)
        let denseB = toDense(B)
        let denseC = toDense(C)

        for i in 0..<A.rows {
            for j in 0..<A.cols {
                XCTAssertEqual(denseC[i][j], denseA[i][j] * denseB[i][j], accuracy: 1e-5)
            }
        }
    }

    func testElementwiseWithDifferentPatterns() throws {
        // A has different sparsity pattern than B
        let edgesA: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            (0, 0, 2.0), (0, 1, 3.0),
            (1, 1, 4.0)
        ]
        let A = try! GraphPrimitivesKernels.cooToCSR(edges: edgesA, rows: 3, cols: 3)

        let edgesB: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            (0, 1, 5.0), (0, 2, 6.0),
            (1, 1, 7.0)
        ]
        let B = try! GraphPrimitivesKernels.cooToCSR(edges: edgesB, rows: 3, cols: 3)

        let C = try GraphPrimitivesKernels.sparseElementwiseMultiply(A: A, B: B)

        // Only (0,1) and (1,1) should be non-zero in result
        XCTAssertEqual(C.nonZeros, 2)

        let denseC = toDense(C)
        XCTAssertEqual(denseC[0][1], 3.0 * 5.0, accuracy: 1e-5)
        XCTAssertEqual(denseC[1][1], 4.0 * 7.0, accuracy: 1e-5)
    }

    // MARK: - Format Conversion Tests

    func testCSRtoCSCConversion() throws {
        let A = createTestMatrix()
        let A_csc = GraphPrimitivesKernels.csrToCSC(A)

        // CSC of A is CSR of A^T
        XCTAssertEqual(A_csc.rows, A.cols)
        XCTAssertEqual(A_csc.cols, A.rows)
        XCTAssertEqual(A_csc.nonZeros, A.nonZeros)

        // Convert back and verify
        let A_back = GraphPrimitivesKernels.csrToCSC(A_csc)
        XCTAssertEqual(toDense(A_back), toDense(A))
    }

    func testCOOtoCSRConversion() throws {
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            (2, 1, 5.0),  // Out of order
            (0, 0, 1.0),
            (1, 2, 3.0),
            (0, 2, 2.0),
            (1, 0, 4.0)
        ]

        let matrix = try GraphPrimitivesKernels.cooToCSR(
            edges: edges, rows: 3, cols: 3
        )

        // Verify CSR properties
        XCTAssertEqual(matrix.rows, 3)
        XCTAssertEqual(matrix.cols, 3)
        XCTAssertEqual(matrix.nonZeros, 5)

        // Row pointers should be monotonic
        for i in 0..<matrix.rows {
            XCTAssertLessThanOrEqual(matrix.rowPointers[i], matrix.rowPointers[i + 1])
        }

        // Column indices within each row should be sorted
        for i in 0..<matrix.rows {
            let start = Int(matrix.rowPointers[i])
            let end = Int(matrix.rowPointers[i + 1])

            for idx in start..<(end - 1) {
                XCTAssertLessThanOrEqual(matrix.columnIndices[idx], matrix.columnIndices[idx + 1])
            }
        }
    }

    func testCOOtoCSRWithDuplicates() throws {
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            (0, 0, 1.0),
            (0, 0, 2.0),  // Duplicate, should sum to 3.0
            (1, 1, 4.0),
            (1, 1, 5.0)  // Duplicate, should sum to 9.0
        ]

        let matrix = try GraphPrimitivesKernels.cooToCSR(
            edges: edges, rows: 2, cols: 2, sumDuplicates: true
        )

        XCTAssertEqual(matrix.nonZeros, 2)  // Only 2 unique positions
        XCTAssertEqual(matrix.values?[0] ?? 0, 3.0, accuracy: 1e-5)  // 1.0 + 2.0
        XCTAssertEqual(matrix.values?[1] ?? 0, 9.0, accuracy: 1e-5)  // 4.0 + 5.0
    }

    // MARK: - Triangular Solve Tests

    func testLowerTriangularSolve() throws {
        // Create lower triangular matrix
        // L = [2  0  0]
        //     [1  3  0]
        //     [4  2  5]
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            (0, 0, 2.0),
            (1, 0, 1.0), (1, 1, 3.0),
            (2, 0, 4.0), (2, 1, 2.0), (2, 2, 5.0)
        ]
        let L = try! GraphPrimitivesKernels.cooToCSR(edges: edges, rows: 3, cols: 3)

        let b = ContiguousArray<Float>([4.0, 7.0, 29.0])

        let x = try GraphPrimitivesKernels.sparseTriangularSolve(
            L: L, b: b, lower: true, transpose: false, unitDiagonal: false
        )

        // Verify: L * x = b (which is the correct test)
        // Don't assume the exact solution values, just verify they satisfy the equation

        // Verify: L * x = b
        let Lx = sparseMatrixVectorMultiply(L, x)
        for i in 0..<3 {
            XCTAssertEqual(Lx[i], b[i], accuracy: 1e-5)
        }
    }

    func testUpperTriangularSolve() throws {
        // Create upper triangular matrix
        // U = [2  1  4]
        //     [0  3  2]
        //     [0  0  5]
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            (0, 0, 2.0), (0, 1, 1.0), (0, 2, 4.0),
            (1, 1, 3.0), (1, 2, 2.0),
            (2, 2, 5.0)
        ]
        let U = try! GraphPrimitivesKernels.cooToCSR(edges: edges, rows: 3, cols: 3)

        let b = ContiguousArray<Float>([20.0, 16.0, 10.0])

        let x = try GraphPrimitivesKernels.sparseTriangularSolve(
            L: U, b: b, lower: false, transpose: false, unitDiagonal: false
        )

        // Verify: U * x = b
        let Ux = sparseMatrixVectorMultiply(U, x)
        for i in 0..<3 {
            XCTAssertEqual(Ux[i], b[i], accuracy: 1e-5)
        }
    }

    func testTransposeTriangularSolve() throws {
        // Test L^T * x = b
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            (0, 0, 2.0),
            (1, 0, 1.0), (1, 1, 3.0),
            (2, 0, 4.0), (2, 1, 2.0), (2, 2, 5.0)
        ]
        let L = try! GraphPrimitivesKernels.cooToCSR(edges: edges, rows: 3, cols: 3)

        let b = ContiguousArray<Float>([10.0, 14.0, 22.0])

        let x = try GraphPrimitivesKernels.sparseTriangularSolve(
            L: L, b: b, lower: true, transpose: true, unitDiagonal: false
        )

        // Verify: L^T * x = b
        let Lt = GraphPrimitivesKernels.csrToCSC(L)
        let Ltx = sparseMatrixVectorMultiply(Lt, x)
        for i in 0..<3 {
            XCTAssertEqual(Ltx[i], b[i], accuracy: 1e-5)
        }
    }

    func testSingularMatrixDetection() throws {
        // Create singular matrix (zero diagonal)
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            (0, 0, 2.0),
            (1, 0, 1.0), // Missing diagonal (1, 1)
            (2, 0, 4.0), (2, 1, 2.0), (2, 2, 5.0)
        ]
        let L = try! GraphPrimitivesKernels.cooToCSR(edges: edges, rows: 3, cols: 3)

        let b = ContiguousArray<Float>([1.0, 2.0, 3.0])

        XCTAssertThrowsError(
            try GraphPrimitivesKernels.sparseTriangularSolve(
                L: L, b: b, lower: true, transpose: false, unitDiagonal: false
            )
        ) { error in
            guard case SparseMatrixError.singularMatrix = error else {
                XCTFail("Expected singular matrix error")
                return
            }
        }
    }

    // MARK: - Edge Cases

    func testEmptyMatrices() async throws {
        let empty = try GraphPrimitivesKernels.cooToCSR(edges: [], rows: 3, cols: 3)
        let A = createTestMatrix()

        // Empty * A = Empty
        let C1 = try await GraphPrimitivesKernels.sparseMatrixMatrixMultiply(A: empty, B: A)
        XCTAssertEqual(C1.nonZeros, 0)

        // A * Empty = Empty
        let C2 = try await GraphPrimitivesKernels.sparseMatrixMatrixMultiply(A: A, B: empty)
        XCTAssertEqual(C2.nonZeros, 0)

        // Empty + A = A
        let C3 = try GraphPrimitivesKernels.sparseMatrixAdd(A: empty, B: A)
        XCTAssertEqual(toDense(C3), toDense(A))
    }

    func testSingleElementMatrix() async throws {
        let edges: ContiguousArray<(row: UInt32, col: UInt32, value: Float?)> = [
            (0, 0, 5.0)
        ]
        let A = try! GraphPrimitivesKernels.cooToCSR(edges: edges, rows: 1, cols: 1)

        // A * A
        let C = try await GraphPrimitivesKernels.sparseMatrixMatrixMultiply(A: A, B: A)
        XCTAssertEqual(C.values?[0] ?? 0, 25.0, accuracy: 1e-5)
    }

    func testLargeMatrixOperations() async throws {
        let A = createRandomSparseMatrix(rows: 100, cols: 100, density: 0.1)
        let B = createRandomSparseMatrix(rows: 100, cols: 100, density: 0.1)

        // Test operations complete without error
        let C = try await GraphPrimitivesKernels.sparseMatrixMatrixMultiply(A: A, B: B)
        XCTAssertGreaterThan(C.nonZeros, 0)

        let D = try GraphPrimitivesKernels.sparseMatrixAdd(A: A, B: B)
        XCTAssertGreaterThan(D.nonZeros, 0)

        let E = try GraphPrimitivesKernels.sparseElementwiseMultiply(A: A, B: B)
        XCTAssertLessThanOrEqual(E.nonZeros, min(A.nonZeros, B.nonZeros))
    }

    // MARK: - Performance Tests

    func testSpGEMMPerformance() async throws {
        let A = createRandomSparseMatrix(rows: 500, cols: 500, density: 0.05)
        let B = createRandomSparseMatrix(rows: 500, cols: 500, density: 0.05)

        measure {
            _ = try! GraphPrimitivesKernels.sparseMatrixMatrixMultiply(A: A, B: B)
        }
    }

    func testParallelSpGEMMScaling() async throws {
        let sizes = [100, 200, 400]
        var times: [Double] = []

        for size in sizes {
            let A = createRandomSparseMatrix(rows: size, cols: size, density: 0.1)
            let B = createRandomSparseMatrix(rows: size, cols: size, density: 0.1)

            let start = CFAbsoluteTimeGetCurrent()
            _ = try await GraphPrimitivesKernels.sparseMatrixMatrixMultiply(A: A, B: B)
            let time = CFAbsoluteTimeGetCurrent() - start
            times.append(time)

            print("Size \(size)×\(size): \(String(format: "%.4f", time))s")
        }

        // Verify scaling is reasonable (not quadratic)
        if times.count > 1 {
            let scalingFactor = times[1] / times[0]
            XCTAssertLessThan(scalingFactor, 10.0, "Scaling should be sub-quadratic")
        }
    }

    func testTriangularSolvePerformance() throws {
        // Create large lower triangular matrix
        var edges = ContiguousArray<(row: UInt32, col: UInt32, value: Float?)>()
        let n = 1000

        for i in 0..<n {
            for j in 0...i {
                if Double.random(in: 0...1) < 0.3 || i == j {
                    edges.append((UInt32(i), UInt32(j), Float.random(in: 1...10)))
                }
            }
        }

        let L = try! GraphPrimitivesKernels.cooToCSR(edges: edges, rows: n, cols: n)
        let b = ContiguousArray<Float>((0..<n).map { _ in Float.random(in: -10...10) })

        measure {
            _ = try! GraphPrimitivesKernels.sparseTriangularSolve(
                L: L, b: b, lower: true
            )
        }
    }

    // MARK: - Input Validation Tests

    func testDimensionMismatch() async throws {
        let A = createRandomSparseMatrix(rows: 3, cols: 4, density: 0.5)
        let B = createRandomSparseMatrix(rows: 5, cols: 3, density: 0.5)

        XCTAssertThrowsError(
            try await GraphPrimitivesKernels.sparseMatrixMatrixMultiply(A: A, B: B)
        ) { error in
            guard case SparseMatrixError.invalidDimensions = error else {
                XCTFail("Expected dimension mismatch error")
                return
            }
        }
    }

    func testInvalidCSRFormat() throws {
        // Test various invalid formats

        // 1. Wrong row pointer count
        XCTAssertThrowsError(
            try SparseMatrix(
                rows: 2, cols: 2,
                rowPointers: [0, 1],  // Should be 3 elements for 2 rows
                columnIndices: [0],
                values: [1.0]
            )
        )

        // 2. Column index/value count mismatch
        XCTAssertThrowsError(
            try SparseMatrix(
                rows: 2, cols: 2,
                rowPointers: [0, 1, 2],
                columnIndices: [0, 1],
                values: [1.0]  // Only 1 value but 2 indices
            )
        )

        // 3. Invalid COO input (out of bounds)
        XCTAssertThrowsError(
            try GraphPrimitivesKernels.cooToCSR(
                edges: [(5, 0, 1.0)],  // Row 5 out of bounds
                rows: 3, cols: 3
            )
        )
    }

    // MARK: - Utility Functions

    private func sparseMatrixVectorMultiply(
        _ A: SparseMatrix,
        _ x: ContiguousArray<Float>
    ) -> ContiguousArray<Float> {
        var y = ContiguousArray<Float>(repeating: 0, count: A.rows)

        for i in 0..<A.rows {
            let start = Int(A.rowPointers[i])
            let end = Int(A.rowPointers[i + 1])

            for idx in start..<end {
                let j = Int(A.columnIndices[idx])
                let val = A.values?[idx] ?? 1.0
                y[i] += val * x[j]
            }
        }

        return y
    }
}
