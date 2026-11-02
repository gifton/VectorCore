import Foundation
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif
import Testing
@testable import VectorCore

private enum L01Oracle {
    // Double independent loops; Float output rounding allows O(d*ulp) residual.
    // One subnormal ULP covers final scalar rounding; the relative term covers
    // widened working error. Factor reconstruction has its own accumulation allowance.
    static func close(_ actual: Float, _ expected: Double) -> Bool {
        actual.isFinite && expected.isFinite &&
        abs(Double(actual) - expected) <= Double(Float.leastNonzeroMagnitude) + 2e-6 * abs(expected)
    }
    static func orthogonal(_ a: [Float], rows: Int, columns: Int) -> Bool {
        guard a.count == rows * columns, a.allSatisfy(\.isFinite) else { return false }
        for j in 0..<columns {
            for k in 0..<columns {
                var sum = 0.0
                for i in 0..<rows { sum += Double(a[j * rows + i]) * Double(a[k * rows + i]) }
                if !sum.isFinite || abs(sum - (j == k ? 1 : 0)) > 2e-6 { return false }
            }
        }
        return true
    }
    static func eigenResidual(_ a: [Float], values: [Float], vectors: [Float], n: Int) -> Bool {
        guard a.count == n*n, values.count == n, vectors.count == n*n,
              a.allSatisfy(\.isFinite), values.allSatisfy(\.isFinite), vectors.allSatisfy(\.isFinite) else { return false }
        let scale = a.map { abs(Double($0)) }.max() ?? 0
        let divisor = scale == 0 ? 1 : scale
        let allowance = 4e-6 + 8 * Double(Float.leastNonzeroMagnitude) / divisor
        for j in 0..<n {
            for i in 0..<n {
                var av = 0.0
                for k in 0..<n { av += (Double(a[k*n+i])/divisor) * Double(vectors[j*n+k]) }
                let vl = Double(vectors[j*n+i]) * (Double(values[j])/divisor)
                if !av.isFinite || !vl.isFinite || abs(av-vl) > allowance { return false }
            }
        }
        return true
    }
    static func transpose(_ a: [Float], rows: Int, columns: Int) -> [Float] {
        guard a.count == rows * columns else { return [] }
        return (0..<a.count).map { a[($0 % columns) * rows + $0 / columns] }
    }
    static func reconstruct(_ a: [Float], left: [Float], right: [Float], m: Int, k: Int, n: Int,
                            spectrum: [Float]? = nil) -> Bool {
        guard a.count == m*n, left.count == m*k, right.count == k*n,
              a.allSatisfy(\.isFinite), left.allSatisfy(\.isFinite), right.allSatisfy(\.isFinite),
              spectrum == nil || (spectrum!.count == k && spectrum!.allSatisfy(\.isFinite)) else { return false }
        let scale = a.map { abs(Double($0)) }.max() ?? 0
        let divisor = scale == 0 ? 1 : scale
        let allowance = 4e-6 + 16 * Double(Float.leastNonzeroMagnitude) / divisor
        for j in 0..<n {
            for i in 0..<m {
                var sum = 0.0
                for p in 0..<k {
                    let weight = spectrum.map { Double($0[p]) / divisor } ?? (1 / divisor)
                    sum += Double(left[p*m+i]) * weight * Double(right[j*k+p])
                }
                if !sum.isFinite || abs(sum - Double(a[j*m+i])/divisor) > allowance { return false }
            }
        }
        return true
    }
}

@Suite("L01 factorization range")
struct LinearAlgebraRangeTests {
    @Test func auditedQR() throws {
        for s: Float in [1e-25, 1e20] {
            let a = [s,s]
            let f = try SwiftLinearAlgebraProvider().qrThin(a, rows: 2, columns: 1)
            #expect(L01Oracle.orthogonal(f.q, rows: 2, columns: 1), "scale=\(s) Q=\(f.q)")
            #expect(f.r.count == 1 && L01Oracle.close(abs(f.r[0]), sqrt(2)*abs(Double(s))))
            #expect(L01Oracle.reconstruct(a, left: f.q, right: f.r, m: 2, k: 1, n: 1))
        }
    }
    @Test func auditedSingleColumnSVD() throws {
        for s: Float in [1e-25, 1e20] {
            let a = [s,s]
            let f = try SwiftLinearAlgebraProvider().svdThin(a, rows: 2, columns: 1)
            #expect(L01Oracle.orthogonal(f.u, rows: 2, columns: 1), "scale=\(s) U=\(f.u)")
            #expect(f.singularValues.count == 1 && L01Oracle.close(f.singularValues[0], sqrt(2)*abs(Double(s))))
            #expect(L01Oracle.reconstruct(a, left: f.u, right: f.vt, m: 2, k: 1, n: 1, spectrum: f.singularValues))
        }
    }
    @Test func auditedGoldenSVD() throws {
        for s: Float in [1e-12, 1e15] {
            let a: [Float] = [s,0,s,s]
            let f = try SwiftLinearAlgebraProvider().svdThin(a, rows: 2, columns: 2)
            #expect(L01Oracle.orthogonal(f.u, rows: 2, columns: 2), "scale=\(s) U=\(f.u)")
            #expect(L01Oracle.orthogonal(L01Oracle.transpose(f.vt, rows: 2, columns: 2), rows: 2, columns: 2))
            #expect(f.singularValues.count == 2)
            for (actual, factor) in zip(f.singularValues, [(1+sqrt(5))/2, (sqrt(5)-1)/2]) {
                #expect(L01Oracle.close(actual, factor * abs(Double(s))), "scale=\(s) spectrum=\(f.singularValues)")
            }
            #expect(L01Oracle.reconstruct(a, left: f.u, right: f.vt, m: 2, k: 2, n: 2, spectrum: f.singularValues))
        }
    }
    @Test func finiteEigenScale() throws {
        let s: Float = 1e20
        let e = try SwiftLinearAlgebraProvider().symmetricEigen([2*s,s,s,2*s], dimension: 2, computeEigenvectors: true)
        #expect(e.eigenvalues.count == 2 && L01Oracle.close(e.eigenvalues[0], Double(s)) && L01Oracle.close(e.eigenvalues[1], 3*Double(s)))
    }
}


extension LinearAlgebraRangeTests {
    @Test func sharedExponentSweep() throws {
        let provider = SwiftLinearAlgebraProvider()
        for exponent in [-149, -145, -130, -126, -100, -40, 0, 40, 100, 124] {
            for sign: Float in [-1, 1] {
                let s = Float(sign: .plus, exponent: exponent, significand: sign)
                for factors: [Float] in [[1,1], [3,4]] {
                    let a = factors.map { $0*s }
                    let norm = hypot(Double(a[0]), Double(a[1]))
                    let q = try provider.qrThin(a, rows: 2, columns: 1)
                    #expect(L01Oracle.orthogonal(q.q, rows: 2, columns: 1))
                    #expect(q.r.count == 1 && L01Oracle.close(abs(q.r[0]), norm))
                    #expect(L01Oracle.reconstruct(a, left: q.q, right: q.r, m: 2, k: 1, n: 1))
                    for (m,n,input) in [(2,1,a), (1,2,a)] {
                        let f = try provider.svdThin(input, rows: m, columns: n)
                        #expect(L01Oracle.orthogonal(f.u, rows: m, columns: 1))
                        #expect(L01Oracle.orthogonal(L01Oracle.transpose(f.vt, rows: 1, columns: n), rows: n, columns: 1))
                        #expect(f.singularValues.count == 1 && L01Oracle.close(f.singularValues[0], norm))
                        #expect(L01Oracle.reconstruct(input, left: f.u, right: f.vt, m: m, k: 1, n: n, spectrum: f.singularValues))
                    }
                }
                // Full rank golden-ratio spectrum. No null-space exception applies.
                let a: [Float] = [s,0,s,s]
                let f = try provider.svdThin(a, rows: 2, columns: 2)
                #expect(L01Oracle.orthogonal(f.u, rows: 2, columns: 2))
                #expect(L01Oracle.orthogonal(L01Oracle.transpose(f.vt, rows: 2, columns: 2), rows: 2, columns: 2))
                #expect(f.singularValues.count == 2)
                for j in 0..<2 {
                    let factor = j == 0 ? (1+sqrt(5))/2 : (sqrt(5)-1)/2
                    #expect(L01Oracle.close(f.singularValues[j], factor*abs(Double(s))))
                }
                #expect(L01Oracle.reconstruct(a, left: f.u, right: f.vt, m: 2, k: 2, n: 2, spectrum: f.singularValues))
                let symmetric: [Float] = [2*s,s,s,2*s]
                let e = try provider.symmetricEigen(symmetric, dimension: 2, computeEigenvectors: true)
                let expected = [Double(s),3*Double(s)].sorted()
                #expect(e.eigenvalues.count == 2)
                for j in 0..<2 { #expect(L01Oracle.close(e.eigenvalues[j], expected[j])) }
                let v = try #require(e.eigenvectors)
                #expect(L01Oracle.orthogonal(v, rows: 2, columns: 2))
                #expect(L01Oracle.reconstruct(symmetric, left: v, right: L01Oracle.transpose(v, rows: 2, columns: 2), m: 2, k: 2, n: 2, spectrum: e.eigenvalues))
                let scalar = try provider.symmetricEigen([s], dimension: 1, computeEigenvectors: true)
                #expect(scalar.eigenvalues == [s] && scalar.eigenvectors == [1])
            }
        }
    }

    @Test func constructedRectangularAndZero() throws {
        let p = SwiftLinearAlgebraProvider()
        // Columns of H/2 (Hadamard) are orthonormal; spectra 4,2,1.
        let h: [Float] = [1,1,1,1, 1,-1,1,-1, 1,1,-1,-1]
        for exponent in [-140, -100, 0, 100, 124] {
            let s = Float(sign: .plus, exponent: exponent, significand: 1)
            var tall = h
            for j in 0..<3 { for i in 0..<4 { tall[j*4+i] *= [2*s,s,s/2][j] } }
            for (m,n,a) in [(4,3,tall),(3,4,L01Oracle.transpose(tall, rows: 4, columns: 3))] {
                let f = try p.svdThin(a, rows: m, columns: n)
                #expect(L01Oracle.orthogonal(f.u, rows: m, columns: 3))
                #expect(L01Oracle.orthogonal(L01Oracle.transpose(f.vt, rows: 3, columns: n), rows: n, columns: 3))
                #expect(f.singularValues.count == 3)
                for j in 0..<3 { #expect(L01Oracle.close(f.singularValues[j], [4,2,1][j]*Double(s))) }
                #expect(L01Oracle.reconstruct(a, left: f.u, right: f.vt, m: m, k: 3, n: n, spectrum: f.singularValues))
            }
            let q = try p.qrThin(tall, rows: 4, columns: 3)
            #expect(L01Oracle.orthogonal(q.q, rows: 4, columns: 3))
            #expect(L01Oracle.reconstruct(tall, left: q.q, right: q.r, m: 4, k: 3, n: 3))
            for j in 0..<3 { for i in (j+1)..<3 { #expect(q.r[j*3+i].bitPattern == 0) } }
        }
        for (m,n) in [(1,1),(4,3),(3,4)] {
            let a = [Float](repeating: 0, count: m*n)
            let f = try p.svdThin(a, rows: m, columns: n)
            #expect(f.singularValues.allSatisfy { $0 == 0 })
            #expect((m >= n ? f.u : f.vt).allSatisfy { $0 == 0 })
            #expect(L01Oracle.reconstruct(a, left: f.u, right: f.vt, m: m, k: min(m,n), n: n, spectrum: f.singularValues))
            if m >= n {
                let q = try p.qrThin(a, rows: m, columns: n)
                #expect(L01Oracle.orthogonal(q.q, rows: m, columns: n))
            }
        }
        let diagonal: [Float] = [-3,0,0,0,2,0,0,0,0]
        let e = try p.symmetricEigen(diagonal, dimension: 3, computeEigenvectors: true)
        #expect(e.eigenvalues == [-3,0,2])
        #expect(L01Oracle.orthogonal(try #require(e.eigenvectors), rows: 3, columns: 3))
    }

    @Test func predicateFailureControls() {
        #expect(!L01Oracle.close(.nan, 1)); #expect(!L01Oracle.close(.infinity, 1))
        #expect(!L01Oracle.close(0, 5 * Double(Float.leastNonzeroMagnitude)))
        #expect(!L01Oracle.orthogonal([.nan], rows: 1, columns: 1))
        #expect(!L01Oracle.orthogonal([], rows: 1, columns: 1))
        #expect(!L01Oracle.reconstruct([1], left: [.nan], right: [1], m: 1, k: 1, n: 1))
        #expect(!L01Oracle.reconstruct([1], left: [], right: [1], m: 1, k: 1, n: 1))
    }

    @Test func workingByteBoundaries() throws {
        try LinearAlgebraValidation.validateWorkingCount(Int.max / MemoryLayout<Double>.stride)
        #expect(throws: VectorError.self) {
            try LinearAlgebraValidation.validateWorkingCount(Int.max / MemoryLayout<Double>.stride + 1)
        }
        #expect(throws: VectorError.self) { try LinearAlgebraValidation.validateWorkingCount(-1) }
    }

    @Test func pcaSwiftProviderControl() throws {
        let data = [[3,0],[-3,0],[0,1],[0,-1]].map { DynamicVector($0.map(Float.init)) }
        try Operations.$linearAlgebraProvider.withValue(SwiftLinearAlgebraProvider()) {
            let (projected, model) = try Operations.pca(data, components: 2)
            #expect(projected.count == 4 && model.components.count == 4)
            #expect(model.explainedVariance.count == 2)
            #expect(L01Oracle.close(model.explainedVariance[0], 6))
            #expect(L01Oracle.close(model.explainedVariance[1], 2.0/3))
            #expect(model.components.allSatisfy { $0.isFinite })
        }
    }

    @Test(.serialized, arguments: ["nonfinite", "overflow", "lower", "immutability"])
    func boundedDomain(_ scenario: String) throws {
        var env = ProcessInfo.processInfo.environment
        env["L01_SCENARIO"] = scenario
        let directory = env["L01_CAPTURE_DIRECTORY"].map { URL(fileURLWithPath: $0) } ?? FileManager.default.temporaryDirectory
        let path = directory.appendingPathComponent("L01-\(scenario)-\(UUID().uuidString).log")
        FileManager.default.createFile(atPath: path.path, contents: nil)
        let file = try FileHandle(forWritingTo: path)
        let process = try launchChildProbe(.swiftTesting(filter: "LinearAlgebraRangeTests/domainChild"),
                                           environment: env, output: file, deadline: 15,
                                           cleanup: .terminateThenKill(grace: 1, seconds: 1))
        if process.timedOut {
            try file.close()
            print("L01_CHILD_CAPTURE \(path.path)")
            throw NSError(domain: "L01.timeout", code: 1)
        }
        try file.close()
        let output = try String(contentsOf: path, encoding: .utf8)
        print("L01_CHILD_CAPTURE \(path.path)")
        print("L01_CHILD_RESULT \(scenario) status=\(process.statusDescription) reason=\(process.reasonDescription)")
        #expect(process.terminationStatus == 0, "\(output.prefix(6000))")
        #expect(output.contains("L01_COMPLETE \(scenario)"))
    }

    @Test func domainChild() throws {
        guard let scenario = ProcessInfo.processInfo.environment["L01_SCENARIO"] else { return }
        print("L01_INVOCATION " + String(decoding: try JSONSerialization.data(withJSONObject: ["argv": CommandLine.arguments, "scenario": scenario], options: [.sortedKeys]), as: UTF8.self))
        var providers: [any LinearAlgebraProvider] = [SwiftLinearAlgebraProvider()]
        if LAPACKLinearAlgebraProvider.isAvailable { providers.append(LAPACKLinearAlgebraProvider()) }
        for p in providers {
            switch scenario {
            case "nonfinite":
                for bad: Float in [.nan,.infinity,-.infinity] {
                    for operation in 0..<3 {
                        do {
                            if operation == 0 { _ = try p.qrThin([bad], rows: 1, columns: 1) }
                            if operation == 1 { _ = try p.svdThin([bad], rows: 1, columns: 1) }
                            if operation == 2 { _ = try p.symmetricEigen([bad], dimension: 1, computeEigenvectors: false) }
                            Issue.record("accepted nonfinite input")
                        } catch let error as VectorError { #expect(error.kind == .invalidData) }
                    }
                    do { _ = try p.qrThin([bad], rows: 2, columns: 1); Issue.record("shape") }
                    catch let e as VectorError { #expect(e.kind == .dimensionMismatch) }
                }
            case "overflow":
                let huge = Float.greatestFiniteMagnitude
                for operation in 0..<3 {
                    do {
                        if operation == 0 { _ = try p.qrThin([huge,huge], rows: 2, columns: 1) }
                        if operation == 1 { _ = try p.svdThin([huge,huge], rows: 2, columns: 1) }
                        if operation == 2 { _ = try p.symmetricEigen([huge,huge,huge,huge], dimension: 2, computeEigenvectors: true) }
                        Issue.record("accepted unrepresentable output")
                    } catch let e as VectorError { #expect(e.kind == .operationFailed) }
                }
            case "lower":
                for poison: Float in [.nan,.infinity,-.infinity] {
                    for vectors in [false,true] {
                        let e = try p.symmetricEigen([2,1,poison,2], dimension: 2, computeEigenvectors: vectors)
                        #expect(e.eigenvalues.count == 2 && L01Oracle.close(e.eigenvalues[0],1) && L01Oracle.close(e.eigenvalues[1],3))
                        if vectors { #expect(L01Oracle.orthogonal(try #require(e.eigenvectors), rows: 2, columns: 2)) }
                        else { #expect(e.eigenvectors == nil) }
                    }
                }
            case "immutability":
                let a: [Float] = [2,1,1,2]
                let before = a.map(\.bitPattern)
                _ = try p.qrThin(a, rows: 2, columns: 2)
                _ = try p.svdThin(a, rows: 2, columns: 2)
                _ = try p.symmetricEigen(a, dimension: 2, computeEigenvectors: true)
                #expect(a.map(\.bitPattern) == before)
            default: Issue.record("unknown scenario")
            }
        }
        print("L01_COMPLETE \(scenario)")
    }
}

extension LinearAlgebraRangeTests {
    @Test func unequalScalesAndRoundedZeroSpectrum() throws {
        let p = SwiftLinearAlgebraProvider()
        let small = Float.leastNonzeroMagnitude
        let large = Float.greatestFiniteMagnitude / 4
        for a in [[large,0,small,small], [small,small,large,0], [small,0,2*small,small]] {
            let f = try p.svdThin(a, rows: 2, columns: 2)
            #expect(L01Oracle.orthogonal(f.u, rows: 2, columns: 2))
            #expect(L01Oracle.orthogonal(L01Oracle.transpose(f.vt, rows: 2, columns: 2), rows: 2, columns: 2))
            #expect(L01Oracle.reconstruct(a, left: f.u, right: f.vt, m: 2, k: 2, n: 2, spectrum: f.singularValues))
            if a[0] == small && a[1] == 0 {
                // Exact spectrum (sqrt(2)+1, sqrt(2)-1)*small; second rounds to0.
                #expect(f.singularValues.count == 2 && f.singularValues[1].bitPattern == 0)
                #expect(L01Oracle.close(f.singularValues[0], (sqrt(2)+1)*Double(small)))
            } else {
                #expect(f.singularValues.count == 2 && L01Oracle.close(f.singularValues[0], Double(large)))
                #expect(L01Oracle.close(f.singularValues[1], Double(small)))
            }
        }
    }
}


extension LinearAlgebraRangeTests {
    @Test func workspaceErrorMapping() throws {
        try LAPACKLinearAlgebraProvider.check(0, routine: "test")
        for (code, kind, fragment): (Int32, VectorError.ErrorKind, String) in [
            (-1000, .resourceUnavailable, "unavailable or workspace allocation failed"),
            (-1001, .operationFailed, "invalid workspace query"),
            (-1002, .invalidDimension, "local workspace size is not representable"),
            (-3, .invalidOperation, "position 3"),
            (Int32.min, .invalidOperation, "2147483648"),
            (1, .operationFailed, "failed to converge")
        ] {
            do { try LAPACKLinearAlgebraProvider.check(code, routine: "test"); Issue.record("accepted error") }
            catch let e as VectorError { #expect(e.kind == kind && e.description.contains(fragment)) }
        }
    }
}


extension LinearAlgebraRangeTests {
    @Test func mixedKnownSpectrumAcrossScales() throws {
        let p = SwiftLinearAlgebraProvider()
        // H/2 is exactly orthogonal. A=H diag(8,4,2,1) Hᵀ/4 uses only
        // dyadic entries, so these scales introduce no input rounding. Unlike
        // already-orthogonal tall columns, this fixture requires Jacobi rotations.
        let h: [Double] = [1,1,1,1, 1,-1,1,-1, 1,1,-1,-1, 1,-1,-1,1]
        let spectrum: [Double] = [8,4,2,1]
        for exponent in [-140,-80,0,80,124] {
            for sign: Float in [-1,1] {
                let s = Float(sign: .plus, exponent: exponent, significand: sign)
                var symmetric = [Float](repeating: 0, count: 16)
                for j in 0..<4 {
                    for i in 0..<4 {
                        var sum = 0.0
                        for k in 0..<4 { sum += h[k*4+i]*spectrum[k]*h[k*4+j]/4 }
                        symmetric[j*4+i] = Float(sum*Double(s))
                    }
                }
                var tall = [Float](repeating: 0, count: 24)
                for j in 0..<4 { for i in 0..<4 { tall[j*6+i] = symmetric[j*4+i] } }
                for (m,n,a) in [(6,4,tall),(4,6,L01Oracle.transpose(tall, rows: 6, columns: 4))] {
                    let f = try p.svdThin(a, rows: m, columns: n)
                    #expect(L01Oracle.orthogonal(f.u, rows: m, columns: 4))
                    #expect(L01Oracle.orthogonal(L01Oracle.transpose(f.vt, rows: 4, columns: n), rows: n, columns: 4))
                    try #require(f.singularValues.count == 4)
                    for j in 0..<4 { #expect(L01Oracle.close(f.singularValues[j], spectrum[j]*abs(Double(s)))) }
                    #expect(L01Oracle.reconstruct(a, left: f.u, right: f.vt, m: m, k: 4, n: n, spectrum: f.singularValues))
                }
                let q = try p.qrThin(tall, rows: 6, columns: 4)
                #expect(L01Oracle.orthogonal(q.q, rows: 6, columns: 4))
                #expect(L01Oracle.reconstruct(tall, left: q.q, right: q.r, m: 6, k: 4, n: 4))
                for j in 0..<4 { for i in (j+1)..<4 { #expect(q.r[j*4+i].bitPattern == 0) } }
                let e = try p.symmetricEigen(symmetric, dimension: 4, computeEigenvectors: true)
                let v = try #require(e.eigenvectors)
                let expected = spectrum.map { $0*Double(s) }.sorted()
                try #require(e.eigenvalues.count == 4)
                for j in 0..<4 { #expect(L01Oracle.close(e.eigenvalues[j], expected[j])) }
                #expect(L01Oracle.orthogonal(v, rows: 4, columns: 4))
                #expect(L01Oracle.eigenResidual(symmetric, values: e.eigenvalues, vectors: v, n: 4))
                let valuesOnly = try p.symmetricEigen(symmetric, dimension: 4, computeEigenvectors: false)
                #expect(valuesOnly.eigenvectors == nil)
                try #require(valuesOnly.eigenvalues.count == 4)
                for j in 0..<4 { #expect(L01Oracle.close(valuesOnly.eigenvalues[j], expected[j])) }
            }
        }
        #expect(!L01Oracle.eigenResidual([1], values: [.nan], vectors: [1], n: 1))
        #expect(!L01Oracle.eigenResidual([1], values: [1], vectors: [], n: 1))
    }
}
