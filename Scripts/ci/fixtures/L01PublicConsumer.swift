import Foundation
import VectorCore

@main struct L01PublicConsumer {
    static func main() throws {
        let p = SwiftLinearAlgebraProvider()
        let f = try p.svdThin([1e-12,0,1e-12,1e-12], rows: 2, columns: 2)
        precondition(f.u.allSatisfy(\.isFinite) && f.singularValues.count == 2)
        precondition(abs(Double(f.singularValues[0])/1e-12 - (1+sqrt(5))/2) < 1e-5)
        let q = try p.qrThin([3e20,4e20], rows: 2, columns: 1)
        precondition(q.q.allSatisfy(\.isFinite) && abs(abs(Double(q.r[0]))/5e20-1) < 1e-5)
        let e = try p.symmetricEigen([2,1,.nan,2], dimension: 2, computeEigenvectors: false)
        precondition(e.eigenvectors == nil && abs(e.eigenvalues[0]-1) < 1e-5)
        do { _ = try p.svdThin([.nan], rows: 1, columns: 1); fatalError("nonfinite accepted") }
        catch let error as VectorError { precondition(error.kind == .invalidData) }
        do { _ = try p.svdThin([Float.greatestFiniteMagnitude,Float.greatestFiniteMagnitude], rows: 2, columns: 1); fatalError("overflow accepted") }
        catch let error as VectorError { precondition(error.kind == .operationFailed) }
        print("L01_PUBLIC_COMPLETE")
    }
}
