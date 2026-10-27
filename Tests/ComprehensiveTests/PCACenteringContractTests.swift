import Foundation
import Testing
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif
@testable import VectorCore

@Suite struct PCACenteringContractTests {
    @Test(arguments: [false, true])
    func translatedPair(_ swiftProvider: Bool) throws {
        func check() throws {
            for offset: Float in [0, 16777216] {
                let values: [[Float]] = [[offset], [offset+2]]
                let data = values.map(DynamicVector.init)
                let model = try PCAModel.fit(data, components: 1)
                #expect(model.mean.count == 1 && Double(model.mean[0]) == Double(offset)+1)
                #expect(model.explainedVariance.count == 1 && model.explainedVariance[0].isFinite && abs(model.explainedVariance[0]-2) < 4e-6)
                #expect(model.explainedVarianceRatio.count == 1 && model.explainedVarianceRatio[0].isFinite && abs(model.explainedVarianceRatio[0]-1) < 4e-6)
                let batch = try model.transform(data)
                let oneShot = try Operations.pca(data, components: 1).projected
                for result in [batch,oneShot] {
                    try #require(result.count == 2 && result.allSatisfy { $0.count == 1 })
                    #expect(result[0][0].isFinite && abs(result[0][0]+1) < 1e-6)
                    #expect(result[1][0].isFinite && abs(result[1][0]-1) < 1e-6)
                    #expect(result[0][0].isFinite && result[1][0].isFinite && abs(Double(result[0][0])+Double(result[1][0])) < 1e-6)
                }
                for i in 0..<2 {
                    let single = try model.transform(values[i])
                    #expect(single.count == 1 && single[0].isFinite && abs(single[0] - (i == 0 ? -1 : 1)) < 1e-6)
                }
            }
        }
        if swiftProvider {
            try Operations.$linearAlgebraProvider.withValue(SwiftLinearAlgebraProvider()) { try check() }
        } else { try check() }
    }

    @Test(.serialized, arguments: ["oversampling", "finite"])
    func oversamplingInBoundedChild(_ scenario: String) throws {
        var env = ProcessInfo.processInfo.environment
        env["L02_SCENARIO"] = scenario
        let directory = env["L02_CAPTURE_DIRECTORY"].map { URL(fileURLWithPath: $0) } ?? FileManager.default.temporaryDirectory
        let path = directory.appendingPathComponent("L02-\(scenario)-\(UUID().uuidString).log")
        FileManager.default.createFile(atPath: path.path, contents: nil)
        let file = try FileHandle(forWritingTo: path)
        let process = try launchChildProbe(.swiftTesting(filter: "PCACenteringContractTests/childProbe"),
                                           environment: env, output: file, deadline: 15,
                                           cleanup: .terminateThenKill(grace: 1, seconds: 1))
        if process.timedOut {
            try file.close()
            print("L02_CHILD_CAPTURE \(path.path)")
            throw NSError(domain: "L02.timeout", code: 1)
        }
        try file.close()
        let output = try String(contentsOf: path, encoding: .utf8)
        print("L02_CHILD_CAPTURE \(path.path)")
        print("L02_CHILD_RESULT status=\(process.statusDescription) reason=\(process.reasonDescription)")
        #expect(process.terminationStatus == 0, "\(output.prefix(6000))")
        #expect(output.contains("L02_COMPLETE \(scenario)"))
    }

    @Test func childProbe() throws {
        guard let scenario = ProcessInfo.processInfo.environment["L02_SCENARIO"] else { return }
        print("L02_INVOCATION " + String(decoding: try JSONSerialization.data(withJSONObject: ["argv":CommandLine.arguments, "scenario":scenario], options: [.sortedKeys]), as: UTF8.self))
        fflush(stdout)
        if scenario == "oversampling" {
            let model = try PCAModel.fit([DynamicVector([Float(0)]),DynamicVector([Float(2)])], components: 1, config: PCAConfig(oversampling: Int.max))
            #expect(model.componentCount == 1)
        } else if scenario == "finite" { try finiteBoundaries() }
        else { Issue.record("Unknown child scenario") }
        print("L02_COMPLETE \(scenario)")
    }
}


private final class L02Calls: @unchecked Sendable {
    private let lock = NSLock()
    private var entries: [String] = []
    func add(_ entry: String) { lock.lock(); defer { lock.unlock() }; entries.append(entry) }
    func read() -> [String] { lock.lock(); defer { lock.unlock() }; return entries }
}
private struct L02Forwarding: LinearAlgebraProvider {
    let calls: L02Calls
    func qrThin(_ a: [Float], rows: Int, columns: Int) throws -> QRFactorization {
        calls.add("qr:\(rows):\(columns)")
        return try SwiftLinearAlgebraProvider().qrThin(a, rows: rows, columns: columns)
    }
    func svdThin(_ a: [Float], rows: Int, columns: Int) throws -> SingularValueDecomposition {
        calls.add("svd:\(rows):\(columns)")
        return try SwiftLinearAlgebraProvider().svdThin(a, rows: rows, columns: columns)
    }
    func symmetricEigen(_ a: [Float], dimension: Int, computeEigenvectors: Bool) throws -> SymmetricEigenDecomposition {
        try SwiftLinearAlgebraProvider().symmetricEigen(a, dimension: dimension, computeEigenvectors: computeEigenvectors)
    }
}
private final class L02Sentinel: Error, Sendable {}
private struct L02Throwing: LinearAlgebraProvider {
    let error: L02Sentinel
    func qrThin(_ a: [Float], rows: Int, columns: Int) throws -> QRFactorization { throw error }
    func svdThin(_ a: [Float], rows: Int, columns: Int) throws -> SingularValueDecomposition { throw error }
    func symmetricEigen(_ a: [Float], dimension: Int, computeEigenvectors: Bool) throws -> SymmetricEigenDecomposition { throw error }
}

private enum L02Oracle {
    // Small fixture sums/products use Double, independently of PCA helpers.
    static func mean(_ rows: [[Float]]) -> [Double] {
        (0..<rows[0].count).map { j in rows.reduce(0.0) { $0+Double($1[j]) } / Double(rows.count) }
    }
    static func close(_ a: Double, _ b: Double, scale: Double = 1) -> Bool {
        a.isFinite && b.isFinite && scale.isFinite && abs(a-b) <= 4e-6*max(1,abs(scale))
    }
    static func reconstruction(_ original: [[Float]], projected: [[Float]], model: PCAModel) -> Bool {
        let n=original.count, d=model.inputDimension, k=model.componentCount
        guard projected.count == n, original.allSatisfy({ $0.count == d && $0.allSatisfy(\.isFinite) }),
              projected.allSatisfy({ $0.count == k && $0.allSatisfy(\.isFinite) }),
              model.mean.count == d, model.mean.allSatisfy(\.isFinite),
              model.components.count == k*d, model.components.allSatisfy(\.isFinite) else { return false }
        var scale=0.0
        for i in 0..<n { for j in 0..<d { scale=max(scale,abs(Double(original[i][j])-model.mean[j])) } }
        for i in 0..<n {
            for j in 0..<d {
                var residual=0.0
                for r in 0..<k { residual += Double(model.components[r*d+j])*Double(projected[i][r]) }
                if !close(residual,Double(original[i][j])-model.mean[j],scale:scale) { return false }
            }
        }
        return true
    }
    static func kind(_ body: () throws -> Void) -> VectorError.ErrorKind? {
        do { try body(); return nil } catch let e as VectorError { return e.kind } catch { return nil }
    }
}

extension PCACenteringContractTests {
    @Test func sketchBoundariesAndProviderIdentity() throws {
        var data: [DynamicVector] = []
        for i in 0..<6 {
            var row: [Float] = []
            for j in 0..<4 { row.append(Float((i + 1) * (j + 2) % 7)) }
            data.append(DynamicVector(row))
        }
        for p in [0,1,2,3,4,Int.max] {
            let calls=L02Calls()
            let iterations = p >= 3 ? Int.max : 1
            _ = try Operations.$linearAlgebraProvider.withValue(L02Forwarding(calls:calls)) {
                try PCAModel.fit(data, components: 1, config: PCAConfig(oversampling:p,powerIterations:iterations))
            }
            if p >= 3 { #expect(calls.read() == ["svd:6:4"]) }
            else {
                let ell=1+p
                #expect(calls.read() == ["qr:6:\(ell)","qr:4:\(ell)","qr:6:\(ell)","svd:\(ell):4"])
            }
        }
        for config in [PCAConfig(oversampling:-1),PCAConfig(powerIterations:-1)] {
            #expect(L02Oracle.kind { _ = try PCAModel.fit(data, components:1, config:config) } == .invalidDimension)
        }
        let sentinel=L02Sentinel()
        for p in [0,Int.max] {
            do {
                _ = try Operations.$linearAlgebraProvider.withValue(L02Throwing(error:sentinel)) {
                    try PCAModel.fit(data,components:1,config:PCAConfig(oversampling:p))
                }
                Issue.record("provider error lost")
            } catch let e as L02Sentinel { #expect(e === sentinel) }
        }
    }

    @Test func shapeAndSharedFlattenBoundaries() throws {
        #expect(try PCAShape.count(rows:Int.max/4,columns:1) == Int.max/4)
        #expect(L02Oracle.kind { _ = try PCAShape.count(rows:Int.max/4+1,columns:1) } == .invalidDimension)
        #expect(try PCAShape.count(rows:Int.max/8,columns:1,width:8) == Int.max/8)
        #expect(L02Oracle.kind { _ = try PCAShape.count(rows:Int.max/8+1,columns:1,width:8) } == .invalidDimension)
        #expect(L02Oracle.kind { _ = try PCAShape.count(rows:Int.max,columns:2) } == .invalidDimension)
        #expect(L02Oracle.kind { _ = try PCAShape.count(rows:-1,columns:2) } == .invalidDimension)
        #expect(try PCAShape.flatten(rows:0,dimension:3,vectorCount:0) == 0)
        try PCAShape.transform(rows:Int(Int32.max),dimension:1,components:1)
        for value in [0,Int(Int32.max)+1] {
            #expect(L02Oracle.kind { try PCAShape.transform(rows:value,dimension:1,components:1) } == .invalidDimension)
            #expect(L02Oracle.kind { _ = try PCAShape.fit(rows:2,dimension:value,components:1,config:PCAConfig(),flatCount:0) } == .invalidDimension)
        }
        let rows=[DynamicVector([Float(1),2]),DynamicVector([Float(3),4])]
        #expect(L02Oracle.kind { _ = try PCAModel.flattenColumnMajor(rows,count:3,dimension:2) } == .invalidDimension)
        #expect(L02Oracle.kind { _ = try PCAModel.fit(columnMajor:[1],rows:2,dimension:2,components:1,config:PCAConfig()) } == .dimensionMismatch)
        for n in [Int.min,0,1] {
            #expect(L02Oracle.kind { _ = try PCAModel.fit(columnMajor:[],rows:n,dimension:1,components:1,config:PCAConfig()) } == .invalidDimension)
        }
        #expect(L02Oracle.kind { _ = try PCAModel.fit([DynamicVector([Float(1)])],components:0,config:PCAConfig(oversampling:-1)) } == .invalidOperation)
        let ragged=[DynamicVector([Float.nan,0]),DynamicVector([Float(1)])]
        #expect(L02Oracle.kind { _ = try PCAModel.fit(ragged,components:0,config:PCAConfig(oversampling:-1)) } == .dimensionMismatch)
        let payload:[Float]=[Float(bitPattern:0x7fc01234),.infinity]
        let flat=try PCAModel.flattenColumnMajor(payload.map { DynamicVector([$0]) },count:2,dimension:1)
        #expect(flat.map(\.bitPattern) == payload.map(\.bitPattern))
    }

    @Test(arguments:[false,true]) func multicolumnAndRandomizedTranslations(_ swiftProvider: Bool) throws {
        func run() throws {
            let scales:[Float]=[4,3,2,1]
            var original=[[Float]]()
            for j in 0..<4 { for sign:Float in [-1,1] { var row=[Float](repeating:0,count:4);row[j]=sign*scales[j];original.append(row) } }
            for offset:Float in [0,65536,-1048576] {
                let raw=original.map { $0.map { $0+offset } }
                let data=raw.map(DynamicVector.init)
                let model=try PCAModel.fit(data,components:4)
                let projected=try model.transform(data)
                #expect(model.mean == [Double](repeating:Double(offset),count:4))
                try #require(model.explainedVariance.count == 4)
                for j in 0..<4 { #expect(L02Oracle.close(Double(model.explainedVariance[j]),2*Double(scales[j])*Double(scales[j])/7,scale:5)) }
                #expect(L02Oracle.reconstruction(raw,projected:projected,model:model))
                for j in 0..<4 { #expect(L02Oracle.close(projected.reduce(0.0) { $0+Double($1[j]) },0)) }
                for i in 0..<8 {
                    let single=try model.transform(raw[i]);try #require(single.count == 4)
                    for j in 0..<4 { #expect(L02Oracle.close(Double(single[j]),Double(projected[i][j]),scale:4)) }
                }
                let config=PCAConfig(oversampling:1,powerIterations:3,seed:71)
                let random=try PCAModel.fit(data,components:1,config:config)
                let base=try PCAModel.fit(original.map(DynamicVector.init),components:1,config:config)
                try #require(random.components.count == 4 && random.explainedVariance.count == 1)
                #expect(L02Oracle.close(Double(random.explainedVariance[0]),Double(base.explainedVariance[0]),scale:5))
                for j in 0..<4 { #expect(L02Oracle.close(Double(random.components[j]),Double(base.components[j]))) }
            }
            let constant=[[Float](repeating:7,count:3),[Float](repeating:7,count:3),[Float](repeating:7,count:3)]
            let zero=try PCAModel.fit(constant.map(DynamicVector.init),components:1)
            #expect(zero.mean == [7,7,7]);#expect(zero.explainedVariance == [0]);#expect(zero.explainedVarianceRatio == [0])
            #expect(try zero.transform(constant.map(DynamicVector.init)) == [[0],[0],[0]])
            let raw=try PCAModel.fit([DynamicVector([Float(1)]),DynamicVector([Float(3)])],components:1,config:PCAConfig(center:false))
            #expect(raw.mean == [0]);#expect(L02Oracle.close(Double(raw.explainedVariance[0]),10,scale:10))
            #expect(try raw.transform([Float(3)]) == [3])
        }
        if swiftProvider { try Operations.$linearAlgebraProvider.withValue(SwiftLinearAlgebraProvider()) { try run() } }
        else { try run() }
    }

    private func finiteBoundaries() throws {
        for useSwift in [false,true] {
            func run() throws {
                let model=try PCAModel.fit([DynamicVector([Float(0)]),DynamicVector([Float(2)])],components:1)
                for value:Float in [.nan,.infinity,-.infinity] {
                    for center in [false,true] {
                        #expect(L02Oracle.kind { _ = try PCAModel.fit([DynamicVector([value]),DynamicVector([Float(0)])],components:1,config:PCAConfig(center:center)) } == .invalidData)
                    }
                    #expect(L02Oracle.kind { _ = try model.transform([value]) } == .invalidData)
                    #expect(L02Oracle.kind { _ = try model.transform([DynamicVector([value])]) } == .invalidData)
                }
                let maximum=Float.greatestFiniteMagnitude
                #expect(L02Oracle.kind { _ = try PCAModel.fit([-maximum,maximum,maximum].map { DynamicVector([$0]) },components:1) } == .operationFailed)
                #expect(L02Oracle.kind { _ = try PCAModel.fit([-Float(1e20),Float(1e20)].map { DynamicVector([$0]) },components:1) } == .operationFailed)
                let offset=try PCAModel.fit([maximum,maximum].map { DynamicVector([$0]) },components:1)
                #expect(L02Oracle.kind { _ = try offset.transform([-maximum]) } == .operationFailed)
                #expect(L02Oracle.kind { _ = try offset.transform([DynamicVector([-maximum])]) } == .operationFailed)
                let direction=try PCAModel.fit([DynamicVector([Float(-1),-1]),DynamicVector([Float(1),1])],components:1)
                #expect(L02Oracle.kind { _ = try direction.transform([maximum,maximum]) } == .operationFailed)
                #expect(L02Oracle.kind { _ = try direction.transform([DynamicVector([maximum,maximum])]) } == .operationFailed)
                let badRows=[DynamicVector([Float.nan,0]),DynamicVector([Float(1)])]
                #expect(L02Oracle.kind { _ = try direction.transform(badRows) } == .dimensionMismatch)
                #expect(try model.transform([DynamicVector]()).isEmpty)
            }
            if useSwift { try Operations.$linearAlgebraProvider.withValue(SwiftLinearAlgebraProvider()) { try run() } }
            else { try run() }
        }
    }
}

extension PCACenteringContractTests {
    @Test func ordinaryOffsetCovarianceAndImmutability() throws {
        var raw: [[Float]] = []
        for i in 0..<32 {
            let x: Float = Float((i * 7) % 19) / 8 + 1000
            let y: Float = Float((i * 11) % 23) / 4 - 2000
            let z: Float = Float((i * 5) % 17) / 16 + 3000
            raw.append([x, y, z])
        }
        let data=raw.map(DynamicVector.init)
        let bits=raw.map { $0.map(\.bitPattern) }
        let referenceMean=L02Oracle.mean(raw)
        var covariance=[Double](repeating:0,count:9)
        for row in raw {
            for i in 0..<3 {
                for j in 0..<3 { covariance[i*3+j] += (Double(row[i])-referenceMean[i])*(Double(row[j])-referenceMean[j])/31 }
            }
        }
        let total=covariance[0]+covariance[4]+covariance[8]
        for useSwift in [false,true] {
            func run() throws {
                let model=try PCAModel.fit(data,components:3)
                #expect(model.mean == referenceMean)
                try #require(model.components.count == 9 && model.explainedVariance.count == 3 && model.explainedVarianceRatio.count == 3)
                for r in 0..<3 {
                    var rayleigh=0.0
                    for i in 0..<3 { for j in 0..<3 { rayleigh += Double(model.components[r*3+i])*covariance[i*3+j]*Double(model.components[r*3+j]) } }
                    #expect(L02Oracle.close(Double(model.explainedVariance[r]),rayleigh,scale:total))
                    #expect(L02Oracle.close(Double(model.explainedVarianceRatio[r]),rayleigh/total))
                }
                #expect(L02Oracle.reconstruction(raw,projected:try model.transform(data),model:model))
                #expect(data.map { $0.toArray().map(\.bitPattern) } == bits)
            }
            if useSwift { try Operations.$linearAlgebraProvider.withValue(SwiftLinearAlgebraProvider()) { try run() } }
            else { try run() }
        }
        #expect(!L02Oracle.close(.nan,0));#expect(!L02Oracle.close(.infinity,0))
    }
}


extension PCACenteringContractTests {
    @Test func compensatedMeanRetainsSmallTerm() throws {
        let large: Float = 0x1p60
        let data = [DynamicVector([large]), DynamicVector([1]), DynamicVector([-large])]
        for useSwift in [false, true] {
            func run() throws {
                let model = try PCAModel.fit(data, components: 1)
                #expect(model.mean == [1.0 / 3.0])
                #expect(model.explainedVariance.allSatisfy { $0.isFinite })
            }
            if useSwift { try Operations.$linearAlgebraProvider.withValue(SwiftLinearAlgebraProvider()) { try run() } }
            else { try run() }
        }
    }
}
