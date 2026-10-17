import Foundation
import Testing
import VectorCore

// D02 phase 4: VectorTypeFactory keeps its concrete results (Vector<Dim128/256/384/512/768/1536/3072>, DynamicVector
// elsewhere in the checked domain) and never routes to the separate Vector*Optimized types; createVector keeps its
// nonthrowing signature and traps on malformed input (child-isolated). The external consumer
// Scripts/ci/tests/D02/Factory.swift pins the same results without @testable.

private func d02ExpectedType(_ dimension: Int) -> String {
    switch dimension {
    case 128, 256, 384, 512, 768, 1536, 3072: return "Vector<Dim\(dimension)>"
    default: return "DynamicVector"
    }
}

@Suite("D02 VectorTypeFactory contract")
struct D02FactoryContractTests {
    static let named = [128, 256, 384, 512, 768, 1536, 3072]
    static let others = [0, 1, 2, 3, 100, 127, 129, 1024, 2048, 4096]

    @Test("vector(of:from:), zeros and ones return Vector<DimN> for the seven named dimensions, DynamicVector otherwise")
    func concreteResults() throws {
        for dimension in Self.named + Self.others {
            let values = (0..<dimension).map { Float($0 % 7) }
            let made = try VectorTypeFactory.vector(of: dimension, from: values)
            let zeros = VectorTypeFactory.zeros(dimension: dimension)
            let ones = VectorTypeFactory.ones(dimension: dimension)
            let expected = d02ExpectedType(dimension)
            for (route, vector) in [("vector", made), ("zeros", zeros), ("ones", ones)] {
                #expect(String(describing: type(of: vector)) == expected, "\(route) \(dimension)")
                #expect(vector.scalarCount == dimension, "\(route) \(dimension)")
                #expect(!(vector is Vector384Optimized) && !(vector is Vector512Optimized)
                        && !(vector is Vector768Optimized) && !(vector is Vector1536Optimized), "\(route) \(dimension)")
            }
            #expect(made.toArray() == values)
            #expect(zeros.toArray().allSatisfy { $0 == 0 } && ones.toArray().allSatisfy { $0 == 1 })
        }
        #expect(try VectorTypeFactory.vector(of: 512, from: [Float](repeating: 0, count: 512)) is Vector<Dim512>)
        #expect(VectorTypeFactory.zeros(dimension: 4096) is DynamicVector)
    }

    @Test("isSupported(dimension:) is exactly the seven named dimensions (count guard over 0...4096)")
    func supportedListIsTheNamedList() {
        let supported = (-1...4096).filter { VectorTypeFactory.isSupported(dimension: $0) }
        #expect(supported == Self.named)
        #expect(Self.named.allSatisfy { VectorCore.hasOptimizedSupport(for: $0) })
    }

    @Test("The checked entry throws typed errors; createVector delegates without throwing on valid input")
    func checkedEntryThrows() {
        #expect(throws: VectorError.self) { try VectorTypeFactory.vector(of: -1, from: []) }
        #expect(throws: VectorError.self) { try VectorTypeFactory.vector(of: 3, from: [1]) }
        do {
            _ = try VectorTypeFactory.vector(of: -1, from: [])
        } catch let error as VectorError {
            #expect(error.kind == .invalidDimension)
        } catch { Issue.record("unexpected error \(error)") }
        do {
            _ = try VectorTypeFactory.vector(of: 3, from: [1])
        } catch let error as VectorError {
            #expect(error.kind == .dimensionMismatch)
        } catch { Issue.record("unexpected error \(error)") }
        let made: any VectorType = VectorCore.createVector(dimension: 512, data: [Float](repeating: 2, count: 512))
        #expect(made is Vector<Dim512>)
        #expect(VectorCore.createVector(dimension: 5) is DynamicVector)
        #expect([Float](repeating: 1, count: 768).toVector() is Vector<Dim768>)
    }

    @Test func createVectorChildProbe() {
        guard let specification = ProcessInfo.processInfo.environment["D02_CREATE_VECTOR_PROBE"] else { return }
        switch specification {
        case "count-mismatch": _ = VectorCore.createVector(dimension: 3, data: [1])
        case "negative-with-data": _ = VectorCore.createVector(dimension: -1, data: [])
        case "negative-without-data": _ = VectorCore.createVector(dimension: -1)
        default: break
        }
        print("D02_CREATE_VECTOR_RETURNED \(specification)")
    }

    @Test("createVector traps on malformed dimension or data (its documented preconditions; child-isolated)")
    func createVectorTrapsOnMalformedInput() throws {
        if ProcessInfo.processInfo.environment["D02_SKIP_TRAPS"] == "1" { return }
        for specification in ["count-mismatch", "negative-with-data", "negative-without-data"] {
            var environment = ProcessInfo.processInfo.environment
            environment["D02_CREATE_VECTOR_PROBE"] = specification
            environment["SWIFT_BACKTRACE"] = "enable=no"
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("D02-createVector-\(specification)-\(UUID().uuidString).log")
            _ = FileManager.default.createFile(atPath: url.path, contents: nil)
            let handle = try FileHandle(forWritingTo: url)
            let process = try launchChildProbe(.swiftTesting(filter: "D02FactoryContractTests/createVectorChildProbe"),
                                               environment: environment, output: handle, deadline: 15,
                                               cleanup: .terminateThenKill(grace: 1, seconds: 1))
            let timedOut = process.timedOut
            try handle.close()
            let diagnostic = try String(contentsOf: url, encoding: .utf8)
            print("D02_CHILD_CAPTURE \(specification) \(url.path)")
            if timedOut {
                throw NSError(domain: "D02.childTimeout", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "Child deadline exceeded: \(url.path)"])
            }
            #expect(process.terminationReason == .uncaughtSignal, "\(specification): \(diagnostic)")
            #expect(!diagnostic.contains("D02_CREATE_VECTOR_RETURNED"), "\(specification)")
        }
    }
}
