import VectorCoreTestSupport
import Foundation
import Testing
import VectorCore

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
    }

    @Test("The checked factory reports typed validation errors")
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
    }

    @Test("An empty array converts to a zero-dimensional DynamicVector")
    func emptyArrayToVector() {
        let vector: any VectorType = [Float]().toVector()
        #expect(vector is DynamicVector)
        #expect(vector.scalarCount == 0)
        #expect(vector.toArray().isEmpty)
    }

    @Test("An array of a non-specialized width converts to DynamicVector with the original values")
    func dynamicArrayToVector() {
        let values: [Float] = [1.25, -2.5, 4]
        let vector: any VectorType = values.toVector()
        #expect(vector is DynamicVector)
        #expect(vector.toArray().map(\.bitPattern) == values.map(\.bitPattern))
    }

    @Test("Arrays of specialized widths convert to the corresponding fixed-width vector")
    func fixedWidthArrayToVector() {
        for dimension in Self.named {
            let values = (0..<dimension).map { Float($0 % 7) }
            let vector: any VectorType = values.toVector()
            #expect(String(describing: type(of: vector)) == d02ExpectedType(dimension))
            #expect(vector.toArray() == values)
        }
    }

    @Test("Array conversion preserves nonfinite values and signed zero on both factory paths")
    func nonfiniteArrayToVector() {
        let values: [Float] = [Float(bitPattern: 0x7fc0_1234), .infinity, -.infinity, -0.0]
        for dimension in [values.count, 128] {
            let input = values + [Float](repeating: 1, count: dimension - values.count)
            let vector: any VectorType = input.toVector()
            #expect(vector.toArray().map(\.bitPattern) == input.map(\.bitPattern))
        }
    }

    @Test func negativeZerosChildProbe() {
        guard ProcessInfo.processInfo.environment["D02_NEGATIVE_ZEROS_PROBE"] == "1" else { return }
        _ = VectorTypeFactory.zeros(dimension: -1)
        print("D02_NEGATIVE_ZEROS_RETURNED")
    }

    @Test("zeros(dimension:) traps for a negative dimension", .enabled(if: childProbeProcessesSupported, "Requires macOS child-process isolation"))
    func zerosTrapsOnNegativeDimension() throws {
        if ProcessInfo.processInfo.environment["D02_SKIP_TRAPS"] == "1" { return }
        var environment = ProcessInfo.processInfo.environment
        environment["D02_NEGATIVE_ZEROS_PROBE"] = "1"
        environment["SWIFT_BACKTRACE"] = "enable=no"
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("D02-negativeZeros-\(UUID().uuidString).log")
        _ = FileManager.default.createFile(atPath: url.path, contents: nil)
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        let process = try launchChildProbe(
            .swiftTesting(filter: "D02FactoryContractTests/negativeZerosChildProbe"),
            environment: environment,
            output: handle,
            deadline: ChildProbeBudget.standard.deadline,
            cleanup: ChildProbeBudget.standard.cleanup
        )
        let diagnostic = try String(contentsOf: url, encoding: .utf8)
        print("D02_CHILD_CAPTURE negativeZeros \(url.path)")
        if process.timedOut {
            throw NSError(domain: "D02.childTimeout", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Child deadline exceeded: \(url.path)"])
        }
        #expect(process.terminationReason == .uncaughtSignal, "\(diagnostic)")
        #expect(!diagnostic.contains("D02_NEGATIVE_ZEROS_RETURNED"), "\(diagnostic)")
    }
}
