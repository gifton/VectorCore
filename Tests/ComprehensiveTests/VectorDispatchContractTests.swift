import Foundation
import Testing
@testable import VectorCore
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif

private enum S02Sentinel: Error { case factory }

// Stops at the first payload access, so baseline short-RHS probes never overread.
private func s02Access(_ marker: StaticString) -> Never {
    marker.withUTF8Buffer { _ = write(STDERR_FILENO, $0.baseAddress, $0.count) }
    exit(86)
}

private struct S02Spy: VectorProtocol, VectorFactory {
    typealias Scalar = Float
    var storage: [Float]
    var dimension: Int
    var stopOnAccess: Bool = false
    subscript(index: Int) -> Float {
        if stopOnAccess { s02Access("S02_PAYLOAD_ACCESSED\n") }
        return storage[index]
    }
    var scalarCount: Int { dimension }
    init() { storage = []; dimension = 0 }
    init(repeating value: Float) { storage = [value]; dimension = 1 }
    init(_ array: [Float]) { storage = array; dimension = array.count }
    init(dimension: Int, stop: Bool) {
        storage = []; self.dimension = dimension; stopOnAccess = stop
    }
    func toArray() -> [Float] {
        if stopOnAccess { s02Access("S02_PAYLOAD_ACCESSED\n") }
        return storage
    }
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        if stopOnAccess { s02Access("S02_READ_BORROW\n") }
        return try storage.withUnsafeBufferPointer(body)
    }
    mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        if stopOnAccess { s02Access("S02_MUTABLE_BORROW\n") }
        return try storage.withUnsafeMutableBufferPointer { try body($0) }
    }
    static func create(from scalars: [Float]) throws -> Self { throw S02Sentinel.factory }
}

// No VectorFactory conformance: concrete operators resolve to VectorProtocol defaults.
private struct S02ProtocolOnly: VectorProtocol {
    typealias Scalar = Float
    var storage: [Float]
    var stopOnBorrow = false
    subscript(index: Int) -> Float {
        if stopOnBorrow { s02Access("S02_READ_BORROW\n") }
        return storage[index]
    }
    var scalarCount: Int { storage.count }
    init() { storage = [] }
    init(repeating value: Float) { storage = [value] }
    init(_ array: [Float]) { storage = array }
    func toArray() -> [Float] { storage }
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        if stopOnBorrow { s02Access("S02_READ_BORROW\n") }
        return try storage.withUnsafeBufferPointer(body)
    }
    mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        if stopOnBorrow { s02Access("S02_MUTABLE_BORROW\n") }
        return try storage.withUnsafeMutableBufferPointer { try body($0) }
    }
}

@Suite("Vector dispatch and dimension contracts", .serialized)
struct VectorDispatchContractTests {
    private func mutate<V: VectorProtocol>(_ lhs: inout V, _ rhs: V, route: String) {
        switch route {
        case "add": lhs.inPlaceAdd(rhs)
        case "subtract": lhs.inPlaceSubtract(rhs)
        case "multiply": lhs.inPlaceElementwiseMultiply(rhs)
        case "plus": lhs += rhs
        case "minus": lhs -= rhs
        default: preconditionFailure("Unknown route")
        }
    }

    private func direct(_ lhs: inout DynamicVector, _ rhs: DynamicVector, route: String) {
        switch route {
        case "add": lhs.inPlaceAdd(rhs)
        case "subtract": lhs.inPlaceSubtract(rhs)
        case "multiply": lhs.inPlaceElementwiseMultiply(rhs)
        case "plus": lhs += rhs // Concrete VectorFactory overload.
        case "minus": lhs -= rhs
        default: preconditionFailure("Unknown route")
        }
    }

    private func child(_ specification: String) throws -> (Process.TerminationReason, Int32, String) {
        var environment = ProcessInfo.processInfo.environment
        environment["S02_PROBE"] = specification
        environment["SWIFT_BACKTRACE"] = "enable=no"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("S02-child-\(UUID().uuidString).log")
        _ = FileManager.default.createFile(atPath: url.path, contents: nil)
        let output = try FileHandle(forWritingTo: url)
        defer { try? output.close(); try? FileManager.default.removeItem(at: url) }
        // The launcher signals only the child's own process group (never this process's group).
        let process = try launchChildProbe(.swiftTesting(filter: "VectorDispatchContractTests/childProbe"),
                                           environment: environment, output: output, deadline: 15, cleanup: .kill(seconds: 2))
        let diagnostic = try String(contentsOf: url, encoding: .utf8)
        try #require(!process.timedOut, "\(specification): child exceeded 15-second deadline; \(diagnostic)")
        let termination = try process.reaped()
        return (termination.reason, termination.status, diagnostic)
    }

    private func mutationProbe(_ p: [String]) {
        let n = Int(p[3])!, m = Int(p[4])!
        if p[1] == "plain" {
            var lhs = S02ProtocolOnly(Array(repeating: 2, count: n))
            lhs.stopOnBorrow = true
            let rhs = S02ProtocolOnly(Array(repeating: 3, count: m))
            if p[2] == "plus" { lhs += rhs }
            else { lhs -= rhs }
        } else if p[1] == "spy" {
            var lhs = S02Spy(dimension: n, stop: true)
            mutate(&lhs, S02Spy(dimension: m, stop: true), route: p[2])
        } else {
            var lhs = DynamicVector(Array(repeating: 2, count: n))
            let rhs = DynamicVector(Array(repeating: 3, count: m))
            if p[1] == "direct" { direct(&lhs, rhs, route: p[2]) }
            else { mutate(&lhs, rhs, route: p[2]) }
        }
    }

    @Test func childProbe() async throws {
        guard let specification = ProcessInfo.processInfo.environment["S02_PROBE"] else { return }
        let p = specification.split(separator: ":").map(String.init)
        switch p[0] {
        case "mutation":
            mutationProbe(p)
        case "aggregate":
            var rows = [S02Spy](repeating: S02Spy(dimension: 2, stop: true), count: 3)
            rows[Int(p[2])!] = S02Spy(dimension: Int(p[3])!, stop: true)
            do {
                if p[1] == "centroid" { _ = try Operations.centroid(of: rows) }
                else { _ = try Operations.statistics(for: rows) }
            } catch let error as VectorError {
                print("S02_ERROR_\(error.kind.rawValue)"); return
            }
        case "array":
            var row = S02Spy([1, 2, 3]); row.dimension = 2
            do {
                if p[1] == "centroid" { _ = try Operations.centroid(of: [row]) }
                else { _ = try Operations.statistics(for: [row]) }
            } catch let error as VectorError { print("S02_ERROR_\(error.kind.rawValue)"); return }
        case "zero":
            _ = try Operations.centroid(of: [DynamicVector([])])
        case "basis":
            do { _ = try VectorTypeFactory.basis(dimension: Int(p[1])!, index: 0) }
            catch let error as VectorError { print("S02_ERROR_\(error.kind.rawValue)"); return }
        case "sentinel":
            do {
                if p[1] == "centroid" { _ = try Operations.centroid(of: [S02Spy([3, 4])]) }
                else { _ = try await Operations.normalize([S02Spy([3, 4])]) }
            } catch S02Sentinel.factory { print("S02_SENTINEL"); return }
        case "emptyCentroid":
            do {
                if p[1] == "fixed" { _ = try Operations.centroid(of: [Vector<Dim2>]()) }
                else { _ = try Operations.centroid(of: [DynamicVector]()) }
            } catch let error as VectorError { print("S02_ERROR_\(error.kind.rawValue)"); return }
        case "batch":
            do { _ = try VectorTypeFactory.batch(dimension: Int(p[1])!, from: Array(repeating: 1, count: Int(p[2])!)) }
            catch let error as VectorError { print("S02_ERROR_\(error.kind.rawValue)"); return }
        case "constructor":
            let n = Int(p[2])!
            switch p[1] {
            case "random": _ = VectorTypeFactory.random(dimension: n)
            case "zeros": _ = VectorTypeFactory.zeros(dimension: n)
            case "ones": _ = VectorTypeFactory.ones(dimension: n)
            default: _ = VectorTypeFactory.withPattern(dimension: n) { _ in s02Access("S02_PATTERN_ACCESSED\n") }
            }
        case "errorHelper":
            let error = VectorError.indexOutOfBounds(index: Int.min, dimension: Int.min)
            print("S02_ERROR_HELPER_\(error.context.additionalInfo["max_index"] ?? "missing")")
            return
        case "lookup": print("S02_LOOKUP_\(VectorTypeFactory.optimalDimension(for: Int.min))"); return
        default: Issue.record("Unknown probe")
        }
        print("S02_ACCEPTED")
    }

    // Removing any binary preflight is caught before a short RHS can be borrowed.
    @Test func mutationBoundaryMatrix() throws {
        for route in ["add", "subtract", "multiply", "plus", "minus"] {
            for n in 0...3 {
                for m in 0...3 where n != m {
                    // Short-RHS baseline uses the access-stopping conformer only.
                    let modes = m > n ? ["spy", "direct", "generic"] : ["spy"]
                    for mode in modes {
                        let spec = "mutation:\(mode):\(route):\(n):\(m)"
                        let (reason, _, output) = try child(spec)
                        #expect(reason == .uncaughtSignal, "\(spec): \(output)")
                        let factoryRoute = mode == "direct" && ["plus", "minus"].contains(route)
                        #expect(output.lowercased().contains(factoryRoute ? "dimension" : "dimension mismatch"), "\(spec): \(output)")
                        #expect(!output.contains("S02_MUTABLE_BORROW") && !output.contains("S02_READ_BORROW") && !output.contains("S02_ACCEPTED"), "\(spec): \(output)")
                    }
                }
            }
        }
    }

    @Test func validMutationArithmeticAndCopies() {
        for n in [0, 1, 2, 3, 128, 2048] {
            let a = (0..<n).map { Float($0 % 7 - 3) }
            let b = (0..<n).map { Float($0 % 5 + 1) }
            for route in ["add", "subtract", "multiply", "plus", "minus"] {
                let expected = zip(a, b).map { x, y in route == "multiply" ? x * y : (["add", "plus"].contains(route) ? x + y : x - y) }
                for mode in ["direct", "generic"] {
                    var lhs = DynamicVector(a)
                    let copy = lhs
                    if mode == "direct" { direct(&lhs, DynamicVector(b), route: route) }
                    else { mutate(&lhs, DynamicVector(b), route: route) }
                    #expect(lhs.toArray() == expected)
                    #expect(copy.toArray() == a)
                }
            }
        }
    }

    @Test func aggregateMetadataPreflight() throws {
        for operation in ["centroid", "statistics"] {
            for position in 0...2 {
                for size in [1, 3] {
                    let spec = "aggregate:\(operation):\(position):\(size)"
                    let (_, status, output) = try child(spec)
                    #expect(status == 0 && output.contains("S02_ERROR_dimensionMismatch"), "\(spec): \(output)")
                    #expect(!output.contains("S02_PAYLOAD_ACCESSED") && !output.contains("S02_READ_BORROW"), "\(spec): \(output)")
                }
            }
        }
    }

    @Test func genericFactoryFailuresPropagate() throws {
        for operation in ["centroid", "normalize"] {
            let (_, status, output) = try child("sentinel:\(operation)")
            #expect(status == 0 && output.contains("S02_SENTINEL"), "\(operation): \(output)")
        }
        for kind in ["fixed", "dynamic"] {
            let (_, status, output) = try child("emptyCentroid:\(kind)")
            #expect(status == 0 && output.contains("S02_ERROR_insufficientData"), "\(output)")
        }
    }

    @Test func invalidBatchDimensions() throws {
        for dimension in [0, -1, Int.min] {
            for count in [0, 1] {
                let (_, status, output) = try child("batch:\(dimension):\(count)")
                #expect(status == 0 && output.contains("S02_ERROR_invalidDimension"), "\(output)")
            }
        }
    }

    @Test func negativeNonthrowingConstructors() throws {
        for constructor in ["random", "zeros", "ones", "pattern"] {
            for dimension in [-1, Int.min] {
                let (reason, _, output) = try child("constructor:\(constructor):\(dimension)")
                #expect(reason == .uncaughtSignal && output.contains("Dimension must be nonnegative"), "\(output)")
                #expect(!output.contains("S02_PATTERN_ACCESSED"))
            }
        }
    }

    @Test func extremeLookupSeparateProbe() throws {
        let (_, status, output) = try child("lookup")
        #expect(status == 0 && output.contains("S02_LOOKUP_128"), "\(output)")
    }

    @Test func throwingFactoryValidationAndValidBatches() throws {
        for dimension in [-1, Int.min] {
            for values: [Float] in [[], [1]] {
                #expect(throws: (any Error).self) { _ = try VectorTypeFactory.vector(of: dimension, from: values) }
                do { _ = try VectorTypeFactory.vector(of: dimension, from: values) }
                catch let error as VectorError { #expect(error.kind == .invalidDimension) }
            }
        }
        for dimension in [0, -1, Int.min] {
            for index in [-1, 0, Int.max] {
                do { _ = try VectorTypeFactory.basis(dimension: dimension, index: index); Issue.record("Expected invalid dimension") }
                catch let error as VectorError { #expect(error.kind == .invalidDimension) }
            }
        }
        #expect(try VectorTypeFactory.vector(of: 0, from: []).scalarCount == 0)
        #expect(throws: VectorError.self) { _ = try VectorTypeFactory.vector(of: 0, from: [1]) }
        for dimension in [1, 2, 4, 128, 512] {
            for count in [0, dimension, dimension * 3] {
                let values = (0..<count).map { Float($0 % 11) }
                let batch = try VectorTypeFactory.batch(dimension: dimension, from: values)
                #expect(batch.count == count / dimension)
                #expect(batch.flatMap { $0.toArray() } == values)
                #expect(batch.allSatisfy { $0.scalarCount == dimension })
            }
            if dimension > 1 {
                do { _ = try VectorTypeFactory.batch(dimension: dimension, from: [1]); Issue.record("Expected invalid data") }
                catch let error as VectorError { #expect(error.kind == .invalidData) }
            }
            let basis = try VectorTypeFactory.basis(dimension: dimension, index: dimension - 1)
            #expect(basis.toArray() == Array(repeating: 0, count: dimension - 1) + [1])
            for index in [-1, dimension] {
                do { _ = try VectorTypeFactory.basis(dimension: dimension, index: index); Issue.record("Expected invalid index") }
                catch let error as VectorError { #expect(error.kind == .indexOutOfBounds) }
            }
        }
        var calls = 0
        #expect(VectorTypeFactory.withPattern(dimension: 0) { _ in calls += 1; return 1 }.scalarCount == 0)
        #expect(calls == 0)
        #expect(VectorTypeFactory.random(dimension: 0).scalarCount == 0)
        #expect(VectorTypeFactory.zeros(dimension: 0).scalarCount == 0)
        #expect(VectorTypeFactory.ones(dimension: 0).scalarCount == 0)
    }

    @Test func boundedLookupTable() {
        let cases: [(Int, Int)] = [(Int.max, 3072), (-1, 128), (0, 128), (128, 128), (256, 256), (384, 384), (512, 512), (768, 768), (1536, 1536), (3072, 3072), (191, 128),
            (192, 128), (193, 256), (319, 256), (320, 256), (321, 384), (447, 384), (448, 384), (449, 512), (639, 512), (640, 512), (641, 768), (1151, 768), (1152, 768),
            (1153, 1536), (2303, 1536), (2304, 1536), (2305, 3072)]
        for (input, expected) in cases { #expect(VectorTypeFactory.optimalDimension(for: input) == expected) }
    }

    // Run after preflight repair; baseline short-RHS evidence uses S02Spy instead.
    @Test func concreteShortRHSRejection() throws {
        for route in ["add", "subtract", "multiply", "plus", "minus"] {
            for n in 1...3 {
                for m in 0..<n {
                    for mode in ["direct", "generic"] {
                        let spec = "mutation:\(mode):\(route):\(n):\(m)"
                        let (reason, _, output) = try child(spec)
                        #expect(reason == .uncaughtSignal, "\(spec): \(output)")
                        #expect(output.lowercased().contains("dimension"), "\(spec): \(output)")
                        #expect(!output.contains("S02_ACCEPTED"))
                    }
                }
            }
        }
    }

    private func near(_ actual: Float, _ expected: Double) {
        #expect(actual.isFinite && expected.isFinite)
        #expect(abs(Double(actual) - expected) <= 0.00001 * max(1, abs(expected)))
    }

    private func aggregateReference<V: VectorProtocol & VectorFactory>(_ rows: [V], arrays: [[Float]]) throws where V.Scalar == Float {
        let centroid = try Operations.centroid(of: rows)
        let stats = try Operations.statistics(for: rows)
        let dimension = arrays[0].count
        #expect(centroid.scalarCount == dimension)
        #expect(stats.count == arrays.count && stats.dimensions == dimension)
        #expect(stats.mean.count == dimension && stats.min.count == dimension && stats.max.count == dimension)
        for i in 0..<dimension {
            let column = arrays.map { Double($0[i]) }
            let mean = column.reduce(0, +) / Double(arrays.count)
            near(centroid[i], mean); near(stats.mean[i], mean)
            near(stats.min[i], column.min()!); near(stats.max[i], column.max()!)
        }
        let magnitudes = arrays.map { $0.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot() }
        near(stats.magnitudes.min, magnitudes.min()!)
        near(stats.magnitudes.max, magnitudes.max()!)
        near(stats.magnitudes.mean, magnitudes.reduce(0, +) / Double(arrays.count))
    }

    @Test func validAggregatesAndEmptyPolicies() async throws {
        for arrays: [[Float]] in [[[3, 4]], [[-2, 4], [2, 0], [6, -4]], [[], [], []]] {
            try aggregateReference(arrays.map(DynamicVector.init), arrays: arrays)
        }
        try aggregateReference([Vector<Dim2>(x: 3, y: 4)], arrays: [[3, 4]])
        let a = Array(repeating: Float(1), count: 512)
        let b = Array(repeating: Float(3), count: 512)
        try aggregateReference([try Vector<Dim512>(a), try Vector<Dim512>(b)], arrays: [a, b])
        let optimized = try Operations.statistics(for: [try Vector512Optimized(a), try Vector512Optimized(b)])
        #expect(optimized.count == 2 && optimized.dimensions == 512)
        for mean in optimized.mean { near(mean, 2) }
        near(optimized.magnitudes.mean, 2 * Double(512).squareRoot())
        for stats in [try Operations.statistics(for: [DynamicVector]()), try Operations.statistics(for: [Vector<Dim2>]())] {
            #expect(stats.count == 0 && stats.dimensions == 0)
            #expect(stats.mean.isEmpty && stats.min.isEmpty && stats.max.isEmpty)
            near(stats.magnitudes.min, 0); near(stats.magnitudes.max, 0); near(stats.magnitudes.mean, 0)
        }
        let normalized = try await Operations.normalize([DynamicVector([3, 4]), DynamicVector([2])])
        #expect(normalized.map(\.scalarCount) == [2, 1])
        near(normalized[0][0], 0.6); near(normalized[0][1], 0.8); near(normalized[1][0], 1)
        let empty = try await Operations.normalize([DynamicVector]())
        #expect(empty.isEmpty)
    }

    @Test func recoverableAggregateDimensionsAndArrayCounts() throws {
        for operation in ["centroid", "statistics"] {
            for position in 0...2 {
                for size in [0, 1, 3] {
                    var rows = Array(repeating: DynamicVector([1, 2]), count: 3)
                    rows[position] = DynamicVector(Array(repeating: 1, count: size))
                    do {
                        if operation == "centroid" { _ = try Operations.centroid(of: rows) }
                        else { _ = try Operations.statistics(for: rows) }
                        Issue.record("Expected dimension mismatch")
                    } catch let error as VectorError { #expect(error.kind == .dimensionMismatch) }
                }
                for actual in [0, 1, 3] {
                    var rows = Array(repeating: S02Spy([1, 2]), count: 3)
                    rows[position].storage = Array(repeating: 1, count: actual)
                    do {
                        if operation == "centroid" { _ = try Operations.centroid(of: rows) }
                        else { _ = try Operations.statistics(for: rows) }
                        Issue.record("Expected actual array count mismatch")
                    } catch let error as VectorError { #expect(error.kind == .dimensionMismatch) }
                }
            }
        }
    }

    @Test func arrayBoundaryAndZeroCentroidProbes() throws {
        for operation in ["centroid", "statistics"] {
            let (_, status, output) = try child("array:\(operation)")
            #expect(status == 0 && output.contains("S02_ERROR_dimensionMismatch"), "\(output)")
        }
        let (_, status, output) = try child("zero")
        #expect(status == 0 && output.contains("S02_ACCEPTED"), "\(output)")
        for dimension in [0, -1, Int.min] {
            let (_, status, output) = try child("basis:\(dimension)")
            #expect(status == 0 && output.contains("S02_ERROR_invalidDimension"), "\(output)")
        }
    }

    @Test func boundsErrorExtremeProbe() throws {
        let (_, status, output) = try child("errorHelper")
        #expect(status == 0 && output.contains("S02_ERROR_HELPER_-1"), "\(output)")
    }

    @Test func boundsErrorMetadataTable() {
        let dimensions: [(Int, Int)] = [(Int.min, -1), (-1, -1), (0, -1), (1, 0), (Int.max, Int.max - 1)]
        for (dimension, maximum) in dimensions {
            for index in [Int.min, -1, 0, Int.max] {
                let error = VectorError.indexOutOfBounds(index: index, dimension: dimension)
                #expect(error.kind == .indexOutOfBounds)
                #expect(error.errorCode == 1100)
                #expect(error.context.additionalInfo["index"] == String(index))
                #expect(error.context.additionalInfo["max_index"] == String(maximum))
                #expect(error.context.additionalInfo["message"] == "Index \(index) is out of bounds for dimension \(dimension)")
            }
        }
    }

    @Test func concreteProtocolOnlyOperators() throws {
        for route in ["plus", "minus"] {
            for n in 0...3 {
                for m in 0...3 where n != m {
                    let (reason, _, output) = try child("mutation:plain:\(route):\(n):\(m)")
                    #expect(reason == .uncaughtSignal && output.contains("Dimension mismatch"), "\(output)")
                    #expect(!output.contains("S02_MUTABLE_BORROW") && !output.contains("S02_READ_BORROW"))
                }
            }
            for n in [0, 1, 2, 3, 2048] {
                var lhs = S02ProtocolOnly(Array(repeating: 2, count: n))
                let copy = lhs
                let rhs = S02ProtocolOnly(Array(repeating: 3, count: n))
                if route == "plus" { lhs += rhs }
                else { lhs -= rhs }
                #expect(lhs.toArray() == Array(repeating: route == "plus" ? 5 : -1, count: n))
                #expect(copy.toArray() == Array(repeating: 2, count: n))
            }
        }
    }

}
