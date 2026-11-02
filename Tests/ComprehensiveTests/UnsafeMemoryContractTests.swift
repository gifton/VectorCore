import Foundation
import Testing
@testable import VectorCore

@Suite struct UnsafeMemoryContractTests {
    final class Ledger { var deaths: [Int: Int] = [:] }
    final class Token {
        let id: Int
        let ledger: Ledger
        init(_ id: Int, _ ledger: Ledger) { self.id = id; self.ledger = ledger }
        deinit { ledger.deaths[id, default: 0] += 1 }
    }
    struct Payload { let token: Token; let text: String; let values: [Int] }

    @Test func initializedCopyLifetimes() throws {
        for sourceOffset in [0, 1] {
            for destinationOffset in [0, 1] {
                let ledger = Ledger()
                let source = try AlignedMemory.allocateAligned(type: Payload.self, count: 4)
                let destination = try AlignedMemory.allocateAligned(type: Payload.self, count: 4)
                let s = source + sourceOffset, d = destination + destinationOffset
                s.initialize(to: Payload(token: Token(1, ledger), text: "source", values: [2, 3]))
                d.initialize(to: Payload(token: Token(2, ledger), text: "old", values: [4]))
                alignedCopy(from: s, to: d, count: 0)
                #expect(ledger.deaths.isEmpty)
                alignedCopy(from: s, to: d, count: 1)
                #expect(ledger.deaths[2] == 1)
                #expect(d.pointee.text == "source" && d.pointee.values == [2, 3])
                s.deinitialize(count: 1)
                #expect(ledger.deaths[1] == nil)
                d.deinitialize(count: 1)
                #expect(ledger.deaths[1] == 1)
                AlignedMemory.deallocate(source); AlignedMemory.deallocate(destination)
            }
        }
    }

    @Test func pointerOracle() throws {
        for offset in 0...3 {
            for dimension in Array(0...5) + [15, 16, 17, 31, 32, 33] {
                let p = try AlignedMemory.allocateAligned(count: 64)
                p.initialize(repeating: -12345, count: 64)
                defer { p.deinitialize(count: 64); AlignedMemory.deallocate(p) }
                let start = 16 + offset
                let values: [Float] = (0..<dimension).map { i in
                    let magnitude = Float((i % 7) + 1)
                    return i % 2 == 0 ? magnitude : -magnitude
                }
                for i in 0..<dimension { p[start + i] = values[i] }
                NormalizeKernels.normalizeUnchecked(p + start, dimension: dimension)
                let norm = sqrt(values.reduce(0.0) { $0 + Double($1) * Double($1) })
                var resultSquared = 0.0
                for i in 0..<dimension {
                    #expect(p[start + i].isFinite)
                    // 33 ordinary coordinates: 2e-6 exceeds accumulated Float rounding.
                    #expect(abs(Double(p[start + i]) - Double(values[i]) / norm) < 2e-6)
                    resultSquared += Double(p[start + i]) * Double(p[start + i])
                }
                if dimension > 0 { #expect(abs(sqrt(resultSquared) - 1) < 2e-6) }
                for i in 0..<start { #expect(p[i].bitPattern == Float(-12345).bitPattern) }
                for i in (start + dimension)..<64 { #expect(p[i].bitPattern == Float(-12345).bitPattern) }
            }
        }
    }
}

extension UnsafeMemoryContractTests {
    final class BorrowError: Error, @unchecked Sendable {}

    private func roundTrip<V: VectorProtocol>(_ type: V.Type, width: Int) throws
    where V.Scalar == Float, V.Storage == ContiguousArray<SIMD4<Float>> {
        var source = (0..<width).map { Float($0 % 19 - 9) }
        var vector = try V(source)
        let copy = vector
        for i in 0..<(width / 4) {
            for lane in 0..<4 { #expect(vector.storage[i][lane].bitPattern == source[4 * i + lane].bitPattern) }
        }
        #expect(vector.toArray() == source)
        let error = BorrowError()
        do {
            try vector.withUnsafeBufferPointer { buffer in
                #expect(Array(buffer) == source)
                throw error
            }
            Issue.record("Expected borrow error")
        } catch let caught { let sameError: Bool = (caught as AnyObject) === error; #expect(sameError) }
        do {
            try vector.withUnsafeMutableBufferPointer { buffer in
                buffer[0] = 42
                buffer[width - 1] = 99
                throw error
            }
            Issue.record("Expected borrow error")
        } catch let caught { let sameError: Bool = (caught as AnyObject) === error; #expect(sameError) }
        #expect(vector.storage[0][0].bitPattern == Float(42).bitPattern)
        #expect(vector.storage[width / 4 - 1][3].bitPattern == Float(99).bitPattern)
        #expect(copy.toArray() == source)
        source[0] = 88
        #expect(copy[0].bitPattern == Float(-9).bitPattern)
        #expect(vector[0].bitPattern == Float(42).bitPattern)
    }

    @Test func allWidthScopedBorrows() throws {
        #expect(MemoryLayout<SIMD4<Float>>.size == 4 * MemoryLayout<Float>.stride)
        #expect(MemoryLayout<SIMD4<Float>>.stride == MemoryLayout<SIMD4<Float>>.size)
        #expect(MemoryLayout<SIMD4<Float>>.alignment >= MemoryLayout<Float>.alignment)
        try roundTrip(Vector384Optimized.self, width: 384)
        try roundTrip(Vector512Optimized.self, width: 512)
        try roundTrip(Vector768Optimized.self, width: 768)
        try roundTrip(Vector1536Optimized.self, width: 1536)
    }
}

extension UnsafeMemoryContractTests {
    @Test func allocatorValuesAndCounts() throws {
        func check<T>(_ value: T) throws {
            for count in [0, 1, 3, 17, 257] {
                let p = try AlignedMemory.allocateAligned(type: T.self, count: count, alignment: 64)
                #expect(Int(bitPattern: p) % 64 == 0)
                p.initialize(repeating: value, count: count)
                p.deinitialize(count: count)
                AlignedMemory.deallocate(p)
            }
        }
        try check(Float(3)); try check(SIMD4<Float>(1, 2, 3, 4)); try check(SIMD8<Double>(repeating: 7))
        let ledger = Ledger()
        try check(Payload(token: Token(3, ledger), text: "owned", values: [1, 3]))
        #expect(ledger.deaths[3] == 1)
        for count in [-1, Int.max] {
            do {
                let p = try AlignedMemory.allocateAligned(type: SIMD4<Float>.self, count: count)
                AlignedMemory.deallocate(p)
                Issue.record("Invalid count accepted")
            } catch let error as VectorError { #expect(error.kind == .invalidDimension) }
        }
    }

    @Test func publicStorageControls() {
        for count in [0, 1, 3, 17, 257] {
            for alignment in [16, 32, 64, 256, 4096] {
                let values = (0..<count).map(Float.init)
                let storages = [AlignedDynamicArrayStorage(dimension: count, alignment: alignment),
                                AlignedDynamicArrayStorage(dimension: count, repeating: 7, alignment: alignment),
                                AlignedDynamicArrayStorage(from: values, alignment: alignment),
                                AlignedDynamicArrayStorage(from: values)]
                for (index, storage) in storages.enumerated() {
                    #expect(storage.count == count)
                    #expect(storage.guaranteedAlignment == (index == 3 ? AlignedMemory.optimalAlignment : alignment))
                    #expect(storage.verifyAlignment())
                    storage.withUnsafeBufferPointer { buffer in
                        #expect(Int(bitPattern: buffer.baseAddress!) % storage.guaranteedAlignment == 0)
                        let expected = index == 0 ? Array(repeating: Float(0), count: count) : index == 1 ? Array(repeating: Float(7), count: count) : values
                        #expect(Array(buffer) == expected)
                    }
                }
            }
        }
    }
}

extension UnsafeMemoryContractTests {
    @Test func forcedPrimaryFailurePreservesAlignment() {
        for count in [0, 1, 3, 17, 257] {
            for alignment in [16, 32, 64, 256, 4096] {
                var calls = 0
                weak var released: AlignedDynamicArrayStorage?
                do {
                    let storage = AlignedDynamicArrayStorage(dimension: count, repeating: 9, alignment: alignment, primaryAllocator: { n, a in
                        calls += 1
                        #expect(n == count && a == alignment)
                        throw VectorError.allocationFailed(size: 0, reason: "O04 injected primary failure")
                    })
                    released = storage
                    #expect(calls == 1)
                    #expect(!storage.usedPosixAlignedAlloc)
                    #expect(storage.guaranteedAlignment == alignment && storage.verifyAlignment())
                    storage.withUnsafeMutableBufferPointer { buffer in
                        #expect(buffer.count == count)
                        #expect(Int(bitPattern: buffer.baseAddress!) % alignment == 0)
                        for i in buffer.indices { #expect(buffer[i].bitPattern == Float(9).bitPattern); buffer[i] = Float(i) }
                    }
                    #expect(storage.toArray() == (0..<count).map(Float.init))
                }
                #expect(released == nil)
            }
        }
    }

    @Test func alignmentPredicates() throws {
        let p = try AlignedMemory.allocateAligned(count: 1)
        defer { AlignedMemory.deallocate(p) }
        for boundary in [-64, 0, 1, 3, 16, 64] {
            let expected = boundary > 0 && Int(bitPattern: p) % boundary == 0
            #expect(isAligned(UnsafePointer(p), to: boundary) == expected)
            #expect(AlignedMemory.isAligned(p, to: boundary) == expected)
            #expect(AlignedMemory.isAligned(UnsafePointer(p), to: boundary) == expected)
        }
    }
}

extension UnsafeMemoryContractTests {
    @Test func invalidChildProbe() throws {
        guard let spec = ProcessInfo.processInfo.environment["O04_INVALID"] else { return }
        let p = try AlignedMemory.allocateAligned(count: 1)
        p.initialize(to: 1)
        defer { p.deinitialize(count: 1); AlignedMemory.deallocate(p) }
        switch spec {
        case "normalize": NormalizeKernels.normalizeUnchecked(p, dimension: -1)
        case "copy-count": alignedCopy(from: p, to: p, count: -1)
        case "copy-zero-hint": alignedCopy(from: p, to: p, count: 0, preferredAlignment: 0)
        case "copy-negative-hint": alignedCopy(from: p, to: p, count: 0, preferredAlignment: -1)
        case "alloc-zero": _ = try AlignedMemory.allocateAligned(count: 0, alignment: 0)
        case "alloc-small": _ = try AlignedMemory.allocateAligned(count: 0, alignment: 8)
        case "alloc-odd": _ = try AlignedMemory.allocateAligned(count: 0, alignment: 24)
        case "storage-negative": _ = AlignedDynamicArrayStorage(dimension: -1)
        case "storage-overflow": _ = AlignedDynamicArrayStorage(dimension: Int.max, repeating: 1)
        case "storage-zero-alignment": _ = AlignedDynamicArrayStorage(from: [], alignment: 0)
        case "storage-odd-alignment": _ = AlignedDynamicArrayStorage(dimension: 0, alignment: 24)
        default: Issue.record("Unknown invalid probe")
        }
        print("O04_INVALID_ACCEPTED")
    }

    @Test func invalidInputsRejectInBoundedChildren() throws {
        for spec in ["normalize", "copy-count", "copy-zero-hint", "copy-negative-hint", "alloc-zero", "alloc-small", "alloc-odd", "storage-negative", "storage-overflow",
            "storage-zero-alignment", "storage-odd-alignment"] {
            let environment = ProcessInfo.processInfo.environment
                .merging(["O04_INVALID": spec, "SWIFT_BACKTRACE": "enable=no"]) { _, new in new }
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("O04-child-\(UUID().uuidString).log")
            _ = FileManager.default.createFile(atPath: url.path, contents: nil)
            let output = try FileHandle(forWritingTo: url)
            defer { try? output.close() }
            let process = try launchChildProbe(.swiftTesting(filter: "UnsafeMemoryContractTests/invalidChildProbe"),
                                               environment: environment, output: output, deadline: 15, cleanup: .kill(seconds: 2))
            if process.timedOut {
                throw NSError(domain: "UnsafeMemoryContractTests.childTimeout", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "\(spec): child deadline; cleanup=\(process.cleanup); log=\(url.path)"])
            }
            let diagnostic = try String(contentsOf: url, encoding: .utf8)
            #expect(process.terminationReason == .uncaughtSignal, "\(spec): \(diagnostic)")
            let phrase: String
            switch spec {
            case "normalize": phrase = "normalizeUnchecked: dimension must be nonnegative"
            case "storage-negative": phrase = "Dimension must be nonnegative"
            case "copy-count": phrase = "Count must be nonnegative"
            case "copy-zero-hint", "copy-negative-hint": phrase = "Preferred alignment must be positive"
            case "storage-overflow": phrase = "Dimension byte count must be representable"
            case "storage-zero-alignment", "storage-odd-alignment": phrase = "Alignment must be a power of two and at least 16 bytes"
            case "alloc-small": phrase = "Alignment must be at least 16 bytes"
            default: phrase = "Alignment must be a power of 2"
            }
            if !spec.hasPrefix("alloc-") {
                #expect(diagnostic.contains(phrase), "\(spec): \(diagnostic)")
            }
            #if DEBUG
            if spec.hasPrefix("alloc-") { #expect(diagnostic.contains(phrase), "\(spec): \(diagnostic)") }
            #endif
            #expect(!diagnostic.contains("O04_INVALID_ACCEPTED"))
            print("O04 child \(spec): \(url.path)")
        }
    }
}

extension UnsafeMemoryContractTests {
    @Test func referenceCopyReleasesEveryReplacedOwner() throws {
        for sourceOffset in [0, 1] {
            for destinationOffset in [0, 1] {
                let ledger = Ledger()
                let source = try AlignedMemory.allocateAligned(type: Token.self, count: 5)
                let destination = try AlignedMemory.allocateAligned(type: Token.self, count: 5)
                let s = source + sourceOffset, d = destination + destinationOffset
                for i in 0..<3 { (s + i).initialize(to: Token(i, ledger)); (d + i).initialize(to: Token(10 + i, ledger)) }
                weak var old = d.pointee
                weak var retained = s.pointee
                alignedCopy(from: s, to: d, count: 3)
                #expect(old == nil)
                for i in 0..<3 { #expect(ledger.deaths[10 + i] == 1); #expect(d[i].id == i) }
                s.deinitialize(count: 3)
                #expect(retained != nil)
                d.deinitialize(count: 3)
                #expect(retained == nil)
                for i in 0..<3 { #expect(ledger.deaths[i] == 1) }
                AlignedMemory.deallocate(source); AlignedMemory.deallocate(destination)
            }
        }
    }
}
