import Testing
import VectorCore

private enum StorageBorrowFailure: Error { case expected }

@Suite struct ReferenceStorageValueTests {
    private func checkCopies<S: VectorStorage & VectorStorageOperations>(_ type: S.Type) throws where S.Scalar == Float {
        let original = S(from: [1, 2, 3, 4])
        var scalarCopy = original
        scalarCopy[1] = 20
        #expect(original.withUnsafeBufferPointer { Array($0) } == [1, 2, 3, 4])
        #expect(scalarCopy.withUnsafeBufferPointer { Array($0) } == [1, 20, 3, 4])

        var bufferCopy = original
        let result = bufferCopy.withUnsafeMutableBufferPointer { buffer in
            buffer[2] = 30
            return buffer.count
        }
        #expect(result == 4)
        #expect(original.withUnsafeBufferPointer { Array($0) } == [1, 2, 3, 4])
        #expect(bufferCopy.withUnsafeBufferPointer { Array($0) } == [1, 2, 30, 4])

        var throwingCopy = original
        #expect(throws: StorageBorrowFailure.self) {
            try throwingCopy.withUnsafeMutableBufferPointer { buffer in
                buffer[3] = 40
                throw StorageBorrowFailure.expected
            }
        }
        #expect(original.withUnsafeBufferPointer { Array($0) } == [1, 2, 3, 4])
        #expect(throwingCopy.withUnsafeBufferPointer { Array($0) } == [1, 2, 3, 40])
        throwingCopy[0] = 10
        #expect(throwingCopy[0] == 10)
        #expect(original.dotProduct(original) == 30)
        #expect(original.dotProduct(scalarCopy) == 66)
        #expect(scalarCopy.dotProduct(original) == 66)

        let empty = S(from: [])
        var emptyCopy = empty
        #expect(emptyCopy.withUnsafeMutableBufferPointer { $0.count } == 0)
        #expect(empty.count == 0 && empty.dotProduct(empty) == 0)
    }

    @Test func dynamicCopies() throws { try checkCopies(DynamicArrayStorage.self) }
    @Test func alignedCopies() throws { try checkCopies(AlignedDynamicArrayStorage.self) }
    @Test func wrapperCopies() throws { try checkCopies(COWDynamicStorage.self) }

    private func concurrentCopies<S: VectorStorage>(_ original: S) async where S.Scalar == Float {
        await withTaskGroup(of: [Float].self) { group in
            for task in 1...16 {
                group.addTask {
                    var local = original
                    for _ in 0..<100 {
                        local[0] = Float(task)
                        local.withUnsafeMutableBufferPointer { buffer in
                            for i in buffer.indices { buffer[i] = Float(task) }
                        }
                    }
                    return local.withUnsafeBufferPointer { Array($0) }
                }
            }
            var values: Set<Float> = []
            for await result in group {
                #expect(result.count == 257 && result.allSatisfy { $0 == result[0] })
                values.insert(result[0])
            }
            #expect(values == Set((1...16).map(Float.init)))
        }
        #expect(original.withUnsafeBufferPointer { $0.allSatisfy { $0 == -1 } })
    }

    @Test func concurrentIndependentCopies() async {
        await concurrentCopies(DynamicArrayStorage(dimension: 257, repeating: -1))
        await concurrentCopies(AlignedDynamicArrayStorage(dimension: 257, repeating: -1))
        await concurrentCopies(COWDynamicStorage(dimension: 257, repeating: -1))
    }

    @Test func alignedCopyRetainsRequestedAlignment() {
        for alignment in [16, 64, 256, 4096] {
            let original = AlignedDynamicArrayStorage(from: [1, 2, 3], alignment: alignment)
            var copy = original
            copy[0] = 9
            #expect(copy.guaranteedAlignment == alignment && copy.verifyAlignment())
            copy.withUnsafeMutableBufferPointer { buffer in
                #expect(Int(bitPattern: buffer.baseAddress!) % alignment == 0)
                buffer[1] = 8
            }
            #expect(original.withUnsafeBufferPointer { Array($0) } == [1, 2, 3])
            #expect(copy.withUnsafeBufferPointer { Array($0) } == [9, 8, 3])
        }
    }
}
