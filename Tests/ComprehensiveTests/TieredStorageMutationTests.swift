import Testing
@testable import VectorCore

@Suite("TieredStorageMutationTests")
struct TieredStorageMutationTests {
    private final class BorrowError: Error {}
    private func address<T>(_ value: TieredStorage<T>) -> UInt {
        value.withUnsafeBufferPointer { UInt(bitPattern: $0.baseAddress!) }
    }

    @Test(arguments: [0, 1, 4, 5, 512, 513, 2048, 2049], [false, true])
    func normalBulkWriteback(count: Int, shared: Bool) {
        var value = TieredStorage(Array(0..<count))
        let original: TieredStorage<Int>? = shared ? value : nil
        var iterator = original?.makeIterator()
        let result = value.withUnsafeMutableBufferPointer { buffer in
            #expect(buffer.count == count)
            if count > 0 { buffer[0] = -1; buffer[count - 1] = -2 }
            return 73
        }
        #expect(result == 73)
        var expected = Array(0..<count)
        if count > 0 { expected[0] = -1; expected[count - 1] = -2 }
        #expect(Array(value) == expected)
        if let original {
            #expect(Array(original) == Array(0..<count))
            for i in 0..<count { #expect(iterator?.next() == i) }
            #expect(iterator?.next() == nil)
        }
    }

    @Test(arguments: [0, 1, 4, 5, 512, 513, 2048, 2049], [false, true])
    func throwingBulkWriteback(count: Int, shared: Bool) {
        var value = TieredStorage(Array(0..<count))
        let original: TieredStorage<Int>? = shared ? value : nil
        var iterator = original?.makeIterator()
        let error = BorrowError()
        do {
            try value.withUnsafeMutableBufferPointer { buffer in
                #expect(buffer.count == count)
                if count > 0 { buffer[0] = -1; buffer[count - 1] = -2 }
                throw error
            }
            Issue.record("Expected throw")
        } catch let caught { #expect(caught as AnyObject === error) }
        var expected = Array(0..<count)
        if count > 0 { expected[0] = -1; expected[count - 1] = -2 }
        #expect(Array(value) == expected)
        if let original {
            #expect(Array(original) == Array(0..<count))
            for i in 0..<count { #expect(iterator?.next() == i) }
            #expect(iterator?.next() == nil)
        }
    }

    @Test(arguments: [5, 512, 513, 2048, 2049], [false, true])
    func scalarAndBulkReuseDetachedAllocation(count: Int, shared: Bool) {
        var value = TieredStorage(Array(0..<count))
        let original: TieredStorage<Int>? = shared ? value : nil
        let before = address(value)
        value[0] = -1
        let after = address(value)
        #expect(shared ? after != before : after == before)
        for i in 0..<10 {
            value[count - 1] = -i
            #expect(address(value) == after)
            value.withUnsafeMutableBufferPointer { $0[0] = -i }
            #expect(address(value) == after)
        }
        if let original { #expect(Array(original) == Array(0..<count)) }
        withExtendedLifetime(original) {}
    }

    @Test(arguments: [3, 4, 32, 33, 64, 65])
    func matrixMutationAndConstruction(size: Int) {
        let original = TriangularMatrix<Int>(size: size, defaultValue: 7)
        var changed = original
        var iterator = original.makeIterator()
        changed[0, 1] = -1
        let detached = address(changed.storage)
        for i in 0..<10 {
            changed[0, 1] = -i
            let set = changed.setAt(size - 2, size - 1, to: i)
            #expect(set)
            if changed.elementCount > 4 { #expect(address(changed.storage) == detached) }
        }
        #expect(changed[0, 1] == -9)
        #expect(changed[size - 2, size - 1] == 9)
        #expect(original.allSatisfy { $0.value == 7 })
        while let entry = iterator.next() { #expect(entry.value == 7) }
        let full = (0..<size).map { row in (0..<size).map { row * size + $0 } }
        let imported = TriangularMatrix<Int>.fromFullMatrix(full)!
        for row in 0..<size { for col in (row + 1)..<size {
            #expect(imported[row, col] == full[row][col])
        } }
    }

    @Test(arguments: [0, 5, 513, 2049])
    func emptyReservedTierBorrows(capacity: Int) {
        var value = TieredStorage<Int>(capacity: capacity)
        let original = value
        var iterator = value.makeIterator()
        let result = value.withUnsafeMutableBufferPointer { buffer in
            #expect(buffer.isEmpty)
            return 73
        }
        #expect(result == 73)
        let error = BorrowError()
        do {
            try value.withUnsafeMutableBufferPointer { buffer in
                #expect(buffer.isEmpty)
                throw error
            }
            Issue.record("Expected throw")
        } catch let caught { #expect(caught as AnyObject === error) }
        #expect(value.count == 0)
        #expect(Array(original).isEmpty)
        #expect(iterator.next() == nil)
    }

    private final class Ledger { var destroyed: [Int] = [] }
    private final class Sentinel {
        let id: Int
        let ledger: Ledger
        init(_ id: Int, _ ledger: Ledger) { self.id = id; self.ledger = ledger }
        deinit { ledger.destroyed.append(id) }
    }
    private final class WeakBox {
        weak var value: Sentinel?
        init(_ value: Sentinel) { self.value = value }
    }

    @Test(arguments: [1, 4, 5, 512, 513, 2048, 2049], [false, true])
    func throwingReferenceWritebackLifetime(count: Int, shared: Bool) {
        let ledger = Ledger()
        var boxes: [WeakBox] = []
        var replacementBox: WeakBox?
        func scope() {
            var value = TieredStorage((0..<count).map { Sentinel($0, ledger) })
            boxes = value.map(WeakBox.init)
            var original: TieredStorage<Sentinel>? = shared ? value : nil
            var iterator = original?.makeIterator()
            do {
                try value.withUnsafeMutableBufferPointer { buffer in
                    let replacement = Sentinel(count, ledger)
                    replacementBox = WeakBox(replacement)
                    buffer[0] = replacement
                    throw BorrowError()
                }
                Issue.record("Expected throw")
            } catch {}
            #expect(value[0].id == count)
            #expect(replacementBox?.value != nil)
            #expect((boxes[0].value != nil) == shared)
            #expect(ledger.destroyed == (shared ? [] : [0]))
            original = nil
            if shared {
                for i in 0..<count { #expect(iterator?.next()?.id == i) }
                iterator = nil
                #expect(boxes[0].value == nil)
            }
            #expect(boxes.dropFirst().allSatisfy { $0.value != nil })
            withExtendedLifetime(value) {}
        }
        scope()
        #expect(boxes.allSatisfy { $0.value == nil })
        #expect(replacementBox?.value == nil)
        #expect(ledger.destroyed.sorted() == Array(0...count))
    }
}
