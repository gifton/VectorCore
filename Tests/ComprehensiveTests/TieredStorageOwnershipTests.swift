import Testing
@testable import VectorCore

@Suite("TieredStorageOwnership")
struct TieredStorageOwnershipTests {
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
    private enum BorrowError: Error { case expected }
    private func address<T>(_ storage: TieredStorage<T>) -> UInt {
        storage.withUnsafeBufferPointer { UInt(bitPattern: $0.baseAddress!) }
    }

    @Test(arguments: 5...512)
    func compactCopiesAreIsolated(count: Int) {
        let original = TieredStorage(Array(0..<count))
        var changed = original
        var iterator = changed.makeIterator()
        changed[0] = -1
        #expect(Array(original) == Array(0..<count))
        #expect(changed[0] == -1)
        #expect(iterator.next() == 0)
        var bulk = original
        bulk.withUnsafeMutableBufferPointer { $0[count - 1] = -2 }
        #expect(Array(original) == Array(0..<count))
        #expect(bulk[count - 1] == -2)
    }

    @Test(arguments: [0, 1, 4, 5, 512, 513, 2048, 2049])
    func tierBoundariesPreserveValues(count: Int) {
        let original = TieredStorage(Array(0..<count))
        #expect(original.count == count)
        #expect(Array(original) == Array(0..<count))
        guard count > 0 else { return }
        var copy = original
        copy[0] = -1
        #expect(original[0] == 0)
        #expect(copy[0] == -1)
    }

    @Test(arguments: [3, 4, 32, 33, 64, 65])
    func matrixCopiesAndIteratorsAreSnapshots(size: Int) {
        let original = TriangularMatrix<Int>(size: size, defaultValue: 7)
        var numeric = original
        var iterator = numeric.storage.makeIterator()
        var matrixIterator = numeric.makeIterator()
        numeric[0, 1] = 8
        #expect(numeric[0, 1] == 8)
        #expect(original[0, 1] == 7)
        #expect(iterator.next() == 7)
        #expect(matrixIterator.next()?.value == 7)
        var generic = original
        let didSet = generic.setAt(0, size - 1, to: 9)
        #expect(didSet)
        #expect(generic[0, size - 1] == 9)
        #expect(original.allSatisfy { $0.value == 7 })
    }

    @Test(arguments: [0, 1, 4, 5, 512, 513, 2048, 2049])
    func initializedElementsAreReleased(count: Int) {
        let ledger = Ledger()
        var boxes: [WeakBox] = []
        func scope() {
            let elements = (0..<count).map { Sentinel($0, ledger) }
            boxes = elements.map(WeakBox.init)
            let storage = TieredStorage(elements)
            #expect(Array(storage).map(\.id) == Array(0..<count))
            withExtendedLifetime(storage) { #expect(boxes.allSatisfy { $0.value != nil }) }
        }
        scope()
        #expect(boxes.allSatisfy { $0.value == nil })
        #expect(ledger.destroyed.sorted() == Array(0..<count))
    }

    @Test
    func compactDetachReplacementAndIteratorLifetime() {
        let ledger = Ledger()
        var boxes: [WeakBox] = []
        func scope() {
            var original: TieredStorage<Sentinel>? = TieredStorage((0..<5).map { Sentinel($0, ledger) })
            boxes = original!.map(WeakBox.init)
            var copy = original!
            var iterator: IndexingIterator<TieredStorage<Sentinel>>? = original!.makeIterator()
            copy[0] = Sentinel(5, ledger)
            #expect(original![0].id == 0)
            original = nil
            #expect(boxes.allSatisfy { $0.value != nil })
            #expect(iterator!.next()?.id == 0)
            iterator = nil
            #expect(boxes[0].value == nil)
            #expect(boxes.dropFirst().allSatisfy { $0.value != nil })
            #expect(ledger.destroyed == [0])
            copy[0] = Sentinel(6, ledger)
            #expect(ledger.destroyed.sorted() == [0, 5])
            withExtendedLifetime(copy) {}
        }
        scope()
        #expect(boxes.allSatisfy { $0.value == nil })
        #expect(ledger.destroyed.sorted() == Array(0...6))
    }

    @Test(arguments: [1, 4, 5, 512, 513, 2048, 2049])
    func repeatedReferenceIsReleasedOnce(count: Int) {
        let ledger = Ledger()
        var box: WeakBox?
        func scope() {
            let object = Sentinel(42, ledger)
            box = WeakBox(object)
            let storage = TieredStorage(Array(repeating: object, count: count))
            withExtendedLifetime(storage) { #expect(box?.value != nil) }
        }
        scope()
        #expect(box?.value == nil)
        #expect(ledger.destroyed == [42])
    }

    @Test
    func emptyReservedCompactCapacityIsNotInitialized() {
        let storage = TieredStorage<Sentinel>(capacity: 5)
        #expect(storage.count == 0)
        #expect(Array(storage).isEmpty)
    }

    @Test
    func throwingCompactBorrowPreservesWritesAndIsolation() {
        let ledger = Ledger()
        var boxes: [WeakBox] = []
        func scope() {
            let original = TieredStorage((0..<5).map { Sentinel($0, ledger) })
            boxes = original.map(WeakBox.init)
            var copy = original
            do {
                try copy.withUnsafeMutableBufferPointer {
                    $0[0] = Sentinel(5, ledger)
                    throw BorrowError.expected
                }
                Issue.record("Borrow should throw")
            } catch BorrowError.expected {} catch { Issue.record("Unexpected error: \(error)") }
            #expect(copy[0].id == 5)
            #expect(original[0].id == 0)
            #expect(boxes.allSatisfy { $0.value != nil })
            #expect(ledger.destroyed.isEmpty)
            withExtendedLifetime((original, copy)) {}
        }
        scope()
        #expect(boxes.allSatisfy { $0.value == nil })
        #expect(ledger.destroyed.sorted() == Array(0...5))
    }

    @Test(arguments: 5...512)
    func compactDetachesOnlyWhenShared(count: Int) {
        let original = TieredStorage(Array(0..<count))
        var copy = original
        let originalAddress = address(original)
        copy[0] = -1
        let detachedAddress = address(copy)
        #expect(detachedAddress != originalAddress)
        copy[1] = -2
        #expect(address(copy) == detachedAddress)
        copy.withUnsafeMutableBufferPointer { $0[2] = -3 }
        #expect(address(copy) == detachedAddress)
        withExtendedLifetime(original) {}
    }

    @Test
    func fullMatrixImportsValidateShape() {
        #expect(TriangularMatrix<Int>.fromFullMatrix([]) == nil)
        #expect(TriangularMatrix<Int>.fromFullMatrix([[]]) == nil)
        #expect(TriangularMatrix<Int>.fromFullMatrix([[1, 2], [3]]) == nil)
        let two = TriangularMatrix<Int>.fromFullMatrix([[1, 2], [3, 4]])
        #expect(two?.toFullMatrix() == [[0, 2], [2, 0]])
    }
}

@Suite("TieredStorageOwnership singleton")
struct TieredStorageOwnershipSingletonTests {
    @Test
    func singletonFullMatrixImport() {
        let singleton = TriangularMatrix<Int>.fromFullMatrix([[7]])
        #expect(singleton?.size == 1)
        #expect(singleton?.elementCount == 0)
        #expect(singleton?.toFullMatrix() == [[0]])
    }
}
