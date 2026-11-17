import VectorCore

@main struct StorageValueConsumer {
    static func check<S: VectorStorage>(_ original: S) async where S.Scalar == Float {
        await withTaskGroup(of: Float.self) { group in
            for task in 1...16 {
                group.addTask {
                    var copy = original
                    for _ in 0..<100 {
                        copy[0] = Float(task)
                        copy.withUnsafeMutableBufferPointer { buffer in
                            for i in buffer.indices { buffer[i] = Float(task) }
                        }
                    }
                    precondition(copy.withUnsafeBufferPointer { $0.allSatisfy { $0 == Float(task) } })
                    return copy[0]
                }
            }
            var values: Set<Float> = []
            for await value in group { values.insert(value) }
            precondition(values == Set((1...16).map(Float.init)))
        }
        precondition(original.withUnsafeBufferPointer { $0.allSatisfy { $0 == -1 } })
    }

    static func main() async {
        await check(AlignedDynamicArrayStorage(dimension: 257, repeating: -1))
        await check(DynamicArrayStorage(dimension: 257, repeating: -1))
        await check(COWDynamicStorage(dimension: 257, repeating: -1))
        print("F001_VALUE_STORAGE_OK")
    }
}
