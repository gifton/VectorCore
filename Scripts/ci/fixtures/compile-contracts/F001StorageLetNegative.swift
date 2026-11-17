import VectorCore

@main struct StorageLetNegative {
    static func main() {
        let aligned = AlignedDynamicArrayStorage(dimension: 4)
        let dynamic = DynamicArrayStorage(dimension: 4)
        aligned[0] = 1
        dynamic[0] = 1
        aligned.withUnsafeMutableBufferPointer { $0[0] = 1 }
        dynamic.withUnsafeMutableBufferPointer { $0[0] = 1 }
    }
}
