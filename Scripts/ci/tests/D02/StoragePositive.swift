// D02 S3 / S4 (positive): the independently declared DynamicDimension.Storage alias stays, and Vector<D>.Storage is
// DimensionStorage<D, Float> for a value-only custom dimension. Before D02, V7 does not conform to StaticDimension.
import VectorCore

enum V7: StaticDimension { static let value = 7 }

@main
struct D02StoragePositive {
    static func main() {
        let dynamicStorage: ArrayStorage<DynamicDimension>.Type = DynamicDimension.Storage.self
        let customStorage: DimensionStorage<V7, Float>.Type = Vector<V7>.Storage.self
        let builtInStorage: DimensionStorage<Dim512, Float>.Type = Vector<Dim512>.Storage.self
        print("D02_STORAGE_POSITIVE_COMPLETE \(dynamicStorage) \(customStorage) \(builtInStorage)")
    }
}
