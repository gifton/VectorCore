import VectorCore
func requiresBinaryCodable<T: BinaryCodable>(_ type:T.Type) {}
func check() {
#if B01_FIXED
    requiresBinaryCodable(Vector<Dim3>.self)
#elseif B01_DYNAMIC
    requiresBinaryCodable(DynamicVector.self)
#elseif B01_OPTIMIZED
    requiresBinaryCodable(Vector512Optimized.self)
#elseif B01_INT8
    requiresBinaryCodable(Vector512INT8.self)
#endif
}
