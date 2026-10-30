// D02 F1/F2/F6 (positive, before and after D02): VectorTypeFactory's concrete results as an ordinary consumer sees
// them: Vector<DimN> for the seven named dimensions, DynamicVector elsewhere, never a Vector*Optimized type.
import VectorCore

@main
struct D02Factory {
    static func main() throws {
        let named = [128, 256, 384, 512, 768, 1536, 3072]
        guard (0...4096).filter({ VectorTypeFactory.isSupported(dimension: $0) }) == named else {
            fatalError("D02 factory: supported list")
        }
        for dimension in named + [0, 1, 100, 4096] {
            let expected = named.contains(dimension) ? "Vector<Dim\(dimension)>" : "DynamicVector"
            let routes: [any VectorType] = [
                try VectorTypeFactory.vector(of: dimension, from: [Float](repeating: 1, count: dimension)),
                VectorTypeFactory.zeros(dimension: dimension), VectorTypeFactory.ones(dimension: dimension)]
            for vector in routes {
                guard String(describing: type(of: vector)) == expected, vector.scalarCount == dimension,
                      !(vector is Vector384Optimized), !(vector is Vector512Optimized),
                      !(vector is Vector768Optimized), !(vector is Vector1536Optimized) else {
                    fatalError("D02 factory: \(dimension) gave \(type(of: vector))")
                }
            }
        }
        print("D02_FACTORY_COMPLETE")
    }
}
