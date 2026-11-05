import Foundation
import VectorCore

@_silgen_name("vectorcoreLeakFixture")
func vectorcoreLeakFixture(_ clean: Int32)

@inline(never)
func exerciseConsumer() throws -> Float {
    var checksum: Float = 0
    for iteration: Int in 0..<64 {
        let vector: DynamicVector = DynamicVector((0..<257).map { Float($0 + iteration + 1) })
        let normalized: DynamicVector = try vector.normalized().get()
        precondition(abs(normalized.magnitude - 1) <= 1e-5)
        checksum += normalized.magnitude
        let a: Vector<Dim32> = Vector(repeating: Float(iteration + 1))
        let b: Vector<Dim32> = Vector(repeating: 2)
        precondition(abs(a.dotProduct(b) - Float(64 * (iteration + 1))) <= 1e-4)
    }
    return checksum
}

let checksum: Float = try exerciseConsumer()
precondition(abs(checksum - 64) <= 1e-3)
let leakMode: Bool = CommandLine.arguments.dropFirst().contains("leak")
vectorcoreLeakFixture(leakMode ? 0 : 1)
print("VECTORCORE_LEAK_CONSUMER_PASS iterations=64 dynamicDimension=257 checksum=\(checksum) leakMode=\(leakMode)")
