import Foundation
import VectorCore

final class Token {}

@main
struct Probe {
    static func main() throws {
        for size in [3, 4, 16, 32, 33, 65] {
            let original = TriangularMatrix<Float>(size: size, defaultValue: 1)
            var copy = original
            copy[0, 1] = 99
            print("matrix", size, "original", original[0, 1], "copy", copy[0, 1])
        }
        weak var observed: Token?
        do {
            let token = Token()
            observed = token
            let matrix = TriangularMatrix<Token>(size: 4, defaultValue: token)
            withExtendedLifetime(matrix) {}
        }
        print("compact element remains alive", observed != nil)
        let empty = DynamicVector()
        let encoded = empty.encodeBinary()
        do {
            let restored = try DynamicVector.decodeBinary(from: encoded)
            print("empty binary restored", restored.scalarCount)
        } catch {
            print("empty binary rejected", encoded.count, error)
        }
        let json = try JSONEncoder().encode(empty)
        print("empty JSON restored", try JSONDecoder().decode(DynamicVector.self, from: json).scalarCount)
    }
}
