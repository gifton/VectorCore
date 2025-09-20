import Foundation

// Shared Duration helper for converting to nanoseconds as Double
extension Duration {
    var nanoseconds: Double {
        let (sec, attos) = self.components
        return Double(sec) * 1_000_000_000 + Double(attos) * 1e-9
    }
}

