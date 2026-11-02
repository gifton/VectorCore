import Foundation

/// Internal synchronous magnitude statistics.
internal enum SyncBatchOperations {
    /// Population statistics of delivered Float magnitudes using centered Double accumulation.
    /// Each magnitude is evaluated once; row dimensions may differ. Empty input
    /// returns positive zeros. Any nonfinite magnitude retains count and yields two NaNs,
    /// including an overflowing magnitude of a vector with finite coordinates.
    internal static func statistics<V: VectorProtocol>(
        for vectors: [V]
    ) -> BatchStatistics where V.Scalar == Float {
        var accumulator = MagnitudeStatisticsAccumulator()
        for vector in vectors { accumulator.add(vector.magnitude) }
        return accumulator.statistics
    }

}

internal extension Array where Element: VectorProtocol {
    /// Get batch statistics
    func batchStatistics() -> BatchStatistics where Element.Scalar == Float {
        SyncBatchOperations.statistics(for: self)
    }
}
