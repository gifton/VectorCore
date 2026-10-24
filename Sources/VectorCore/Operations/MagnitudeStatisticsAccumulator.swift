// Centered population statistics of the Float measurements delivered by magnitude.
internal struct MagnitudeStatisticsAccumulator: Sendable {
    private var count = 0
    private var mean: Double = 0
    private var m2: Double = 0
    private var invalid = false

    mutating func add(_ magnitude: Float) {
        let (nextCount, overflow) = count.addingReportingOverflow(1)
        precondition(!overflow, "Magnitude statistics count overflow")
        let oldCount = count
        count = nextCount
        guard magnitude.isFinite else { invalid = true; return }
        guard !invalid else { return }
        let value = Double(magnitude)
        if oldCount == 0 { mean = value; return }
        let delta = value - mean
        mean += delta / Double(count)
        // Welford's centered update, with an explicitly nonnegative increment.
        m2 += delta * delta * (Double(oldCount) / Double(count))
    }

    mutating func merge(_ other: Self) {
        let (combinedCount, overflow) = count.addingReportingOverflow(other.count)
        precondition(!overflow, "Magnitude statistics count overflow")
        if other.count == 0 { return }
        if count == 0 { self = other; return }
        let oldCount = count
        count = combinedCount
        invalid = invalid || other.invalid
        guard !invalid else { return }
        let delta = other.mean - mean
        let fraction = Double(other.count) / Double(count)
        mean += delta * fraction
        // Chan's centered merge; count products are widened before multiplication.
        m2 += other.m2 + delta * delta * Double(oldCount) * fraction
    }

    var statistics: BatchStatistics {
        guard count > 0 else { return BatchStatistics(count: 0, meanMagnitude: 0, stdMagnitude: 0) }
        guard !invalid else { return BatchStatistics(count: count, meanMagnitude: .nan, stdMagnitude: .nan) }
        return BatchStatistics(count: count, meanMagnitude: Float(mean), stdMagnitude: Float((m2 / Double(count)).squareRoot()))
    }
}
