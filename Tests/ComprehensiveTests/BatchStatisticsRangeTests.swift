import Foundation
import Testing
@testable import VectorCore

@Suite struct BatchStatisticsRangeTests {
    private func assertStatistics(_ result: BatchStatistics, count: Int, mean: Float, deviation: Float, exactZero: Bool = false) {
        #expect(result.count == count)
        #expect(result.meanMagnitude.isFinite)
        #expect(result.stdMagnitude.isFinite)
        if count == 0 { #expect(result.meanMagnitude.bitPattern == 0) }
        if deviation > 0 { #expect(result.stdMagnitude > 0) }
        let meanAllowance = 16 * Double(mean.ulp) + 8 * Double(Float.leastNonzeroMagnitude)
        let stdAllowance = 16 * Double(deviation.ulp) + 8 * Double(Float.leastNonzeroMagnitude)
        #expect(abs(Double(result.meanMagnitude) - Double(mean)) <= meanAllowance)
        if exactZero { #expect(result.stdMagnitude.bitPattern == 0) }
        else { #expect(abs(Double(result.stdMagnitude) - Double(deviation)) <= stdAllowance) }
    }
    private func checkPublic(_ values: [Float], mean: Float, deviation: Float, exactZero: Bool = false) async {
        let vectors = values.map { DynamicVector([$0]) }
        assertStatistics(await BatchOperations.statistics(for: vectors), count: values.count, mean: mean, deviation: deviation, exactZero: exactZero)
        assertStatistics(SyncBatchOperations.statistics(for: vectors), count: values.count, mean: mean, deviation: deviation, exactZero: exactZero)
        assertStatistics(vectors.batchStatistics(), count: values.count, mean: mean, deviation: deviation, exactZero: exactZero)
    }
    @Test func thresholdDiscontinuity() async {
        for count in [999, 1000, 1024] {
            let nA = (count + 1) / 2, nB = count / 2
            let values = (0..<count).map { $0 % 2 == 0 ? Float(10000) : Float(10001) }
            let mean = Float(10000 + Double(nB) / Double(count))
            let std = Float((Double(nA) * Double(nB)).squareRoot() / Double(count))
            await checkPublic(values, mean: mean, deviation: std)
        }
    }
    @Test func identicalMeasurements() async {
        for value: Float in [0, .leastNonzeroMagnitude, Float.leastNormalMagnitude.nextDown, .leastNormalMagnitude, Float.leastNormalMagnitude.nextUp, 1, 1e18, .greatestFiniteMagnitude] {
            for count in [1, 999, 1000, 1024] {
                await checkPublic(Array(repeating: value, count: count), mean: value, deviation: 0, exactZero: true)
            }
        }
        await checkPublic([], mean: 0, deviation: 0, exactZero: true)
    }
    @Test func decimalFixtures() async {
        for fixture in n03MagnitudeFixtures {
            await checkPublic(fixture.bits.map(Float.init(bitPattern:)), mean: Float(bitPattern: fixture.mean), deviation: Float(bitPattern: fixture.deviation))
        }
    }
    @Test func invalidMeasurements() async {
        for count in [3, 1000, 1024] { for index in [0, count / 2, count - 1] { for bad: Float in [.nan, .infinity, -.infinity] {
            var vectors = Array(repeating: DynamicVector([1]), count: count); vectors[index] = DynamicVector([bad])
            let results = [await BatchOperations.statistics(for: vectors), SyncBatchOperations.statistics(for: vectors), vectors.batchStatistics()]
            for result in results { #expect(result.count == count); #expect(result.meanMagnitude.isNaN); #expect(result.stdMagnitude.isNaN) }
        } } }
        let overflow = DynamicVector([Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude])
        #expect(overflow.toArray().allSatisfy { $0.isFinite })
        for vectors in [[overflow], [DynamicVector([1]), overflow]] {
            for result in [await BatchOperations.statistics(for: vectors), SyncBatchOperations.statistics(for: vectors), vectors.batchStatistics(),
                SyncBatchOperations.finiteStatistics(for: vectors)] {
                #expect(result.count == vectors.count); #expect(result.meanMagnitude.isNaN); #expect(result.stdMagnitude.isNaN)
            }
        }
        let filtered = SyncBatchOperations.finiteStatistics(for: [DynamicVector([3, 4]), DynamicVector([.nan]), DynamicVector([.infinity])])
        assertStatistics(filtered, count: 1, mean: 5, deviation: 0, exactZero: true)
        assertStatistics(SyncBatchOperations.finiteStatistics(for: [DynamicVector([.nan])]), count: 0, mean: 0, deviation: 0, exactZero: true)
    }
    @Test func ordinaryAndEmptyDimensions() async throws {
        let mixed = [DynamicVector([]), DynamicVector([3, 4]), DynamicVector([0, 0, 5])]
        assertStatistics(await BatchOperations.statistics(for: mixed), count: 3, mean: Float(10.0 / 3), deviation: Float(sqrt(50.0 / 9)))
        assertStatistics(SyncBatchOperations.statistics(for: mixed), count: 3, mean: Float(10.0 / 3), deviation: Float(sqrt(50.0 / 9)))
        let fixed = try Vector<Dim3>([0, 3, 4])
        assertStatistics(await BatchOperations.statistics(for: [fixed, fixed]), count: 2, mean: 5, deviation: 0, exactZero: true)
        var coordinates = Array(repeating: Float(0), count: 512); coordinates[0] = 3; coordinates[1] = 4
        let optimized = try Vector512Optimized(coordinates)
        assertStatistics(await BatchOperations.statistics(for: [optimized, optimized]), count: 2, mean: 5, deviation: 0, exactZero: true)
        await checkPublic([0, 0], mean: 0, deviation: 0, exactZero: true)
    }
}

extension BatchStatisticsRangeTests {
    private func orders(_ values: [Float]) -> [[Float]] {
        var shuffled = values
        var seed: UInt64 = 0x4e3033
        if shuffled.count > 1 {
            for index in stride(from: shuffled.count - 1, through: 1, by: -1) {
                seed = seed &* 6364136223846793005 &+ 1
                shuffled.swapAt(index, Int(seed % UInt64(index + 1)))
            }
        }
        return [values, Array(values.reversed()), shuffled]
    }
    private func checkPartitions(_ values: [Float], mean: Float, deviation: Float, exactZero: Bool = false) async {
        for order in orders(values) {
            let vectors = order.map { DynamicVector([$0]) }
            for chunk in [1, 2, 3, 127, 256, 1024, values.count + 7, Int.max] {
                let first = await BatchOperations.statisticsParallel(for: vectors, chunkSize: chunk)
                assertStatistics(first, count: values.count, mean: mean, deviation: deviation, exactZero: exactZero)
                // Repeated actual tasks must merge by index, not completion order.
                let second = await BatchOperations.statisticsParallel(for: vectors, chunkSize: chunk)
                #expect(first.meanMagnitude.bitPattern == second.meanMagnitude.bitPattern)
                #expect(first.stdMagnitude.bitPattern == second.stdMagnitude.bitPattern)
            }
            await checkPublic(order, mean: mean, deviation: deviation, exactZero: exactZero)
        }
    }
    @Test func partitionAndOrderMatrix() async {
        for count in [0, 1, 2, 3, 999, 1000, 1024] {
            let nA = (count + 1) / 2, nB = count / 2
            let values = (0..<count).map { $0 % 2 == 0 ? Float(10000) : Float(10001) }
            let mean = count == 0 ? 0 : Float(10000 + Double(nB) / Double(count))
            let std = count == 0 ? 0 : Float((Double(nA) * Double(nB)).squareRoot() / Double(count))
            await checkPartitions(values, mean: mean, deviation: std, exactZero: count < 2)
        }
        for fixture in n03MagnitudeFixtures {
            await checkPartitions(fixture.bits.map(Float.init(bitPattern:)), mean: Float(bitPattern: fixture.mean), deviation: Float(bitPattern: fixture.deviation))
        }
        for value: Float in [0, .leastNonzeroMagnitude, Float.leastNormalMagnitude.nextDown, .leastNormalMagnitude, Float.leastNormalMagnitude.nextUp, 1, 1e18, .greatestFiniteMagnitude] {
            await checkPartitions(Array(repeating: value, count: 9), mean: value, deviation: 0, exactZero: true)
        }
    }
    @Test func invalidPartialsAndOverflowingFiniteRows() async {
        for index in [0, 4, 8] { for bad: Float in [.nan, .infinity, -.infinity] {
            var values = Array(repeating: Float(1), count: 9); values[index] = bad
            for order in orders(values) { for chunk in [1, 2, 3, 127, 256, 1024, 16, Int.max] {
                let result = await BatchOperations.statisticsParallel(for: order.map { DynamicVector([$0]) }, chunkSize: chunk)
                #expect(result.count == 9 && result.meanMagnitude.isNaN && result.stdMagnitude.isNaN)
            } }
        } }
        let overflow = DynamicVector([Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude])
        var vectors = Array(repeating: DynamicVector([1]), count: 1000); vectors[999] = overflow
        for result in [await BatchOperations.statistics(for: vectors), SyncBatchOperations.statistics(for: vectors), vectors.batchStatistics(), SyncBatchOperations.finiteStatistics(for: vectors)] {
            #expect(result.count == 1000 && result.meanMagnitude.isNaN && result.stdMagnitude.isNaN)
        }
        let zeroDimensions = Array(repeating: DynamicVector([]), count: 1000)
        assertStatistics(await BatchOperations.statistics(for: zeroDimensions), count: 1000, mean: 0, deviation: 0, exactZero: true)
        assertStatistics(zeroDimensions.batchStatistics(), count: 1000, mean: 0, deviation: 0, exactZero: true)
    }
    @Test func accumulatorDomainAndMergeIdentities() {
        for bad: Float in [.nan, .infinity, -.infinity] {
            for position in 0..<3 {
                var accumulator = MagnitudeStatisticsAccumulator()
                for index in 0..<3 { accumulator.add(index == position ? bad : 1) }
                let result = accumulator.statistics
                #expect(result.count == 3 && result.meanMagnitude.isNaN && result.stdMagnitude.isNaN)
                var empty = MagnitudeStatisticsAccumulator(); empty.merge(accumulator)
                accumulator.merge(MagnitudeStatisticsAccumulator())
                #expect(empty.statistics.count == 3 && empty.statistics.meanMagnitude.isNaN && empty.statistics.stdMagnitude.isNaN)
                #expect(accumulator.statistics.count == 3 && accumulator.statistics.meanMagnitude.isNaN && accumulator.statistics.stdMagnitude.isNaN)
            }
        }
        for fixture in n03MagnitudeFixtures {
            let values = fixture.bits.map(Float.init(bitPattern:))
            for split in 0...values.count {
                var first = MagnitudeStatisticsAccumulator(), second = MagnitudeStatisticsAccumulator()
                for value in values.prefix(split) { first.add(value) }
                for value in values.dropFirst(split) { second.add(value) }
                first.merge(second)
                assertStatistics(first.statistics, count: values.count, mean: Float(bitPattern: fixture.mean), deviation: Float(bitPattern: fixture.deviation))
            }
        }
    }
}
