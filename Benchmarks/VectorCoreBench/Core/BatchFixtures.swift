import Foundation

/// The immutable FP32 query/candidate fixture of the batch benchmarks. The seeds and generator are those of the
/// default `batch` suite (`InputSeeds.batch`, `InputFactory.randomArray`), so a companion in the `boundary` suite and a
/// default batch case at the same width, count and run seed measure the same inputs.
enum BatchFixtures {
    struct Fixture<Vector> {
        let query: Vector
        let candidates: [Vector]
    }

    static func makeFixture<Vector>(dim: Int, count: Int, runSeed: UInt64, vector: ([Float]) throws -> Vector) throws -> Fixture<Vector> {
        let seeds = InputSeeds.batch(dim: dim, n: count, runSeed: runSeed)
        let querySeed = InputSeeds.batch(dim: dim, n: 0, runSeed: runSeed).query
        let query = try vector(InputFactory.randomArray(count: dim, seed: querySeed))
        let candidates = try (0..<count).map { try vector(InputFactory.randomArray(count: dim, seed: seeds.baseCandidates &+ UInt64($0))) }
        return Fixture(query: query, candidates: candidates)
    }
}
