import Benchmark
import VectorCore

@main
struct BenchmarkRunner: BenchmarkRunnerHooks {
    static func registerBenchmarks() {
        // Configure benchmarks
        Benchmark.defaultConfiguration = .init(
            metrics: [
                .throughput,
                .wallClock,
                .cpuTotal,
                .mallocCountTotal,
                .memoryLeaked
            ]
        )
        
        // Register benchmarks
        vectorOperationBenchmarks()
        distanceBenchmarks()
        storageBenchmarks() 
        batchOperationBenchmarks()
    }
}