import Benchmark
import VectorCore

func batchOperationBenchmarks() {
    // findNearest benchmarks with various dataset sizes
    Benchmark("findNearest - 100 vectors x 768D") { benchmark in
        let dataset = (0..<100).map { _ in Vector<Dim768>.random(in: -1...1) }
        let query = Vector<Dim768>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<100 {
            blackHole(SyncBatchOperations.findNearest(to: query, in: dataset, k: 10))
        }
    }
    
    Benchmark("findNearest - 1000 vectors x 768D") { benchmark in
        let dataset = (0..<1000).map { _ in Vector<Dim768>.random(in: -1...1) }
        let query = Vector<Dim768>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<10 {
            blackHole(SyncBatchOperations.findNearest(to: query, in: dataset, k: 10))
        }
    }
    
    Benchmark("findNearest - 10000 vectors x 768D") { benchmark in
        let dataset = (0..<10000).map { _ in Vector<Dim768>.random(in: -1...1) }
        let query = Vector<Dim768>.random(in: -1...1)
        
        benchmark.startMeasurement()
        blackHole(SyncBatchOperations.findNearest(to: query, in: dataset, k: 10))
    }
    
    // Different dimensions
    Benchmark("findNearest - 1000 vectors x 128D") { benchmark in
        let dataset = (0..<1000).map { _ in Vector<Dim128>.random(in: -1...1) }
        let query = Vector<Dim128>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<10 {
            blackHole(SyncBatchOperations.findNearest(to: query, in: dataset, k: 10))
        }
    }
    
    Benchmark("findNearest - 1000 vectors x 1536D") { benchmark in
        let dataset = (0..<1000).map { _ in Vector<Dim1536>.random(in: -1...1) }
        let query = Vector<Dim1536>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<10 {
            blackHole(SyncBatchOperations.findNearest(to: query, in: dataset, k: 10))
        }
    }
    
    // Different k values
    Benchmark("findNearest k=1 - 1000 vectors x 768D") { benchmark in
        let dataset = (0..<1000).map { _ in Vector<Dim768>.random(in: -1...1) }
        let query = Vector<Dim768>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<10 {
            blackHole(SyncBatchOperations.findNearest(to: query, in: dataset, k: 1))
        }
    }
    
    Benchmark("findNearest k=50 - 1000 vectors x 768D") { benchmark in
        let dataset = (0..<1000).map { _ in Vector<Dim768>.random(in: -1...1) }
        let query = Vector<Dim768>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<10 {
            blackHole(SyncBatchOperations.findNearest(to: query, in: dataset, k: 50))
        }
    }
    
    Benchmark("findNearest k=100 - 1000 vectors x 768D") { benchmark in
        let dataset = (0..<1000).map { _ in Vector<Dim768>.random(in: -1...1) }
        let query = Vector<Dim768>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<10 {
            blackHole(SyncBatchOperations.findNearest(to: query, in: dataset, k: 100))
        }
    }
    
    // DynamicVector batch operations
    Benchmark("DynamicVector findNearest - 1000 vectors x 768D") { benchmark in
        let dataset = (0..<1000).map { _ in DynamicVector.random(dimension: 768, in: -1...1) }
        let query = DynamicVector.random(dimension: 768, in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<10 {
            blackHole(SyncBatchOperations.findNearest(to: query, in: dataset, k: 10))
        }
    }
    
    // Different distance metrics
    Benchmark("findNearest Euclidean - 1000 vectors x 768D") { benchmark in
        let dataset = (0..<1000).map { _ in Vector<Dim768>.random(in: -1...1) }
        let query = Vector<Dim768>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<10 {
            blackHole(SyncBatchOperations.findNearest(to: query, in: dataset, k: 10, metric: EuclideanDistance()))
        }
    }
    
    Benchmark("findNearest Cosine - 1000 vectors x 768D") { benchmark in
        let dataset = (0..<1000).map { _ in Vector<Dim768>.random(in: -1...1) }
        let query = Vector<Dim768>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<10 {
            blackHole(SyncBatchOperations.findNearest(to: query, in: dataset, k: 10, metric: CosineDistance()))
        }
    }
    
    // Batch mean calculation
    Benchmark("Batch Mean - 100 vectors x 768D") { benchmark in
        let vectors = (0..<100).map { _ in Vector<Dim768>.random(in: -1...1) }
        
        benchmark.startMeasurement()
        for _ in 0..<100 {
            if let meanVector = SyncBatchOperations.mean(vectors) {
                blackHole(meanVector)
            }
        }
    }
    
    Benchmark("Batch Mean - 1000 vectors x 768D") { benchmark in
        let vectors = (0..<1000).map { _ in Vector<Dim768>.random(in: -1...1) }
        
        benchmark.startMeasurement()
        for _ in 0..<10 {
            if let meanVector = SyncBatchOperations.mean(vectors) {
                blackHole(meanVector)
            }
        }
    }
    
    // Batch normalization
    Benchmark("Batch Normalize - 100 vectors x 768D") { benchmark in
        let vectors = (0..<100).map { _ in Vector<Dim768>.random(in: -1...1) }
        
        benchmark.startMeasurement()
        // Batch normalize not available, normalize individually
        blackHole(vectors.map { $0.normalized() })
    }
    
    Benchmark("Batch Normalize - 1000 vectors x 768D") { benchmark in
        let vectors = (0..<1000).map { _ in Vector<Dim768>.random(in: -1...1) }
        
        benchmark.startMeasurement()
        // Batch normalize not available, normalize individually
        blackHole(vectors.map { $0.normalized() })
    }
}