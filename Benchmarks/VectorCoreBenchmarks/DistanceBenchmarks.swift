import Benchmark
import VectorCore

func distanceBenchmarks() {
    // Create distance metric instances
    let euclideanMetric = EuclideanDistance()
    let cosineMetric = CosineDistance()
    let manhattanMetric = ManhattanDistance()
    
    // Euclidean distance benchmarks
    Benchmark("Euclidean Distance - 128D") { benchmark in
        let a = Vector<Dim128>.random(in: -1...1)
        let b = Vector<Dim128>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(euclideanMetric.distance(a, b))
        }
    }
    
    Benchmark("Euclidean Distance - 768D") { benchmark in
        let a = Vector<Dim768>.random(in: -1...1)
        let b = Vector<Dim768>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(euclideanMetric.distance(a, b))
        }
    }
    
    Benchmark("Euclidean Distance - 1536D") { benchmark in
        let a = Vector<Dim1536>.random(in: -1...1)
        let b = Vector<Dim1536>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(euclideanMetric.distance(a, b))
        }
    }
    
    // Cosine similarity benchmarks
    Benchmark("Cosine Similarity - 128D") { benchmark in
        let a = Vector<Dim128>.random(in: -1...1)
        let b = Vector<Dim128>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(cosineMetric.distance(a, b))
        }
    }
    
    Benchmark("Cosine Similarity - 768D") { benchmark in
        let a = Vector<Dim768>.random(in: -1...1)
        let b = Vector<Dim768>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(cosineMetric.distance(a, b))
        }
    }
    
    Benchmark("Cosine Similarity - 1536D") { benchmark in
        let a = Vector<Dim1536>.random(in: -1...1)
        let b = Vector<Dim1536>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(cosineMetric.distance(a, b))
        }
    }
    
    // Manhattan distance benchmarks
    Benchmark("Manhattan Distance - 128D") { benchmark in
        let a = Vector<Dim128>.random(in: -1...1)
        let b = Vector<Dim128>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(manhattanMetric.distance(a, b))
        }
    }
    
    Benchmark("Manhattan Distance - 768D") { benchmark in
        let a = Vector<Dim768>.random(in: -1...1)
        let b = Vector<Dim768>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(manhattanMetric.distance(a, b))
        }
    }
    
    Benchmark("Manhattan Distance - 1536D") { benchmark in
        let a = Vector<Dim1536>.random(in: -1...1)
        let b = Vector<Dim1536>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(manhattanMetric.distance(a, b))
        }
    }
    
    // DynamicVector distance benchmarks
    Benchmark("DynamicVector Euclidean Distance - 768D") { benchmark in
        let a = DynamicVector.random(dimension: 768, in: -1...1)
        let b = DynamicVector.random(dimension: 768, in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(euclideanMetric.distance(a, b))
        }
    }
    
    Benchmark("DynamicVector Cosine Similarity - 768D") { benchmark in
        let a = DynamicVector.random(dimension: 768, in: -1...1)
        let b = DynamicVector.random(dimension: 768, in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(cosineMetric.distance(a, b))
        }
    }
    
    Benchmark("DynamicVector Manhattan Distance - 768D") { benchmark in
        let a = DynamicVector.random(dimension: 768, in: -1...1)
        let b = DynamicVector.random(dimension: 768, in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(manhattanMetric.distance(a, b))
        }
    }
    
    // Batch distance calculations
    Benchmark("Batch Euclidean Distance - 100x768D") { benchmark in
        let vectors = (0..<100).map { _ in Vector<Dim768>.random(in: -1...1) }
        let target = Vector<Dim768>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for vector in vectors {
            blackHole(euclideanMetric.distance(vector, target))
        }
    }
    
    Benchmark("Batch Cosine Similarity - 100x768D") { benchmark in
        let vectors = (0..<100).map { _ in Vector<Dim768>.random(in: -1...1) }
        let target = Vector<Dim768>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for vector in vectors {
            blackHole(cosineMetric.distance(vector, target))
        }
    }
    
    // Normalized vector distance (pre-normalized)
    Benchmark("Cosine Similarity Normalized - 768D") { benchmark in
        let a = Vector<Dim768>.random(in: -1...1).normalized()
        let b = Vector<Dim768>.random(in: -1...1).normalized()
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(cosineMetric.distance(a, b))
        }
    }
}