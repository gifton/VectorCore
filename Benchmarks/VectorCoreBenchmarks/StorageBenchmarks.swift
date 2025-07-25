import Benchmark
import VectorCore

func storageBenchmarks() {
    // Memory allocation benchmarks
    Benchmark("Vector Allocation - 128D") { benchmark in
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(Vector<Dim128>.zeros())
        }
    }
    
    Benchmark("Vector Allocation - 768D") { benchmark in
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(Vector<Dim768>.zeros())
        }
    }
    
    Benchmark("Vector Allocation - 1536D") { benchmark in
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(Vector<Dim1536>.zeros())
        }
    }
    
    // DynamicVector allocation benchmarks
    Benchmark("DynamicVector Allocation - 128D") { benchmark in
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(DynamicVector(dimension: 128, repeating: 0.0))
        }
    }
    
    Benchmark("DynamicVector Allocation - 768D") { benchmark in
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(DynamicVector(dimension: 768, repeating: 0.0))
        }
    }
    
    Benchmark("DynamicVector Allocation - 1536D") { benchmark in
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(DynamicVector(dimension: 1536, repeating: 0.0))
        }
    }
    
    // Random initialization benchmarks
    Benchmark("Vector Random Initialization - 768D") { benchmark in
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(Vector<Dim768>.random(in: -1...1))
        }
    }
    
    Benchmark("DynamicVector Random Initialization - 768D") { benchmark in
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(DynamicVector.random(dimension: 768, in: -1...1))
        }
    }
    
    // Copy benchmarks
    Benchmark("Vector Copy - 768D") { benchmark in
        let source = Vector<Dim768>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            let copy = source
            blackHole(copy)
        }
    }
    
    Benchmark("DynamicVector Copy - 768D") { benchmark in
        let source = DynamicVector.random(dimension: 768, in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            let copy = source
            blackHole(copy)
        }
    }
    
    // Array initialization benchmarks
    Benchmark("Vector from Array - 768D") { benchmark in
        let array = (0..<768).map { Float($0) }
        
        benchmark.startMeasurement()
        for _ in 0..<100 {
            blackHole(Vector<Dim768>(array))
        }
    }
    
    Benchmark("DynamicVector from Array - 768D") { benchmark in
        let array = (0..<768).map { Float($0) }
        
        benchmark.startMeasurement()
        for _ in 0..<100 {
            blackHole(DynamicVector(array))
        }
    }
    
    // Memory access patterns
    Benchmark("Vector Sequential Access - 768D") { benchmark in
        let vector = Vector<Dim768>.random(in: -1...1)
        
        benchmark.startMeasurement()
        var sum: Float = 0
        for _ in 0..<100 {
            for i in 0..<768 {
                sum += vector[i]
            }
        }
        blackHole(sum)
    }
    
    Benchmark("DynamicVector Sequential Access - 768D") { benchmark in
        let vector = DynamicVector.random(dimension: 768, in: -1...1)
        
        benchmark.startMeasurement()
        var sum: Float = 0
        for _ in 0..<100 {
            for i in 0..<768 {
                sum += vector[i]
            }
        }
        blackHole(sum)
    }
    
    // Batch allocation benchmarks
    Benchmark("Vector Batch Allocation - 100x768D") { benchmark in
        benchmark.startMeasurement()
        var vectors: [Vector<Dim768>] = []
        vectors.reserveCapacity(100)
        for _ in 0..<100 {
            vectors.append(Vector<Dim768>.zeros())
        }
        blackHole(vectors)
    }
    
    Benchmark("DynamicVector Batch Allocation - 100x768D") { benchmark in
        benchmark.startMeasurement()
        var vectors: [DynamicVector] = []
        vectors.reserveCapacity(100)
        for _ in 0..<100 {
            vectors.append(DynamicVector(dimension: 768, repeating: 0.0))
        }
        blackHole(vectors)
    }
}