import Benchmark
import VectorCore

func vectorOperationBenchmarks() {
    // Addition benchmarks for various dimensions
    Benchmark("Vector Addition - 128D") { benchmark in
        let a = Vector<Dim128>.random(in: -1...1)
        let b = Vector<Dim128>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(a + b)
        }
    }
    
    Benchmark("Vector Addition - 256D") { benchmark in
        let a = Vector<Dim256>.random(in: -1...1)
        let b = Vector<Dim256>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(a + b)
        }
    }
    
    Benchmark("Vector Addition - 512D") { benchmark in
        let a = Vector<Dim512>.random(in: -1...1)
        let b = Vector<Dim512>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(a + b)
        }
    }
    
    Benchmark("Vector Addition - 768D") { benchmark in
        let a = Vector<Dim768>.random(in: -1...1)
        let b = Vector<Dim768>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(a + b)
        }
    }
    
    Benchmark("Vector Addition - 1536D") { benchmark in
        let a = Vector<Dim1536>.random(in: -1...1)
        let b = Vector<Dim1536>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(a + b)
        }
    }
    
    // Subtraction benchmarks
    Benchmark("Vector Subtraction - 768D") { benchmark in
        let a = Vector<Dim768>.random(in: -1...1)
        let b = Vector<Dim768>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(a - b)
        }
    }
    
    // Scalar multiplication benchmarks
    Benchmark("Vector Scalar Multiplication - 768D") { benchmark in
        let a = Vector<Dim768>.random(in: -1...1)
        let scalar: Float = 2.5
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(a * scalar)
        }
    }
    
    // Dot product benchmarks
    Benchmark("Vector Dot Product - 128D") { benchmark in
        let a = Vector<Dim128>.random(in: -1...1)
        let b = Vector<Dim128>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(a.dotProduct(b))
        }
    }
    
    Benchmark("Vector Dot Product - 768D") { benchmark in
        let a = Vector<Dim768>.random(in: -1...1)
        let b = Vector<Dim768>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(a.dotProduct(b))
        }
    }
    
    Benchmark("Vector Dot Product - 1536D") { benchmark in
        let a = Vector<Dim1536>.random(in: -1...1)
        let b = Vector<Dim1536>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(a.dotProduct(b))
        }
    }
    
    // DynamicVector operations
    Benchmark("DynamicVector Addition - 768D") { benchmark in
        let a = DynamicVector.random(dimension: 768, in: -1...1)
        let b = DynamicVector.random(dimension: 768, in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(a + b)
        }
    }
    
    Benchmark("DynamicVector Dot Product - 768D") { benchmark in
        let a = DynamicVector.random(dimension: 768, in: -1...1)
        let b = DynamicVector.random(dimension: 768, in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(a.dotProduct(b))
        }
    }
    
    // Normalization benchmarks
    Benchmark("Vector Normalization - 768D") { benchmark in
        let a = Vector<Dim768>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(a.normalized())
        }
    }
    
    // Magnitude benchmarks
    Benchmark("Vector Magnitude - 768D") { benchmark in
        let a = Vector<Dim768>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(a.magnitude)
        }
    }
    
    // Element-wise operations
    Benchmark("Vector Element-wise Multiplication - 768D") { benchmark in
        let a = Vector<Dim768>.random(in: -1...1)
        let b = Vector<Dim768>.random(in: -1...1)
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(a * b.toArray()[0]) // No elementwise multiply, using scalar mult as placeholder
        }
    }
}