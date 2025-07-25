# Task: Implement Performance Benchmarking Framework for VectorCore

## Context

You are implementing the first task of Phase 1 in a comprehensive refactoring of VectorCore, a high-performance vector mathematics library for Swift. The library is being completely redesigned to eliminate ~1,500 lines of code duplication while maintaining performance.

### Project Status
- **Current State**: VectorCore exists with working functionality but significant duplication
- **Target**: Swift 6.0 modern architecture with zero duplication
- **Constraint**: Maximum 5% performance regression tolerance
- **Platform**: Apple platforms (macOS 13+, iOS 16+)

### Architecture Decisions Already Made
- Using Swift 6.0 exclusively (bleeding edge)
- Protocol-based design to unify Vector<Dimension> and DynamicVector
- ExecutionContext pattern for CPU/GPU operations
- Platform-specific optimizations for Apple Silicon
- No legacy compatibility needed (complete freedom to redesign)

## Your Task

Implement the performance benchmarking framework as specified in Phase 1, line item 1:

### Deliverables

1. **Update Package.swift** to include:
   - Swift Benchmark package dependency
   - Benchmark executable target
   - Appropriate build settings

2. **Create benchmark directory structure**:
   ```
   Benchmarks/
   ├── VectorCoreBenchmarks/
   │   ├── VectorOperationBenchmarks.swift
   │   ├── StorageBenchmarks.swift
   │   ├── DistanceBenchmarks.swift
   │   └── BatchOperationBenchmarks.swift
   └── main.swift
   ```

3. **Implement comprehensive benchmarks** for:
   - Basic vector operations (add, subtract, multiply, dot product)
   - Distance calculations (Euclidean, Cosine, Manhattan)
   - Batch operations (findNearest with various dataset sizes)
   - Memory allocation patterns
   - Both Vector<Dim128>, Vector<Dim768> and DynamicVector variants

4. **Establish baseline metrics** by:
   - Running benchmarks on current implementation
   - Documenting results in `Benchmarks/baseline_metrics.json`
   - Creating comparison script to detect regressions

### Technical Requirements

- Use Apple's swift-benchmark package (not third-party alternatives)
- Benchmark names should be descriptive: "Vector Addition - 512D"
- Include both micro-benchmarks (single operations) and macro-benchmarks (realistic workflows)
- Test dimensions relevant to ML: 128, 256, 384, 512, 768, 1024, 1536
- Measure both throughput (ops/sec) and memory allocation

### Code Example to Follow

```swift
import Benchmark
import VectorCore

let benchmarks = {
    Benchmark("Vector Addition - 768D") { benchmark in
        let a = Vector<Dim768>.random()
        let b = Vector<Dim768>.random()
        
        benchmark.startMeasurement()
        for _ in 0..<1000 {
            blackHole(a + b)
        }
    }
}
```

### Important Notes

1. The current VectorCore implementation is in `/Users/goftin/dev/gsuite/VSK/VectorCore/Sources/VectorCore/`
2. Focus on measuring the CURRENT implementation first (not the refactored version)
3. Ensure benchmarks can run via: `swift run -c release VectorCoreBenchmarks`
4. Results should be reproducible and statistically significant
5. Consider thermal throttling - add warmup iterations

### Success Criteria

- [ ] Benchmarking framework compiles and runs
- [ ] All major operations have benchmarks
- [ ] Baseline metrics captured and saved
- [ ] Can detect 5% performance regressions
- [ ] Results are consistent across runs
- [ ] Documentation explains how to run and interpret benchmarks

## Additional Context

The benchmark suite you create will be used throughout the refactoring to ensure we don't regress performance. It's critical that the benchmarks are comprehensive and accurately reflect real-world usage patterns of a vector mathematics library used in machine learning applications.

Focus on getting accurate measurements of the current implementation. We'll use these as our baseline for all future optimizations.