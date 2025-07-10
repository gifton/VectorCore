# VectorCore Benchmarks

Performance benchmarks for VectorCore operations.

## Running Benchmarks

### Run all benchmarks:
```bash
swift run VectorCoreBenchmarks
```

### Run specific benchmark suites:
```bash
# SIMD storage benchmarks only
swift run VectorCoreBenchmarks --simd

# Parallel batch operation benchmarks only
swift run VectorCoreBenchmarks --parallel

# Show help
swift run VectorCoreBenchmarks --help
```

## Available Benchmark Suites

### SIMD Storage Benchmarks
Measures performance of different SIMD storage implementations:
- Vector128 operations (creation, access, math)
- Vector256 operations
- Comparison between different storage types

### Parallel Batch Benchmarks
Compares serial vs parallel implementations for:
- k-nearest neighbor search
- Pairwise distance calculations
- Statistical computations
- Various dataset sizes (1K, 5K, 10K, 50K vectors)

## Interpreting Results

### SIMD Storage Results
- **Time**: Nanoseconds per operation
- **Rate**: Operations per second
- **Allocations**: Heap allocations (0 is ideal)

### Parallel Batch Results
- **Speedup**: How much faster parallel is vs serial (e.g., 3.5x)
- **Efficiency**: Speedup divided by core count (closer to 100% is better)
- **Recommendation**: Automatic suggestion based on your hardware

## Performance Tips

1. **Warm-up**: Benchmarks include warm-up iterations to ensure consistent results
2. **Multiple Runs**: Results are averaged over multiple iterations
3. **System Load**: Run benchmarks on an idle system for best accuracy
4. **Release Mode**: For most accurate results, build in release mode:
   ```bash
   swift run -c release VectorCoreBenchmarks
   ```

## Adding New Benchmarks

1. Create a new benchmark file in `Benchmarks/Sources/VectorCoreBenchmarks/`
2. Import VectorCore with `@testable import VectorCore`
3. Add benchmark execution to `main.swift`
4. Update this README with the new benchmark description