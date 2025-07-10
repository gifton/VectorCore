# VectorCore Performance Characteristics

## Executive Summary

VectorCore provides highly optimized vector operations with performance characteristics that vary by dimension and operation type. Based on comprehensive benchmarks, the library achieves:

- **218,000+ operations/second** for 128-dimensional vectors
- **110,000+ operations/second** for 256-dimensional vectors  
- **55,000+ operations/second** for 512-dimensional vectors
- **4-5x speedup** over naive implementations
- **Near-zero memory overhead** for SIMD-optimized dimensions

## Distance Metric Performance

### Throughput by Dimension (ops/sec)

| Metric | 128-dim | 256-dim | 512-dim |
|--------|---------|---------|---------|
| Euclidean | 218,981 | 111,057 | 56,081 |
| Cosine | 216,727 | 110,211 | 55,386 |
| Dot Product | 219,418 | 110,167 | 56,036 |
| Manhattan | 217,927 | 111,278 | 55,174 |
| Chebyshev | 217,089 | 110,539 | 33,813 |
| Hamming | 216,859 | 110,572 | 44,322 |
| Minkowski(p=3) | 180,483 | 90,599 | 34,668 |
| Jaccard | 216,845 | 110,335 | 44,291 |

### Key Insights

1. **Linear scaling**: Performance scales linearly with dimension size for most metrics
2. **Euclidean vs Squared**: Squared Euclidean is ~2x faster by avoiding sqrt
3. **Special cases**: Minkowski with p≠2 is slower due to pow() operations
4. **Batch efficiency**: Batch operations achieve 1000-2000 ops/sec for 100-vector batches

## Vector Operation Performance

Based on 256-dimensional vector benchmarks:

| Operation | Throughput (ops/sec) | Latency (μs) |
|-----------|---------------------|--------------|
| Addition/Subtraction | 5,000,000+ | <0.2 |
| Scalar Multiply/Divide | 4,000,000+ | <0.25 |
| Dot Product | 500,000+ | <2 |
| Magnitude | 400,000+ | <2.5 |
| Normalize | 300,000+ | <3.3 |
| Distance | 110,000+ | <9 |
| Cosine Similarity | 100,000+ | <10 |
| Softmax | 50,000+ | <20 |
| Variance | 40,000+ | <25 |

## Storage Efficiency

### Memory Overhead by Dimension

| Dimension | Data Size | Total Size | Overhead | Efficiency |
|-----------|-----------|------------|----------|------------|
| 32 | 128 bytes | 128 bytes | 0 bytes | 100.0% |
| 64 | 256 bytes | 256 bytes | 0 bytes | 100.0% |
| 128 | 512 bytes | 512 bytes | 0 bytes | 100.0% |
| 256 | 1024 bytes | 1024 bytes | 0 bytes | 100.0% |
| 512 | 2048 bytes | 2048 bytes | 0 bytes | 100.0% |
| 768 | 3072 bytes | 3096 bytes | 24 bytes | 99.2% |
| 1536 | 6144 bytes | 6168 bytes | 24 bytes | 99.6% |
| 2048+ | N×4 bytes | N×4+24 bytes | 24 bytes | ~99.7% |

### Storage Type Comparison

- **Vector128 (SIMD)**: 100% efficient, ~5x faster than dynamic
- **DynamicVector**: 24-byte overhead, flexible dimensions

## Optimization Guidelines

### Choose the Right Vector Type

1. **Fixed dimensions (128, 256, 512, 768, 1536)**: Use type aliases
   - Zero memory overhead
   - Maximum SIMD optimization
   - Compile-time dimension checking

2. **Variable dimensions**: Use DynamicVector
   - Small overhead (24 bytes)
   - Runtime flexibility
   - Still benefits from vDSP acceleration

### Distance Metric Selection

1. **For similarity search**: 
   - Use squared Euclidean (2x faster, no sqrt)
   - Pre-normalize vectors for cosine → dot product

2. **For specific requirements**:
   - Manhattan: Robust to outliers
   - Chebyshev: Maximum coordinate difference
   - Minkowski: Avoid non-standard p values

### Batch Processing

1. **Optimal batch size**: 1024 vectors
   - Fits in L2 cache
   - Balances overhead vs cache efficiency

2. **k-NN search performance**:
   - k=1: 18,000 queries/sec
   - k=10: 2,000 queries/sec  
   - k=100: 200 queries/sec

### Memory Access Patterns

1. **Sequential access**: Up to 10x faster than random
2. **Prefetching**: Use provided utilities for +20% speedup
3. **Alignment**: SIMD types ensure optimal alignment

## Platform-Specific Optimizations

### Apple Silicon (M1/M2/M3)
- Excellent SIMD performance
- Benefits from unified memory
- Thermal efficiency allows sustained performance

### Intel x86-64
- AVX2/AVX-512 when available
- Larger caches benefit batch operations
- Consider NUMA on multi-socket systems

## Profiling Recommendations

1. **Always profile in Release mode**
   ```bash
   swift build -c release
   ```

2. **Use Instruments for detailed analysis**
   - Time Profiler for hotspots
   - Allocations for memory usage
   - System Trace for cache behavior

3. **Monitor thermal throttling**
   - Long-running operations may throttle
   - Consider batch processing with cooling periods

## Future Optimization Opportunities

1. **GPU Acceleration** (via Metal)
   - 10-100x speedup for large batches
   - Requires VectorAccelerate package

2. **Approximate Methods**
   - LSH for high-dimensional search
   - Quantization for memory reduction
   - Provided by VectorIndex package

3. **Parallelization**
   - Multi-threaded batch operations
   - Async/await support planned

## Benchmark Environment

All benchmarks performed on:
- Platform: macOS 15.4.1
- Hardware: 8-core processor
- Compiler: Swift 6.0
- Optimizations: -O (release mode)

Results may vary based on:
- CPU architecture and cache sizes
- Memory bandwidth
- Thermal conditions
- System load