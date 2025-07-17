# VectorCore Performance Benchmarks

Comprehensive performance benchmark suite for the VectorCore library.

## Overview

The VectorCore benchmark suite provides detailed performance analysis including:

- **Vector Creation**: Performance of creating vectors from arrays, zeros, and random values
- **Basic Operations**: Addition, subtraction, multiplication, division
- **Advanced Operations**: Dot product, normalization, distance calculations, cosine similarity
- **Storage Types**: Performance comparison of Small, Medium, and Large storage implementations
- **Batch Operations**: Serial vs parallel performance for batch processing
- **Dimension Analysis**: Performance characteristics across different vector dimensions
- **Regression Detection**: Automated performance regression detection against baselines

## Running Benchmarks

### Quick Start

```bash
# Run all benchmarks
swift run VectorCoreBenchmarks

# Run quick benchmarks (fewer iterations)
swift run VectorCoreBenchmarks --quick

# Run specific benchmark suite
swift run VectorCoreBenchmarks --suite --baseline --json
```

### Command Line Options

- `--all`: Run all benchmarks (default)
- `--parallel`: Run auto-parallelization benchmarks
- `--comprehensive`: Run original comprehensive benchmarks
- `--suite`, `--new`: Run new comprehensive benchmark suite
- `--dimensions`: Run dimension comparison analysis
- `--regression`: Run regression detection
- `--quick`: Run quick benchmarks (fewer iterations)
- `--baseline`: Include baseline comparisons (slower but more informative)
- `--json`: Save results to JSON file
- `--update-baseline`: Update regression baseline after running
- `--help`, `-h`: Show help message

### Examples

```bash
# Run comprehensive suite with baseline comparisons and JSON output
swift run VectorCoreBenchmarks --suite --baseline --json

# Run dimension comparison analysis
swift run VectorCoreBenchmarks --dimensions

# Run regression detection and update baseline
swift run VectorCoreBenchmarks --regression --update-baseline

# Quick parallel benchmarks
swift run VectorCoreBenchmarks --quick --parallel
```

## Benchmark Framework Features

### 1. Statistical Analysis
- Multiple runs for statistical significance
- Mean, median, min, max, standard deviation
- 95th and 99th percentile measurements
- Warmup iterations before measurement

### 2. Memory Tracking
- Allocation counting
- Bytes allocated per operation
- Memory efficiency analysis

### 3. Baseline Comparisons
- Compare VectorCore performance against naive Swift Array implementations
- Automatic speedup calculation
- Identify optimization effectiveness

### 4. JSON Output
- Machine-readable results for CI/CD integration
- Historical performance tracking
- Regression detection support

### 5. Dimension Analysis
- Performance scaling across dimensions (32 to 3072)
- SIMD utilization efficiency
- Cache effect analysis
- Optimal batch size recommendations

## Interpreting Results

### Performance Metrics
- **ops/sec**: Operations per second (higher is better)
- **elements/sec**: Elements processed per second for vector operations
- **Time metrics**: Mean time with standard deviation
- **Memory metrics**: Allocations and bytes per operation

### Speedup Values
- `1.0x`: Same performance as baseline
- `>1.0x`: Faster than baseline (e.g., 5.2x = 5.2 times faster)
- `<1.0x`: Slower than baseline (indicates potential issue)

### Regression Detection
- ✅ **Improvement**: Performance improved by >10%
- ✓ **Stable**: Performance within ±5%
- ⚠️ **Minor Regression**: 5-10% performance decrease
- ❌ **Major Regression**: >10% performance decrease
- ⚠️ **CRITICAL REGRESSION**: Critical operation regressed (build should fail)

## Integration with CI/CD

### Regression Detection in CI

```bash
# Run regression detection in CI pipeline
swift run VectorCoreBenchmarks --regression

# Exit code 1 if critical regressions detected
if [ $? -eq 1 ]; then
    echo "Critical performance regression detected!"
    exit 1
fi
```

### Baseline Management

```bash
# Establish initial baseline
swift run VectorCoreBenchmarks --regression --update-baseline

# Commit baseline to repository
git add vectorcore_baseline.json
git commit -m "Update performance baseline"
```

### JSON Output for Analysis

```bash
# Generate JSON results for external analysis
swift run VectorCoreBenchmarks --suite --json

# Results saved to vectorcore_benchmark_results.json
```

## Performance Tips

1. **Run on consistent hardware**: Performance can vary between machines
2. **Close other applications**: Reduce system noise during benchmarking
3. **Use Release builds**: Always benchmark optimized code
4. **Multiple runs**: Use statistical runs for reliable results
5. **Thermal considerations**: Allow cooling between extended benchmark runs

## Adding New Benchmarks

To add new benchmarks, modify the appropriate file:
- `ComprehensiveBenchmarkSuite.swift`: For general operation benchmarks
- `DimensionComparisonBenchmark.swift`: For dimension-specific analysis
- `RegressionDetectionBenchmark.swift`: For regression-critical operations

Example:
```swift
benchmarks.append((
    name: "YourOperation",
    category: "Category",
    vectorSize: dimension,
    baseline: {
        // Naive implementation for comparison
    },
    block: {
        // VectorCore implementation
    }
))
```