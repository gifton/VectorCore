# VectorCore Performance Baseline Metrics

## Overview

This document describes the performance baseline metrics system for VectorCore, used to track and prevent performance regressions during the refactoring process.

> **Note**: For implementation details and technical architecture, see the [Performance Baseline Implementation Guide](./performance_baseline_implementation.md).

## Baseline Structure

The performance baseline captures comprehensive metrics across three key areas:

### 1. Throughput Metrics (ops/sec)
- **Vector Addition**: Operations per second for vector addition
- **Scalar Multiplication**: Vector-scalar multiplication throughput
- **Element-wise Multiplication**: Element-wise vector multiplication
- **Dot Product**: Vector dot product calculations
- **Distance Metrics**: Euclidean, Cosine, Manhattan distance calculations
- **Normalization**: Vector normalization operations

### 2. Memory Metrics
- **Bytes per Operation**: Average memory allocated per vector operation
- **Peak Memory Usage**: Maximum memory used during benchmarks
- **Allocation Rate**: Number of allocations per second
- **Bytes per Vector**: Memory footprint by dimension (128D, 256D, etc.)

### 3. Parallelization Metrics
- **Parallel Speedup**: Performance gain from parallelization
- **Scaling Efficiency**: How well performance scales with threads (0-1)
- **Optimal Batch Size**: Best batch size for parallel operations
- **Thread Utilization**: Percentage of available threads utilized

## Usage

### Capturing Baseline

Run the capture script to create a baseline:

```bash
swift Scripts/capture_baseline.swift baseline_metrics.json
```

This will:
1. Run all VectorCore benchmarks
2. Calculate aggregate metrics
3. Save results to JSON file
4. Print performance summary

### Comparing Performance

Compare current performance against baseline:

```bash
swift Scripts/compare_baseline.swift baseline_metrics.json current_metrics.json
```

With custom regression threshold (default 5%):

```bash
swift Scripts/compare_baseline.swift baseline_metrics.json current_metrics.json --threshold 0.10
```

### Example Output

```
VectorCore Performance Comparison
=================================
Baseline: baseline_metrics.json (6.0)
Current:  current_metrics.json (6.0)
Regression Threshold: 5%

Metric                              Baseline         Current     Change Status
------------------------------------------------------------------------------
Vector Addition (768D)                222222          215000      -3.2% ✓ Acceptable
Dot Product (768D)                   312500          325000      +4.0% ✅ Improvement
Euclidean Distance                   263157          250000      -5.0% ⚠️  REGRESSION
Memory per Operation                     512             520      +1.6% ✓ Acceptable

✅ All metrics within acceptable range
   1 metrics showed improvement
```

## Regression Detection

A regression is detected when:
- **Throughput metrics** decrease by more than the threshold
- **Memory metrics** increase by more than the threshold
- **Parallelization efficiency** decreases by more than the threshold

The default threshold is 5%, following industry standards for major refactoring.

## Integration with CI/CD

The comparison script exits with code 1 if regression is detected, making it suitable for CI/CD pipelines:

```yaml
- name: Check Performance
  run: |
    swift Scripts/capture_baseline.swift current_metrics.json
    swift Scripts/compare_baseline.swift baseline_metrics.json current_metrics.json
```

## Baseline File Format

The baseline is stored as JSON with the following structure:

```json
{
  "timestamp": "2024-01-18T10:30:00Z",
  "swiftVersion": "6.0",
  "platform": {
    "os": "macOS",
    "osVersion": "14.2",
    "architecture": "arm64",
    "cpuCores": 10,
    "memoryGB": 32.0
  },
  "throughput": {
    "vectorAddition": 222222.0,
    "dotProduct": 312500.0,
    ...
  },
  "memory": {
    "bytesPerOperation": 512,
    "peakMemoryUsage": 52428800,
    ...
  },
  "parallelization": {
    "parallelSpeedup": 3.2,
    "scalingEfficiency": 0.8,
    ...
  },
  "benchmarks": [...]
}
```

## Best Practices

1. **Capture baseline on quiet system** - Minimize background processes
2. **Use release builds** - Always benchmark optimized code
3. **Run multiple times** - Average results for stability
4. **Document changes** - Note any changes that affect performance
5. **Platform-specific baselines** - Maintain separate baselines per platform

## Troubleshooting

### Inconsistent Results
- Ensure system is idle during benchmarking
- Check for thermal throttling on extended runs
- Verify release build configuration

### Large Variations
- Increase iteration count in benchmarks
- Add warmup iterations
- Check for background system updates

### Memory Metrics Missing
- Ensure memory profiling is enabled
- Check for proper instrumentation in benchmarks