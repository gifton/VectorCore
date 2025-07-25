# VectorCore Benchmark Usage

## Running Benchmarks

To run the VectorCore benchmarks:

```bash
swift run -c release VectorCoreBenchmarks
```

## Benchmark Categories

The benchmark suite includes:

1. **Vector Operations** - Basic arithmetic operations (add, subtract, multiply, dot product)
2. **Storage Operations** - Memory allocation, initialization, and access patterns
3. **Distance Metrics** - Euclidean, Cosine, and Manhattan distance calculations
4. **Batch Operations** - findNearest, mean, and normalize operations on vector collections

## Dimensions Tested

- 128D - Small embeddings
- 256D - Medium embeddings (where available)
- 512D - Standard transformer outputs (where available)
- 768D - BERT-style embeddings
- 1536D - Large language model embeddings

## Performance Regression Detection

To detect performance regressions:

1. Run benchmarks and save baseline:
   ```bash
   swift run -c release VectorCoreBenchmarks --format json > baseline_metrics.json
   ```

2. After changes, run benchmarks again:
   ```bash
   swift run -c release VectorCoreBenchmarks --format json > current_metrics.json
   ```

3. Compare results to detect regressions > 5%

## Important Notes

- Always run benchmarks in release mode (`-c release`)
- Run on a quiet system to minimize variance
- Consider thermal throttling - benchmarks include warmup iterations
- Results are statistically significant with multiple runs