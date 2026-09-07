---
name: Performance regression
about: Report a performance degradation
title: '[PERF] '
labels: "performance, regression"
assignees: ''
---

## Performance Regression Description
<!-- Describe the performance issue you're experiencing -->

## Affected Operations
<!-- Which operations are slower? -->
- [ ] Vector Addition
- [ ] Vector Multiplication
- [ ] Dot Product
- [ ] Distance Calculations
- [ ] Batch Operations
- [ ] Other: 

## Benchmark Results

### Baseline (Previous Version)
```
Operation: 
Throughput: 
Memory Usage: 
Version: 
```

### Current Version
```
Operation: 
Throughput: 
Memory Usage: 
Version: 
```

### Performance Delta
- Throughput change: ___%
- Memory change: ___%

## Reproduction Steps
1. Build with: `swift build -c release --product vectorcore-bench`
2. Run a warm-up, then measure: `.build/release/vectorcore-bench --suites dot --dims 512 --samples 5 --min-time 0.2 --run-seed 1 --format json --out /tmp/vectorcore-benchmark.json`
3. Repeat on the baseline revision with the same cases and seed; attach raw results and summarize medians/percentiles. See [benchmark guidance](https://github.com/gifton/VectorCore/blob/main/CONTRIBUTING.md#performance-changes).

## Environment
- Platform: 
- Architecture (arm64/x86_64): 
- Hardware model and memory:
- OS and Xcode versions:
- Warm-up and sample counts:
- Power mode and thermal conditions:
- Baseline/current commit IDs:
- Swift Version: 
- Build Configuration: 
- Optimization Flags Used: 

## Analysis
<!-- Any profiling or analysis you've done -->
- Instruments traces: 
- Hot spots identified: 
- Suspected cause: 

## Additional Context
<!-- Any other relevant information -->
