---
name: Performance regression
about: Report a performance degradation
title: '[PERF] '
labels: 'performance', 'regression'
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
1. Build with: `./Scripts/build_optimized.sh`
2. Run benchmark: 
3. Compare with baseline

## Environment
- Platform: 
- Architecture (arm64/x86_64): 
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