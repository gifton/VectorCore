# Measuring Performance

> **Reading time:** 10 minutes
> **Prerequisites:** None (standalone guide)

---

## The Concept

Performance measurement seems simple: run the code, time it. But getting *accurate, reproducible* measurements is surprisingly hard.

Common measurement mistakes:

1. **Measuring debug builds** (10-100x slower than release)
2. **Ignoring warmup** (first runs are slower due to caching)
3. **Not accounting for variance** (system noise affects results)
4. **Measuring too little** (single runs are not statistically meaningful)

---

## Why It Matters

### The Misleading Benchmark

```swift
func benchmark() {
    let start = Date()

    // Do some work
    var sum = 0
    for i in 0..<1_000_000 {
        sum += i
    }

    let elapsed = Date().timeIntervalSince(start)
    print("Took: \(elapsed) seconds")
}
```

Problems with this:
- `Date()` has microsecond precision at best
- Single measurement doesn't account for variance
- No warmup—first run includes JIT/cache effects
- The optimizer might eliminate the loop entirely!

### A Better Approach

```swift
import Foundation

func benchmarkCorrectly() {
    let iterations = 100
    var times = [Double]()

    // Warmup (not measured)
    for _ in 0..<10 {
        _ = doWork()
    }

    // Measured runs
    for _ in 0..<iterations {
        let start = DispatchTime.now()
        _ = doWork()
        let end = DispatchTime.now()

        let nanos = Double(end.uptimeNanoseconds - start.uptimeNanoseconds)
        times.append(nanos)
    }

    // Statistics
    times.sort()
    let median = times[times.count / 2]
    let p90 = times[Int(Double(times.count) * 0.9)]
    let min = times.first!
    let max = times.last!

    print("Median: \(median / 1_000_000) ms")
    print("P90:    \(p90 / 1_000_000) ms")
    print("Min:    \(min / 1_000_000) ms")
    print("Max:    \(max / 1_000_000) ms")
}
```

---

## The Technique

### Rule 1: Always Use Release Builds

```bash
# Debug build (slow, unoptimized)
swift build

# Release build (fast, optimized)
swift build -c release
```

Debug builds include:
- Bounds checking on all array access
- Debug assertions
- No inlining
- No optimization passes

Release builds can be 10-100x faster for numerical code.

### Rule 2: Use High-Resolution Timers

| Timer | Precision | Use Case |
|-------|-----------|----------|
| `Date()` | ~milliseconds | Wall clock time |
| `CFAbsoluteTimeGetCurrent()` | ~microseconds | General benchmarking |
| `DispatchTime.now()` | ~nanoseconds | Microbenchmarks |
| `mach_absolute_time()` | ~nanoseconds | Lowest overhead |

For VectorCore operations (100ns range), use `DispatchTime` or `mach_absolute_time`.

### Rule 3: Include Warmup Runs

```swift
// Warmup: Populate caches, trigger any lazy initialization
for _ in 0..<10 {
    _ = vector.dotProduct(other)
}

// Now measure
let start = DispatchTime.now()
for _ in 0..<1000 {
    _ = vector.dotProduct(other)
}
let elapsed = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
let perOperation = Double(elapsed) / 1000.0
```

First runs are slower because:
- Instruction cache is cold
- Data cache is cold
- Branch predictors haven't learned
- Memory might not be paged in

### Rule 4: Report Percentiles, Not Averages

Averages are skewed by outliers (GC pauses, OS interrupts):

```swift
let times = [100, 101, 102, 5000, 103]  // One outlier
let average = times.reduce(0, +) / times.count  // 1081 ns (misleading!)
let median = times.sorted()[times.count / 2]    // 102 ns (representative)
```

Report:
- **Median** (P50): The typical case
- **P90/P99**: Tail latency (important for real systems)
- **Min**: Best achievable performance
- **Max**: Worst-case performance

### Rule 5: Prevent Dead Code Elimination

The optimizer may remove code whose result isn't used:

```swift
// ❌ Optimizer might eliminate this loop
for _ in 0..<1000 {
    _ = vector.dotProduct(other)  // Result unused, might be eliminated
}

// ✅ Use the result somehow
var sum: Float = 0
for _ in 0..<1000 {
    sum += vector.dotProduct(other)
}
// Use sum at the end
print(sum)  // Or assign to a global, or call a @noinline function
```

Or use `@inline(never)`:

```swift
@inline(never)
func blackhole<T>(_ value: T) {
    // Prevents the compiler from optimizing away the computation
}

for _ in 0..<1000 {
    blackhole(vector.dotProduct(other))
}
```

---

## In VectorCore

VectorCore includes a benchmarking module:

**📍 See:** `Sources/VectorCoreBenchmarking/`

```swift
// Example usage
let benchmark = Benchmark(name: "512-dim dot product")

benchmark.measure {
    _ = vector.dotProduct(other)
}

print(benchmark.report())
// Outputs: min, median, p90, p99, max, std dev
```

### Running VectorCore Benchmarks

```bash
# Build and run the benchmark tool
swift build -c release
./.build/release/vectorcore-bench

# Run specific suites
./.build/release/vectorcore-bench --suites dot,euclidean

# Vary dimensions
./.build/release/vectorcore-bench --dims 512,768,1536
```

---

## Using Instruments

For deeper analysis, use Xcode Instruments:

### Time Profiler
Shows where CPU time is spent:

```bash
# Profile a command
xcrun xctrace record --template 'Time Profiler' \
    --launch ./.build/release/vectorcore-bench
```

### Allocations
Shows memory allocation patterns:

```bash
xcrun xctrace record --template 'Allocations' \
    --launch ./.build/release/your-program
```

### What to Look For

| Symptom | Possible Cause |
|---------|---------------|
| Many small allocations | Object churn, box/unbox |
| High time in `malloc`/`free` | Too many allocations |
| Time in `objc_msgSend` | Dynamic dispatch overhead |
| Time in `swift_retain`/`release` | Reference counting overhead |
| Long functions you didn't write | Library inefficiency |

---

## Benchmarking Checklist

Before sharing benchmark results:

- [ ] Release build (`-c release`)
- [ ] Warmup runs included
- [ ] Multiple iterations (100+)
- [ ] Reported percentiles (not just average)
- [ ] Prevented dead code elimination
- [ ] Run on quiet system (no background tasks)
- [ ] Documented hardware and OS version
- [ ] Disabled Turbo Boost if measuring absolute times

---

## Key Takeaways

1. **Always benchmark release builds.** Debug can be 100x slower.

2. **Include warmup runs.** First runs aren't representative.

3. **Report percentiles, not averages.** Median and P99 matter more.

4. **Prevent dead code elimination.** Use results or `@inline(never)` blackhole.

5. **Use high-resolution timers.** `DispatchTime.now()` for nanosecond precision.

6. **Measure, don't guess.** Profile before optimizing.

---

## Next Up

Now that you can measure, let's understand what makes code fast or slow at the memory level:

**[→ Cache-Friendly Code](./02-Cache-Friendly-Code.md)**
