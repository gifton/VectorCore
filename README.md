# VectorCore

[![Swift](https://img.shields.io/badge/Swift-6.0-orange.svg?style=flat)](https://swift.org)
[![Platforms](https://img.shields.io/badge/Platforms-macOS%2014%2B%20|%20iOS%2017%2B%20|%20tvOS%2017%2B%20|%20watchOS%2010%2B%20|%20visionOS%201%2B-blue.svg?style=flat)](https://swift.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)
[![SPM](https://img.shields.io/badge/SPM-compatible-brightgreen.svg?style=flat)](https://swift.org/package-manager/)
[![CI](https://github.com/gifton/VectorCore/actions/workflows/ci.yml/badge.svg)](https://github.com/gifton/VectorCore/actions/workflows/ci.yml)

VectorCore provides CPU vector math for Swift on Apple platforms, with generic
fixed dimensions, specialized SIMD vector types, distance metrics, and batch
operations. It has no third-party package dependencies; its implementation uses
Swift, a `VectorCoreC` target, and Apple's Accelerate framework.

## Installation

Add VectorCore to your Swift Package Manager dependencies:

```swift
dependencies: [
    .package(url: "https://github.com/gifton/VectorCore.git", from: "0.3.3")
]
```

Then add it to your target:

```swift
.target(
    name: "YourTarget",
    dependencies: ["VectorCore"]
)
```

## Quick Start

This complete program can be used as an executable target's `main.swift`:

```swift
import VectorCore

let first = Vector512Optimized(repeating: 1.0)
let second = Vector512Optimized(repeating: 2.0)

let dot: Float = first.dotProduct(second)
let distance: Float = first.euclideanDistance(to: second)
let unit = try first.normalized().get()

// Generic vectors encode their dimension in the type.
let embedding = Vector<Dim768>(repeating: 0.5)

print("Dot product: \(dot), distance: \(distance)")
print("Unit magnitude: \(unit.magnitude), embedding magnitude: \(embedding.magnitude)")
```

## API and performance

- `Vector<D>` encodes fixed dimensions in its type; `DynamicVector` accepts
  runtime dimensions. Runtime input and buffer validation still matter.
- `Vector512Optimized`, `Vector768Optimized`, and `Vector1536Optimized` provide
  specialized SIMD storage and kernels for common embedding dimensions.
- `Operations`, `BatchOperations`, and `MatrixDistance` expose vector, batch,
  and query-by-candidate matrix operations. Routing depends on operation,
  input size, configuration, and provider availability.
- Built-in metrics include Euclidean, cosine, Manhattan, Chebyshev, Hamming,
  Minkowski, and dot-product distance.
- Provider overrides use `@TaskLocal`; see the
  [API overview](Docs/API_Overview_Map.md) for the public surface.

Allocation behavior depends on the type and operation. Generic
[`DimensionStorage`](Sources/VectorCore/Storage/DimensionStorage.swift) defaults
to managed heap storage above 16 elements. Results, copies, and batch scratch
storage may allocate. There is no package-wide allocation-free guarantee.

Measure performance in Release on your workload. The
[Top-K benchmark record](Benchmarks/TopKNaNContract/RESULTS.md) gives a specific
before/after comparison with hardware, samples, and limitations; it does not
measure allocations or establish a universal speedup. See
[contribution benchmark instructions](CONTRIBUTING.md#performance-changes).

## Numerical behavior and unsafe buffers

Floating-point operations can overflow, propagate non-finite values, and produce
slightly different results across kernels. Validation and error behavior are
API-specific: check the selected API's contract when handling zero norms,
NaNs, infinities, or untrusted dimensions. A throwing initializer or checked
operation does not imply that every subsequent operation validates its inputs.

`TopKSelection` orders numeric scores, including infinities, before NaNs.
Its default tie policy prefers smaller original indices; exact ties include
signed zeros and pairs of NaNs. For positive `k`, selection retains NaNs when
needed to return `min(k, count)` candidates. These contracts are exercised by
[TopKNaNContractTests](Tests/ComprehensiveTests/TopKNaNContractTests.swift).
Metric computation can round differently before selection; this ordering rule
does not promise identical results for different computed scores.

For unsafe APIs, callers must provide valid counts, initialized elements,
required alignment, and sufficient storage; respect aliasing and exclusivity
rules. Pointers borrowed by a closure must not escape that closure. Keep owned
allocations alive until all consumers finish, and coordinate mutation and
ownership transfer across tasks. See
[Memory Alignment](Docs/Memory_Alignment.md) and
[UnifiedVectorBuffer](Sources/VectorCore/Storage/UnifiedVectorBuffer.swift).

The package enables Swift 6 strict concurrency checking, but some buffer and
pool types use `@unchecked Sendable`, which relies on manually maintained
invariants. `MemoryPool` is a class with explicit synchronization. Neither
conformance nor compiler checking makes arbitrary shared pointer mutation safe.

## GPU integration

VectorCore contains CPU implementations and buffer/provider interfaces for
integration with the separate VectorAccelerate package. `UnifiedVectorBuffer`
provides a scoped contiguous read view; it does **not** imply page alignment.
`PageAlignedBuffer` and opt-in page-aligned SoA storage provide additional
allocation contracts. GPU import, ownership, and synchronization require the
consumer to follow those contracts. See
[Package Boundaries](Docs/Package_Boundaries.md) and
[SoA Layout Contract](Docs/SoA_Layout_Contract.md).

## Requirements and compatibility

The manifest requires Swift tools 6.0 and declares these minimum deployment
targets: macOS 14, iOS 17, tvOS 17, watchOS 10, and visionOS 1.

Deployment targets are distinct from tested configurations. The
[September 6 verification record](Docs/verification-topk-nan-contract-2026-09-05.md)
records full Debug and Release tests on Apple M3 Max, macOS 26.5.2, Swift 6.3.3.
The [CI workflow](.github/workflows/ci.yml) defines the current macOS test and
Apple platform compile matrix; consult its run results for coverage on a given
commit. Simulator compilation does not establish runtime behavior on devices.
Linux is not a supported or tested platform for this package.

VectorCore is pre-1.0. Minor releases may change APIs or numerical behavior;
patch releases aim to preserve source compatibility but may correct documented
bugs. Read release notes and test updates against your workload. See
[Contributing](CONTRIBUTING.md#compatibility-and-support) for support policy.

## Documentation and contributing

- [API Overview](Docs/API_Overview_Map.md)
- [Package Boundaries](Docs/Package_Boundaries.md)
- [Performance Guide](Docs/Performance_Guide.md)
- [Memory Alignment](Docs/Memory_Alignment.md)
- [Contributing](CONTRIBUTING.md)
- [Security reporting](SECURITY.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Maintainer and fork guidance](Docs/Maintaining.md)

## License

VectorCore is released under the MIT License. See [LICENSE](LICENSE) for details.
