# Changelog

All notable changes to VectorCore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-11-26

### Added

#### Dim384 Optimized Vector Type
- **`Vector384Optimized`**: New SIMD-accelerated 384-dimensional vector type for MiniLM and Sentence-BERT embeddings
- Full kernel support: dot product, Euclidean distance, cosine distance, normalization
- SoA (Structure-of-Arrays) layout support via `SoA384` typealias
- `Dim384` dimension type with `VectorTypeFactory` integration
- Performance: ~75ns dot product, ~90ns distance computation

#### `normalizedUnchecked()` API
- New `normalizedUnchecked() -> Self` method on all vector types
- Bypasses zero-vector validation for maximum performance in hot paths
- Debug-only assertions catch misuse during development
- Optimized SIMD kernels for Vector384/512/768/1536Optimized
- Estimated 15-20% faster than `normalized()` in hot paths

#### SIMD Manhattan Distance
- Generic `ManhattanDistance` now uses SIMD4 vectorization
- Processes 4 elements per iteration with scalar tail loop
- Works with `DynamicVector` and arbitrary dimensions
- Estimated 3-4x faster than previous scalar implementation

#### Top-K Selection Public API
- **`TopKSelection`** enum with efficient k-nearest neighbor selection
- `select(k:from:)` - O(n log k) heap-based selection from pre-computed distances
- `nearest(k:query:candidates:metric:)` - Generic k-NN with any distance metric
- Optimized variants: `nearestEuclidean384/512/768/1536()`, `nearestCosinePreNormalized512()`, `nearestCosineFused512()`, `nearestDotProduct512()`
- `batchNearest()` for multi-query batch operations
- **`TopKResult`** struct with Collection and Codable conformance
- Adaptive algorithm: heap for k < n/10, sort for larger k

#### VectorIndex Integration Protocols
- **`IndexableVector`** protocol extending `VectorProtocol` with optimization hints
  - `isNormalized: Bool` - indicates pre-normalized vectors
  - `cachedMagnitude: Float?` - cached magnitude value
  - `isApproximatelyNormalized` computed property
- **`NormalizationHint<V>`** wrapper for tracking normalization status
- **`SearchResult<ID>`** - Single search result with id, distance, score
- **`SearchResults<ID>`** - Collection with metadata (candidatesSearched, searchTimeNanos, isExhaustive)
- **`VectorCollection`** protocol for searchable vector collections
  - Default brute-force `search(query:k:metric:)` implementation
  - Convenience methods: `searchEuclidean()`, `searchCosine()`, `searchDotProduct()`
- **`MutableVectorCollection`** protocol with add/remove/update operations
- **`SimpleVectorCollection<V>`** - Reference implementation for testing

### Changed

- `VectorTypeFactory` now returns `Vector384Optimized` for dimension 384
- `optimalDimension(for:)` includes 384 in supported dimensions
- Updated documentation to reflect 384-dimension support

### Tests

- 27 new tests for Top-K Selection
- 24 new tests for Integration Protocols
- 9 new tests for `normalizedUnchecked()`
- 7 new tests for SIMD Manhattan Distance
- All 140 tests in relevant suites pass

---

## [0.1.4] - Previous Release

Initial pre-release version with:
- Vector512/768/1536Optimized types
- SIMD-accelerated kernels
- Distance metrics (Euclidean, Cosine, Manhattan, Chebyshev, Dot Product)
- Batch operations with async support
- Mixed-precision computation support
