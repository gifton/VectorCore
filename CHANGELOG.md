# Changelog

All notable changes to VectorCore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Performance baseline system with automated regression detection
- `BenchmarkAdapter` for bridging swift-benchmark package with baseline system
- `PerformanceRegressionSuite` for comprehensive performance testing
- CI/CD integration for automatic performance monitoring on PRs
- Dimension extraction from benchmark names for better metric categorization

### Fixed
- Compilation errors in test suite
- VectorError pattern matching updated to use `.kind` property
- PropertyTest API usage updated for compatibility
- Method naming consistency (euclideanDistance → distance, dot → dotProduct)

## [0.1.0] - 2025-01-15

### Added
- Initial release of VectorCore
- Generic `Vector<D: Dimension>` type with compile-time dimensions
- `DynamicVector` for runtime-determined dimensions
- 3-tier storage system optimized for different vector sizes:
  - `SmallVectorStorage` (1-64 elements) - stack allocated
  - `MediumVectorStorage` (65-512 elements) - heap allocated with fixed buffer
  - `LargeVectorStorage` (513+ elements) - dynamically sized
- Copy-on-Write (COW) optimization for efficient value semantics
- SIMD acceleration via Accelerate framework
- Quality metrics:
  - `.sparsity` - percentage of near-zero elements
  - `.entropy` - Shannon entropy for information content
  - `.quality` - composite quality score
- Binary serialization with CRC32 checksums
- Base64 encoding/decoding support
- Comprehensive mathematical operations:
  - Element-wise arithmetic
  - Dot product
  - Norms (L1, L2, L∞)
  - Distance metrics
  - Statistical functions
- Swift 6 compatibility with full `Sendable` conformance
- Extensive test coverage including edge cases
- Performance benchmarks
- Documentation and examples

### Performance
- Zero-copy operations where possible
- Optimized memory alignment for SIMD operations
- Efficient storage selection based on vector size
- Minimal overhead for small vectors

### Swift Integration
- Natural Swift API design
- Support for common Swift patterns
- Integration with standard library protocols
- Type-safe dimension handling