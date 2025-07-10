Building for production...
[0/2] Write swift-version--58304C5D6DBC2206.txt
Build of product 'VectorCoreBenchmarks' complete! (0.26s)
VectorCore Performance Benchmarks
=================================
Platform: Version 15.4.1 (Build 24E263)
Processor: 8 cores
Date: 2025-07-06 01:26:23 +0000

=== Distance Metric Benchmarks ===


--- Vector128 (SIMD optimized) ---
Euclidean (single) (dim=128):
  Total time: 0.046s
  Avg time: 0.004567ms
  Throughput: 218981 ops/sec
Euclidean (batch-100) (dim=128):
  Total time: 0.049s
  Avg time: 0.489759ms
  Throughput: 2042 ops/sec
Cosine (single) (dim=128):
  Total time: 0.046s
  Avg time: 0.004614ms
  Throughput: 216727 ops/sec
Cosine (batch-100) (dim=128):
  Total time: 0.048s
  Avg time: 0.477279ms
  Throughput: 2095 ops/sec
DotProduct (single) (dim=128):
  Total time: 0.046s
  Avg time: 0.004558ms
  Throughput: 219418 ops/sec
DotProduct (batch-100) (dim=128):
  Total time: 0.047s
  Avg time: 0.472620ms
  Throughput: 2116 ops/sec
Manhattan (single) (dim=128):
  Total time: 0.046s
  Avg time: 0.004589ms
  Throughput: 217927 ops/sec
Manhattan (batch-100) (dim=128):
  Total time: 0.047s
  Avg time: 0.473360ms
  Throughput: 2113 ops/sec
Chebyshev (single) (dim=128):
  Total time: 0.046s
  Avg time: 0.004606ms
  Throughput: 217089 ops/sec
Chebyshev (batch-100) (dim=128):
  Total time: 0.048s
  Avg time: 0.475811ms
  Throughput: 2102 ops/sec
Hamming (single) (dim=128):
  Total time: 0.046s
  Avg time: 0.004611ms
  Throughput: 216859 ops/sec
Hamming (batch-100) (dim=128):
  Total time: 0.048s
  Avg time: 0.475590ms
  Throughput: 2103 ops/sec
Minkowski(p=3) (single) (dim=128):
  Total time: 0.055s
  Avg time: 0.005541ms
  Throughput: 180483 ops/sec
Minkowski(p=3) (batch-100) (dim=128):
  Total time: 0.057s
  Avg time: 0.569971ms
  Throughput: 1754 ops/sec
Jaccard (single) (dim=128):
  Total time: 0.046s
  Avg time: 0.004612ms
  Throughput: 216845 ops/sec
Jaccard (batch-100) (dim=128):
  Total time: 0.048s
  Avg time: 0.476489ms
  Throughput: 2099 ops/sec

--- Vector256 (SIMD optimized) ---
Euclidean (single) (dim=256):
  Total time: 0.090s
  Avg time: 0.009004ms
  Throughput: 111057 ops/sec
Euclidean (batch-100) (dim=256):
  Total time: 0.093s
  Avg time: 0.929780ms
  Throughput: 1076 ops/sec
Cosine (single) (dim=256):
  Total time: 0.091s
  Avg time: 0.009074ms
  Throughput: 110211 ops/sec
Cosine (batch-100) (dim=256):
  Total time: 0.092s
  Avg time: 0.916250ms
  Throughput: 1091 ops/sec
DotProduct (single) (dim=256):
  Total time: 0.091s
  Avg time: 0.009077ms
  Throughput: 110167 ops/sec
DotProduct (batch-100) (dim=256):
  Total time: 0.094s
  Avg time: 0.939560ms
  Throughput: 1064 ops/sec
Manhattan (single) (dim=256):
  Total time: 0.090s
  Avg time: 0.008986ms
  Throughput: 111278 ops/sec
Manhattan (batch-100) (dim=256):
  Total time: 0.091s
  Avg time: 0.911360ms
  Throughput: 1097 ops/sec
Chebyshev (single) (dim=256):
  Total time: 0.090s
  Avg time: 0.009047ms
  Throughput: 110539 ops/sec
Chebyshev (batch-100) (dim=256):
  Total time: 0.092s
  Avg time: 0.917140ms
  Throughput: 1090 ops/sec
Hamming (single) (dim=256):
  Total time: 0.090s
  Avg time: 0.009044ms
  Throughput: 110572 ops/sec
Hamming (batch-100) (dim=256):
  Total time: 0.092s
  Avg time: 0.921310ms
  Throughput: 1085 ops/sec
Minkowski(p=3) (single) (dim=256):
  Total time: 0.110s
  Avg time: 0.011038ms
  Throughput: 90599 ops/sec
Minkowski(p=3) (batch-100) (dim=256):
  Total time: 0.112s
  Avg time: 1.115080ms
  Throughput: 897 ops/sec
Jaccard (single) (dim=256):
  Total time: 0.091s
  Avg time: 0.009063ms
  Throughput: 110335 ops/sec
Jaccard (batch-100) (dim=256):
  Total time: 0.092s
  Avg time: 0.916620ms
  Throughput: 1091 ops/sec

--- Vector512 (SIMD optimized) ---
Euclidean (single) (dim=512):
  Total time: 0.178s
  Avg time: 0.017831ms
  Throughput: 56081 ops/sec
Euclidean (batch-100) (dim=512):
  Total time: 0.182s
  Avg time: 1.820670ms
  Throughput: 549 ops/sec
Cosine (single) (dim=512):
  Total time: 0.181s
  Avg time: 0.018055ms
  Throughput: 55386 ops/sec
Cosine (batch-100) (dim=512):
  Total time: 0.185s
  Avg time: 1.852000ms
  Throughput: 540 ops/sec
DotProduct (single) (dim=512):
  Total time: 0.178s
  Avg time: 0.017846ms
  Throughput: 56036 ops/sec
DotProduct (batch-100) (dim=512):
  Total time: 0.180s
  Avg time: 1.800320ms
  Throughput: 555 ops/sec
Manhattan (single) (dim=512):
  Total time: 0.181s
  Avg time: 0.018124ms
  Throughput: 55174 ops/sec
Manhattan (batch-100) (dim=512):
  Total time: 0.213s
  Avg time: 2.132471ms
  Throughput: 469 ops/sec
Chebyshev (single) (dim=512):
  Total time: 0.296s
  Avg time: 0.029575ms
  Throughput: 33813 ops/sec
Chebyshev (batch-100) (dim=512):
  Total time: 0.225s
  Avg time: 2.254050ms
  Throughput: 444 ops/sec
Hamming (single) (dim=512):
  Total time: 0.226s
  Avg time: 0.022562ms
  Throughput: 44322 ops/sec
Hamming (batch-100) (dim=512):
  Total time: 0.233s
  Avg time: 2.331489ms
  Throughput: 429 ops/sec
Minkowski(p=3) (single) (dim=512):
  Total time: 0.288s
  Avg time: 0.028845ms
  Throughput: 34668 ops/sec
Minkowski(p=3) (batch-100) (dim=512):
  Total time: 0.297s
  Avg time: 2.968550ms
  Throughput: 337 ops/sec
Jaccard (single) (dim=512):
  Total time: 0.232s
  Avg time: 0.023242ms
  Throughput: 43025 ops/sec
Jaccard (batch-100) (dim=512):
  Total time: 0.234s
  Avg time: 2.341311ms
  Throughput: 427 ops/sec


=== Vector Operation Benchmarks ===

Addition (dim=256):
  Total time: 0.003s
  Avg time: 0.000317ms
  Throughput: 3155510 ops/sec
Subtraction (dim=256):
  Total time: 0.003s
  Avg time: 0.000324ms
  Throughput: 3088248 ops/sec
Scalar Multiply (dim=256):
  Total time: 0.002s
  Avg time: 0.000154ms
  Throughput: 6493736 ops/sec
Scalar Divide (dim=256):
  Total time: 0.002s
  Avg time: 0.000163ms
  Throughput: 6131127 ops/sec
Dot Product (dim=256):
  Total time: 0.003s
  Avg time: 0.000261ms
  Throughput: 3837248 ops/sec
Magnitude (dim=256):
  Total time: 0.002s
  Avg time: 0.000235ms
  Throughput: 4251702 ops/sec
Normalize (dim=256):
  Total time: 0.004s
  Avg time: 0.000427ms
  Throughput: 2340310 ops/sec
Distance (dim=256):
  Total time: 0.006s
  Avg time: 0.000550ms
  Throughput: 1816542 ops/sec
Cosine Similarity (dim=256):
  Total time: 0.007s
  Avg time: 0.000706ms
  Throughput: 1417019 ops/sec
Element-wise Multiply (dim=256):
  Total time: 0.003s
  Avg time: 0.000327ms
  Throughput: 3057073 ops/sec
Element-wise Divide (dim=256):
  Total time: 0.004s
  Avg time: 0.000371ms
  Throughput: 2696868 ops/sec
L1 Norm (dim=256):
  Total time: 0.004s
  Avg time: 0.000399ms
  Throughput: 2505633 ops/sec
L∞ Norm (dim=256):
  Total time: 0.001s
  Avg time: 0.000130ms
  Throughput: 7662929 ops/sec
Mean (dim=256):
  Total time: 0.001s
  Avg time: 0.000120ms
  Throughput: 8305552 ops/sec
Variance (dim=256):
  Total time: 0.005s
  Avg time: 0.000518ms
  Throughput: 1931257 ops/sec
Softmax (dim=256):
  Total time: 0.006s
  Avg time: 0.000638ms
  Throughput: 1567407 ops/sec
Clamp (dim=256):
  Total time: 0.002s
  Avg time: 0.000156ms
  Throughput: 6418707 ops/sec


=== Batch Operation Benchmarks ===

k-NN (k=1) (dim=256):
  Total time: 1.201s
  Avg time: 120.130503ms
  Throughput: 8 ops/sec
k-NN (k=10) (dim=256):
  Total time: 1.205s
  Avg time: 120.451295ms
  Throughput: 8 ops/sec
k-NN (k=100) (dim=256):
  Total time: 1.181s
  Avg time: 118.126702ms
  Throughput: 8 ops/sec
Pairwise Distances (100x100) (dim=256):
  Total time: 0.550s
  Avg time: 54.950893ms
  Throughput: 18 ops/sec
Batch Normalize (10k vectors) (dim=256):
  Total time: 0.096s
  Avg time: 9.648192ms
  Throughput: 104 ops/sec


=== Storage Type Benchmarks ===

Vector128 (SIMD optimized) (dim=128):
  Total time: 0.025s
  Avg time: 0.000245ms
  Throughput: 4074315 ops/sec
DynamicVector (array-based) (dim=128):
  Total time: 0.004s
  Avg time: 0.000045ms
  Throughput: 22306568 ops/sec
  → Vector128 is 0.2x faster than DynamicVector


=== Memory Usage Analysis ===

Dimension 32:
  Data size: 128 bytes
  Total size: ~136 bytes
  Overhead: 8 bytes
  Efficiency: 94.1%
Dimension 64:
  Data size: 256 bytes
  Total size: ~264 bytes
  Overhead: 8 bytes
  Efficiency: 97.0%
Dimension 128:
  Data size: 512 bytes
  Total size: ~520 bytes
  Overhead: 8 bytes
  Efficiency: 98.5%
Dimension 256:
  Data size: 1024 bytes
  Total size: ~1032 bytes
  Overhead: 8 bytes
  Efficiency: 99.2%
Dimension 512:
  Data size: 2048 bytes
  Total size: ~16 bytes
  Overhead: -2032 bytes
  Efficiency: 12800.0%
Dimension 768:
  Data size: 3072 bytes
  Total size: ~3096 bytes
  Overhead: 24 bytes
  Efficiency: 99.2%
Dimension 1536:
  Data size: 6144 bytes
  Total size: ~6168 bytes
  Overhead: 24 bytes
  Efficiency: 99.6%
Dimension 2048:
  Data size: 8192 bytes
  Total size: ~8216 bytes
  Overhead: 24 bytes
  Efficiency: 99.7%


=== Implementation Comparison ===

Naive Euclidean (dim=256):
  Total time: 0.000s
  Avg time: 0.000000ms
  Throughput: inf ops/sec
Optimized Euclidean (dim=256):
  Total time: 0.010s
  Avg time: 0.001021ms
  Throughput: 979234 ops/sec
  → Optimized is 0.0x faster than naive
Squared Euclidean (dim=256):
  Total time: 0.000s
  Avg time: 0.000001ms
  Throughput: 1421797966 ops/sec
  → Squared is 1451.9x faster (no sqrt)


Benchmark completed!
