# Shared source-file planning map

[Planning draft](README.md) · [Work items](WORK_ITEMS.md)

Generated from audited source/read references, so this is a conservative conflict map, not permission to edit every listed file. The dispatch brief must add any test/doc/generated files it will write and reserve them too. Disjoint source lists alone do not establish interface independence. Shared edits serialize even across worktrees.

| Source file | Candidate owners |
|---|---|
| `Sources/VectorCore/Factory/VectorTypeFactory.swift` | S02, D02 |
| `Sources/VectorCore/Integration/SearchResult.swift` | S03, R05 |
| `Sources/VectorCore/Integration/VectorCollection.swift` | R01, R02, R05, D01 |
| `Sources/VectorCore/Interop/CKernels.swift` | V02, D02 |
| `Sources/VectorCore/LinearAlgebra/PCA.swift` | V04, L02, L03, L04 |
| `Sources/VectorCore/LinearAlgebra/SwiftLinearAlgebraProvider.swift` | V04, L01, L04 |
| `Sources/VectorCore/ManifoldLearning/FuzzySimplicialSet.swift` | L03, L04 |
| `Sources/VectorCore/ManifoldLearning/UMAP.swift` | L03, L04 |
| `Sources/VectorCore/Operations/BatchOperations.swift` | R03, E01, E02, E03, E06, N03, P01 |
| `Sources/VectorCore/Operations/ExecutionOperations.swift` | R03, E02 |
| `Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift` | V04, R03 |
| `Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift` | V04, O05, I02, Q02, Q03, Q04, D02 |
| `Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift` | O04, N01 |
| `Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift` | S01, S03, Q01, Q02, B01 |
| `Sources/VectorCore/Operations/Kernels/SoAFP16Cache.swift` | I02, D02 |
| `Sources/VectorCore/Operations/Operations.swift` | S02, E05, P01, P02, D02 |
| `Sources/VectorCore/Operations/TopKSelection.swift` | S03, R01, P01 |
| `Sources/VectorCore/Operations/TransformOperations.swift` | R03, E02 |
| `Sources/VectorCore/Operations/VectorCoreOperations.swift` | O03, D02 |
| `Sources/VectorCore/Operations/VectorMath.swift` | N04, P01, D02 |
| `Sources/VectorCore/Optimization/OptimizationAttributes.swift` | V04, E01, D02 |
| `Sources/VectorCore/Protocols/BatchKernelProvider.swift` | E02, D01 |
| `Sources/VectorCore/Protocols/ComputeProvider.swift` | E01, E02 |
| `Sources/VectorCore/Protocols/VectorProtocol.swift` | S01, S02, N01, N02, D02 |
| `Sources/VectorCore/Providers/CPUComputeProvider.swift` | E01, E02 |
| `Sources/VectorCore/Storage/DimensionStorage.swift` | O03, O05 |
| `Sources/VectorCore/Utilities/TieredStorage.swift` | O01, O02 |
| `Sources/VectorCore/Utilities/TriangularMatrix.swift` | O01, O02, E02 |
| `Sources/VectorCore/Vectors/DynamicVector.swift` | V04, S01, O03, B01 |
| `Sources/VectorCore/Vectors/Vector1536Optimized.swift` | S01, P01 |
| `Sources/VectorCore/Vectors/Vector384Optimized.swift` | S01, O04, P01 |
| `Sources/VectorCore/Vectors/Vector512Optimized.swift` | V04, S01, I02, P01 |
| `Sources/VectorCore/Vectors/Vector768Optimized.swift` | S01, P01 |
| `Sources/VectorCoreBenchmarking/EnvCapture.swift` | V02, D02 |
