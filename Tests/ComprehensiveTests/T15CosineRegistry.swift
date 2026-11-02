import Foundation
@testable import VectorCore

// T15 cosine implementation registry (class closure, T15 ruling section 7).
//
// `T15CosineRegistry.sites` lists every cosine source site, one entry per function body: the Sources and Benchmarks
// bodies that `Scripts/ci/tests/test_cosine_registry.py` discovers (a function or computed property whose name
// contains "cosine", a body that uses a cosine clamp idiom or calls `calculateCosineDistance`, and every body declared
// in a type whose name contains "Cosine"). The guard parses the string literals between the BEGIN and END markers and
// fails on a discovered site that is missing here (unregistered) or a listed site that no longer exists (stale).
// Each site names how its exceptional behaviour is pinned:
//   .table(prefixes)   rows of the T2 exceptional table (T15CosineExceptionalTableTests) whose id starts with a prefix;
//   .selection(routes) T1 routes (T15CosineSelectionTests, `T15SelectionRoute` raw values) that reach the body;
//   .external(test)    a test outside this target (`path::Suite.test`), checked to exist by the guard;
//   .exempt(reason)    not a distance or similarity computation (reviewed reason);
//   .tableAndSelection(prefixes, routes) both of the first two, for a body that rows and T1 routes reach (P02 phase C
//                      review m3: the cosine clamp shared by the public matrix entries and the `.gram` route).
// `T15CosineExceptionalTableTests.registryCoverage` checks every .table prefix matches an evaluated row and every
// .selection route exists (both for .tableAndSelection).

enum T15SiteCoverage: Sendable {
    case table([String])
    case selection([String])
    case tableAndSelection([String], [String])
    case external(String)
    case exempt(String)
}

enum T15CosineRegistry {
    static let sites: [(site: String, coverage: T15SiteCoverage)] = [
        // BEGIN T15 COSINE SITES
        ("Sources/VectorCore/Operations/Kernels/CosineKernels.swift::calculateCosineDistance", .table(["helper.H", "CK.fused."])),
        ("Sources/VectorCore/Operations/Kernels/CosineKernels.swift::fused", .table(["CK.fused."])),
        ("Sources/VectorCore/Operations/Kernels/CosineKernels.swift::fusedSquaredNorm",
         .exempt("squared norm in CosineKernels.fused's accumulator layout; computes no distance")),
        ("Sources/VectorCore/Operations/Kernels/CosineKernels.swift::fusedCandidate", .table(["topkKernel.fused.512"])),
        ("Sources/VectorCore/Operations/Kernels/CosineKernels.swift::fusedPair",
         .table(["BK.fused.384.pair", "BK.fused.512.pair", "BK.fused.768.pair", "BK.fused.1536.pair"])),
        ("Sources/VectorCore/Operations/Kernels/CosineKernels.swift::distance384_fused", .table(["CK.fused.384"])),
        ("Sources/VectorCore/Operations/Kernels/CosineKernels.swift::distance512_fused", .table(["CK.fused.512"])),
        ("Sources/VectorCore/Operations/Kernels/CosineKernels.swift::distance768_fused", .table(["CK.fused.768"])),
        ("Sources/VectorCore/Operations/Kernels/CosineKernels.swift::distance1536_fused", .table(["CK.fused.1536"])),
        ("Sources/VectorCore/Operations/Kernels/CosineKernels.swift::distance384_preNormalized", .table(["CK.pre.384"])),
        ("Sources/VectorCore/Operations/Kernels/CosineKernels.swift::distance512_preNormalized", .table(["CK.pre.512"])),
        ("Sources/VectorCore/Operations/Kernels/CosineKernels.swift::distance768_preNormalized", .table(["CK.pre.768"])),
        ("Sources/VectorCore/Operations/Kernels/CosineKernels.swift::distance1536_preNormalized", .table(["CK.pre.1536"])),
        ("Sources/VectorCore/Operations/DistanceMetrics.swift::name", .exempt("CosineDistance metric name string")),
        ("Sources/VectorCore/Operations/DistanceMetrics.swift::relevanceScore",
         .exempt("maps a distance to a relevance score; computes no distance")),
        // T14 phase 3 removed the six concrete CosineDistance overloads; concrete calls bind this witness (concrete.* rows).
        ("Sources/VectorCore/Operations/DistanceMetrics.swift::distance", .table(["witness.", "concrete."])),
        ("Sources/VectorCore/Operations/Kernels/NormCache.swift::batch_cosine_preNormalized_512", .table(["helpers.pre.512"])),
        ("Sources/VectorCore/Operations/Kernels/NormCache.swift::batch_cosine_preNormalized_768", .table(["helpers.pre.768"])),
        ("Sources/VectorCore/Operations/Kernels/NormCache.swift::batch_cosine_preNormalized_1536", .table(["helpers.pre.1536"])),
        ("Sources/VectorCore/Operations/Kernels/NormCache.swift::buildNormalized_512",
         .exempt("normalizes candidates; computes no distance")),
        ("Sources/VectorCore/Operations/Kernels/NormCache.swift::buildNormalized_768",
         .exempt("normalizes candidates; computes no distance")),
        ("Sources/VectorCore/Operations/Kernels/NormCache.swift::buildNormalized_1536",
         .exempt("normalizes candidates; computes no distance")),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels.swift::range_cosine_preNorm_384", .table(["BK.pre.384."])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels.swift::range_cosine_preNorm_512", .table(["BK.pre.512."])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels.swift::range_cosine_preNorm_768", .table(["BK.pre.768."])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels.swift::range_cosine_preNorm_1536", .table(["BK.pre.1536."])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels.swift::range_cosine_fused_384", .table(["BK.fused.384."])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels.swift::range_cosine_fused_512", .table(["BK.fused.512."])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels.swift::range_cosine_fused_768", .table(["BK.fused.768."])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels.swift::range_cosine_fused_1536", .table(["BK.fused.1536."])),
        ("Sources/VectorCore/Operations/Kernels/TopKSelectionKernels.swift::range_topk_cosine_preNorm_512", .table(["topkKernel.pre.512"])),
        ("Sources/VectorCore/Operations/Kernels/TopKSelectionKernels.swift::range_topk_cosine_fused_512", .table(["topkKernel.fused.512"])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift::transformToDistance", .table(["SoA.pre."])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift::cosine_fused_blocked_2way",
         .table(["SoA.fused.512.2way", "SoA.fused.768.2way", "SoA.fused.1536.2way"])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift::computeDistance", .table(["SoA.fused.512.2way"])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift::cosine_fused_blocked_4way",
         .table(["SoA.fused.512.4way", "SoA.fused.768.4way", "SoA.fused.1536.4way"])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift::computeDistance", .table(["SoA.fused.512.4way"])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift::cosine_fused_tiled",
         .table(["SoA.fused.512.tiled", "SoA.fused.768.tiled", "SoA.fused.1536.tiled"])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift::computeDistance", .table(["SoA.fused.512.tiled"])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift::cosine_prenorm_blocked", .table(["SoA.pre."])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift::batchCosineDistance512", .table(["SoA.fused.512.safe"])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift::batchCosineDistance768", .table(["SoA.fused.768."])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift::batchCosineDistance1536", .table(["SoA.fused.1536."])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift::batchCosineDistancePreNorm512", .table(["SoA.pre.512."])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift::batchCosineDistancePreNorm768", .table(["SoA.pre.768."])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift::batchCosineDistancePreNorm1536", .table(["SoA.pre.1536."])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift::batchCosineDistance512", .table(["SoA.fused.512.2way"])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift::batchCosineDistance768", .table(["SoA.fused.768.2way"])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift::batchCosineDistance1536", .table(["SoA.fused.1536.2way"])),
        ("Sources/VectorCore/Operations/Kernels/BatchKernels_SoA.swift::batchCosineDistance512Safe", .table(["SoA.fused.512.safe"])),
        ("Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift::cosineFP16Core", .table(["fp16.core."])),
        ("Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift::cosineMixedCore", .table(["fp16.q32c16.", "fp16.q16c32."])),
        ("Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift::cosine512", .table(["fp16.pair.512"])),
        ("Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift::cosine768", .table(["fp16.pair.768"])),
        ("Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift::cosine1536", .table(["fp16.pair.1536"])),
        ("Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift::cosine512", .table(["fp16.q32c16.512"])),
        ("Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift::cosine768", .table(["fp16.q32c16.768"])),
        ("Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift::cosine1536", .table(["fp16.q32c16.1536"])),
        ("Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift::cosine512", .table(["fp16.q16c32.512"])),
        ("Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift::cosine768", .table(["fp16.q16c32.768"])),
        ("Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift::cosine1536", .table(["fp16.q16c32.1536"])),
        ("Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift::range_cosine_mixed_core", .table(["mixedRange."])),
        ("Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift::range_cosine_mixed_512", .table(["mixedRange.512."])),
        ("Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift::range_cosine_mixed_768", .table(["mixedRange.768."])),
        ("Sources/VectorCore/Operations/Kernels/MixedPrecisionKernels.swift::range_cosine_mixed_1536", .table(["mixedRange.1536."])),
        ("Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift::cosine_generic", .table(["int8."])),
        ("Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift::cosine512", .table(["int8.512."])),
        ("Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift::cosine768", .table(["int8.768."])),
        ("Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift::cosine1536", .table(["int8.1536."])),
        ("Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift::cosineDistance", .table(["int8.512."])),
        ("Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift::cosineDistance", .table(["int8.768."])),
        ("Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift::cosineDistance", .table(["int8.1536."])),
        ("Sources/VectorCore/Operations/MatrixDistance.swift::cosineDistanceMatrix", .table(["gemm."])),
        ("Sources/VectorCore/Operations/MatrixDistance.swift::cosineDistanceMatrix", .selection(["matrixPairwiseGEMM"])),
        // P02 phase C (RD13) and its review (m3): the clamp shared by the public entries ("gemm." rows, T1 route
        // `matrixPairwiseGEMM`) and the explicit `.gram` route of `Operations.findNearestBatch` ("gram." rows, T1 route
        // `operationsBatchGEMM`).
        ("Sources/VectorCore/Operations/MatrixDistance.swift::cosinePostprocess",
         .tableAndSelection(["gemm.", "gram."], ["matrixPairwiseGEMM", "operationsBatchGEMM"])),
        ("Sources/VectorCore/Operations/MatrixDistance.swift::cosineDistanceMatrix", .table(["gemm."])),
        ("Sources/VectorCore/Operations/TransformOperations.swift::cosineSimilarityMatrix", .table(["transform.768"])),
        ("Sources/VectorCore/Platform/ArraySIMDProvider.swift::cosineSimilarity", .table(["sim.array.default"])),
        ("Sources/VectorCore/Platform/ArraySIMDProvider.swift::cosineSimilarity", .table(["sim.array.swift"])),
        ("Sources/VectorCore/Platform/ArraySIMDProvider.swift::cosineSimilarity", .table(["sim.array.accelerate"])),
        ("Sources/VectorCore/Protocols/VectorProtocol.swift::cosineSimilarity", .table(["sim.vp."])),
        ("Sources/VectorCore/Protocols/VectorProtocol.swift::cosineDistance", .table(["vpext."])),
        ("Sources/VectorCore/Vectors/Vector384Optimized.swift::cosineSimilarity", .table(["sim.opt.384"])),
        ("Sources/VectorCore/Vectors/Vector512Optimized.swift::cosineSimilarity", .table(["sim.opt.512"])),
        ("Sources/VectorCore/Vectors/Vector768Optimized.swift::cosineSimilarity", .table(["sim.opt.768"])),
        ("Sources/VectorCore/Vectors/Vector1536Optimized.swift::cosineSimilarity", .table(["sim.opt.1536"])),
        ("Sources/VectorCore/Operations/TopKSelection.swift::nearestCosinePreNormalized512", .table(["topk.pre.512"])),
        ("Sources/VectorCore/Operations/TopKSelection.swift::nearestCosineFused512", .table(["topk.fused.512"])),
        ("Sources/VectorCore/Operations/Operations.swift::topk_cosine_fused_512", .selection(["operationsV512"])),
        ("Sources/VectorCore/Operations/Operations.swift::batchKernelDistances_cosineFused", .selection(["operationsV512"])),
        ("Sources/VectorCore/Operations/Operations.swift::batchKernelDistances_cosineFused", .selection(["operationsV768"])),
        ("Sources/VectorCore/Operations/Operations.swift::batchKernelDistances_cosineFused", .selection(["operationsV1536"])),
        ("Sources/VectorCore/Operations/BatchOperations.swift::computeDistances_cosine_fused_serial", .selection(["batchV512"])),
        ("Sources/VectorCore/Operations/BatchOperations.swift::computeDistances_cosine_fused_serial", .selection(["batchV768"])),
        ("Sources/VectorCore/Operations/BatchOperations.swift::computeDistances_cosine_fused_serial", .selection(["batchV1536"])),
        ("Sources/VectorCore/Operations/BatchOperations.swift::computeDistances_cosine_fused_parallel", .selection(["batchV512Parallel"])),
        ("Sources/VectorCore/Operations/BatchOperations.swift::computeDistances_cosine_fused_parallel", .selection(["batchV512Parallel"])),
        ("Sources/VectorCore/Operations/BatchOperations.swift::computeDistances_cosine_fused_parallel", .selection(["batchV512Parallel"])),
        ("Sources/VectorCore/Integration/VectorCollection.swift::searchCosine", .selection(["collectionSearchCosine"])),
        ("Benchmarks/VectorCoreBench/Utils/Correctness.swift::cosineDist",
         .external(
             "Tests/BenchmarkContractTests/SwiftTestingEntryTests.swift::T15DoubleRefCosineTests.doubleRefCosineFollowsThePolicy")),
        ("Benchmarks/VectorCoreBench/Suites/BatchBench.swift::runBatchOptimizedCosineFused",
         .exempt("benchmark driver over the registered range-fused kernels")),
        ("Benchmarks/VectorCoreBench/Suites/BatchBench.swift::runBatchOptimizedCosineFused",
         .exempt("benchmark driver over the registered range-fused kernels")),
        ("Benchmarks/VectorCoreBench/Suites/BatchBench.swift::runBatchOptimizedCosineFused",
         .exempt("benchmark driver over the registered range-fused kernels")),
        ("Benchmarks/VectorCoreBench/Suites/BatchBench.swift::runBatchOptimizedCosinePreNorm",
         .exempt("benchmark driver over the registered range pre-normalized kernels")),
        ("Benchmarks/VectorCoreBench/Suites/BatchBench.swift::runBatchOptimizedCosinePreNorm",
         .exempt("benchmark driver over the registered range pre-normalized kernels")),
        ("Benchmarks/VectorCoreBench/Suites/BatchBench.swift::runBatchOptimizedCosinePreNorm",
         .exempt("benchmark driver over the registered range pre-normalized kernels")),
        ("Benchmarks/VectorCoreBench/Suites/BatchBench.swift::runBatchCosineFP16_512",
         .exempt("benchmark driver over the registered mixed-precision range kernel")),
        ("Benchmarks/VectorCoreBench/Suites/BatchBench.swift::runBatchCosineFP16_768",
         .exempt("benchmark driver over the registered mixed-precision range kernel")),
        ("Benchmarks/VectorCoreBench/Suites/BatchBench.swift::runBatchCosineFP16_1536",
         .exempt("benchmark driver over the registered mixed-precision range kernel"))
        // END T15 COSINE SITES
    ]
}
