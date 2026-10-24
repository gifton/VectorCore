//
//  OptimizedDistanceMetrics.swift
//  VectorCore
//
//  The metric factory
//

import Foundation

// T14 removed every concrete `distance`/`batchDistance` overload for Vector512Optimized, Vector768Optimized and
// Vector1536Optimized: the Euclidean, dot-product, Manhattan and Chebyshev ones in phase 2 and the cosine ones in
// phase 3. A call with those concrete types binds the DistanceMetric witness: MetricWitnessEquivalenceTests checks
// that concrete, generic and existential calls agree, and MetricSearchWitnessConsistencyTests that the search APIs
// return the witness's result outside the GEMM routes. The Euclidean and cosine GEMM routes are a different arithmetic
// and are not covered: Operations.findNearestBatch with the explicit BatchSearchCPUStrategy.gram (the default,
// .perQuery, runs the per-candidate routes), and the off-diagonal entries of BatchOperations.pairwiseDistances when
// enableMatrixRouting is set and the count reaches matrixRoutingMinN. A new concrete overload needs a key in MetricWitnessEquivalenceTests.concreteOverloadRegistry
// (Scripts/ci/tests/test_metric_overload_registry.py).

// MARK: - Distance Metric Factory

/// Factory for creating distance metrics with optimized implementations
public struct OptimizedDistanceMetrics {

    /// Available optimized metrics
    public enum MetricType: String, CaseIterable {
        case euclidean
        case cosine
        case dotProduct
        case manhattan
        case chebyshev
        case hamming

        /// Create the corresponding distance metric
        public func createMetric() -> any DistanceMetric<Float> {
            switch self {
            case .euclidean:
                return EuclideanDistance()
            case .cosine:
                return CosineDistance()
            case .dotProduct:
                return DotProductDistance()
            case .manhattan:
                return ManhattanDistance()
            case .chebyshev:
                return ChebyshevDistance()
            case .hamming:
                return HammingDistance()
            }
        }
    }

    /// Create a distance metric by type
    public static func create(_ type: MetricType) -> any DistanceMetric<Float> {
        type.createMetric()
    }

    /// Create a distance metric by name
    public static func create(named name: String) -> (any DistanceMetric<Float>)? {
        MetricType.allCases.first { $0.rawValue.lowercased() == name.lowercased() }?.createMetric()
    }
}
