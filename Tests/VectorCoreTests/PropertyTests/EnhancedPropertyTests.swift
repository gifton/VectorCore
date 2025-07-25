import XCTest
@testable import VectorCore

/// Enhanced property-based tests for vector operations with comprehensive coverage
final class EnhancedPropertyTests: XCTestCase {
    
    // MARK: - Configuration
    
    let config = PropertyTest.Config(iterations: 100, seed: 42, verbose: false)
    let quickConfig = PropertyTest.Config.quick
    let thoroughConfig = PropertyTest.Config.thorough
    
    // MARK: - Arithmetic Properties
    
    func testVectorAdditionProperties() {
        PropertyTest.forAll(
            config,
            generator: { (generator: inout SeededRandomGenerator) in
                let (v1, v2) = PropertyTest.Gen.vectorPair(
                    type: Vector768.self,
                    constraint: .normal(-100...100),
                    using: &generator
                )
                let v3 = PropertyTest.Gen.vector(
                    type: Vector768.self,
                    constraint: .normal(-100...100),
                    using: &generator
                )
                return (v1, v2, v3)
            },
            property: { (v1, v2, v3) in
                // Commutative: a + b = b + a
                let sum1 = v1 + v2
                let sum2 = v2 + v1
                guard vectorsApproximatelyEqual(sum1, sum2, tolerance: 1e-5) else {
                    return false
                }
                
                // Associative: (a + b) + c = a + (b + c)
                let leftAssoc = (v1 + v2) + v3
                let rightAssoc = v1 + (v2 + v3)
                guard vectorsApproximatelyEqual(leftAssoc, rightAssoc, tolerance: 1e-5) else {
                    return false
                }
                
                // Identity: a + 0 = a
                let zero = Vector768.zero
                let identity = v1 + zero
                guard vectorsApproximatelyEqual(identity, v1, tolerance: 1e-7) else {
                    return false
                }
                
                // Inverse: a + (-a) = 0
                let negated = -v1
                let shouldBeZero = v1 + negated
                for i in 0..<shouldBeZero.scalarCount {
                    if abs(shouldBeZero[i]) > 1e-5 {
                        return false
                    }
                }
                
                return true
            },
            message: "Vector addition properties failed"
        )
    }
    
    func testScalarMultiplicationProperties() {
        PropertyTest.forAll(
            config,
            generator: { (generator: inout SeededRandomGenerator) in
                let v = PropertyTest.Gen.vector(
                    type: Vector768.self,
                    constraint: .normal(-100...100),
                    using: &generator
                )
                let v2 = PropertyTest.Gen.vector(
                    type: Vector768.self,
                    constraint: .normal(-100...100),
                    using: &generator
                )
                let s1 = PropertyTest.Gen.float(in: -10...10, using: &generator)
                let s2 = PropertyTest.Gen.float(in: -10...10, using: &generator)
                return (v, v2, s1, s2)
            },
            property: { (v, v2, s1, s2) in
                // Distributive over vector addition
                let distributed1 = s1 * (v + v2)
                let distributed2 = (s1 * v) + (s1 * v2)
                guard vectorsApproximatelyEqual(distributed1, distributed2, tolerance: 1e-4) else {
                    return false
                }
                
                // Distributive over scalar addition
                let scalar_distributed1 = (s1 + s2) * v
                let scalar_distributed2 = (s1 * v) + (s2 * v)
                guard vectorsApproximatelyEqual(scalar_distributed1, scalar_distributed2, tolerance: 1e-4) else {
                    return false
                }
                
                // Associative: (s1 * s2) * v = s1 * (s2 * v)
                let assoc1 = (s1 * s2) * v
                let assoc2 = s1 * (s2 * v)
                guard vectorsApproximatelyEqual(assoc1, assoc2, tolerance: 1e-4) else {
                    return false
                }
                
                // Identity: 1 * v = v
                let identity = 1.0 * v
                guard vectorsApproximatelyEqual(identity, v, tolerance: 1e-7) else {
                    return false
                }
                
                // Zero: 0 * v = 0
                let zeroMult = 0.0 * v
                for i in 0..<zeroMult.scalarCount {
                    if abs(zeroMult[i]) > 1e-7 {
                        return false
                    }
                }
                
                return true
            },
            message: "Scalar multiplication properties failed"
        )
    }
    
    // MARK: - Distance Metric Properties
    
    func testDistanceMetricProperties() {
        PropertyTest.forAll(
            config,
            generator: { (generator: inout SeededRandomGenerator) in
                let v1 = PropertyTest.Gen.vector(
                    type: Vector768.self,
                    constraint: .normal(-50...50),
                    using: &generator
                )
                let v2 = PropertyTest.Gen.vector(
                    type: Vector768.self,
                    constraint: .normal(-50...50),
                    using: &generator
                )
                let v3 = PropertyTest.Gen.vector(
                    type: Vector768.self,
                    constraint: .normal(-50...50),
                    using: &generator
                )
                return (v1, v2, v3)
            },
            property: { (v1, v2, v3) in
                // Test Euclidean distance
                let euclidean12 = v1.distance(to: v2)
                let euclidean21 = v2.distance(to: v1)
                
                // Symmetry: d(a,b) = d(b,a)
                guard abs(euclidean12 - euclidean21) < 1e-5 else { return false }
                
                // Non-negativity: d(a,b) >= 0
                guard euclidean12 >= 0 else { return false }
                
                // Identity: d(a,a) = 0
                let selfDistance = v1.distance(to: v1)
                guard abs(selfDistance) < 1e-7 else { return false }
                
                // Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
                let d_ac = v1.distance(to: v3)
                let d_ab = v1.distance(to: v2)
                let d_bc = v2.distance(to: v3)
                guard d_ac <= d_ab + d_bc + 1e-5 else { return false }
                
                // Test Manhattan distance
                // Manhattan distance using L1 norm of difference
                let manhattan12 = (v1 - v2).l1Norm
                let manhattan21 = (v2 - v1).l1Norm
                
                // Symmetry
                guard abs(manhattan12 - manhattan21) < 1e-5 else { return false }
                
                // Relationship: Manhattan >= Euclidean
                guard manhattan12 >= euclidean12 - 1e-5 else { return false }
                
                return true
            },
            message: "Distance metric properties failed"
        )
    }
    
    // MARK: - Normalization Properties
    
    func testNormalizationProperties() {
        PropertyTest.forAll(
            config,
            generator: { (generator: inout SeededRandomGenerator) in
                let v = PropertyTest.Gen.vector(
                    type: Vector768.self,
                    constraint: .nonZero(-100...100),
                    using: &generator
                )
                let scalar = PropertyTest.Gen.float(
                    constraint: .positive,
                    using: &generator
                )
                return (v, scalar)
            },
            property: { (v, scalar) in
                let normalized = v.normalized()
                
                // Unit length: ||normalized|| = 1
                let magnitude = normalized.magnitude
                guard abs(magnitude - 1.0) < 1e-5 else { return false }
                
                // Direction preservation: normalized = v / ||v||
                let manualNorm = v * (1.0 / v.magnitude)
                guard vectorsApproximatelyEqual(normalized, manualNorm, tolerance: 1e-5) else { return false }
                
                // Idempotent: normalize(normalize(v)) = normalize(v)
                let doubleNorm = normalized.normalized()
                guard vectorsApproximatelyEqual(normalized, doubleNorm, tolerance: 1e-6) else { return false }
                
                // Scalar multiplication property
                let scaled = v * scalar
                let scaledNorm = scaled.normalized()
                guard vectorsApproximatelyEqual(normalized, scaledNorm, tolerance: 1e-5) else { return false }
                
                return true
            },
            message: "Normalization properties failed"
        )
    }
    
    // MARK: - Dot Product Properties
    
    func testDotProductProperties() {
        PropertyTest.forAll(
            config,
            generator: { (generator: inout SeededRandomGenerator) in
                let v1 = PropertyTest.Gen.vector(
                    type: Vector768.self,
                    constraint: .normal(-10...10),
                    using: &generator
                )
                let v2 = PropertyTest.Gen.vector(
                    type: Vector768.self,
                    constraint: .normal(-10...10),
                    using: &generator
                )
                let v3 = PropertyTest.Gen.vector(
                    type: Vector768.self,
                    constraint: .normal(-10...10),
                    using: &generator
                )
                let s = PropertyTest.Gen.float(in: -5...5, using: &generator)
                return (v1, v2, v3, s)
            },
            property: { (v1, v2, v3, s) in
                // Commutative: a · b = b · a
                let dot12 = v1.dotProduct(v2)
                let dot21 = v2.dotProduct(v1)
                guard abs(dot12 - dot21) < 1e-5 else { return false }
                
                // Distributive: a · (b + c) = a · b + a · c
                let sum23 = v2 + v3
                let dotSum = v1.dotProduct(sum23)
                let dotDistributed = v1.dotProduct(v2) + v1.dotProduct(v3)
                guard abs(dotSum - dotDistributed) < 1e-4 else { return false }
                
                // Scalar multiplication: (s * a) · b = s * (a · b)
                let scaled = s * v1
                let dotScaled = scaled.dotProduct(v2)
                let scalarDot = s * v1.dotProduct(v2)
                guard abs(dotScaled - scalarDot) < 1e-4 else { return false }
                
                // Self dot product is squared magnitude
                let selfDot = v1.dotProduct(v1)
                let magnitudeSquared = v1.magnitude * v1.magnitude
                guard abs(selfDot - magnitudeSquared) < 1e-4 else { return false }
                
                // Orthogonality test
                if abs(dot12) < 1e-6 {
                    // If nearly orthogonal, cosine similarity should be near 0
                    let cosine = v1.cosineSimilarity(to: v2)
                    guard abs(cosine) < 1e-5 else { return false }
                }
                
                return true
            },
            message: "Dot product properties failed"
        )
    }
    
    // MARK: - Edge Case Properties
    
    func testEdgeCaseHandling() {
        PropertyTest.forAll(
            quickConfig,
            generator: { (generator: inout SeededRandomGenerator) in
                // Test with very small values
                let small = PropertyTest.Gen.vector(
                    type: Vector768.self,
                    constraint: .small,
                    using: &generator
                )
                
                // Test with very large values
                let large = PropertyTest.Gen.vector(
                    type: Vector768.self,
                    constraint: .large,
                    using: &generator
                )
                
                return (small, large)
            },
            property: { (small, large) in
                // Operations should not underflow
                let squared = small .* small  // Element-wise multiplication
                for i in 0..<squared.scalarCount {
                    if squared[i].isSubnormal {
                        return false
                    }
                }
                
                // Check for overflow in magnitude calculation
                let magnitude = large.magnitude
                if magnitude.isInfinite {
                    return false
                }
                
                // Test zero vector handling
                let zero = Vector768.zero
                if zero.magnitude != 0.0 {
                    return false
                }
                
                // Normalization of zero should work (returns zero vector)
                let normalizedZero = zero.normalized()
                if normalizedZero.magnitude != 0.0 {
                    return false
                }
                
                return true
            },
            message: "Edge case handling failed"
        )
    }
    
    // MARK: - Performance Regression Properties
    
    func testPerformanceInvariants() {
        // Test that optimizations don't break mathematical properties
        PropertyTest.forAll(
            quickConfig,
            generator: { (generator: inout SeededRandomGenerator) in
            let v1 = PropertyTest.Gen.vector(
                type: Vector768.self,
                constraint: .normal(-100...100),
                using: &generator
            )
            let v2 = PropertyTest.Gen.vector(
                type: Vector768.self,
                constraint: .normal(-100...100),
                using: &generator
            )
            
            // Measure operations
            let addTime = measureAverageTime(iterations: 100) {
                _ = v1 + v2
            }
            
            let dotTime = measureAverageTime(iterations: 100) {
                _ = v1.dotProduct(v2)
            }
            
            let distTime = measureAverageTime(iterations: 100) {
                _ = v1.distance(to: v2)
            }
            
            // Basic performance sanity checks
            // Addition should be faster than distance calculation
            XCTAssertLessThan(addTime, distTime * 2,
                             "Addition should not be much slower than distance")
            
            // Dot product should be reasonably fast
            XCTAssertLessThan(dotTime, distTime,
                             "Dot product should be faster than full distance")
            
            return (v1, v2)
            },
            property: { _ in
                // Performance tests always pass if they don't crash
                return true
            },
            message: "Performance invariants should hold"
        )
    }
    
    // MARK: - Batch Operation Properties
    
    func testBatchOperationConsistency() {
        PropertyTest.forAll(
            quickConfig,
            generator: { (generator: inout SeededRandomGenerator) in
            let vectors = (0..<10).map { _ in
                PropertyTest.Gen.vector(
                    type: Vector768.self,
                    constraint: .normal(-50...50),
                    using: &generator
                )
            }
            
            let query = PropertyTest.Gen.vector(
                type: Vector768.self,
                constraint: .normal(-50...50),
                using: &generator
            )
            
            // Individual distance calculations
            let individualDistances = vectors.map { $0.distance(to: query) }
            
            // Find nearest individually
            var minDistance = Float.infinity
            var nearestIndex = 0
            for (index, distance) in individualDistances.enumerated() {
                if distance < minDistance {
                    minDistance = distance
                    nearestIndex = index
                }
            }
            
            // Batch operation should give same result
            // Note: Array doesn't have findNearest method, using sync batch operations
            let batchResults = SyncBatchOperations.findNearest(to: query, in: vectors, k: 1)
            if let nearest = batchResults.first {
                XCTAssertEqual(nearest.index, nearestIndex,
                              "Batch and individual nearest should match")
                XCTAssertEqual(nearest.distance, minDistance, accuracy: 1e-5,
                              "Batch and individual distance should match")
            } else {
                XCTFail("Batch operation should find nearest vector")
            }
            
            return (vectors, query)
            },
            property: { _ in
                // Consistency tests always pass if assertions don't fail
                return true
            },
            message: "Batch operations should be consistent with individual operations"
        )
    }
}

// MARK: - Additional Test Utilities

extension EnhancedPropertyTests {
    
    /// Run property test with multiple constraints
    func testWithMultipleConstraints<T>(
        _ test: (VectorConstraint, inout SeededRandomGenerator) throws -> T
    ) rethrows {
        let constraints: [VectorConstraint] = [
            .normal(-100...100),
            .normalized,
            .unit,
            .nonNegative,
            .small,
            .large
        ]
        
        var generator = SeededRandomGenerator(seed: 42)
        
        for constraint in constraints {
            _ = try test(constraint, &generator)
        }
    }
}

// MARK: - Vector Constraint Extensions

extension VectorConstraint {
    static let small = VectorConstraint.normal(-1e-6...1e-6)
    static let large = VectorConstraint.normal(1e6...1e8)
}

// Helper to get float for vector constraint
private func floatForVectorConstraint(
    _ constraint: VectorConstraint,
    using generator: inout SeededRandomGenerator
) -> Float {
    switch constraint {
    case .normal(let range):
        return Float.random(in: range, using: &generator)
    case .normalized:
        return Float.random(in: -1...1, using: &generator)
    case .unit:
        // For unit vectors, we'll generate normalized components
        return Float.random(in: -1...1, using: &generator)
    case .nonNegative:
        return Float.random(in: 0...100, using: &generator)
    default:
        return Float.random(in: -100...100, using: &generator)
    }
}