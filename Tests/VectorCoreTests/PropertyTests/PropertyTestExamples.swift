import XCTest
@testable import VectorCore

/// Example tests demonstrating the property testing framework
final class PropertyTestExamples: XCTestCase {
    
    // MARK: - Using the Property Testing Helper
    
    func testVectorAdditionCommutativityWithHelper() {
        PropertyTest.testCommutative(
            PropertyTest.Config(iterations: 200),
            generator: { rng in
                PropertyTest.Gen.vectorPair(
                    type: Vector<Dim128>.self,
                    constraint: .normal(-100...100),
                    using: &rng
                )
            },
            operation: (+),
            equality: { vectorsApproximatelyEqual($0, $1, tolerance: 1e-5) },
            message: "Vector addition should be commutative"
        )
    }
    
    func testScalarMultiplicationAssociativityWithHelper() {
        PropertyTest.forAll(
            PropertyTest.Config.thorough,
            generator: { rng in
                let a = PropertyTest.Gen.float(in: -10...10, using: &rng)
                let b = PropertyTest.Gen.float(in: -10...10, using: &rng)
                let v = PropertyTest.Gen.vector(
                    type: Vector<Dim64>.self,
                    constraint: .normal(-50...50),
                    using: &rng
                )
                return (a, b, v)
            },
            property: { a, b, v in
                let left = (a * b) * v
                let right = a * (b * v)
                return vectorsApproximatelyEqual(left, right, tolerance: 1e-4)
            },
            message: "Scalar multiplication should be associative: (a*b)*v = a*(b*v)"
        )
    }
    
    func testNormalizationIdempotence() {
        PropertyTest.forAll(
            generator: { rng in
                PropertyTest.Gen.vector(
                    type: Vector<Dim256>.self,
                    constraint: .nonZero(-100...100),
                    using: &rng
                )
            },
            property: { v in
                guard v.magnitude > 1e-6 else { return true }
                
                let once = v.normalized()
                let twice = once.normalized()
                return vectorsApproximatelyEqual(once, twice, tolerance: 1e-5)
            },
            message: "Normalization should be idempotent: normalize(normalize(v)) = normalize(v)"
        )
    }
    
    // MARK: - Testing with Different Vector Types
    
    func testDynamicVectorProperties() {
        let dimensions = [32, 64, 128, 256, 512, 768, 1536]
        
        for dim in dimensions {
            PropertyTest.forAll(
                PropertyTest.Config(iterations: 50),
                generator: { rng in
                    let v1 = PropertyTest.Gen.dynamicVector(
                        dimension: dim,
                        constraint: .normal(-50...50),
                        using: &rng
                    )
                    let v2 = PropertyTest.Gen.dynamicVector(
                        dimension: dim,
                        constraint: .normal(-50...50),
                        using: &rng
                    )
                    return (v1, v2)
                },
                property: { v1, v2 in
                    // Test multiple properties at once
                    let addCommutes = dynamicVectorsApproximatelyEqual(v1 + v2, v2 + v1)
                    let dotCommutes = approximately(v1.dotProduct(v2), v2.dotProduct(v1))
                    let distanceSymmetric = approximately(v1.distance(to: v2), v2.distance(to: v1))
                    
                    return addCommutes && dotCommutes && distanceSymmetric
                },
                message: "DynamicVector[\(dim)] operations should satisfy basic properties"
            )
        }
    }
    
    // MARK: - Testing Special Constraints
    
    func testUnitVectorProperties() {
        PropertyTest.forAll(
            generator: { rng in
                // Generate a random vector and normalize it
                let v = PropertyTest.Gen.vector(
                    type: Vector<Dim128>.self,
                    constraint: .nonZero(-100...100),
                    using: &rng
                )
                return v.magnitude > 1e-6 ? v.normalized() : Vector<Dim128>(repeating: 1).normalized()
            },
            property: { unitVector in
                let magnitude = unitVector.magnitude
                let selfDot = unitVector.dotProduct(unitVector)
                
                // Unit vector properties
                return approximately(magnitude, 1.0, tolerance: 1e-5) &&
                       approximately(selfDot, 1.0, tolerance: 1e-5) &&
                       unitVector.lInfinityNorm <= 1.0 + 1e-5
            },
            message: "Unit vectors should have magnitude 1 and satisfy related properties"
        )
    }
    
    func testOrthogonalVectorProperties() {
        PropertyTest.forAll(
            PropertyTest.Config(iterations: 50),
            generator: { rng in
                // Generate two orthogonal vectors
                let dim = 128
                var v1Values = [Float](repeating: 0, count: dim)
                var v2Values = [Float](repeating: 0, count: dim)
                
                // Make them orthogonal by construction
                v1Values[0] = PropertyTest.Gen.float(constraint: .nonZero(-10...10), using: &rng)
                v2Values[1] = PropertyTest.Gen.float(constraint: .nonZero(-10...10), using: &rng)
                
                return (Vector<Dim128>(v1Values), Vector<Dim128>(v2Values))
            },
            property: { v1, v2 in
                let dot = v1.dotProduct(v2)
                let cosine = v1.cosineSimilarity(to: v2)
                
                // Orthogonal vectors have dot product 0 and cosine similarity 0
                return approximately(dot, 0, tolerance: 1e-6) &&
                       approximately(cosine, 0, tolerance: 1e-6)
            },
            message: "Orthogonal vectors should have zero dot product and cosine similarity"
        )
    }
    
    // MARK: - Testing Numerical Stability
    
    func testNumericalStabilityProperties() {
        // Test with extreme values
        PropertyTest.forAll(
            PropertyTest.Config(iterations: 50),
            generator: { rng in
                let constraint: VectorConstraint
                switch Int.random(in: 0...2, using: &rng) {
                case 0:
                    constraint = .normal(1e-10...1e-8)  // Very small
                case 1:
                    constraint = .normal(1e8...1e10)    // Very large
                default:
                    constraint = .normal(-1e10...1e10)  // Mixed range
                }
                
                return PropertyTest.Gen.vectorPair(
                    type: Vector<Dim64>.self,
                    constraint: constraint,
                    using: &rng
                )
            },
            property: { v1, v2 in
                // Operations should not produce NaN or Infinity
                let sum = v1 + v2
                let diff = v1 - v2
                let magnitude = v1.magnitude
                let distance = v1.distance(to: v2)
                
                let noNaN = !sum.toArray().contains { $0.isNaN } &&
                           !diff.toArray().contains { $0.isNaN } &&
                           !magnitude.isNaN &&
                           !distance.isNaN
                
                let noInf = !sum.toArray().contains { $0.isInfinite } &&
                           !diff.toArray().contains { $0.isInfinite } &&
                           !magnitude.isInfinite &&
                           !distance.isInfinite
                
                return noNaN && noInf
            },
            message: "Operations on extreme values should not produce NaN or Infinity"
        )
    }
    
    // MARK: - Testing Complex Properties
    
    func testDistributivityOfDotProductOverAddition() {
        PropertyTest.forAll(
            PropertyTest.Config(iterations: 200),
            generator: { rng in
                PropertyTest.Gen.vectorTriple(
                    type: Vector<Dim128>.self,
                    constraint: .normal(-10...10),
                    using: &rng
                )
            },
            property: { a, b, c in
                // a·(b + c) = a·b + a·c
                let left = a.dotProduct(b + c)
                let right = a.dotProduct(b) + a.dotProduct(c)
                return approximately(left, right, tolerance: 1e-4)
            },
            message: "Dot product should distribute over vector addition"
        )
    }
    
    func testParallelogramLaw() {
        PropertyTest.forAll(
            generator: { rng in
                PropertyTest.Gen.vectorPair(
                    type: Vector<Dim64>.self,
                    constraint: .normal(-50...50),
                    using: &rng
                )
            },
            property: { a, b in
                // ||a + b||² + ||a - b||² = 2(||a||² + ||b||²)
                let sumMagSquared = (a + b).magnitude * (a + b).magnitude
                let diffMagSquared = (a - b).magnitude * (a - b).magnitude
                let left = sumMagSquared + diffMagSquared
                
                let aMagSquared = a.magnitude * a.magnitude
                let bMagSquared = b.magnitude * b.magnitude
                let right = 2 * (aMagSquared + bMagSquared)
                
                return approximately(left, right, tolerance: 1e-3)
            },
            message: "Vectors should satisfy the parallelogram law"
        )
    }
    
    // MARK: - Shrinking Example
    
    func testWithMinimalFailingCase() {
        // Example of finding minimal failing case
        var minimalFailure: (Vector<Dim32>, Float)? = nil
        var minimalMagnitude = Float.infinity
        
        PropertyTest.forAll(
            PropertyTest.Config(iterations: 1000),
            generator: { rng in
                let v = PropertyTest.Gen.vector(
                    type: Vector<Dim32>.self,
                    constraint: .normal(-1000...1000),
                    using: &rng
                )
                let scalar = PropertyTest.Gen.float(in: -100...100, using: &rng)
                return (v, scalar)
            },
            property: { v, scalar in
                // Artificial property that might fail
                let scaled = v * scalar
                let rescaled = scaled / scalar
                
                // Check if division by scalar recovers original (might fail for scalar ≈ 0)
                if abs(scalar) > 1e-6 {
                    let equal = vectorsApproximatelyEqual(v, rescaled, tolerance: 1e-4)
                    if !equal && v.magnitude < minimalMagnitude {
                        minimalFailure = (v, scalar)
                        minimalMagnitude = v.magnitude
                    }
                    return equal
                }
                return true
            },
            message: "Scaling and rescaling should recover original vector"
        )
        
        if let (v, s) = minimalFailure {
            print("Minimal failing case found: vector magnitude = \(v.magnitude), scalar = \(s)")
        }
    }
}