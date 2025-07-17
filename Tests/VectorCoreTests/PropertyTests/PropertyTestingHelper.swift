import XCTest
@testable import VectorCore

/// Simple property-based testing framework for VectorCore
public struct PropertyTest {
    
    // MARK: - Configuration
    
    public struct Config {
        public let iterations: Int
        public let seed: UInt64?
        public let verbose: Bool
        
        public init(iterations: Int = 100, seed: UInt64? = nil, verbose: Bool = false) {
            self.iterations = iterations
            self.seed = seed
            self.verbose = verbose
        }
        
        public static let `default` = Config()
        public static let quick = Config(iterations: 10)
        public static let thorough = Config(iterations: 1000)
    }
    
    // MARK: - Generators
    
    /// Generate random vectors with various constraints
    public struct Gen {
        
        /// Generate a random Float in the given range
        public static func float(in range: ClosedRange<Float>, 
                               using generator: inout SeededRandomGenerator) -> Float {
            Float.random(in: range, using: &generator)
        }
        
        /// Generate a random Float with special constraints
        public static func float(constraint: FloatConstraint,
                               using generator: inout SeededRandomGenerator) -> Float {
            switch constraint {
            case .normal(let range):
                return float(in: range, using: &generator)
            case .nonZero(let range):
                var value = float(in: range, using: &generator)
                while value == 0 {
                    value = float(in: range, using: &generator)
                }
                return value
            case .positive:
                return float(in: 0.001...1000, using: &generator)
            case .negative:
                return float(in: -1000...(-0.001), using: &generator)
            case .small:
                return float(in: -1e-6...1e-6, using: &generator)
            case .large:
                return float(in: 1e6...1e10, using: &generator)
            case .normalized:
                return float(in: -1...1, using: &generator)
            }
        }
        
        /// Generate a fixed-size vector
        public static func vector<D: Dimension>(
            type: Vector<D>.Type,
            constraint: VectorConstraint = .normal(-100...100),
            using generator: inout SeededRandomGenerator
        ) -> Vector<D> {
            let values = (0..<D.value).map { _ in
                floatForVectorConstraint(constraint, using: &generator)
            }
            return Vector<D>(values)
        }
        
        /// Generate a dynamic vector
        public static func dynamicVector(
            dimension: Int,
            constraint: VectorConstraint = .normal(-100...100),
            using generator: inout SeededRandomGenerator
        ) -> DynamicVector {
            let values = (0..<dimension).map { _ in
                floatForVectorConstraint(constraint, using: &generator)
            }
            return DynamicVector(values)
        }
        
        /// Generate a pair of vectors with same dimension
        public static func vectorPair<D: Dimension>(
            type: Vector<D>.Type,
            constraint: VectorConstraint = .normal(-100...100),
            using generator: inout SeededRandomGenerator
        ) -> (Vector<D>, Vector<D>) {
            return (
                vector(type: type, constraint: constraint, using: &generator),
                vector(type: type, constraint: constraint, using: &generator)
            )
        }
        
        /// Generate a triple of vectors with same dimension
        public static func vectorTriple<D: Dimension>(
            type: Vector<D>.Type,
            constraint: VectorConstraint = .normal(-100...100),
            using generator: inout SeededRandomGenerator
        ) -> (Vector<D>, Vector<D>, Vector<D>) {
            return (
                vector(type: type, constraint: constraint, using: &generator),
                vector(type: type, constraint: constraint, using: &generator),
                vector(type: type, constraint: constraint, using: &generator)
            )
        }
        
        // Helper for vector constraints
        private static func floatForVectorConstraint(
            _ constraint: VectorConstraint,
            using generator: inout SeededRandomGenerator
        ) -> Float {
            switch constraint {
            case .normal(let range):
                return float(in: range, using: &generator)
            case .unit:
                // Generate random unit vector components (will be normalized)
                return float(in: -1...1, using: &generator)
            case .nonZero(let range):
                return float(constraint: .nonZero(range), using: &generator)
            case .orthogonalBasis(let index, let total):
                // Standard basis vector
                return 0 // Will be set to 1 at the specific index
            }
        }
    }
    
    // MARK: - Property Testing Functions
    
    /// Test a property with a single input
    public static func forAll<A>(
        _ config: Config = .default,
        generator: (inout SeededRandomGenerator) -> A,
        property: (A) throws -> Bool,
        message: @autoclosure () -> String,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        var rng = SeededRandomGenerator(seed: config.seed ?? UInt64.random(in: 0...UInt64.max))
        var failures: [(A, Error?)] = []
        
        for i in 0..<config.iterations {
            let input = generator(&rng)
            
            do {
                if try !property(input) {
                    failures.append((input, nil))
                    if config.verbose {
                        print("❌ Failed on iteration \(i): \(input)")
                    }
                }
            } catch {
                failures.append((input, error))
                if config.verbose {
                    print("💥 Error on iteration \(i): \(error)")
                }
            }
        }
        
        if !failures.isEmpty {
            let failureRate = Float(failures.count) / Float(config.iterations) * 100
            let firstFailure = failures.first!
            
            var failureMessage = "\(message()) - Failed \(failures.count)/\(config.iterations) times (%.1f%%)"
            failureMessage = String(format: failureMessage, failureRate)
            
            if let error = firstFailure.1 {
                failureMessage += "\nFirst error: \(error)"
            }
            failureMessage += "\nFirst failing input: \(firstFailure.0)"
            
            if let seed = config.seed {
                failureMessage += "\nSeed: \(seed)"
            }
            
            XCTFail(failureMessage, file: file, line: line)
        }
    }
    
    /// Test a property with two inputs
    public static func forAll<A, B>(
        _ config: Config = .default,
        generator: (inout SeededRandomGenerator) -> (A, B),
        property: (A, B) throws -> Bool,
        message: @autoclosure () -> String,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        forAll(config, generator: generator, property: { tuple in try property(tuple.0, tuple.1) },
               message: message(), file: file, line: line)
    }
    
    /// Test a property with three inputs
    public static func forAll<A, B, C>(
        _ config: Config = .default,
        generator: (inout SeededRandomGenerator) -> (A, B, C),
        property: (A, B, C) throws -> Bool,
        message: @autoclosure () -> String,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        forAll(config, generator: generator, property: { tuple in try property(tuple.0, tuple.1, tuple.2) },
               message: message(), file: file, line: line)
    }
    
    // MARK: - Common Property Patterns
    
    /// Test that an operation is commutative: f(a,b) = f(b,a)
    public static func testCommutative<T: Equatable>(
        _ config: Config = .default,
        generator: (inout SeededRandomGenerator) -> (T, T),
        operation: (T, T) -> T,
        equality: (T, T) -> Bool = (==),
        message: @autoclosure () -> String,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        forAll(config, generator: generator, property: { a, b in
            let ab = operation(a, b)
            let ba = operation(b, a)
            return equality(ab, ba)
        }, message: message(), file: file, line: line)
    }
    
    /// Test that an operation is associative: f(f(a,b),c) = f(a,f(b,c))
    public static func testAssociative<T: Equatable>(
        _ config: Config = .default,
        generator: (inout SeededRandomGenerator) -> (T, T, T),
        operation: (T, T) -> T,
        equality: (T, T) -> Bool = (==),
        message: @autoclosure () -> String,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        forAll(config, generator: generator, property: { a, b, c in
            let left = operation(operation(a, b), c)
            let right = operation(a, operation(b, c))
            return equality(left, right)
        }, message: message(), file: file, line: line)
    }
    
    /// Test that an element is an identity: f(a, identity) = a
    public static func testIdentity<T: Equatable>(
        _ config: Config = .default,
        generator: (inout SeededRandomGenerator) -> T,
        identity: T,
        operation: (T, T) -> T,
        equality: (T, T) -> Bool = (==),
        message: @autoclosure () -> String,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        forAll(config, generator: generator, property: { a in
            let result = operation(a, identity)
            return equality(result, a)
        }, message: message(), file: file, line: line)
    }
}

// MARK: - Constraint Types

public enum FloatConstraint {
    case normal(ClosedRange<Float>)
    case nonZero(ClosedRange<Float>)
    case positive
    case negative
    case small
    case large
    case normalized
}

public enum VectorConstraint {
    case normal(ClosedRange<Float>)
    case unit
    case nonZero(ClosedRange<Float>)
    case orthogonalBasis(index: Int, total: Int)
}

// MARK: - Approximate Equality

public func approximately<T: BinaryFloatingPoint>(_ a: T, _ b: T, tolerance: T = 1e-6) -> Bool {
    abs(a - b) <= tolerance
}

public func vectorsApproximatelyEqual<D: Dimension>(
    _ a: Vector<D>, 
    _ b: Vector<D>, 
    tolerance: Float = 1e-6
) -> Bool {
    guard a.scalarCount == b.scalarCount else { return false }
    for i in 0..<a.scalarCount {
        if !approximately(a[i], b[i], tolerance: tolerance) {
            return false
        }
    }
    return true
}

public func dynamicVectorsApproximatelyEqual(
    _ a: DynamicVector,
    _ b: DynamicVector,
    tolerance: Float = 1e-6
) -> Bool {
    guard a.dimension == b.dimension else { return false }
    for i in 0..<a.dimension {
        if !approximately(a[i], b[i], tolerance: tolerance) {
            return false
        }
    }
    return true
}