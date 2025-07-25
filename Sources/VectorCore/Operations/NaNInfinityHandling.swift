// VectorCore: NaN and Infinity Handling
//
// Comprehensive utilities for handling non-finite values in vectors
//

import Foundation
#if canImport(Accelerate)
import Accelerate
#endif

/// Options for handling NaN and Infinity values
public struct NonFiniteHandling: OptionSet, Sendable {
    public let rawValue: Int
    
    public init(rawValue: Int) {
        self.rawValue = rawValue
    }
    
    /// Replace NaN values with zero
    public static let replaceNaNWithZero = NonFiniteHandling(rawValue: 1 << 0)
    
    /// Replace Infinity values with maximum finite value
    public static let replaceInfinityWithMax = NonFiniteHandling(rawValue: 1 << 1)
    
    /// Replace -Infinity values with minimum finite value
    public static let replaceNegInfinityWithMin = NonFiniteHandling(rawValue: 1 << 2)
    
    /// Propagate NaN (any NaN in input produces NaN in output)
    public static let propagateNaN = NonFiniteHandling(rawValue: 1 << 3)
    
    /// Throw error on non-finite values
    public static let throwOnNonFinite = NonFiniteHandling(rawValue: 1 << 4)
    
    /// Common presets
    public static let replaceAll: NonFiniteHandling = [.replaceNaNWithZero, .replaceInfinityWithMax, .replaceNegInfinityWithMin]
    public static let strict: NonFiniteHandling = [.throwOnNonFinite]
    public static let propagate: NonFiniteHandling = [.propagateNaN]
}

/// Errors related to non-finite values
public enum NonFiniteError: Error, LocalizedError {
    case nanDetected(indices: [Int])
    case infinityDetected(indices: [Int])
    case negativeInfinityDetected(indices: [Int])
    case mixedNonFiniteValues(nanIndices: [Int], infIndices: [Int], negInfIndices: [Int])
    
    public var errorDescription: String? {
        switch self {
        case .nanDetected(let indices):
            return "NaN values detected at indices: \(indices)"
        case .infinityDetected(let indices):
            return "Infinity values detected at indices: \(indices)"
        case .negativeInfinityDetected(let indices):
            return "Negative infinity values detected at indices: \(indices)"
        case .mixedNonFiniteValues(let nan, let inf, let negInf):
            return "Non-finite values detected - NaN: \(nan), Inf: \(inf), -Inf: \(negInf)"
        }
    }
}

/// Result of non-finite value check
public struct NonFiniteCheckResult {
    public let hasNaN: Bool
    public let hasInfinity: Bool
    public let hasNegativeInfinity: Bool
    public let nanIndices: [Int]
    public let infinityIndices: [Int]
    public let negativeInfinityIndices: [Int]
    
    public var hasNonFinite: Bool {
        hasNaN || hasInfinity || hasNegativeInfinity
    }
    
    public var totalNonFiniteCount: Int {
        nanIndices.count + infinityIndices.count + negativeInfinityIndices.count
    }
}

// MARK: - Vector Extensions for NaN/Infinity Handling

public extension Vector where D.Storage: VectorStorageOperations {
    
    /// Check for non-finite values in the vector
    ///
    /// - Returns: Detailed information about non-finite values
    /// - Complexity: O(n) where n is the dimension
    func checkNonFinite() -> NonFiniteCheckResult {
        var nanIndices: [Int] = []
        var infinityIndices: [Int] = []
        var negativeInfinityIndices: [Int] = []
        
        for i in 0..<D.value {
            let value = self[i]
            if value.isNaN {
                nanIndices.append(i)
            } else if value == .infinity {
                infinityIndices.append(i)
            } else if value == -.infinity {
                negativeInfinityIndices.append(i)
            }
        }
        
        return NonFiniteCheckResult(
            hasNaN: !nanIndices.isEmpty,
            hasInfinity: !infinityIndices.isEmpty,
            hasNegativeInfinity: !negativeInfinityIndices.isEmpty,
            nanIndices: nanIndices,
            infinityIndices: infinityIndices,
            negativeInfinityIndices: negativeInfinityIndices
        )
    }
    
    /// Check if all values are finite
    ///
    /// - Returns: true if all values are finite
    /// - Complexity: O(n) where n is the dimension
    @inlinable
    var isFinite: Bool {
        for i in 0..<D.value {
            if !self[i].isFinite {
                return false
            }
        }
        return true
    }
    
    /// Check if vector contains any NaN values
    ///
    /// - Returns: true if any element is NaN
    /// - Complexity: O(n) where n is the dimension
    @inlinable
    var hasNaN: Bool {
        for i in 0..<D.value {
            if self[i].isNaN {
                return true
            }
        }
        return false
    }
    
    /// Check if vector contains any infinity values
    ///
    /// - Returns: true if any element is ±infinity
    /// - Complexity: O(n) where n is the dimension
    @inlinable
    var hasInfinity: Bool {
        for i in 0..<D.value {
            let value = self[i]
            if value == .infinity || value == -.infinity {
                return true
            }
        }
        return false
    }
    
    /// Handle non-finite values according to specified options
    ///
    /// - Parameter options: How to handle non-finite values
    /// - Returns: New vector with non-finite values handled
    /// - Throws: NonFiniteError if throwOnNonFinite option is set
    func handleNonFinite(options: NonFiniteHandling) throws -> Vector<D> {
        // Quick check if all values are finite
        if isFinite {
            return self
        }
        
        let checkResult = checkNonFinite()
        
        // Throw if requested
        if options.contains(.throwOnNonFinite) && checkResult.hasNonFinite {
            if checkResult.hasNaN && !checkResult.hasInfinity && !checkResult.hasNegativeInfinity {
                throw NonFiniteError.nanDetected(indices: checkResult.nanIndices)
            } else if checkResult.hasInfinity && !checkResult.hasNaN && !checkResult.hasNegativeInfinity {
                throw NonFiniteError.infinityDetected(indices: checkResult.infinityIndices)
            } else if checkResult.hasNegativeInfinity && !checkResult.hasNaN && !checkResult.hasInfinity {
                throw NonFiniteError.negativeInfinityDetected(indices: checkResult.negativeInfinityIndices)
            } else {
                throw NonFiniteError.mixedNonFiniteValues(
                    nanIndices: checkResult.nanIndices,
                    infIndices: checkResult.infinityIndices,
                    negInfIndices: checkResult.negativeInfinityIndices
                )
            }
        }
        
        // Propagate NaN if requested
        if options.contains(.propagateNaN) && checkResult.hasNaN {
            return Vector<D>(repeating: .nan)
        }
        
        // Replace values as needed
        var result = self
        
        for i in 0..<D.value {
            let value = result[i]
            
            if value.isNaN && options.contains(.replaceNaNWithZero) {
                result[i] = 0
            } else if value == .infinity && options.contains(.replaceInfinityWithMax) {
                result[i] = Float.greatestFiniteMagnitude
            } else if value == -.infinity && options.contains(.replaceNegInfinityWithMin) {
                result[i] = -Float.greatestFiniteMagnitude
            }
        }
        
        return result
    }
    
    /// Replace non-finite values with specified value
    ///
    /// - Parameter replacement: Value to use for non-finite elements
    /// - Returns: New vector with non-finite values replaced
    func replacingNonFinite(with replacement: Float) -> Vector<D> {
        var result = self
        
        for i in 0..<D.value {
            if !result[i].isFinite {
                result[i] = replacement
            }
        }
        
        return result
    }
    
    /// Filter out non-finite values and return indices of finite values
    ///
    /// - Returns: Tuple of (finite values array, original indices)
    func finiteValues() -> (values: [Float], indices: [Int]) {
        var values: [Float] = []
        var indices: [Int] = []
        
        for i in 0..<D.value {
            let value = self[i]
            if value.isFinite {
                values.append(value)
                indices.append(i)
            }
        }
        
        return (values, indices)
    }
}

// MARK: - DynamicVector Extensions

public extension DynamicVector {
    
    /// Check for non-finite values in the vector
    func checkNonFinite() -> NonFiniteCheckResult {
        var nanIndices: [Int] = []
        var infinityIndices: [Int] = []
        var negativeInfinityIndices: [Int] = []
        
        for i in 0..<dimension {
            let value = self[i]
            if value.isNaN {
                nanIndices.append(i)
            } else if value == .infinity {
                infinityIndices.append(i)
            } else if value == -.infinity {
                negativeInfinityIndices.append(i)
            }
        }
        
        return NonFiniteCheckResult(
            hasNaN: !nanIndices.isEmpty,
            hasInfinity: !infinityIndices.isEmpty,
            hasNegativeInfinity: !negativeInfinityIndices.isEmpty,
            nanIndices: nanIndices,
            infinityIndices: infinityIndices,
            negativeInfinityIndices: negativeInfinityIndices
        )
    }
    
    /// Check if all values are finite
    @inlinable
    var isFinite: Bool {
        for i in 0..<dimension {
            if !self[i].isFinite {
                return false
            }
        }
        return true
    }
    
    /// Handle non-finite values
    func handleNonFinite(options: NonFiniteHandling) throws -> DynamicVector {
        if isFinite {
            return self
        }
        
        let checkResult = checkNonFinite()
        
        // Throw if requested
        if options.contains(.throwOnNonFinite) && checkResult.hasNonFinite {
            throw NonFiniteError.mixedNonFiniteValues(
                nanIndices: checkResult.nanIndices,
                infIndices: checkResult.infinityIndices,
                negInfIndices: checkResult.negativeInfinityIndices
            )
        }
        
        // Propagate NaN if requested
        if options.contains(.propagateNaN) && checkResult.hasNaN {
            return DynamicVector(dimension: dimension, repeating: .nan)
        }
        
        // Replace values
        var result = self
        for i in 0..<dimension {
            let value = result[i]
            
            if value.isNaN && options.contains(.replaceNaNWithZero) {
                result[i] = 0
            } else if value == .infinity && options.contains(.replaceInfinityWithMax) {
                result[i] = Float.greatestFiniteMagnitude
            } else if value == -.infinity && options.contains(.replaceNegInfinityWithMin) {
                result[i] = -Float.greatestFiniteMagnitude
            }
        }
        
        return result
    }
}

// MARK: - Safe Mathematical Operations

public extension Vector where D.Storage: VectorStorageOperations {
    
    /// Safe division that handles division by zero
    ///
    /// - Parameters:
    ///   - divisor: The divisor
    ///   - options: How to handle division by zero
    /// - Returns: Result of division with non-finite values handled
    func safeDivide(by divisor: Float, options: NonFiniteHandling = .replaceNaNWithZero) throws -> Vector<D> {
        if divisor == 0 {
            if options.contains(.throwOnNonFinite) {
                throw VectorError.divisionByZero(operation: "safeDivide")
            }
            
            if options.contains(.propagateNaN) {
                return Vector<D>(repeating: .nan)
            }
            
            // Replace with zero by default
            return Vector<D>.zeros()
        }
        
        let result = self / divisor
        return try result.handleNonFinite(options: options)
    }
    
    /// Safe normalization that handles zero vectors
    ///
    /// - Parameter options: How to handle zero magnitude
    /// - Returns: Normalized vector or handled according to options
    func safeNormalized(options: NonFiniteHandling = .replaceAll) throws -> Vector<D> {
        let mag = magnitude
        
        if mag == 0 {
            if options.contains(.throwOnNonFinite) {
                throw VectorError.zeroVectorError(operation: "normalization")
            }
            
            if options.contains(.propagateNaN) {
                return Vector<D>(repeating: .nan)
            }
            
            // Return zero vector by default
            return self
        }
        
        return try safeDivide(by: mag, options: options)
    }
    
    /// Safe logarithm that handles non-positive values
    ///
    /// - Parameter options: How to handle non-positive values
    /// - Returns: Vector with log of each element
    func safeLog(options: NonFiniteHandling = .replaceNaNWithZero) throws -> Vector<D> {
        var result = Vector<D>()
        
        for i in 0..<D.value {
            let value = self[i]
            if value <= 0 {
                if options.contains(.throwOnNonFinite) {
                    throw VectorError.invalidValues(indices: [i], reason: "Logarithm of non-positive value")
                }
                
                if options.contains(.propagateNaN) {
                    return Vector<D>(repeating: .nan)
                }
                
                if options.contains(.replaceNaNWithZero) {
                    result[i] = 0
                } else {
                    result[i] = -Float.greatestFiniteMagnitude
                }
            } else {
                result[i] = log(value)
            }
        }
        
        return result
    }
}

// MARK: - Batch Operations with NaN/Infinity Handling

public extension SyncBatchOperations {
    
    /// Filter out vectors containing non-finite values
    ///
    /// - Parameter vectors: Input vectors
    /// - Returns: Only vectors with all finite values
    static func filterFinite<V: ExtendedVectorProtocol>(_ vectors: [V]) -> [V] {
        vectors.filter { vector in
            for i in 0..<vector.scalarCount {
                if !vector[i].isFinite {
                    return false
                }
            }
            return true
        }
    }
    
    /// Find vectors containing non-finite values
    ///
    /// - Parameter vectors: Input vectors
    /// - Returns: Indices of vectors containing non-finite values
    static func findNonFinite<V: ExtendedVectorProtocol>(_ vectors: [V]) -> [Int] {
        vectors.enumerated().compactMap { index, vector in
            for i in 0..<vector.scalarCount {
                if !vector[i].isFinite {
                    return index
                }
            }
            return nil
        }
    }
    
    /// Compute statistics excluding non-finite values
    ///
    /// - Parameter vectors: Input vectors
    /// - Returns: Statistics computed only from finite values
    static func finiteStatistics<V: ExtendedVectorProtocol>(for vectors: [V]) -> BatchStatistics {
        let finiteVectors = filterFinite(vectors)
        
        guard !finiteVectors.isEmpty else {
            return BatchStatistics(count: 0, meanMagnitude: 0, stdMagnitude: 0)
        }
        
        return statistics(for: finiteVectors)
    }
}

// MARK: - Global Validation Functions

/// Validate that a value is finite
///
/// - Parameter value: Value to check
/// - Throws: NonFiniteError if value is not finite
public func validateFinite(_ value: Float) throws {
    if value.isNaN {
        throw NonFiniteError.nanDetected(indices: [0])
    } else if value == .infinity {
        throw NonFiniteError.infinityDetected(indices: [0])
    } else if value == -.infinity {
        throw NonFiniteError.negativeInfinityDetected(indices: [0])
    }
}

/// Validate that all values in an array are finite
///
/// - Parameter values: Array to check
/// - Throws: NonFiniteError if any value is not finite
public func validateFinite(_ values: [Float]) throws {
    var nanIndices: [Int] = []
    var infIndices: [Int] = []
    var negInfIndices: [Int] = []
    
    for (i, value) in values.enumerated() {
        if value.isNaN {
            nanIndices.append(i)
        } else if value == .infinity {
            infIndices.append(i)
        } else if value == -.infinity {
            negInfIndices.append(i)
        }
    }
    
    if !nanIndices.isEmpty || !infIndices.isEmpty || !negInfIndices.isEmpty {
        throw NonFiniteError.mixedNonFiniteValues(
            nanIndices: nanIndices,
            infIndices: infIndices,
            negInfIndices: negInfIndices
        )
    }
}