#!/usr/bin/swift

// VectorCore: NaN and Infinity Handling Example
//
// Demonstrates comprehensive handling of non-finite values
//

import VectorCore
import Foundation

// MARK: - Example 1: Basic Detection

print("=== NaN and Infinity Detection ===\n")

// Create vectors with various non-finite values
var values = Array(repeating: Float(1.0), count: 32)
values[5] = .nan
values[10] = .infinity
values[15] = -.infinity
values[20] = 0.0 / 0.0  // Another way to get NaN
values[25] = 1.0 / 0.0  // Another way to get infinity

let problematicVector = Vector<Dim32>(values)

print("Vector contains:")
print("  - Is finite: \(problematicVector.isFinite)")
print("  - Has NaN: \(problematicVector.hasNaN)")
print("  - Has Infinity: \(problematicVector.hasInfinity)")

// Detailed check
let check = problematicVector.checkNonFinite()
print("\nDetailed check:")
print("  - NaN at indices: \(check.nanIndices)")
print("  - Infinity at indices: \(check.infinityIndices)")
print("  - -Infinity at indices: \(check.negativeInfinityIndices)")
print("  - Total non-finite count: \(check.totalNonFiniteCount)")

// MARK: - Example 2: Handling Strategies

print("\n\n=== Handling Strategies ===\n")

// Strategy 1: Replace all non-finite values
if let handled1 = try? problematicVector.handleNonFinite(options: .replaceAll) {
    print("Replace all strategy:")
    print("  - NaN → 0: value at index 5 = \(handled1[5])")
    print("  - Inf → max: value at index 10 = \(handled1[10])")
    print("  - -Inf → min: value at index 15 = \(handled1[15])")
    print("  - Result is finite: \(handled1.isFinite)")
}

// Strategy 2: Propagate NaN (useful for detecting issues)
if let handled2 = try? problematicVector.handleNonFinite(options: .propagateNaN) {
    print("\nPropagate NaN strategy:")
    print("  - All values become NaN: \(handled2[0].isNaN)")
}

// Strategy 3: Throw on non-finite (strict validation)
do {
    _ = try problematicVector.handleNonFinite(options: .strict)
} catch let error as NonFiniteError {
    print("\nStrict validation error: \(error.localizedDescription)")
}

// Strategy 4: Custom replacement
let handled3 = problematicVector.replacingNonFinite(with: -999.0)
print("\nCustom replacement (with -999):")
print("  - Value at index 5 (was NaN): \(handled3[5])")
print("  - Value at index 10 (was Inf): \(handled3[10])")

// MARK: - Example 3: Safe Mathematical Operations

print("\n\n=== Safe Mathematical Operations ===\n")

// Division by zero handling
let numerator = Vector<Dim32>.ones()

print("Safe division by zero:")
if let result1 = try? numerator.safeDivide(by: 0, options: .replaceNaNWithZero) {
    print("  - With replacement: all values = \(result1[0])")
}

if let result2 = try? numerator.safeDivide(by: 0, options: .propagateNaN) {
    print("  - With propagation: all values are NaN = \(result2[0].isNaN)")
}

// Safe normalization of zero vector
let zeroVector = Vector<Dim32>.zeros()
print("\nSafe normalization of zero vector:")
if let normalized = try? zeroVector.safeNormalized(options: .replaceAll) {
    print("  - Result magnitude: \(normalized.magnitude)")
    print("  - Result is zero vector: \(normalized == zeroVector)")
}

// Safe logarithm with non-positive values
let mixedValues = Vector<Dim32>([-2, -1, 0, 1, 2, 3, 4, 5] + Array(repeating: 1.0, count: 24))
print("\nSafe logarithm of mixed values:")
if let logResult = try? mixedValues.safeLog(options: .replaceNaNWithZero) {
    print("  - log(-2) → \(logResult[0])")
    print("  - log(-1) → \(logResult[1])")
    print("  - log(0) → \(logResult[2])")
    print("  - log(1) → \(logResult[3])")
    print("  - log(2) → \(logResult[4])")
}

// MARK: - Example 4: Real-world Scenarios

print("\n\n=== Real-world Scenarios ===\n")

// Scenario 1: Computing cosine similarity with potential zero vectors
func safeCosineSimiliarity(_ v1: Vector<Dim128>, _ v2: Vector<Dim128>) -> Float {
    do {
        let norm1 = try v1.safeNormalized(options: .throwOnNonFinite)
        let norm2 = try v2.safeNormalized(options: .throwOnNonFinite)
        return norm1.dotProduct(norm2)
    } catch {
        // Handle zero vectors or non-finite values
        return 0.0  // Or return NaN to indicate undefined
    }
}

let v1 = Vector<Dim128>.random(in: -1...1)
let v2 = Vector<Dim128>.zeros()  // Problematic!
let similarity = safeCosineSimiliarity(v1, v2)
print("Cosine similarity with zero vector: \(similarity)")

// Scenario 2: Robust statistics computation
func computeRobustMean<V: ExtendedVectorProtocol>(_ vectors: [V]) -> Float? {
    let finiteVectors = SyncBatchOperations.filterFinite(vectors)
    
    guard !finiteVectors.isEmpty else {
        print("  Warning: No finite vectors found!")
        return nil
    }
    
    if finiteVectors.count < vectors.count {
        print("  Warning: Filtered out \(vectors.count - finiteVectors.count) non-finite vectors")
    }
    
    let stats = SyncBatchOperations.statistics(for: finiteVectors)
    return stats.meanMagnitude
}

let vectorSet = [
    Vector<Dim64>.ones(),
    Vector<Dim64>([.nan] + Array(repeating: 1.0, count: 63)),
    Vector<Dim64>.ones() * 2,
    Vector<Dim64>([.infinity] + Array(repeating: 1.0, count: 63)),
    Vector<Dim64>.ones() * 3
]

if let robustMean = computeRobustMean(vectorSet) {
    print("\nRobust mean magnitude: \(robustMean)")
}

// MARK: - Example 5: Batch Processing with Non-finite Values

print("\n\n=== Batch Processing ===\n")

// Generate dataset with some problematic vectors
let dataset = (0..<100).map { i -> Vector<Dim128> in
    if i % 20 == 0 {
        // Insert problematic vector
        var vals = Array(repeating: Float(i), count: 128)
        vals[0] = [Float.nan, .infinity, -.infinity].randomElement()!
        return Vector<Dim128>(vals)
    } else {
        return Vector<Dim128>.random(in: -10...10)
    }
}

// Find non-finite vectors
let nonFiniteIndices = SyncBatchOperations.findNonFinite(dataset)
print("Non-finite vectors at indices: \(nonFiniteIndices)")

// Filter and process only finite vectors
let finiteDataset = SyncBatchOperations.filterFinite(dataset)
print("Filtered dataset size: \(finiteDataset.count) (removed \(dataset.count - finiteDataset.count))")

// Compute statistics on finite vectors only
let finiteStats = SyncBatchOperations.finiteStatistics(for: dataset)
print("Statistics (finite values only):")
print("  - Count: \(finiteStats.count)")
print("  - Mean magnitude: \(finiteStats.meanMagnitude)")
print("  - Std deviation: \(finiteStats.stdMagnitude)")

// MARK: - Example 6: Validation Pipeline

print("\n\n=== Validation Pipeline ===\n")

// Create a validation pipeline for incoming data
func validateAndProcessVector(_ vector: Vector<Dim256>) -> Vector<Dim256>? {
    do {
        // Step 1: Check for non-finite values
        let check = vector.checkNonFinite()
        if check.hasNonFinite {
            print("  ⚠️  Non-finite values detected:")
            print("     - NaN count: \(check.nanIndices.count)")
            print("     - Inf count: \(check.infinityIndices.count + check.negativeInfinityIndices.count)")
        }
        
        // Step 2: Handle based on severity
        let handled: Vector<Dim256>
        if check.nanIndices.count > 10 {
            // Too many NaN values - reject
            print("  ❌ Rejected: Too many NaN values")
            return nil
        } else if check.hasNonFinite {
            // Some non-finite values - try to fix
            handled = try vector.handleNonFinite(options: .replaceAll)
            print("  ✅ Fixed: Replaced non-finite values")
        } else {
            // All good
            handled = vector
            print("  ✅ Passed: All values finite")
        }
        
        // Step 3: Additional validation
        let magnitude = handled.magnitude
        if magnitude == 0 {
            print("  ⚠️  Warning: Zero vector")
            return nil
        } else if magnitude > 1000 {
            print("  ⚠️  Warning: Very large magnitude (\(magnitude))")
            // Normalize to prevent numerical issues
            return handled.normalized()
        }
        
        return handled
        
    } catch {
        print("  ❌ Error: \(error)")
        return nil
    }
}

// Test the pipeline
print("Testing validation pipeline:")
let testVectors = [
    Vector<Dim256>.random(in: -1...1),  // Good vector
    Vector<Dim256>([.nan, .nan] + Array(repeating: 1.0, count: 254)),  // Few NaN
    Vector<Dim256>(Array(repeating: .nan, count: 256)),  // All NaN
    Vector<Dim256>.ones() * 2000,  // Very large
    Vector<Dim256>.zeros()  // Zero vector
]

for (i, vector) in testVectors.enumerated() {
    print("\nVector \(i):")
    _ = validateAndProcessVector(vector)
}

// MARK: - Example 7: Custom Handling Options

print("\n\n=== Custom Handling Options ===\n")

// Combine multiple handling strategies
let customOptions: NonFiniteHandling = [.replaceNaNWithZero, .throwOnNonFinite]

// This will replace NaN but throw on infinity
var customValues = Array(repeating: Float(1.0), count: 32)
customValues[5] = .nan
let vectorWithNaN = Vector<Dim32>(customValues)

do {
    let handled = try vectorWithNaN.handleNonFinite(options: customOptions)
    print("Successfully handled NaN: value at index 5 = \(handled[5])")
} catch {
    print("Error: \(error)")
}

// Now try with infinity
customValues[10] = .infinity
let vectorWithInf = Vector<Dim32>(customValues)

do {
    _ = try vectorWithInf.handleNonFinite(options: customOptions)
} catch {
    print("Infinity correctly triggered error: \(error)")
}

print("\n✅ NaN/Infinity handling examples completed!")