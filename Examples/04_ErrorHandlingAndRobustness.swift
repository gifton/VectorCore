// VectorCore: Error Handling and Robustness
//
// This example demonstrates proper error handling and defensive programming

import VectorCore
import Foundation

// MARK: - 1. Handling Non-Finite Values

func nonFiniteHandling() {
    print("=== Handling Non-Finite Values ===\n")
    
    // Example: Processing data that might contain NaN or Infinity
    let problematicData = [1.0, 2.0, Float.nan, 4.0, Float.infinity, -3.0, -Float.infinity]
    
    // Method 1: Check before creating vector
    do {
        print("Checking data before vector creation:")
        try validateFinite(problematicData)
        let vector = Vector<Dim7>(problematicData)
        print("Vector created successfully: \(vector)")
    } catch let error as NonFiniteError {
        print("Error detected: \(error.localizedDescription)")
        
        // Handle the error
        switch error {
        case .nanDetected(let indices):
            print("NaN values at indices: \(indices)")
        case .infinityDetected(let indices):
            print("Infinity values at indices: \(indices)")
        case .negativeInfinityDetected(let indices):
            print("Negative infinity values at indices: \(indices)")
        case .multipleNonFinite(let nanIndices, let infIndices, let negInfIndices):
            print("Multiple issues found:")
            if !nanIndices.isEmpty { print("  NaN at: \(nanIndices)") }
            if !infIndices.isEmpty { print("  Infinity at: \(infIndices)") }
            if !negInfIndices.isEmpty { print("  -Infinity at: \(negInfIndices)") }
        }
    } catch {
        print("Unexpected error: \(error)")
    }
    
    // Method 2: Clean the data
    print("\nCleaning problematic data:")
    let handler = NonFiniteHandler(handling: [.replaceWithZero, .skipOperation])
    let cleaned = handler.handle(problematicData)
    
    switch cleaned {
    case .success(let cleanData):
        print("Cleaned data: \(cleanData)")
        let vector = Vector<Dim7>(cleanData)
        print("Vector created from cleaned data: \(vector)")
    case .failure(let error):
        print("Failed to clean data: \(error)")
    }
}

// MARK: - 2. Safe Vector Operations

func safeVectorOperations() {
    print("\n\n=== Safe Vector Operations ===\n")
    
    // Division by zero protection
    let numerator = Vector32.random(in: -10...10)
    var denominator = Vector32.random(in: -0.1...0.1)
    
    print("Unsafe division might produce infinities:")
    
    // Make some elements exactly zero
    for i in stride(from: 0, to: 32, by: 8) {
        denominator[i] = 0.0
    }
    
    // Safe division with check
    let safeDivision = numerator.toArray().enumerated().map { index, num in
        let denom = denominator[index]
        return denom != 0 ? num / denom : 0.0  // Replace division by zero with 0
    }
    
    let safeResult = Vector32(safeDivision)
    print("Safe division result has finite values: \(safeResult.toArray().allSatisfy { $0.isFinite })")
    
    // Normalization safety
    print("\nSafe normalization:")
    let zeroVector = Vector64.zeros()
    let normalizedZero = zeroVector.normalized()
    print("Zero vector normalized magnitude: \(normalizedZero.magnitude)")
    print("Is still zero: \(normalizedZero == zeroVector)")
}

// MARK: - 3. Dimension Mismatch Handling

func dimensionMismatchHandling() {
    print("\n\n=== Dimension Mismatch Handling ===\n")
    
    // Using VectorFactory for safe creation
    let data128 = Array(repeating: 1.0, count: 128)
    let data256 = Array(repeating: 2.0, count: 256)
    
    do {
        // This will succeed
        let vector128 = try VectorFactory.create(Dim128.self, from: data128)
        print("Created 128-dim vector successfully")
        
        // This will fail
        let vector128Wrong = try VectorFactory.create(Dim128.self, from: data256)
        print("This won't print")
    } catch let error as VectorError {
        print("Caught dimension mismatch: \(error.localizedDescription)")
        print("Recovery suggestion: \(error.recoverySuggestion ?? "None")")
    } catch {
        print("Unexpected error: \(error)")
    }
    
    // Dynamic dimension handling
    print("\nDynamic dimension handling:")
    let unknownData = loadDataFromSource() // Simulated data of unknown size
    
    let dynamicVector = DynamicVector(unknownData)
    print("Created dynamic vector with \(dynamicVector.dimension) dimensions")
    
    // Convert to strongly-typed if dimension matches
    if unknownData.count == 512 {
        let stronglyTyped = Vector512(unknownData)
        print("Converted to strongly-typed Vector512")
    } else {
        print("Dimension \(unknownData.count) doesn't match any strongly-typed vector")
    }
}

// MARK: - 4. Serialization Error Handling

func serializationErrorHandling() {
    print("\n\n=== Serialization Error Handling ===\n")
    
    let vector = Vector256.random(in: -100...100)
    
    // Binary serialization with error handling
    do {
        let data = try vector.encodeBinary()
        print("Encoded to \(data.count) bytes")
        
        // Simulate data corruption
        var corruptedData = data
        corruptedData[10] = 255  // Corrupt a byte
        
        // Try to decode corrupted data
        do {
            let decoded = try Vector256.decodeBinary(from: corruptedData)
            print("Decoded successfully (shouldn't happen with corruption)")
        } catch {
            print("Failed to decode corrupted data: \(error)")
            
            // Use VectorError for rich error handling
            let richError = VectorError(
                .dataCorruption,
                message: "Binary data failed CRC32 validation",
                underlying: error
            )
            print("Rich error: \(richError)")
        }
    } catch {
        print("Encoding error: \(error)")
    }
    
    // Base64 handling
    print("\nBase64 error handling:")
    let invalidBase64 = "This is not valid base64!"
    
    do {
        let decoded = try Vector256.base64Decoded(from: invalidBase64)
        print("This shouldn't print")
    } catch let error as VectorError {
        print("Vector error: \(error.localizedDescription)")
    } catch {
        print("Decoding error: \(error)")
    }
}

// MARK: - 5. Performance Regression Detection

func performanceRegressionDetection() {
    print("\n\n=== Performance Regression Detection ===\n")
    
    // Setup regression test suite
    let suite = PerformanceRegressionSuite(
        name: "Core Operations",
        configuration: RegressionTestConfig(
            iterations: 100,
            warmupIterations: 10,
            acceptableVariancePercent: 10.0,
            failOnRegression: true
        )
    )
    
    // Add tests
    suite.addTest("Vector Creation") { timer in
        timer.start()
        let _ = Vector512.random(in: -1...1)
        timer.stop()
    }
    
    suite.addTest("Dot Product") { timer in
        let v1 = Vector512.random(in: -1...1)
        let v2 = Vector512.random(in: -1...1)
        timer.start()
        let _ = v1.dotProduct(v2)
        timer.stop()
    }
    
    // Run tests
    do {
        let results = try suite.run()
        
        print("Performance test results:")
        for result in results.results {
            print("\(result.testName):")
            print("  Mean time: \(String(format: "%.6f", result.meanTime))s")
            print("  Std dev: \(String(format: "%.6f", result.stdDeviation))s")
            print("  Throughput: \(String(format: "%.0f", result.throughput)) ops/sec")
        }
        
        // Check against baseline (if exists)
        if let baselineData = loadBaseline() {
            let baseline = try PerformanceBaseline.decode(from: baselineData)
            let comparison = try suite.compareAgainst(baseline: baseline)
            
            if comparison.hasRegressions {
                print("\nPerformance regressions detected!")
                for regression in comparison.regressions {
                    print("  \(regression.test): \(String(format: "%.1f%%", regression.percentageChange)) slower")
                }
            } else {
                print("\nNo performance regressions detected ✓")
            }
        }
    } catch {
        print("Performance test error: \(error)")
    }
}

// MARK: - 6. Defensive Programming Patterns

func defensiveProgrammingPatterns() {
    print("\n\n=== Defensive Programming Patterns ===\n")
    
    // Pattern 1: Validate inputs
    func processVectors(_ vectors: [any VectorType]) throws -> Float {
        guard !vectors.isEmpty else {
            throw VectorError.invalidOperation(
                name: "processVectors",
                reason: "Empty vector array"
            )
        }
        
        // Check all vectors have same dimension
        let firstDim = vectors[0].scalarCount
        guard vectors.allSatisfy({ $0.scalarCount == firstDim }) else {
            throw VectorError.dimensionMismatch(
                expected: firstDim,
                actual: vectors.first(where: { $0.scalarCount != firstDim })?.scalarCount ?? 0
            )
        }
        
        // Process safely
        return vectors.map { $0.magnitude }.reduce(0, +) / Float(vectors.count)
    }
    
    // Pattern 2: Use Result type for fallible operations
    func safeDivide(_ a: Vector128, by b: Vector128) -> Result<Vector128, VectorError> {
        // Check for zeros in denominator
        if b.toArray().contains(where: { $0 == 0 }) {
            return .failure(.invalidOperation(
                name: "division",
                reason: "Division by zero detected"
            ))
        }
        
        return .success(a ./ b)
    }
    
    // Pattern 3: Graceful degradation
    func findSimilarVectors(to query: Vector256, in database: [Vector256], k: Int) -> [Int] {
        let safeK = min(max(1, k), database.count)  // Clamp k to valid range
        
        if database.isEmpty {
            return []
        }
        
        let distances = database.enumerated().map { index, vector in
            (index: index, distance: query.distance(to: vector))
        }
        
        return distances
            .sorted { $0.distance < $1.distance }
            .prefix(safeK)
            .map { $0.index }
    }
    
    print("Defensive patterns implemented successfully")
}

// MARK: - Helper Functions

func loadDataFromSource() -> [Float] {
    // Simulate loading data of unknown size
    let randomSize = Int.random(in: 100...1000)
    return (0..<randomSize).map { _ in Float.random(in: -1...1) }
}

func loadBaseline() -> Data? {
    // Simulate loading baseline data
    return nil  // Return nil to simulate no baseline
}

// MARK: - Main

func main() {
    nonFiniteHandling()
    safeVectorOperations()
    dimensionMismatchHandling()
    serializationErrorHandling()
    performanceRegressionDetection()
    defensiveProgrammingPatterns()
}

// Run the example
main()