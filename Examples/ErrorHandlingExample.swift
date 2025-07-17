// VectorCore: Error Handling Example
//
// Demonstrates error handling patterns in VectorCore
//

import Foundation
import VectorCore

/// Example showing various error handling scenarios
struct ErrorHandlingExample {
    
    static func run() {
        print("VectorCore Error Handling Examples")
        print("==================================\n")
        
        // Example 1: Dimension mismatch
        do {
            let v1 = Vector512.random(in: -1...1)
            let v2data = Array(repeating: Float(1.0), count: 256)
            _ = try VectorFactory.vector(of: 512, from: v2data)
        } catch let error as VectorError {
            print("1. Dimension Mismatch Error:")
            print("   Error Type: \(error.kind.rawValue)")
            print("   Description: \(error.errorDescription ?? "")")
            print("   Severity: \(error.kind.severity.rawValue)\n")
        } catch {
            print("Unexpected error: \(error)")
        }
        
        // Example 2: Index out of bounds
        do {
            let vector = Vector256.zeros()
            // This would be caught at compile time in real usage
            if let dynamicVector = vector as? DynamicVector {
                _ = dynamicVector[300] // Out of bounds
            }
        } catch {
            print("2. Index Out of Bounds: This is prevented by type safety\n")
        }
        
        // Example 3: Numerical validation
        demonstrateNumericalValidation()
        
        // Example 4: Batch operation errors
        demonstrateBatchErrorHandling()
        
        // Example 5: Custom error handling
        demonstrateCustomErrorHandling()
    }
    
    /// Demonstrates numerical validation errors
    static func demonstrateNumericalValidation() {
        print("3. Numerical Validation:")
        
        // Create vectors with invalid values
        var values = Array(repeating: Float(1.0), count: 512)
        values[10] = .nan
        values[20] = .infinity
        values[30] = -.infinity
        
        do {
            let vector = Vector512(values)
            
            // Check for invalid values
            let invalidIndices = values.enumerated().compactMap { index, value in
                (!value.isFinite) ? index : nil
            }
            
            if !invalidIndices.isEmpty {
                throw VectorError.invalidValues(
                    indices: invalidIndices,
                    reason: "Vector contains non-finite values"
                )
            }
            
            // Attempt normalization
            let normalized = vector.normalized()
            print("   Vector normalized successfully")
        } catch let error as VectorError {
            print("   Error: \(error.errorDescription ?? "")")
            print("   Error Type: \(error.kind.rawValue)\n")
        } catch {
            print("   Unexpected error: \(error)\n")
        }
    }
    
    /// Demonstrates batch operation error handling
    static func demonstrateBatchErrorHandling() {
        print("4. Batch Operation Error Handling:")
        
        // Create a batch with mixed valid/invalid vectors
        var vectors: [Vector256] = []
        for i in 0..<10 {
            if i == 5 {
                // Create a zero vector that will fail normalization
                vectors.append(Vector256.zeros())
            } else {
                vectors.append(Vector256.random(in: -1...1))
            }
        }
        
        // Process batch with error collection
        var results: [Vector256] = []
        var errors: [(index: Int, error: VectorError)] = []
        
        for (index, vector) in vectors.enumerated() {
            do {
                // Check for zero vector
                if vector.magnitude == 0 {
                    throw VectorError.zeroVectorError(operation: "normalization")
                }
                
                let normalized = vector.normalized()
                results.append(normalized)
            } catch let error as VectorError {
                errors.append((index: index, error: error))
            } catch {
                let genericError = VectorError.invalidData(
                    "Unknown error at index \(index)"
                )
                errors.append((index: index, error: genericError))
            }
        }
        
        print("   Processed \(vectors.count) vectors:")
        print("   - Successful: \(results.count)")
        print("   - Failed: \(errors.count)")
        
        for (index, error) in errors {
            print("   - Error at index \(index): \(error.errorDescription ?? "")")
        }
        print()
    }
    
    /// Demonstrates custom error handling patterns
    static func demonstrateCustomErrorHandling() {
        print("5. Custom Error Handling Patterns:")
        
        // Pattern 1: Result type for cleaner error handling
        func safeDotProduct(_ v1: Vector512, _ v2: Vector512) -> Result<Float, VectorError> {
            // Validate inputs
            if v1.magnitude == 0 || v2.magnitude == 0 {
                return .failure(VectorError.zeroVectorError(operation: "dot product"))
            }
            
            let result = v1.dotProduct(v2)
            
            // Check for numerical issues
            if !result.isFinite {
                return .failure(VectorError.invalidValues(indices: [], reason: "Numerical instability in dot product"))
            }
            
            return .success(result)
        }
        
        // Use the safe function
        let v1 = Vector512.random(in: -1...1)
        let v2 = Vector512.zeros()
        
        switch safeDotProduct(v1, v2) {
        case .success(let result):
            print("   Dot product: \(result)")
        case .failure(let error):
            print("   Safe operation prevented error: \(error.errorDescription ?? "")")
        }
        
        // Pattern 2: Error aggregation
        func processVectorBatch(_ vectors: [Vector512]) -> (results: [Float], errors: [VectorError]) {
            var results: [Float] = []
            var errors: [VectorError] = []
            
            for vector in vectors {
                if vector.magnitude == 0 {
                    errors.append(VectorError.zeroVectorError(operation: "magnitude calculation"))
                } else if vector.toArray().contains(where: { !$0.isFinite }) {
                    let indices = vector.toArray().enumerated().compactMap { 
                        !$1.isFinite ? $0 : nil 
                    }
                    errors.append(.invalidValues(indices: indices, reason: "Non-finite values"))
                } else {
                    results.append(vector.magnitude)
                }
            }
            
            return (results, errors)
        }
        
        // Test batch processing
        let testBatch = [
            Vector512.random(in: -1...1),
            Vector512.zeros(),
            Vector512.random(in: -1...1)
        ]
        
        let (magnitudes, batchErrors) = processVectorBatch(testBatch)
        print("   Batch processing results:")
        print("   - Computed \(magnitudes.count) magnitudes")
        print("   - Encountered \(batchErrors.count) errors")
    }
}

// Main entry point
@main
struct ErrorHandlingExampleMain {
    static func main() async throws {
        ErrorHandlingExample.run()
    }
}