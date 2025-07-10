// VectorCore - Error Handling Example
//
// Demonstrates error handling and logging
//

import VectorCore
import Foundation

// MARK: - Basic Error Handling

func basicErrorHandling() {
    print("=== Basic Error Handling ===\n")
    
    // Example 1: Dimension mismatch
    do {
        let v1 = Vector512.random(in: -1...1)
        let v2data = Array(repeating: Float(1.0), count: 256)
        _ = try VectorFactory.vector(of: 512, from: v2data)
    } catch let error as VectorError {
        // Log the error
        error.log(message: "Failed to create vector")
        
        print("Caught error: \(error)")
        print("Error kind: \(error.kind)\n")
    } catch {
        coreLogger.error("Unexpected error: \(error)")
    }
    
    // Example 2: Index out of bounds
    do {
        _ = try VectorFactory.basis(dimension: 10, index: 15)
    } catch let error as VectorError {
        // Log with default logger
        error.log()
        print("Caught error: \(error)\n")
    } catch {
        coreLogger.error("Unexpected error: \(error)")
    }
}

// MARK: - Error Chaining

func errorChainingExample() throws {
    print("=== Error Chaining ===\n")
    
    func loadVectorData() throws -> [Float] {
        coreLogger.debug("Attempting to load vector data")
        throw VectorError.invalidData("File format incorrect")
    }
    
    func processVector() throws -> Vector512 {
        do {
            let data = try loadVectorData()
            return try Vector512(data)
        } catch let error as VectorError {
            coreLogger.warning("Failed to process vector: \(error)")
            throw error.chain(with: VectorError(
                .operationFailed,
                message: "Failed to process vector from file"
            ))
        }
    }
    
    func analyzeVectors() throws {
        do {
            _ = try processVector()
        } catch let error as VectorError {
            coreLogger.error("Analysis failed: \(error)")
            throw error.chain(with: VectorError(
                .operationFailed,
                message: "Analysis pipeline failed"
            ))
        }
    }
    
    // Try the chained operation
    do {
        try analyzeVectors()
    } catch let error as VectorError {
        error.log(message: "Complete pipeline failure")
        print("Error with chain:")
        print(error)
        print("\nChain length: \(error.errorChain.count)")
    }
}

// MARK: - Performance Logging

func performanceLoggingExample() {
    print("\n=== Performance Logging ===\n")
    
    // Time a vector operation
    let timer = PerformanceTimer(operation: "vector_normalization")
    let vector = Vector512.random(in: -1...1)
    let normalized = vector.normalized()
    timer.log() // Only logs if > 1ms by default
    
    print("Normalized vector magnitude: \(normalized.magnitude)")
    
    // Time a batch operation
    let batchTimer = PerformanceTimer(operation: "batch_creation")
    let vectors = (0..<1000).map { _ in Vector256.random(in: -1...1) }
    batchTimer.log(threshold: 0.0001) // Log if > 0.1ms
    
    batchLogger.info("Created \(vectors.count) vectors")
}

// MARK: - Custom Error Building

func customErrorBuildingExample() {
    print("\n=== Custom Error Building ===\n")
    
    // Build a detailed error
    let error = ErrorBuilder(.allocationFailed)
        .message("Failed to allocate memory for large vector batch")
        .parameter("requested_size", value: "1073741824")
        .parameter("available_memory", value: "536870912")
        .parameter("vector_count", value: "10000")
        .parameter("dimension", value: "3072")
        .build()
    
    // Log the error
    error.log(to: storageLogger, message: "Memory allocation failed")
    
    print("Custom error:")
    print(error.debugDescription)
}

// MARK: - Different Log Levels

func logLevelExample() {
    print("\n=== Log Levels Example ===\n")
    
    // Different loggers for different subsystems
    coreLogger.debug("This is a debug message")
    storageLogger.info("Storage initialized successfully")
    batchLogger.warning("Batch size exceeds recommended limit")
    metricsLogger.error("Failed to compute distance metric")
    performanceLogger.critical("Performance degradation detected")
    
    // Conditional logging
    let vectorCount = 10000
    if vectorCount > 5000 {
        batchLogger.warning("Processing large batch of \(vectorCount) vectors")
    }
}

// MARK: - Real-World Scenario

func realWorldScenario() throws {
    print("\n=== Real-World Scenario ===\n")
    
    // Configure minimum log level for production
    #if !DEBUG
    Logger.configuration.minimumLevel = .warning
    #endif
    
    // Batch processing with error handling and logging
    func processBatch(_ vectors: [Vector512]) throws -> [Float] {
        batchLogger.info("Starting batch processing of \(vectors.count) vectors")
        
        var results: [Float] = []
        var errors: [VectorError] = []
        
        for (index, vector) in vectors.enumerated() {
            do {
                // Time individual operations
                let timer = PerformanceTimer(operation: "process_vector_\(index)")
                
                // Simulate processing that might fail
                if vector.magnitude < 0.1 {
                    throw VectorError.invalidOperation(
                        "process",
                        reason: "Vector magnitude too small"
                    )
                }
                
                let result = vector.magnitude
                results.append(result)
                
                timer.log(threshold: 0.01) // Log if slower than 10ms
                
            } catch let error as VectorError {
                // Log individual errors
                error.log(to: batchLogger, message: "Failed at index \(index)")
                
                // Chain with batch context
                let batchError = error.chain(with: VectorError(
                    .operationFailed,
                    message: "Failed at batch index \(index)"
                ))
                errors.append(batchError)
            }
        }
        
        // Log batch summary
        if errors.isEmpty {
            batchLogger.info("Batch completed successfully: \(results.count) vectors processed")
        } else {
            batchLogger.warning("Batch completed with \(errors.count) errors out of \(vectors.count) vectors")
            
            // Log aggregated error
            let batchError = ErrorBuilder(.operationFailed)
                .message("Batch processing completed with errors")
                .parameter("total_vectors", value: "\(vectors.count)")
                .parameter("failed_vectors", value: "\(errors.count)")
                .parameter("success_rate", value: "\(Int(100 * results.count / vectors.count))%")
                .chain(errors[0])  // Include first error for context
                .build()
            
            batchError.log(to: batchLogger)
        }
        
        return results
    }
    
    // Create test batch
    let timer = PerformanceTimer(operation: "create_test_batch")
    let vectors = (0..<100).map { _ in Vector512.random(in: -1...1) }
    timer.log()
    
    // Process
    let results = try processBatch(vectors)
    print("\nProcessed \(results.count) vectors successfully")
}

// MARK: - Main

@main
struct ErrorHandlingExample {
    static func main() throws {
        print("VectorCore Error Handling and Logging Examples\n")
        
        basicErrorHandling()
        
        try errorChainingExample()
        
        performanceLoggingExample()
        
        customErrorBuildingExample()
        
        logLevelExample()
        
        try realWorldScenario()
        
        print("\n=== Summary ===")
        print("✓ Simple, lightweight logging system")
        print("✓ Integration with os.log for system console")
        print("✓ Performance timing with minimal overhead")
        print("✓ Rich error context with logging")
        print("✓ Different log levels for different scenarios")
        print("✓ Zero overhead with @autoclosure for disabled logs")
    }
}