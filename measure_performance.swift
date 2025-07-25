#!/usr/bin/env swift

import Foundation
import VectorCore

// Performance measurement script
print("\n=== VectorCore Performance Analysis ===")
print("Date: \(Date())")
print("Platform: \(ProcessInfo.processInfo.operatingSystemVersionString)")
print("Processors: \(ProcessInfo.processInfo.activeProcessorCount)")
print("")

// Helper to measure execution time
func measure(name: String, iterations: Int = 1, _ block: () throws -> Void) rethrows {
    let start = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        try block()
    }
    let end = CFAbsoluteTimeGetCurrent()
    let time = (end - start) / Double(iterations)
    print("\(name): \(String(format: "%.4f", time))s")
}

// Test data
let dimensions = [128, 256, 512, 1024]
let vectorCounts = [100, 500, 1000]

print("--- Vector Creation Performance ---")
for dim in dimensions {
    measure(name: "Create \(dim)D vector", iterations: 10000) {
        _ = DynamicVector(dimension: dim)
    }
}

print("\n--- Basic Operations Performance ---")
let v1 = Vector<D512>.random(in: -1...1)
let v2 = Vector<D512>.random(in: -1...1)

measure(name: "Vector addition (512D)", iterations: 10000) {
    _ = v1 + v2
}

measure(name: "Dot product (512D)", iterations: 10000) {
    _ = v1.dot(v2)
}

measure(name: "Normalization (512D)", iterations: 10000) {
    _ = v1.normalized()
}

print("\n--- Distance Computation Performance ---")
let vectors = (0..<1000).map { _ in Vector<D512>.random(in: -1...1) }
let query = Vector<D512>.random(in: -1...1)

measure(name: "Euclidean distance (1000 vectors)", iterations: 10) {
    for v in vectors {
        _ = query.distance(to: v, using: EuclideanDistance())
    }
}

measure(name: "Cosine distance (1000 vectors)", iterations: 10) {
    for v in vectors {
        _ = query.distance(to: v, using: CosineDistance())
    }
}

print("\n--- Nearest Neighbor Search Performance ---")
for count in [100, 500, 1000] {
    let testVectors = Array(vectors.prefix(count))
    
    measure(name: "Find 10 nearest in \(count) vectors", iterations: 10) {
        _ = query.findNearest(in: testVectors, k: 10)
    }
}

print("\n--- Batch Operations Performance ---")
let batchSize = 100
let batchVectors = Array(vectors.prefix(batchSize))

measure(name: "Batch centroid (\(batchSize) vectors)") {
    _ = BatchOperations.centroid(of: batchVectors)
}

measure(name: "Pairwise distances (\(batchSize) vectors)") {
    _ = BatchOperations.pairwiseDistances(of: batchVectors)
}

// Memory usage
var info = mach_task_basic_info()
var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

let result = withUnsafeMutablePointer(to: &info) {
    $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
        task_info(mach_task_self_,
                 task_flavor_t(MACH_TASK_BASIC_INFO),
                 $0,
                 &count)
    }
}

if result == KERN_SUCCESS {
    let memoryMB = Double(info.resident_size) / 1024 / 1024
    print("\n--- Memory Usage ---")
    print("Resident memory: \(String(format: "%.1f", memoryMB)) MB")
}

print("\n=== Analysis Complete ===\n")