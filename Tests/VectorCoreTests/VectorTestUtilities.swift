import XCTest
@testable import VectorCore

// MARK: - Vector Test Utilities

public extension XCTestCase {
    
    // MARK: - Vector Assertions
    
    func assertVectorsEqual<V: BaseVectorProtocol>(
        _ v1: V,
        _ v2: V,
        accuracy: Float = 1e-6,
        _ message: String = "",
        file: StaticString = #file,
        line: UInt = #line
    ) {
        XCTAssertEqual(v1.scalarCount, v2.scalarCount,
                      "Vector dimensions don't match",
                      file: file, line: line)
        
        for i in 0..<v1.scalarCount {
            XCTAssertEqual(v1[i], v2[i], accuracy: accuracy,
                          "\(message) - Vectors differ at index \(i)",
                          file: file, line: line)
        }
    }
    
    func assertVectorNear<V: ExtendedVectorProtocol>(
        _ vector: V,
        magnitude: Float,
        accuracy: Float = 1e-5,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        XCTAssertEqual(vector.magnitude, magnitude, accuracy: accuracy,
                      "Vector magnitude mismatch",
                      file: file, line: line)
    }
    
    // MARK: - Random Test Data
    
    func createRandomTestVectors<V: BaseVectorProtocol>(
        type: V.Type,
        count: Int,
        range: ClosedRange<Float> = -1...1,
        seed: UInt64? = nil
    ) -> [V] {
        if let seed = seed {
            // Use seeded random for reproducible tests
            var generator = SeededRandomGenerator(seed: seed)
            return (0..<count).map { _ in
                let values = (0..<V.dimensions).map { _ in
                    Float.random(in: range, using: &generator)
                }
                return V(from: values)
            }
        } else {
            return (0..<count).map { _ in
                let values = (0..<V.dimensions).map { _ in
                    Float.random(in: range)
                }
                return V(from: values)
            }
        }
    }
    
    // MARK: - Performance Utilities
    
    func measureAverageTime(
        iterations: Int = 10,
        _ block: () throws -> Void
    ) rethrows -> TimeInterval {
        var times: [TimeInterval] = []
        
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            try block()
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            times.append(elapsed)
        }
        
        return times.reduce(0, +) / Double(iterations)
    }
    
    func measureMemoryFootprint<T>(
        _ block: () throws -> T
    ) rethrows -> (result: T, memoryUsed: Int64) {
        let startMemory = getMemoryUsage()
        let result = try block()
        let endMemory = getMemoryUsage()
        
        return (result, endMemory - startMemory)
    }
    
    // MARK: - Async Test Helpers
    
    func asyncTest<T>(
        timeout: TimeInterval = 5.0,
        _ block: () async throws -> T
    ) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask {
                try await block()
            }
            
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                throw TestError.timeout
            }
            
            guard let result = try await group.next() else {
                throw TestError.unexpectedNil
            }
            
            group.cancelAll()
            return result
        }
    }
    
    // MARK: - Private Helpers
    
    private func getMemoryUsage() -> Int64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout.size(ofValue: info) / MemoryLayout<integer_t>.size)
        
        let result = withUnsafeMutablePointer(to: &info) { infoPtr in
            infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), intPtr, &count)
            }
        }
        
        return result == KERN_SUCCESS ? Int64(info.resident_size) : 0
    }
}

// MARK: - Supporting Types

enum TestError: Error {
    case timeout
    case unexpectedNil
}

struct SeededRandomGenerator: RandomNumberGenerator {
    private var state: UInt64
    
    init(seed: UInt64) {
        self.state = seed
    }
    
    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
}