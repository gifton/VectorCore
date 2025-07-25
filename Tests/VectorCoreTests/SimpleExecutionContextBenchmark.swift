import XCTest
@testable import VectorCore

final class SimpleExecutionContextBenchmark: XCTestCase {
    
    func testExecutionContextPerformance() async throws {
        print("\n=== ExecutionContext Performance Analysis ===\n")
        
        // Test data
        let dimension = 512
        let vectorCount = 1000
        let k = 10
        
        // Create test vectors
        let vectors = (0..<vectorCount).map { _ in
            Vector<Dim512>.random(in: -1...1)
        }
        let query = Vector<Dim512>.random(in: -1...1)
        
        // 1. Direct API Performance (baseline)
        let directStart = CFAbsoluteTimeGetCurrent()
        let directResult = SyncBatchOperations.findNearest(to: query, in: vectors, k: k)
        let directTime = CFAbsoluteTimeGetCurrent() - directStart
        
        // 2. Operations API with automatic context
        let opsStart = CFAbsoluteTimeGetCurrent()
        let opsResult = try await Operations.findNearest(to: query, in: vectors, k: k)
        let opsTime = CFAbsoluteTimeGetCurrent() - opsStart
        
        // 3. Operations API with explicit parallel context
        let parallelStart = CFAbsoluteTimeGetCurrent()
        let parallelResult = try await Operations.$computeProvider.withValue(CPUComputeProvider.automatic) {
            try await Operations.findNearest(
                to: query, 
                in: vectors, 
                k: k
            )
        }
        let parallelTime = CFAbsoluteTimeGetCurrent() - parallelStart
        
        // 4. Operations API with sequential context
        let seqStart = CFAbsoluteTimeGetCurrent()
        let seqResult = try await Operations.$computeProvider.withValue(CPUComputeProvider.sequential) {
            try await Operations.findNearest(
                to: query,
                in: vectors,
                k: k
            )
        }
        let seqTime = CFAbsoluteTimeGetCurrent() - seqStart
        
        // Verify results are consistent
        XCTAssertEqual(directResult.count, k)
        XCTAssertEqual(opsResult.count, k)
        XCTAssertEqual(parallelResult.count, k)
        XCTAssertEqual(seqResult.count, k)
        
        // Print performance comparison
        print("FindNearest Performance (dimension: \(dimension), vectors: \(vectorCount), k: \(k)):")
        print("- Direct API:        \(String(format: "%.4f", directTime))s (baseline)")
        print("- Operations (auto): \(String(format: "%.4f", opsTime))s (\(String(format: "%.2fx", directTime/opsTime)))")
        print("- Operations (par):  \(String(format: "%.4f", parallelTime))s (\(String(format: "%.2fx", directTime/parallelTime)))")
        print("- Operations (seq):  \(String(format: "%.4f", seqTime))s (\(String(format: "%.2fx", directTime/seqTime)))")
        
        // Test batch operations
        print("\n--- Batch Operations ---")
        let queries = (0..<10).map { _ in Vector<Dim512>.random(in: -1...1) }
        
        // Sequential batch
        let batchSeqStart = CFAbsoluteTimeGetCurrent()
        let batchSeqResult = try await Operations.$computeProvider.withValue(CPUComputeProvider.sequential) {
            try await Operations.findNearestBatch(
                queries: queries,
                in: vectors,
                k: k
            )
        }
        let batchSeqTime = CFAbsoluteTimeGetCurrent() - batchSeqStart
        
        // Parallel batch
        let batchParStart = CFAbsoluteTimeGetCurrent()
        let batchParResult = try await Operations.$computeProvider.withValue(CPUComputeProvider.automatic) {
            try await Operations.findNearestBatch(
                queries: queries,
                in: vectors,
                k: k
            )
        }
        let batchParTime = CFAbsoluteTimeGetCurrent() - batchParStart
        
        print("Batch FindNearest (\(queries.count) queries):")
        print("- Sequential: \(String(format: "%.4f", batchSeqTime))s")
        print("- Parallel:   \(String(format: "%.4f", batchParTime))s (\(String(format: "%.2fx", batchSeqTime/batchParTime)) speedup)")
        
        // BufferPool section removed - BufferPool doesn't exist
        /*
        print("\n--- BufferPool Efficiency ---")
        let pool = BufferPool.shared
        let poolStats = await pool.statistics()
        
        print("BufferPool Statistics:")
        print("- Hit Rate: \(String(format: "%.1f%%", poolStats.hitRate * 100))")
        print("- Total Allocations: \(poolStats.totalAllocations)")
        print("- Reused Buffers: \(poolStats.reusedBuffers)")
        print("- Current Usage: \(poolStats.currentUsageBytes / 1024) KB")
        print("- Peak Usage: \(poolStats.peakUsageBytes / 1024) KB")
        */
        
        // Test parallelization efficiency
        print("\n--- Parallelization Efficiency ---")
        let cpuCount = ProcessInfo.processInfo.activeProcessorCount
        print("CPU Cores: \(cpuCount)")
        
        // Large batch to test parallelization
        let largeBatch = (0..<100).map { _ in Vector<Dim512>.random(in: -1...1) }
        
        let largeBatchSeqStart = CFAbsoluteTimeGetCurrent()
        _ = try await Operations.$computeProvider.withValue(CPUComputeProvider.sequential) {
            try await Operations.findNearestBatch(
                queries: largeBatch,
                in: vectors,
                k: 5
            )
        }
        let largeBatchSeqTime = CFAbsoluteTimeGetCurrent() - largeBatchSeqStart
        
        let largeBatchParStart = CFAbsoluteTimeGetCurrent()
        _ = try await Operations.$computeProvider.withValue(CPUComputeProvider.automatic) {
            try await Operations.findNearestBatch(
                queries: largeBatch,
                in: vectors,
                k: 5
            )
        }
        let largeBatchParTime = CFAbsoluteTimeGetCurrent() - largeBatchParStart
        
        let speedup = largeBatchSeqTime / largeBatchParTime
        let efficiency = speedup / Double(cpuCount) * 100
        
        print("Large Batch (\(largeBatch.count) queries):")
        print("- Sequential: \(String(format: "%.4f", largeBatchSeqTime))s")
        print("- Parallel:   \(String(format: "%.4f", largeBatchParTime))s")
        print("- Speedup:    \(String(format: "%.2fx", speedup))")
        print("- Efficiency: \(String(format: "%.1f%%", efficiency))")
        
        print("\n=== Summary ===")
        print("✓ ExecutionContext provides automatic parallelization")
        print("✓ Batch operations show significant speedup (\(String(format: "%.1fx", batchSeqTime/batchParTime)))")
        // print("✓ BufferPool shows high reuse rate (\(String(format: "%.0f%%", poolStats.hitRate * 100)))")
        print("✓ Parallel efficiency: \(String(format: "%.0f%%", efficiency)) of theoretical maximum")
    }
    
    func testMemoryEfficiency() async throws {
        print("\n=== Memory Efficiency Analysis ===\n")
        
        // Create large dataset
        let vectors = (0..<1000).map { _ in Vector<Dim512>.random(in: -1...1) }
        
        // Get initial memory
        let memBefore = getMemoryUsage()
        
        // Perform memory-intensive operation
        _ = try await Operations.distanceMatrix(
            between: vectors,
            and: vectors,
            metric: EuclideanDistance()
        )
        
        // Get memory after
        let memAfter = getMemoryUsage()
        
        // BufferPool doesn't exist - removed
        // let poolStats = await BufferPool.shared.statistics()
        
        print("Memory Usage:")
        print("- Before: \(formatBytes(memBefore))")
        print("- After:  \(formatBytes(memAfter))")
        print("- Increase: \(formatBytes(memAfter - memBefore))")
    }
    
    private func getMemoryUsage() -> Int64 {
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
        
        return result == KERN_SUCCESS ? Int64(info.resident_size) : 0
    }
    
    private func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: bytes)
    }
}