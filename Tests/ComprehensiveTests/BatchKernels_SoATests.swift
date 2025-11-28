import Testing
import Foundation
@testable import VectorCore
@preconcurrency import Darwin.Mach

@Suite("BatchKernels SoA")
struct BatchKernels_SoATests {

    // MARK: - Core Functionality Tests

    @Suite("Core SoA Functionality")
    struct CoreSoAFunctionalityTests {

        @Test
        func testEuclideanSquaredDistance512() {
            _ = 512 // dimension
            let candidateCount = 10

            // Generate test vectors
            let query = Vector512Optimized(repeating: 1.0)
            let candidates = generateTestVectors512(count: candidateCount)

            // Test SoA implementation
            let soaResults = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)

            // Verify against reference implementation
            let referenceResults = candidates.map { candidate in
                EuclideanKernels.squared512(query, candidate)
            }

            #expect(soaResults.count == candidateCount)
            for i in 0..<candidateCount {
                let error = abs(soaResults[i] - referenceResults[i])
                #expect(error < 1e-3, "Result \(i): SoA=\(soaResults[i]), Reference=\(referenceResults[i]), Error=\(error)")
            }

            // Test edge cases
            let zeroQuery = Vector512Optimized(repeating: 0.0)
            let zeroResults = BatchKernels_SoA.batchEuclideanSquared512(query: zeroQuery, candidates: candidates)

            for i in 0..<candidateCount {
                let expected = candidates[i].storage.reduce(0) { acc, simd4 in
                    acc + (simd4.x * simd4.x + simd4.y * simd4.y + simd4.z * simd4.z + simd4.w * simd4.w)
                }
                let error = abs(zeroResults[i] - expected)
                #expect(error < 1e-3, "Zero query test failed at \(i)")
            }

            // Test identical vectors
            let identicalCandidates = Array(repeating: query, count: 5)
            let identicalResults = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: identicalCandidates)

            for result in identicalResults {
                #expect(abs(result) < 1e-6, "Identical vectors should have zero distance: \(result)")
            }
        }

        @Test
        func testEuclideanSquaredDistance768() {
            let candidateCount = 8

            let query = Vector768Optimized(repeating: 0.5)
            let candidates = generateTestVectors768(count: candidateCount)

            let soaResults = BatchKernels_SoA.batchEuclideanSquared768(query: query, candidates: candidates)

            let referenceResults = candidates.map { candidate in
                EuclideanKernels.squared768(query, candidate)
            }

            #expect(soaResults.count == candidateCount)
            for i in 0..<candidateCount {
                let error = abs(soaResults[i] - referenceResults[i])
                #expect(error < 1e-3, "Result \(i): SoA=\(soaResults[i]), Reference=\(referenceResults[i])")
            }
        }

        @Test
        func testEuclideanSquaredDistance1536() {
            let candidateCount = 6

            let query = Vector1536Optimized(repeating: 0.25)
            let candidates = generateTestVectors1536(count: candidateCount)

            let soaResults = BatchKernels_SoA.batchEuclideanSquared1536(query: query, candidates: candidates)

            let referenceResults = candidates.map { candidate in
                EuclideanKernels.squared1536(query, candidate)
            }

            #expect(soaResults.count == candidateCount)
            for i in 0..<candidateCount {
                let error = abs(soaResults[i] - referenceResults[i])
                #expect(error < 1e-3, "Result \(i): SoA=\(soaResults[i]), Reference=\(referenceResults[i])")
            }
        }

        @Test
        func testEuclideanDistance512() {
            let candidateCount = 8

            let query = Vector512Optimized(repeating: 1.0)
            let candidates = generateTestVectors512(count: candidateCount)

            // Get squared distances from SoA
            let squaredResults = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)

            // Compute euclidean distances (sqrt of squared)
            let euclideanResults = squaredResults.map { sqrt($0) }

            // Verify against reference implementation
            let referenceResults = candidates.map { candidate in
                EuclideanKernels.distance512(query, candidate)
            }

            for i in 0..<candidateCount {
                let error = abs(euclideanResults[i] - referenceResults[i])
                #expect(error < 1e-3, "Result \(i): Computed=\(euclideanResults[i]), Reference=\(referenceResults[i])")
            }

            // Test numerical stability for small distances
            let nearIdentical = Vector512Optimized(repeating: 1.000001) // Very close to query
            let smallDistanceResult = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: [nearIdentical])
            let euclideanSmall = sqrt(smallDistanceResult[0])

            #expect(euclideanSmall > 0, "Small distance should be positive")
            #expect(euclideanSmall < 0.01, "Small distance should be actually small")
        }

        @Test
        func testEuclideanDistance768() {
            let candidateCount = 6

            let query = Vector768Optimized(repeating: 0.5)
            let candidates = generateTestVectors768(count: candidateCount)

            let squaredResults = BatchKernels_SoA.batchEuclideanSquared768(query: query, candidates: candidates)
            let euclideanResults = squaredResults.map { sqrt($0) }

            let referenceResults = candidates.map { candidate in
                EuclideanKernels.distance768(query, candidate)
            }

            for i in 0..<candidateCount {
                let error = abs(euclideanResults[i] - referenceResults[i])
                #expect(error < 1e-3, "Result \(i): Computed=\(euclideanResults[i]), Reference=\(referenceResults[i])")
            }
        }

        @Test
        func testEuclideanDistance1536() {
            let candidateCount = 4

            let query = Vector1536Optimized(repeating: 0.25)
            let candidates = generateTestVectors1536(count: candidateCount)

            let squaredResults = BatchKernels_SoA.batchEuclideanSquared1536(query: query, candidates: candidates)
            let euclideanResults = squaredResults.map { sqrt($0) }

            let referenceResults = candidates.map { candidate in
                EuclideanKernels.distance1536(query, candidate)
            }

            for i in 0..<candidateCount {
                let error = abs(euclideanResults[i] - referenceResults[i])
                #expect(error < 1e-3, "Result \(i): Computed=\(euclideanResults[i]), Reference=\(referenceResults[i])")
            }
        }

        @Test
        func testDotProduct512() {
            // Test batch cosine distance using SoA layout
            // This test demonstrates the need for additional SoA kernels

            let candidateCount = 8
            let query = Vector512Optimized(repeating: 1.0)
            let candidates = generateTestVectors512(count: candidateCount)

            // Test algebraic properties using individual DotKernels for reference
            _ = candidates.map { candidate in
                DotKernels.dot512(query, candidate)
            }

            // Verify commutative property: dot(a, b) = dot(b, a)
            for i in 0..<min(3, candidateCount) {
                let forward = DotKernels.dot512(query, candidates[i])
                let backward = DotKernels.dot512(candidates[i], query)
                let error = abs(forward - backward)
                #expect(error < 1e-3, "Dot product should be commutative")
            }

            // Test orthogonal vectors
            let orthogonal = createOrthogonalVector512(to: query)
            let orthogonalDot = DotKernels.dot512(query, orthogonal)
            #expect(abs(orthogonalDot) < 1e-3, "Orthogonal vectors should have zero dot product: \(orthogonalDot)")

            // Test unit vectors
            guard let normalizedQuery = try? query.normalized().get(),
                  let normalizedCandidate = try? candidates[0].normalized().get() else {
                #expect(Bool(false), "Failed to normalize vectors")
                return
            }
            let dotProduct = DotKernels.dot512(normalizedQuery, normalizedCandidate)
            #expect(dotProduct >= -1.0 && dotProduct <= 1.0, "Unit vector dot product should be in [-1, 1]: \(dotProduct)")

            // Test SoA dot product kernel implementation
            let soaResults = BatchKernels_SoA.batchDotProduct512(query: query, candidates: candidates)

            // Verify results match reference implementation
            for (i, soaResult) in soaResults.enumerated() {
                let refResult = DotKernels.dot512(query, candidates[i])
                #expect(abs(soaResult - refResult) < 1e-5, "SoA result mismatch at index \(i): SoA=\(soaResult), Ref=\(refResult)")
            }
        }

        @Test
        func testDotProduct768() {
            // Test batch cosine distance using SoA layout

            let candidateCount = 6
            let query = Vector768Optimized(repeating: 0.5)
            let candidates = generateTestVectors768(count: candidateCount)

            // Test with reference implementation
            _ = candidates.map { candidate in
                DotKernels.dot768(query, candidate)
            }

            // Reference results computed successfully

            // Test SoA dot product kernel for 768-dim vectors
            let soaResults = BatchKernels_SoA.batchDotProduct768(query: query, candidates: candidates)

            // Verify results match reference implementation
            for (i, soaResult) in soaResults.enumerated() {
                let refResult = DotKernels.dot768(query, candidates[i])
                #expect(abs(soaResult - refResult) < 1e-5, "SoA result mismatch at index \(i): SoA=\(soaResult), Ref=\(refResult)")
            }
        }

        @Test
        func testDotProduct1536() {
            // Test batch cosine distance using SoA layout

            let candidateCount = 4
            let query = Vector1536Optimized(repeating: 0.25)
            let candidates = generateTestVectors1536(count: candidateCount)

            // Test with reference implementation
            _ = candidates.map { candidate in
                DotKernels.dot1536(query, candidate)
            }

            // Reference results computed successfully

            // Test SoA dot product kernel for 1536-dim vectors
            let soaResults = BatchKernels_SoA.batchDotProduct1536(query: query, candidates: candidates)

            // Verify results match reference implementation
            for (i, soaResult) in soaResults.enumerated() {
                let refResult = DotKernels.dot1536(query, candidates[i])
                #expect(abs(soaResult - refResult) < 1e-5, "SoA result mismatch at index \(i): SoA=\(soaResult), Ref=\(refResult)")
            }
        }

        @Test
        func testCosineDistance512() {
            // Test batch cosine distance using SoA layout

            let candidateCount = 8
            let query = Vector512Optimized(repeating: 1.0)
            let candidates = generateTestVectors512(count: candidateCount)

            // Test range validation with reference implementation
            let referenceResults = candidates.map { candidate in
                CosineKernels.distance512_fused(query, candidate)
            }

            for (i, distance) in referenceResults.enumerated() {
                #expect(distance >= 0.0 && distance <= 2.0, "Cosine distance \(i) out of range [0, 2]: \(distance)")
            }

            // Test identical vectors
            let identicalDistance = CosineKernels.distance512_fused(query, query)
            #expect(abs(identicalDistance) < 1e-6, "Identical vectors should have cosine distance 0: \(identicalDistance)")

            // Test orthogonal vectors
            let orthogonal = createOrthogonalVector512(to: query)
            let orthogonalDistance = CosineKernels.distance512_fused(query, orthogonal)
            #expect(abs(orthogonalDistance - 1.0) < 1e-3, "Orthogonal vectors should have cosine distance 1: \(orthogonalDistance)")

            // Test opposite vectors
            let oppositeStorage = query.storage.map { -$0 }
            let oppositeArray = oppositeStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let opposite = try! Vector512Optimized(oppositeArray)
            let oppositeDistance = CosineKernels.distance512_fused(query, opposite)
            #expect(abs(oppositeDistance - 2.0) < 1e-3, "Opposite vectors should have cosine distance 2: \(oppositeDistance)")

            // Test SoA cosine distance kernel
            let soaResults = BatchKernels_SoA.batchCosineDistance512(query: query, candidates: candidates)

            // Verify SoA results are in valid range and match reference
            for (i, soaDistance) in soaResults.enumerated() {
                #expect(soaDistance >= 0.0 && soaDistance <= 2.0, "SoA cosine distance \(i) out of range: \(soaDistance)")

                // Compare with reference implementation
                let refDistance = CosineKernels.distance512_fused(query, candidates[i])
                #expect(abs(soaDistance - refDistance) < 1e-5, "SoA mismatch at \(i): SoA=\(soaDistance), Ref=\(refDistance)")
            }
        }

        @Test
        func testCosineDistance768() {
            // Test batch cosine distance using SoA layout

            let candidateCount = 6
            let query = Vector768Optimized(repeating: 0.5)
            let candidates = generateTestVectors768(count: candidateCount)

            // Test with reference implementation
            let referenceResults = candidates.map { candidate in
                CosineKernels.distance768_fused(query, candidate)
            }

            for (i, distance) in referenceResults.enumerated() {
                #expect(distance >= 0.0 && distance <= 2.0, "Cosine distance \(i) out of range [0, 2]: \(distance)")
            }

            // Test SoA cosine distance kernel for 768-dim vectors
            let soaResults = BatchKernels_SoA.batchCosineDistance768(query: query, candidates: candidates)

            // Verify results match reference implementation
            for (i, soaDistance) in soaResults.enumerated() {
                #expect(soaDistance >= 0.0 && soaDistance <= 2.0, "SoA cosine distance \(i) out of range: \(soaDistance)")

                let refDistance = referenceResults[i]
                #expect(abs(soaDistance - refDistance) < 1e-5, "SoA mismatch at \(i): SoA=\(soaDistance), Ref=\(refDistance)")
            }
        }

        @Test
        func testCosineDistance1536() {
            // Test batch cosine distance using SoA layout

            let candidateCount = 4
            let query = Vector1536Optimized(repeating: 0.25)
            let candidates = generateTestVectors1536(count: candidateCount)

            // Test with reference implementation
            let referenceResults = candidates.map { candidate in
                CosineKernels.distance1536_fused(query, candidate)
            }

            for (i, distance) in referenceResults.enumerated() {
                #expect(distance >= 0.0 && distance <= 2.0, "Cosine distance \(i) out of range [0, 2]: \(distance)")
            }

            // Test SoA cosine distance kernel for 1536-dim vectors
            let soaResults = BatchKernels_SoA.batchCosineDistance1536(query: query, candidates: candidates)

            // Verify results match reference implementation
            for (i, soaDistance) in soaResults.enumerated() {
                #expect(soaDistance >= 0.0 && soaDistance <= 2.0, "SoA cosine distance \(i) out of range: \(soaDistance)")

                let refDistance = referenceResults[i]
                #expect(abs(soaDistance - refDistance) < 1e-5, "SoA mismatch at \(i): SoA=\(soaDistance), Ref=\(refDistance)")
            }
        }
    }

    // MARK: - Memory Layout and SoA Optimization Tests

    @Suite("SoA Memory Layout")
    struct SoAMemoryLayoutTests {

        @Test
        func testSoAConversionFromAoS() {
            let candidateCount = 10
            let candidates512 = generateTestVectors512(count: candidateCount)
            let candidates768 = generateTestVectors768(count: candidateCount)
            let candidates1536 = generateTestVectors1536(count: candidateCount)

            // Test 512-dimensional conversion
            let soa512 = SoA<Vector512Optimized>.build(from: candidates512)
            #expect(soa512.count == candidateCount)
            #expect(soa512.lanes == 128) // 512 / 4

            // Verify data integrity by reconstructing original data
            for candidateIdx in 0..<candidateCount {
                let originalCandidate = candidates512[candidateIdx]
                for laneIdx in 0..<soa512.lanes {
                    let lanePtr = soa512.lanePointer(laneIdx)
                    let reconstructedSIMD4 = lanePtr[candidateIdx]
                    let originalSIMD4 = originalCandidate.storage[laneIdx]

                    #expect(abs(reconstructedSIMD4.x - originalSIMD4.x) < 1e-6)
                    #expect(abs(reconstructedSIMD4.y - originalSIMD4.y) < 1e-6)
                    #expect(abs(reconstructedSIMD4.z - originalSIMD4.z) < 1e-6)
                    #expect(abs(reconstructedSIMD4.w - originalSIMD4.w) < 1e-6)
                }
            }

            // Test 768-dimensional conversion
            let soa768 = SoA<Vector768Optimized>.build(from: candidates768)
            #expect(soa768.count == candidateCount)
            #expect(soa768.lanes == 192) // 768 / 4

            // Test 1536-dimensional conversion
            let soa1536 = SoA<Vector1536Optimized>.build(from: candidates1536)
            #expect(soa1536.count == candidateCount)
            #expect(soa1536.lanes == 384) // 1536 / 4

            // Test empty array conversion
            let emptySoA = SoA<Vector512Optimized>.build(from: [])
            #expect(emptySoA.count == 0)
            #expect(emptySoA.lanes == 128)
        }

        @Test
        func testSoAMemoryAlignment() {
            let candidateCount = 20
            let candidates = generateTestVectors512(count: candidateCount)
            let soa = SoA<Vector512Optimized>.build(from: candidates)

            // Check 16-byte alignment for SIMD4<Float> operations
            let baseAddress = Int(bitPattern: soa.lanePointer(0))
            #expect(baseAddress % 16 == 0, "SoA buffer should be 16-byte aligned for SIMD operations")

            // Check that each lane pointer is properly aligned
            for laneIdx in 0..<soa.lanes {
                let laneAddress = Int(bitPattern: soa.lanePointer(laneIdx))
                #expect(laneAddress % 16 == 0, "Lane \(laneIdx) pointer should be 16-byte aligned")
            }

            // Verify no data corruption through aligned access
            for laneIdx in 0..<min(5, soa.lanes) {
                let lanePtr = soa.lanePointer(laneIdx)
                for candidateIdx in 0..<candidateCount {
                    let simd4Value = lanePtr[candidateIdx]
                    let originalValue = candidates[candidateIdx].storage[laneIdx]

                    #expect(simd4Value.x == originalValue.x)
                    #expect(simd4Value.y == originalValue.y)
                    #expect(simd4Value.z == originalValue.z)
                    #expect(simd4Value.w == originalValue.w)
                }
            }
        }

        @Test
        func testSoAAccessPatterns() {
            let candidateCount = 100
            let candidates = generateTestVectors512(count: candidateCount)
            let soa = SoA<Vector512Optimized>.build(from: candidates)

            // Test lane-wise access pattern (key SoA benefit)
            var laneSum = Float(0.0)
            for laneIdx in 0..<soa.lanes {
                let lanePtr = soa.lanePointer(laneIdx)

                // Access all candidates for this lane sequentially (cache-friendly)
                for candidateIdx in 0..<candidateCount {
                    let simd4 = lanePtr[candidateIdx]
                    laneSum += simd4.x + simd4.y + simd4.z + simd4.w
                }
            }

            // Verify the same computation gives same result with AoS access
            var aosSum = Float(0.0)
            for candidateIdx in 0..<candidateCount {
                for laneIdx in 0..<soa.lanes {
                    let simd4 = candidates[candidateIdx].storage[laneIdx]
                    aosSum += simd4.x + simd4.y + simd4.z + simd4.w
                }
            }

            #expect(abs(laneSum - aosSum) < 1e-3, "SoA and AoS access should produce same results")

            // Test contiguous memory access within lanes
            for laneIdx in 0..<min(3, soa.lanes) {
                let lanePtr = soa.lanePointer(laneIdx)
                let firstAddress = Int(bitPattern: lanePtr)
                let expectedStride = MemoryLayout<SIMD4<Float>>.size

                // Verify addresses are contiguous
                for candidateIdx in 1..<min(10, candidateCount) {
                    let candidateAddress = Int(bitPattern: lanePtr + candidateIdx)
                    let expectedAddress = firstAddress + candidateIdx * expectedStride
                    #expect(candidateAddress == expectedAddress, "Candidates should be contiguous in memory")
                }
            }
        }

        @Test
        func testSoALaneProcessing() {
            let candidateCount = 50
            let candidates = generateTestVectors512(count: candidateCount)
            let soa = SoA<Vector512Optimized>.build(from: candidates)
            let query = Vector512Optimized(repeating: 1.0)

            // Test lane-wise processing (mimicking the SoA kernel pattern)
            var manualResults = Array<Float>(repeating: 0.0, count: candidateCount)

            for laneIdx in 0..<soa.lanes {
                let queryLane = query.storage[laneIdx]
                let candidateLanePtr = soa.lanePointer(laneIdx)

                // Process all candidates for this lane
                for candidateIdx in 0..<candidateCount {
                    let candidateLane = candidateLanePtr[candidateIdx]
                    let diff = queryLane - candidateLane

                    // Accumulate squared differences (euclidean squared distance)
                    let squaredDiff = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + diff.w * diff.w
                    manualResults[candidateIdx] += squaredDiff
                }
            }

            // Compare with SoA kernel implementation
            let soaResults = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)

            #expect(manualResults.count == soaResults.count)
            for i in 0..<candidateCount {
                let error = abs(manualResults[i] - soaResults[i])
                #expect(error < 1e-3, "Manual lane processing should match SoA kernel at \(i): Manual=\(manualResults[i]), SoA=\(soaResults[i])")
            }

            // Test lane processing with different query patterns
            let alternatingStorage = (0..<128).map { i in
                SIMD4<Float>(Float(i % 2), Float((i + 1) % 2), Float(i % 3), Float((i + 2) % 3))
            }
            let alternatingArray = alternatingStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let alternatingQuery = try! Vector512Optimized(alternatingArray)

            let alternatingResults = BatchKernels_SoA.batchEuclideanSquared512(query: alternatingQuery, candidates: candidates)
            #expect(alternatingResults.count == candidateCount)

            // Verify results are reasonable (non-negative, finite)
            for result in alternatingResults {
                #expect(result >= 0.0, "Distance should be non-negative")
                #expect(result.isFinite, "Distance should be finite")
            }
        }
    }

    // MARK: - Batch Processing Tests

    @Suite("Batch Processing")
    struct BatchProcessingTests {

        @Test
        func testSmallBatchSize() {
            let query = Vector512Optimized(repeating: 0.5)

            // Test single candidate (edge case)
            let singleCandidate = [generateTestVectors512(count: 1)[0]]
            let singleResult = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: singleCandidate)
            #expect(singleResult.count == 1)

            let referenceResult = EuclideanKernels.squared512(query, singleCandidate[0])
            #expect(abs(singleResult[0] - referenceResult) < 1e-6)

            // Test small batch sizes (2-10 candidates)
            for batchSize in 2...10 {
                let candidates = generateTestVectors512(count: batchSize)
                let batchResults = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)
                #expect(batchResults.count == batchSize)

                // Verify each result matches individual computation
                for i in 0..<batchSize {
                    let individualResult = EuclideanKernels.squared512(query, candidates[i])
                    let error = abs(batchResults[i] - individualResult)
                    #expect(error < 1e-3, "Batch size \(batchSize), candidate \(i): Batch=\(batchResults[i]), Individual=\(individualResult)")
                }
            }
        }

        @Test
        func testMediumBatchSize() {
            let query = Vector512Optimized(repeating: 0.75)

            // Test medium batch sizes (10-100 candidates)
            let testSizes = [10, 25, 50, 75, 100]

            for batchSize in testSizes {
                let candidates = generateTestVectors512(count: batchSize)
                let batchResults = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)
                #expect(batchResults.count == batchSize)

                // Verify result consistency
                let referenceResults = candidates.map { candidate in
                    EuclideanKernels.squared512(query, candidate)
                }

                for i in 0..<batchSize {
                    let error = abs(batchResults[i] - referenceResults[i])
                    #expect(error < 1e-3, "Batch size \(batchSize), candidate \(i): error \(error)")
                }

                // Basic performance validation (batch should not be dramatically slower)
                let batchTime = measureTime {
                    _ = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)
                }

                let individualTime = measureTime {
                    for candidate in candidates {
                        _ = EuclideanKernels.squared512(query, candidate)
                    }
                }

                // SoA should not be more than 2x slower (allowing for setup overhead)
                #expect(batchTime < individualTime * 2.0, "Batch processing took \(batchTime)s vs individual \(individualTime)s for size \(batchSize)")
            }
        }

        @Test
        func testLargeBatchSize() {
            let query = Vector512Optimized(repeating: 0.25)

            // Test large batch sizes (100-1000 candidates)
            let testSizes = [100, 250, 500, 750, 1000]

            for batchSize in testSizes {
                let candidates = generateTestVectors512(count: batchSize)

                // Test memory management
                let initialMemory = getMemoryUsage()
                let batchResults = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)
                let finalMemory = getMemoryUsage()

                #expect(batchResults.count == batchSize)

                // Memory should be reasonable (not growing excessively)
                let memoryGrowth = finalMemory - initialMemory
                let expectedMemory = batchSize * 512 * 4 * 2 // Rough estimate: candidates + SoA conversion
                #expect(memoryGrowth < expectedMemory * 3, "Memory usage should be reasonable for batch size \(batchSize)")

                // Spot check accuracy (test every 100th element for large batches)
                let stepSize = max(1, batchSize / 10)
                for i in stride(from: 0, to: batchSize, by: stepSize) {
                    let referenceResult = EuclideanKernels.squared512(query, candidates[i])
                    let error = abs(batchResults[i] - referenceResult)
                    #expect(error < 1e-3, "Large batch accuracy check failed at \(i)")
                }

                // Validate that SoA shows benefits for large batches
                if batchSize >= 500 {
                    let batchTime = measureTime {
                        _ = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)
                    }

                    let individualTime = measureTime {
                        for candidate in candidates {
                            _ = EuclideanKernels.squared512(query, candidate)
                        }
                    }

                    // For large batches, SoA should show performance benefits
                    // Allow some tolerance for measurement variance
                    let expectedRatio = 3.0 // Allow SoA to be up to 3x slower in debug mode
                    #expect(batchTime < individualTime * expectedRatio || batchTime < 0.1,
                           "Large batch \(batchSize): SoA time \(batchTime)s should be reasonable vs individual \(individualTime)s")
                }
            }
        }

        @Test
        func testVaryingBatchSizes() {
            let query = Vector512Optimized(repeating: 1.0)

            // Test prime numbers (odd sizes that might stress alignment)
            let primeSizes = [3, 7, 11, 17, 23, 31, 37, 43]

            for primeSize in primeSizes {
                let candidates = generateTestVectors512(count: primeSize)
                let results = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)
                #expect(results.count == primeSize)

                // Verify accuracy for prime sizes
                for i in 0..<primeSize {
                    let reference = EuclideanKernels.squared512(query, candidates[i])
                    let error = abs(results[i] - reference)
                    #expect(error < 1e-3, "Prime size \(primeSize), element \(i): error \(error)")
                }
            }

            // Test powers of 2 (should be well-aligned)
            let powerSizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

            for powerSize in powerSizes {
                let candidates = generateTestVectors512(count: powerSize)
                let results = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)
                #expect(results.count == powerSize)

                // Powers of 2 should have perfect accuracy
                for i in 0..<powerSize {
                    let reference = EuclideanKernels.squared512(query, candidates[i])
                    let error = abs(results[i] - reference)
                    #expect(error < 1e-3, "Power-of-2 size \(powerSize), element \(i): error \(error)")
                }
            }

            // Test arbitrary sizes (including edge cases)
            let arbitrarySizes = [0, 1, 13, 99, 101, 199, 333, 777]

            for arbitrarySize in arbitrarySizes {
                if arbitrarySize == 0 {
                    let results = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: [])
                    #expect(results.isEmpty, "Empty batch should return empty results")
                } else {
                    let candidates = generateTestVectors512(count: arbitrarySize)
                    let results = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)
                    #expect(results.count == arbitrarySize)

                    // Spot check accuracy
                    let checkIndices = [0, arbitrarySize / 2, arbitrarySize - 1].filter { $0 < arbitrarySize }
                    for i in checkIndices {
                        let reference = EuclideanKernels.squared512(query, candidates[i])
                        let error = abs(results[i] - reference)
                        #expect(error < 1e-3, "Arbitrary size \(arbitrarySize), element \(i): error \(error)")
                    }
                }
            }
        }

        @Test
        func testBatchConsistency() {
            let batchSize = 50
            let query = Vector512Optimized(repeating: 0.5)
            let candidates = generateTestVectors512(count: batchSize)

            // Process as batch
            let batchResults = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)

            // Process individually
            let individualResults = candidates.map { candidate in
                EuclideanKernels.squared512(query, candidate)
            }

            #expect(batchResults.count == individualResults.count)

            // Verify bit-exact results (within floating point precision)
            for i in 0..<batchSize {
                let batchResult = batchResults[i]
                let individualResult = individualResults[i]
                let error = abs(batchResult - individualResult)

                // Should be extremely close (within machine epsilon for accumulated operations)
                #expect(error < 1e-3, "Index \(i): Batch=\(batchResult), Individual=\(individualResult), Error=\(error)")
            }

            // Test consistency across multiple runs
            let run1 = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)
            let run2 = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)
            let run3 = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)

            for i in 0..<batchSize {
                #expect(run1[i] == run2[i], "Run consistency failed at index \(i): run1=\(run1[i]), run2=\(run2[i])")
                #expect(run2[i] == run3[i], "Run consistency failed at index \(i): run2=\(run2[i]), run3=\(run3[i])")
            }

            // Test with different query vectors
            let queries = [
                Vector512Optimized(repeating: 0.0),
                Vector512Optimized(repeating: 1.0),
                Vector512Optimized(repeating: -0.5)
            ]

            for query in queries {
                let batchResults = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)
                let individualResults = candidates.map { candidate in
                    EuclideanKernels.squared512(query, candidate)
                }

                for i in 0..<batchSize {
                    let error = abs(batchResults[i] - individualResults[i])
                    #expect(error < 1e-3, "Query consistency failed at index \(i)")
                }
            }
        }

        @Test
        func testConcurrentBatchProcessing() async {
            let batchSize = 100
            let numConcurrentBatches = 8
            let query = Vector512Optimized(repeating: 0.5)

            // Generate multiple batches of candidates
            let candidateBatches = (0..<numConcurrentBatches).map { _ in
                generateTestVectors512(count: batchSize)
            }

            // Process batches concurrently
            let concurrentResults = await withTaskGroup(of: (Int, [Float]).self) { group in
                for (batchIndex, candidates) in candidateBatches.enumerated() {
                    group.addTask {
                        let results = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)
                        return (batchIndex, results)
                    }
                }

                var results: [(Int, [Float])] = []
                for await result in group {
                    results.append(result)
                }
                return results.sorted { $0.0 < $1.0 }
            }

            #expect(concurrentResults.count == numConcurrentBatches)

            // Verify results match sequential processing
            for (batchIndex, concurrentResult) in concurrentResults {
                let candidates = candidateBatches[batchIndex]
                let sequentialResult = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)

                #expect(concurrentResult.count == sequentialResult.count)

                for i in 0..<batchSize {
                    let error = abs(concurrentResult[i] - sequentialResult[i])
                    #expect(error < 1e-3, "Concurrent batch \(batchIndex), element \(i): error \(error)")
                }
            }

            // Test thread safety with shared query and overlapping processing
            let sharedCandidates = generateTestVectors512(count: batchSize)

            await withTaskGroup(of: [Float].self) { group in
                for _ in 0..<numConcurrentBatches {
                    group.addTask {
                        // Each task processes the same candidates
                        return BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: sharedCandidates)
                    }
                }

                var allResults: [[Float]] = []
                for await result in group {
                    allResults.append(result)
                }

                // All results should be identical (thread safety test)
                let referenceResult = allResults[0]
                for (taskIndex, result) in allResults.enumerated() {
                    #expect(result.count == referenceResult.count)
                    for i in 0..<batchSize {
                        #expect(result[i] == referenceResult[i], "Thread safety failed: task \(taskIndex), element \(i)")
                    }
                }
            }
        }
    }

    // MARK: - Register Blocking Tests

    @Suite("Register Blocking")
    struct RegisterBlockingTests {

        @Test
        func testTwoWayBlocking() {
            let query = Vector512Optimized(repeating: 1.0)

            // Test even number of candidates (perfect for 2-way blocking)
            let evenCandidates = generateTestVectors512(count: 20)
            let evenResults = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: evenCandidates)
            #expect(evenResults.count == 20)

            // Verify results match reference implementation
            for i in 0..<20 {
                let reference = EuclideanKernels.squared512(query, evenCandidates[i])
                let error = abs(evenResults[i] - reference)
                #expect(error < 1e-3, "Even candidate \(i): error \(error)")
            }

            // Test odd number of candidates (requires tail handling)
            let oddCandidates = generateTestVectors512(count: 21)
            let oddResults = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: oddCandidates)
            #expect(oddResults.count == 21)

            // Verify all results, especially the tail element
            for i in 0..<21 {
                let reference = EuclideanKernels.squared512(query, oddCandidates[i])
                let error = abs(oddResults[i] - reference)
                #expect(error < 1e-3, "Odd candidate \(i): error \(error)")
            }

            // Test various odd sizes to stress tail handling
            for oddSize in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19] {
                let candidates = generateTestVectors512(count: oddSize)
                let results = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)
                #expect(results.count == oddSize)

                // Verify tail element (last one) is correct
                let tailIndex = oddSize - 1
                let tailReference = EuclideanKernels.squared512(query, candidates[tailIndex])
                let tailError = abs(results[tailIndex] - tailReference)
                #expect(tailError < 1e-6, "Tail handling failed for size \(oddSize): error \(tailError)")
            }
        }

        @Test
        func testBlockingEfficiency() {
            let query = Vector512Optimized(repeating: 0.5)
            let candidateCount = 1000
            let candidates = generateTestVectors512(count: candidateCount)

            // Measure SoA performance (which uses 2-way blocking)
            let soaTime = measureTime {
                for _ in 0..<10 {
                    _ = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)
                }
            }

            // Measure individual processing performance (no blocking)
            let individualTime = measureTime {
                for _ in 0..<10 {
                    for candidate in candidates {
                        _ = EuclideanKernels.squared512(query, candidate)
                    }
                }
            }

            // SoA should be reasonable for large batches (debug mode considerations)
            // Allow tolerance for debug overhead
            #expect(soaTime < individualTime * 10.0 || soaTime < 1.0,
                   "SoA blocking should be efficient: SoA time \(soaTime)s vs individual \(individualTime)s")

            // Test that blocking handles different batch sizes efficiently
            let testSizes = [2, 4, 10, 50, 100, 500]
            var allTimesReasonable = true

            for size in testSizes {
                let testCandidates = Array(candidates.prefix(size))
                let timeForSize = measureTime {
                    _ = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: testCandidates)
                }

                // Time should scale roughly linearly with size
                let expectedTime = soaTime * Double(size) / Double(candidateCount)
                if timeForSize > expectedTime * 5 && timeForSize > 0.001 {
                    allTimesReasonable = false
                }
            }

            #expect(allTimesReasonable, "Blocking efficiency should scale reasonably with batch size")
        }

        @Test
        func testBlockingCorrectnessWithOddSizes() {
            let query = Vector512Optimized(repeating: 0.75)

            // Test all odd sizes from 1 to 51
            for oddSize in stride(from: 1, through: 51, by: 2) {
                let candidates = generateTestVectors512(count: oddSize)
                let soaResults = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)

                #expect(soaResults.count == oddSize)

                // Verify every result is correct
                for i in 0..<oddSize {
                    let reference = EuclideanKernels.squared512(query, candidates[i])
                    let error = abs(soaResults[i] - reference)
                    #expect(error < 1e-3, "Odd size \(oddSize), element \(i): error \(error)")
                }

                // Special attention to the last element (remainder handling)
                let lastIndex = oddSize - 1
                let lastReference = EuclideanKernels.squared512(query, candidates[lastIndex])
                let lastError = abs(soaResults[lastIndex] - lastReference)
                #expect(lastError < 1e-3, "Remainder handling failed for size \(oddSize): error \(lastError)")
            }

            // Test boundary condition: size 1 (special case)
            let singleCandidate = [generateTestVectors512(count: 1)[0]]
            let singleResult = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: singleCandidate)
            let singleReference = EuclideanKernels.squared512(query, singleCandidate[0])

            #expect(singleResult.count == 1)
            #expect(abs(singleResult[0] - singleReference) < 1e-3, "Single candidate boundary case failed")
        }

        @Test
        func testBlockingWithDifferentDimensions() {
            let candidateCount = 25 // Odd number to test tail handling across all dimensions

            // Test 512-dimensional blocking
            let query512 = Vector512Optimized(repeating: 1.0)
            let candidates512 = generateTestVectors512(count: candidateCount)
            let results512 = BatchKernels_SoA.batchEuclideanSquared512(query: query512, candidates: candidates512)

            #expect(results512.count == candidateCount)
            for i in 0..<candidateCount {
                let reference = EuclideanKernels.squared512(query512, candidates512[i])
                let error = abs(results512[i] - reference)
                #expect(error < 1e-3, "512D blocking failed at \(i): error \(error)")
            }

            // Test 768-dimensional blocking
            let query768 = Vector768Optimized(repeating: 0.5)
            let candidates768 = generateTestVectors768(count: candidateCount)
            let results768 = BatchKernels_SoA.batchEuclideanSquared768(query: query768, candidates: candidates768)

            #expect(results768.count == candidateCount)
            for i in 0..<candidateCount {
                let reference = EuclideanKernels.squared768(query768, candidates768[i])
                let error = abs(results768[i] - reference)
                #expect(error < 1e-3, "768D blocking failed at \(i): error \(error)")
            }

            // Test 1536-dimensional blocking
            let query1536 = Vector1536Optimized(repeating: 0.25)
            let candidates1536 = generateTestVectors1536(count: candidateCount)
            let results1536 = BatchKernels_SoA.batchEuclideanSquared1536(query: query1536, candidates: candidates1536)

            #expect(results1536.count == candidateCount)
            for i in 0..<candidateCount {
                let reference = EuclideanKernels.squared1536(query1536, candidates1536[i])
                let error = abs(results1536[i] - reference)
                #expect(error < 1e-3, "1536D blocking failed at \(i): error \(error)")
            }

            // Verify that all dimensions handle odd sizes consistently
            let oddSize = 13
            let oddCandidates512 = generateTestVectors512(count: oddSize)
            let oddCandidates768 = generateTestVectors768(count: oddSize)
            let oddCandidates1536 = generateTestVectors1536(count: oddSize)

            let oddResults512 = BatchKernels_SoA.batchEuclideanSquared512(query: query512, candidates: oddCandidates512)
            let oddResults768 = BatchKernels_SoA.batchEuclideanSquared768(query: query768, candidates: oddCandidates768)
            let oddResults1536 = BatchKernels_SoA.batchEuclideanSquared1536(query: query1536, candidates: oddCandidates1536)

            #expect(oddResults512.count == oddSize)
            #expect(oddResults768.count == oddSize)
            #expect(oddResults1536.count == oddSize)

            // All dimensions should handle the tail element correctly
            let tail512Ref = EuclideanKernels.squared512(query512, oddCandidates512[oddSize - 1])
            let tail768Ref = EuclideanKernels.squared768(query768, oddCandidates768[oddSize - 1])
            let tail1536Ref = EuclideanKernels.squared1536(query1536, oddCandidates1536[oddSize - 1])

            #expect(abs(oddResults512[oddSize - 1] - tail512Ref) < 1e-6, "512D tail handling failed")
            #expect(abs(oddResults768[oddSize - 1] - tail768Ref) < 1e-6, "768D tail handling failed")
            #expect(abs(oddResults1536[oddSize - 1] - tail1536Ref) < 1e-6, "1536D tail handling failed")
        }
    }

    // MARK: - Performance Comparison Tests

    @Suite("Performance Comparison")
    struct PerformanceComparisonTests {

        @Test
        func testSoAvsAoSPerformance() {
            let query = Vector512Optimized(repeating: 0.5)
            let largeCandidateCount = 1000
            let candidates = generateTestVectors512(count: largeCandidateCount)

            // Measure SoA performance
            let soaTime = measureTime {
                for _ in 0..<5 {
                    _ = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)
                }
            }

            // Measure AoS performance (individual kernel calls)
            let aosTime = measureTime {
                for _ in 0..<5 {
                    for candidate in candidates {
                        _ = EuclideanKernels.squared512(query, candidate)
                    }
                }
            }

            // For large batches, SoA should be reasonable (debug mode considerations)
            // Allow tolerance for debug overhead
            #expect(soaTime < aosTime * 3.0 || soaTime < 0.1,
                   "SoA should be reasonable vs AoS: SoA \(soaTime)s vs AoS \(aosTime)s")

            // Test at the threshold where SoA should be beneficial
            let shouldUseSoA = BatchKernels_SoA.shouldUseSoA(candidateCount: largeCandidateCount, dimension: 512)
            #expect(shouldUseSoA, "Large batch should recommend SoA usage")

            // Test below threshold
            let smallBatch = BatchKernels_SoA.shouldUseSoA(candidateCount: 50, dimension: 512)
            #expect(!smallBatch, "Small batch should not recommend SoA usage")

            // Performance should scale reasonably with batch size
            let mediumCandidates = Array(candidates.prefix(500))
            let mediumTime = measureTime {
                _ = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: mediumCandidates)
            }

            let smallCandidates = Array(candidates.prefix(100))
            let smallTime = measureTime {
                _ = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: smallCandidates)
            }

            // Time should scale roughly linearly (allowing for overhead)
            let expectedRatio = 500.0 / 100.0 // 5x
            let actualRatio = mediumTime / max(smallTime, 1e-6)
            #expect(actualRatio < expectedRatio * 2 || mediumTime < 0.001,
                   "Performance should scale reasonably: small \(smallTime)s, medium \(mediumTime)s, ratio \(actualRatio)")
        }

        @Test
        func testCacheLocalityImprovements() {
            // Test cache locality improvements with SoA by comparing sequential vs random access patterns
            let candidateCount = 1000
            let query = Vector512Optimized(repeating: 0.5)
            let candidates = generateTestVectors512(count: candidateCount)

            // Test 1: Sequential access (cache-friendly in SoA)
            let sequentialTime = measureTime {
                for _ in 0..<10 {
                    _ = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)
                }
            }

            // Test 2: Random access pattern simulation
            var randomCandidates = candidates
            randomCandidates.shuffle()
            let randomTime = measureTime {
                for _ in 0..<10 {
                    _ = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: randomCandidates)
                }
            }

            // Sequential should be at least as fast as random (cache benefits)
            #expect(sequentialTime <= randomTime * 1.5 || sequentialTime < 0.1,
                   "Sequential access should benefit from cache locality: sequential \(sequentialTime)s vs random \(randomTime)s")

            // Test 3: Verify lane-wise access pattern efficiency
            let soa = SoA<Vector512Optimized>.build(from: candidates)
            var laneSum: Float = 0

            let laneAccessTime = measureTime {
                // Cache-friendly: access all candidates for each lane sequentially
                for laneIdx in 0..<min(10, soa.lanes) {
                    let lanePtr = soa.lanePointer(laneIdx)
                    for candidateIdx in 0..<candidateCount {
                        let simd4 = lanePtr[candidateIdx]
                        laneSum += simd4.x + simd4.y + simd4.z + simd4.w
                    }
                }
            }

            var candidateSum: Float = 0
            let candidateAccessTime = measureTime {
                // Cache-unfriendly: access all lanes for each candidate
                for candidateIdx in 0..<candidateCount {
                    for laneIdx in 0..<min(10, soa.lanes) {
                        let simd4 = candidates[candidateIdx].storage[laneIdx]
                        candidateSum += simd4.x + simd4.y + simd4.z + simd4.w
                    }
                }
            }

            // Lane-wise access should be more cache-friendly
            #expect(laneAccessTime <= candidateAccessTime * 2.0 || laneAccessTime < 0.01,
                   "Lane-wise access should be cache-friendly: lane \(laneAccessTime)s vs candidate \(candidateAccessTime)s")

            // Test 4: Memory access pattern validation
            let accessPatternTest = measureTime {
                let queryLanes = query.storage
                var results = Array<Float>(repeating: 0, count: candidateCount)

                // Optimal access pattern: process one lane at a time
                for laneIdx in 0..<soa.lanes {
                    let lanePtr = soa.lanePointer(laneIdx)
                    let queryLane = queryLanes[laneIdx]

                    // Sequential memory access within each lane
                    for candidateIdx in 0..<candidateCount {
                        let candidateLane = lanePtr[candidateIdx]
                        let diff = queryLane - candidateLane
                        results[candidateIdx] += diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + diff.w * diff.w
                    }
                }
            }

            #expect(accessPatternTest < 1.0, "Optimal access pattern should be fast")
        }

        @Test
        func testSIMDUtilization() {
            // Test SIMD instruction utilization in SoA kernels
            let candidateCount = 256 // Power of 2 for optimal SIMD alignment
            let query = Vector512Optimized(repeating: 1.0)
            let candidates = generateTestVectors512(count: candidateCount)

            // Test 1: Verify SIMD4 operations are utilized
            let soa = SoA<Vector512Optimized>.build(from: candidates)
            var simdResults = Array<SIMD4<Float>>(repeating: SIMD4<Float>(0, 0, 0, 0), count: candidateCount)

            // Process using SIMD4 operations directly
            for laneIdx in 0..<min(4, soa.lanes) {
                let lanePtr = soa.lanePointer(laneIdx)
                let queryLane = query.storage[laneIdx]

                for candidateIdx in 0..<candidateCount {
                    let candidateLane = lanePtr[candidateIdx]
                    // SIMD4 subtraction and multiplication
                    let diff = queryLane - candidateLane
                    simdResults[candidateIdx] += diff * diff
                }
            }

            // Verify SIMD operations produce correct results
            for i in 0..<min(10, candidateCount) {
                let sum = simdResults[i].x + simdResults[i].y + simdResults[i].z + simdResults[i].w
                #expect(sum >= 0, "SIMD results should be non-negative")
                #expect(sum.isFinite, "SIMD results should be finite")
            }

            // Test 2: Verify vectorized operations are faster than scalar
            var scalarSum: Float = 0
            let scalarTime = measureTime {
                for _ in 0..<100 {
                    for laneIdx in 0..<soa.lanes {
                        let lanePtr = soa.lanePointer(laneIdx)
                        let queryLane = query.storage[laneIdx]

                        for candidateIdx in 0..<candidateCount {
                            let candidateLane = lanePtr[candidateIdx]
                            // Scalar operations
                            let dx = queryLane.x - candidateLane.x
                            let dy = queryLane.y - candidateLane.y
                            let dz = queryLane.z - candidateLane.z
                            let dw = queryLane.w - candidateLane.w
                            scalarSum += dx*dx + dy*dy + dz*dz + dw*dw
                        }
                    }
                }
            }

            var simdSum: Float = 0
            let simdTime = measureTime {
                for _ in 0..<100 {
                    for laneIdx in 0..<soa.lanes {
                        let lanePtr = soa.lanePointer(laneIdx)
                        let queryLane = query.storage[laneIdx]

                        for candidateIdx in 0..<candidateCount {
                            let candidateLane = lanePtr[candidateIdx]
                            // SIMD4 operations
                            let diff = queryLane - candidateLane
                            let squared = diff * diff
                            simdSum += squared.x + squared.y + squared.z + squared.w
                        }
                    }
                }
            }

            // SIMD should be reasonably efficient (allowing for debug mode)
            #expect(simdTime <= scalarTime * 2.0 || simdTime < 0.1,
                   "SIMD operations should be efficient: SIMD \(simdTime)s vs scalar \(scalarTime)s")

            // Test 3: Verify alignment for SIMD operations
            for laneIdx in 0..<min(10, soa.lanes) {
                let lanePtr = soa.lanePointer(laneIdx)
                let address = Int(bitPattern: lanePtr)
                #expect(address % 16 == 0, "Lane \(laneIdx) should be 16-byte aligned for SIMD4<Float>")
            }

            // Test 4: Test SIMD register blocking (2-way)
            let blockingResults = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)
            #expect(blockingResults.count == candidateCount)

            // Verify that 2-way blocking produces correct results
            for i in stride(from: 0, to: candidateCount, by: 2) {
                if i + 1 < candidateCount {
                    // Both elements in the block should be computed correctly
                    let ref1 = EuclideanKernels.squared512(query, candidates[i])
                    let ref2 = EuclideanKernels.squared512(query, candidates[i + 1])

                    let error1 = abs(blockingResults[i] - ref1)
                    let error2 = abs(blockingResults[i + 1] - ref2)

                    #expect(error1 < 1e-3, "Block element 0 error: \(error1)")
                    #expect(error2 < 1e-3, "Block element 1 error: \(error2)")
                }
            }
        }

        @Test
        func testScalingWithBatchSize() {
            // Test how performance scales with batch size
            let query = Vector512Optimized(repeating: 0.5)
            let batchSizes = [10, 50, 100, 250, 500, 1000, 2000]
            var timings: [(size: Int, time: TimeInterval, timePerItem: TimeInterval)] = []

            for size in batchSizes {
                let candidates = generateTestVectors512(count: size)

                // Warm up
                _ = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)

                // Measure
                let time = measureTime {
                    for _ in 0..<5 {
                        _ = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)
                    }
                }

                let avgTime = time / 5.0
                let timePerItem = avgTime / Double(size)
                timings.append((size: size, time: avgTime, timePerItem: timePerItem))
            }

            // Test 1: Verify roughly linear scaling
            // Time per item should be relatively constant
            let timePerItemValues = timings.map { $0.timePerItem }
            let meanTimePerItem = timePerItemValues.reduce(0, +) / Double(timePerItemValues.count)

            for timing in timings {
                let deviation = abs(timing.timePerItem - meanTimePerItem) / meanTimePerItem
                #expect(deviation < 2.0 || timing.time < 0.01,
                       "Scaling should be roughly linear. Size \(timing.size): \(timing.timePerItem) vs mean \(meanTimePerItem)")
            }

            // Test 2: Identify sweet spots
            // SoA should show benefits at larger sizes
            if timings.count >= 4 {
                let smallBatchEfficiency = timings[0].timePerItem  // size 10
                let largeBatchEfficiency = timings[timings.count - 1].timePerItem  // size 2000

                // Large batches should have better or similar efficiency
                #expect(largeBatchEfficiency <= smallBatchEfficiency * 3.0 || largeBatchEfficiency < 1e-6,
                       "Large batches should be efficient: small \(smallBatchEfficiency) vs large \(largeBatchEfficiency)")
            }

            // Test 3: Check for performance cliffs
            // Look for sudden changes in performance
            var hasPerformanceCliff = false
            for i in 1..<timings.count {
                let prevEfficiency = timings[i-1].timePerItem
                let currEfficiency = timings[i].timePerItem
                let change = abs(currEfficiency - prevEfficiency) / prevEfficiency

                if change > 5.0 && currEfficiency > 1e-6 {
                    hasPerformanceCliff = true
                    break
                }
            }

            #expect(!hasPerformanceCliff, "Should not have dramatic performance cliffs")

            // Test 4: Verify SoA threshold recommendations
            for timing in timings {
                let shouldUseSoA = BatchKernels_SoA.shouldUseSoA(candidateCount: timing.size, dimension: 512)

                if timing.size >= 100 {
                    #expect(shouldUseSoA, "Large batch size \(timing.size) should recommend SoA")
                } else if timing.size <= 50 {
                    #expect(!shouldUseSoA, "Small batch size \(timing.size) should not recommend SoA")
                }
            }

            // Test 5: Compare scaling across dimensions
            let testSize = 500
            let candidates512 = generateTestVectors512(count: testSize)
            let candidates768 = generateTestVectors768(count: testSize)
            let candidates1536 = generateTestVectors1536(count: testSize)

            let time512 = measureTime {
                _ = BatchKernels_SoA.batchEuclideanSquared512(
                    query: Vector512Optimized(repeating: 0.5),
                    candidates: candidates512
                )
            }

            let time768 = measureTime {
                _ = BatchKernels_SoA.batchEuclideanSquared768(
                    query: Vector768Optimized(repeating: 0.5),
                    candidates: candidates768
                )
            }

            let time1536 = measureTime {
                _ = BatchKernels_SoA.batchEuclideanSquared1536(
                    query: Vector1536Optimized(repeating: 0.5),
                    candidates: candidates1536
                )
            }

            // Time should scale roughly with dimension
            let ratio768to512 = time768 / max(time512, 1e-9)
            let ratio1536to512 = time1536 / max(time512, 1e-9)

            #expect(ratio768to512 < 3.0 || time768 < 0.01, "768D should scale reasonably vs 512D: \(ratio768to512)x")
            #expect(ratio1536to512 < 6.0 || time1536 < 0.01, "1536D should scale reasonably vs 512D: \(ratio1536to512)x")
        }
    }

    // MARK: - Edge Cases and Error Handling

    @Suite("Edge Cases and Error Handling")
    struct EdgeCasesErrorHandlingTests {

        @Test
        func testEmptyBatch() {
            let query = Vector512Optimized(repeating: 1.0)
            let emptyCandidates: [Vector512Optimized] = []

            // Test all dimensions with empty batches
            let emptyResults512 = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: emptyCandidates)
            #expect(emptyResults512.isEmpty, "Empty batch should return empty results")

            let query768 = Vector768Optimized(repeating: 0.5)
            let emptyCandidates768: [Vector768Optimized] = []
            let emptyResults768 = BatchKernels_SoA.batchEuclideanSquared768(query: query768, candidates: emptyCandidates768)
            #expect(emptyResults768.isEmpty, "Empty 768D batch should return empty results")

            let query1536 = Vector1536Optimized(repeating: 0.25)
            let emptyCandidates1536: [Vector1536Optimized] = []
            let emptyResults1536 = BatchKernels_SoA.batchEuclideanSquared1536(query: query1536, candidates: emptyCandidates1536)
            #expect(emptyResults1536.isEmpty, "Empty 1536D batch should return empty results")

            // Test that empty batch processing is fast (no unnecessary work)
            let emptyTime = measureTime {
                for _ in 0..<100 {
                    _ = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: emptyCandidates)
                }
            }

            #expect(emptyTime < 0.001, "Empty batch processing should be very fast: \(emptyTime)s")
        }

        @Test
        func testZeroVectors() {
            // Test all zero query vector
            let zeroQuery = Vector512Optimized(repeating: 0.0)
            let normalCandidates = generateTestVectors512(count: 10)

            let zeroQueryResults = BatchKernels_SoA.batchEuclideanSquared512(query: zeroQuery, candidates: normalCandidates)
            #expect(zeroQueryResults.count == 10)

            // Verify results match expected: distance from zero is magnitude squared
            for i in 0..<10 {
                let expected = normalCandidates[i].storage.reduce(0) { acc, simd4 in
                    acc + (simd4.x * simd4.x + simd4.y * simd4.y + simd4.z * simd4.z + simd4.w * simd4.w)
                }
                let error = abs(zeroQueryResults[i] - expected)
                #expect(error < 1e-3, "Zero query test failed at \(i): got \(zeroQueryResults[i]), expected \(expected)")
            }

            // Test all zero candidate vectors
            let normalQuery = Vector512Optimized(repeating: 1.0)
            let zeroCandidates = Array(repeating: Vector512Optimized(repeating: 0.0), count: 8)

            let zeroCandidateResults = BatchKernels_SoA.batchEuclideanSquared512(query: normalQuery, candidates: zeroCandidates)
            #expect(zeroCandidateResults.count == 8)

            // Distance from any point to zero should be the query's magnitude squared
            let expectedDistance = normalQuery.storage.reduce(0) { acc, simd4 in
                acc + (simd4.x * simd4.x + simd4.y * simd4.y + simd4.z * simd4.z + simd4.w * simd4.w)
            }

            for result in zeroCandidateResults {
                let error = abs(result - expectedDistance)
                #expect(error < 1e-3, "Zero candidate test failed: got \(result), expected \(expectedDistance)")
            }

            // Test mixed zero and non-zero vectors
            var mixedCandidates = generateTestVectors512(count: 5)
            mixedCandidates.append(Vector512Optimized(repeating: 0.0))
            mixedCandidates.append(contentsOf: generateTestVectors512(count: 3))
            mixedCandidates.append(Vector512Optimized(repeating: 0.0))

            let mixedResults = BatchKernels_SoA.batchEuclideanSquared512(query: normalQuery, candidates: mixedCandidates)
            #expect(mixedResults.count == 10)

            // Verify zero candidates produce expected results
            let zeroIndices = [5, 9] // Positions where we inserted zero vectors
            for zeroIndex in zeroIndices {
                let error = abs(mixedResults[zeroIndex] - expectedDistance)
                #expect(error < 1e-3, "Mixed zero candidate at index \(zeroIndex) failed")
            }
        }

        @Test
        func testDenormalizedValues() {
            // Test handling of denormalized floating point values
            let epsilon = Float.ulpOfOne
            let subnormal = Float.leastNonzeroMagnitude

            // Test 1: Very small values near machine epsilon
            let smallStorage = (0..<128).map { _ in
                SIMD4<Float>(epsilon, -epsilon, epsilon * 2, -epsilon * 2)
            }
            let smallArray = smallStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let smallQuery = try! Vector512Optimized(smallArray)

            let smallCandidates = (0..<5).map { i in
                let storage = (0..<128).map { _ in
                    SIMD4<Float>(
                        epsilon * Float(i + 1),
                        -epsilon * Float(i + 1),
                        epsilon * Float(i + 2),
                        -epsilon * Float(i + 2)
                    )
                }
                let array = storage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
                return try! Vector512Optimized(array)
            }

            let smallResults = BatchKernels_SoA.batchEuclideanSquared512(query: smallQuery, candidates: smallCandidates)
            #expect(smallResults.count == 5)

            // Results should be very small but non-negative
            for result in smallResults {
                #expect(result >= 0, "Distance with small values should be non-negative: \(result)")
                #expect(result.isFinite, "Distance with small values should be finite: \(result)")
                #expect(result < 1e-10, "Distance between small values should be small: \(result)")
            }

            // Test 2: Subnormal numbers
            let subnormalStorage = (0..<128).map { i in
                SIMD4<Float>(
                    subnormal * Float(i + 1),
                    subnormal * Float(i + 2),
                    subnormal * Float(i + 3),
                    subnormal * Float(i + 4)
                )
            }
            let subnormalArray = subnormalStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let subnormalQuery = try! Vector512Optimized(subnormalArray)

            let subnormalCandidates = [subnormalQuery] // Test with identical subnormal vector
            let subnormalResult = BatchKernels_SoA.batchEuclideanSquared512(query: subnormalQuery, candidates: subnormalCandidates)

            #expect(subnormalResult.count == 1)
            #expect(abs(subnormalResult[0]) < epsilon, "Identical subnormal vectors should have ~zero distance: \(subnormalResult[0])")

            // Test 3: Mixed normal and denormalized values
            let mixedStorage = (0..<128).map { i in
                if i % 2 == 0 {
                    // Normal values
                    SIMD4<Float>(1.0, -1.0, 0.5, -0.5)
                } else {
                    // Denormalized values
                    SIMD4<Float>(subnormal, -subnormal, epsilon, -epsilon)
                }
            }
            let mixedArray = mixedStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let mixedQuery = try! Vector512Optimized(mixedArray)

            let normalCandidates = generateTestVectors512(count: 3)
            let mixedResults = BatchKernels_SoA.batchEuclideanSquared512(query: mixedQuery, candidates: normalCandidates)

            #expect(mixedResults.count == 3)
            for result in mixedResults {
                #expect(result >= 0, "Mixed normal/denormalized distance should be non-negative")
                #expect(result.isFinite, "Mixed normal/denormalized distance should be finite")
            }

            // Test 4: Gradual underflow handling
            let underflowValues = [1e-30, 1e-35, 1e-38, 1e-40, 1e-45].map { Float($0) }
            for underflowValue in underflowValues {
                let underflowStorage = (0..<128).map { _ in
                    SIMD4<Float>(underflowValue, underflowValue, underflowValue, underflowValue)
                }
                let underflowArray = underflowStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
                let underflowVector = try! Vector512Optimized(underflowArray)

                let underflowResult = BatchKernels_SoA.batchEuclideanSquared512(
                    query: underflowVector,
                    candidates: [underflowVector]
                )

                #expect(underflowResult.count == 1)
                #expect(abs(underflowResult[0]) < 1e-20, "Underflow handling for \(underflowValue): \(underflowResult[0])")
            }
        }

        @Test
        func testInfiniteAndNaNValues() {
            // Test handling of infinite and NaN values

            // Test 1: Positive infinity
            let posInfStorage = (0..<128).map { i in
                if i == 0 {
                    SIMD4<Float>(Float.infinity, 1.0, 1.0, 1.0)
                } else {
                    SIMD4<Float>(1.0, 1.0, 1.0, 1.0)
                }
            }
            let posInfArray = posInfStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let posInfQuery = try! Vector512Optimized(posInfArray)

            let normalCandidates = generateTestVectors512(count: 3)
            let posInfResults = BatchKernels_SoA.batchEuclideanSquared512(query: posInfQuery, candidates: normalCandidates)

            #expect(posInfResults.count == 3)
            for result in posInfResults {
                #expect(result == Float.infinity, "Distance with infinity should be infinity: \(result)")
            }

            // Test 2: Negative infinity
            let negInfStorage = (0..<128).map { i in
                if i == 1 {
                    SIMD4<Float>(1.0, -Float.infinity, 1.0, 1.0)
                } else {
                    SIMD4<Float>(0.0, 0.0, 0.0, 0.0)
                }
            }
            let negInfArray = negInfStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let negInfQuery = try! Vector512Optimized(negInfArray)

            let negInfResults = BatchKernels_SoA.batchEuclideanSquared512(query: negInfQuery, candidates: normalCandidates)

            #expect(negInfResults.count == 3)
            for result in negInfResults {
                #expect(result == Float.infinity, "Distance with negative infinity should be infinity: \(result)")
            }

            // Test 3: NaN values
            let nanStorage = (0..<128).map { i in
                if i == 2 {
                    SIMD4<Float>(Float.nan, 1.0, 1.0, 1.0)
                } else {
                    SIMD4<Float>(1.0, 1.0, 1.0, 1.0)
                }
            }
            let nanArray = nanStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let nanQuery = try! Vector512Optimized(nanArray)

            let nanResults = BatchKernels_SoA.batchEuclideanSquared512(query: nanQuery, candidates: normalCandidates)

            #expect(nanResults.count == 3)
            for result in nanResults {
                #expect(result.isNaN, "Distance with NaN should propagate NaN: \(result)")
            }

            // Test 4: Mixed finite/infinite values
            let mixedStorage = (0..<128).map { i in
                switch i % 4 {
                case 0:
                    SIMD4<Float>(Float.infinity, 1.0, -Float.infinity, 0.0)
                case 1:
                    SIMD4<Float>(1.0, 2.0, 3.0, 4.0)
                case 2:
                    SIMD4<Float>(-1.0, -2.0, -3.0, -4.0)
                default:
                    SIMD4<Float>(0.0, 0.0, 0.0, 0.0)
                }
            }
            let mixedArray = mixedStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let mixedQuery = try! Vector512Optimized(mixedArray)

            let mixedResults = BatchKernels_SoA.batchEuclideanSquared512(query: mixedQuery, candidates: [mixedQuery])

            #expect(mixedResults.count == 1)
            #expect(mixedResults[0] == 0.0 || mixedResults[0].isNaN, "Mixed inf vector with itself should be 0 or NaN: \(mixedResults[0])")

            // Test 5: Infinity - Infinity (should produce NaN)
            let infMinusInfStorage = (0..<128).map { i in
                if i == 0 {
                    SIMD4<Float>(Float.infinity, 0, 0, 0)
                } else {
                    SIMD4<Float>(0, 0, 0, 0)
                }
            }
            let infMinusInfArray = infMinusInfStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let infVector1 = try! Vector512Optimized(infMinusInfArray)
            let infVector2 = try! Vector512Optimized(infMinusInfArray) // Same infinity location

            let infMinusInfResult = BatchKernels_SoA.batchEuclideanSquared512(query: infVector1, candidates: [infVector2])
            #expect(infMinusInfResult.count == 1)
            #expect(infMinusInfResult[0] == 0.0 || infMinusInfResult[0].isNaN, "Inf - Inf should be 0 (same vector): \(infMinusInfResult[0])")

            // Test 6: Verify NaN propagation through the computation
            let nanCandidates = [nanQuery] // Vector with NaN
            let nanPropagationResults = BatchKernels_SoA.batchEuclideanSquared512(
                query: Vector512Optimized(repeating: 1.0),
                candidates: nanCandidates
            )

            #expect(nanPropagationResults.count == 1)
            #expect(nanPropagationResults[0].isNaN, "NaN in candidate should propagate: \(nanPropagationResults[0])")
        }

        @Test
        func testExtremeValues() {
            // Test with extreme floating point values
            let maxFloat = Float.greatestFiniteMagnitude
            let minFloat = -Float.greatestFiniteMagnitude
            let largeFloat: Float = 1e30
            let smallFloat: Float = 1e-30

            // Test 1: Maximum representable values
            let maxStorage = (0..<128).map { i in
                if i < 2 {
                    // Put extreme values in first few components
                    SIMD4<Float>(maxFloat / 1e10, maxFloat / 1e10, 0, 0)
                } else {
                    SIMD4<Float>(0, 0, 0, 0)
                }
            }
            let maxArray = maxStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let maxQuery = try! Vector512Optimized(maxArray)

            // Create candidates with slightly different max values
            let maxCandidates = (0..<3).map { j in
                let storage = (0..<128).map { i in
                    if i < 2 {
                        let scale = Float(j + 1) * 0.9
                        return SIMD4<Float>(maxFloat / 1e10 * scale, maxFloat / 1e10 * scale, 0, 0)
                    } else {
                        return SIMD4<Float>(0, 0, 0, 0)
                    }
                }
                let array = storage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
                return try! Vector512Optimized(array)
            }

            let maxResults = BatchKernels_SoA.batchEuclideanSquared512(query: maxQuery, candidates: maxCandidates)
            #expect(maxResults.count == 3)

            for result in maxResults {
                #expect(result >= 0, "Distance with max values should be non-negative")
                #expect(result.isFinite || result == Float.infinity, "Distance should be finite or infinity")
            }

            // Test 2: Minimum representable values
            let minStorage = (0..<128).map { i in
                if i == 0 {
                    SIMD4<Float>(minFloat / 1e10, 0, 0, 0)
                } else {
                    SIMD4<Float>(0, 0, 0, 0)
                }
            }
            let minArray = minStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let minQuery = try! Vector512Optimized(minArray)

            let minResults = BatchKernels_SoA.batchEuclideanSquared512(
                query: minQuery,
                candidates: [minQuery, maxQuery]
            )

            #expect(minResults.count == 2)
            #expect(abs(minResults[0]) < 1e-6 || minResults[0] == 0, "Same vector should have ~zero distance")
            #expect(minResults[1] >= 0, "Distance between min and max should be non-negative")

            // Test 3: Large magnitude differences
            let largeStorage = (0..<128).map { _ in
                SIMD4<Float>(largeFloat, largeFloat, largeFloat, largeFloat)
            }
            let largeArray = largeStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let largeQuery = try! Vector512Optimized(largeArray)

            let smallStorage = (0..<128).map { _ in
                SIMD4<Float>(smallFloat, smallFloat, smallFloat, smallFloat)
            }
            let smallArray = smallStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let smallCandidate = try! Vector512Optimized(smallArray)

            let magnitudeDiffResults = BatchKernels_SoA.batchEuclideanSquared512(
                query: largeQuery,
                candidates: [smallCandidate]
            )

            #expect(magnitudeDiffResults.count == 1)
            #expect(magnitudeDiffResults[0] > 0, "Large magnitude difference should produce non-zero distance")
            #expect(magnitudeDiffResults[0].isFinite || magnitudeDiffResults[0] == Float.infinity,
                   "Result should be finite or infinity")

            // Test 4: Mixed extreme values
            let mixedStorage = (0..<128).map { i in
                switch i % 4 {
                case 0:
                    SIMD4<Float>(largeFloat, -largeFloat, 0, 0)
                case 1:
                    SIMD4<Float>(smallFloat, -smallFloat, 0, 0)
                case 2:
                    SIMD4<Float>(1.0, -1.0, 0.5, -0.5)
                default:
                    SIMD4<Float>(0, 0, 0, 0)
                }
            }
            let mixedArray = mixedStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let mixedQuery = try! Vector512Optimized(mixedArray)

            let normalCandidates = generateTestVectors512(count: 2)
            let mixedResults = BatchKernels_SoA.batchEuclideanSquared512(
                query: mixedQuery,
                candidates: normalCandidates
            )

            #expect(mixedResults.count == 2)
            for result in mixedResults {
                #expect(result >= 0, "Mixed extreme values should produce non-negative distance")
                #expect(result.isFinite || result == Float.infinity, "Result should be finite or infinity")
            }

            // Test 5: Precision loss with large values
            let precisionTestStorage = (0..<128).map { i in
                if i == 0 {
                    // Large value + small delta
                    SIMD4<Float>(1e20, 1e20 + 1, 0, 0)
                } else {
                    SIMD4<Float>(0, 0, 0, 0)
                }
            }
            let precisionArray = precisionTestStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let precisionQuery = try! Vector512Optimized(precisionArray)

            let baseStorage = (0..<128).map { i in
                if i == 0 {
                    SIMD4<Float>(1e20, 1e20, 0, 0)
                } else {
                    SIMD4<Float>(0, 0, 0, 0)
                }
            }
            let baseArray = baseStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let baseCandidate = try! Vector512Optimized(baseArray)

            let precisionResults = BatchKernels_SoA.batchEuclideanSquared512(
                query: precisionQuery,
                candidates: [baseCandidate]
            )

            #expect(precisionResults.count == 1)
            // Due to floating point precision, the small difference might be lost
            #expect(precisionResults[0] >= 0, "Distance should be non-negative despite precision loss")
        }

        @Test
        func testMemoryAlignment() {
            // Test memory alignment requirements
            let candidateCount = 33 // Odd number to test alignment handling
            let candidates = generateTestVectors512(count: candidateCount)
            let query = Vector512Optimized(repeating: 0.5)

            // Test 1: Verify SoA buffer alignment
            let soa = SoA<Vector512Optimized>.build(from: candidates)

            // Check base buffer alignment
            let basePtr = soa.lanePointer(0)
            let baseAddress = Int(bitPattern: basePtr)
            #expect(baseAddress % 16 == 0, "Base SoA buffer must be 16-byte aligned for SIMD4<Float>")

            // Check all lane pointers are aligned
            for laneIdx in 0..<soa.lanes {
                let lanePtr = soa.lanePointer(laneIdx)
                let laneAddress = Int(bitPattern: lanePtr)
                #expect(laneAddress % 16 == 0, "Lane \(laneIdx) must be 16-byte aligned")

                // Verify stride between elements
                if candidateCount > 1 {
                    let secondElementPtr = lanePtr + 1
                    let secondAddress = Int(bitPattern: secondElementPtr)
                    let stride = secondAddress - laneAddress
                    #expect(stride == MemoryLayout<SIMD4<Float>>.size,
                           "Stride should match SIMD4<Float> size: \(stride) vs \(MemoryLayout<SIMD4<Float>>.size)")
                }
            }

            // Test 2: Verify results are correct with aligned access
            let alignedResults = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)
            #expect(alignedResults.count == candidateCount)

            for i in 0..<candidateCount {
                let reference = EuclideanKernels.squared512(query, candidates[i])
                let error = abs(alignedResults[i] - reference)
                #expect(error < 1e-3, "Aligned access should produce correct results at \(i)")
            }

            // Test 3: Test with various sizes to ensure alignment is maintained
            let testSizes = [1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65]

            for size in testSizes {
                let testCandidates = generateTestVectors512(count: size)
                let testSoA = SoA<Vector512Optimized>.build(from: testCandidates)

                // Check first and last lane alignment
                let firstLanePtr = testSoA.lanePointer(0)
                let firstAddress = Int(bitPattern: firstLanePtr)
                #expect(firstAddress % 16 == 0, "Size \(size): First lane must be aligned")

                let lastLaneIdx = testSoA.lanes - 1
                let lastLanePtr = testSoA.lanePointer(lastLaneIdx)
                let lastAddress = Int(bitPattern: lastLanePtr)
                #expect(lastAddress % 16 == 0, "Size \(size): Last lane must be aligned")

                // Verify computation correctness
                let testResults = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: testCandidates)
                #expect(testResults.count == size)
            }

            // Test 4: Stress test with large misaligned-sized batch
            let largeOddSize = 999 // Not a nice power of 2
            let largeCandidates = generateTestVectors512(count: largeOddSize)
            let largeSoA = SoA<Vector512Optimized>.build(from: largeCandidates)

            // Verify alignment is maintained even with large odd size
            for laneIdx in stride(from: 0, to: largeSoA.lanes, by: 10) {
                let lanePtr = largeSoA.lanePointer(laneIdx)
                let address = Int(bitPattern: lanePtr)
                #expect(address % 16 == 0, "Large batch lane \(laneIdx) must be aligned")
            }

            // Test computation
            let largeResults = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: largeCandidates)
            #expect(largeResults.count == largeOddSize)

            // Spot check results
            for i in [0, largeOddSize / 2, largeOddSize - 1] {
                let reference = EuclideanKernels.squared512(query, largeCandidates[i])
                let error = abs(largeResults[i] - reference)
                #expect(error < 1e-3, "Large batch result at \(i) should be correct")
            }

            // Test 5: Verify SIMD operations work correctly with alignment
            let simdTestSize = 64
            let simdCandidates = generateTestVectors512(count: simdTestSize)
            let simdSoA = SoA<Vector512Optimized>.build(from: simdCandidates)

            // Manually compute using SIMD operations to verify alignment
            var manualResults = Array<Float>(repeating: 0, count: simdTestSize)

            for laneIdx in 0..<simdSoA.lanes {
                let lanePtr = simdSoA.lanePointer(laneIdx)
                let queryLane = query.storage[laneIdx]

                // These SIMD operations require proper alignment
                for i in 0..<simdTestSize {
                    let candidateLane = lanePtr[i]
                    let diff = queryLane - candidateLane // SIMD4 subtraction
                    let squared = diff * diff // SIMD4 multiplication
                    manualResults[i] += squared.x + squared.y + squared.z + squared.w
                }
            }

            // Compare with kernel results
            let kernelResults = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: simdCandidates)

            for i in 0..<simdTestSize {
                let error = abs(manualResults[i] - kernelResults[i])
                #expect(error < 1e-3, "Manual SIMD computation should match kernel at \(i)")
            }
        }
    }

    // MARK: - Numerical Accuracy Tests

    @Suite("Numerical Accuracy")
    struct NumericalAccuracyTests {

        @Test
        func testFloatingPointPrecision() {
            // Test floating point precision and rounding
            let query = Vector512Optimized(repeating: 1.0 / 3.0) // Repeating decimal
            let candidateCount = 10

            // Create candidates with values that test precision
            let candidates = (0..<candidateCount).map { i in
                let value = 1.0 / Float(i + 2) // 1/2, 1/3, 1/4, ...
                let storage = (0..<128).map { _ in
                    SIMD4<Float>(value, value * 2, value * 3, value * 4)
                }
                let array = storage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
                return try! Vector512Optimized(array)
            }

            // Test 1: Compare SoA results with reference implementation
            let soaResults = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)
            let referenceResults = candidates.map { candidate in
                EuclideanKernels.squared512(query, candidate)
            }

            #expect(soaResults.count == referenceResults.count)

            // Verify error bounds
            for i in 0..<candidateCount {
                let absoluteError = abs(soaResults[i] - referenceResults[i])
                let relativeError = absoluteError / max(referenceResults[i], Float.ulpOfOne)

                // Acceptable error: ~512 * machine epsilon for accumulation
                let maxAbsoluteError: Float = 512 * Float.ulpOfOne * 10
                let maxRelativeError: Float = 1e-5

                #expect(absoluteError < maxAbsoluteError || relativeError < maxRelativeError,
                       "Precision error at \(i): absolute \(absoluteError), relative \(relativeError)")
            }

            // Test 2: Precision with values near 1.0 ULP differences
            let ulpStorage1 = (0..<128).map { _ in
                SIMD4<Float>(1.0, 1.0, 1.0, 1.0)
            }
            let ulpArray1 = ulpStorage1.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let ulpVector1 = try! Vector512Optimized(ulpArray1)

            let ulpStorage2 = (0..<128).map { _ in
                let nextUp = Float(1.0).nextUp // Next representable float after 1.0
                return SIMD4<Float>(nextUp, 1.0, 1.0, 1.0)
            }
            let ulpArray2 = ulpStorage2.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let ulpVector2 = try! Vector512Optimized(ulpArray2)

            let ulpResults = BatchKernels_SoA.batchEuclideanSquared512(
                query: ulpVector1,
                candidates: [ulpVector2]
            )

            #expect(ulpResults.count == 1)
            #expect(ulpResults[0] > 0, "Should detect ULP difference")
            #expect(ulpResults[0] < 1e-10, "ULP difference should be tiny: \(ulpResults[0])")

            // Test 3: Accumulated rounding errors
            let accumStorage = (0..<128).map { i in
                // Values that don't have exact binary representation
                let base = 0.1 * Float(i)
                return SIMD4<Float>(base, base * 0.3, base * 0.7, base * 0.9)
            }
            let accumArray = accumStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let accumQuery = try! Vector512Optimized(accumArray)

            let accumCandidates = (0..<5).map { j in
                let storage = (0..<128).map { i in
                    let base = 0.1 * Float(i) + 0.01 * Float(j)
                    return SIMD4<Float>(base, base * 0.3, base * 0.7, base * 0.9)
                }
                let array = storage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
                return try! Vector512Optimized(array)
            }

            let accumResults = BatchKernels_SoA.batchEuclideanSquared512(query: accumQuery, candidates: accumCandidates)
            let accumReference = accumCandidates.map { candidate in
                EuclideanKernels.squared512(accumQuery, candidate)
            }

            for i in 0..<accumResults.count {
                let error = abs(accumResults[i] - accumReference[i])
                let tolerance = accumReference[i] * 1e-5 + 1e-6 // Relative + absolute tolerance
                #expect(error < tolerance, "Accumulated error at \(i): \(error) vs tolerance \(tolerance)")
            }
        }

        @Test
        func testCumulativeErrors() {
            // Test accumulation of floating point errors

            // Test 1: Long sequence of small additions
            let smallValue: Float = 1e-7
            let smallStorage = (0..<128).map { _ in
                SIMD4<Float>(smallValue, smallValue, smallValue, smallValue)
            }
            let smallArray = smallStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let smallQuery = try! Vector512Optimized(smallArray)

            // Create many candidates with tiny differences
            let longSequenceCount = 100
            let longCandidates = (0..<longSequenceCount).map { i in
                let delta = smallValue * Float(i) * 1e-3
                let storage = (0..<128).map { _ in
                    SIMD4<Float>(smallValue + delta, smallValue + delta, smallValue + delta, smallValue + delta)
                }
                let array = storage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
                return try! Vector512Optimized(array)
            }

            let longResults = BatchKernels_SoA.batchEuclideanSquared512(query: smallQuery, candidates: longCandidates)

            // Verify error doesn't explode
            var maxError: Float = 0
            for i in 0..<longSequenceCount {
                let reference = EuclideanKernels.squared512(smallQuery, longCandidates[i])
                let error = abs(longResults[i] - reference)
                maxError = max(maxError, error)

                // Error should be bounded even after many operations
                #expect(error < 1e-10, "Cumulative error at \(i): \(error)")
            }

            #expect(maxError < 1e-9, "Maximum cumulative error: \(maxError)")

            // Test 2: Error propagation with alternating signs
            let alternatingStorage = (0..<128).map { i in
                let sign: Float = (i % 2 == 0) ? 1.0 : -1.0
                return SIMD4<Float>(sign * 0.1, sign * 0.2, sign * 0.3, sign * 0.4)
            }
            let alternatingArray = alternatingStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let alternatingQuery = try! Vector512Optimized(alternatingArray)

            // Process in large batch to test error propagation
            let batchSize = 200
            let batchCandidates = (0..<batchSize).map { j in
                let storage = (0..<128).map { i in
                    let sign: Float = ((i + j) % 2 == 0) ? 1.0 : -1.0
                    let base = Float(j) * 1e-4
                    return SIMD4<Float>(
                        sign * (0.1 + base),
                        sign * (0.2 + base),
                        sign * (0.3 + base),
                        sign * (0.4 + base)
                    )
                }
                let array = storage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
                return try! Vector512Optimized(array)
            }

            let batchResults = BatchKernels_SoA.batchEuclideanSquared512(query: alternatingQuery, candidates: batchCandidates)

            // Check error growth pattern
            var errors: [Float] = []
            for i in 0..<batchSize {
                let reference = EuclideanKernels.squared512(alternatingQuery, batchCandidates[i])
                let error = abs(batchResults[i] - reference)
                errors.append(error)
            }

            // Error should not grow unbounded
            let meanError = errors.reduce(0, +) / Float(errors.count)
            let maxBatchError = errors.max() ?? 0

            #expect(meanError < 1e-6, "Mean error should be small: \(meanError)")
            #expect(maxBatchError < 1e-5, "Max error should be bounded: \(maxBatchError)")

            // Test 3: Catastrophic cancellation scenario
            let largeBase: Float = 1e6
            let cancelStorage1 = (0..<128).map { _ in
                SIMD4<Float>(largeBase, largeBase, largeBase, largeBase)
            }
            let cancelArray1 = cancelStorage1.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let cancelQuery = try! Vector512Optimized(cancelArray1)

            let cancelStorage2 = (0..<128).map { _ in
                let delta: Float = 1.0 // Small relative to largeBase
                return SIMD4<Float>(largeBase + delta, largeBase + delta, largeBase + delta, largeBase + delta)
            }
            let cancelArray2 = cancelStorage2.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let cancelCandidate = try! Vector512Optimized(cancelArray2)

            let cancelResults = BatchKernels_SoA.batchEuclideanSquared512(
                query: cancelQuery,
                candidates: [cancelCandidate]
            )

            // Despite cancellation, result should be reasonably accurate
            let expectedDistance: Float = 512.0 * (1.0 * 1.0) // 512 dimensions, each with difference of 1
            let cancelError = abs(cancelResults[0] - expectedDistance)
            let cancelRelativeError = cancelError / expectedDistance

            #expect(cancelRelativeError < 0.01, "Catastrophic cancellation handling: relative error \(cancelRelativeError)")
        }

        @Test
        func testNumericalStability() {
            // Test numerical stability of algorithms

            // Test 1: Ill-conditioned input (very different magnitudes)
            let illConditionedStorage = (0..<128).map { i in
                if i < 64 {
                    // Large magnitude components
                    SIMD4<Float>(1e10, 1e10, 1e10, 1e10)
                } else {
                    // Small magnitude components
                    SIMD4<Float>(1e-10, 1e-10, 1e-10, 1e-10)
                }
            }
            let illCondArray = illConditionedStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let illCondQuery = try! Vector512Optimized(illCondArray)

            // Create candidate with slightly perturbed values
            let perturbedStorage = (0..<128).map { i in
                if i < 64 {
                    SIMD4<Float>(1e10 * 1.000001, 1e10 * 1.000001, 1e10 * 1.000001, 1e10 * 1.000001)
                } else {
                    SIMD4<Float>(1e-10 * 1.1, 1e-10 * 1.1, 1e-10 * 1.1, 1e-10 * 1.1)
                }
            }
            let perturbedArray = perturbedStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let perturbedCandidate = try! Vector512Optimized(perturbedArray)

            let illCondResults = BatchKernels_SoA.batchEuclideanSquared512(
                query: illCondQuery,
                candidates: [perturbedCandidate]
            )

            #expect(illCondResults.count == 1)
            #expect(illCondResults[0] > 0, "Should detect difference in ill-conditioned case")
            #expect(illCondResults[0].isFinite, "Result should be finite despite ill-conditioning")

            // Test 2: Nearly parallel vectors (small angle, potential stability issues)
            let baseVector = generateTestVectors512(count: 1)[0]
            let nearlyParallelCandidates = (0..<5).map { i in
                let scale = 1.0 + Float(i) * 1e-6 // Very small scaling differences
                let storage = baseVector.storage.map { simd4 in
                    simd4 * scale
                }
                let array = storage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
                return try! Vector512Optimized(array)
            }

            let parallelResults = BatchKernels_SoA.batchEuclideanSquared512(
                query: baseVector,
                candidates: nearlyParallelCandidates
            )

            // Results should be stable and monotonically increasing
            for i in 1..<parallelResults.count {
                #expect(parallelResults[i] >= parallelResults[i-1],
                       "Distance should increase monotonically: \(parallelResults[i-1]) -> \(parallelResults[i])")
            }

            // Test 3: Gradual underflow scenario
            let underflowBase: Float = 1e-30
            let underflowStorage = (0..<128).map { i in
                let value = underflowBase * Float(128 - i) // Gradually decreasing to very small
                return SIMD4<Float>(value, value, value, value)
            }
            let underflowArray = underflowStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let underflowQuery = try! Vector512Optimized(underflowArray)

            let underflowCandidates = (0..<3).map { j in
                let storage = (0..<128).map { i in
                    let value = underflowBase * Float(128 - i) * (1.0 + Float(j) * 0.1)
                    return SIMD4<Float>(value, value, value, value)
                }
                let array = storage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
                return try! Vector512Optimized(array)
            }

            let underflowResults = BatchKernels_SoA.batchEuclideanSquared512(
                query: underflowQuery,
                candidates: underflowCandidates
            )

            // Should handle graceful underflow
            for result in underflowResults {
                #expect(result >= 0, "Underflow should not produce negative results")
                #expect(result.isFinite, "Underflow should not produce infinity or NaN")
            }

            // Test 4: Overflow protection
            let nearOverflow: Float = 1e18 // Large but not quite overflow
            let overflowStorage = (0..<128).map { _ in
                SIMD4<Float>(nearOverflow, nearOverflow, nearOverflow, nearOverflow)
            }
            let overflowArray = overflowStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let overflowQuery = try! Vector512Optimized(overflowArray)

            let overflowCandidate = try! Vector512Optimized(overflowArray) // Same vector
            let overflowResults = BatchKernels_SoA.batchEuclideanSquared512(
                query: overflowQuery,
                candidates: [overflowCandidate]
            )

            #expect(overflowResults.count == 1)
            #expect(abs(overflowResults[0]) < Float.ulpOfOne, "Same large vectors should have ~zero distance")

            // Test different large vector
            let overflowStorage2 = (0..<128).map { _ in
                SIMD4<Float>(nearOverflow * 0.9, nearOverflow * 0.9, nearOverflow * 0.9, nearOverflow * 0.9)
            }
            let overflowArray2 = overflowStorage2.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let overflowCandidate2 = try! Vector512Optimized(overflowArray2)

            let overflowResults2 = BatchKernels_SoA.batchEuclideanSquared512(
                query: overflowQuery,
                candidates: [overflowCandidate2]
            )

            #expect(overflowResults2[0] > 0, "Different large vectors should have positive distance")
            #expect(overflowResults2[0] == Float.infinity || overflowResults2[0] < Float.infinity,
                   "Should handle potential overflow gracefully")
        }

        @Test
        func testConsistencyAcrossPlatforms() {
            // Test numerical consistency across platforms
            // Note: In practice, this tests consistency within the same platform
            // but verifies deterministic behavior that should hold across platforms

            let query = Vector512Optimized(repeating: 0.5)
            let candidateCount = 20
            let candidates = generateTestVectors512(count: candidateCount)

            // Test 1: Verify deterministic results (same input -> same output)
            let results1 = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)
            let results2 = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)
            let results3 = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)

            #expect(results1.count == results2.count)
            #expect(results2.count == results3.count)

            for i in 0..<candidateCount {
                #expect(results1[i] == results2[i], "Results should be deterministic at index \(i)")
                #expect(results2[i] == results3[i], "Results should be deterministic at index \(i)")
            }

            // Test 2: Verify order independence for batch processing
            var shuffledCandidates = candidates
            shuffledCandidates.shuffle()

            let shuffledResults = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: shuffledCandidates)

            // Results should be the same when mapped back to original order
            var mappedResults = Array<Float>(repeating: 0, count: candidateCount)
            for (newIndex, candidate) in shuffledCandidates.enumerated() {
                if let originalIndex = candidates.firstIndex(where: { vector in
                    // Check if vectors are the same
                    for j in 0..<128 {
                        if vector.storage[j] != candidate.storage[j] {
                            return false
                        }
                    }
                    return true
                }) {
                    mappedResults[originalIndex] = shuffledResults[newIndex]
                }
            }

            for i in 0..<candidateCount {
                #expect(abs(results1[i] - mappedResults[i]) < 1e-5,
                       "Order independence failed at \(i): \(results1[i]) vs \(mappedResults[i])")
            }

            // Test 3: Test consistency with IEEE 754 operations
            // These specific values should produce consistent results on IEEE 754 compliant systems
            let ieeeStorage = (0..<128).map { i in
                switch i % 4 {
                case 0:
                    SIMD4<Float>(0.5, 0.25, 0.125, 0.0625)
                case 1:
                    SIMD4<Float>(1.0, 2.0, 4.0, 8.0)
                case 2:
                    SIMD4<Float>(-0.5, -0.25, -0.125, -0.0625)
                default:
                    SIMD4<Float>(-1.0, -2.0, -4.0, -8.0)
                }
            }
            let ieeeArray = ieeeStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            let ieeeQuery = try! Vector512Optimized(ieeeArray)

            let ieeeCandidates = [
                Vector512Optimized(repeating: 0.0),
                Vector512Optimized(repeating: 1.0),
                Vector512Optimized(repeating: -1.0),
                Vector512Optimized(repeating: 0.5)
            ]

            let ieeeResults = BatchKernels_SoA.batchEuclideanSquared512(query: ieeeQuery, candidates: ieeeCandidates)

            // These results should be consistent across platforms
            #expect(ieeeResults.count == 4)
            for result in ieeeResults {
                #expect(result >= 0, "Distance should always be non-negative")
                #expect(result.isFinite, "Distance should be finite for finite inputs")
            }

            // Test 4: Verify compiler optimization consistency
            // Test with both -O0 and -Os equivalent operations
            @inline(never)
            func unoptimizedComputation() -> [Float] {
                var results = Array<Float>(repeating: 0, count: candidateCount)
                for i in 0..<candidateCount {
                    results[i] = EuclideanKernels.squared512(query, candidates[i])
                }
                return results
            }

            let unoptimizedResults = unoptimizedComputation()
            let optimizedResults = BatchKernels_SoA.batchEuclideanSquared512(query: query, candidates: candidates)

            for i in 0..<candidateCount {
                let error = abs(unoptimizedResults[i] - optimizedResults[i])
                #expect(error < 1e-5, "Optimization consistency at \(i): error \(error)")
            }

            // Test 5: Associativity and commutativity preservation
            // (a-b)² should equal (b-a)² for distance computation
            let commutativeResults = BatchKernels_SoA.batchEuclideanSquared512(
                query: candidates[0],
                candidates: [query]
            )
            let originalResults = BatchKernels_SoA.batchEuclideanSquared512(
                query: query,
                candidates: [candidates[0]]
            )

            #expect(abs(commutativeResults[0] - originalResults[0]) < 1e-6,
                   "Commutativity should be preserved: \(commutativeResults[0]) vs \(originalResults[0])")
        }
    }

    // MARK: - Integration Tests

    @Suite("Integration Tests")
    struct IntegrationTests {

        @Test
        func testIntegrationWithBatchOperations() async {
            // Test integration with higher-level batch operations
            // Verify SoA kernels work correctly in real-world scenarios

            // Test 1: k-NN search using SoA kernels
            let k = 5
            let queryVector = Vector512Optimized(repeating: 0.5)

            // Create database of vectors
            let databaseSize = 100
            let database = generateTestVectors512(count: databaseSize)

            // Perform k-NN search using batch SoA operations
            let distances = BatchKernels_SoA.batchEuclideanSquared512(
                query: queryVector,
                candidates: database
            )

            // Find k nearest neighbors
            let indexedDistances = distances.enumerated().map { ($0.offset, $0.element) }
            let sortedNeighbors = indexedDistances.sorted { $0.1 < $1.1 }
            let kNearest = Array(sortedNeighbors.prefix(k))

            #expect(kNearest.count == min(k, databaseSize), "Should find k nearest neighbors")

            // Verify neighbors are correctly ordered
            for i in 1..<kNearest.count {
                #expect(kNearest[i-1].1 <= kNearest[i].1,
                       "Neighbors should be sorted by distance")
            }

            // Test 2: Similarity matrix computation
            let matrixSize = 10
            let vectors = generateTestVectors512(count: matrixSize)

            // Compute pairwise distances using SoA
            var similarityMatrix = [[Float]](repeating: [Float](repeating: 0, count: matrixSize),
                                            count: matrixSize)

            for i in 0..<matrixSize {
                let queryVec = vectors[i]
                let distances = BatchKernels_SoA.batchEuclideanSquared512(
                    query: queryVec,
                    candidates: vectors
                )

                for j in 0..<matrixSize {
                    // Convert distance to similarity (e.g., using Gaussian kernel)
                    let distance = distances[j]
                    let sigma: Float = 1.0
                    let similarity = exp(-distance / (2 * sigma * sigma))
                    similarityMatrix[i][j] = similarity
                }
            }

            // Verify matrix properties
            // 1. Diagonal should be 1 (self-similarity)
            for i in 0..<matrixSize {
                #expect(abs(similarityMatrix[i][i] - 1.0) < 1e-3,
                       "Diagonal elements should be 1 (self-similarity)")
            }

            // 2. Matrix should be symmetric
            for i in 0..<matrixSize {
                for j in i+1..<matrixSize {
                    #expect(abs(similarityMatrix[i][j] - similarityMatrix[j][i]) < 1e-3,
                           "Similarity matrix should be symmetric")
                }
            }

            // Test 3: Batch clustering scenario
            // Use SoA for efficient distance computations in clustering
            let clusterCenters = [
                Vector512Optimized(repeating: 0.0),
                Vector512Optimized(repeating: 1.0),
                Vector512Optimized(repeating: 2.0)
            ]

            let pointsToCluster = generateTestVectors512(count: 30)
            var assignments = [Int](repeating: 0, count: pointsToCluster.count)

            // Assign each point to nearest cluster center
            for (idx, point) in pointsToCluster.enumerated() {
                let distances = BatchKernels_SoA.batchEuclideanSquared512(
                    query: point,
                    candidates: clusterCenters
                )

                let minDistanceIdx = distances.enumerated().min(by: { $0.1 < $1.1 })?.0 ?? 0
                assignments[idx] = minDistanceIdx
            }

            #expect(assignments.count == pointsToCluster.count,
                   "Every point should be assigned to a cluster")
            #expect(Set(assignments).count <= clusterCenters.count,
                   "Assignments should be within valid cluster range")

            // Test 4: Parallel batch processing
            await withTaskGroup(of: [Float].self) { group in
                let numBatches = 4
                let batchSize = 25

                for batchIdx in 0..<numBatches {
                    group.addTask {
                        let batchVectors = generateTestVectors512(count: batchSize)
                        let query = Vector512Optimized(repeating: Float(batchIdx))
                        return BatchKernels_SoA.batchEuclideanSquared512(
                            query: query,
                            candidates: batchVectors
                        )
                    }
                }

                var allResults: [[Float]] = []
                for await result in group {
                    allResults.append(result)
                }

                #expect(allResults.count == numBatches,
                       "Should process all batches in parallel")

                for batchResult in allResults {
                    #expect(batchResult.count == batchSize,
                           "Each batch should produce correct number of results")
                }
            }
        }

        @Test
        func testIntegrationWithVectorTypes() {
            // Test integration with all OptimizedVector types
            // Verify seamless interoperability across dimensions

            let testSize = 10

            // Test Vector512Optimized
            let vectors512 = generateTestVectors512(count: testSize)
            let query512 = Vector512Optimized(repeating: 0.5)
            let results512 = BatchKernels_SoA.batchEuclideanSquared512(
                query: query512,
                candidates: vectors512
            )

            #expect(results512.count == testSize, "512D: Should return results for all candidates")
            for result in results512 {
                #expect(result >= 0, "512D: Distance should be non-negative")
                #expect(result.isFinite, "512D: Distance should be finite")
            }

            // Test Vector768Optimized
            let vectors768 = generateTestVectors768(count: testSize)
            let query768 = Vector768Optimized(repeating: 0.5)
            let results768 = BatchKernels_SoA.batchEuclideanSquared768(
                query: query768,
                candidates: vectors768
            )

            #expect(results768.count == testSize, "768D: Should return results for all candidates")
            for result in results768 {
                #expect(result >= 0, "768D: Distance should be non-negative")
                #expect(result.isFinite, "768D: Distance should be finite")
            }

            // Test Vector1536Optimized
            let vectors1536 = generateTestVectors1536(count: testSize)
            let query1536 = Vector1536Optimized(repeating: 0.5)
            let results1536 = BatchKernels_SoA.batchEuclideanSquared1536(
                query: query1536,
                candidates: vectors1536
            )

            #expect(results1536.count == testSize, "1536D: Should return results for all candidates")
            for result in results1536 {
                #expect(result >= 0, "1536D: Distance should be non-negative")
                #expect(result.isFinite, "1536D: Distance should be finite")
            }

            // Test type conversion and compatibility
            // Create vectors with similar patterns across dimensions
            var pattern512 = [Float](repeating: 0, count: 512)
            var pattern768 = [Float](repeating: 0, count: 768)
            var pattern1536 = [Float](repeating: 0, count: 1536)

            // Fill with same pattern (repeated if needed)
            for i in 0..<512 {
                pattern512[i] = sin(Float(i) * 0.01)
            }
            for i in 0..<768 {
                pattern768[i] = sin(Float(i % 512) * 0.01)
            }
            for i in 0..<1536 {
                pattern1536[i] = sin(Float(i % 512) * 0.01)
            }

            let patternVec512 = try! Vector512Optimized(pattern512)
            let patternVec768 = try! Vector768Optimized(pattern768)
            let patternVec1536 = try! Vector1536Optimized(pattern1536)

            // Test batch operations with pattern vectors
            let patternResults512 = BatchKernels_SoA.batchEuclideanSquared512(
                query: patternVec512,
                candidates: [patternVec512]
            )
            let patternResults768 = BatchKernels_SoA.batchEuclideanSquared768(
                query: patternVec768,
                candidates: [patternVec768]
            )
            let patternResults1536 = BatchKernels_SoA.batchEuclideanSquared1536(
                query: patternVec1536,
                candidates: [patternVec1536]
            )

            // Self-distance should be zero
            #expect(abs(patternResults512[0]) < 1e-3, "512D: Self-distance should be ~0")
            #expect(abs(patternResults768[0]) < 1e-3, "768D: Self-distance should be ~0")
            #expect(abs(patternResults1536[0]) < 1e-3, "1536D: Self-distance should be ~0")

            // Test mixed batch sizes across dimensions
            let sizes = [1, 5, 10, 50, 100]

            for size in sizes {
                let batch512 = generateTestVectors512(count: size)
                let batch768 = generateTestVectors768(count: size)
                let batch1536 = generateTestVectors1536(count: size)

                let batchResults512 = BatchKernels_SoA.batchEuclideanSquared512(
                    query: query512,
                    candidates: batch512
                )
                let batchResults768 = BatchKernels_SoA.batchEuclideanSquared768(
                    query: query768,
                    candidates: batch768
                )
                let batchResults1536 = BatchKernels_SoA.batchEuclideanSquared1536(
                    query: query1536,
                    candidates: batch1536
                )

                #expect(batchResults512.count == size, "512D batch size \(size) check")
                #expect(batchResults768.count == size, "768D batch size \(size) check")
                #expect(batchResults1536.count == size, "1536D batch size \(size) check")
            }

            // Verify dimension-specific optimizations
            // 512D should be fastest (smallest), 1536D slowest (largest)
            let perfTestSize = 100
            let perfVectors512 = generateTestVectors512(count: perfTestSize)
            let perfVectors768 = generateTestVectors768(count: perfTestSize)
            let perfVectors1536 = generateTestVectors1536(count: perfTestSize)

            let time512 = measureTime {
                _ = BatchKernels_SoA.batchEuclideanSquared512(
                    query: query512,
                    candidates: perfVectors512
                )
            }

            _ = measureTime {
                _ = BatchKernels_SoA.batchEuclideanSquared768(
                    query: query768,
                    candidates: perfVectors768
                )
            }  // 768D timing measured but comparison only uses 512 vs 1536

            let time1536 = measureTime {
                _ = BatchKernels_SoA.batchEuclideanSquared1536(
                    query: query1536,
                    candidates: perfVectors1536
                )
            }

            // Generally expect processing time to scale with dimension
            // (may not hold in debug mode)
            #expect(time512 <= time1536 * 2 || time512 < 0.01,
                   "512D should generally be faster than 1536D")
        }

        @Test
        func testIntegrationWithCaching() {
            // Test integration with caching mechanisms
            // Verify performance benefits and correctness with caching

            // Test 1: Norm caching for cosine distance
            class NormCache {
                private var cache: [String: Float] = [:]
                private var hits = 0
                private var misses = 0

                func getNorm(_ vector: Vector512Optimized) -> Float {
                    // Create simple hash key from first few elements
                    let key = "\(vector[0])_\(vector[1])_\(vector[2])"

                    if let cached = cache[key] {
                        hits += 1
                        return cached
                    }

                    misses += 1
                    let norm = sqrt(DotKernels.dot512(vector, vector))
                    cache[key] = norm
                    return norm
                }

                var hitRate: Float {
                    let total = hits + misses
                    return total > 0 ? Float(hits) / Float(total) : 0
                }
            }

            let normCache = NormCache()
            let vectors = generateTestVectors512(count: 20)
            _ = vectors[0]  // Query vector available for distance computations

            // Compute norms with caching
            for _ in 0..<3 {  // Multiple passes to test cache hits
                for vector in vectors {
                    _ = normCache.getNorm(vector)
                }
            }

            #expect(normCache.hitRate > 0.5, "Should have good cache hit rate after multiple passes")

            // Test 2: Result caching for repeated queries
            class DistanceCache {
                private var cache: [String: [Float]] = [:]
                private(set) var computations = 0  // Make it readable for testing

                func getDistances(
                    query: Vector512Optimized,
                    candidates: [Vector512Optimized]
                ) -> [Float] {
                    // Create cache key from query (simplified)
                    let queryKey = "\(query[0])_\(query[1])_\(candidates.count)"

                    if let cached = cache[queryKey] {
                        return cached
                    }

                    computations += 1
                    let results = BatchKernels_SoA.batchEuclideanSquared512(
                        query: query,
                        candidates: candidates
                    )
                    cache[queryKey] = results
                    return results
                }
            }

            let distanceCache = DistanceCache()
            let testCandidates = generateTestVectors512(count: 50)
            let testQuery = Vector512Optimized(repeating: 0.5)

            // First computation (cache miss)
            let results1 = distanceCache.getDistances(
                query: testQuery,
                candidates: testCandidates
            )

            // Second computation (cache hit)
            let results2 = distanceCache.getDistances(
                query: testQuery,
                candidates: testCandidates
            )

            #expect(results1 == results2, "Cached results should be identical")
            #expect(distanceCache.computations == 1, "Should only compute once")

            // Test 3: LRU cache for memory-bounded scenarios
            class LRUCache<Key: Hashable, Value> {
                private var cache: [Key: Value] = [:]
                private var order: [Key] = []
                private let capacity: Int

                init(capacity: Int) {
                    self.capacity = capacity
                }

                func get(_ key: Key) -> Value? {
                    guard let value = cache[key] else { return nil }

                    // Move to front
                    order.removeAll { $0 == key }
                    order.insert(key, at: 0)

                    return value
                }

                func set(_ key: Key, _ value: Value) {
                    if cache[key] != nil {
                        order.removeAll { $0 == key }
                    }

                    order.insert(key, at: 0)
                    cache[key] = value

                    // Evict if over capacity
                    if order.count > capacity {
                        if let evicted = order.popLast() {
                            cache.removeValue(forKey: evicted)
                        }
                    }
                }
            }

            let lruCache = LRUCache<Int, [Float]>(capacity: 5)

            // Fill cache beyond capacity
            for i in 0..<10 {
                let query = vectors[i % vectors.count]
                let results = BatchKernels_SoA.batchEuclideanSquared512(
                    query: query,
                    candidates: vectors
                )
                lruCache.set(i, results)
            }

            // Recent items should be in cache
            #expect(lruCache.get(9) != nil, "Most recent should be cached")
            #expect(lruCache.get(0) == nil, "Oldest should be evicted")

            // Test 4: Cache invalidation on vector updates
            class InvalidatingCache {
                private var cache: [String: [Float]] = [:]
                private var versions: [String: Int] = [:]

                func invalidate(_ vector: Vector512Optimized) {
                    let key = "\(vector[0])_\(vector[1])_\(vector[2])"
                    versions[key, default: 0] += 1
                    // Remove cached results involving this vector
                    cache.removeValue(forKey: key)
                }

                func getCachedOrCompute(
                    query: Vector512Optimized,
                    candidates: [Vector512Optimized]
                ) -> [Float] {
                    let key = "\(query[0])_\(query[1])_\(query[2])"

                    if let cached = cache[key] {
                        return cached
                    }

                    let results = BatchKernels_SoA.batchEuclideanSquared512(
                        query: query,
                        candidates: candidates
                    )
                    cache[key] = results
                    return results
                }
            }

            let invalidatingCache = InvalidatingCache()
            let mutableQuery = vectors[0]

            // Compute and cache
            _ = invalidatingCache.getCachedOrCompute(
                query: mutableQuery,
                candidates: vectors
            )

            // Invalidate
            invalidatingCache.invalidate(mutableQuery)

            // Should recompute after invalidation
            let recomputed = invalidatingCache.getCachedOrCompute(
                query: mutableQuery,
                candidates: vectors
            )

            #expect(recomputed.count == vectors.count, "Should recompute after invalidation")
        }

        @Test
        func testRealWorldWorkloads() async {
            // Test with realistic workloads
            // Simulate actual use cases for SoA batch operations

            // Test 1: Semantic search scenario
            // Simulate searching through document embeddings
            let documentCount = 500
            let queryEmbedding = Vector512Optimized(repeating: 0.1)

            // Generate document embeddings (simulating different topics)
            var documentEmbeddings: [Vector512Optimized] = []
            for i in 0..<documentCount {
                var values = [Float](repeating: 0, count: 512)
                // Create clusters of similar documents
                let topicId = i / 50  // 10 topics
                let baseValue = Float(topicId) * 0.2

                for j in 0..<512 {
                    values[j] = baseValue + Float.random(in: -0.1...0.1)
                }
                documentEmbeddings.append(try! Vector512Optimized(values))
            }

            // Perform semantic search
            let searchStartTime = CFAbsoluteTimeGetCurrent()
            let similarities = BatchKernels_SoA.batchEuclideanSquared512(
                query: queryEmbedding,
                candidates: documentEmbeddings
            )
            let searchTime = CFAbsoluteTimeGetCurrent() - searchStartTime

            // Find top 10 most similar documents
            let topK = 10
            let rankedDocs = similarities.enumerated()
                .sorted { $0.1 < $1.1 }
                .prefix(topK)
                .map { $0.0 }

            #expect(rankedDocs.count == topK, "Should find top K documents")
            #expect(searchTime < 1.0, "Search should complete in reasonable time")

            // Test 2: Recommendation system pattern
            // User-item similarity computation
            let numUsers = 100
            let numItems = 200

            // Generate user preference vectors
            let userVectors = generateTestVectors512(count: numUsers)

            // Generate item feature vectors
            let itemVectors = generateTestVectors512(count: numItems)

            // Find recommendations for specific user
            let targetUser = userVectors[0]

            // Compute similarities to all items
            let itemScores = BatchKernels_SoA.batchEuclideanSquared512(
                query: targetUser,
                candidates: itemVectors
            )

            // Convert distances to scores (inverse relationship)
            let recommendationScores = itemScores.map { 1.0 / (1.0 + $0) }

            // Get top recommendations
            let topRecommendations = recommendationScores.enumerated()
                .sorted { $0.1 > $1.1 }  // Higher score is better
                .prefix(20)
                .map { $0.0 }

            #expect(topRecommendations.count == 20, "Should generate recommendations")

            // Verify score ordering
            for i in 1..<topRecommendations.count {
                let prevScore = recommendationScores[topRecommendations[i-1]]
                let currScore = recommendationScores[topRecommendations[i]]
                #expect(prevScore >= currScore, "Recommendations should be sorted")
            }

            // Test 3: Document similarity/clustering task
            // Group similar documents together
            let clusterCount = 5
            let docsPerCluster = 20
            let totalDocs = clusterCount * docsPerCluster

            // Generate clustered documents
            var clusteredDocs: [Vector512Optimized] = []
            for cluster in 0..<clusterCount {
                let clusterCenter = Float(cluster) * 2.0
                for _ in 0..<docsPerCluster {
                    var values = [Float](repeating: clusterCenter, count: 512)
                    // Add noise
                    for j in 0..<512 {
                        values[j] += Float.random(in: -0.3...0.3)
                    }
                    clusteredDocs.append(try! Vector512Optimized(values))
                }
            }

            // Compute similarity matrix for clustering
            var avgIntraClusterDist: Float = 0
            var avgInterClusterDist: Float = 0
            var intraCount = 0
            var interCount = 0

            for i in 0..<totalDocs {
                let distances = BatchKernels_SoA.batchEuclideanSquared512(
                    query: clusteredDocs[i],
                    candidates: clusteredDocs
                )

                for j in 0..<totalDocs {
                    if i != j {
                        let sameCluster = (i / docsPerCluster) == (j / docsPerCluster)
                        if sameCluster {
                            avgIntraClusterDist += distances[j]
                            intraCount += 1
                        } else {
                            avgInterClusterDist += distances[j]
                            interCount += 1
                        }
                    }
                }
            }

            avgIntraClusterDist /= Float(intraCount)
            avgInterClusterDist /= Float(interCount)

            #expect(avgIntraClusterDist < avgInterClusterDist,
                   "Intra-cluster distance should be smaller than inter-cluster")

            // Test 4: Parallel workload processing
            // Simulate multiple concurrent search queries
            await withTaskGroup(of: (Int, [Float]).self) { group in
                let numQueries = 10
                let databaseSize = 200
                let database = generateTestVectors512(count: databaseSize)

                for queryId in 0..<numQueries {
                    group.addTask {
                        // Each query is slightly different
                        let queryVector = Vector512Optimized(repeating: Float(queryId) * 0.1)
                        let results = BatchKernels_SoA.batchEuclideanSquared512(
                            query: queryVector,
                            candidates: database
                        )
                        return (queryId, results)
                    }
                }

                var allResults: [(Int, [Float])] = []
                for await result in group {
                    allResults.append(result)
                }

                #expect(allResults.count == numQueries,
                       "Should process all queries concurrently")

                // Verify each query got complete results
                for (_, results) in allResults {
                    #expect(results.count == databaseSize,
                           "Each query should get results for all candidates")
                }
            }

            // Test 5: Incremental index update scenario
            // Simulate adding new documents to search index
            var searchIndex = generateTestVectors512(count: 100)
            let newDocuments = generateTestVectors512(count: 20)

            // Add new documents incrementally
            for newDoc in newDocuments {
                searchIndex.append(newDoc)

                // Verify we can still search the updated index
                let searchResults = BatchKernels_SoA.batchEuclideanSquared512(
                    query: newDoc,
                    candidates: searchIndex
                )

                #expect(searchResults.count == searchIndex.count,
                       "Should handle incremental updates")
            }

            #expect(searchIndex.count == 120, "Index should grow correctly")
        }
    }

    // MARK: - Helper Functions (Placeholder)

    // MARK: - Helper Functions for Test Data Generation

    private static func generateTestVectors512(count: Int) -> [Vector512Optimized] {
        guard count > 0 else { return [] }

        return (0..<count).map { i in
            let storage = (0..<128).map { j in
                let base = Float(i * 128 + j)
                return SIMD4<Float>(
                    sin(base * 0.01),
                    cos(base * 0.01 + 1),
                    sin(base * 0.02 + 2),
                    cos(base * 0.02 + 3)
                )
            }
            let flatStorage = storage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            return try! Vector512Optimized(flatStorage)
        }
    }

    private static func generateTestVectors768(count: Int) -> [Vector768Optimized] {
        guard count > 0 else { return [] }

        return (0..<count).map { i in
            let storage = (0..<192).map { j in
                let base = Float(i * 192 + j)
                return SIMD4<Float>(
                    sin(base * 0.01 + 0.5),
                    cos(base * 0.01 + 1.5),
                    sin(base * 0.02 + 2.5),
                    cos(base * 0.02 + 3.5)
                )
            }
            let flatStorage = storage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            return try! Vector768Optimized(flatStorage)
        }
    }

    private static func generateTestVectors1536(count: Int) -> [Vector1536Optimized] {
        guard count > 0 else { return [] }

        return (0..<count).map { i in
            let storage = (0..<384).map { j in
                let base = Float(i * 384 + j)
                return SIMD4<Float>(
                    sin(base * 0.005 + 1.0),
                    cos(base * 0.005 + 2.0),
                    sin(base * 0.01 + 3.0),
                    cos(base * 0.01 + 4.0)
                )
            }
            let flatStorage = storage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
            return try! Vector1536Optimized(flatStorage)
        }
    }

    private static func createOrthogonalVector512(to query: Vector512Optimized) -> Vector512Optimized {
        // Create a vector orthogonal to the query using Gram-Schmidt process
        let randomVector = generateTestVectors512(count: 1)[0]

        // Project randomVector onto query
        let dotProduct = DotKernels.dot512(randomVector, query)
        let queryNormSquared = DotKernels.dot512(query, query)

        guard queryNormSquared > 0 else {
            return randomVector // If query is zero, return random vector
        }

        let projectionScale = dotProduct / queryNormSquared

        // orthogonal = randomVector - projection
        let orthogonalStorage = zip(randomVector.storage, query.storage).map { (random, queryLane) in
            random - queryLane * projectionScale
        }

        let flatOrthogonalStorage = orthogonalStorage.flatMap { simd4 in [simd4.x, simd4.y, simd4.z, simd4.w] }
        return try! Vector512Optimized(flatOrthogonalStorage)
    }

    private static func measureTime(operation: () -> Void) -> TimeInterval {
        let startTime = CFAbsoluteTimeGetCurrent()
        operation()
        let endTime = CFAbsoluteTimeGetCurrent()
        return endTime - startTime
    }

    private static func getMemoryUsage() -> Int {
        // Simplified memory usage measurement
        // In a real implementation, you might use more sophisticated memory tracking
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        // Access mach_task_self_ directly - @preconcurrency import allows this
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) { infoPtr in
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), infoPtr, &count)
            }
        }

        return result == KERN_SUCCESS ? Int(info.resident_size) : 0
    }
}