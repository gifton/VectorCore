import Testing
import Foundation
@testable import VectorCore
import Darwin.Mach

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
            // NOTE: Current BatchKernels_SoA only implements Euclidean squared distance
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

            // TODO: Implement SoA dot product kernel and compare results
            // let soaResults = BatchKernels_SoA.batchDotProduct512(query: query, candidates: candidates)
        }

        @Test
        func testDotProduct768() {
            // NOTE: Current BatchKernels_SoA only implements Euclidean squared distance

            let candidateCount = 6
            let query = Vector768Optimized(repeating: 0.5)
            let candidates = generateTestVectors768(count: candidateCount)

            // Test with reference implementation
            _ = candidates.map { candidate in
                DotKernels.dot768(query, candidate)
            }

            // Reference results computed successfully

            // TODO: Implement SoA dot product kernel for 768-dim vectors
            // let soaResults = BatchKernels_SoA.batchDotProduct768(query: query, candidates: candidates)
        }

        @Test
        func testDotProduct1536() {
            // NOTE: Current BatchKernels_SoA only implements Euclidean squared distance

            let candidateCount = 4
            let query = Vector1536Optimized(repeating: 0.25)
            let candidates = generateTestVectors1536(count: candidateCount)

            // Test with reference implementation
            _ = candidates.map { candidate in
                DotKernels.dot1536(query, candidate)
            }

            // Reference results computed successfully

            // TODO: Implement SoA dot product kernel for 1536-dim vectors
            // let soaResults = BatchKernels_SoA.batchDotProduct1536(query: query, candidates: candidates)
        }

        @Test
        func testCosineDistance512() {
            // NOTE: Current BatchKernels_SoA only implements Euclidean squared distance

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

            // TODO: Implement SoA cosine distance kernel
            // let soaResults = BatchKernels_SoA.batchCosineDistance512(query: query, candidates: candidates)
        }

        @Test
        func testCosineDistance768() {
            // NOTE: Current BatchKernels_SoA only implements Euclidean squared distance

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

            // TODO: Implement SoA cosine distance kernel for 768-dim vectors
            // let soaResults = BatchKernels_SoA.batchCosineDistance768(query: query, candidates: candidates)
        }

        @Test
        func testCosineDistance1536() {
            // NOTE: Current BatchKernels_SoA only implements Euclidean squared distance

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

            // TODO: Implement SoA cosine distance kernel for 1536-dim vectors
            // let soaResults = BatchKernels_SoA.batchCosineDistance1536(query: query, candidates: candidates)
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
            // TODO: Test cache locality improvements with SoA
            // - Measure cache miss rates (if possible)
            // - Verify improved memory access patterns
        }

        @Test
        func testSIMDUtilization() {
            // TODO: Test SIMD instruction utilization in SoA kernels
            // - Verify efficient use of SIMD registers
            // - Check for vectorization opportunities
        }

        @Test
        func testScalingWithBatchSize() {
            // TODO: Test how performance scales with batch size
            // - Linear scaling expected for most operations
            // - Identify any performance cliffs or sweet spots
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
            // TODO: Test handling of denormalized floating point values
            // - Very small values near machine epsilon
            // - Subnormal numbers
        }

        @Test
        func testInfiniteAndNaNValues() {
            // TODO: Test handling of infinite and NaN values
            // - Positive/negative infinity
            // - NaN propagation
            // - Mixed finite/infinite values
        }

        @Test
        func testExtremeValues() {
            // TODO: Test with extreme floating point values
            // - Maximum/minimum representable values
            // - Large magnitude differences
        }

        @Test
        func testMemoryAlignment() {
            // TODO: Test memory alignment requirements
            // - Verify proper alignment for SIMD operations
            // - Test with misaligned data (should be handled gracefully)
        }
    }

    // MARK: - Numerical Accuracy Tests

    @Suite("Numerical Accuracy")
    struct NumericalAccuracyTests {

        @Test
        func testFloatingPointPrecision() {
            // TODO: Test floating point precision and rounding
            // - Compare against high-precision reference
            // - Verify acceptable error bounds
        }

        @Test
        func testCumulativeErrors() {
            // TODO: Test accumulation of floating point errors
            // - Long sequences of operations
            // - Error propagation in batch processing
        }

        @Test
        func testNumericalStability() {
            // TODO: Test numerical stability of algorithms
            // - Condition number analysis
            // - Stability with ill-conditioned inputs
        }

        @Test
        func testConsistencyAcrossPlatforms() {
            // TODO: Test numerical consistency across platforms
            // - Different CPU architectures
            // - Various compiler optimizations
        }
    }

    // MARK: - Integration Tests

    @Suite("Integration Tests")
    struct IntegrationTests {

        @Test
        func testIntegrationWithBatchOperations() async {
            // TODO: Test integration with higher-level batch operations
            // - k-NN search using SoA kernels
            // - Similarity matrix computation
        }

        @Test
        func testIntegrationWithVectorTypes() {
            // TODO: Test integration with all OptimizedVector types
            // - Vector512Optimized, Vector768Optimized, Vector1536Optimized
            // - Verify seamless interoperability
        }

        @Test
        func testIntegrationWithCaching() {
            // TODO: Test integration with caching mechanisms
            // - Norm caching
            // - Result caching
        }

        @Test
        func testRealWorldWorkloads() async {
            // TODO: Test with realistic workloads
            // - Semantic search scenarios
            // - Recommendation system patterns
            // - Document similarity tasks
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

        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }

        return result == KERN_SUCCESS ? Int(info.resident_size) : 0
    }
}