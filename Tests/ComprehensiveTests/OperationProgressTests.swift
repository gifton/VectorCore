import Testing
import Foundation
@testable import VectorCore

@Suite("OperationProgress")
struct OperationProgressSuite {

    // MARK: - Initialization Tests

    @Test
    func testOperationProgress_BasicInit() {
        let progress = OperationProgress(
            current: 5,
            total: 10,
            phase: "Loading"
        )

        #expect(progress.current == 5)
        #expect(progress.total == 10)
        #expect(progress.phase == "Loading")
        #expect(progress.message == nil)
        #expect(progress.estimatedTimeRemaining == nil)
    }

    @Test
    func testOperationProgress_FullInit() {
        let timestamp = Date()
        let progress = OperationProgress(
            current: 3,
            total: 7,
            phase: "Processing",
            message: "Processing items",
            estimatedTimeRemaining: 10.0,
            timestamp: timestamp
        )

        #expect(progress.current == 3)
        #expect(progress.total == 7)
        #expect(progress.phase == "Processing")
        #expect(progress.message == "Processing items")
        #expect(progress.estimatedTimeRemaining == 10.0)
        #expect(progress.timestamp == timestamp)
    }

    @Test
    func testOperationProgress_DefaultTimestamp() {
        let before = Date()
        let progress = OperationProgress(current: 1, total: 10)
        let after = Date()

        #expect(progress.timestamp >= before)
        #expect(progress.timestamp <= after)
    }

    @Test
    func testOperationProgress_NegativeValuesClamped() {
        let progress = OperationProgress(current: -5, total: -10)

        #expect(progress.current == 0)
        #expect(progress.total == 0)
    }

    // MARK: - Computed Properties

    @Test
    func testOperationProgress_Fraction() {
        let half = OperationProgress(current: 5, total: 10)
        #expect(half.fraction == 0.5)

        let quarter = OperationProgress(current: 25, total: 100)
        #expect(quarter.fraction == 0.25)

        let complete = OperationProgress(current: 10, total: 10)
        #expect(complete.fraction == 1.0)

        let none = OperationProgress(current: 0, total: 10)
        #expect(none.fraction == 0.0)
    }

    @Test
    func testOperationProgress_FractionZeroTotal() {
        let progress = OperationProgress(current: 5, total: 0)
        #expect(progress.fraction == 0.0)
    }

    @Test
    func testOperationProgress_Percentage() {
        let p50 = OperationProgress(current: 50, total: 100)
        #expect(p50.percentage == 50)

        let p75 = OperationProgress(current: 75, total: 100)
        #expect(p75.percentage == 75)

        let p0 = OperationProgress(current: 0, total: 100)
        #expect(p0.percentage == 0)

        let p100 = OperationProgress(current: 100, total: 100)
        #expect(p100.percentage == 100)
    }

    @Test
    func testOperationProgress_IsComplete() {
        let incomplete = OperationProgress(current: 5, total: 10)
        #expect(!incomplete.isComplete)

        let complete = OperationProgress(current: 10, total: 10)
        #expect(complete.isComplete)

        let overComplete = OperationProgress(current: 15, total: 10)
        #expect(overComplete.isComplete)
    }

    // MARK: - Factory Methods

    @Test
    func testOperationProgress_Started() {
        let started = OperationProgress.started(total: 100)

        #expect(started.current == 0)
        #expect(started.total == 100)
        #expect(started.phase == "Starting")
        #expect(!started.isComplete)
        #expect(started.fraction == 0.0)
    }

    @Test
    func testOperationProgress_StartedCustomPhase() {
        let started = OperationProgress.started(total: 50, phase: "Initializing")

        #expect(started.current == 0)
        #expect(started.total == 50)
        #expect(started.phase == "Initializing")
    }

    @Test
    func testOperationProgress_Completed() {
        let completed = OperationProgress.completed(total: 100)

        #expect(completed.current == 100)
        #expect(completed.total == 100)
        #expect(completed.phase == "Complete")
        #expect(completed.isComplete)
        #expect(completed.fraction == 1.0)
        #expect(completed.percentage == 100)
    }

    @Test
    func testOperationProgress_CompletedCustomPhase() {
        let completed = OperationProgress.completed(total: 75, phase: "Finished")

        #expect(completed.current == 75)
        #expect(completed.total == 75)
        #expect(completed.phase == "Finished")
    }

    @Test
    func testOperationProgress_Indeterminate() {
        let indeterminate = OperationProgress.indeterminate(current: 42, phase: "Processing")

        #expect(indeterminate.current == 42)
        #expect(indeterminate.total == 0)
        #expect(indeterminate.phase == "Processing")
        #expect(indeterminate.fraction == 0.0)
    }

    // MARK: - Equatable Tests

    @Test
    func testOperationProgress_Equatable() {
        let timestamp = Date()

        let p1 = OperationProgress(
            current: 5,
            total: 10,
            phase: "Test",
            message: "msg",
            estimatedTimeRemaining: 5.0,
            timestamp: timestamp
        )

        let p2 = OperationProgress(
            current: 5,
            total: 10,
            phase: "Test",
            message: "msg",
            estimatedTimeRemaining: 5.0,
            timestamp: timestamp
        )

        #expect(p1 == p2)
    }

    @Test
    func testOperationProgress_NotEquatable() {
        let timestamp = Date()
        let base = OperationProgress(current: 5, total: 10, timestamp: timestamp)

        let differentCurrent = OperationProgress(current: 6, total: 10, timestamp: timestamp)
        #expect(base != differentCurrent)

        let differentTotal = OperationProgress(current: 5, total: 11, timestamp: timestamp)
        #expect(base != differentTotal)

        let differentPhase = OperationProgress(current: 5, total: 10, phase: "Other", timestamp: timestamp)
        #expect(base != differentPhase)
    }

    // MARK: - Sendable Tests

    @Test
    func testOperationProgress_Sendable() async {
        let progress = OperationProgress(current: 5, total: 10)

        // Should compile without warnings
        let result = await Task {
            progress.percentage
        }.value

        #expect(result == 50)
    }

    // MARK: - Real-World Scenarios

    @Test
    func testOperationProgress_BatchProcessing() {
        let batchSize = 10
        var progresses: [OperationProgress] = []

        for i in 0...batchSize {
            let progress = OperationProgress(
                current: i,
                total: batchSize,
                phase: i == batchSize ? "Complete" : "Processing"
            )
            progresses.append(progress)
        }

        #expect(progresses.first?.fraction == 0.0)
        #expect(progresses.last?.fraction == 1.0)
        #expect(progresses.last?.isComplete == true)
    }

    @Test
    func testOperationProgress_WithEstimatedTime() {
        let progress = OperationProgress(
            current: 25,
            total: 100,
            phase: "Downloading",
            estimatedTimeRemaining: 30.0
        )

        #expect(progress.percentage == 25)
        #expect(progress.estimatedTimeRemaining == 30.0)
    }
}

// MARK: - ProgressStream Tests

@Suite("ProgressStream")
struct ProgressStreamSuite {

    @Test
    func testProgressStream_BasicIteration() async throws {
        let (stream, continuation) = AsyncThrowingStream.makeStream(of: (String, OperationProgress).self)
        let progressStream = ProgressStream(stream)

        // Send some progress updates
        continuation.yield(("item1", OperationProgress(current: 1, total: 3)))
        continuation.yield(("item2", OperationProgress(current: 2, total: 3)))
        continuation.yield(("item3", OperationProgress(current: 3, total: 3)))
        continuation.finish()

        var results: [(String, OperationProgress)] = []
        for try await update in progressStream {
            results.append(update)
        }

        #expect(results.count == 3)
        #expect(results[0].0 == "item1")
        #expect(results[1].0 == "item2")
        #expect(results[2].0 == "item3")

        #expect(results[0].1.current == 1)
        #expect(results[1].1.current == 2)
        #expect(results[2].1.current == 3)
    }

    @Test
    func testProgressStream_EmptyStream() async throws {
        let (stream, continuation) = AsyncThrowingStream.makeStream(of: (Int, OperationProgress).self)
        let progressStream = ProgressStream(stream)

        continuation.finish()

        var count = 0
        for try await _ in progressStream {
            count += 1
        }

        #expect(count == 0)
    }

    @Test
    func testProgressStream_ErrorPropagation() async throws {
        let (stream, continuation) = AsyncThrowingStream.makeStream(of: (String, OperationProgress).self)
        let progressStream = ProgressStream(stream)

        continuation.yield(("item1", OperationProgress(current: 1, total: 3)))
        continuation.finish(throwing: VectorError(.operationFailed, message: "Test error"))

        do {
            var count = 0
            for try await _ in progressStream {
                count += 1
            }
            #expect(count == 1)  // Should get the first item before error
            Issue.record("Expected error to be thrown")
        } catch let error as VectorError {
            #expect(error.kind == .operationFailed)
        } catch {
            Issue.record("Unexpected error type: \(error)")
        }
    }

    @Test
    func testProgressStream_IntElements() async throws {
        let (stream, continuation) = AsyncThrowingStream.makeStream(of: (Int, OperationProgress).self)
        let progressStream = ProgressStream(stream)

        for i in 0..<5 {
            continuation.yield((i, OperationProgress(current: i, total: 4)))
        }
        continuation.finish()

        var sum = 0
        for try await (value, progress) in progressStream {
            sum += value
            #expect(progress.current == value)
        }

        #expect(sum == 10)  // 0 + 1 + 2 + 3 + 4
    }

    @Test
    func testProgressStream_ProgressTracking() async throws {
        let (stream, continuation) = AsyncThrowingStream.makeStream(of: (String, OperationProgress).self)
        let progressStream = ProgressStream(stream)

        let items = ["a", "b", "c", "d", "e"]
        for (index, item) in items.enumerated() {
            continuation.yield((item, OperationProgress(current: index + 1, total: items.count)))
        }
        continuation.finish()

        var progressValues: [Int] = []
        for try await (_, progress) in progressStream {
            progressValues.append(progress.percentage)
        }

        #expect(progressValues == [20, 40, 60, 80, 100])
    }

    @Test
    func testProgressStream_CancellationSupport() async throws {
        let (stream, continuation) = AsyncThrowingStream.makeStream(of: (Int, OperationProgress).self)
        let progressStream = ProgressStream(stream)

        let task = Task {
            var count = 0
            for try await _ in progressStream {
                count += 1
                if count >= 2 {
                    break
                }
            }
            return count
        }

        continuation.yield((1, OperationProgress(current: 1, total: 5)))
        continuation.yield((2, OperationProgress(current: 2, total: 5)))
        continuation.yield((3, OperationProgress(current: 3, total: 5)))
        continuation.finish()

        let count = try await task.value
        #expect(count == 2)
    }
}
