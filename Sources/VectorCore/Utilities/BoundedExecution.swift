import Foundation

/// Internal structured worker scheduling. Public contracts are tested by
/// ExecutionSchedulingContractTests; this does not bound user-created tasks or threads.
internal enum BoundedExecution {
    @TaskLocal static var isInsideRegion = false

    /// Task-scoped observation is nil in ordinary execution; recorders synchronize their state.
    struct Event: Sendable {
        enum Kind: Sendable, Equatable { case begin, submitted, completed, end; case logicalRange(Range<Int>) }
        let id: UUID
        let phase: String
        let kind: Kind
        let worker: Int
        let limit: Int
    }
    @TaskLocal static var observation: (@Sendable (Event) -> Void)?

    static var hardwareLimit: Int { max(1, ProcessInfo.processInfo.activeProcessorCount) }

    /// Ordered parent callbacks share the region and cancellation policy; exercised by
    /// ExecutionSchedulingContractTests.reductionCombineCancellationAndRegionAcrossProvidersAndOverloads.
    static func combinePartials<R: Sendable>(
        _ parts: [R], initial: R, _ combine: @Sendable (R, R) -> R
    ) throws -> R {
        try $isInsideRegion.withValue(true) {
            var result = initial
            for part in parts {
                try Task.checkCancellation()
                result = combine(result, part)
                try Task.checkCancellation()
            }
            try Task.checkCancellation()
            return result
        }
    }

    static func validate(limit: Int) throws {
        guard limit > 0 else { throw VectorError(.invalidConfiguration, message: "maxConcurrency/maxThreadCount must be positive") }
    }

    /// The range length must already be representable and the hint positive.
    private static func chunk(count: Int, limit: Int, minimum: Int) -> Int {
        max(minimum, ComputeRangeBoundary.ceilingDivide(count, by: min(count, limit)))
    }

    /// Invokes one body per contiguous partition; at most limit worker tasks are submitted.
    /// The serial/nested path invokes the body once with the entire nonempty range.
    static func ranges<T: Sendable>(
        _ items: Range<Int>, limit: Int, parallel: Bool = true, minimum: Int = 1,
        phase: String, body: @Sendable @escaping (Range<Int>) async throws -> T
    ) async throws -> [T] {
        let count = try ComputeRangeBoundary.count(items)
        try validate(limit: limit)
        try ComputeRangeBoundary.validate(minChunk: minimum)
        try Task.checkCancellation()
        let observer = observation
        let id = observer == nil ? nil : UUID()
        func emit(_ kind: Event.Kind, _ worker: Int = -1) {
            if let id { observer?(Event(id: id, phase: phase, kind: kind, worker: worker, limit: limit)) }
        }
        emit(.begin); defer { emit(.end) }
        guard count > 0 else { return [] }
        let spawn = parallel && !isInsideRegion && !CPUComputeProvider.isInsideParallelRegion
        return try await $isInsideRegion.withValue(true) {
            if !spawn {
                try Task.checkCancellation()
                let value = try await body(items)
                try Task.checkCancellation()
                return [value]
            }
            let width = chunk(count: count, limit: limit, minimum: minimum)
            return try await withThrowingTaskGroup(of: (Int, T).self) { group in
                do {
                    var start = items.lowerBound
                    var index = 0
                    while start < items.upperBound {
                        try Task.checkCancellation()
                        let end = start + min(width, items.upperBound - start)
                        let range = start..<end, worker = index
                        emit(.submitted, worker)
                        group.addTask {
                            defer {
                                if let id { observer?(Event(id: id, phase: phase, kind: .completed, worker: worker, limit: limit)) }
                            }
                            try Task.checkCancellation()
                            let result = try await body(range)
                            try Task.checkCancellation()
                            return (worker, result)
                        }
                        start = end; index += 1
                    }
                    var partials: [(Int, T)] = []
                    for try await part in group { partials.append(part) }
                    try Task.checkCancellation()
                    return partials.sorted { $0.0 < $1.0 }.map { $0.1 }
                } catch {
                    group.cancelAll()
                    throw error
                }
            }
        }
    }

    static func map<T: Sendable>(
        _ items: Range<Int>, limit: Int, parallel: Bool = true, minimum: Int = 1,
        phase: String, body: @Sendable @escaping (Int) async throws -> T
    ) async throws -> [T] {
        let parts = try await ranges(items, limit: limit, parallel: parallel, minimum: minimum, phase: phase) { range in
            var results: [T] = []
            results.reserveCapacity(range.count)
            for index in range {
                try Task.checkCancellation()
                let value = try await body(index)
                try Task.checkCancellation()
                results.append(value)
            }
            return results
        }
        let result = parts.flatMap { $0 }
        try Task.checkCancellation()
        return result
    }

    static func forEach(
        _ items: Range<Int>, limit: Int, parallel: Bool = true, minimum: Int = 1,
        phase: String, body: @Sendable @escaping (Int) async throws -> Void
    ) async throws {
        let _: [Bool] = try await ranges(items, limit: limit, parallel: parallel, minimum: minimum, phase: phase) { range in
            for index in range {
                try Task.checkCancellation()
                try await body(index)
                try Task.checkCancellation()
            }
            return true
        }
        try Task.checkCancellation()
    }

    /// Nonthrowing routes deliberately complete all work, even when cancelled.
    static func completeMap<T: Sendable>(
        _ items: Range<Int>, limit: Int, parallel: Bool = true, minimum: Int = 1,
        phase: String, body: @Sendable @escaping (Int) async -> T
    ) async -> [T] {
        // All callers provide array indices, a positive hardware limit and positive hints.
        precondition(limit > 0 && minimum > 0)
        let observer = observation
        let id = observer == nil ? nil : UUID()
        func emit(_ kind: Event.Kind, _ worker: Int = -1) {
            if let id { observer?(Event(id: id, phase: phase, kind: kind, worker: worker, limit: limit)) }
        }
        emit(.begin); defer { emit(.end) }
        guard !items.isEmpty else { return [] }
        let spawn = parallel && !isInsideRegion && !CPUComputeProvider.isInsideParallelRegion
        return await $isInsideRegion.withValue(true) {
            if !spawn {
                var values: [T] = []
                values.reserveCapacity(items.count)
                for index in items { values.append(await body(index)) }
                return values
            }
            let width = chunk(count: items.count, limit: limit, minimum: minimum)
            return await withTaskGroup(of: (Int, [T]).self) { group in
                var start = items.lowerBound, index = 0
                while start < items.upperBound {
                    let end = start + min(width, items.upperBound - start)
                    let range = start..<end, worker = index
                    emit(.submitted, worker)
                    group.addTask {
                        defer {
                            if let id { observer?(Event(id: id, phase: phase, kind: .completed, worker: worker, limit: limit)) }
                        }
                        var values: [T] = []
                        values.reserveCapacity(range.count)
                        for index in range { values.append(await body(index)) }
                        return (worker, values)
                    }
                    start = end; index += 1
                }
                var parts: [(Int, [T])] = []
                for await part in group { parts.append(part) }
                return parts.sorted { $0.0 < $1.0 }.flatMap { $0.1 }
            }
        }
    }
}
