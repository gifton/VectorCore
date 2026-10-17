import VectorCore

#if E03_NEGATIVE_CAPTURE
final class MutableState {
    var threshold = 7
}

func rejectedCapture(_ state: MutableState) async throws {
    try await BatchOperations.updateConfiguration { configuration in
        configuration.parallelThreshold = state.threshold
    }
}
#else
struct ConsumerValue: Sendable {
    var count = 0
}

enum ConsumerFailure: Error { case expected }

func nonisolatedConsumer(_ amount: Int) async throws {
    let snapshot = amount
    let configuration = ThreadSafeConfiguration(ConsumerValue())
    await configuration.modify { $0.count += snapshot }
    try await configuration.modify { value in
        value.count += snapshot
        if snapshot < 0 { throw ConsumerFailure.expected }
    }
    await configuration.update(ConsumerValue(count: snapshot))
    await configuration.update(\.count, to: snapshot)
    _ = await configuration.get()
    try await BatchOperations.updateConfiguration { $0.minimumChunkSize = snapshot }
    try await BatchOperations.updateConfiguration { value in
        value.parallelThreshold = snapshot
        if snapshot < 0 { throw ConsumerFailure.expected }
    }
}

actor ConsumerActor {
    private var amount = 3
    func apply(_ configuration: ThreadSafeConfiguration<ConsumerValue>) async throws {
        let snapshot = amount
        await configuration.modify { $0.count += snapshot }
        try await configuration.modify { value in
            value.count += snapshot
            if snapshot < 0 { throw ConsumerFailure.expected }
        }
        try await BatchOperations.updateConfiguration { $0.defaultBatchSize = snapshot }
        try await BatchOperations.updateConfiguration { value in
            value.matrixRoutingMinN = snapshot
            if snapshot < 0 { throw ConsumerFailure.expected }
        }
    }
}

@MainActor
final class ConsumerMainActor {
    private var amount = 4
    func apply(_ configuration: ThreadSafeConfiguration<ConsumerValue>) async throws {
        let snapshot = amount
        await configuration.modify { $0.count += snapshot }
        try await configuration.modify { value in
            value.count += snapshot
            if snapshot < 0 { throw ConsumerFailure.expected }
        }
        try await BatchOperations.updateConfiguration { $0.oversubscription = Double(snapshot) }
        try await BatchOperations.updateConfiguration { value in
            value.enableMatrixRouting = snapshot > 0
            if snapshot < 0 { throw ConsumerFailure.expected }
        }
    }
}
#endif
