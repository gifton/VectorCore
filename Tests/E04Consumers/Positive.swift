import VectorCore

func nonisolatedConsumer(_ provider: any BufferProvider, _ typed: TypedBufferPool<String>) async throws {
    // Exact BufferProvider documentation example.
    let address = try await provider.withBuffer(size: 1024) { handle in
        UInt(bitPattern: handle.pointer)
    }
    precondition(address != 0)
    let count = await typed.withBuffer(count: 2) { buffer in
        buffer.baseAddress!.initialize(repeating: "value", count: buffer.count)
        defer { buffer.baseAddress!.deinitialize(count: buffer.count) }
        return buffer.count
    }
    precondition(count == 2)
    let handle = await typed.acquire(count: 1)
    handle.pointer.initialize(to: "explicit lease")
    precondition(handle.pointer.pointee == "explicit lease")
    handle.pointer.deinitialize(count: 1)
    await typed.release(handle)
}

actor ConsumerActor {
    func run(_ provider: any BufferProvider, _ typed: TypedBufferPool<String>) async throws {
        try await nonisolatedConsumer(provider, typed)
        let count = await typed.withBuffer(count: 0) { $0.count }
        precondition(count == 0)
    }
}

@MainActor func mainActorConsumer(_ provider: any BufferProvider, _ typed: TypedBufferPool<String>) async throws {
    let count = try await provider.withBuffer(size: 1) { $0.size }
    precondition(count == 1)
    let typedCount = await typed.withBuffer(count: 0) { $0.count }
    precondition(typedCount == 0)
}

@main struct Consumer {
    static func main() async throws {
        let provider = SwiftBufferPool(configuration: BufferConfiguration(maxPoolSize: 64))
        let typed = TypedBufferPool<String>(maxCachedBytes: 128, maxBuffersPerSize: 2)
        try await nonisolatedConsumer(provider, typed)
        try await ConsumerActor().run(provider, typed)
        try await mainActorConsumer(provider, typed)
        await provider.clear(); await typed.clear()
        print("E04_PUBLIC_SCOPED_CONSUMERS_COMPLETE")
    }
}
