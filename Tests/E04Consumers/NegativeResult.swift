import VectorCore
final class NonSendableResult { var value = 0 }
func rejected(_ provider: any BufferProvider, _ typed: TypedBufferPool<Float>) async throws {
    let _: NonSendableResult = try await provider.withBuffer(size: 1) { _ in NonSendableResult() }
    let _: NonSendableResult = await typed.withBuffer(count: 1) { _ in NonSendableResult() }
}
