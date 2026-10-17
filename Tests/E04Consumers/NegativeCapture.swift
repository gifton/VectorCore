import VectorCore
final class NonSendableCapture { var value = 0 }
func rejected(_ provider: any BufferProvider, _ typed: TypedBufferPool<Float>) async throws {
    let value = NonSendableCapture()
    let _: Int = try await provider.withBuffer(size: 1) { _ in value.value }
    let _: Int = await typed.withBuffer(count: 1) { _ in value.value }
}
