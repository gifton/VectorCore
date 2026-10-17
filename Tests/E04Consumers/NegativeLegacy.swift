import VectorCore
func rejected() async {
    let _ = TypedBufferPool<Float>()
    let pool = TypedBufferPool<Float>(maxCachedBytes: 64)
    let buffer: UnsafeMutableBufferPointer<Float> = await pool.acquire(count: 1)
    await pool.release(buffer)
}
