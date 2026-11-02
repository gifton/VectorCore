import VectorCore
// The metric-free property was removed. This control compiles if a property of
// either shape (Float or Float?) exists, so it fails only when none does.
func oldProperty(_ result: SearchResult<Int>) {
    let _: Float? = result.computedScore
}
