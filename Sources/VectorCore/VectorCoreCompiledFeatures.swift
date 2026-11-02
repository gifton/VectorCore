/// Immutable diagnostics for two compile conditions, not full binary provenance.
public enum VectorCoreCompiledFeatures {
    #if VC_USE_C_KERNELS
    public static let useCKernels = true
    #else
    public static let useCKernels = false
    #endif
    #if VC_SWIFTPM_DEBUG
    public static let configuration = "debug"
    #elseif VC_SWIFTPM_RELEASE
    public static let configuration = "release"
    #else
    public static let configuration = "unknown"
    #endif
}
