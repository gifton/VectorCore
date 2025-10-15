// FeatureFlags.swift — compile-time toggles for optional features
// Default is conservative: C kernels disabled unless VC_USE_C_KERNELS is defined.

enum FeatureFlags {
    #if VC_USE_C_KERNELS
    static let useCKernels = true
    #else
    static let useCKernels = false
    #endif
}

