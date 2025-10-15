// Runtime CPU feature queries (scaffold implementations)
#include "VectorCoreC.h"

#if defined(__x86_64__) || defined(_M_X64)
  #if defined(__GNUC__) || defined(__clang__)
    #define VC_BUILTIN_CPU_SUPPORTS 1
  #endif
#endif

int vc_has_avx2(void) {
#if VC_BUILTIN_CPU_SUPPORTS
    return __builtin_cpu_supports("avx2");
#else
    return 0;
#endif
}

int vc_has_avx512f(void) {
#if VC_BUILTIN_CPU_SUPPORTS
    return __builtin_cpu_supports("avx512f");
#else
    return 0;
#endif
}

int vc_has_neon(void) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return 1;
#else
    return 0;
#endif
}

int vc_has_dotprod(void) {
#if defined(__ARM_FEATURE_DOTPROD)
    return 1;
#else
    return 0;
#endif
}

