Numerical Rigor & Edge Cases
Issue 4.1: Int16 Arithmetic Overflow in Quantized Euclidean
Target File: Sources/VectorCore/Operations/Kernels/QuantizedKernels.swift
Target Method: accumulateEuclidDiffSq (Around line 167)
The Defect: The kernel calculates Int32(diff.x &* diff.x) where diff.x is an Int16. The maximum difference between two Int8 components is 255. Squaring it yields 65025. Because Int16.max is 32767, this operation overflows the 16-bit multiplication before it is cast to Int32, wrapping to -511 in two's complement. This mathematically corrupts distance calculations.
Agent Action:
Cast the Int16 components to Int32 before performing the multiplication.
Correct Logic:
Swift
let dx = Int32(diff.x)
let dy = Int32(diff.y)
// ...
acc &+= (dx &* dx) &+ (dy &* dy) // ...
Issue 4.2: Infinity Overflow in Cosine Similarity Denominator
Target File: Sources/VectorCore/Operations/Kernels/CosineKernels.swift
Target Method: calculateCosineDistance(dot:sumAA:sumBB:)
The Defect: The code evaluates let denomSq = sumAA * sumBB followed by let denom = sqrt(denomSq). If sumAA and sumBB belong to large unnormalized vectors (e.g., 1.0e20), their product becomes 1.0e40. This exceeds the upper bound of an IEEE-754 32-bit Float (~3.4e38), overflowing to +Infinity. The subsequent division results in 0.0.
Agent Action:
Compute the square roots independently before multiplication to keep intermediate values safely inside the FP32 domain.
Correct Logic: let denom = sqrt(sumAA) * sqrt(sumBB)
Issue 4.3: Memory Leak on Asynchronous Pool Return
Target File: Sources/VectorCore/Utilities/MemoryPool.swift
Target Method: returnBuffer (Around line 224)
The Defect: The method casts a raw pointer to an Int and dispatches it to an async queue with a [weak self] capture. If the pool has been deallocated, guard let self = self else { return } silently returns. Because the pointer was allocated with posix_memalign (which ignores ARC), the memory is permanently leaked into the OS.
Agent Action:
In the else block of the guard statement, explicitly free the memory.
Correct Logic:
Swift
let rawPointerToFree = UnsafeMutableRawPointer(bitPattern: pointerAddress)!
guard let self = self else { 
    AlignedMemory.deallocate(rawPointerToFree)
    return 
}
Issue 4.4: Subnormal Reciprocal Overflow (NaN Poisoning)
Target File: Sources/VectorCore/Operations/Kernels/NormalizeKernels.swift
Target Methods: magnitude, magnitudeSquared, normalizeUnchecked
The Defect: Kahan's stabilization algorithm calculates an inverse scale: let scale = 1.0 / maxAbs. If the vector consists entirely of subnormal (denormalized) floats near zero (e.g., 1e-40), the division 1.0 / 1e-40 overflows and becomes +Infinity. Multiplying the array elements by +Infinity poisons the entire vector with NaNs.
Agent Action:
Clamp maxAbs to a safe minimum threshold before calculating the reciprocal.
Add Check: let safeMaxAbs = Swift.max(maxAbs, Float.leastNormalMagnitude) before calculating the scale.
Issue 4.5: Flawed Epsilon Truncation Rejecting Valid Micro-Vectors
Target Files: Sources/VectorCore/Operations/DistanceMetrics.swift (CosineDistance), Sources/VectorCore/Operations/Kernels/CosineKernels.swift
Target Area: Zero-vector detection guards (e.g., guard magnitudeProduct > Float.ulpOfOne else { return 1.0 })
The Defect: Float.ulpOfOne (approx. 1.19e-7) denotes precision steps at 1.0, not the absolute limit of float representation near zero. If two valid dense micro-vectors have magnitudes of 1e-4, their magnitudeProduct is 1e-8. Because 1e-8 < 1.19e-7, valid vectors are wrongly rejected and returned with maximum distance.
Agent Action:
Change all absolute magnitude product checks from Float.ulpOfOne (and any hardcoded 1e-9 checks) to Float.leastNormalMagnitude (or a strict absolute domain epsilon like 1e-15) to properly distinguish mathematical zero spaces from valid micro-vector spaces.
