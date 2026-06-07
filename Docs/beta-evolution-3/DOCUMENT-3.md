Metal Compute Pipeline & Bridging Integrity
Issue 3.1: GPU Zero-Copy Page Alignment Violation
Target Files: Sources/VectorCore/Storage/AlignedMemory.swift, Sources/VectorCore/Platform/PlatformConfiguration.swift
Target Variable: optimalAlignment
The Defect: The framework advertises .gpu routing but hardcodes CPU memory alignment to 64 bytes (a standard CPU cache line). For Apple Silicon's Unified Memory Architecture (UMA) to achieve zero-copy GPU execution (via MTLDevice.makeBuffer(bytesNoCopy:)), CPU allocations must be strictly aligned to the OS memory page size. If they aren't, the GPU driver silently initiates a blocking memcpy across the PCIe bus, destroying batch performance.
Agent Action:
Modify PlatformConfiguration.optimalAlignment and AlignedMemory.optimalAlignment.
Use a dynamic query Int(getpagesize()) for Apple platforms (or hardcode 16384 for Apple Silicon arch(arm64)), which satisfies the bytesNoCopy alignment restrictions.
Add documentation noting that this page-alignment is strictly required for zero-copy MTLBuffer bridging.
