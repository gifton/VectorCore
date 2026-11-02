// D02 RD-R8/RD-R2 (negative): each line uses a public member removed by D02's routed fixes: the four flag helpers
// of OptimizationAttributes.swift (Q14; never true, never run or always enforced in package builds),
// `SoA.memoryEfficiency` and `BatchKernels_SoA.estimatePerformanceGain` (fixed text, no callers). After D02 each line
// is one compile error. Before D02 the file compiles.
import VectorCore

func simdEnabled() -> Bool { isSIMDEnabled }
func releaseMode() -> Bool { isReleaseMode }
func releaseOnlyBlock() { releaseOnly { } }
func enforced() { performancePrecondition(true) }
func memoryEfficiency(_ soa: SoA512) -> String { soa.memoryEfficiency }
func performanceGain() -> String { BatchKernels_SoA.estimatePerformanceGain(candidateCount: 1, dimension: 512) }
