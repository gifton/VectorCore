import Foundation
import VectorCore
@_silgen_name("q02_start") func start(_ threshold: UInt)
@_silgen_name("q02_stop") func stop()
@_silgen_name("q02_read") func read(_ field: Int32) -> UInt64
@_silgen_name("q02_calibrate_filter") func calibrateFilter()
@_silgen_name("q02_calibrate_c") func calibrateC()
@inline(never) func arrayControl(_ count: Int) -> [Float] { Array(repeating: 3, count: count) }

@main struct Q02SelectionProbe {
    static func select(_ width: Int, _ count: Int) async throws -> MixedPrecisionStrategy {
        #if Q02_BASELINE
        return await MixedPrecisionAutoTuner.shared.selectOptimalStrategy(candidateCount: count, accuracyRequirement: 0.001, dimension: width)
        #else
        return try await MixedPrecisionAutoTuner.shared.selectStrategy(candidateCount: count, maximumSyntheticRelativeError: 0.001, dimension: width)
        #endif
    }
    static func seconds(_ duration: Duration) -> Double {
        Double(duration.components.seconds) + Double(duration.components.attoseconds) * 1e-18
    }
    static func main() async throws {
        let mode = CommandLine.arguments[1]
        print("Q02_BODY_BEGIN mode=\(mode)")
        if mode == "semantics" {
            var failures = 0
            func check(_ condition: Bool, _ name: String) {
                print("CHECK \(name) passed=\(condition)")
                if !condition { failures += 1 }
            }
            let a = Vector512Optimized(repeating: 1), b = Vector512Optimized(repeating: 1.0001)
            let reference = a.euclideanDistance(to: b)
            let reduced = MixedPrecisionKernels.adaptiveEuclideanDistance(query: a, candidate: b, threshold: 0)
            print("ADAPTIVE nearTie directBits=\(reference.bitPattern) resultBits=\(reduced.bitPattern)")
            check(reference.bitPattern == reduced.bitPattern, "thresholdZero")
            let zero = Vector512Optimized(repeating: 0), large = Vector512Optimized(repeating: 70000)
            for (q,c) in [(zero,large),(large,zero)] {
                let direct = q.euclideanDistance(to:c), actual = MixedPrecisionKernels.adaptiveEuclideanDistance(query:q,candidate:c,threshold:0.001)
                print("ADAPTIVE range direct=\(direct) result=\(actual)")
                check(actual.isFinite && abs(Double(actual)-Double(direct)) <= 0.001*Double(direct), "bothOperandRanges")
            }
            let quantized = QuantizedAutoTuner.shared
            for hints: [Float] in [[0.02,0.2],[0.2,0.02]] {
                quantized.clearCache()
                var warm: [String] = []
                for hint in hints {
                    #if Q02_BASELINE
                    let result = quantized.selectOptimalQuantization(for:"dot",dimension:512,candidateCount:2000,accuracyRequirement:hint)
                    #else
                    let result = try quantized.suggestQuantization(for:"dot",dimension:512,candidateCount:2000,relativeErrorHint:hint)
                    #endif
                    warm.append(String(describing:result))
                }
                for (index,hint) in hints.enumerated() {
                    quantized.clearCache()
                    #if Q02_BASELINE
                    let result = quantized.selectOptimalQuantization(for:"dot",dimension:512,candidateCount:2000,accuracyRequirement:hint)
                    #else
                    let result = try quantized.suggestQuantization(for:"dot",dimension:512,candidateCount:2000,relativeErrorHint:hint)
                    #endif
                    print("QUANTIZED hint=\(hint) warm=\(warm[index]) fresh=\(result)")
                    check(warm[index] == String(describing:result), "quantizedOrder")
                }
            }
            #if Q02_BASELINE
            let kernel = KernelAutoTuner.shared
            let ordinary = WorkloadCharacteristics(dimension:512,batchSize:4,precisionMode:.fp32Full)
            let filtered = WorkloadCharacteristics(dimension:512,batchSize:4,precisionMode:.fp32Full,memoryBudget:0,latencyTarget:1e-30)
            let unfilteredResult = await kernel.selectStrategy(for:ordinary)
            let filteredResult = await kernel.selectStrategy(for:filtered)
            print("KERNEL ignoredConstraints ordinary=\(unfilteredResult) filtered=\(filteredResult)")
            let payload: Int
            switch filteredResult {
            case .cpuAoS, .cpuSoA: payload = (4+1)*512*4
            case .cpuMixedPrecision: payload = 512*4+4*512*2
            case .cpuQuantized: payload = (4+1)*512
            case .auto: payload = Int.max
            }
            check(payload <= 0, "infeasibleMustThrow")
            await kernel.setOverride(.cpuQuantized,for:ordinary)
            check(await kernel.selectStrategy(for:ordinary) != .cpuQuantized, "fullOverridePrecedence")
            await kernel.removeOverride(for:ordinary)
            await MixedPrecisionAutoTuner.shared.clearCache()
            await MixedPrecisionAutoTuner.shared.setOverride(.queryFP32Blocked,candidateCount:4,accuracyRequirement:0.0001,dimension:512)
            check(await kernel.selectMixedPrecisionStrategy(for:ordinary) == .fullFP32, "fullIntegrationPrecedence")
            await MixedPrecisionAutoTuner.shared.clearCache()
            #endif
            print("Q02_RESULT semantics failures=\(failures)")
            print("Q02_BODY_END mode=\(mode)")
            if failures > 0 { exit(1) }
            return
        }
        let width = Int(CommandLine.arguments[2])!, count = Int(CommandLine.arguments[3])!
        await MixedPrecisionAutoTuner.shared.clearCache()
        if mode == "work" {
            MixedPrecisionDiagnostics.shared.enable()
            let strategy = try await select(width,count)
            let conversions = MixedPrecisionDiagnostics.shared.getStatistics().totalConversions
            MixedPrecisionDiagnostics.shared.disable()
            print("WORK width=\(width) count=\(count) convertedElements=\(conversions) strategy=\(strategy)")
        } else {
            calibrateFilter(); calibrateC()
            _ = arrayControl(4096).count
            start(0); stop()
            precondition(read(0)==0 && read(1)==0)
            print("CALIBRATION noop events=\(read(0)) bytes=\(read(1))")
            start(0); let array=arrayControl(4096); stop()
            precondition(read(0)>0 && read(1)>=16384 && array.reduce(0,+)==12288)
            print("CALIBRATION array events=\(read(0)) bytes=\(read(1))")
            for phase in ["cold","warm"] {
                let clock = ContinuousClock(), before=clock.now
                start(0)
                let strategy = try await select(width,count)
                stop()
                let elapsed=seconds(before.duration(to:clock.now)), events=read(0), bytes=read(1)
                print("MEASURE phase=\(phase) width=\(width) count=\(count) events=\(events) bytes=\(bytes) elapsedSeconds=\(elapsed) strategy=\(strategy)")
            }
        }
        print("Q02_RESULT \(mode)")
        print("Q02_BODY_END mode=\(mode)")
    }
}
