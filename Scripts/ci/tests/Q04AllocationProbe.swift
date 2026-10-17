import Foundation
import VectorCore
@_silgen_name("q04_start") func start(_ threshold: UInt)
@_silgen_name("q04_stop") func stop()
@_silgen_name("q04_read") func read(_ field: Int32) -> UInt64
@_silgen_name("q04_calibrate_filter") func calibrateFilter()
@_silgen_name("q04_calibrate_c") func calibrateC()
@inline(never) func arrayControl(_ n: Int) -> [Float] { Array(repeating: 3, count: n) }
calibrateFilter()
calibrateC()
_ = arrayControl(4096).count
start(0); stop()
precondition(read(0) == 0 && read(1) == 0)
print("CALIBRATION noop events=\(read(0)) bytes=\(read(1))")
start(0)
let control = arrayControl(4096)
stop()
precondition(read(0) > 0 && read(1) >= 16384 && control.reduce(0,+) == 12288)
print("CALIBRATION array events=\(read(0)) bytes=\(read(1))")

func measure512() {
 let query = Vector512Optimized(repeating: 1)
 let output = UnsafeMutablePointer<Float>.allocate(capacity: 5)
 output.initialize(repeating: -777, count: 5)
 defer { output.deinitialize(count: 5); output.deallocate() }
 for n in [3, 64, 1024, 4096] {
  let candidates = (0..<n).map { index in MixedPrecisionKernels.Vector512FP16(fp16Values: Array(repeating: Float16(1 + Float(index % 7) / 8).bitPattern, count: 512)) }
  for count in 0...3 {
   for lower in [0, n - count] {
    let range = lower..<(lower + count)
    let out = UnsafeMutableBufferPointer(start: output, count: 5)
    for metric in 0..<3 {
     func invoke() {
      switch metric {
      case 0: MixedPrecisionKernels.range_euclid2_mixed_512(query: query, candidatesFP16: candidates, range: range, out: out)
      case 1: MixedPrecisionKernels.range_cosine_mixed_512(query: query, candidatesFP16: candidates, range: range, out: out)
      case 2: MixedPrecisionKernels.range_dot_mixed_512(query: query, candidatesFP16: candidates, range: range, out: out)
      default: fatalError("metric")
      }
     }
     invoke(); invoke()
     for i in 0..<5 { output[i] = -777 }
     start(0)
     invoke()
     stop()
     let events = read(0), bytes = read(1)
     for j in 0..<count {
      let value = Double(1 + Float((lower + j) % 7) / 8)
      let expected = metric == 0 ? Double(512) * (1-value)*(1-value) : (metric == 1 ? 0 : Double(512)*value)
      precondition(output[j].isFinite && abs(Double(output[j])-expected) <= 0.00001 * max(1,abs(expected)))
     }
     for j in count..<5 { precondition(output[j].bitPattern == Float(-777).bitPattern) }
     print("ROW width=512 metric=\(metric) n=\(n) count=\(count) lower=\(lower) events=\(events) bytes=\(bytes)")
    }
   }
  }
 }
}
measure512()

func measure768() {
 let query = Vector768Optimized(repeating: 1)
 let output = UnsafeMutablePointer<Float>.allocate(capacity: 5)
 output.initialize(repeating: -777, count: 5)
 defer { output.deinitialize(count: 5); output.deallocate() }
 for n in [3, 64, 1024, 4096] {
  let candidates = (0..<n).map { index in MixedPrecisionKernels.Vector768FP16(fp16Values: Array(repeating: Float16(1 + Float(index % 7) / 8).bitPattern, count: 768)) }
  for count in 0...3 {
   for lower in [0, n - count] {
    let range = lower..<(lower + count)
    let out = UnsafeMutableBufferPointer(start: output, count: 5)
    for metric in 0..<3 {
     func invoke() {
      switch metric {
      case 0: MixedPrecisionKernels.range_euclid2_mixed_768(query: query, candidatesFP16: candidates, range: range, out: out)
      case 1: MixedPrecisionKernels.range_cosine_mixed_768(query: query, candidatesFP16: candidates, range: range, out: out)
      case 2: MixedPrecisionKernels.range_dot_mixed_768(query: query, candidatesFP16: candidates, range: range, out: out)
      default: fatalError("metric")
      }
     }
     invoke(); invoke()
     for i in 0..<5 { output[i] = -777 }
     start(0)
     invoke()
     stop()
     let events = read(0), bytes = read(1)
     for j in 0..<count {
      let value = Double(1 + Float((lower + j) % 7) / 8)
      let expected = metric == 0 ? Double(768) * (1-value)*(1-value) : (metric == 1 ? 0 : Double(768)*value)
      precondition(output[j].isFinite && abs(Double(output[j])-expected) <= 0.00001 * max(1,abs(expected)))
     }
     for j in count..<5 { precondition(output[j].bitPattern == Float(-777).bitPattern) }
     print("ROW width=768 metric=\(metric) n=\(n) count=\(count) lower=\(lower) events=\(events) bytes=\(bytes)")
    }
   }
  }
 }
}
measure768()

func measure1536() {
 let query = Vector1536Optimized(repeating: 1)
 let output = UnsafeMutablePointer<Float>.allocate(capacity: 5)
 output.initialize(repeating: -777, count: 5)
 defer { output.deinitialize(count: 5); output.deallocate() }
 for n in [3, 64, 1024, 4096] {
  let candidates = (0..<n).map { index in MixedPrecisionKernels.Vector1536FP16(fp16Values: Array(repeating: Float16(1 + Float(index % 7) / 8).bitPattern, count: 1536)) }
  for count in 0...3 {
   for lower in [0, n - count] {
    let range = lower..<(lower + count)
    let out = UnsafeMutableBufferPointer(start: output, count: 5)
    for metric in 0..<3 {
     func invoke() {
      switch metric {
      case 0: MixedPrecisionKernels.range_euclid2_mixed_1536(query: query, candidatesFP16: candidates, range: range, out: out)
      case 1: MixedPrecisionKernels.range_cosine_mixed_1536(query: query, candidatesFP16: candidates, range: range, out: out)
      case 2: MixedPrecisionKernels.range_dot_mixed_1536(query: query, candidatesFP16: candidates, range: range, out: out)
      default: fatalError("metric")
      }
     }
     invoke(); invoke()
     for i in 0..<5 { output[i] = -777 }
     start(0)
     invoke()
     stop()
     let events = read(0), bytes = read(1)
     for j in 0..<count {
      let value = Double(1 + Float((lower + j) % 7) / 8)
      let expected = metric == 0 ? Double(1536) * (1-value)*(1-value) : (metric == 1 ? 0 : Double(1536)*value)
      precondition(output[j].isFinite && abs(Double(output[j])-expected) <= 0.00001 * max(1,abs(expected)))
     }
     for j in count..<5 { precondition(output[j].bitPattern == Float(-777).bitPattern) }
     print("ROW width=1536 metric=\(metric) n=\(n) count=\(count) lower=\(lower) events=\(events) bytes=\(bytes)")
    }
   }
  }
 }
}
measure1536()
print("Q04_PROBE_COMPLETE")
