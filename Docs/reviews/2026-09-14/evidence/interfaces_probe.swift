import Foundation
import VectorCore
@main struct Probe {
 static func main() throws {
  let c = SimpleVectorCollection<DynamicVector>(dimension: 2)
  try c.add(id: 1, vector: DynamicVector([3, 0]))
  try c.add(id: 2, vector: DynamicVector([2, 2]))
  let result = c.search(query: DynamicVector([0,0]), k: 1, metric: ChebyshevDistance())
  print("chebyshev expected ID2 distance2; actual", result.ids, result.distances)
  var u = SimpleVectorCollection<DynamicVector>(dimension: 2)
  try u.add(id: 7, vector: DynamicVector([1, 2]))
  do { try u.update(id: 7, vector: DynamicVector([3])) } catch { print("update rejected:", error) }
  print("original exists after failed update", u.vector(for: 7) != nil)
  let auto = SimpleVectorCollection<DynamicVector>(dimension: 2)
  try auto.add(id: 0, vector: DynamicVector([1, 2]))
  let id = try auto.add(vector: DynamicVector([3, 4]))
  print("auto ID after explicit0", id, "count", auto.count, "id0", auto.vector(for: 0)!.toArray())
  for d: Float in [-2, -1, -0.5, 0, 1] { print("score", d, SearchResult(id: 0, distance: d).computedScore) }
  guard CommandLine.arguments.contains("zero") else { return }
  let zero = Vector512Optimized().normalizedUncheckedWithHint()
  print("zero hint", zero.isNormalized, "claimed", zero.magnitude, "actual", zero.vector.magnitude)
 }
}
