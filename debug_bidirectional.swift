import Foundation

// Simple implementation to debug the issue
func bidirectionalBFSDebug(
    adjList: [[Int]],
    source: Int,
    target: Int
) -> (path: [Int]?, distance: Int) {
    if source == target {
        return ([source], 0)
    }

    let n = adjList.count
    var forwardVisited = Array(repeating: false, count: n)
    var backwardVisited = Array(repeating: false, count: n)
    var forwardParent = Array(repeating: -1, count: n)
    var backwardParent = Array(repeating: -1, count: n)
    var forwardDist = Array(repeating: -1, count: n)
    var backwardDist = Array(repeating: -1, count: n)

    forwardVisited[source] = true
    forwardDist[source] = 0
    backwardVisited[target] = true
    backwardDist[target] = 0

    var forwardQueue = [source]
    var backwardQueue = [target]
    var meeting = -1

    var level = 0
    while !forwardQueue.isEmpty && !backwardQueue.isEmpty {
        print("\n=== Level \(level) ===")
        print("Forward frontier: \(forwardQueue)")
        print("Backward frontier: \(backwardQueue)")

        // Expand smaller frontier
        if forwardQueue.count <= backwardQueue.count {
            var nextQueue: [Int] = []
            for u in forwardQueue {
                for v in adjList[u] {
                    if backwardVisited[v] {
                        print("Meeting at node \(v): forward from \(u), already visited from backward")
                        print("Forward distance to \(u): \(forwardDist[u])")
                        print("Backward distance to \(v): \(backwardDist[v])")
                        meeting = v
                        forwardParent[v] = u
                        let totalDist = forwardDist[u] + 1 + backwardDist[v]
                        print("Total distance: \(forwardDist[u]) + 1 + \(backwardDist[v]) = \(totalDist)")

                        // Reconstruct path
                        var path = [Int]()
                        var curr = v
                        while curr != source {
                            path.append(curr)
                            curr = forwardParent[curr]
                        }
                        path.append(source)
                        path.reverse()

                        curr = backwardParent[v]
                        while curr != -1 && curr != target {
                            path.append(curr)
                            curr = backwardParent[curr]
                        }
                        if curr == target {
                            path.append(target)
                        }

                        return (path, totalDist)
                    }

                    if !forwardVisited[v] {
                        forwardVisited[v] = true
                        forwardDist[v] = forwardDist[u] + 1
                        forwardParent[v] = u
                        nextQueue.append(v)
                    }
                }
            }
            forwardQueue = nextQueue
        } else {
            var nextQueue: [Int] = []
            for u in backwardQueue {
                for v in adjList[u] {
                    if forwardVisited[v] {
                        print("Meeting at node \(v): backward from \(u), already visited from forward")
                        print("Forward distance to \(v): \(forwardDist[v])")
                        print("Backward distance to \(u): \(backwardDist[u])")
                        meeting = v
                        backwardParent[v] = u
                        let totalDist = forwardDist[v] + backwardDist[u] + 1
                        print("Total distance: \(forwardDist[v]) + \(backwardDist[u]) + 1 = \(totalDist)")

                        // Reconstruct path
                        var path = [Int]()
                        var curr = v
                        while curr != source {
                            path.append(curr)
                            curr = forwardParent[curr]
                        }
                        path.append(source)
                        path.reverse()

                        curr = backwardParent[v]
                        while curr != -1 && curr != target {
                            path.append(curr)
                            curr = backwardParent[curr]
                        }
                        if curr == target {
                            path.append(target)
                        }

                        return (path, totalDist)
                    }

                    if !backwardVisited[v] {
                        backwardVisited[v] = true
                        backwardDist[v] = backwardDist[u] + 1
                        backwardParent[v] = u
                        nextQueue.append(v)
                    }
                }
            }
            backwardQueue = nextQueue
        }
        level += 1
    }

    return (nil, -1)
}

// Test graph:
//   0 -- 1 -- 2
//   |    |    |
//   3 -- 4 -- 5
let adjList = [
    [1, 3],     // 0
    [0, 2, 4],  // 1
    [1, 5],     // 2
    [0, 4],     // 3
    [1, 3, 5],  // 4
    [2, 4]      // 5
]

let (path, dist) = bidirectionalBFSDebug(adjList: adjList, source: 0, target: 5)
print("\n=== Final Result ===")
print("Path: \(path ?? [])")
print("Distance: \(dist)")