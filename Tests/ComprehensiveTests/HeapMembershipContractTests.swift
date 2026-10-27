import Foundation
import Testing
#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif
@testable import VectorCore

@Suite struct HeapMembershipContractTests {
    @Test(arguments:[false,true]) func bulkMatchesStreaming(_ reversed:Bool) {
        let compare:(Int,Int)->Bool = { a,b in reversed ? a > b : a < b }
        let bulk=MinHeap(from:[1,2,3],capacity:2,compare:compare)
        var stream=MinHeap<Int>(capacity:2,compare:compare)
        for value in [1,2,3] { stream.insert(value) }
        #expect(bulk.sorted() == stream.sorted())
        #expect(bulk.sorted() == (reversed ? [2,1] : [2,3]))
    }
    @Test func numericEvictsNaN() {
        var heap=KNearestHeap(k:1)
        let first=heap.insert(index:0,distance:.nan);#expect(first)
        let second=heap.insert(index:1,distance:1);#expect(second)
        let result=heap.getSorted()
        #expect(result.count == 1 && result[0].index == 1 && result[0].distance.isFinite)
    }
    @Test(.serialized,arguments:["empty","bulk","nearest"])
    func negativeCapacity(_ scenario:String) throws {
        var env=ProcessInfo.processInfo.environment;env["R04_SCENARIO"]=scenario
        // Never write into the package tree: default to a unique temporary directory.
        let fallback=FileManager.default.temporaryDirectory.appendingPathComponent("R04-child-captures-\(UUID().uuidString)").path
        let directory=env["R04_CAPTURE_DIRECTORY"] ?? fallback
        try FileManager.default.createDirectory(atPath:directory,withIntermediateDirectories:true)
        let path=URL(fileURLWithPath:directory).appendingPathComponent("R04-\(scenario)-\(UUID().uuidString).log")
        FileManager.default.createFile(atPath:path.path,contents:nil)
        let file=try FileHandle(forWritingTo:path)
        let child = try launchChildProbe(.swiftTesting(filter: "HeapMembershipContractTests/childProbe"), environment: env, output: file,
                                         deadline: 15, cleanup: .terminateThenKill(grace: 1, seconds: 1))
        if child.timedOut {
            try file.close();print("R04_CHILD_TIMEOUT \(scenario) capture=\(path.path)")
            throw NSError(domain:"R04.timeout",code:1)
        }
        try file.close();let output=try String(contentsOf:path,encoding:.utf8)
        print("R04_CHILD_RESULT \(scenario) status=\(child.statusDescription) reason=\(child.reasonDescription) capture=\(path.path)")
        #expect(child.terminationReason == .uncaughtSignal)
        #expect(output.contains("R04_STARTED \(scenario)"))
        #expect(!output.contains("R04_ACCEPTED"))
        #if DEBUG
        #expect(output.contains("capacity must be nonnegative"))
        #endif
    }
    @Test func childProbe() throws {
        guard let scenario=ProcessInfo.processInfo.environment["R04_SCENARIO"] else { return }
        print("R04_INVOCATION "+String(decoding:try JSONSerialization.data(withJSONObject:["argv":CommandLine.arguments,"scenario":scenario],options:[.sortedKeys]),as:UTF8.self))
        print("R04_STARTED \(scenario)");fflush(stdout)
        switch scenario {
        case "empty": _ = MinHeap<Int>(capacity:-1,compare:<)
        case "bulk": _ = MinHeap<Int>(from:[],capacity:-1,compare:<)
        case "nearest": _ = KNearestHeap(k:-1)
        default:throw NSError(domain:"R04.scenario",code:1)
        }
        print("R04_ACCEPTED")
    }
}

private func r04Permutations<T>(_ values:[T]) -> [[T]] {
    if values.isEmpty { return [[]] }
    var result:[[T]]=[]
    for i in values.indices {
        var rest=values;let first=rest.remove(at:i)
        for tail in r04Permutations(rest) { result.append([first]+tail) }
    }
    return result
}
private struct R04Record { let key:Int;let label:Int }
private final class R04Box { let key:Int;init(_ key:Int) { self.key=key } }

extension HeapMembershipContractTests {
    @Test func genericMembershipMatrix() {
        for input in [[Int](),[7],[1,2,3,4],[1,1,2,3]] {
            let capacities:[Int?] = [nil]+Set([0,1,max(0,input.count-1),input.count,input.count+1,Int.max]).sorted().map { Optional($0) }
            for values in r04Permutations(input) {
                for descending in [false,true] {
                    let compare:(Int,Int)->Bool = { a,b in descending ? a > b : a < b }
                    let ordered=values.sorted(by:compare)
                    for capacity in capacities {
                        let expected=Array(ordered.suffix(min(capacity ?? values.count,values.count)))
                        let bulk=MinHeap(from:values,capacity:capacity,compare:compare)
                        var streaming=MinHeap<Int>(capacity:capacity,compare:compare)
                        for value in values {
                            streaming.insert(value)
                            #expect(streaming.count <= (capacity ?? values.count))
                        }
                        #expect(bulk.sorted() == expected)
                        #expect(streaming.sorted() == expected)
                        #expect(Array(bulk).sorted(by:compare) == expected)
                        #expect(bulk.count == expected.count)
                        var copied=bulk
                        let removed=copied.removeMin()
                        #expect(removed == expected.first)
                        #expect(copied.sorted() == Array(expected.dropFirst()))
                        #expect(bulk.sorted() == expected)
                    }
                }
            }
        }
    }

    @Test func recordEquivalenceAndInsertionResults() {
        let records=[R04Record(key:1,label:9),R04Record(key:1,label:2),R04Record(key:2,label:8),R04Record(key:3,label:1)]
        for values in r04Permutations(records) {
            for descending in [false,true] {
                let compare:(R04Record,R04Record)->Bool = { a,b in descending ? a.key > b.key : a.key < b.key }
                let expected=Array(values.map(\.key).sorted(by:{ descending ? $0 > $1 : $0 < $1 }).suffix(2))
                let bulk=MinHeap(from:values,capacity:2,compare:compare)
                var streaming=MinHeap<R04Record>(capacity:2,compare:compare)
                for value in values { streaming.insert(value) }
                // Equivalent keys have no promised identity or stable drain order.
                #expect(bulk.sorted().map(\.key) == expected)
                #expect(streaming.sorted().map(\.key) == expected)
            }
            let strict:(R04Record,R04Record)->Bool = { a,b in a.key == b.key ? a.label < b.label : a.key < b.key }
            let expected=Array(values.sorted(by:strict).suffix(2)).map(\.label)
            #expect(MinHeap(from:values,capacity:2,compare:strict).sorted().map(\.label) == expected)
        }
        var heap=MinHeap<R04Record>(capacity:1,compare:{ $0.key < $1.key })
        let under=heap.insert(.init(key:2,label:7));#expect(under)
        let equal=heap.insert(.init(key:2,label:9));#expect(!equal)
        #expect(heap.min?.label == 7)
        let worse=heap.insert(.init(key:1,label:3));#expect(!worse)
        let better=heap.insert(.init(key:3,label:4));#expect(better)
        #expect(heap.min?.label == 4 && heap.count == 1)
        var zero=MinHeap<Int>(capacity:0,compare:<)
        let rejected=zero.insert(1);#expect(!rejected && zero.isEmpty)
        var unbounded=MinHeap<Int>(compare:<)
        let a=unbounded.insert(1);let b=unbounded.insert(1);#expect(a && b && unbounded.count == 2)
    }

    @Test func retainedValuesSurviveCopyMutation() {
        weak var observed:R04Box?
        do {
            var first=MinHeap<R04Box>(capacity:1,compare:{ $0.key < $1.key })
            do { let value=R04Box(1);observed=value;first.insert(value) }
            let copy=first
            first.insert(R04Box(2))
            #expect(observed != nil && copy.min?.key == 1 && first.min?.key == 2)
        }
        #expect(observed == nil)
    }

    @Test func nearestPermutationMatrix() {
        typealias Pair=(index:Int,distance:Float)
        func before(_ a:Pair,_ b:Pair)->Bool {
            if a.distance.isNaN { return b.distance.isNaN && a.index < b.index }
            if b.distance.isNaN { return true }
            if a.distance < b.distance { return true }
            if a.distance > b.distance { return false }
            return a.index < b.index
        }
        let fixtures:[[Pair]] = [[],[(4,2)],[(9,.nan),(8,.infinity),(7,-.infinity),(4,-0.0),(2,0.0),(5,2)],[(8,2),(3,2),(9,.nan),(1,.nan)],[(3,.nan),(2,.nan),(1,.nan)],[(4,1),(4,1),(2,.nan)]]
        for fixture in fixtures {
            for values in r04Permutations(fixture) {
                let ordered=values.sorted(by:before)
                for k in Set([0,1,max(0,values.count-1),values.count,values.count+1,Int.max]) {
                    var heap=KNearestHeap(k:k)
                    for value in values { heap.insert(index:value.index,distance:value.distance);#expect(heap.count <= k) }
                    let actual=heap.getSorted();let expected=Array(ordered.prefix(min(k,values.count)))
                    #expect(actual.count == expected.count)
                    for i in expected.indices {
                        #expect(actual[i].index == expected[i].index)
                        if expected[i].distance.isNaN { #expect(actual[i].distance.isNaN) }
                        else { #expect(actual[i].distance.bitPattern == expected[i].distance.bitPattern) }
                    }
                }
            }
        }
        // Modern public route uses positional IDs and remains a control.
        let scores:[Float] = [.nan,2,-.infinity,-0.0,0.0,.infinity,.nan]
        let candidates=scores.enumerated().map { Pair(index:$0.offset,distance:$0.element) }
        for k in [0,1,3,scores.count,scores.count+1] {
            let expected=Array(candidates.sorted(by:before).prefix(min(k,scores.count)))
            let actual=TopKSelection.select(k:k,from:scores)
            #expect(actual.indices == expected.map(\.index))
            for i in expected.indices {
                if expected[i].distance.isNaN { #expect(actual.distances[i].isNaN) }
                else { #expect(actual.distances[i].bitPattern == expected[i].distance.bitPattern) }
            }
        }
    }
}
