import Foundation
import Testing
@testable import VectorCore

@Suite struct VectorCollectionTransactionTests {
    @Test func failedUpdatePreservesBinding() throws {
        let collection = SimpleVectorCollection<DynamicVector>(dimension: 2)
        try collection.add(id: 7, vector: DynamicVector([3, 4]))
        #expect(throws: VectorError.self) { try collection.update(id: 7, vector: DynamicVector([9])) }
        #expect(collection.vector(for: 7)?.toArray() == [3, 4])
        #expect(collection.count == 1)
        #expect(Set(collection.allIDs) == [7])
        #expect(collection.searchEuclidean(query: DynamicVector([3,4]), k: 1).ids == [7])
    }

    @Test func generatedAndExplicitIDsNeverOverwrite() throws {
        let collection = SimpleVectorCollection<DynamicVector>(dimension: 2)
        try collection.add(id: 0, vector: DynamicVector([3, 4]))
        let generated = try collection.add(vector: DynamicVector([5, 6]))
        #expect(generated == 1)
        #expect(collection.vector(for: 0)?.toArray() == [3, 4])
        #expect(collection.vector(for: generated)?.toArray() == [5, 6])
        #expect(collection.count == 2)
        #expect(throws: VectorError.self) { try collection.add(id: 0, vector: DynamicVector([7, 8])) }
        #expect(collection.vector(for: 0)?.toArray() == [3, 4])
        #expect(collection.count == 2)
    }

    private func expectError(_ kind: VectorError.ErrorKind, _ operation: () throws -> Void) {
        do { try operation(); Issue.record("Expected \(kind)") }
        catch let error as VectorError { #expect(error.kind == kind) }
        catch { Issue.record("Unexpected error: \(error)") }
    }

    @Test func allUpdateFailuresPreserveEveryBindingAndSearch() throws {
        let collection = SimpleVectorCollection<DynamicVector>(dimension: 3)
        try collection.add(id: 7, vector: DynamicVector([3,-0.0,4]))
        try collection.add(id: 9, vector: DynamicVector([-2,0,0]))
        let original = collection.vectors(for: [7,9]).mapValues { $0.toArray().map(\.bitPattern) }
        let query = DynamicVector([0,0,0])
        let before = collection.searchEuclidean(query:query,k:10)
        let cases: [(Int,[Float],VectorError.ErrorKind)] = [(7,[1,2],.dimensionMismatch),(33,[1,2,3],.indexOutOfBounds),(33,[1,2],.dimensionMismatch)]
        for (id,values,kind) in cases {
            expectError(kind) { try collection.update(id:id,vector:DynamicVector(values)) }
            #expect(collection.count == 2)
            #expect(Set(collection.allIDs) == [7,9])
            #expect(collection.vectors(for:[7,9,33]).mapValues { $0.toArray().map(\.bitPattern) } == original)
            let after = collection.searchEuclidean(query:query,k:10)
            #expect(after.ids == before.ids)
            #expect(after.distances.map(\.bitPattern) == before.distances.map(\.bitPattern))
            #expect(after.candidatesSearched == 2)
        }
        try collection.update(id:7,vector:DynamicVector([0,6,8]))
        #expect(collection.count == 2)
        #expect(collection.vector(for:7)?.toArray() == [0,6,8])
        #expect(collection.searchEuclidean(query:query,k:10).distances == [2,10])
    }

    @Test func explicitAndGeneratedInsertionMatrix() throws {
        let collection = SimpleVectorCollection<DynamicVector>(dimension: 2)
        let value = DynamicVector([1,2])
        try collection.add(id:-9,vector:value)
        try collection.add(id:Int.max,vector:DynamicVector([3,4]))
        try collection.add(id:3,vector:DynamicVector([5,6]))
        try collection.add(id:100,vector:DynamicVector([7,8]))
        #expect(try collection.add(vector:DynamicVector([9,10])) == 0)
        #expect(try collection.add(vector:DynamicVector([11,12])) == 1)
        for id in [-9,0,3,Int.max] {
            let before = collection.vectors(for:collection.allIDs)
            expectError(.invalidConfiguration) { try collection.add(id:id,vector:value) }
            #expect(collection.vectors(for:collection.allIDs) == before)
            #expect(collection.count == 6)
        }
        #expect(try collection.add(vector:DynamicVector([13,14])) == 2)
        #expect(try collection.add(vector:DynamicVector([15,16])) == 4)
        #expect(collection.remove(id:1)?.toArray() == [11,12])
        #expect(try collection.add(vector:DynamicVector([17,18])) == 5)
        try collection.add(id:1,vector:DynamicVector([19,20]))
        #expect(collection.vector(for:1)?.toArray() == [19,20])
        #expect(collection.vector(for:-9)?.toArray() == [1,2])
        #expect(collection.vector(for:Int.max)?.toArray() == [3,4])
        #expect(collection.vector(for:3)?.toArray() == [5,6])
        #expect(collection.vector(for:100)?.toArray() == [7,8])
        #expect(collection.vector(for:0)?.toArray() == [9,10])
        #expect(collection.vector(for:2)?.toArray() == [13,14])
        #expect(collection.vector(for:4)?.toArray() == [15,16])
        #expect(collection.vector(for:5)?.toArray() == [17,18])
        #expect(collection.count == 9)
        collection.removeAll()
        #expect(collection.isEmpty && collection.allIDs.isEmpty)
        #expect(try collection.add(vector:value) == 6)
        try collection.add(id:0,vector:DynamicVector([21,22]))
        #expect(collection.vector(for:0)?.toArray() == [21,22])
        #expect(collection.count == 2)
    }

    @Test func failedInsertsPreserveCursorAndDimensionPrecedesDuplicate() throws {
        let collection = SimpleVectorCollection<DynamicVector>(dimension:2)
        try collection.add(id:0,vector:DynamicVector([3,4]))
        expectError(.dimensionMismatch) { _ = try collection.add(vector:DynamicVector([1])) }
        expectError(.dimensionMismatch) { try collection.add(id:0,vector:DynamicVector([1])) }
        expectError(.dimensionMismatch) { try collection.add(id:99,vector:DynamicVector([1])) }
        expectError(.invalidConfiguration) { try collection.add(id:0,vector:DynamicVector([5,6])) }
        #expect(collection.count == 1)
        #expect(collection.vector(for:0)?.toArray() == [3,4])
        // Removing the occupied cursor distinguishes unchanged cursor from eager advance.
        _ = collection.remove(id:0)
        #expect(try collection.add(vector:DynamicVector([7,8])) == 0)
        #expect(collection.count == 1)
        do { try collection.add(id:0,vector:DynamicVector([9,10])); Issue.record("Expected duplicate") }
        catch let error as VectorError {
            #expect(error.kind == .invalidConfiguration)
            let message = error.localizedDescription.lowercased()
            #expect(message.contains("duplicate") && message.contains("id") && message.contains("0"))
        }
        #expect(try collection.add(vector:DynamicVector([11,12])) == 1)
    }

    @Test func nearMaximumCursorFreeOccupiedAndExhausted() throws {
        let value = DynamicVector([1,2])
        let free = SimpleVectorCollection<DynamicVector>(dimension:2,nextGeneratedID:Int.max-1)
        #expect(try free.add(vector:value) == Int.max-1)
        #expect(try free.add(vector:DynamicVector([3,4])) == Int.max)
        expectError(.resourceExhausted) { _ = try free.add(vector:DynamicVector([5,6])) }
        expectError(.dimensionMismatch) { _ = try free.add(vector:DynamicVector([5])) }
        #expect(free.count == 2)
        #expect(free.vector(for:Int.max-1)?.toArray() == [1,2])
        #expect(free.vector(for:Int.max)?.toArray() == [3,4])
        _ = free.remove(id:Int.max)
        expectError(.resourceExhausted) { _ = try free.add(vector:value) }
        free.removeAll()
        expectError(.resourceExhausted) { _ = try free.add(vector:value) }
        try free.add(id:Int.max,vector:value)
        #expect(free.count == 1)
        expectError(.resourceExhausted) { _ = try free.add(vector:value) }

        let occupied = SimpleVectorCollection<DynamicVector>(dimension:2,nextGeneratedID:Int.max-1)
        try occupied.add(id:Int.max-1,vector:value)
        try occupied.add(id:Int.max,vector:DynamicVector([3,4]))
        expectError(.resourceExhausted) { _ = try occupied.add(vector:DynamicVector([5,6])) }
        #expect(occupied.count == 2)
        #expect(occupied.vector(for:Int.max-1)?.toArray() == [1,2])
        #expect(occupied.vector(for:Int.max)?.toArray() == [3,4])
        // A failed scan must not commit even its first occupied successor.
        _ = occupied.remove(id:Int.max-1)
        #expect(try occupied.add(vector:DynamicVector([7,8])) == Int.max-1)
        expectError(.resourceExhausted) { _ = try occupied.add(vector:value) }
        _ = occupied.remove(id:Int.max)
        #expect(try occupied.add(vector:DynamicVector([9,10])) == Int.max)
        #expect(occupied.count == 2)
        #expect(occupied.vector(for:Int.max-1)?.toArray() == [7,8])
        #expect(occupied.vector(for:Int.max)?.toArray() == [9,10])

        let lastFree = SimpleVectorCollection<DynamicVector>(dimension:2,nextGeneratedID:Int.max)
        expectError(.dimensionMismatch) { _ = try lastFree.add(vector:DynamicVector([1])) }
        #expect(try lastFree.add(vector:value) == Int.max)
        expectError(.resourceExhausted) { _ = try lastFree.add(vector:value) }
    }

    private struct RejectingCollection: MutableVectorCollection {
        enum Rejection: Error { case injected }
        var values: [Int: DynamicVector] = [7:DynamicVector([1,2])]
        var count: Int { values.count }
        var dimension: Int { 2 }
        var allIDs: [Int] { Array(values.keys) }
        func vector(for id: Int) -> DynamicVector? { values[id] }
        mutating func add(id:Int,vector:DynamicVector) throws { values[id] = vector }
        mutating func remove(id:Int) -> DynamicVector? { values.removeValue(forKey:id) }
        mutating func removeAll() { values.removeAll() }
        mutating func update(id:Int,vector:DynamicVector) throws {
            guard vector.scalarCount == dimension else { throw VectorError.dimensionMismatch(expected:dimension,actual:vector.scalarCount) }
            guard values[id] != nil else { throw VectorError.indexOutOfBounds(index:0,dimension:dimension) }
            guard vector[0] >= 0 else { throw Rejection.injected }
            values[id] = vector
        }
    }

    private func replace<C: MutableVectorCollection>(_ collection: inout C, id:C.ID, vector:C.Vector) throws {
        try collection.update(id:id,vector:vector)
    }

    @Test func customTransactionalWitnessDispatchesAndPreservesInjectedFailure() throws {
        var collection = RejectingCollection()
        #expect(throws: RejectingCollection.Rejection.self) { try replace(&collection,id:7,vector:DynamicVector([-1,2])) }
        #expect(collection.vector(for:7)?.toArray() == [1,2])
        #expect(collection.count == 1 && collection.allIDs == [7])
        #expect(collection.searchEuclidean(query:DynamicVector([1,2]),k:1).distances == [0])
        try replace(&collection,id:7,vector:DynamicVector([3,4]))
        #expect(collection.vector(for:7)?.toArray() == [3,4])
        var simple = SimpleVectorCollection<DynamicVector>(dimension:2)
        try simple.add(id:7,vector:DynamicVector([1,2]))
        try replace(&simple,id:7,vector:DynamicVector([3,4]))
        #expect(simple.vector(for:7)?.toArray() == [3,4])
    }

    @Test func batchRetrievalPreservesRequestedBindings() throws {
        let collection = SimpleVectorCollection<DynamicVector>(dimension:2)
        try collection.add(id:0,vector:DynamicVector([1,2]))
        try collection.add(id:1,vector:DynamicVector([3,4]))
        #expect(collection.vectors(for:[]).isEmpty)
        let captured = collection.vectors(for:[1,0,1,99])
        try collection.update(id:0,vector:DynamicVector([5,6]))
        _ = collection.remove(id:1)
        #expect(captured.mapValues { $0.toArray() } == [0:[1,2],1:[3,4]])
        #expect(collection.vectors(for:[0,1,99]).mapValues { $0.toArray() } == [0:[5,6]])
    }

    private struct Insertion: Sendable {
        let generated: Bool
        let id: Int
        let payload: Float
        let inserted: Bool
        let duplicate: Bool
    }

    @Test func concurrentExplicitAndGeneratedInsertionsKeepEverySuccessfulPayload() async throws {
        let collection = SimpleVectorCollection<DynamicVector>(dimension:2)
        try collection.add(id:0,vector:DynamicVector([999,-999]))
        let outcomes = await withTaskGroup(of:Insertion.self,returning:[Insertion].self) { group in
            for index in 0..<128 {
                group.addTask {
                    let payload = Float(1000+index)
                    do {
                        let id = try collection.add(vector:DynamicVector([payload,-payload]))
                        return Insertion(generated:true,id:id,payload:payload,inserted:true,duplicate:false)
                    } catch {
                        return Insertion(generated:true,id:-1,payload:payload,inserted:false,duplicate:false)
                    }
                }
                group.addTask {
                    let id = 2*index
                    let payload = Float(2000+index)
                    do {
                        try collection.add(id:id,vector:DynamicVector([payload,-payload]))
                        return Insertion(generated:false,id:id,payload:payload,inserted:true,duplicate:false)
                    } catch let error as VectorError {
                        return Insertion(generated:false,id:id,payload:payload,inserted:false,duplicate:error.kind == .invalidConfiguration)
                    } catch {
                        return Insertion(generated:false,id:id,payload:payload,inserted:false,duplicate:false)
                    }
                }
            }
            var outcomes: [Insertion] = []
            for await outcome in group { outcomes.append(outcome) }
            return outcomes
        }
        let generated = outcomes.filter(\.generated)
        #expect(generated.count == 128)
        #expect(generated.allSatisfy { $0.inserted && $0.id >= 1 })
        #expect(Set(generated.map(\.id)).count == 128)
        let successes = outcomes.filter(\.inserted)
        for outcome in outcomes {
            if outcome.inserted { #expect(collection.vector(for:outcome.id)?.toArray() == [outcome.payload,-outcome.payload]) }
            else { #expect(!outcome.generated && outcome.duplicate) }
        }
        #expect(collection.vector(for:0)?.toArray() == [999,-999])
        #expect(collection.count == successes.count+1)
        #expect(Set(collection.allIDs) == Set(successes.map(\.id)).union([0]))
    }

    private struct CoherentPayloadMetric: DistanceMetric {
        let name = "coherent-payload"
        func distance<V: VectorProtocol>(_ a:V,_ b:V) -> Float where V.Scalar == Float {
            let values = b.toArray()
            guard values.count == 8, values.dropFirst().allSatisfy({ $0 == values[1] }) else { return .nan }
            return values[0]
        }
    }

    @Test func concurrentUpdatesBatchReadsAndSearchesCaptureCompleteBindings() async throws {
        let collection = SimpleVectorCollection<DynamicVector>(dimension:8)
        for id in 0..<16 { try collection.add(id:id,vector:DynamicVector([Float(id)]+Array(repeating:0,count:7))) }
        await withTaskGroup(of:Void.self) { group in
            for writer in 0..<4 {
                group.addTask {
                    for generation in 1...64 {
                        for id in stride(from:writer,to:16,by:4) {
                            let values = [Float(id)]+Array(repeating:Float(generation),count:7)
                            do { try collection.update(id:id,vector:DynamicVector(values)) }
                            catch { Issue.record("Unexpected replacement failure \(error)") }
                        }
                    }
                }
            }
            group.addTask {
                for generation in 1...64 {
                    for id in 16..<24 {
                        do { try collection.add(id:id,vector:DynamicVector([Float(id)]+Array(repeating:Float(generation),count:7))) }
                        catch { Issue.record("Unexpected explicit insert failure \(error)") }
                    }
                    for id in 16..<24 { _ = collection.remove(id:id) }
                }
            }
            for _ in 0..<4 {
                group.addTask {
                    for _ in 0..<64 {
                        let batch = collection.vectors(for:Array(0..<24))
                        #expect(Set(0..<16).isSubset(of:Set(batch.keys)))
                        for (id,vector) in batch {
                            let values = vector.toArray()
                            #expect(values.count == 8 && values[0] == Float(id))
                            #expect(values.dropFirst().allSatisfy { $0 == values[1] })
                        }
                        let result = collection.search(query:DynamicVector(Array(repeating:0,count:8)),k:32,metric:CoherentPayloadMetric())
                        #expect(result.candidatesSearched == result.ids.count)
                        #expect(Set(result.ids).count == result.ids.count)
                        #expect(Set(0..<16).isSubset(of:Set(result.ids)))
                        #expect(result.ids.allSatisfy { (0..<24).contains($0) })
                        for (id,distance) in zip(result.ids,result.distances) { #expect(distance.isFinite && distance == Float(id)) }
                    }
                }
            }
        }
        #expect(collection.count == 16)
        for id in 0..<16 { #expect(collection.vector(for:id)?.toArray() == [Float(id)]+Array(repeating:64,count:7)) }
    }

    private struct ReentrantMetric: DistanceMetric {
        let collection: SimpleVectorCollection<DynamicVector>
        let scale: Float
        let offset: Float
        var name: String { "configured-reentrant" }
        func distance<V: VectorProtocol>(_ a:V,_ b:V) -> Float where V.Scalar == Float {
            let id = Int(b[0])
            _ = collection.vector(for:id)
            _ = collection.vectors(for:[id])
            try? collection.update(id:7,vector:DynamicVector([999,999]))
            _ = collection.remove(id:id)
            return offset + scale*b[0]
        }
    }

    @Test func reentrantSearchChildProbe() throws {
        guard ProcessInfo.processInfo.environment["R02_REENTRANT_PROBE"] == "1" else { return }
        let collection = SimpleVectorCollection<DynamicVector>(dimension:2)
        for id in 0..<8 { try collection.add(id:id,vector:DynamicVector([Float(id),Float(id)])) }
        let result = collection.search(query:DynamicVector([0,0]),k:8,metric:ReentrantMetric(collection:collection,scale:2,offset:7))
        #expect(result.ids == Array(0..<8))
        #expect(result.distances == [7,9,11,13,15,17,19,21])
        #expect(result.candidatesSearched == 8)
        #expect(collection.count == 0)
        print("R02_REENTRANT_SNAPSHOT_COMPLETE")
    }

    @Test func reentrantMetricMutatesOutsideLockAndKeepsCapturedCandidates() throws {
        var environment = ProcessInfo.processInfo.environment
        environment["R02_REENTRANT_PROBE"] = "1"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("R02-reentrant-\(UUID().uuidString).log")
        _ = FileManager.default.createFile(atPath:url.path,contents:nil)
        let handle = try FileHandle(forWritingTo:url)
        let process = try launchChildProbe(.swiftTesting(filter: "VectorCollectionTransactionTests/reentrantSearchChildProbe"),
                                           environment: environment, output: handle, deadline: 15,
                                           cleanup: .terminateThenKill(grace: 1, seconds: 1))
        let timedOut = process.timedOut
        try handle.close()
        let diagnostic = try String(contentsOf:url,encoding:.utf8)
        print("R02_CHILD_CAPTURE reentrant \(url.path)")
        if timedOut {
            throw NSError(domain: "R02.childTimeout", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Reentrant search deadline: \(url.path)"])
        }
        #expect(process.terminationStatus == 0,"\(diagnostic)")
        #expect(diagnostic.contains("R02_REENTRANT_SNAPSHOT_COMPLETE"),"\(diagnostic)")
    }
}
