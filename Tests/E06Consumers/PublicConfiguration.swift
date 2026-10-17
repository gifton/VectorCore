import Foundation
import VectorCore

@main struct E06PublicConfiguration {
    static func bits(_ c: BatchOperations.Configuration) -> [UInt64] {
        [UInt64(c.parallelThreshold),c.oversubscription.bitPattern,UInt64(c.minimumChunkSize),UInt64(c.defaultBatchSize),c.enableMatrixRouting ? 1:0,UInt64(c.matrixRoutingMinN)]
    }
    static func main() async throws {
        let original = await BatchOperations.configuration()
        do {
            try await BatchOperations.updateConfiguration { $0.defaultBatchSize = 2; $0.parallelThreshold = 1 }
            let rows = (0..<5).map { DynamicVector([Float($0)]) }
            let shapes: [Int] = try await BatchOperations.process(rows) { [$0.count] }
            guard shapes == [2,2,1] else { fatalError("Live default batch size not observed") }
            let before = await BatchOperations.configuration()
            do {
                try await BatchOperations.updateConfiguration { $0.defaultBatchSize = 0; $0.enableMatrixRouting.toggle() }
                fatalError("Invalid configuration accepted")
            } catch let error as VectorError {
                guard error.kind == .invalidConfiguration else { throw error }
            }
            guard bits(await BatchOperations.configuration()) == bits(before) else { fatalError("Rollback failed") }
        } catch {
            try await BatchOperations.updateConfiguration { $0 = original }
            throw error
        }
        try await BatchOperations.updateConfiguration { $0 = original }
        guard bits(await BatchOperations.configuration()) == bits(original) else { fatalError("Restore failed") }
        print("E06_PUBLIC_VALIDATED_CONFIGURATION_COMPLETE")
    }
}
