// D02 H1/H2/H7 (negative): each line uses the removed `Operations.bufferProvider` hook (or its projected binding),
// the removed `VectorCore.Configuration` type or one of its members, or the removed `VectorCore.configuration`, so
// after D02 each line is one compile error. Before D02 the file compiles.
import VectorCore

func bufferProvider() -> any BufferProvider { Operations.bufferProvider }
func projectedBufferProvider() -> Any { Operations.$bufferProvider }
func configuration() -> Any { VectorCore.configuration }
func configurationType() -> Any.Type { VectorCore.Configuration.self }
func batchOperations() -> BatchOperations.Configuration { VectorCore.Configuration().batchOperations }
func memoryAlignment() -> Int { VectorCore.Configuration().memoryAlignment }
