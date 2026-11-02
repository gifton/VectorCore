// D02 R9 / S1 / S2 (negative): each line names the removed `StaticDimension.Storage` requirement or a removed
// built-in `DimN.Storage` alias, so after D02 each line is one compile error. Before D02 the file compiles.
import VectorCore

func requirement<D: StaticDimension>(_: D.Type) -> Int where D.Storage: VectorStorageOperations { D.value }
func aliasOf512() -> Any.Type { Dim512.Storage.self }
func aliasOf2() -> Any.Type { Dim2.Storage.self }
