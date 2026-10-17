// D02 RD-R5 (negative): with both `os` and VectorCore imported and no scoped import, a bare `Logger` type is
// ambiguous, and `VectorCore.Logger` names a member of the `VectorCore` enum, which has none; each line is one
// compile error, before and after D02 (Logger.swift documents the scoped import instead).
import os
import VectorCore

func bare() -> Any.Type { Logger.self }
func enumQualified() -> Any.Type { VectorCore.Logger.self }
