// D02 RD-R5 (negative): with both `os` and VectorCore imported and no scoped import, `VectorCore.Logger` names a
// member of the `VectorCore` enum, which has none, so line 10 is one compile error on every supported compiler. A bare
// `Logger` (line 9) is ambiguous under Swift 6.4 and accepted by Swift 6.0 (Xcode 16.2), so the compile contract
// (consumer_smoke.py, row D02-logger-negative) requires line 10 and tolerates line 9; those numbers are absolute, so a
// change to this comment must update the row. Logger.swift documents the scoped import instead.
import os
import VectorCore

func bare() -> Any.Type { Logger.self }
func enumQualified() -> Any.Type { VectorCore.Logger.self }
