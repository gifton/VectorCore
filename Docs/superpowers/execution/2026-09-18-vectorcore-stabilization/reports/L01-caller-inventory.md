# L01 caller inventory

Base 101b3749bf531cfdbc42d61198b503e94af4aa5b. Production consumers are PCA.swift (three QR calls and one SVD call), the two provider implementations, and the C shim. Existing direct test consumers are LinearAlgebraProviderTests.swift and PCATests.swift. No standalone Scripts caller was found. The source-established workspace hazards remain distinct from numerical runtime RED.

Existing typed error assertions cover wide QR invalidOperation, count mismatch dimensionMismatch, and nonpositive dimensions invalidDimension. No caller expects the obsolete Float Frobenius invalidData error, nonfinite successful results, or raw workspace sentinels. No additional existing-test migration is needed beyond the brief-authorized invariant helper repair and PCA override control. Public protocol signatures remain unchanged.
