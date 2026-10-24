# B01 initial source and coverage inventory

Base is accepted R04 commit 1d6b3b3ee3ae5e3fa24c90768ef805125969c41f. All 5,934 archive files matched canonical base.tar before edits; its SHA256 is 02b3c21f62c39b01f6adc2fb844b2babc8bcff1919650f25d591a17665c1d99d. Initial RED adds B01BinaryContractTests, B01Golden, and the independent Python generator only. Production is unchanged pending actual RED.

Read the binding brief, design inventory/review, post-S03 readiness review, SER-01 through SER-07, full BinaryFormat/BinaryProtocols/VectorSerialization boundaries and both existing serialization suites. Source confirms zero is the only header-domain change needed. The existing default DynamicVector and dropLast() of a singleton both have scalarCount zero. Binary encoding emits their existing 16-byte frame, while header validation rejects zero before size/CRC. Nonzero direct slices use buffer-relative reads; no slice defect is inferred.

The generator explicitly packs UInt32/UInt16 little-endian integers, compares zlib.crc32 with a separate bit-at-a-time reflected CRC, and verifies the standard 123456789 check value CBF43926. It emits three complete Swift frame tables: empty, ordinary three-word, and sixteen pathological FP32 words. Expectations never call Core writers or CRC.

A complete Sources/Tests identifier search found the old SerializationTestDataGenerator, SerializationValidator, measureSerializationMetrics, SerializationMetrics, SerializationPattern, CorruptionType and TestFixtures definitions only in SerializationEdgeCaseTests.swift, with no callers outside their own unused helper definitions. Their removal will not remove an active supported test oracle.

## Approved original stub disposition plan

| Original active comment-only name | Supported replacement or explicit unsupported removal |
| --- | --- |
| testJSONToBinaryConversion | Actual JSON→Dynamic→binary round trip |
| testFormatDetectionAndAutoSelection | Remove; automatic detection/selection is not implemented |
| testMixedFormatBatchProcessing | Explicit per-element binary/JSON choice and round trips, no automatic detection |
| testDynamicVectorSerialization | Concrete Dynamic JSON and binary, including actual empty values |
| testQuantizedVectorSerialization | Concrete INT8 JSON raw codes/parameters and dequantized FP32 binary destinations at all three widths |
| testMixedPrecisionVectorSerialization | Explicit FP16→FP32 conversion before binary, all three widths |
| testVersionMigrationHandling | Remove migration claim; version rejection remains an executable framing control |
| testResourceExhaustionHandling | Remove OOM/descriptor/timeout claim; bounded dimension-cap/tiny-header rejection is executable |
| testStreamingSerializationPerformance | Remove; streaming/performance not implemented or measured |
| testCompressionEffectiveness | Remove; compressed storage format not implemented |
| testCacheFriendlySerialization | Remove; cache optimization/performance not measured |
| testBatchSerializationOptimization | Remove; optimization/performance not measured |
| testIOSMacOSCompatibility | Remove; no platform emulator/cross-platform run |
| testLinuxCompatibility | Remove; no Linux execution |
| test32BitVs64BitCompatibility | Remove; no 32-bit execution |

The sixteen real SerializationSuite functions remain active; deterministic data/bit checks and typed failures can strengthen them. Concrete optimized tests must instantiate the four optimized structs, not generic aliases or encode extracted arrays. Old broad/misleading edge names will be narrowed to their actual schema, formatting, bounded shape, framing or concurrent-value controls.

Root authorized one additional comment-only path: VectorShapeContractTests.swift's stale binary-zero comment. Keep its count>0 guard and all behavior unchanged; refer to B01BinaryContractTests.emptyDynamicRoundTrip for empty coverage. No other S01/S03/R04 production/test migration is authorized.

SER-03 remains unoptimized and unmeasured. The consumer-owned BinaryCodable adapter disposition does not add built-in protocol adoption or change the distinct VectorType checked default's positive domain.
