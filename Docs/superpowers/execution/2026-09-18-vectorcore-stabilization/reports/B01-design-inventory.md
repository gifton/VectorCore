# B01 design inventory

Read-only inventory of WORK_ITEMS B01, canonical SER-01–07, BinaryFormat/BinaryProtocols/VectorSerialization, the complete SerializationEdgeCaseTests and regular VectorSerializationTests, concrete representation helpers and callers. No implementation/optimization or contract choice is made. S01/S03 are prerequisites; V04 is conditional for scalar-loop optimization or speed/allocation claims.

## Existing public contracts and reachable paths

VectorSerialization.swift:60 supplies nonthrowing encodeBinary to every VectorProtocol with Scalar == Float. It materializes toArray, writes a12-byte header, appends each Float bit pattern little-endian and appends CRC32. Exact frame length is16+4N. Numeric magic0x56454354 is emitted as bytes54 43 45 56, not ASCII VECT; version1 is UInt16 little-endian, dimension UInt32 at offset6, flags UInt16 at10 (encoder zero), payload at12, CRC last4 bytes little-endian. CRC32.swift uses reflected polynomial0xEDB88320, initial/final xor0xffffffff over the complete header+payload. CRC detects corruption, not authenticity.

BinaryFormat.validateHeader:135–183 checks minimum16, magic, version, dimension1...10000, then exact total length; flags are returned but ignored. Public decoders then check CRC. Vector<D> additionally checks D.value before reading payload; DynamicVector allocates the decoded array. Oversize/trailing frames are rejected, not streamed/concatenated; the exact-size error uses insufficientData even for too many bytes. Header checks therefore precede CRC and fixed-dimension mismatch. Preserve this ordering unless explicitly changed. Zero-length Data is invalid independently of a valid zero-dimensional16-byte frame.

The nonthrowing encoder does not enforce the decode cap. Dimension10001 encodes but decode rejects, explicitly expected by the existing test. BinaryHeader.init(dimension:) directly converts Int to UInt32; generic encoder size arithmetic also assumes representable valid counts. Do not remove the10000 cap as an incidental empty fix or advertise checked arbitrary-count encoding. Internal scalar read helpers accept offsets/counts without full negative/overflow guards, but public validated decoding supplies bounded nonnegative values; no public overflow exploit is inferred. validateChecksum likewise relies on header validation first.

BinaryFormat reads use raw-buffer-relative byte offsets and unaligned loads, not Data integer indices. Thus direct nonzero-startIndex Data slices must be tested without rebuilding Data(slice); an alignment label on a rebased copy does not prove this route. The custom generic unaligned-load extension allocates scratch in source, but actual concrete overload/Release optimization requires emitted-code/allocation evidence before claiming per-scalar allocations (SER-03).

Concrete matrix:

| Type | JSON | Binary | Base64 convenience |
|---|---|---|---|
| Vector<D> | keyed object `elements`; initializer validates D | encodes and same-type decodes | both, VectorMath.swift:535–552; wraps the binary frame |
| DynamicVector | keyed `dimension` + `elements`; counts must agree; empty supported | encodes; same-type decoder currently rejects dimension0 | none found |
| Vector384/512/768/1536Optimized | single-value Float array in each concrete Codable extension | common FP32 encoder; no same-type decoder | none found |
| Vector512/768/1536INT8 | keyed storage + quantizationParams, explicit validated Codable after S01/Q01 | common encoder emits dequantized Float values,16+4N; no same-type decoder | none found |
| MixedPrecisionKernels.Vector512/768/1536FP16 | Sendable storage structs, no Codable/VectorProtocol conformance | none directly; explicit toFP32() then common FP32 encoding possible | none directly |

INT8 reports Scalar=Float and toArray delegates through toFP32 (QuantizedKernels.swift:309–510). Binary interchange loses INT8 storage/parameter identity; JSON retains that schema and now validates parameter/lane constraints. Do not describe current JSON as merely synthesized or a new compressed binary format. FP16-to-FP32 serialization preserves the already-converted Float values, not original FP32 pre-rounding values or necessarily arbitrary FP16 NaN payloads. JSON nonfinite strategy is caller-configurable; default Foundation rejection and explicit string classification must be distinct from binary bit preservation. sortedKeys affects key order, not numerical precision.

No actual BinaryEncodable/BinaryDecodable adoption is declared by these types. Nonthrowing methods can potentially witness a throwing protocol requirement, but declaration/witness selection must be explicit. **Additional reachable default:** CoreProtocols.swift:163–196 defines another throwing encodeBinary on VectorType where Self: BinaryEncodable, validating1...10000 before writing the same layout. Vector and DynamicVector are VectorType. Adding conformances can expose competing defaults and change empty/cap dispatch unless root deliberately resolves the witness. This file is outside the initial B01 source list and needs narrow authorization if modified. Generic/existential/concrete external consumers must exercise the chosen witness, not merely compile a method-name call. Consumer-defined conformers using the existing checked default are a separate compatibility concern.

Production binary callers are the fixed-vector Base64 wrapper and the two decoders; CoreProtocols has a VectorSerializable documentation example and the conditional default above. Searches found no internal generic BinaryCodable consumer or benchmark binary runner. Tests supply the principal in-tree direct calls. No new optimized/INT8 decoder is necessary merely to explain encoding into DynamicVector or generic Vector<D>; adding same-type decoding would be a separate root API choice.

## Exactly15 active comment-only tests

Mechanical comment-stripped body inventory in SerializationEdgeCaseTests.swift:

| Line | Function |
|---|---|
|611|testJSONToBinaryConversion|
|619|testFormatDetectionAndAutoSelection|
|627|testMixedFormatBatchProcessing|
|707|testDynamicVectorSerialization|
|715|testQuantizedVectorSerialization|
|723|testMixedPrecisionVectorSerialization|
|763|testVersionMigrationHandling|
|771|testResourceExhaustionHandling|
|823|testStreamingSerializationPerformance|
|831|testCompressionEffectiveness|
|839|testCacheFriendlySerialization|
|847|testBatchSerializationOptimization|
|861|testIOSMacOSCompatibility|
|869|testLinuxCompatibility|
|877|test32BitVs64BitCompatibility|

These require actual supported behavior assertions or removal from active tests into explicit future-work notes, not implementing every imaginary feature. Empty fixture Data/arrays at1114 onward are not independent goldens. measureSerializationMetrics at1050 returns constants, not measurements. The unused SerializationValidator checks only magic/version despite its broad label; compareVectors can accept NaN-vs-finite because `diff > tolerance` is false for NaN. If retained/activated it needs finite-aware predicates; otherwise do not make it the new oracle. Searches within the file found definitions, no callers for these support placeholders.

Other misleading coverage: testEmptyVectorSerialization uses one element; three optimized-named tests at641/662/685 instantiate generic Vector and JSON-encode extracted arrays; testMemoryAlignmentRequirements rebuilds Data from the slice; special-float test checks NaNs only by classification, not payload bits; fixed dimension corruption also invalidates frame length/CRC and does not isolate dimension mismatch; JSONBackwardCompatibility swallows fixed-vector decode errors and merely documents unsupported old array JSON. JSON escape/nesting/precision names overstate pretty-formatting, shallow-array and ordinary-value checks. Binary smaller-than-JSON assertions apply only to their fixtures, not a universal size property. Regular SerializationSuite has16 actual tests covering ordinary round trips, typed framing/CRC failures, Base64 and JSON; preserve those real assertions while strengthening exact bits and labels.

## Bounded choices for root

The smallest compatible empty repair candidate is to allow dimension0 through shared header framing, still require the16-byte frame and valid CRC, let DynamicVector construct[], and retain Vector<D>'s exact-dimension check. This changes no encoder bytes or cap. A destination-specific allowEmpty header parameter is another bounded option if root wants shared validation's default positive domain preserved. Checked encoding rejection instead changes the current nonthrowing API and needs explicit migration; this inventory chooses neither.

Conformance choices are likewise separate: Vector/Dynamic can be declared full roundtrip participants only after witness behavior is fixed; optimized/INT8 can be encode-only if desired. Protocol adoption need not create same-concrete decoders or new storage framing. Explicit documentation that protocols are extension points is also possible but must not imply current adoption. Decide flag acceptance/version policy explicitly; current nonzero reserved flags with recomputed valid CRC are accepted. Preserve that compatibility or document a deliberate new rejection; no hidden future flag semantics.

## Independent bounded acceptance

Generate full golden frames from explicit UInt32 payload words plus independent little-endian packing and a separate bitwise CRC implementation or Python zlib, without production writers/CRC. Include a tiny ordinary frame, actual empty frame if accepted, signed zeros, infinities, signed quiet/signaling NaN words and subnormal/extreme finite words. Compare every encoded byte and every decoded Float.bitPattern; avoid arithmetic on NaN fixtures before encoding. Fixed and dynamic decoding plus optimized/INT8-to-explicit-destination examples must name source/destination and expected16+4N length. CRC known-answer `123456789`→0xCBF43926 is a separate control, not a substitute for a full-frame oracle.

Malformed matrix: every truncation boundary, appended bytes, header-only/empty Data, wrong magic/version, cap9999/10000/10001, UInt32.max tiny bomb, invalid CRC in header/payload/checksum, valid CRC on a wrong fixed dimension, and valid nonzero flags according to the chosen policy. Recompute independent CRC when testing a later validation stage; do not let stale CRC mask the intended assertion. Use direct slices with nonzero startIndex and offsets1/2/3/5 around a valid frame, retaining exact length and payload bits. Test default DynamicVector, [] construction, and removal of its last element for the real empty contract. No OOM or huge allocation probe is necessary.

Public Swift6 generic BinaryEncodable/BinaryDecodable/BinaryCodable helpers should compile and execute selected conformances, including existential encode, concrete encode and generic roundtrip at empty/cap boundaries. Unsupported FP16/direct optimized decode should remain explicitly unsupported unless root expands scope. JSON tests should invoke concrete Codable types, roundtrip actual INT8 storage/parameters under accepted Q01/S01 rules, verify malformed lanes/parameters stay rejected, and use explicit string strategies for nonfinite values without promising NaN payload preservation. Same-source binary-vs-JSON conversions should compare the appropriate values/schema, not infer one universal representation.

SER-03 remains a qualified profiling candidate unless root elects optimization after V04. A no-algorithm B01 can document that disposition without claiming speedup. Any later bulk/scalar change requires CRC-inclusive Release encode/decode allocation/timing records, unaligned slices, independent byte equality, concrete linked emitted-code inspection and exact workload/build provenance. No streaming, compression, automatic format selection, migration engine, cross-platform runtime validation or resource-exhaustion recovery is established by the current API or this inventory.
