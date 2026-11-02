# Component 08 — Serialization and input boundaries

Revision reviewed: `8b02e75bca7055ae11170873d15c06f1538709b7`.

## Scope and evidence

Read all of `Vectors/VectorSerialization.swift`, `Serialization/BinaryProtocols.swift`, `Utilities/BinaryFormat.swift`, `Utilities/CRC32.swift`, `Types/VectorData.swift`, `Quantization/QuantizationSchemes.swift`; reviewed constructors and Codable sections in `Vector.swift`, `DynamicVector.swift`, and all four optimized FP32 vectors; reviewed inherited Codable and storage/conversion implementations in `Operations/Kernels/QuantizedKernels.swift`, the Codable requirement in `Protocols/VectorProtocol.swift`, and Base64 wrappers in `Operations/VectorMath.swift:525–555`. Also inspected Codable implementations in `TopKSelection.swift` and `Integration/IndexableVector.swift` for coordination, rather than duplicating their component reviews.

Read `VectorSerializationTests.swift`, `SerializationEdgeCaseTests.swift`, relevant serialization/parameter tests in `QuantizedKernelsTests.swift` and `QuantizationTests.swift`, plus Package Boundaries, API Overview Map, Numerical Behavior and relevant Memory Alignment documentation.

Evidence here is source inspection and existing tests. No production/test files changed, no shared builds, crash probes, or malformed-input experiments. Root reported full Debug suite passed and owns Release validation. No new timing or allocation measurement was conducted.

## Prioritized ordinary-format finding

### P2 candidate — A valid empty DynamicVector cannot round-trip through its own binary encoder

- Category: correctness / API consistency.
- Locations: `Sources/VectorCore/Utilities/BinaryFormat.swift:164`; `Sources/VectorCore/Vectors/VectorSerialization.swift:61`; `Sources/VectorCore/Vectors/DynamicVector.swift:49`.
- Valid trigger: create `DynamicVector()` (explicitly documented as a zero-dimensional vector), call `encodeBinary()`, then `DynamicVector.decodeBinary(from:)` on the returned bytes.
- Direct derivation: encoding writes dimension 0 and a valid CRC into a 16-byte record; the decoder rejects every dimension 0 before inspecting the CRC. Codable JSON allows the same valid vector because dimension 0 matches an empty elements array. `dropLast()` on a one-element dynamic vector also produces this state.
- Impact: successful serialization is not sufficient to establish that the library can recover a supported vector value. This affects generic save/load paths containing empty dynamic vectors, including default values and empty batch results.
- Existing coverage hides the issue: `Tests/ComprehensiveTests/SerializationEdgeCaseTests.swift:22–25` names an “Empty vector serialization” test but substitutes a one-element vector and incorrectly comments that VectorCore does not support dimension 0. No runtime reproduction claimed; the contradiction follows directly from adjacent checks and constructor code.
- Recommended remedy: decide the binary format's empty-vector contract explicitly; preferably accept an exact 16-byte CRC-valid dimension-0 record for DynamicVector while preserving fixed-dimension validation. If empty values must be unsupported by this format, expose a checked encoding API and document its supported dimensions. Add real empty JSON and binary round-trip tests, including default construction and removal of the last element.
- Confidence: high. The >10,000 asymmetry below is already deliberately covered, so do not conflate that with a newly discovered decoder size-limit defect.

## Evidence-backed quality and performance opportunities

### P3 — Serialization coverage overstates implemented behavior

`Tests/ComprehensiveTests/SerializationEdgeCaseTests.swift` contains 15 comment-only `@Test` functions (mechanically counted from source): JSON-to-binary conversion at 611; format detection at 619; mixed-format processing at 627; dynamic/quantized/mixed-precision serialization at 707/715/723; migration/resource exhaustion at 763/771; streaming/compression/cache-friendly/batch performance at 823/831/839/847; platform compatibility at 861/869/877. Their successful completion establishes none of those behaviors. `testVector512OptimizedSerialization` at 641 actually constructs `Vector<Dim512>`, then encodes an extracted `[Float]` rather than the vector's Codable implementation; the analogous 768/1536 tests do the same. These should become real targeted tests or clearly separated future-test notes. This is most relevant to interpreting a green comprehensive suite, rather than evidence that Core should own migration, streaming, or compression.

Tests use round-trip comparisons but provide no populated golden fixtures (`TestFixtures` at the end contains empty Data values/arrays). Add independently specified complete bytes for a small vector, including checksum; test bit patterns for signed zero and NaN payloads rather than only approximate float equality. Keep actual optimized Codable round trips separate from generic vectors and raw array encoding.

### Source-derived allocation opportunity — scalar-at-a-time binary conversion

`BinaryFormat.readFloatArray` (`:109–124`) creates a Float array and calls `readFloat` for every scalar. The local `loadUnaligned` implementation (`:204–225`) explicitly allocates aligned scratch memory, copies one value and deallocates it. This duplicates functionality available in the Swift toolchain and creates a source-level allocation/copy operation per scalar if that local path survives optimization. The optimizer may eliminate allocations, so this review does **not** claim measured Release heap allocations or a throughput regression.

Encoding (`VectorSerialization.swift:71`) materializes `toArray()` and appends each Float individually. Decoding materializes a Float array before constructing vector storage. These are O(n) scans/copies in addition to CRC and final vector storage; encoding a quantized vector additionally dequantizes through its `toArray()` route. Candidate improvement: scoped buffer access, bulk payload transfer on supported little-endian platforms, and a bounded unaligned read implementation using the toolchain facility or stack storage. Verify generated Release code/allocations and benchmark before asserting a speedup. Preserve exact bit patterns and unaligned Data support.

### Representation and API clarity opportunities

- Binary encoding is available for every Float `VectorProtocol`, including optimized and quantized types. Decoding is provided only on generic `Vector<D>` and `DynamicVector`. Cross-type FP32 value interchange works by design, but same-concrete-type round trips are not offered for optimized types.
- INT8 types inherit the generic `encodeBinary()` implementation, which writes **dequantized Float32 values**, with size `16 + 4*n`; it does not preserve Int8 codes, quantization parameters, or 4x compression. Their synthesized JSON Codable representation instead preserves storage and parameters. This is a distinction to document rather than a proven violation of a compressed-binary contract: no such contract is declared.
- `BinaryEncodable` / `BinaryDecodable` are public protocols, but no production vector explicitly adopts them (only the doc example does). A concrete vector having identically named methods does not satisfy a generic `T: BinaryCodable` API. Consider adding appropriate conformances or documenting the protocols as extension points. Source search found no internal callers relying on these conformances, so this is an API integration opportunity rather than an established downstream break.
- `BinaryHeader` describes the magic as ASCII “VECT” (`BinaryProtocols.swift:46,51`), but little-endian writing of `0x56454354` produces bytes `54 43 45 56`, ASCII “TCEV”. `testEndiannessHandling` explicitly tests those actual bytes. Keep the existing wire format; clarify numeric magic versus actual byte sequence, and document all header fields as little-endian for independent readers.

## Documented or directly evident limitations, not new findings

- Decoder accepts dimensions 1 through 10,000 only, while encoder permits larger valid dynamic vectors. `testMaximumDimensionVectorSerialization` explicitly expects successful encoding and rejected decoding at 10,001. This is an existing deliberate test contract, although the public encode/decode docs should expose the asymmetry and cap more clearly.
- The format is version 1, uncompressed, exactly one whole vector per Data. There is no actual optional compression mode, stream reader, chunk framing, batch serialization API, or migration implementation in this component. Reserved flags are written as zero and read but unused. No compression bug was found. Future flag semantics must either reject unsupported flags or use a version change before assigning incompatible meanings.
- Generic vector JSON is `{ "elements": [...] }`; DynamicVector JSON adds `dimension`; optimized FP32 vectors encode as bare arrays; INT8 types inherit synthesized structured Codable. These representations round-trip within their own type, but are not interchangeable JSON schemas. No promise of a universal JSON schema was found.
- `VectorData` uses synthesized Codable for its SIMD value, ID, metadata and timestamp. It does not add vector binary framing or a separate compressed representation. Its own initializer does not promise extra semantic validation; no new serialization defect was identified there.
- Binary public decode validates header size, magic, version, dimension cap and exact total length **before** payload allocation; then checks CRC and fixed dimension. Size products reached through that route are bounded. Private negative-offset or overflow hypotheticals do not establish a public decoder exploit and are not findings here.
- Binary reads use the Data buffer's byte offsets, and CRC uses `prefix` iteration; neither relies on Data collection indices being zero. The test rewraps its unaligned slice in Data, so a direct nonzero-startIndex slice test would strengthen coverage, but no slice-indexing defect was found by inspection.
- CRC covers header and payload and excludes its own trailing UInt32, consistently on write and read. It is an integrity checksum, not an authentication feature. Existing tests separately exercise header corruption, payload tampering, checksum tampering, truncation, dimension mismatch and valid round trips. No new CRC implementation defect found.

## Boundary observations for aggregation only

Root requested no further malformed-input/crash probes and already has boundary/quantizer findings. Do not inflate the final issue count by splitting this existing class:

- `QuantizedVectorINT8` inherits Codable through `VectorProtocol`. The three INT8 structs synthesize decoding of raw `storage` and `quantizationParams`, unlike the checked FP32 array decoders. Their raw storage constructors at `QuantizedKernels.swift:175,188,201` also omit length guards; `toFP32` at `:148–161` indexes `Self.laneCount` entries. Any class-wide storage-invariant remedy should cover decoded construction as well as direct initializers and setters, and add malformed length tests for all three types.
- `LinearQuantizationParams` synthesizes Codable over derived scale and zero point. This bypasses the calibration constructor and permits inconsistent derived fields. Include decoded scale/range/zeroPoint consistency in the existing quantization-boundary remedy; no additional runtime failure is claimed here.
- `TopKResult.init(from:)` at `TopKSelection.swift:651` decodes its two arrays independently with no matching-count check. Forwarded for the result/selection component to assess against its constructors and collection operations.

## Strengths

The single BinaryFormat implementation gives generic and dynamic decoders a consistent version/size/CRC path. Float binary storage uses bit patterns with explicit endianness, avoiding decimal conversion loss. Fixed FP32 Codable decoding routes through the existing checked array constructors for all four optimized types and generic vectors. Dynamic JSON explicitly compares declared dimension with actual count. Base64 simply wraps the same checked binary decoder. Current whole-record framing has bounded decode allocation and rejects trailing bytes rather than silently accepting partial records.

## Limitations of this review

No new round-trip execution, golden-vector reference generation, Release allocation profiling, alternate-platform verification or external consumer compile was performed in this component. Existing test names are cited only for what their bodies actually exercise. Root's green suite provides broad regression evidence, not proof of the unimplemented placeholder behaviors or all possible encodings.
