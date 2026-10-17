# Vector serialization contracts

Version1 uses numeric magic `0x56454354`, whose little-endian wire bytes are `54 43 45 56`. They are not the ASCII byte order of VECT. Every integer and FP32 word is little-endian.

| Offset | Field |
| --- | --- |
| 0 | UInt32 magic |
| 4 | UInt16 version, currently 1 |
| 6 | UInt32 dimension N |
| 10 | UInt16 reserved flags |
| 12 | N UInt32 Float bit patterns |
| 12 + 4N | UInt32 CRC32 |

Total length is 16 + 4N. CRC32 covers the header and payload, using reflected polynomial `0xEDB88320`, initial value and final xor `0xffffffff`. It detects corruption, not authentication. Version1 encoders emit zero flags; decoders ignore reserved bits when the complete frame and CRC are valid.

Independent full frames from `Scripts/ci/fixtures/B01_generate_goldens.py`:

* Empty: `54 43 45 56 01 00 00 00 00 00 00 00 74 2c 73 be`
* `[1, -2.5, +0]`: `54 43 45 56 01 00 03 00 00 00 00 00 00 00 80 3f 00 00 20 c0 00 00 00 00 ed 72 75 5e`

The maintained generator packs integer words directly, cross-checks zlib against a bit-at-a-time CRC and the `123456789` → `CBF43926` control. `B01Golden.swift` also contains a complete special-bit frame. `B01BinaryContractTests` compares every encoded byte and decoded stored word, including signed zeros, infinities, quiet/signaling NaN payloads and finite extremes. This is binary storage preservation, not arithmetic accuracy or JSON NaN-payload preservation.

Validation order is minimum 16 bytes, magic, version, dimension cap, reserved flags read, exact length, CRC, destination dimension, payload. Trailing bytes and concatenated frames fail with `insufficientData`, as do short frames. Only version1 is accepted. Decode supports N in 0...10000; an empty DynamicVector is an ordinary 16-byte frame. Positive fixed dimensions reject that valid empty frame after CRC validation. Empty Data remains invalid. The nonthrowing built-in encoder still emits dimension10001, but decode rejects it. It does not promise checked arbitrary-count encoding: BinaryHeader retains its representable nonnegative-dimension precondition. Tiny oversized-header tests never allocate a giant payload.

## Concrete representations

| Concrete type | JSON | Binary route |
| --- | --- | --- |
| Vector<D> | Object with `elements` | Encode and same-type decode; exact D required |
| DynamicVector | Object with `dimension` and `elements` | Encode and decode, including default/explicit/removed-last empty |
| Optimized FP32 384/512/768/1536 | Concrete array | Common FP32 encoding; decode explicitly as DynamicVector or matching Vector<D> |
| INT8 512/768/1536 | Validated `storage` and `quantizationParams` | Dequantized FP32 coordinates, 16+4N bytes; no compressed-code preservation |
| Raw FP16 512/768/1536 | No direct Codable API | Explicit `toFP32().encodeBinary()` serializes already-rounded values |

The FP16 route cannot recover pre-rounding FP32 or promise arbitrary FP16 NaN-payload preservation. `SerializationEdgeCaseTests` instantiates each concrete optimized/INT8 type and all three FP16 conversion routes. JSON's default nonfinite rejection and matching explicit string strategies are separate tests; string strategies preserve classification, not NaN bits. The generic fixed object schema accepts `elements`; an old bare array is rejected. These are explicit destination choices, not automatic format detection.

## Protocol adoption is consumer-owned

Built-in methods do not imply `BinaryEncodable`, `BinaryDecodable`, or `BinaryCodable` conformance. Optimized/INT8 methods are encode-only; no same-type binary decoder is added. A complete adapter is:

```swift
import Foundation
import VectorCore

struct StoredVector: BinaryCodable {
    let value: DynamicVector
    func encodeBinary() throws -> Data { value.encodeBinary() }
    static func decodeBinary(from data: Data) throws -> StoredVector {
        StoredVector(value: try DynamicVector.decodeBinary(from: data))
    }
}
```

`B01PublicConsumer.swift` exercises these explicit witnesses through concrete, generic and existential encoding and generic decoding. Separate negative typechecks verify bare fixed/dynamic/optimized/INT8 types do not meet the protocol constraints. The distinct throwing default on `VectorType where Self: BinaryEncodable` still checks 1...10000. It must not be confused with the adapter's nonthrowing built-in delegation and empty-value contract. Future built-in adoption needs its own witness/cap decision; no downstream generic regression is alleged.

## Test inventory and unsupported work

The 16 real SerializationSuite tests remain. Two misleading names are narrowed: InvalidUTF8String → NonJSONText and RandomizedFuzzSmall → DeterministicWordsSmall. The scoped lexical inventory checks registered bodies and required golden tables, with empty-body/empty-golden mutation controls. This is not proof of arbitrary test semantics.

The former 15 comment-only registrations have the following dispositions:

| Old name | Disposition |
| --- | --- |
| testJSONToBinaryConversion | Real JSON→Dynamic→binary golden test |
| testFormatDetectionAndAutoSelection | Removed: automatic detection/selection unsupported |
| testMixedFormatBatchProcessing | Replaced by explicit mixed-format round trips |
| testDynamicVectorSerialization | Real concrete JSON/binary test including empty |
| testQuantizedVectorSerialization | Real three-width concrete JSON/FP32 binary test |
| testMixedPrecisionVectorSerialization | Replaced by explicit three-width FP16→FP32 test |
| testVersionMigrationHandling | Removed: no migration engine; version rejection tested |
| testResourceExhaustionHandling | Removed: no OOM/descriptor guarantee; bounded caps/header bombs tested |
| testStreamingSerializationPerformance | Removed: streaming/performance unsupported/unmeasured |
| testCompressionEffectiveness | Removed: no compressed format |
| testCacheFriendlySerialization | Removed: no measured cache optimization |
| testBatchSerializationOptimization | Removed: no measured batch optimization |
| testIOSMacOSCompatibility | Removed: no platform emulator or cross-platform execution |
| testLinuxCompatibility | Removed: no Linux execution in this work |
| test32BitVs64BitCompatibility | Removed: no 32-bit execution in this work |

Old broad escape/nesting/compatibility tests are replaced by explicit schema, whitespace, nested-vector-array, and fixture-specific formatting controls. Pretty-JSON size observations apply only to the named fixture, not universal compression. No compression, streaming, platform emulation, format selection or production performance target is introduced. SER-03 remains an unoptimized, unmeasured profiling opportunity requiring accepted V04 and separate authorization before scalar-loop changes or speed claims.
