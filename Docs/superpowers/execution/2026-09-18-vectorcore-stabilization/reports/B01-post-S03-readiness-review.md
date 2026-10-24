# B01 post-S03 dispatch readiness — 2026-09-29

**READY within the approved B01 scope; no S03-driven brief correction required.** Read-only check at integration `ccc432c92e18981ab02de4657ab14d31adb55867`, with production S03 commit `e0e0e74b5128419482d1e1993d58c2184adc01ba`. B01 brief SHA256 matches `bf6d4f1bfc605d19d0b90d6b4ee24929c1fee66d10bdabd75b08e343a7635a44`. Read the approved brief and prior inventory/review, inspected S03's production/test/fixture diff and current serialization test bodies/declarations. No build, runtime re-audit or new serialization-defect investigation was performed.

## Write ownership and preserved contracts

B01's production edit remains the bounded zero-dimension acceptance in `Sources/VectorCore/Utilities/BinaryFormat.swift`, plus directly necessary empty handling only. Documentation ownership remains `Sources/VectorCore/Serialization/BinaryProtocols.swift`, `Sources/VectorCore/Vectors/VectorSerialization.swift`, `Sources/VectorCore/Protocols/CoreProtocols.swift`, and a focused serialization guide. Existing scalar/CRC/unaligned arithmetic, JSON production decoders, vector/quantization storage, encoder signatures and cap remain outside behavior edits absent separately approved evidence.

The existing writable test suites are `Tests/ComprehensiveTests/SerializationEdgeCaseTests.swift` and `Tests/ComprehensiveTests/VectorSerializationTests.swift`; new B01 fixtures, independent generator, mechanical inventory and external consumer drivers can be added under the approved scope. Preserve the 16 real `SerializationSuite` tests, strengthening their assertions where specified. Accepted S03's `DecodedInvariantTests.swift`, four `Scripts/ci/fixtures/S03-*.json` fixtures and `TopKSelection.swift` are regression inputs, not required B01 edit targets. No B01 change is needed in R04's heap/selection ownership or V02's benchmark ownership.

S03 changes only TopK's JSON decoder production behavior: equal parallel-array counts with a keyed dataCorrupted error, then assignment. Its generic/optimized/Dynamic/INT8/parameter/hint tests retain their existing schemas and error families. B01's version-1 binary empty-frame repair does not conflict with those JSON boundaries. A valid empty Dynamic JSON object already succeeds; B01 adds the missing corresponding binary acceptance without making empty Data valid. Positive fixed dimensions must continue to reject a valid empty frame after normal framing/CRC validation.

S03's concrete JSON tests overlap intentionally with B01's supported-representation controls; B01 additionally establishes actual binary source/destination behavior and bit-preserving framing. Do not change TopK label/score/metadata policy, valid quantization tuples, advisory-hint reset semantics, or introduce built-in BinaryCodable adoption. Direct INT8 NormalizationHint remains unavailable; use standalone/nested INT8 decoding as S03 already does. The independent binary goldens must be new expectations, not generated from S03's JSON fixtures or Core writers.

## The 15-stub inventory remains current

The two existing serialization files and `DecodedInvariantTests.swift` are byte-identical to their accepted S03-commit versions. A bounded function-body scan with comments removed, followed by inspection of the registered bodies, still finds exactly these 15 empty tests in SerializationEdgeCaseTests.swift:

| Current line | Function |
| --- | --- |
| 611 | testJSONToBinaryConversion |
| 619 | testFormatDetectionAndAutoSelection |
| 627 | testMixedFormatBatchProcessing |
| 707 | testDynamicVectorSerialization |
| 715 | testQuantizedVectorSerialization |
| 723 | testMixedPrecisionVectorSerialization |
| 763 | testVersionMigrationHandling |
| 771 | testResourceExhaustionHandling |
| 823 | testStreamingSerializationPerformance |
| 831 | testCompressionEffectiveness |
| 839 | testCacheFriendlySerialization |
| 847 | testBatchSerializationOptimization |
| 861 | testIOSMacOSCompatibility |
| 869 | testLinuxCompatibility |
| 877 | test32BitVs64BitCompatibility |

S03's accepted run explicitly excluded the three vector-type stubs from exercised-contract credit; it did not implement or remove them. All 15 still need B01's old-name→actual replacement/documented removal table and maintained mechanical guard. The three optimized-named real tests still instantiate generic Vector and encode arrays; B01's planned concrete-type correction remains necessary. Their current declaration lines are 641/660/679 (the older prose inventory's approximate 662/685 anchors do not alter the identified tests).

## Exact existing focused-test identities

Use source test identities, not filenames or only human display labels:

| Purpose | Actual filter identity |
| --- | --- |
| All 16 ordinary serialization functions | `SerializationSuite` (display name “Serialization”; file VectorSerializationTests.swift) |
| All existing edge suites and stubs to replace | `SerializationEdgeCaseTests` (display “Serialization Edge Cases”) |
| All six accepted S03 functions | `DecodedInvariantTests` |
| S01 shape suite | `VectorShapeContractTests` |
| Q01 existing parameter Codable XCTest | `QuantizationRangeContractTests/testParameterDecodeMatrix` |
| Existing TopK Codable control | `TopKSelectionSuite/testTopKResult_Codable` |
| Existing advisory hint decode control | `NormalizationHintMutationTests/decodedMetadataIsAdvisory` |

A concrete source-derived existing-suite filter is:

```text
SerializationSuite|SerializationEdgeCaseTests|DecodedInvariantTests|VectorShapeContractTests|QuantizationRangeContractTests/testParameterDecodeMatrix|TopKSelectionSuite/testTopKResult_Codable|NormalizationHintMutationTests/decodedMetadataIsAdvisory
```

Add the actual names of new B01 suites after implementation; do not substitute a nonexistent B01 name now. Run this focused matrix in Debug and Release with the brief's unique scratches/job bound. `VectorShapeContractTests` includes its child dispatcher and actual bounded shape children, so retain the resulting child evidence and separate parent no-op counts if using the entire suite. The individual Q01 parameter method avoids accidentally scheduling unrelated Q01 trap/calibration children. The TopK Codable method avoids expanding this serialization check into all R04 selection behavior. `DecodedInvariantTests` supplies both TopK mismatch and valid-adapter cases, all-width JSON shape/parameter checks and legacy fixtures.

Within the edge suite, actual nested source names are `BinarySerializationTests`, `JSONSerializationTests`, `CrossFormatCompatibilityTests`, `VectorTypeSerializationTests`, `ErrorRecoveryTests`, `SerializationPerformanceTests`, and `PlatformCompatibilityTests`. Selecting the outer type covers them; their display names are not separate filenames. Actual post-change discovery/logs must confirm intended function and parameterized-case execution rather than treating this source-derived filter as runtime proof.

ASan remains a separately selected valid-slice/bounded-malformed B01 matrix, not an automatic rerun of intentional shape trap children. External positive/negative Swift6 adapter fixtures, generator reproduction and the empty-body/golden mutation checks are separate acceptance commands, not covered merely by a Swift suite filter. Existing S03 tests may serve as unchanged regression coverage but do not replace B01's actual empty-frame RED, independent full binary goldens or precise later-stage error assertions.

No source, test, brief, state, index or commit changes were made. Only this readiness report and its canonical copy were written.
