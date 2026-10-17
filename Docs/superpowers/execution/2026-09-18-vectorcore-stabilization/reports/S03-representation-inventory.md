# S03 representation and independent case inventory

Base `813255cb389891590db172480667bb47d83fbdd1`; entire dispatch archive verified equal to all 5,842 base regular files before edits. The initial brief hash is `5ec57e87db1b00f85fe82167c214411b4f3e0627514d9c199c83bc1eb2273c1c`. Root's subsequent correction makes direct NormalizationHint<INT8> non-applicable: built-in INT8 types inherit VectorProtocol through QuantizedVectorINT8, but do not conform to IndexableVector. No conformance or fake wrapper is added.

| Representation | Existing wire and invariant | New maintained matrix |
| --- | --- | --- |
| Generic Vector | keyed elements, exact static count | 17 widths: custom zero plus every built-in 2/3/4/8/16/32/64/128/256/384/512/768/1024/1536/2048/3072; 17 valid, 33 invalid count cases |
| DynamicVector | keyed dimension/elements, equal count; zero valid | 3 valid lengths 0/1/3; 5 mismatches including Int.min, -1, Int.max and empty/nonempty |
| Optimized Float | single Float array, exact scalar width | 384/512/768/1536: 4 valid, 8 short/long cases |
| INT8 | keyed storage/quantizationParams; 128/192/384 SIMD4 lanes | 9 empty/short/long lane-count cases; 12 ordinary SIMD lane-shape/Int8-range failures |
| LinearQuantizationParams | five stored keys, valid Q01 tuple | 31 independent cases in each of four contexts (standalone plus 3 INT8 widths): 6 valid, 13 invalid-domain, 5 missing keys, 5 wrong types, 2 Int32 overflow = 124 executions |
| NormalizationHint | authoritative nested vector; ignore advisory keys | generic Dim3, optimized512, Dynamic: four advisory variants each; 12 valid/cleared hints and 12 malformed nested vectors |
| TopKResult | indices/distances keys and equal counts | 4 exact mismatch cases; 3 valid empty/finite/nonfinite cases with arbitrary labels and full adapter checks; 6 ordinary decoder key/type/numeric errors |
| SearchResult/SearchResults | nested records, optional score/time | base-generated collection fixture plus single-record round trip; TopK adapter preserves every pair through Collection/toTuples/mapIDs |

The parameter valid cases retain arbitrary legacy scale/bounds, both Int32 zero-point extremes in asymmetric tuples, positive subnormal scale with equal bounds, finite greatest scale whose later dequantization can overflow, and symmetric zero offset. Invalid tables independently enumerate zero/negative/NaN/both-infinite scales, each nonfinite bound, reversed bounds, symmetric nonzero offset and each malformed key. No production validity predicate supplies the expected verdict. Numeric tuple fields and raw codes are checked without dequantizing to impose a new metric constraint.

The four base-generated sorted-key fixtures cover TopK, SearchResults, DynamicVector, and INT8. Their exact generator source, unchanged production module and actual successful generation marker remain in the RED archive. JSON key order is not generalized into a wire contract: exact fixture comparison uses the explicitly selected sorted-key encoder; arbitrary parameter values use typed field-bit and key-set checks.

## Bounded non-Codable inventory

Source declarations and repository-wide conformance search show `MixedPrecisionKernels.Vector512FP16`, `Vector768FP16`, and `Vector1536FP16` are Sendable wrappers, not Codable/VectorProtocol. SoA<Vector>, SoAFP16<VectorType>, SoAINT8<VectorType>, and SoALayout likewise have no Codable conformance. Storage wrappers under Sources/VectorCore/Storage (ArrayStorage, dimension-specific Storage2...Storage3072, DimensionStorage, HybridStorage, ManagedStorage/StorageBuffer in UnifiedStorage.swift, AlignedValueStorage, aligned dynamic storage, and COWDynamicStorage) expose storage/buffer behavior without Codable conformance. Generic Vector serializing scalar elements does not confer serialization on these wrappers. This is the named representation inventory, not a claim that every library Codable type is tested.

## Existing focused controls

Selected actual symbols: TopKSelectionTests, TopKNaNContractTests, SerializationSuite, SerializationEdgeCaseTests/JSONSerializationTests, SerializationEdgeCaseTests/VectorTypeSerializationTests, VectorShapeContractTests/int8DecodingValidatesShapeAndPreservesWire, QuantizationRangeContractTests/testParameterDecodeMatrix, and NormalizationHintMutationTests. No deliberate-trap parent matrices or ASan are needed for this recoverable decoder task.

S01's construction/shape and Q01's tuple invariants are regression prerequisites, not new closure credit. I01's ignored advisory metadata behavior is preserved. Only the TopK decoder changes production. No binary format, score policy, metadata plausibility rule, finite-score restriction, uniqueness/order rule or new serialization support is added.
