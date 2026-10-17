# B01 implementation and verification — final v1

B01 repairs decoding of the already-encodable empty DynamicVector frame. The only production behavior change is the shared BinaryFormat header domain from 1...10000 to 0...10000 and its diagnostic. All other production changes are documentation. Header version, cap, signatures, exact-size classification, validation order, CRC, scalar writers/readers, reserved-flag acceptance and fixed-destination checks remain unchanged. The approved VectorShapeContractTests change is comment-only.

Base is accepted commit `1d6b3b3ee3ae5e3fa24c90768ef805125969c41f`. The binding brief hash is `bf6d4f1bfc605d19d0b90d6b4ee24929c1fee66d10bdabd75b08e343a7635a44`, plus root's narrow shape-comment addendum. No source refinements followed v1. The final archive is byte-identical to its frozen inventory: 5,934 base files, 5,943 final files, 321 base code files, 329 final code files, and 16 changed/new paths. No Package, JSON production, storage, quantization, provider, S03 decoder, heap, benchmark or routing edits occurred.

Canonical evidence root is `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-02/B01`. The complete source snapshot, full diff (including new files), full inventories and source manifest are in `v1`. Diff SHA256 is `fa091f10be7c25c3c9252fa32523d9f405331bbc7b238f4ed7e1a5abd1a51b82`. Root's reconstruction/review records have their own lifecycle and are excluded from the worker index.

## Actual RED and controls

Root launched the unchanged-production RED after the worker's capture invocation remained approval-bound without a process or log. `red/command.json` records the actual completed command, exit 1, and 338,935-byte log SHA256 `bad6de79724933d986c39da5f66e9028b3c4ed730af61c0b963b3ba5f11e8963`. Default, explicit-empty and removed-last DynamicVector routes each failed with invalidDimension 0. The nonempty independent golden/direct-slice function passed. No malformed-result indexing, giant allocation or expected trap is used for this RED.

The RED source is retained as `red/complete-source.tar`, reconstructed from the immutable base plus the three exact RED test/generator files; production is unchanged. `red/source-association.json` records the hashes. The original canonical `build-red-debug` scratch retains 4,363 actual files inventoried in `red/artifacts.json`. It was never reused or rebuilt. The separate lexical RED found exactly the 15 old comment-only registrations and no empty registered body in SerializationSuite; its actual log remains `red/empty-body-guard.log`. This is a coverage finding, not a fabricated behavioral failure.

## Executable contracts and API disposition

`B01BinaryContractTests` covers the three actual empty values, complete ordinary/empty/special-bit goldens, exact stored FP32 words, both dynamic and supported fixed direct slices at offsets 1/2/3/5, every truncation of a small frame, appended and concatenated frames, corrupt magic/version/payload/checksum, reserved flags, valid-CRC destination mismatch, every supported positive fixed dimension rejecting empty, cap 9999/10000, the existing 10001 encode/decode asymmetry and a 16-byte UInt32.max header bomb. All malformed-frame assertions check the error family/kind. The pathological fixture includes signed zeros, both infinities, positive/negative quiet and signaling NaNs with distinct payloads, least subnormal/normal and maximum finite words; both tested generic/dynamic constructors retain those words. No arithmetic forms the input payload before serialization.

The maintained Python generator uses integer packing and zlib CRC, cross-checks a separate bit-at-a-time reference and `123456789` → `CBF43926`, and reproduces all three complete Swift frame tables. Core writers/CRC do not generate expectations. Root's separate CRC proof is independent review evidence, not substituted for these executed controls.

`SerializationEdgeCaseTests` now exercises concrete optimized types at all four widths, concrete INT8 JSON and explicit FP32 binary destinations at all three widths, and the three explicit FP16→FP32 serialization routes. INT8 binary preserves dequantized FP32 coordinates, not raw compressed storage. FP16 conversion preserves its already-rounded FP32 values, not pre-rounding inputs or arbitrary FP16 NaN payloads. JSON default nonfinite rejection is separate from an explicit string-strategy classification round trip. Schema, whitespace, nested vector arrays, fixture-specific size and concurrent independent-value controls replace broad unsupported/misleading assertions.

All 15 former empty names map to executable supported behavior or documented unsupported removal in `Docs/Serialization.md`. The 16 real SerializationSuite functions remain. InvalidUTF8String is narrowed to NonJSONText; RandomizedFuzzSmall becomes deterministic pathological words. Exact binary and ordinary JSON checks replace tolerance comparisons; no arbitrary thrown error is treated as sufficient. Unused metrics, validators and empty fixtures had no external callers and were removed with the old helper block.

The maintained lexical guard covers only the two serialization suites and three required complete golden tables. Its seven tests include empty-body/comment-only mutations, empty-golden mutations, string/comment brace controls and the exact sixteen-function inventory. It is explicitly not semantic proof of arbitrary test bodies.

No built-in BinaryEncodable/BinaryDecodable conformance is added. A consumer-owned BinaryCodable wrapper provides explicit witnesses delegating to DynamicVector. The separate checked VectorType default keeps its positive 1...10000 domain. Public Swift 6 fixtures prove concrete/generic/existential encoding and generic decoding, actual empty values and the cap. Four separate bare fixed/dynamic/optimized/INT8 typechecks fail specifically for missing protocol conformance. No new same-type optimized/INT8 binary decoder or downstream regression is claimed.

## Runtime evidence

All builds use unique canonical scratches and at most six jobs. No source was edited during a build. All actual binaries, modules, objects and test bundles remain in those immutable scratches, not merely as hashes. The environment is Apple Swift 6.3.3 on arm64 macOS, recorded in `v1/environment.json`. Existing unrelated compiler warnings were retained in complete logs.

| Check | Actual result |
| --- | --- |
| Focused Debug | Exit 0; 51 Swift Testing functions in 7 suites: 50 exercised plus one parent dispatcher no-op; separately 1 XCTest parameter-decoder method |
| Focused Release | Same counts, exit 0 |
| Shape children per configuration | 126 actual cases: 42 valid and 84 expected storage rejections; all captures retained |
| AddressSanitizer | Exit 0; 7 actual B01 functions in 1 suite; no dispatcher no-op or intentional trap children |
| Supplementary direct Debug/Release/ASan | 7 B01 functions each, using explicitly recorded helper and loaded bundle |
| External Swift 6 | Positive compile and execution pass; four intended missing-conformance typechecks fail as expected |
| Maintained B01 lexical/mutation suite | 7 checks pass |
| Existing child-process scanner | 7 checks pass via `python3 -m unittest Scripts.ci.tests.test_child_process_bounds -v` |
| Independent golden reproduction | Pass |

The full focused filter and exact Swift argv are recorded in `v1/debug-command.json` and `v1/release-command.json`: B01BinaryContractTests, SerializationSuite, SerializationEdgeCaseTests, DecodedInvariantTests, VectorShapeContractTests, the actual quantization parameter decoder method, TopK Codable test and advisory normalization-hint decode test. Debug log SHA256 is `083e88069e7b4112a2db61fb7d7d26603fa40eb644419766dd7e20c09ac891e6`; Release is `6c45b8887060019219eb3fbb6adcfdbbb062a7da633be23b793266703e714fa1`. ASan filters only B01BinaryContractTests; its log SHA256 is `cd6e59ca77243b583ae7bf50990f9d698b31450f18ebc773080f3dd8020fcc97`.

The existing S01 parent creates temporary capture directories. Each complete directory was copied immediately after its run into `v1/debug-children` or `v1/release-children`; the corresponding JSON records original and canonical paths plus every capture hash. The capture basename is the exact S01_SHAPE_PROBE value. The unchanged helper forwards its actual parent executable/bundle arguments, replaces the filter, installs its termination handler before launch, uses file output and a 15-second deadline plus at most 2 seconds cleanup, and throws on timeout before termination-status access. The parent childProbe return is not counted as exercised child coverage.

SwiftPM's test host is distinct from the loaded `.xctest/Contents/MacOS/VectorCorePackageTests` bundle. Supplemental direct runs retain exact host argv, environment, executable bytes/hash and the exact loaded bundle hash in `v1/direct-{debug,release,asan}/provenance.json`; each reports seven actual functions, not additional unique coverage. The direct ASan run also retains the actual preloaded ASan runtime. Child argv forwarding is source-associated; it is not presented as independently sampled process argv for each of the 252 children.

Public commands, statuses and complete logs are in `v1/public-escalated`. Its `inputs.json` records all 131 actual linked objects, the VectorCore module artifacts, fixture/driver source and executable hashes, pointing to the immutable Release scratch. The initial sandbox-denied Clang ModuleCache compile is separately retained in `v1/public`; it is an environmental failure, not a negative protocol test. The same frozen driver succeeded after approved escalation. External processes have prelaunch SIGCHLD registration, file captures, 15-second deadlines and at most 2 seconds cleanup; timeout raises before any acceptance/status result.

`v1/source-runtime-association.json` connects each command/log, source version, artifact inventory, host, bundle, child index and public consumer record. Actual artifact counts are RED 4,363, Debug 4,363, Release 2,891 and ASan 4,364. There were no source revisions or overwritten tested build versions.

## Qualifications

SER-01/02/06 receive the concrete empty-frame, documentation and executable-coverage repairs. SER-04/05 receive explicit precision/protocol dispositions and working public examples. SER-07 framing behavior is preserved with broader class coverage. SER-03 remains unoptimized and unmeasured; V04 and separate scope are required before scalar-loop optimization or speed claims. No compressed format, streaming, automatic detection, migration engine, platform emulation, cross-platform execution, universal precision guarantee, production performance target or downstream adoption is asserted. Existing nonthrowing encoding beyond the decoder cap and BinaryHeader's representability precondition remain explicit limitations.

The final worker-only sealed index is `B01-worker-evidence-index.json` in the canonical evidence root. It includes the actual retained sources/build artifacts and report, excludes itself to avoid a circular hash, and excludes Python bytecode caches. No root or independent-review records are included.
