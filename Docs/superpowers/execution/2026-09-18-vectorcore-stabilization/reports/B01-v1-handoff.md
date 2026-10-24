# B01 v1 source handoff

Version v1 is frozen in the canonical 2026-10-02/B01/v1 directory. Full diff SHA256 is fa091f10be7c25c3c9252fa32523d9f405331bbc7b238f4ed7e1a5abd1a51b82. The complete archive contains 5,934 base files and 5,943 final files; 16 paths changed or were added. All other base files are byte-identical. The only production behavior change is accepting dimension zero in BinaryFormat's existing shared header validator. The VectorShapeContractTests change is the root-approved comment only.

Actual unchanged-production RED is retained in red/debug.log and command.json: exit 1, three empty-route invalidDimension issues, successful nonempty golden and direct-slice control. The immutable build-red-debug directory retains all 4,363 inventoried artifacts. Source reconstruction is base.tar plus the three files in red (B01BinaryContractTests.swift, B01Golden.swift, B01_generate_goldens.py).

Seven lexical/mutation inventory checks and independent golden regeneration pass. Focused Debug compilation/runtime is underway in build-v1-debug; no final runtime claim is made yet. Release, valid-path ASan and external Swift6 consumers remain to run. Consumer driver, fixtures and maintained scanner are included in the frozen full diff.

All 15 prior comment-only names are mapped in Docs/Serialization.md. Sixteen real SerializationSuite functions remain; two misleading names are narrowed while preserving their tested behavior. Exact bit, framing, slices, caps, concrete JSON/FP32 destination and explicit FP16 conversion coverage is in the changed suites. No compressed format, streaming, built-in BinaryCodable conformance, scalar optimization, performance result or cross-platform execution is claimed.
