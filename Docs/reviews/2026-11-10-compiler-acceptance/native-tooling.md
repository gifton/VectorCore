# Native tooling validation — 2026-11-10

Fresh native evidence satisfies the adopted bounded policy for the current
alpha compiled inputs on Xcode 27.2 build 27B52. Supported numerical checks pass;
the compiler remains **known-failing** and **releaseAdmitted=false**. This is
Task 2 tooling validation. The final independent integrated review, iOS scopes,
hosted qualification and stable compiled-candidate selection remain pending.

## Identity and preservation

- Source checkout: `/Users/goftin/dev/projects/VectorCore-archive/VectorCore-fixes-2026-10-27-84798d3`.
- Captured source HEAD: `e4c711d7beccfe99f3f2a20997d0ec117be00bb8`; tooling edits were present and recorded in `source-state.json`.
- All 389 tracked compiled inputs match the adopted alpha baseline. No production Swift/C or version constants changed.
- Capture source-tree manifest: `d6a5711b1571cc853c571a4312a619d512b12459bb8e644d3bb78fdfcdc291db`. It conservatively includes three existing `.DS_Store` files in addition to those 389 inputs, so its digest differs from the alpha's 389-file manifest. Before/after manifests match.
- Compiler frontend: `448457da9a8a56741f2c8c4b8931e85b7377fc009c16918cd8ac1a8a3210ea1f`.
- SDK settings: `82165b15ee21419d4cffaba8895797391b908ac9499ba70f561f3e1bd16758d6`.
- Runtime: macOS 27.0.1 (26A434), arm64; consumer target `arm64-apple-macosx14.0`, Swift 6, ordinary `-Onone`/`-O`.
- Invocation: `71ca0f26-2fd0-4491-a9c7-2dc1e78ed516`.

## Results

| Scope | Observed result |
|---|---|
| Native Debug suite | 1,947 passed, 46 skipped, zero failures; 1,993 declared |
| Native Release suite | 1,946 passed, 46 skipped, one unchanged lost-store canary failure; 1,993 declared |
| Debug/Release producer × Debug/Release consumer | All 96 by-value numerical observations pass; all 96 omitted-write observations match their independent control oracles |
| Separate generic witness and migration contracts | Witness object/module/link binding verified; all 12 migration contract cells match the expected success or diagnostic |
| Original standalone compiler matrix | 11/14 positive cells correct; three optimized failures retain `40000000` (2.0) instead of `3f800000` (1.0); both no-write controls complete |
| Concrete stdlib diagnostic | Each Release client retains the concrete-inout failure; all concrete by-value observations pass |
| Residual helper/conformer matrix | Debug clients: 17/17 correct. Release clients: 14/17 correct, with exactly the three reviewed inout failures, for both producer modes. All seven by-value rows stay strict; all 68 omitted-write observations match |
| Optional C-on/off, Debug/Release | Four producer cells, 144 numerical/forwarding routes and 40 tail controls complete. Matching/default requests pass and all four contradictory requests reject |

The Release suite's only failure is
`ComprehensiveTests.LostStoreCanaryTests/literalArrayStoreIsObservedAfterOutOfLineClosure()`
at `LostStoreCanaryTests.swift:47:9`. XML retains the observed 2.0,
bits `0x40000000`, and the unchanged expectation of 1. Every other failure is
rejected. Individual XCTest records and Swift Testing XML outcomes are reconciled,
including exact skips and discovery inventories. No test is filtered out.

The three residual failures are `new.concrete.copied-inout`,
`new.inside.copied-inout`, and `new.aggregate.inout`. Their raw process exits and
incorrect positive values remain unchanged. A diagnostic's name does not exempt
any supported group or required-correct row.

## Evidence and repeatable assessment

Canonical local evidence:
`/Users/goftin/dev/projects/VectorCore-archive/VectorCore-compiler-acceptance-2026-11-10-evidence/native-tooling`.
The 2.7 GB archive retains build products, receipts, streams, XML, sources, SIL/IR,
consumer objects/modules, controls and C-route records. It was copied without
rewriting receipts and revalidated through explicit root remapping. Original
captured paths under `/private/tmp/vectorcore-native-qualification-2026-11-10-02`
remain in the profile and receipts as historical provenance.

```sh
python3 -B Scripts/ci/compiler_qualification.py \
  --profile Scripts/ci/qualification_profiles/macos-27.0.1-arm64-alpha-2026-11-10.json \
  --evidence-root /Users/goftin/dev/projects/VectorCore-archive/VectorCore-compiler-acceptance-2026-11-10-evidence/native-tooling \
  --source-root /Users/goftin/dev/projects/VectorCore-archive/VectorCore-fixes-2026-10-27-84798d3 \
  --output /private/tmp/vectorcore-native-assessment-new.json
```

The output path must not already exist. The reviewed profile pins required data
and helper identities; the profile-directory README is explicitly context only.
Changes to a pinned helper require a reviewed evidence/profile refresh and must
not be hidden by replacing a digest during verification. The checker assesses
retained evidence, not hostile-host authenticity.

The first attempted capture at `/private/tmp/vectorcore-native-qualification-2026-11-10-01`
remains incomplete. Its Debug suite passed, but a collector product-discovery bug
mistook a deprecation warning on stderr for part of the stdout path. A regression
test reproduced this, the collector was corrected, and capture 02 rebuilt and
reran the matrix in a new directory. Capture 01 supplies no qualification result.

## Remaining work

No iOS simulator/device acceptance, hosted run, minimum-OS/Intel/other-platform
acceptance, sanitizer refresh, performance acceptance or stable release approval
is established here. Physical iOS requires the separately approved app-hosted
Release run. No branch was pushed, no workflow dispatched, and no release tagged
or published by this validation.
