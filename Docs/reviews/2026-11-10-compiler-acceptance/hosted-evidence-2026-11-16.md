# Hosted qualification evidence — 2026-11-16

[Collection run 44016980622](https://github.com/gifton/VectorCore/actions/runs/44016980622)
completed both macOS and iOS simulator collection, archive and upload steps.
Each platform then deliberately failed **Collection is not qualification**.
That red result prevents unreviewed collection from satisfying the qualification
aggregate. Full hosted verification of the reviewed profiles is pending.

The immutable collector candidate is
`7c115cd927c76275b73a08b0e6fb178636a3c4b9`; collection policy is
`a4f8bb8c9442b141412d55d1279246b1e25bb68e`. Its 389 tracked compiled-input files
match frozen stable candidate `af315c56c52de737f868fd969e6e1f5ae59eaa48` byte for
byte. The clean source manifest is
`2191f5d4663a071e9a537be4a525bb353798370d7c7db44980fcd15b0b1f521e`.
Only three untracked `.DS_Store` files distinguish this manifest from the older
local capture. Earlier profiles and evidence remain unchanged.

## Numerical review

The exact hosted Xcode 27.2 / 27B52 frontend and Swift-driver pair matches the
[adopted identity](decision.md#additional-exact-hosted-compiler-identity).
Numerical observations were compared with the existing local stable profiles'
literal `correct` and `observed` tables, without changing an oracle, exception,
required-correct row or negative-control relationship.

| Scope | Reviewed observations |
| --- | --- |
| macOS 27.0.1 (26A434), arm64 | 36 groups; 96 supported by-value rows correct across all Debug/Release producer/client pairs; 166 omitted-write control rows; 11 incorrect diagnostic rows retained |
| Native full suites | 1,993 declared tests per mode; Debug 1,946 pass / 47 skip; Release 1,945 pass / 47 skip / one unchanged lost-store canary failure; exact test identities and outcomes match the prior reviewed profile |
| Native auxiliary checks | 12 migration-contract checks; four C-on/off build cells, 144 numerical routes and 40 direct-C tails match their existing contracts |
| iOS simulator 27.0 (24A434), arm64 | 52 app-hosted groups across all four producer/client pairs; 96 supported rows correct; 168 omitted-write controls; 14 incorrect diagnostic rows retained |

All required-correct diagnostic rows also pass. The observed failures remain
numerically incorrect; they are explicit diagnostic evidence of the known
compiler defect. The simulator profile covers the consumer app matrix. It does
not claim a fresh hosted full-package suite or a physical-device execution.
The separately retained [local stable evidence](stable-candidate.md) supplies
those earlier scopes.

## Artifact and profile review

Both downloaded ZIPs match GitHub's recorded size and SHA-256. Their nested
archives match the upload manifests. Extraction accepted only regular files and
directories, with no links, traversal or duplicate members; archived helpers
and binaries were not executed during review.

| Artifact | ID | ZIP SHA-256 | Reviewed nested archive SHA-256 |
| --- | --- | --- | --- |
| native-evidence | 14300739713 | `dafb88ca35034c550a7799757e5a8c104681a54be6d8e32714109bc8011acecd` | `29533c1fa5a844c3b6216d994488f82c2079461550f5bacfadd9a33ea000ff3f` |
| ios-simulator-evidence | 14302155396 | `1f7139423d08f9bff0f103765e54965e24b3364a09a4e407f3cd2246cda90b1b` | `bfa16117a5953bb9c23310cd0c848348b956b2f2fde7e85456b4466ce91d48fa` |

The retained artifact inventories are complete: 1,518 native and 59,813
simulator entries, with no reported omissions. Their capture digests and every
retained/source file digest were reconciled. External compiler, SDK and system
tool hashes remain exact pins for the full hosted verifier to check against the
installed files. Both `swift` and `swiftc` aliases are pinned to the frontend
digest from the retained four-tool preflight. Referenced original C products,
retained copies and source files add explicit pins from their retained bytes.

The reviewed profiles contain 2,140 native and 60,748 simulator path pins:

- [Native profile](../../../Scripts/ci/qualification_profiles/macos-27.0.1-arm64-0.4.0-hosted-2026-11-16.json)
- [Simulator profile](../../../Scripts/ci/qualification_profiles/ios-simulator-27.0-arm64-0.4.0-hosted-2026-11-16.json)

Canonical evidence is retained under
`/Users/goftin/dev/projects/VectorCore-archive/VectorCore-compiler-acceptance-2026-11-10-evidence/hosted-qualification-44016980622/`.
Each platform directory contains the original ZIP, nested archive, complete job
log, safely extracted review data and `retention.json`. Separate review reports
and profile snapshots are retained with the review handoff.

Profile preparation and numerical comparison are not a full qualification
result. Verification must use the exact candidate, these tracked profiles, the
reviewed run/archive provenance and the unchanged complete validator. It must
return `vectorCoreQualification=accepted`, `compilerProfile=known-failing`, an
empty problem list and `releaseAdmitted=false` before the hosted scope is
accepted. Broader platform, minimum-OS, Intel, sanitizer, performance and release
reference/admission decisions remain outside this evidence.

## Integration checks

Independent read-only review found no Critical, Important or Minor findings.
It independently reparsed the observations and checked the retained/source
pins. Nine native and twelve simulator external pins remain for the actual
hosted verifier to check. The two new data files receive exact scanner
exemptions; adjacent unreviewed files remain covered by the existing controls.

All 756 Python policy tests passed in 271.015 seconds, eight focused layout
tests passed, ten GitHub policies passed, and whitespace checks passed. The
original dirty-main state, 25 protected historical files, four historical
profiles, alpha tag and all 389 frozen tracked compiled inputs are unchanged.
