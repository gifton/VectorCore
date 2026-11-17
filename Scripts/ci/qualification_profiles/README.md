# Reviewed compiler qualification profiles

Profiles for `compiler_qualification.py` are reviewed data, not generated
acceptance decisions. The first native profile is
`macos-27.0.1-arm64-alpha-2026-11-10.json`; its exact scope and
[reviewed result](../../../Docs/reviews/2026-11-10-compiler-acceptance/native-tooling.md)
cover tooling validation of the current alpha inputs. It does not admit stable 0.4.0.

A profile binds the complete source manifest, compiler/SDK/runtime identity,
producer modules and objects, command arguments/environment/exits, artifact
digests, full observation inventory and literal numerical oracles. The `correct`
oracle states mathematical expectations; `observed` records only the reviewed
observation allowed for that exact diagnostic/control composition. Supported
groups must match `correct`, regardless of their names or `observed` values.

The `pins` mapping is the integrity trust anchor. Receipt contents cannot replace
an earlier digest. `captureRoots` maps the original source and evidence locations
to the explicit current roots; other absolute files must have individual pins.
The validator does not authenticate the original execution or protect a profile
from an author deliberately approving false evidence.

Each supported observation group requires a complete omitted-write control for
the same producer/client configuration. Compiler diagnostic failures remain
numerically incorrect in the report. Scoped acceptance always retains
`releaseAdmitted=false`; platform coverage and other release gates remain separate.

Review a new profile when source, compiler, SDK, flags, runtime, architecture or
the observation inventory changes. An unexpectedly corrected diagnostic also
requires review. Preserve historical profiles and raw outputs rather than
relabeling them for another candidate.

The separate `bounded_native_qualification.py` and its adjacent profile retain
their original historical purpose and have not been changed by this mechanism.

Fresh captures and profiles use schema 2. The identity includes
`swiftDriverSHA256`; `identityFiles` contains exactly the frontend, Swift driver
and SDK settings paths. The driver must be the pinned sibling of the selected
frontend, and both whole-file fingerprints must form one adopted pair. Both
adjacent `swift` and `swiftc` files must also be pinned and have the frontend's
fingerprint. Their command roles remain separately bound by paths and prior pins.

Historical schema-1 profiles remain unchanged and are limited to the original
local frontend fingerprint. They retain their original frontend-bound scope;
they do not make a retrospective claim about a separately captured Swift driver.
Hosted verification requires schema 2, so a legacy-profile downgrade cannot
satisfy the new hosted scope. Numerical expectations and review requirements are
unchanged.
