# Reviewed compiler qualification profiles

Profiles for `compiler_qualification.py` are reviewed data, not generated
acceptance decisions. No live candidate profile has been installed yet.

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
