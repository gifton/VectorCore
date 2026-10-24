# capture_swift.py orchestration review

**Verdict: changes required before reusable approval.** Read-only source review; no Swift invocation, subprocess control, production edit or build was performed.

Reviewed source SHA256 `ed795563966d9cd15f316b2e37ca54d4c45ca18425261a02b4ea98badf0ce35e`.

## Blocking argument-validation gaps

1. `one_option` recognizes only an exact split option token. An accepted `--jobs 4` can coexist with an unchecked `--jobs=99` or Swift's `-j 99` spelling. Likewise `--scratch-path <canonical>` can coexist with `--scratch-path=<other location>`. Depending on Swift argument-parser duplicate handling, the process may reject or use an unchecked value; the wrapper has not established its advertised effective jobs/path restriction. Reject alternate and duplicate spellings of these controlled options, or parse all supported spellings and reconstruct one canonical occurrence. Do not rely on the particular Swift release choosing the first occurrence.

2. A relative scratch path is resolved against the driver's current directory during validation but the original relative token is passed to Swift with `cwd=workdir`. Those can identify different directories, so a successful canonical-path check does not bound the effective build directory. Replace the validated argv scratch value with its resolved absolute path (or reject relative scratch input). Prefer explicit absolute output argv in the saved record so the effective path is reviewable.

A narrow validation matrix should cover valid split absolute options, duplicate split values, equals spellings, short jobs aliases, out-of-scope paths, and a relative path with different driver/work directories. These can be tested without launching Swift; root retains implementation and validation ownership.

## Sound source properties and evidence limits

The executable is found through PATH and the exact resulting argv is passed directly to Popen without a shell. Workdir is resolved strictly, must contain Package.swift and must lie in recovery storage or equal the integration directory. Package-path/chdir long-form overrides are explicitly rejected. Log and record paths are canonicalized, distinct, constrained to recovery storage with existing parents, and created exclusively (`x`/`xb`). A race that creates an output after validation is rejected by exclusive creation rather than overwriting earlier evidence. Source freezing and library/module association remain caller duties, as the docstring states.

stdout and stderr both go directly to one binary file, so normal child output is not limited by the tool's response cap and no pipe-reading deadlock is introduced. stdin is closed. The driver waits for the Swift process without inventing a build deadline. On observed completion it records the actual signed return code, elapsed time, log size and SHA256; negative signal status is mapped only for the driver's shell exit. A normal nonzero build is still accurately labeled completed with its nonzero return code, not success.

The record is exclusively created once and updated in place for starting/running/completed states; it is a per-invocation status record, not an immutable event journal. Flush is used but neither crash-atomic replacement nor power-loss durability is promised. KeyboardInterrupt explicitly records unverified child completion without fabricating a status or claiming reaping; external abrupt termination may leave the last starting/running state. OSError records a driver error without asserting completion. These limitations are truthful and are not blockers for the stated helper purpose.

A completed record establishes the waited Swift process's completion, not independent proof that arbitrary detached descendants can no longer write an inherited descriptor. Existing bounded test harnesses and final artifact association remain responsible for their own child provenance. No cleanup, cancellation latency, sandbox/security-boundary or universal descendant-output guarantee is inferred. The inherited-environment description is accurate but is not an environment snapshot or toolchain binary hash; callers retain those broader provenance obligations.

Only this review report was written. Root should repair the two validation gaps and run its planned local controls before approval.


## Scoped re-review — source 49189927…

Reviewed final proposed source SHA256 `49189927a51413fc103eeb303ab176df054d9ba57dbba287e5eeb1b3c81fd24b` and validation-controls.json. **Relative scratch mismatch resolved; one remaining recognized-option class requires rejection before approval.**

Timing correction: the first displayed source lacked the alias guard; root added it during that review, before the later hash read recorded `ed795563…`. Consequently the first report incorrectly associated the displayed unguarded version with that later hash. The equals and -j guard was already present at the recorded ed795563 revision. Exact duplicate split tokens were always rejected by one_option; they were a requested control, not an established defect. This append corrects that attribution and supersedes the earlier broad duplicate wording.

Current code resolves relative scratch against effective workdir and substitutes the absolute resolved path into the actual argv. Checked and executed scratch identities therefore agree. Exact split duplicates reject; equals jobs/scratch forms and -j prefixes reject. The retained controls accurately distinguish the 15 earlier alias-guard controls from the final relative-normalization control; they are validation-only evidence, not builds.

Local installed-toolchain `swift test --help-hidden` and `swift build --help-hidden` both expose `--build-path <build-path>` and `--multiroot-data-file <multiroot-data-file>`. These recognized legacy build-location/multiroot switches are not rejected by the wrapper. Reject their split and equals forms to remove competing unchecked build-location/package-root inputs. No actual precedence/bypass execution is claimed: the finding is that recognized location selectors remain admitted without a contract or validation, and the cheap conservative rejection avoids relying on Swift's version-specific precedence. Ordinary help confirms -j/--jobs and --package-path; those current advertised selectors are already handled. Other cache/SDK/compiler/output options are not reclassified as scratch/package aliases or a promised filesystem sandbox.

Only help commands were invoked, with no Swift build/test execution. Capture, exclusive-create and truthful status observations above remain unchanged. Root should add the two location-switch rejections and validation-only negative controls, after which this bounded remaining finding can be closed.


## Final scoped disposition

**ACCEPT for the stated local orchestration purpose**, source SHA256 `14ce36548aad77555b7e9e822bb415801f9ec308ace92e175d49d0c4ee8476e5`. Both remaining recognized location selectors now reject in split and equals forms via the same guard as package/chdir overrides. The absolute scratch normalization remains intact. All findings above are resolved, with the historical timing correction preserved.

Read the complete reproducible validate_capture_swift.py and validation-final.json; independently matched their recorded driver/control-script SHA256 values to current bytes and checked all 23 recorded expected outcomes. The controls call only validate(), with no Swift subprocess, and directly assert the normalized argv path. They include both build-path/multiroot forms, jobs aliases and duplicates, scratch duplicates, path/output restrictions and positive build/test argument controls. No rerun was necessary for this scoped source/evidence check. The control script's own hash is `008da4d10f9a09a865e9288512f7d149d84fd2ed51ebb9e89485fce4420ce3f3`.

Acceptance does not expand this helper into a security sandbox, immutable event journal, environment/toolchain snapshot, child-process cleanup system, or guarantee about detached descendant output. Build execution and full artifact provenance remain with callers; no build/test was executed during this recheck. Only this report was appended.
