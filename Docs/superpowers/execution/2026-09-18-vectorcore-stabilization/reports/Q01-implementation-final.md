# Q01 final implementation handoff

Frozen revision4 is implemented and worker verification is complete. Root retains independent-review acceptance and closure decisions. Complete patch: `Q01-review-v4.diff`, SHA256 `03b1df34f94d20807683da228af2d3f382ee6bd389382d0e63b136bdc03a1117`. Base/final hashes for all7 changed/new files: `Q01-review-v4-manifest.json`. Archive: `/private/tmp/vectorcore-stabilization-Q01-implementation`; base `0dbb18c818f26edd8e28bb303a6fccedbd063a6f`. No commits/index operations, sibling changes or nested delegation.

## Verification on frozen v4

Exact executable argv, scratch paths, working directory, logs and exit statuses are in `Q01-validation-commands.json`.

| Configuration | Exit | XCTest methods | Swift Testing functions | Result |
|---|---:|---:|---:|---|
| Debug, focused correctness |0|17|50|passed|
| Release, same focused correctness |0|17|50|passed|
| Address Sanitizer, valid controls only |0|10|7|passed; no sanitizer finding|

The50-function totals include one parent-process no-op child entry; the parent matrix actually launches99 intended-rejection children and1 finite-outlier success. Frozen-v4 harness installs terminationHandler before run, uses a15-second semaphore execution deadline, and at most2seconds of owned-PID SIGKILL cleanup. It has no waitUntilExit/polling/pipe read. Each child log is retained with a unique Q01 prefix. ASan excludes deliberate traps and covers raw legacy decoding, valid SoA, mutation/rollback and numerical controls plus relevant S01 shape/postreplacement controls. Existing unrelated compiler warnings are retained in the complete logs.

`Q01-final-test-inventory.log` is the refreshed compiled symbol inventory (exit0). `Q01-scope-inventory.json` checks224 base source/test files and finds only the7 owned changed/new paths. `Q01-archive-integrity.json` verifies all778 base tracked paths exist; final worker hash check confirms all7 current file bytes equal frozen-v4 manifest. Root has additionally backed up frozen source outside temporary storage.

## Numerical and memory evidence

The full case matrix is detailed in `Q01-implementation-review-v1.md`, with subsequent corrections in `Q01-review-revisions.md`. Final coverage includes both calibration APIs; all256 codes for the deterministic extreme/exponent tuples; invalid Codable tuple matrix and exact legacy fields; all three widths and conversion/borrow families;99 named rejection cases;21 EX04 witnesses; public concrete and generic metrics; both mixed orders; both SoA routes/counts0...5/prefix sentinels; exact dot cancellation and neighboring nonzero; signed wide carries/negative magnitude conversion; required exact affine differences over five powers-of-two scale families; and96 independently generated exact-dyadic difference fixtures.

Independent arithmetic provenance: `Q01-exact-oracle.json`, `Q01-exact-oracle.py`, and the later exact centered-square correction in `Q01-release-guards.py`. The required INT8 raw-zero difference−1 fixture remains scalar because the supported SoA Float conversion yields code2; its supported SoA test uses independently computed difference−16777221. Mixed difference+1 is constructible as SoA code0 and is tested. No unchecked SoA insertion API was added.

Root's linked Release probe found the v2 dynamic expansion and flatMap calibration allocation regressions, then authorized the v4 fixed SIMD4 expansion and streaming calibration repair. Root reports all22 measured v4 cases request0allocations/0bytes and preserve every v2 result bit. Probe evidence is retained separately by root in `Q01-root-allocation-*` and `/private/tmp/vectorcore-Q01-root-allocation-v4/{provenance.json,phase-2.log}`. This is bounded requested-allocation evidence, not latency/throughput or scale-readiness evidence.

## Retained failures and migrations

All behavioral RED and failed compile/test attempts are retained; see the revision narrative and individual `Q01-red-*`, `Q01-green-*`, and earlier focused logs. In particular, original calibration collapse/outlier trap/EX04 failures, final-scalar-rounding RED, Release diagnostic-stripping failure, and bounded-harness review correction remain inspectable. An exploratory broad test filter accidentally selected existing performance functions; that failed run is retained and explicitly supplies no performance evidence.

Only two pre-existing test expectations changed, each with root approval: asymmetric scale expectations now include zero ([-100,-10] uses100/255 instead of90/255), and the old nonfinite `from` success test now checks throwing array `.invalidData` with indices0,1,2. The new child matrix supplies nonthrowing rejection evidence. No further existing test migrations occurred.

No VectorProtocol/ordinary FP32 metric range, N01 helper, Package/backend flag, tuner, GPU/downstream integration or broad accuracy-analysis change is included. Optional C alternatives remain uncompiled/unverified. No universal floating-point error theorem or production scale/performance claim is made. The review package preserves revisions1–4; final acceptance should use revision4.
