# Q02 independent frozen-v2 scoped review

**Source disposition: ACCEPT. Final implementation/runtime acceptance remains pending the sealed evidence handoff.** This supplements the preserved v1 production review; it does not replace its mathematical, persistence, measurement-boundary or evidence qualifications.

## Freeze and scope

Reviewed canonical `2026-09-24/Q02/frozen-v2`, not the advancing workspace. Verified full patch SHA256 `a2d335f084ca332782fd99bf5582ded5d13e142404589d3b81580fdb95599334`, source manifest `ec8a102fbd7ee1ec850f11d957061863e71bf16c20daa6a4e6b08d63aa602dd6`, frozen hash map `e2208200e20f456f5083f1645b151a07cc4e46fad959a2332506e7c1471f33f8`, and every file recorded by the frozen hash map. Base remains `101b3749bf531cfdbc42d61198b503e94af4aa5b`. Manifest inventories contain328 baseline and338 final files, with17 owned changed/new paths. Comparing all v1/v2 final inventory hashes identifies exactly eight changed/added paths: four existing test files, the public consumer, and three new probe/driver files. No production file changes between the reviewed v1 and v2.

Read the complete v1-to-v2 test/consumer deltas and all three new files: `Q02CalibrationProbe.swift`, `Q02RunCalibrationProbe.py`, and `Q02RunPublicConsumer.py`.

## Earlier findings

**IR1 source coverage resolved.** Already-cancelled tasks self-cancel before entering each actor calibration at512/768/1536; they require CancellationError, zero observed work, retained prior profiles, and retained override behavior. This complements the existing during-work and commit-boundary controls.

The standalone public consumer establishes a qualifying reduced shared mixed profile and override, proves the direct reduced lookup is active, then checks full-mode bridge precedence and reduced-mode delegation. The singleton is confined to this fresh consumer process and cleared. The control is no longer vacuous.

The public persistence consumer independently mutates15 profile fields covering invalid finite metrics, percentile ordering, estimate, accuracy, counts, strategy/precision compatibility, per-profile provenance and workload identity. It requires the affected512 family to miss while another valid768 family survives. It does not use the production validity predicate as its oracle. The unavailable-hardware test explicitly checks disabled persistence and exercises profile/clear with a URL that must remain unused.

**IR2 substantive migration resolved.** Touched lookup tests now identify unprofiled fallback across sizes/limits instead of claiming blocked-kernel performance or memory-pressure decisions. Synthetic-error comments no longer equate the limit to one-minus-accuracy; the fixed Kernel fixture is labeled as a deterministic synthetic dataset. One residual assertion message in `MixedPrecisionKernelTests.testAutoTunerThreadSafety` says “Cached strategies should be identical” although it observes repeated misses; this is a nonblocking wording nit, not evidence of cache reuse. No timing or general accuracy claim is accepted from these tests.

## New driver and probe assessment

Both Python drivers start separate process groups, impose15-second execution deadlines and bounded2-second post-kill waits, record timeout/cleanup outcomes and fail the driver on any timeout. They redirect output to files, avoiding pipe-drain deadlock. Public compilation/typechecking and real consumer execution remain distinct recorded phases. Linked objects/modules, fixtures, executable hashes and logs are captured; ordinary consumer environment removes the listed CI/test cache-bypass variables. Final review must associate these records with the exact frozen sources and retained binaries rather than merely trusting source-side provenance generation.

The calibration probe separates explicit calibration from selection. Its work control independently expects25 executions across conversion-bearing strategies, totaling `25*(4*N+2)*D` converted elements, and consumes five returned strategy records. Allocation mode includes the whole explicit calibration call, uses no-op/C/Swift-array instrumentation controls, and reports elapsed time plus cumulative allocator events/requested bytes. It does not establish peak live memory, RSS, production workload performance, or an acceleration guarantee. Coverage-instrumented work observations must remain distinct from matched ordinary allocation/timing runs.

Reported successful public runs and preliminary tests are not final acceptance here. Remaining adjudication requires the immutable worker index, exact final commands/statuses/logs, baseline versus setup-failure distinctions, matching source/object/executable associations, and the disclosed scope of measurements. No builds, source/test edits, index operations or nested delegation were performed.

## Announced follow-up

Root reports a pending test-only v3: two legacy print-only lookup controls gain explicit fallback assertions, the legacy strict-limit multi-strategy assertion is narrowed, and Kernel invalid iterations/warmup/overflow gains a no-work counterpart. These are not part of this frozen-v2 review or claimed execution. Preserve v2 evidence and review the exact v3 delta before final acceptance; production, public consumer and probes are expected unchanged.
