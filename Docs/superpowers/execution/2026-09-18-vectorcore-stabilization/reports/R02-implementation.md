# R02 implementation and review package

Frozen archive: `/private/tmp/vectorcore-stabilization-R02-implementation`, base `84f42d01489a526b8ae40bd77fe7618c0de33f34`. `R02-full.diff` includes all three changed/new files; `R02-source-manifest.json` records their SHA256 values and confirms all 1899 baseline files were present. Diff SHA256: `30825c993d28937ce5f7dcaa26ca5f67ebbf4d586f2293cc597a2081c8f3c7cf`.

Canonical recovery backup: `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-09-23/R02/R02-frozen-source.tar.gz`, SHA256 `383b76986a7908b3986ab3ba81e18f06f1d22f0f0fb895b351e006a7da3cc40e`. `R02-backup-manifest.json` also identifies the exact pre-repair RED archive. No N01, configuration, R05 score-policy, Package, sibling, index, or commit changes occurred.

## Implementation

The destructive generic update default is removed. `MutableVectorCollection.update` remains a requirement with an explicit failure-preserving contract. `SimpleVectorCollection.update` validates dimension before locking, then checks membership and replaces the existing dictionary binding under the lock. It does not call duplicate-rejecting add or perform generic rollback. Dimension mismatch precedes missing-ID lookup and retains the existing error families.

Explicit insertion rejects occupied IDs with `.invalidConfiguration` and a duplicate/ID diagnostic; any Int remains accepted when unused. It leaves the generated cursor unchanged. Generated insertion scans occupied successors using a local cursor and checked arithmetic, commits only upon finding an unused nonnegative ID, and sets its private optional cursor to nil after allocating Int.max. Failed dimension/duplicate/exhaustion paths do not advance the cursor. Removal and removeAll do not rewind it. The only new construction seam is an internal nonnegative near-limit seed, not a public configuration API.

Simple batch retrieval reads requested bindings under one lock. Simple search captures all dictionary bindings under one lock and releases it before invoking the supplied metric and TopK. The small private scoring helper is shared with the generic default; it preserves the actual Float metric witness, captured candidate count, existing TopK indices/ties/NaNs and result construction. No ID sorting or new score policy was introduced. The default collection implementation documents that composing allIDs and individual lookups is not an atomic snapshot. Stored V bindings are captured, not universally deep-copied; custom reference aliases remain the conformer's responsibility.

## Behavioral RED and focused evidence

Before production changes, `R02-red-source.tar.gz` captured Sources, Package.swift and the exact two-test RED file. `R02-red-01.log` exited 1 with 10 behavioral issues: invalid-dimension update deleted the old binding/count/search entry, generated ID0 overwrote explicit ID0, and duplicate explicit insertion overwrote again without throwing. `R02-green-01.log` then exited 0 with both tests passing.

The expanded `R02-green-02.log` is a retained compile-only failure: a Swift Testing macro containing a mapped numeric expression exceeded the compiler's type-checking budget. Replacing that test-only expression with independent literal expected scores resolved it. `R02-debug-final-01.log` exited 0: 80 reported test functions across five suites. The parent `reentrantSearchChildProbe` entry is a no-op without its environment; the deadline-bounded parent test launches and verifies the actual child probe separately.

The new tests cover all thrown Simple update paths with original coordinate bits, membership/count and subsequent search preservation; valid/missing-ID controls; an explicit custom rejecting replacement witness invoked through generic MutableVectorCollection; explicit/generated ordering, sparse positive IDs, negatives/Int.max, duplicates, dimension precedence, removal/reinsertion, removeAll, and free/occupied/exhausted near-Int.max cursors. Concurrent generated/explicit insertions retain every successful distinct payload, enforce unique generated IDs and exact final counts, and allow only the intended duplicate collisions.

Concurrent update/read/search stress checks complete payloads and internally consistent candidate/result membership. These bounded observations do not prove all schedules: the single-lock batch/search capture is established directly by the source's lock boundary. No atomic multi-operation transaction or deep alias immutability is promised. The configured reentrant metric reads, updates and removes collection entries during scoring; captured candidates retain their original values and the child completes with exact expected IDs/scores. Its file capture and termination handler are installed before launch; deadline is 15 seconds, cleanup at most two additional seconds, and only its owned PID can be killed. No waitUntilExit is used.

## Public Swift 6 migration fixtures

`Scripts/ci/tests/R02CollectionUpdateConsumer.swift` imports only public VectorCore. With `-D MISSING_UPDATE`, Swift 6 complete-concurrency typechecking exits 1 specifically because update(id:vector:) is missing (`R02-consumer-missing-witness.log`). The explicit transactional witness version compiles/links (`R02-consumer-positive-compile-02.log`, exit 0), runs (`R02-consumer-positive-run.log`, exit 0), preserves an injected rejection and succeeds via generic MutableVectorCollection dispatch, printing `R02_PUBLIC_TRANSACTIONAL_WITNESS_OK`. This proves the concrete fixture behavior, not arbitrary external conformers' transaction correctness.

A preliminary positive-link command used a nonmatching top-level C object glob; zsh exited 1 before invoking swiftc. SwiftPM's C objects are nested one directory down. The corrected glob is retained in the final command record; this was tooling only, with no source workaround.

Migration for downstream conformers that inherited update: implement a transactional witness that validates and commits within the conformer's own synchronization or storage transaction boundary. Catch-and-reinsert is not a generic guarantee. Existing callers must handle duplicate explicit insertion and generated-ID exhaustion instead of relying on overwrite/wrap behavior. No VectorIndex or VectorAccelerate concurrency claim is made.

## Final checks

Source bytes were frozen and backed up before the Release build. Exact commands, exits and source/log hashes are in `R02-validation.json`. Final Debug and Release both exited 0 with 80 reported test functions across five suites. Independent review and root API-09/API-10 adjudication remain separate.


## Final validation and retained child provenance

Final Debug and normal Release each passed the exact focused filter `VectorCollectionTransactionTests|MetricDispatchContractTests|IntegrationProtocolsSuite|TopKNaN`, using separate R02 scratches and `--jobs 6`. Each parent run reports 80 functions across five suites. Within the new suite, 11 parent functions actually execute and `reentrantSearchChildProbe` is a no-op in the parent; the bounded parent probe then executes that child function in a separate process, verifies exit 0 and the completion marker, and checks captured candidate IDs/scores/count. Do not count the parent no-op as additional exercised reentrant coverage.

`R02-child-capture-manifest.json` hashes both actual unique child captures, preserves their original paths and timestamps, and associates each with its exact Debug or Release parent log. The original capture creation time falls within exactly one parent log lifetime; the files themselves show one passed child test and `R02_REENTRANT_SNAPSHOT_COMPLETE`. They are retained under `R02-child-logs` and copied to the canonical recovery directory. No frozen source edit was needed to add logging. Parent and child source bytes match the same frozen diff hash.

`R02-validation.json` contains the exact public Swift 6 negative typecheck, positive compile/link and positive runtime commands, their exits and log SHA256 values. The negative conformance fails solely on missing update, the positive consumer runs its own transactional witness through MutableVectorCollection, and all three changed/new source hashes still match the frozen manifest after validation. Debug/Release matched test-symbol inventories are retained as `R02-debug-symbols.log` and `R02-release-symbols.log`.

There were no existing test edits and no performance or external VectorIndex/VectorAccelerate concurrency claims. Bounded stress execution complements the source-derived lock invariant; it is not proof of every schedule or arbitrary conformer behavior.
