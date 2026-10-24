# V03 post-prerequisite readiness review

**READY for dispatch after final V02 acceptance/import.** No additional architecture blocker found in this bounded source-readiness check. This does not accept V02's unsealed final evidence or establish instrumented/hosted CI execution. Integration inspected: `1eea5ec95fcaabc48c07447976575a70bb68aacc`. Current V03 brief SHA256: `7ccf4f729b881ba6ae08f31f8bb5715af6d8e4b8d597ee1b5a4b4f31f17bef9e`. Read the full brief, prior design inventory/review, current affected tests/workflow, and the V02 workspace driver; no builds or probes.

## Resolved handoff

`Scripts/ci/check_benchmark_build_modes.py --scratch-root DIR --output DIR --jobs N` implements the brief's exact interface, bounds jobs to 1...6, creates four distinct Debug/Release × optional-C-off/on scratch directories, and rejects existing cell scratch/output reuse. Activation is `-Xswiftc -DVC_USE_C_KERNELS`. The v5 executable-discovery correction excludes the same-named dSYM DWARF file by requiring the configuration-directory parent.

Each cell runs the real benchmark for absent/matching requests and rejects a contradictory request; links the public INT8 consumer with the renamed real dispatcher and forwarding sentinel; checks 36 public route records and ten direct C tail controls; and retains source/module/object/executable and command/status/output provenance. The driver verifies unchanged ordinary compiled artifacts after replacement linking. Only after all four cells and final source hashes pass does it write `result.json` with `V02_BUILD_MODES_COMPLETE` and print that marker. `progress.json` and `failure.json` are not completion evidence. V03's semantic policy can now bind to this actual implementation rather than a future unspecified interface. Preserve the C-off/LAPACK distinction and public-route fixture scope; this does not prove arbitrary raw-C lengths or FP32 dormant routing.

## Q02 timing migration

Accepted Q02 leaves two measured-clock predicates in `Tests/ComprehensiveTests/KernelAutoTunerTests.swift:162–163`: positive throughput and mean latency after real profiling. Both remain in V03's strict opt-out scope. Adjacent estimated payload, synthetic accuracy bounds, and profile sorting assertions remain functional contracts, not speed targets. The injected score fixtures at lines 360 onward and `PrecisionSelectionContractTests` injected latency/quantile/resource-filter controls remain active: their values are deterministic inputs, not environmental performance assertions. Q02's new names/fields require refreshing the old inventory anchors, not changing tuner production.

Implementation detail to make explicit at handoff: `KernelAutoTunerTests` is XCTest while the brief's central helper paragraph names Swift Testing. Preserve framework-appropriate failure reporting for those XCTest sites (a lazy XCTest-facing adapter under the same strict predicate/site inventory is sufficient); do not assume a Swift Testing issue alone establishes an XCTest failure. Test the actual adapter path with the requested strict/off process controls. This is within the existing test-only ownership and does not require a new API or production change.

## Sanitizer manifest readiness

All suite identifiers in the current standalone sanitizer workflow still resolve to actual current declarations, including `MemoryPoolTests`; none can simply be removed as a stale name. Their actual selected bodies still need the implementation's promised supported-execution audit and runtime manifest. The existing workflow has neither O05 nor Q03's new coverage and therefore cannot be reused unchanged as evidence.

O05's `StorageBorrowContractTests` supplies seven real valid bodies: `fp16Borrow512`, `fp16Borrow768`, `fp16Borrow1536`, `directFP16KernelMatrix`, `externalConformerLayoutMatrix`, `ownedDynamicSliceSurvivesSourceChange`, and `fp16SoAStorageRemainsOwned`. These are concrete ASan candidates without intentional invalid lifetime accesses.

For Q03 TSan, select `FP16DiagnosticsBoundaryTests/isolatedConcurrency` and validate its actual child rather than counting the no-op `childProbe` entry. Set `Q03_TSAN=1`, create a retained `Q03_CHILD_LOG_DIR`, and keep `Q03_UNSAFE_BASELINE` disabled. The accepted helper locates and injects the actual loaded TSan runtime, records host versus loaded bundle, applies a 45-second child deadline plus bounded cleanup, and verifies body/result/end markers. The concurrency evidence should include all six completed 2000-iteration loops plus the stable-total and atomic-batch completion markers. The accepted helper's presence is not fresh TSan runtime evidence.

E02 provides direct valid candidates in `ExecutionSchedulingContractTests.swift`, including `reductionIdentityAndOrderedAssociativity`, `preCancelledThrowingRoutesAndExecute`, `cancellationBetweenUnitsAndOriginalBodyError`, and `reductionCombineCancellationAndRegionAcrossProvidersAndOverloads`. Prefer explicitly named valid bodies over broad selection of its rejection-child and watchdog controls. Pin framework, exact body, expected markers, and no-op exclusions in the independent required-body policy as the brief already demands; instrumented execution remains acceptance work.

## Ownership and remaining gate

Active L03 owns UMAP/FuzzySimplicialSet/KNNGraph production and UMAPTests. V03's timing inventory and driver/workflow/lint work need no edits to those paths. Preserve L03 numerical assertions and avoid expanding benchmark-formatting ownership to library files. Q02's accepted test migration is the relevant refreshed overlap; its tuner production remains excluded.

The remaining dispatch gate is final V02 evidence acceptance and import of its 33-file source, including the narrow internal ABI-visibility annotations. Parent-reported four-cell verification and the completed-runtime checkpoint are useful prerequisites, not a substitute for that final adjudication. No new performance target, remote branch-protection claim, or whole-suite sanitizer guarantee is introduced.
