# Milestone09: combined tuning and PCA integration

Focused Debug and Release checks pass at source revision `813255cb389891590db172480667bb47d83fbdd1`, after accepted L01, L02 and Q02. Each configuration reports 24 XCTest cases and 96 Swift Testing functions in 18 suites, plus nine separately captured actual child scenarios: four L01 factorization-domain, two L02 PCA-domain and three Q03 FP16-diagnostics children. Reported parent/no-op test entries are not counted as actual child executions.

The filter combines precision selection/calibration and its migrated callers, PCA/factorization/KNN/UMAP controls, FP16 diagnostics and the two valid finite/exceptional range-work functions. Q04 rejection children, unrelated suites, new sanitizers and public consumers were not rerun. The earlier isolated task evidence remains separately identified in the closures.

Root verifies all 346 frozen code/source hashes unchanged, each command/exit/log, child invocation and completion marker, copied actual helper and independently hashed loaded bundle. Canonical versioned build trees retain 4,340 Debug and 2,879 Release file records without symlink aliases. Commands, driver, source map, inventories and root verification are in the adjacent milestone09 reports; actual logs, child captures, host and build bytes remain under `.stabilization-recovery/2026-09-29/milestone09` in the canonical repository.

The first Debug attempt failed at the sandboxed compiler module cache before tests. Its log/command were preserved before the approved retry. The final driver's debug artifact map captured the preceding command.json before replacing that path with the final command record; its original bytes are retained as sandbox-failed-command.json. Root verification maps that entry explicitly and hashes the final command record separately. No failed setup run is counted as successful validation.

This checkpoint establishes the selected integrated behavior only. It does not establish full-suite coverage, production workload readiness, downstream integration or a new performance result. Independent evidence review is recorded separately in milestone09-review.md.
