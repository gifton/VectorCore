## Problem and change

<!-- Describe the concrete problem and resulting behavior. Link relevant issues. -->

## Validation

<!-- List commands and outcomes, including Swift/Xcode, platform, and Debug/Release.
     Report skipped tests and checks you could not run. For documentation-only
     changes, link/example validation may be sufficient locally; CI still runs. -->

## UMAP / graph / PCA reference evidence

<!-- Required for covered changes, including comments in those source areas;
     otherwise explain why out of scope. See Docs/UMAP_Reference_Checks.md.
     Missing or stale exact evidence blocks approval/merge. -->

| Run | Full revision | Production / fixture / runner / profile hashes | Host/profile | Debug cases/words | Release cases/words | Evidence |
|---|---|---|---|---|---|---|
| Before editing | | | | | | |
| After final change | | | | | | |

<!-- Record ordinary CI maximum deviations and any budget misses separately.
     New environment profiles need reviewed alpha/control re-capture. -->

## Compatibility and performance

<!-- Describe API/numerical behavior changes and migration, or write "Not affected".
     For performance claims, include Release measurements, baseline/current commits,
     hardware, warm-up, samples, and raw results. Measure allocations separately. -->

## Review checklist

- [ ] I reviewed the diff and updated affected documentation.
- [ ] I added appropriate regression coverage for behavior changes, or explained why it is not applicable.
- [ ] I considered non-finite values, tolerances versus exact ordering, and boundary inputs where relevant.
- [ ] I reviewed bounds, alignment, lifetimes, ownership, and concurrency for unsafe code changes, if any.
- [ ] I checked newly reachable paths when changing routing or configuration gates, if any.

<!-- Follow CONTRIBUTING.md and CODE_OF_CONDUCT.md. Report vulnerabilities through
     SECURITY.md before opening a public PR with sensitive details. -->
