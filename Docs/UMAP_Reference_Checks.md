# UMAP reference checks

These development checks protect the alpha-derived UMAP, graph and PCA outputs.
They do not qualify a release or change the compiler-support policy.

## Required before and after changes

Before editing UMAP, graph or PCA code, commit the pre-change revision and run:

```sh
python3 -B Scripts/ci/check_umap_reference.py --output "$TMPDIR/umap-reference-before"
```

After the last covered change, commit the candidate and run:

```sh
python3 -B Scripts/ci/check_umap_reference.py --output "$TMPDIR/umap-reference-after"
```

Each output directory must be new. Both commands run Debug and Release. A clean
tree, matching reviewed host profile, successful test process and complete exact
comparison inventory are required. A mismatch or missing check is a failure,
never a successful skip. Ordinary CI cannot replace these exact runs.

The scope includes all files under `Sources/VectorCore/ManifoldLearning/` and
`Sources/VectorCore/LinearAlgebra/`, their shared normalization kernels, package/build
settings, reference fixtures/harness, and any newly introduced dependencies that
can affect these routes. Comments-only edits in the two source directories follow
the same simple rule. Unrelated prose changes do not require numerical reruns.

Record both results in the PR: full before/after revisions, production-input,
fixture, runner and profile SHA-256 values, host identity, Debug/Release outcomes,
case/word counts and the retained evidence location. Both sides must use the same
reviewed profile and fixture/harness. A later covered edit invalidates the after
result. Missing, stale, failed or incomparable results prevent reviewer approval
and merge. This is a maintainer/reviewer requirement; GitHub's existing rules do
not automatically attest local execution.

If the harness itself changes, keep the previous approved oracle available for
both source revisions. Review added cases separately; do not remove or rewrite an
existing case to absorb a candidate change. An incompatible harness migration
needs explicit review, rather than two unrelated green runs.

Bootstrap exception for introducing this runner: commit only tests/harness first,
including the default coefficient pair; record that production inputs still match
`a676d9745dec60450a2f1ff86da94824eb8b4f5e`; run the before command; then change
production sources and run after. This does not retroactively satisfy the missing
tracked-execution condition in the earlier progress delivery.

## What must execute

`UMAPReferenceAssertions.compare` always checks shape, finiteness and absolute
tolerance. The runner sets `VECTORCORE_UMAP_CHECK_BASELINE_BITS=1` in the final
Swift child environment, after removing inherited test/provider controls. Exact
mode additionally requires actual Float words to equal the frozen words.

Each successful comparison writes a unique receipt containing the fresh run ID,
configuration, profile/fixture identity, provider, logical case, count and actual
words. The runner independently checks those words against the versioned profile.
It rejects duplicate, missing, unexpected, malformed or stale receipts and any
record whose mode is not `exact`. A green process exit or a flag banner is insufficient.

| Group | Cases per configuration | Compared Float words |
|---|---:|---:|
| Six fit routes × dimensions 2/5 × legacy/explicit Euclidean | 24 | 672 |
| Legacy/explicit Euclidean graph | 2 | 48 |
| Curve layout: two entries × two dimensions × two curve pairs | 8 | 112 |
| Default public coefficient pair | 1 | 2 |
| Total | 35 | 834 |

The 834 comparisons reuse 250 distinct reference words; they are not 834 independent
fixtures. Same-build handler/no-handler checks remain separate controls. The runner
tests and controlled Swift mutations exercise lost flags, empty selections and
sub-tolerance word changes. See `test_umap_reference.py` for receipt failure cases.

The initial versioned profile is
[`xcode27.2-m3max.json`](../Scripts/ci/umap-reference/xcode27.2-m3max.json):
Xcode 27.2 (27B52), Swift 6.4.2 (swiftlang-6.4.2.1.9, driver 1.168.6), arm64,
macOS 27.0.1 (26A434), Apple M3 Max / Mac15,9 and the default LAPACK provider.
It is deliberately narrow. No cross-environment exact-word promise is made.

## OS, Xcode or machine updates

Preserve old profiles and expected words. An environment update never authorizes
replacing the expected outputs with those of the candidate.

1. Save the current profile, pre-change revision and original before evidence.
2. On the new environment run the independent observation client in Debug and Release:

   ```sh
   python3 -B Scripts/ci/observe_umap_reference.py --configuration debug --output "$TMPDIR/umap-new-host-debug"
   python3 -B Scripts/ci/observe_umap_reference.py --configuration release --output "$TMPDIR/umap-new-host-release"
   ```

   The client extracts unchanged alpha `44c9ffaf145a59688f420023b28cdcf87c158070`
   and retained accepted snapshot `a676d9745dec60450a2f1ff86da94824eb8b4f5e` from Git.
   Each is built separately, and each binary runs in two fresh processes. No UMAP
   implementation is copied into the client. Alpha legacy routes anchor the new
   default/explicit Euclidean routes; the alpha does not expose the new selectors.
   The public coefficient pair is anchored to the accepted snapshot, because the
   alpha has no public curve symbol. Captures and failures are retained even when
   historical tolerance checks fail. Run the same fixture client against the saved
   pre-change revision as well, using an isolated checkout.
3. Compare every graph/layout word from alpha, accepted snapshot, saved base and
   candidate on the new environment. Require exact agreement and repeated-process
   agreement in both configurations. Compare public coefficients to the accepted
   snapshot separately. Check discrete graph identities and provider identity too.
4. If the only difference is between the old and new environments, propose a **new**
   named profile in a review PR. Include actual OS/compiler/hardware identities,
   fixture/harness hashes, captures, and measured differences. Preserve the old
   profile. A profile is usable for the merge rule only after review/approval.
5. If unchanged alpha cannot build, repeats differ, or the candidate disagrees with
   its control, investigate. Do not patch alpha, rebaseline from candidate alone,
   widen tolerances or report a pass. Use the original supported environment if
   available; otherwise the exact merge condition remains unmet.
6. After approval, run paired base/head checks with `--profile` selecting the new
   tracked profile. For an update during a PR, retain the chronological pre-edit
   pass and add paired replays on the new environment, labeled as replays. Missing
   original pre-edit evidence requires an explicit owner exception.

This procedure supports additional reviewed machines without making the check
permanently dependent on one physical Mac. It is not compiler/release admission.

## Ordinary CI measurements and fixed budgets

The existing macOS 15 matrix runs Xcode 16.2/26.3 in Debug/Release. The observation
step records actual host/provider identities, candidate/source/fixture hashes,
repeated alpha/control/candidate captures, and a per-case maximum absolute
deviation against both historical words and the same-runner control. It includes
the worst index, ULP diagnostics, and discrete graph identity in its report.

Budgets chosen before measurement are absolute `1e-4` for layout coordinates,
`1e-5` for the compact six-dimensional graph-distance fixture, and `1e-6` for the
default coefficient pair. These are fixture budgets, not general numerical error
bounds. Both historical and same-runner comparisons must fit their budget.

A nonfinite value, missing case, shape/index change or budget miss fails the lane
and blocks cleanup readiness. Report the measured maxima in the PR; local zero
deviation does not establish CI coverage. Investigate misses with the same-runner
alpha. Do not automatically enlarge a limit, omit a route or skip a comparison.
An environment-specific reference or a changed numerical policy needs a separate
justified review. A control that cannot run is unassessed coverage, not a pass.

Keep raw local captures/logs outside Git. CI uploads compact captures, summaries
and deviation tables; do not commit full build trees or raw logs.
