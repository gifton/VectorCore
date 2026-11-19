# 0.4.0 release evidence

Start with [release status](../RELEASE_STATUS.md) for current support and remaining
gates. The documents below distinguish the adopted policy and bounded results
from earlier investigations. Archiving a record does not qualify a new candidate
or admit a release.

## Current policy and results

| Record | Purpose |
|---|---|
| [Platform coverage decision](reviews/2026-11-17-platform-coverage/decision.md) | iOS 18 minimum and required runtime checks, deferred platform scopes and fresh-candidate status |
| [Platform qualification results](reviews/2026-11-17-platform-coverage/results.md) | Fresh native, iOS 18/current simulator and stable physical-iOS results; private evidence digests and remaining gates |
| [Compiler acceptance decision](reviews/2026-11-10-compiler-acceptance/decision.md) | Adopted support restrictions, numerical requirements and admission boundaries |
| [Frozen candidate](reviews/2026-11-10-compiler-acceptance/stable-candidate.md) | Local macOS, simulator and physical-device results, with remaining decisions |
| [Hosted qualification results](reviews/2026-11-10-compiler-acceptance/hosted-evidence-2026-11-16.md) | Reviewed hosted profiles and the collection/verification runs |
| [Hosted compiler identity](reviews/2026-11-10-compiler-acceptance/hosted-compiler-identity.md) | Exact adopted hosted compiler pair and byte comparison |
| [Hosted workflow instructions](reviews/2026-11-10-compiler-acceptance/hosted-qualification.md) | Two-stage evidence collection and reviewed verification |
| [Qualification profiles](../Scripts/ci/qualification_profiles/README.md) | Exact profile selection and verification inputs |
| [SIMDStorage migration](SIMDStorage_Migration.md) | Consumer and conformer migration requirements |

The retained qualification results still report `compilerProfile=known-failing`
and `releaseAdmitted=false`. The compiler defect remains visible; these results
do not extend support to arbitrary helper/conformer compositions.

## Historical investigations

Longer dated write-ups now link to their full immutable snapshots. Their original
paths and section headings remain available. Large historical JSON reports use
the same checksum-verified archive references as qualification profiles. See
[evidence recovery](evidence-storage.md) to read the complete records.

| Investigation | Retained entry point |
|---|---|
| October 27 audit repairs and storage measurements | [Repair record](reviews/2026-10-27-audit-repairs/README.md) |
| October 28 oracle and progress repairs | [Oracles](reviews/2026-10-28-oracle-repairs/README.md), [progress](reviews/2026-10-28-progress-repair/README.md) |
| October 29 remaining repairs and coverage | [Repair record](reviews/2026-10-29-remaining-repairs/README.md), [coverage inventory](reviews/2026-10-29-remaining-repairs/COVERAGE.md) |
| November 1 compiler reproduction and core repairs | [Compiler instructions](reviews/2026-11-01-compiler-qualification/README.md), [repairs](reviews/2026-11-01-core-remediations-batch/README.md) |
| November 2 readiness investigation | [Readiness record](reviews/2026-11-02-readiness-batch/README.md) |
| November 4 preparation, remediation and hosted qualification | [Preparation](reviews/2026-11-04-release-preparation/README.md), [phase one](reviews/2026-11-04-release-phase-one/README.md), [phase two](reviews/2026-11-04-release-phase-two/README.md), [qualification](reviews/2026-11-04-release-qualification/README.md) |
| November 5 remaining release gates | [Gate record](reviews/2026-11-05-release-gates/README.md) |
| November 6 bounded qualification and buffer migration | [Qualification](reviews/2026-11-06-bounded-native-qualification/README.md), [migration](reviews/2026-11-06-buffer-api-migration/README.md) |
| November 6 residual exposure and compiler research | [Workaround audit](reviews/2026-11-06-workaround-audit/README.md), [workaround qualification](reviews/2026-11-06-workaround-qualification/README.md), [research](reviews/2026-11-06-compiler-research-handoff/README.md) |

Executable reproducers, reusable qualification tooling and test fixtures remain
in the current tree. The iOS capability inventory also remains inline because
CI checks its entries against individual test gates. Its compact formatting does
not change those entries.

The benchmark [numerical bound analysis](reviews/2026-11-04-release-phase-two/benchmarks/dot-bound-analysis.md)
and [quiet-host procedure](reviews/2026-11-04-release-phase-two/benchmarks/quiet-host-procedure.md)
remain available in full for ongoing work.

## Storage boundary

This reduction affects current files and the PR diff. Published Git history is
unchanged, and the maintainer retains private snapshots of the original trees.
Archive references identify exact published bytes; a previously edited portable
presentation is still a presentation, not an original raw receipt. The
[publication catalog](evidence-publication.json), itself an archive reference,
locates the raw originals preserved during the earlier path cleanup.
