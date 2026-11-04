# Hosted CI preflight

The existing GitHub Actions matrix is active and accessible. The carried qualification candidate has not been published to a remote branch, so no hosted run currently qualifies these local repairs.

Read-only observations on 2026-11-04:

- Repository: `gifton/VectorCore`, public; default branch `main`.
- Local qualification input: `77defdf6161ab7600309d3432691fa9d388d088e` (the qualification plan commit; package inputs equal the preceding accepted candidate).
- Remote main: `7d9c962af774d9aae3cd2445d487ec5a941a7a65`.
- Local branch is 55 commits ahead and one commit behind that remote main. The upstream-only commit updates `actions/upload-artifact` from 7.0.1 to 7.0.2; it has not been merged into the carried branch by this pass.
- [CI run 41378600891](https://github.com/gifton/VectorCore/actions/runs/41378600891) passed all 14 jobs, including Xcode 16.2/26.3 Debug/Release, both sanitizers, consumers, Apple compilation and the required aggregate. Its source is the older remote-main SHA above. This confirms a recent available hosted configuration, not acceptance of the local candidate.

The local `.github/workflows/ci.yml` supports `workflow_dispatch`. Its required aggregate depends on tests, lint, repository checks, consumers, platforms, benchmark contracts and sanitizers. A push to a new qualification branch alone will not start this workflow: its push trigger is restricted to `main`; the manual dispatch is required unless a pull request is opened.

After local qualification and review, the next authorized qualification operation is to publish the exact reviewed candidate to the new `codex/qualification-0.4.0-2026-11-04` branch, then dispatch `ci.yml` on that ref. Record the final candidate SHA and ensure the returned run's `head_sha` matches before accepting results. This does not require a main-branch merge, tag or release. No remote mutation was performed during this read-only preflight; the candidate-specific hosted run is a separate execution record.

Retained raw API responses and bounded command records: `hosted-evidence/` in the qualification evidence archive. `hosted-preflight.json` captures the observed source/ref/workflow relationships and deliberately records `qualifies_local_candidate: false`.
