# Maintaining VectorCore and publishing forks

## Reviewing and merging

Review the diff and any workflow changes before approving an outside
contributor's Actions run. Use ordinary pull-request workflows on hosted
runners; do not give untrusted contributor code repository secrets or write
credentials. Check that new tests are discovered, not merely compiled.

The main-branch baseline requires a PR, an up-to-date branch, resolved
conversations, and the stable `CI Required` check from GitHub Actions. Keep this
baseline free of bypass actors. A separate owner-review rule can require one
code-owner approval and dismiss stale approvals while allowing a PR-only admin
bypass for solo maintenance. That exception must not bypass the CI baseline;
use it sparingly and record the reason on the PR. Copilot review is additional
feedback, not a substitute for maintainer judgment.

When editing CI, keep `CI Required` stable and run it on every PR, including
documentation changes. Its aggregation must reject failed, cancelled, missing,
or unexpectedly skipped required jobs. Check full Debug and Release summaries
for both XCTest and Swift Testing, recording disabled/skipped cases explicitly.
Do not replace full discovery with suite inclusion lists. If a quarantine is
needed, name the excluded case, reason, and follow-up.

Keep Action references pinned to verified full commit SHAs, version comments
current, token permissions explicit, and checkout credential persistence off.
Review Dependabot changes as executable dependency updates. Keep SwiftLint's
version and download checksum in sync with contributor instructions. Scan
results and timing measurements remain advisory until their baseline is
reviewed; investigate findings and document limits instead of treating a green
badge as a source audit.

Repository policy is enforced by live GitHub settings as well as files. After
changes, read back effective rules, required check names/app identity,
CODEOWNERS errors, reporting availability, dependency alerts/updates, and
Actions restrictions. Keep the allowlist aligned with Actions actually used.

## Releases

Do not move or delete published version tags. For each new release, use a
SwiftPM-compatible semantic version tag such as `vX.Y.Z`, verify its commit and
CI, and test resolution/build from a clean downstream consumer. Explain API,
numerical behavior, compatibility, and security changes in release notes.

Prepare assets and notes in a draft before publication. Tag update/deletion
protection and release immutability complement each other; enabling immutability
does not make earlier releases immutable. Verify protections and the final
published state. Coordinate publication with the repository owner; keep release
identity and license changes separate from routine contributions.

Maintenance is best-effort and focuses on the latest release and `main`;
[CONTRIBUTING.md](../CONTRIBUTING.md) and [SECURITY.md](../SECURITY.md) define the
public support scope. Do not publish response deadlines that cannot be met.

## Fork checklist

Before promoting a maintained fork, review these identities and settings:

- Replace `.github/CODEOWNERS` with real maintainers who can review in the fork.
  Remove unavailable teams and verify GitHub reports no ownership errors.
- Update README badges, package URLs, issue links, contribution destinations,
  and release links to make the fork's relationship with upstream clear.
- Configure the fork's security intake, then update `SECURITY.md` and the issue
  chooser contact link. Provide a real private conduct-reporting contact in
  [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md).
- Review Actions permissions, SHA pins, allowed Actions, outside-contributor
  approvals, secrets, repository rules, and required checks. Verify them on the
  fork; do not assume upstream settings were inherited.
- Configure Dependabot, security scanning, supported platforms, and support
  commitments for the fork's actual capacity. Preserve upstream attribution
  and the existing license obligations.
- Establish the fork's own release namespace and package identity; explain
  compatibility and migration. Do not imply upstream maintainers support it.

To contribute upstream without publishing a separate product, keep the PR
focused and avoid committing fork-specific identity changes to upstream.
