# GitHub and public-readiness review

Reviewed September 6, 2026 (America/Los_Angeles), September 7 UTC.
Repository: [gifton/VectorCore](https://github.com/gifton/VectorCore).

VectorCore already has useful safeguards, but its merge checks, test coverage,
ownership configuration, and release naming need attention before promoting it
to a wider audience. This document records verified findings and a proposed
rollout. It does not apply GitHub settings or change package behavior.

## Evidence and scope

- Authenticated, read-only GitHub REST inspection with repository admin access.
- Remote `main`: `fca4b602383589c46b627d8a0de2b6b2a68cd07d`.
- Local HEAD: `a508562009b5ffbcf1e646ef08c3caf18e3d98d0`. GitHub's compare API
  reports one additional commit on remote main and **no file differences**.
- Reviewed every active workflow, CODEOWNERS, Dependabot, issue/PR templates,
  package manifest, README, release metadata, and recent completed CI jobs.
- Downloaded the comprehensive job log for [PR CI run 34074267969](https://github.com/gifton/VectorCore/actions/runs/34074267969).
- Final readback confirmed [main CI run 34088443832](https://github.com/gifton/VectorCore/actions/runs/34088443832)
  completed successfully on the published commit. This is the existing filtered
  CI, with the coverage limitations described below.
- Reproduced version-tag resolution using Swift 6.3.3 in a throwaway local
  package. Parsed all three issue templates with Ruby's YAML parser.
- Did not perform a full source vulnerability audit, rerun the full test suite,
  inspect secret values, or audit account-level 2FA/recovery, installed GitHub
  App grants, deploy keys, environments, or all historical commits for secrets.
  Repository settings do not establish that those account/access controls are safe.

## Current settings

| Control | Observed state | Assessment |
| --- | --- | --- |
| Visibility and license | Public, MIT; forking enabled | Appropriate for contribution and reuse |
| Default branch protection | Active `protect` ruleset, default branch only | PR required; deletion and force pushes blocked |
| Required checks | None | Failed CI does not prevent merging |
| Human review | Zero approvals; code-owner and conversation-resolution requirements off | No enforced human review gate |
| Ruleset bypass | No bypass actors; current admin cannot bypass | Preserve an unbypassable CI baseline |
| CODEOWNERS | 16 `Unknown owner` errors from GitHub | Every owner is the placeholder `@yourusername` |
| Direct collaborators | Only `gifton`, admin | Avoid rules that make solo maintenance impossible |
| Actions token | Default read; Actions cannot approve PR reviews | Good baseline; also declare permissions in workflow files |
| Allowed Actions | All; SHA pinning not required | Restrict after workflow references are pinned |
| Fork workflow approval | First-time contributors | Consider approval for all outside collaborators |
| Runners and webhooks | Zero repository self-hosted runners; zero webhooks | Active workflows use GitHub-hosted macOS runners |
| Secret scanning | Enabled | Keep enabled |
| Secret push protection | Enabled | Keep enabled |
| Private vulnerability reporting | Disabled | No GitHub private intake channel |
| Dependabot alerts | Disabled; API explicitly returned that state | Enable alerts |
| Dependabot security updates | Disabled | Enable alongside alerts |
| Dependabot version updates | Weekly Actions configuration exists | Reviewer is still a placeholder |
| CodeQL default setup | Not configured; no active CodeQL workflow in tree | Add tested scanning in a subsequent pass |
| Release immutability | Disabled; latest five releases report mutable | Establish release integrity after correcting version naming |
| Tag rulesets | None returned | Published version tags have no ruleset protection |
| Branch cleanup | Automatic deletion after merge disabled | Optional hygiene improvement |
| Discussions / wiki | Both disabled; Issues enabled | Issues are sufficient initially |

The legacy branch-protection endpoint returns “Branch not protected,” but
`main` **is protected by a ruleset**. The effective branch-rules API and
[ruleset 10525559](https://github.com/gifton/VectorCore/rules/10525559) confirm
the controls above. Do not replace this with a claim that main is unprotected.

## Findings, ordered by urgency

### 1. The documented installation requirement cannot select the 0.3.3 release

[README installation](https://github.com/gifton/VectorCore/blob/fca4b602383589c46b627d8a0de2b6b2a68cd07d/README.md)
requests `from: "0.3.3"`. The complete GitHub tag inventory contains `vc0.3.3`
and `v0.3.2`, but no `v0.3.3` or `0.3.3`. The
[published 0.3.3 release](https://github.com/gifton/VectorCore/releases/tag/vc0.3.3)
points to remote main.

A local fixture with tags `v0.3.2` and `vc0.3.3` failed resolution with:

```text
no versions of 'dependency' match the requirement 0.3.3..<1.0.0
```

Adding `v0.3.3` to the same fixture commit made resolution succeed at 0.3.3.
This verifies the naming behavior locally; it is not a fresh network-based
consumer build of VectorCore.

Proposed repair: create the canonical `v0.3.3` tag at the already-published
commit `fca4b602383589c46b627d8a0de2b6b2a68cd07d`, after rechecking its CI.
Retain the existing `vc0.3.3` reference for anyone already using it. Prepare a
canonical release and explain the duplicate tag in its notes. Before publishing,
verify a clean consumer resolves and builds the README dependency. Do not move
an existing published tag to a different commit.

### 2. CI is informational rather than an enforced merge condition

The current ruleset contains deletion, non-fast-forward, pull-request, and
Copilot-review rules; it has no `required_status_checks` rule. Copilot review
does not replace a passing build or a maintainer's review.

Observed successful check names are `Build (Debug)`, `Test Minimal`,
`Test Comprehensive`, `Test (Release, smoke)`, and `Perf Smoke (artifact only)`.
Do not blindly require all existing checks: the separately named
`Run SwiftLint` job has workflow path filters, and performance measurements
should not become timing gates on shared runners.

Proposed repair: after CI coverage is corrected, require a stable `CI Required`
aggregate check, associated with the GitHub Actions app. Its job must run even
when a dependency fails or is skipped and must explicitly reject failed,
cancelled, or unexpectedly skipped required jobs. Trigger the workflow for every
PR to main, including documentation-only PRs. Test successful, failed, and
skipped-job scenarios before enforcing it. Require the branch to be up to date
with main and require conversation resolution.

### 3. The comprehensive job silently omits most of the modern test inventory

[ci.yml](https://github.com/gifton/VectorCore/blob/fca4b602383589c46b627d8a0de2b6b2a68cd07d/.github/workflows/ci.yml#L122)
uses three suite allowlists. The completed job ran **105 + 107 + 96 = 308**
Swift Testing tests and zero XCTest cases. In comparison, the checked-in
[September 6 verification record](verification-topk-nan-contract-2026-09-05.md)
reports 1,138 Swift Testing tests and 91 XCTest cases for full local runs,
including their reported skips. Different toolchains/environments mean these
totals are context, not an exact calculation of missing CI test cases.

The filters demonstrably exclude `TopKNaNContractTests`, `TopKNaNWrapperTests`,
`TopKTieBreakingTests`, provider parity, GEMM routing, pointer seams, and many
numerical, serialization-edge, and quantization suites. `DistanceMetricsSuite`
also does not match the actual `VectorDistanceMetricsSuite` name. The workflow's
claim that everything except MemoryPool ran is inaccurate. Compilation of a
test file in a log is not evidence that its tests executed.

Proposed repair: run the full normally enabled suite in Debug and Release.
Keep extended/performance cases opt-in and keep timing comparisons advisory.
Recheck MemoryPool on the CI toolchain; local success alone does not disprove the
old CI hang. If quarantine is still necessary, use explicit, narrow exclusions
with a documented reason and follow-up, rather than an inclusion list. Retain
job timeouts. Verify discovered/executed/skipped counts for both test frameworks.
If tests must be sharded, mechanically verify that discovered tests are assigned
exactly once or explicitly quarantined. New suites must enter CI automatically.

### 4. Review routing and contributor intake contain unfinished scaffolding

GitHub rejects every entry in `.github/CODEOWNERS`. Replace the entire repeated
placeholder mapping with `* @gifton`; add path-specific owners only when there
are distinct maintainers. Validate with GitHub's CODEOWNERS errors endpoint
after the file lands. Remove the placeholder Dependabot reviewer; omit the
optional reviewer field unless an actual supported reviewer is needed.

The performance issue template has invalid YAML:

```yaml
labels: 'performance', 'regression'
```

Use a supported scalar such as `labels: "performance, regression"`. Its
reproduction instructions also reference nonexistent `Scripts/build_optimized.sh`.
Replace those with the real Release benchmark command. Bug and feature template
frontmatter parsed successfully. Validate every issue template, not just this
one, to prevent this error class from recurring.

`CONTRIBUTING.md`, `SECURITY.md`, and a code of conduct are absent. GitHub's
community-profile endpoint reports 57%; that is a discoverability metric, not
a security score. The endpoint reports no issue template despite files existing;
the API response alone does not establish that all templates are unusable.

### 5. Workflow supply-chain controls should travel with the repository

All active workflows use mutable Action tags. They omit explicit token
permissions and checkout credential persistence settings. The repository's
read-only default currently limits token privileges, but a fork may use
different repository defaults.

Proposed repair across **all** active workflows:

- Declare `permissions: { contents: read }` and disable checkout credential
  persistence for build/test jobs.
- Pin each Action to a verified full commit SHA with its version in a comment;
  retain Dependabot updates. Verify the commit belongs to the intended upstream.
- Use an explicit supported runner/toolchain, bounded job timeouts, and
  concurrency cancellation for obsolete PR runs. SwiftLint currently runs
  twice and repeatedly invokes the linter; consolidate and version its tool.
- Review the broad `.build`/DerivedData cache restore prefixes. Include
  architecture, Xcode/Swift version, and build configuration in cache identity;
  avoid sharing analysis/release artifacts with unrelated builds. This audit
  did not establish a cache-poisoning exploit.
- Preserve ordinary `pull_request` execution on hosted runners. There is no
  active `pull_request_target` workflow to fix.

After pinned workflows pass, enable the repository SHA requirement and allow
only the Actions actually used. Enabling enforcement first would break the
current workflows. GitHub recommends full-SHA pins and narrowly scoped token
permissions in its [secure-use reference](https://docs.github.com/en/actions/reference/security/secure-use).

### 6. Security reporting, dependency alerts, and release integrity are incomplete

Enable private vulnerability reporting and link `SECURITY.md` and the issue
chooser to `https://github.com/gifton/VectorCore/security/advisories/new` only
after verifying that intake is enabled. Define what information to include,
which versions receive fixes, and that handling is best-effort; do not invent
a guaranteed response deadline or an unverified private email address.
See [GitHub's private-reporting guidance](https://docs.github.com/en/code-security/how-tos/report-and-fix-vulnerabilities/report-privately).

Enable Dependabot alerts and security updates. Zero third-party package
dependencies do not remove the separate GitHub Actions dependency surface.
Add CodeQL for Actions and the Swift/C sources after verifying analysis on a
supported macOS toolchain. Start advisory and inspect the baseline before
making it required. A Swift build and a C analysis need actual extraction;
do not assume a generic Ubuntu/no-build configuration covers this package.
See [CodeQL build modes](https://docs.github.com/en/code-security/how-tos/find-and-fix-code-vulnerabilities/manage-your-configuration/codeql-for-compiled-languages).

After correcting canonical release naming, enable immutable releases for
future publications and protect version tags against updates/deletion with a
tag ruleset. Enabling immutability is not retroactive. Prepare release assets
in a draft before publication; see [immutable releases](https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases)
and [enabling immutability](https://docs.github.com/en/enterprise-cloud%40latest/code-security/how-tos/secure-your-supply-chain/establish-provenance-and-integrity/prevent-release-changes).

### 7. Public documentation overstates some guarantees

Before promoting the README, reconcile these claims with adjacent code/tests:

- “Pure Swift” conflicts with the `VectorCoreC` target and its C kernels.
- “All core operations avoid heap allocations” is too broad: generic
  `DimensionStorage` defaults dimensions above 16 to managed heap storage.
- “Buffer pooling via actor” conflicts with `MemoryPool`, which is a final
  class with `@unchecked Sendable`. This is a documentation mismatch, not a
  finding that its synchronization is broken.
- Blanket Sendable/data-race and speedup claims need narrower scope and named
  evidence. Existing benchmark records explicitly do not measure allocations.

Document tested platforms separately from deployment targets, the pre-1.0
compatibility policy, expected behavior for non-finite numbers, and caller
responsibilities around unsafe buffers. Compile the introductory examples in
a small consumer fixture. Avoid presenting untested Linux support; the issue
template currently lists Linux while the package imports Accelerate directly.

## Proposed contribution model

Default recommendation pending owner preference: retain solo maintenance while
making external contributions pass CI and receive maintainer review.

Use two layers if enforced code-owner approval is desired:

1. Baseline main ruleset: PRs, no deletion/force push, required `CI Required`,
   up-to-date branch, resolved conversations, and **no bypass actors**.
2. Separate review ruleset: one approval, code-owner review, stale approval
   dismissal, with an admin bypass limited to pull requests. This lets the sole
   owner merge their own PRs while the baseline still enforces CI. Treat bypass
   as an auditable maintainer exception; technically it can also bypass review
   on an outside contribution.

Do not add an admin bypass to a combined ruleset containing required CI. Do not
require independent approval with no solo-maintainer exception until another
trusted reviewer exists. An alternative is zero enforced approvals while only
the owner has write access, with review as maintainer policy; that is simpler
but provides less mechanical protection. Rulesets compose with the most
restrictive applicable rules; see [GitHub rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets).

Require workflow approval for all outside collaborators if the owner accepts
the extra approval step. This controls when contributor code runs; it does not
replace code review. Read the proposed workflow and code before approving it.

## Concrete rollout and acceptance checks

| Order | Work | Acceptance evidence |
| --- | --- | --- |
| 1 | Repair canonical 0.3.3 tag/release naming | Exact target SHA verified; published-commit CI passes; clean consumer resolves 0.3.3 and builds |
| 2 | One PR fixing CODEOWNERS, Dependabot placeholder, issue templates, README claims; add contributor/security guidance | All template YAML parses; no owner placeholders; local links resolve; public examples compile; review support commitments |
| 3 | Correct full Debug/Release test coverage, harden all workflows, add stable aggregate check | Successful PR run includes newly covered suites; failure/skip cases fail aggregate; documentation-only PR gets aggregate; Actions pins verified |
| 4 | Apply main/review rulesets and Actions restrictions after new checks exist on main | Read back effective rules and settings; pending/failing check blocks merge; solo-owner path still requires CI |
| 5 | Enable reporting, dependency alerts/updates, tag protection and future release immutability | Read back each setting; private report link available; tag protections target intended versions |
| 6 | Add baseline CodeQL, targeted sanitizers, and supported-toolchain/platform checks | Actual analysis results; sanitizer findings triaged; supported configurations demonstrated before required status checks are added |

Contributor guidance should include fork/branch/PR steps, Swift/Xcode
requirements, `swift test`, `swift test -c release`, linting, Release benchmarks
with warm-up/hardware/sample information, regression-test expectations,
numerical tolerances versus exact ordering contracts, and how to discuss API
changes before substantial work. Hooks should be optional: the current installer
assumes `.git` is a directory and is not suitable for linked worktrees as written.
Document fork-specific changes to CODEOWNERS, security intake, badges, release
identity, and upstream links so a fork does not inadvertently direct its users
to the original maintainer.

A short code of conduct can follow once the maintainer chooses an actual private
enforcement contact. Keep support capacity and compatibility promises realistic.
No CLA, mandatory DCO, signing requirement, new license, or organizational move
is needed for this first pass.

## Audit artifacts and unperformed actions

The downloaded job log is at
`/private/tmp/vectorcore-ci-comprehensive-audit.log`. The local SwiftPM fixture
is at `/private/tmp/vectorcore-tag-audit-h6_h4sx9`; it contains only throwaway
packages and locally created tags. These temporary artifacts are not committed.

Only this review document was added to the project. No GitHub setting, remote
tag, release, branch, collaborator, or workflow was changed; no messages or
issues were sent. Existing untracked `.antigravitycli/` and `GEMINI.md` were
left untouched. This is a preparation review, not a claim that VectorCore is
now hardened or that its source is vulnerability-free.
