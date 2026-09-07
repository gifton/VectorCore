# GitHub hardening implementation plan

> Execute the approved readiness review in independent tasks; review and verify before integration.

**Objective:** Apply all recommendations in `GitHub_Readiness_Review_2026-09-06.md`
except release naming, which the owner has already corrected.

**Architecture:** Repository-owned workflows provide full tests and a stable
aggregate check. GitHub rules enforce that check separately from review policy.
Contributor documentation explains the verified package and its support limits.

**Constraints:** Keep Swift tools 6.0 and existing deployment targets. No version
or release-tag changes. Preserve unrelated untracked files. No package API or
numerical algorithm changes unless verification identifies an actual defect.
Publish no guaranteed support SLA or unconfirmed conduct-report address.

## 1. Contributor documentation and ownership

- [x] Replace placeholder CODEOWNERS with `* @gifton`; remove Dependabot reviewer placeholders.
- [x] Repair issue frontmatter and real benchmark commands; add private security intake link after enabling reporting.
- [x] Add CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md and maintainer/fork guidance.
- [x] Narrow README allocation, language, concurrency and performance claims to evidence.
- [x] Compile README Quick Start verbatim through a downstream consumer fixture.

Files: README.md, CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md,
Docs/Maintaining.md, .github/CODEOWNERS, .github/dependabot.yml,
.github/ISSUE_TEMPLATE/*, .github/PULL_REQUEST_TEMPLATE.md.

## 2. Mechanical regression checks and hooks

- [x] Add behavioral tests for aggregate success/failure/cancellation/skips/missing results.
- [x] Implement Scripts/ci/check_required.py with expected job IDs
  `test`, `lint`, `repository-checks`, `consumer`, `platforms` supplied as arguments;
  consume GitHub `needs` JSON through environment variable `NEEDS_JSON`.
- [x] Add Scripts/ci/validate_github.rb to parse all GitHub YAML and template frontmatter,
  validate immutable action references, explicit permissions and disabled checkout credentials.
- [x] Fix the optional hook installer to respect git's actual hooks path and existing hooks.
- [x] Verify hook execution with filenames containing spaces and fail on linter failures.

Tests: Python unittest for the gate; fixture-based Ruby validator checks; isolated
temporary Git repositories and linked worktrees for hook installation/execution.

## 3. Workflows

- [x] Replace suite allowlists with full Debug/Release testing on supported Xcode versions.
- [x] Consolidate lint into CI with a fixed SwiftLint binary and checked SHA-256.
- [x] Use clean builds instead of broadly restored compiled-artifact caches.
- [x] Pin every Action to an upstream-verified full SHA; disable checkout credentials.
- [x] Add always-running `CI Required` aggregate with no workflow path filters.
- [x] Add consumer Quick Start and Apple platform compile checks.
- [x] Add advisory CodeQL Actions/Swift/C scans and focused address/thread sanitizer jobs.
- [x] Preserve bounded manual Release benchmarks and advisory PR benchmark artifacts.
- [x] Validate YAML/policy, run local checks, then exercise actual GitHub workflows.

Files: .github/workflows/ci.yml, benchmarks.yml, codeql.yml, sanitizers.yml,
Scripts/ci/*; remove redundant SwiftLint workflow and obsolete workflow archive.

## 4. Live settings and integration

- [x] Save before-state snapshots outside the repository.
- [x] Enable private reporting, Dependabot alerts and updates, all-outside-contributor
  workflow approval, future immutable releases, and branch cleanup; read back results.
- [x] Review and open the implementation PR; resolve CI/scanner/sanitizer findings.
- [x] Merge verified workflows before enabling required checks or Action SHA enforcement.
- [x] Update baseline ruleset: PR, resolved conversations, deletion/force-push blocks,
  strict `CI Required` from GitHub Actions, no bypass.
- [x] Add separate owner-review ruleset with PR-only admin bypass for solo maintenance.
- [x] Protect version tags against updates and deletion; allow new version creation.
- [x] Restrict Actions to the exact action repositories in use and require full SHAs.
- [x] Verify effective rules, CODEOWNERS, reporting, scanning and Actions settings.
- [x] Exercise a documentation-only PR and verify pending/failing gate rejection without
  merging a deliberately failing change. Record proof in a final verification document.

## 5. Acceptance record

- [x] Independent review of workflow, script and policy changes.
- [x] Document actual test/analysis/platform results and any bounded limitations.
- [x] Verify final tree, effective remote state and required-check enforcement.
