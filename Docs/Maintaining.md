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

## Documentation maintenance

Use the [documentation index](README.md) to distinguish active references
from tutorials and historical records. Keep reference examples in the
[downstream consumer check](../Scripts/ci/consumer_smoke.py), and verify the
actual public signatures rather than reproducing internal-only test usage.
The [contributor instructions](../CONTRIBUTING.md#documentation-changes)
describe the local checks.

When a design document becomes historical, label it and preserve its path
when source or external links depend on it. Prefer a current reference over
editing an old proposal into an ambiguous mixture of plan and implementation.
Keep dated verification records tied to their original revision and
environment; append a new record for new evidence.

Do not publish a sibling compatibility matrix without testing the named
revisions together. The Core consumer job does not exercise the whole
ecosystem. Repository setting claims likewise need dated live readback,
not just a local workflow diff.

### Claim ledger

Performance and robustness statements in the files listed under `regions` in
`Scripts/ci/claim_ledger.json` each carry one reviewed disposition: retained
(backed by adjacent code, a named test or a dated in-repository artifact),
corrected, an explicit unverified contract, removed (a tombstone that requires
the old text to stay absent), or not a claim. `Scripts/ci/check_claim_ledger.py`
runs in the repository-policy unittest step through
`Scripts/ci/tests/test_claim_ledger.py`. It fails on a stale anchor, a missing
named test, an uncovered statement that matches the ledger vocabulary, a
reintroduced removed statement or a forbidden phrase. Editing a covered comment
or string literal changes its fingerprint, so re-review the statement and update
its entry; `python3 Scripts/ci/check_claim_ledger.py --list PATH` prints the
statements the vocabulary matches, with their symbols and fingerprints.
A Markdown file named as a region or as a literal forbidden-phrase path is read
as prose, not lexed as Swift: a statement is one paragraph, list item, table
row, heading or fenced-code line, and its symbol is the heading chain.
`Docs/Performance_Guide.md`, `Docs/API_Overview_Map.md`,
`Docs/HowTo_NearestNeighbor.md`, `README.md`, `Operations.swift` and the
optimized vector files are regions. `CHANGELOG.md` is a section region
(`sectionRegions`): only statements under its `[Unreleased]` heading are counted,
an entry outside that section fails the check, and so does a listed section that
no longer exists. Every `acceptedContract` an entry names must be a key of the
ledger's `contracts` table, with a description and a source. One rule rejects per-operation times,
speed multipliers and percentage speedups, including spelled-out forms such as
"twice as fast", "three times faster", "N-fold", "an order of magnitude" and
"N percent faster", in that guide and in `README.md` unless the same sentence
carries unverified wording or cites an in-repository artifact path. Another
rejects source line-number citations (`File.swift:NNN`) in the benchmark
sources, `README.md`, `CHANGELOG.md`, the maintained documents in this
directory and the comments of every test file, because they go stale. A rule's
`commentOnlyPaths` checks only the comments of those files, so string fixtures
such as log locations stay legal.

Three rules cover library sources (`Sources/`). The first rejects qualitative
performance and robustness labels, which are listed in the rule's pattern, in
library sources, `README.md` and the maintained documents unless the same
sentence carries unverified wording or an in-repository artifact path; the
branch names "fast path" and "fast route" are allowed, and its one exemption is
unchanged runtime text owned by D02. The second applies the numeric rule above
to library sources, with a multiplier counted only when a speed word is attached
to it, so a 4x4 tile or 2× unrolling is not a claim. The third forbids the
undefined phrase for copy-free GPU import (`zero[- ]?copy`) in the same files;
name the mechanism instead, such as an `MTLBuffer` created with
`makeBuffer(bytesNoCopy:)` over page-aligned storage. Each rule's fixture table
is its exact scope: a phrasing outside the table may not be seen.

An unverified contract must carry one of the phrases "not verified",
"unverified", "not established", "not promised" or "no ... guarantee". A string
literal is runtime text, so for a literal the phrase may instead sit in one
sentence of its adjacent comment block (the comment lines directly above it, a
trailing comment, or the doc comment of the enclosing declaration), and that
sentence must also name `Q11` or say "runtime", "returned" or "printed"
followed by "text", "string", "output" or "note". A test that only checks a
record flag the note relates to (`evidence.flagTests`) or a drift guard that
hashes the source the note was written from (`evidence.bodyPin`) is recorded for
traceability and is never evidence for the note. A statement whose only evidence
is a record outside the repository stays an unverified contract: its entry cites
the record's path, sha256 and date, because a dated artifact must be in the
repository.

Forbidden phrases (`forbiddenPhrases`) list literal file paths or fnmatch globs;
an empty list, or a path that matches no file, fails the check. Each rule carries
must-fail fixtures (and may carry must-pass fixtures) that the check runs against
its pattern and its screen. One rule covers positive memory, RAM, bandwidth,
storage or footprint saving and reduction claims in every Swift file under
`Sources/`, `Tests/` and `Benchmarks/`; its fixture table is its exact scope.
Describe stored scalar payload as a payload model, not a memory saving. A claim
counts as hedged only when the hedge ("not measured", "not verified", "never
claimed", "unverified", ...) immediately follows it and ends its sentence, in
parentheses or as the clause's final words; a hedge elsewhere in the sentence
does not qualify the claim. Its two exemptions are unchanged runtime strings
owned by D02. Statements are matched with Swift escapes neutralized and
multi-line line continuations joined, so a leading `\n` in a print label does not
hide a claim, and a string literal that matches a forbidden phrase is a statement
even when it looks like an identifier. A tier-A word inside an identifier-shaped
literal that matches no forbidden phrase (for example `"optimized"` in a case
ID) is still not a statement; this is a documented limit. The caller inventory (`callerInventory`) pins the uses of
each flag-gated helper per file, including inside its defining file. Test
references are compared by the test they resolve to, so `File::Suite.test` and
`File::test` are one test; a drift guard listed in any `bodyPin` may be cited as
named-test evidence only by an entry listed in `bodyPinDescribers`.

The check is an inventory and lexical guard backed by a manual review. It cannot
tell whether a claim is true, it does not see a claim phrased without its
vocabulary or a paraphrase of a forbidden phrase, and it treats overloads that
share a name as one symbol. A named test reference is matched by its method name
only, and caller counts are lexical, so a homonym counts as a use. A named test
supports a claim only within that test's fixtures. The check reads the ledger,
the region and evidence files, every Swift file under `Sources/`, `Tests/`
and `Benchmarks/`, and the Markdown files named as literal forbidden-phrase
paths.

## Releases

Do not move or delete published version tags. For each new release, use a
SwiftPM-compatible semantic version tag such as `vX.Y.Z`, verify its commit and
CI, and test resolution/build from a clean downstream consumer. Explain API,
numerical behavior, compatibility, and security changes in release notes.

When a release renames `## [Unreleased]` to the version heading, retire the
`[Unreleased]` claim-ledger entries in the same change: delete every
`CHANGELOG.md` entry from `Scripts/ci/claim_ledger.json`, including its
tombstones, and add a new empty `## [Unreleased]` heading, so the section scope
still names an existing section. Released history is then a record, not a
region. Run `python3 Scripts/ci/check_claim_ledger.py` before tagging.

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
