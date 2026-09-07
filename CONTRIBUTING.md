# Contributing to VectorCore

Small fixes, reproducible bug reports, and focused pull requests are welcome.
For substantial API, storage, concurrency, or algorithm changes, open an issue
first to agree on scope and compatibility. Follow the
[Code of Conduct](CODE_OF_CONDUCT.md); report vulnerabilities through
[Security reporting](SECURITY.md), not public issues.

## Development workflow

1. Fork the repository, clone your fork, and create a branch from current `main`.
2. Use macOS with Xcode selected by `xcode-select` and Swift 6.0 or later.
   Check `swift --version` and `xcodebuild -version`; the
   [CI matrix](.github/workflows/ci.yml) names the toolchains used for merge checks.
3. Make a focused change and update affected documentation. Add a regression
   test for a behavior fix and meaningful coverage for new behavior.
4. Run the checks below, then open a PR against `gifton/VectorCore:main`.
   Describe the problem, resulting behavior, compatibility impact, and actual
   validation, including any checks you could not run.
5. Address maintainer feedback and resolve discussions. Keep the branch current
   with `main` when required by GitHub. Outside contributors may need a
   maintainer to approve workflow execution before checks run.

Run from the repository root:

```sh
swift test
swift test -c release
swiftlint version
swiftlint lint --config .swiftlint.yml
```

Use **SwiftLint 0.65.1** to match CI. Install that version from its upstream
release; an unpinned package-manager install may select a different version.
The repository contains both Swift Testing and XCTest tests; inspect both
summaries and report skips separately. Full tests mean all normally enabled
tests; extended and performance cases may remain opt-in. Targeted test filters
help during development but do not replace full Debug and Release checks.

`CI Required` is the stable merge gate. It aggregates the required jobs,
including tests, lint, repository checks, consumer examples, and platform
compilation. Timing benchmarks, sanitizers, and CodeQL are separate advisory
signals; inspect relevant failures even when they do not block the gate.

Local hooks are optional:

```sh
bash .github/hooks/install-hooks.sh
```

Review the installer before running it. Hooks provide early feedback; CI and
maintainer review remain necessary.

## Numerical and systems changes

Document formulas, complexity, failure modes, and stability tradeoffs where they
help review. Use explicit types in complex algorithms and follow nearby Swift
style. Performance claims need measurements; correctness claims must be
immediately evident from adjacent code, cite a test, or be labeled as an
unverified contract.

Choose tolerances from the operation's error budget, including absolute error
near zero and relative error at scale. Use exact comparisons when the contract
requires them, such as Top-K ordering, index identity, signed-zero bit patterns,
and serialization. An epsilon comparator is unsuitable for an ordering contract
because it can violate transitivity. Include relevant empty, singleton,
boundary-size, zero-norm, NaN, infinity, and overflow cases.

For pointers and buffers, review bounds, initialization, alignment, integer
arithmetic, aliasing, lifetime, ownership transfer, and concurrent access. Unsafe
closures must not leak borrowed pointers. Changes to `@unchecked Sendable`
types need an explicit account of synchronization and ownership. Use the
[alignment guide](Docs/Memory_Alignment.md) and the affected API's contract.

When a defect has occurred more than once, add a mechanical test covering the
class of failure. When changing a routing or configuration gate, inspect and
test every path newly reachable through that gate.

## Performance changes

Build and measure in Release. For example, from the repository root:

```sh
swift build -c release --product vectorcore-bench
.build/release/vectorcore-bench --suites dot --dims 512 --samples 5 --min-time 0.2 --run-seed 1 --format json --out /tmp/vectorcore-benchmark.json
```

Run a warm-up before collecting comparison runs; record the warm-up procedure.
Compare the same cases and seed on both revisions, use repeated samples, and
report medians and available percentiles with the raw results. Record commit
IDs, hardware, architecture, OS, Swift/Xcode versions, compiler flags, power
mode, and thermal conditions. Measure allocations separately with an allocation
profiler when making memory claims; elapsed time is not allocation evidence.
Shared-runner timing results are advisory. The
[Top-K benchmark](Benchmarks/TopKNaNContract/README.md) shows a targeted comparison
workflow.

## Compatibility and support

VectorCore is pre-1.0. Minor releases may introduce source or behavior changes;
patch releases aim to retain source compatibility, while bug fixes can change
incorrect numerical results. Explain any API or behavior change in the PR and
release notes. No stable ABI or blanket cross-provider bitwise reproducibility
is promised.

Maintenance focuses on the latest published release and `main`. Older releases
have no guaranteed backports. Issue triage, reviews, fixes, and security response
are best-effort, with no guaranteed response or resolution deadline. Deployment
targets in `Package.swift` are not a promise of runtime tests on every device;
see [Requirements and compatibility](README.md#requirements-and-compatibility).

Contributions use the repository's existing [MIT license](LICENSE). No CLA,
DCO sign-off, or commit-signing requirement is imposed by this contribution
policy. See [Maintaining](Docs/Maintaining.md) if you publish a fork.
