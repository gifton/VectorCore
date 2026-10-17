# N04 dispatch addendum (root, 2026-10-03). Binding together with briefs/N04.md

- Brief: `/private/tmp/vectorcore-stabilization-2026-09-18/Docs/superpowers/execution/2026-09-18-vectorcore-stabilization/briefs/N04.md` (approved, unchanged).
- Workspace (git worktree, detached at integration 0d0bf76, code identical to e6e1bae): `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-03/workspaces/N04`.
- Evidence root: `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-03/N04/`. Report: `.../N04/report.md`.
- Toolchain: macOS 27 / Xcode 27.0 / Swift 6.4 with the default swiftbuild build system. Record it.
- Run every swift build/test through the capture driver, using unique log, record and scratch names:
  `python3 /Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-02/build-tools/capture_swift.py --workdir <workspace> --log <ev>/<name>.log --record <ev>/<name>.json -- test [-c release] --jobs 6 --scratch-path <ev>/build-<name> --filter '<filter>'`
- Order: run the behavioral RED on unchanged production first, through the public entropy route and both entropyFast routes. A compile failure is not RED. Keep its log. Then implement, then run focused Debug and Release.
- Any Python fixture generator must give every subprocess call an explicit finite timeout. Run the program bounds scanner `python3 -B -m unittest discover -s Scripts/ci/tests -p 'test_child_process_bounds.py' -v`. Also run pinned SwiftLint on changed Swift files: `/Users/goftin/dev/gsuite/VSK/VectorCore/.stabilization-recovery/2026-10-02/toolchain/swiftlint-0.65.1/swiftlint lint --config <workspace>/.swiftlint.yml <files>`. Fix any new warnings in files you own.
- Mutation evidence is required. After GREEN, (a) temporarily reintroduce an absolute-input cutoff and (b) remove the NaN propagation, each in a scratch copy of the workspace (for example `rsync -a --exclude .build`). Show each mutation makes a named test fail. Never mutate the workspace itself.
- Another worker is running long builds concurrently. If a deadline-bounded test times out, re-run it once in isolation and report both outcomes.
- No git commits, staging, stash or other index operations. Do not dispatch subagents.
