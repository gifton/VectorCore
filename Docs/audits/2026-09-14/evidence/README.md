# Validation evidence and replay

[Audit guide](../README.md) · [Claim register](../CLAIMS.md) · [Source/provenance manifest](manifest.json)

## Evidence boundaries

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| Actual-library clients in `validate_ownership/`, `validate_numerics/`, `validate_execution_api/`, plus `root/SerializationProbe.swift` | Behavior of the selected local Debug/Release objects on explicit fixtures. Internal-only probes use `@testable import` where needed. | Every platform, every route, sanitizer cleanliness or performance at scale. |
| `validate_ownership/hybrid-release-excerpts.txt` | Actual linked Release retain/uniqueness/copy paths for the inspected specialization. | Universal allocation totals across compilers and storage modes. |
| `legacy-counters/`, replay logs and counter-source audit | Isolated current-source copy/allocation mechanism; instrumentation is described in ownership validation. | Exact production allocation counts without corroboration. |
| `validate_execution_api/ExecutionPoolProbe.swift` | Current pool source with a counted real deallocator; weak owner teardown and cache-budget observations. | RSS, all generic element lifetimes or active-handle policy. |
| `root/main.swift` | Concatenated unchanged benchmarking helpers, removing redundant package imports; finite/shape statistics and A/B behavior. | Injected faulty production kernels or complete CLI numerical validation. |
| `root/cli-results.json` and CLI logs | Commands, exit codes, selected-case counts, requested/effective metadata mismatch and artifact-write failure behavior. | Useful throughput measurements: the filtered cases intentionally execute no timed work. |
| Source reasoning and sizing equations in the register | Reachability, ownership obligations, algebra, asymptotic work and logical payload sizes. | Measured latency, allocation count, peak RSS, deployed incidents or effects of proposed fixes. |

These are **observation programs**. A normal exit can accompany a printed incorrect result. Some special arguments deliberately reproduce traps or nontermination and were run in isolated, bounded subprocesses. Read each source and its recorded command before replaying those modes. The regular test suite was not modified by this audit.

## Fresh baseline

From the reviewed repository root:

```sh
swift test -c release
python3 -m unittest discover -s Scripts/ci/tests
ruby Scripts/ci/validate_github.rb
```

The [Release log](https://github.com/gifton/VectorCore/blob/b2709f5b8c6a171a843813a968e5ca3cfa2e94d3/Docs/audits/2026-09-14/evidence/release-baseline.log) ends with 1,144 Swift Testing tests in 168 suites passing in 22.567 seconds. XCTest reports 91 cases, 27 skipped, no failures. The Python check returned exit 0 with 25 tests in 6.818 seconds; the Ruby check returned exit 0 after validating nine YAML/template files. These are correctness/configuration checks, not controlled performance samples. Normal baseline execution did not set the extended-performance environment flag.

Earlier same-revision evidence is preserved in the [first-pass evidence directory](../../../reviews/2026-09-14/evidence/): Debug/Release observation clients, the original report's Debug/consumer/lint verification record, and initial reproductions. Fresh second-pass observations are identified separately in this audit. Compile logs include corrected initial client syntax errors or sandbox cache failures; these are not library defects.

## Portable client replay

Use the reviewed revision and the matching toolchain. Build the corresponding package configuration first; `swift test` makes internal symbols available to the `@testable` clients. The helper links existing objects, writes only a temporary executable and does not run it.

```sh
python3 Docs/audits/2026-09-14/evidence/build_probe.py \
  Docs/audits/2026-09-14/evidence/root/SerializationProbe.swift \
  --configuration release
```

Run the executable path printed by the helper. The same process applies to the ordinary `Ownership.swift`, `ShapeRoutes.swift`, `NumericProbe.swift`, `DivisionProbe.swift`, `ExtraProbe.swift`, search/interface/batch clients and `ExecutionPublicProbe.swift`. `AlignmentDebug.swift` requires `--configuration debug`; consult `alignment-results.json` before selecting a dimension that intentionally traps.

The historical reviewer notes retain absolute temporary commands from the session. Their probe sources and bounded text output are copied here, so those temporary directories are not the only record. Huge full disassemblies, binaries, debug-symbol bundles and compiler caches are deliberately excluded; the relevant production disassembly excerpts are retained.

### Standalone benchmarking helper

`root/source-paths.txt` identifies the five source files concatenated into `root/main.swift`. The only transformation is removing now-redundant `import VectorCore` / `import VectorCoreBenchmarking`; the footer contains the printed fixtures.

```sh
swiftc -O -module-cache-path /private/tmp/vectorcore-audit-module-cache \
  Docs/audits/2026-09-14/evidence/root/main.swift \
  -o /private/tmp/vectorcore-audit-tooling-probe
/private/tmp/vectorcore-audit-tooling-probe
```

For the pool probe, compile the current `SwiftBufferPool.swift`, `BufferProvider.swift` and preserved `ExecutionPoolProbe.swift` together with `-O -swift-version 6 -parse-as-library`. Its source supplies a counted `AlignedMemory.deallocate` dependency while preserving real allocation/free. See [execution probe provenance](validate_execution_api/commands.md) for the exact original command and cleanup behavior.

### Independent binary fixture

The fresh serialization client uses bit patterns `0`, `0x80000000`, `0x7fc12345`, `0x7f800000`, `0xff800000`, `1`. Python independently generated the record with:

```python
body = struct.pack("<IHIH6I", 0x56454354, 1, 6, 0,
                   0, 0x80000000, 0x7fc12345, 0x7f800000, 0xff800000, 1)
expected = body + struct.pack("<I", zlib.crc32(body))
```

This exactly matched the emitted bytes in [SerializationProbe.log](https://github.com/gifton/VectorCore/blob/b2709f5b8c6a171a843813a968e5ca3cfa2e94d3/Docs/audits/2026-09-14/evidence/root/SerializationProbe.log). It validates the chosen framing/CRC/bit-pattern fixture, not every possible record.

## Provenance and audit checks

[manifest.json](manifest.json) records reviewed revisions, host/toolchain description, tracked-file content hashes and retained evidence hashes. [validate_audit.py](validate_audit.py) checks the register schema, unique IDs, all 46 canonical labels, source paths/line bounds, dependencies, local document links and retained-file hashes. It checks the audit artifact's consistency; it is not a substitute for library correctness tests.

The 46 `.log` files this evidence directory originally held were removed from the repository tree on 2026-10-23 (the repository ignores `*.log`). They are preserved unchanged at the same paths on branch `archive/pr56-build-logs` (commit `b2709f5b8c6a`), and their audit-time hashes are kept under `archived_artifacts` in manifest.json.

No source, package, CI or library test file was changed by this work. The new artifacts live under `Docs/audits/2026-09-14/`; the initial review stays under `Docs/reviews/2026-09-14/`.
