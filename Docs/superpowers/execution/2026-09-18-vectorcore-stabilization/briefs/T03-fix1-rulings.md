# T03 fix round 1 — root rulings (binding)

1. **C1 (Critical) — portable sentinels.**
   - Remove `ProgressReporter` from the Foundation sentinel set. `AttributedString` already proves that FoundationEssentials re-exports are covered.
   - Update `test_missing_sentinel_exits_two` to drop a sentinel that is stable across SDKs.
   - **Verify for real on an older SDK:** run the checker once with `DEVELOPER_DIR=/Library/Developer/CommandLineTools` (Swift 6.2, SDK MacOSX26.0, the local proxy for CI's 26.3). It must exit 0, or 1 only for a genuine collision. Also run it once on Xcode 27. Record both outputs and timings.
   - Root has confirmed that the 26.0 SDK has no ProgressReporter.
2. **I1 — Scope and placement.**
   - Move the checker step out of `platforms` into a new **non-required** job, `sdk-names` (macos-15, Xcode 26.3, `timeout-minutes: 30`). Do NOT add it to the `required` aggregate: the seven-job required set and check_required stay unchanged.
   - Update the validate_github.rb RULES (not the pin) so the job must exist and run the command.
   - If the validator's policy forbids a non-required job, fall back to keeping the step in `platforms`, raise that job's timeout to 45 min, and record the justification.
   - In the step name or a comment, document that it guards the CI SDK only, and that Xcode-27 collisions need an Xcode-27 leg, which belongs to the Xcode-27 CI migration.
   - Report the measured local run time.
3. **I2 — Prove the real build is untouched under swiftbuild.**
   - In `check_benchmark_build_modes.py`, hash the `source_objects` set and the dispatcher side files (.d, .dia, index unit) before and after the replacement compile and link, and assert they are equal.
   - Assert that no remaining path-valued argument of the rewritten command resolves under the real build's Objects-normal directory.
   - Document the module-cache writes as cache-only.
   - Keep every V04-imported function signature.
   - Report the new sha256 of the driver: root re-pins.
4. **I3 — Strengthen the path policy, documented as a heuristic.** In `test_swift_test_output_paths.py`:
   - join multi-line statements before matching;
   - taint the names of functions and computed properties whose body derives a URL or path from `#filePath`/`#file`/`#fileID`;
   - extend the write operations to FileHandle(forWritingAtPath:/forWritingTo:), OutputStream(toFileAtPath:/url:), open with O_WRONLY/O_RDWR/O_CREAT, fopen with a write mode, createSymbolicLink, and write(toFile:/to:);
   - flag relative-path writes, i.e. createDirectory/createFile/write with a non-absolute string literal path, and any write through `currentDirectoryPath`.
   - Add a negative control for EACH new pattern, and keep the existing ones.
   - The docstring must state the heuristic scope and the per-file taint limitation.
   - Re-scan the tree. Fix any true positive in test code; any production hit goes to root.
5. **I4 — Close the native-layout class.**
   - Migrate O03RunAllocationProbe.py, Q04RunAllocationProbe.py, Q02RunCalibrationProbe.py, E03TypecheckDriver.swift and B01_public_driver.py to `swiftpm_products.py`. For the Swift driver, use the equivalent bin-path/products input supplied by its Python caller. Each must work under both layouts. Run whatever each one drives, under swiftbuild 6.4 and, where possible, native CLT 6.2.
   - Add `Scripts/ci/tests/test_no_native_layout_literals.py`: fail on native-layout literals (`.build/debug`, `.build/release`, `VectorCoreC.build/module.modulemap`, `*.build/*.o` globs) outside `swiftpm_products.py`. It needs an explicit justified allowlist and negative controls.
6. **Minors.**
   - M1: add TimeoutExpired/OSError tests for the checker's run().
   - M4: add an alias test for the old driver flags.
   - M2: assert the source_objects set is non-empty and includes the dispatcher object. Full completeness against the merged object is optional.
   - M3: accepted as temporary storage, no change.
7. **xunit:** no change in T03. Record the invariant ("children are awaited inside a parent test") as a documented convention in one shared place, the most central child helper or a short Docs note, citing T03's evidence and stating that native-CI behaviour is unverified. A future class fix (a shared argument filter that strips --xunit-output) stays deferred.
8. **approxEqual:** add a doc comment on `approxEqual` (TestSupport.swift:8). It must state that the default is an absolute tolerance of 1e-5, appropriate only for |values| ≲ 100 or exact results, and that callers comparing larger magnitudes must pass an explicit relative or ULP-derived `tol:`. No call-site changes.

Verification:
- policy discovery, where only the two expected pin failures are allowed;
- gate 5 under both toolchains, with the new I2 assertions passing;
- the checker run on SDK 26.0 and 27, with timings;
- the I4-migrated drivers run under both layouts;
- lint on changed Swift files.

Return the report text in your final message.
