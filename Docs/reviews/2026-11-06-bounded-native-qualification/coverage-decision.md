# Next runtime coverage decision

**Recommendation:** Refresh the available arm64 iOS 27 simulator and current
arm64 Catalyst runtime evidence, and prepare a bounded app-hosted iOS consumer.
Use those concrete results to settle the remaining release coverage. This does
not adopt the proposed compiler restriction or admit 0.4.0.

## Concrete next deliverables

1. Run the unfiltered Release package suites for the unchanged migrated candidate
   on the available iOS simulator and current Catalyst host. Keep original canary
   failures visible and derive actual executed/skipped inventories afresh.
2. Prepare external consumers for each platform, with matching platform targets,
   SDKs, modules and Release producer objects. Exercise both `-Onone` and `-O`
   clients: by-value mutation, external conformer/witness, vector/storage paths,
   full omitted-write controls and residual reconstructed-inout diagnostics.
   Report Release-producer-only scope until a later Debug-producer pair is run.
3. Prepare a small app-hosted physical iOS consumer using the same numerical
   oracles and compiler diagnostics. Build/inspect the concrete harness before
   any signing/install/device step needing a user decision. A bounded consumer
   is a smaller deliverable than hosting the full suite; it must not be described
   as physical-device execution of the entire package suite.
4. Capture structured build results during the new platform work. Preserve the
   original seven console diagnostics; new observations must not relabel them.

The existing native consumer driver and evidence profile are macOS-specific.
Changing only `SDKROOT` cannot qualify simulator or Catalyst consumers: target,
module/object bindings, runner and host identity must agree. Extending the driver
requires focused behavioral tests and platform execution, not a broader version
allowlist.

## Coverage and resources

| Combination | What is available or established | Additional evidence/resources |
| --- | --- | --- |
| Current macOS arm64 | Pinned native producer/client evidence and C-mode routing runs | Chosen sanitizer refresh and controlled performance; final hosted candidate |
| Current iOS simulator | Fresh runtime inventory lists iOS 27.0, build 24A434 | Fresh migrated suites and target-aware consumers; select an available destination |
| Current Catalyst arm64 | Current host and fresh generic Release compilation | Fresh migrated suites/consumers; subprocess capability skips remain platform-specific |
| Physical iOS | Earlier tool-hosted package scheme rejected before test bodies | App-hosted consumer/test harness, usable signing arrangement and reachable device; current availability was not queried |
| macOS 14 and Catalyst floor | Deployment metadata and older coverage records only | Compatible actual macOS 14 hosts/runners; native arm64 and Intel execution |
| Intel | Existing CI cross-compilation requirement; fresh x86 simulator/Catalyst object slices | Actual Intel runtime for numerics, FP16 portability and x86 C dispatch; cross-compilation/Rosetta alone is insufficient for physical Intel dispatch |
| iOS 17 | Advertised deployment floor | Compatible runtime/host and representative physical-device evidence; current iOS 27 execution cannot establish the floor |
| tvOS 17/current | Fresh SDK 27.2 compilation | Compatible simulator runtimes, platform consumers and representative hardware if device execution is required |
| visionOS 1/current | Fresh SDK 27.2 compilation | Compatible simulator runtimes, platform consumers and representative hardware if device execution is required |

The current inventory found only an iOS simulator runtime. Other resource gaps
come from the prior reviewed coverage matrix, not fresh device or remote-host
discovery. Runtime provisioning, installations and support changes are separate
decisions. Compiling `VectorCoreBenchmarking` in the platform matrix does not
advertise that macOS host tool as a mobile product.

## User decision

| Direction | Consequence |
| --- | --- |
| **Local runtime and iOS harness first — recommended** | Complete the available evidence and concrete harness, then decide the remaining minimum-OS/Intel/tvOS/visionOS coverage bar. Release admission remains pending. |
| Require broad runtime coverage before release | Make minimum-OS, Intel, tvOS/visionOS and representative-device execution explicit release blockers and schedule the required resources. |
| Plan explicit deferred coverage | Prepare a precise gap statement and a later release-risk decision. This is not approval of those gaps, reduced platform support or a release. |

Existing support floors are not narrowed by any investigation option. The current
repository does not promise testing on every device; the broader bar is a proposed
release evidence decision, not a newly discovered mandatory CI requirement.

Existing required CI still includes Xcode 16.2/26.3 native Debug/Release tests,
eight Xcode 26.3 product/platform compilation cells, Intel test compilation,
Xcode 16.2 documentation consumers, C-mode contracts and sanitizers. Those jobs
must be requalified for the final candidate or changed through a separate explicit
decision. Local Xcode 27.2 evidence does not replace them.
