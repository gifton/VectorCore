# Compiler acceptance gate for the second remediation phase

The [first-phase exact-binary matrix](../2026-11-04-release-phase-one/compiler/README.md) remains the compiler evidence for this phase. The installed Xcode Release compiler failed three positive shapes, including the original lost-store canary. The exact retained Swift.org 6.4.2 compiler passed those positive probes. No new compiler binary or source-level accommodation has been qualified by this record.

The existing canary is unchanged. Its Release `compiler(>=6.4.1)` path requires the correct result; older compiler paths can record the same failure as a known issue. Consequently the earlier hosted aggregate passing on Xcode 26.3 does not imply that its numerical observation passed. Each release decision must inspect the observed result and exact binary identity as well as CI status.

The two upstream issues [#92562](https://github.com/swiftlang/swift/issues/92562) and [#93141](https://github.com/swiftlang/swift/issues/93141) were both closed when consulted again on November 4, 2026. Their status does not establish which changes are present in a particular installed compiler. The locally observed failing binary remains unqualified until an actual corrected binary passes the original numerical check and the relevant library/client qualification.

The available choices remain:

1. Preserve the intended Xcode/App Store support and keep affected Release combinations gated while qualifying a corrected Xcode compiler when available.
2. Explicitly adopt a narrower exact-toolchain scope and complete its remaining candidate checks. The currently passing Swift.org result is bounded to its tested configuration. Swift.org's [macOS installation guidance](https://www.swift.org/install/macos/package_installer/) requires the compiler bundled with Xcode for App Store submissions; standalone-toolchain qualification cannot establish that distribution path.

This phase preserves the current support policy. Portable test-support repairs, expanded sanitizer execution and benchmark observations can proceed independently of that decision. Their results must retain the compiler and platform actually used. Final hosted candidate acceptance and release publication remain separate gates.
