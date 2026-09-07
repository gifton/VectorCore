# Security policy

## Report a vulnerability privately

Use [GitHub private vulnerability reporting](https://github.com/gifton/VectorCore/security/advisories/new)
for suspected security defects in VectorCore. Do not include exploit details,
sensitive data, or credentials in public issues or pull requests. Ordinary bugs
without security implications can use the public issue templates.

Include the affected version or commit, platform and toolchain, a minimal
reproduction, expected and actual behavior, potential impact, and any proposed
mitigation. For memory-safety or concurrency problems, include sanitized crash
logs or sanitizer output and the relevant buffer ownership/lifetime details.
Use synthetic inputs in place of private application data.

The maintainer will assess reports, discuss reproduction and mitigation through
the private advisory, and coordinate a fix and disclosure when practical.
Handling is best-effort; there is no guaranteed acknowledgment, fix, or
publication deadline. Please coordinate public disclosure in the advisory.

## Supported versions

Security maintenance focuses on the latest published release and `main`.
Older releases have no guaranteed security backports; users may need to upgrade
to receive a fix. Pre-1.0 compatibility and support limits are described in
[Contributing](CONTRIBUTING.md#compatibility-and-support).

VectorCore exposes unsafe buffer APIs whose callers must satisfy the documented
bounds, alignment, lifetime, ownership, and synchronization requirements. Report
suspected violations of the library's own contracts, including unsafe handling
of otherwise valid inputs. A compiler check, test pass, or scanner result is
not a security guarantee.

Fork maintainers must configure their own private reporting channel and update
this policy before inviting reports; see [Maintaining](Docs/Maintaining.md).
