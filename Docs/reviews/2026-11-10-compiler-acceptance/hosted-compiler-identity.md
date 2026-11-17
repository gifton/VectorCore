# Hosted compiler identity — adopted 2026-11-16

The user adopted the exact hosted Xcode 27.2 / 27B52 frontend/driver pair for
fresh bounded qualification after a byte-level comparison. The
[policy amendment](decision.md#additional-exact-hosted-compiler-identity) records
both full fingerprint pairs and retains the numerical checks, support boundaries
and two-stage profile review. No hosted numerical result or release admission
is supplied by this identity decision.

The first collection, [run 43631911981](https://github.com/gifton/VectorCore/actions/runs/43631911981),
stopped at the frontend fingerprint check before compilation/tests and retained
no qualification artifact. [Inventory run 43992711618](https://github.com/gifton/VectorCore/actions/runs/43992711618)
recorded matching compiler build identifiers, UUIDs and signed-code hashes,
valid Apple signatures, and differing whole-file/signature metadata.

At the user's request, [byte-comparison run 43996681385](https://github.com/gifton/VectorCore/actions/runs/43996681385)
then measured all non-signature file regions and individual signature components.
The full-file comparison accounts for 674 differing frontend bytes and 672 driver
bytes through the omitted 30-byte CMS signingTime attribute, changed RSA signature
value and associated length fields. Bytes outside the signature allocation,
CodeDirectory and requirements match. In-memory reconstruction using only those
signing-record changes reproduces both hosted full-file SHA-256 hashes. The
installed compiler binaries were not changed. This does not attribute the signing
process or compare every SDK/toolchain component.

Canonical retained evidence:

- [Initial metadata inventory](/Users/goftin/dev/projects/VectorCore-archive/VectorCore-compiler-acceptance-2026-11-10-evidence/hosted-toolchain-inventory-43992711618/README.md)
- [Byte comparison and exact reconstruction](/Users/goftin/dev/projects/VectorCore-archive/VectorCore-compiler-acceptance-2026-11-10-evidence/hosted-compiler-byte-comparison-43996681385/compiler-byte-comparison-report.md)
- [Adoption receipt](/Users/goftin/dev/projects/VectorCore-archive/VectorCore-compiler-acceptance-2026-11-10-evidence/hosted-compiler-byte-comparison-43996681385/hosted-compiler-identity-adoption.json)

Both inventory jobs completed their metadata steps and uploads, then deliberately
failed their final non-qualification step. They executed no VectorCore numerical
tests. Their raw records and proposed pre-adoption documents remain unchanged in
the archives; the later adoption receipt supplies the actual policy decision.

Fresh captures and reviewed profiles use schema 2 to bind both the frontend and
Swift driver and require an exact adopted pair. Legacy schema-1 local profiles
retain their original frontend-bound assessment; they are not relabeled as
paired captures, and hosted verification requires schema 2. Unlisted compiler
fingerprints and mixed pairs reject. The numerical oracle, diagnostic failure
handling and releaseAdmitted=false behavior remain as adopted.

The implementation uses a new tooling revision while preserving all tracked
Package.swift, Sources, Tests and Benchmarks bytes from the frozen
`af315c56c52de737f868fd969e6e1f5ae59eaa48` candidate. Fresh hosted collection
and [exact profile review](hosted-evidence-2026-11-16.md) are complete; the separate
hosted verification step remains pending.

The clean candidate contains 389 tracked compiled-input files; its serialized
manifest SHA-256 is `2191f5d4663a071e9a537be4a525bb353798370d7c7db44980fcd15b0b1f521e`.
Compared with the prior local 392-file manifest, only three untracked
`.DS_Store` metadata files are absent. Every shared input digest matches. The
earlier local profiles and their original source manifest remain unchanged.

Implementation validation: 754 Python policy tests passed in 270.650 seconds;
122 focused qualification tests and all ten GitHub policy checks passed.
Independent review found one entry-point fingerprint gap in schema-2
verification. Four repinned native/iOS substitutions reproduced it; the verifier
now requires both adjacent `swift` and `swiftc` files to match the frontend.
Those regressions pass after the fix. No other review findings remain.

The final verifier rechecked the unchanged native, simulator and physical-device
archives: 2,520, 60,195 and 3,607 verified files respectively. All retain their
original scoped acceptance, `compilerProfile=known-failing` and
`releaseAdmitted=false`. Preservation checks confirm the dirty-main baseline,
25 protected historical files, four historical profiles and the alpha tag are
unchanged. Detailed records are retained in the canonical
[integration evidence archive](/Users/goftin/dev/projects/VectorCore-archive/VectorCore-compiler-acceptance-2026-11-10-evidence/hosted-identity-integration-2026-11-16/README.md).

Fresh collection [run 44015059588](https://github.com/gifton/VectorCore/actions/runs/44015059588)
used candidate and policy `7c115cd927c76275b73a08b0e6fb178636a3c4b9`.
While preparing review, a pre-existing hosted bundle omission became apparent:
external iOS `clang` and `libtool` paths were listed, but their hashes were not
recorded in command receipts or retained file data. Local profile review had
access to the original tools; hosted review needs those digests in the archive.
The owned attempt was stopped before treating it as qualification evidence.

The corrected policy bundler records a post-collection `artifact-inventory.json`
with every declared artifact's digest, size and resolved path, plus the capture
digest. It includes external tools and source files without uploading those
external binaries. Missing files leave an explicitly incomplete inventory while
raw evidence is still retained. The inventory supplies observed hashes for
separate profile review; it does not generate or approve numerical expectations.
The corrected policy collects the same immutable `7c115cd` candidate. Review
must require a complete inventory and reconcile its digests with the capture,
retained data and independently reviewed profile pins before verification.

Retention correction validation: both new transport regressions failed before
the fix; all 124 focused qualification tests and 756 Python policy tests then
passed, along with ten GitHub policies and whitespace checks. The correction
changes policy bundling and its tests/documentation only. The original
`7c115cd` collector candidate, paired-identity verifier, numerical oracles and
historical profiles are unchanged.
