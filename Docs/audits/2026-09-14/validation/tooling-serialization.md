# Root validation: tooling and serialization

The root reviewer reread the actionable findings and their current source, rebuilt an exact-source benchmark helper, compiled a new Release-linked serialization client, exercised benchmark CLI selection/metadata/write failures, and independently generated a complete binary record with Python `struct` and `zlib`. Final decisions are in [the claim register](../CLAIMS.md).

## Original subsection crosswalk

| Original subsection | Final records / disposition |
|---|---|
| B1 correctness gating | TOOL-01 and TOOL-02. NaN, missing/extra output, missing validation and artifact failures are distinguished. |
| B2 build/kernel metadata | TOOL-03. Requested flags do not establish effective compilation/routing. |
| B3 extended performance assertions | TOOL-04. Source-confirmed workflow reachability; no hosted failure claimed. |
| B4 optimized dot selection | TOOL-05. Fresh Release CLI selects zero cases and exits zero. |
| B5 cosine A/B pairing | TOOL-06. Fresh two-case helper yields no pair; same key defect in console and JSON. |
| B6 observability/timing scope | TOOL-07. Source-confirmed differences; elimination and timing magnitude remain unmeasured. |
| B7 memory footprint | TOOL-08. Fresh handle sizes 16/8 bytes exclude 2048/1024-byte scalar payloads. |
| Trustworthy coverage for downstream scale | TOOL-10, TOOL-11 and API-41. Workload/route coverage gaps, not proof of a scale incident. |
| Matrix suite controls | TOOL-09. Hardcoded workloads, post-execution filtering and missing parsed dimensions. |
| CI/style enforcement | TOOL-11. Test-target/lint/analyzer/sanitizer inventories are source facts; remote branch rules remain uninspected. |
| Metadata/baselines | TOOL-12 and TOOL-03. Historical baselines, comparator provenance, CSV escaping and statistic labeling. |
| Mobile benchmarking product | TOOL-13. Unverified platform question; no Core compile failure alleged. |
| Tooling strengths/rejected suspicions | Shared stats use ordinary median, nearest-rank p90 and sample standard deviation coherently; p90 describes sample averages. Clock uses uptime in the shared harness. Main distance references use separate Double loops. The current sequential CLI caller structure does not demonstrate a BlackHole/Filters race. Existing Top-K benchmark evidence is historical; its timing was not repeated this pass. |
| Empty DynamicVector binary round trip | SER-01. Fresh actual-library reproduction; JSON control restores dimension zero. |
| Serialization coverage and golden fixtures | SER-02. Mechanical count of 15 comment-only tests; three optimized-named tests use generic vectors/extracted arrays. |
| Scalar binary conversion allocations | SER-03. Source opportunity; concrete overload/optimizer behavior must be measured before claiming heap allocation cost. |
| Type/precision representations and JSON schemas | SER-04. Deliberate distinctions need documentation, not an invented compressed-binary promise. |
| Binary protocol conformances | SER-05. Method availability does not declare conformance; no current dependent failure established. |
| Header magic/endian wording | SER-06. Complete bytes match independent Python framing and CRC; preserve wire bytes. |
| Dimension cap, flags, compression, whole records | SER-07. Deliberate 1...10000 decoder cap and one-frame format. Absent compression/streaming/migration is not an implementation bug. |
| Data indexing / unaligned input | SER-07. Fresh direct slice with startIndex 3 decodes and preserves all tested bit patterns. No slice bug found. |
| CRC and bounded decode allocation | SER-07. Source ordering and independent frame check support the limited positive assessment; no authentication claim. |
| INT8 / parameter / TopK decode boundaries | SER-08 cross-references OWN-02, numerical quantization records and API-18. Avoid double-counting. |
| VectorData and Base64 | Source inspection supports the original limited assessment: synthesized Codable for VectorData, Base64 delegates to the checked binary path. No additional defect established. |

## Fresh observations

The benchmark helper is assembled from the unchanged current Models, CaseParsing, JSONWriter, ABComparisons and Correctness files, removing only imports made redundant by concatenation. It reports:

```text
nanScalar       maxAbs=NaN  maxRel=NaN  violates=false
nanVector       maxAbs=0    maxRel=0    violates=false
emptyVsNonempty samples=0   maxRel=0    violates=false
longerOutput    samples=1   maxRel=0    violates=false
cosineABPairCount=0
nanJSON encoding rejected
```

The Release serialization client reports an empty 16-byte frame rejected by binary decoding and an empty JSON round trip accepted. A six-element payload covering positive/negative zero, a NaN payload, both infinities and the smallest subnormal matches independent Python little-endian framing and CRC byte for byte. A direct nonzero-index Data slice preserves those bit patterns. Dimension 10001 encoding is rejected on decode, matching the existing intentional cap test.

The fresh Release CLI's optimized-only dot filter yields `results: []` and exit 0. The existing same-revision Debug CLI reports Release and accepts C/size flags into metadata. A requested JSON destination that is an existing directory prints a write error and exits 0. These are short control-path observations, not useful performance timings.

Fresh CI checks: 25 Python tests passed; the Ruby validator accepted nine GitHub YAML/template files. The Release baseline's terminal summaries report 1,144 Swift Testing tests and 91 XCTest cases, with 27 XCTest skips and no failures. Earlier Debug, lint and documentation-consumer results are retained as first-pass evidence only.

## Limitations

No malformed binary exploit, allocator-failure injection, concrete binary-load allocation trace, mobile benchmarking-product build, current hosted CI/branch-settings inspection or long performance run was performed. Positive controls support only their stated properties. The [evidence index](../evidence/README.md) distinguishes actual-library probes, copied-source fixtures and source reasoning.
