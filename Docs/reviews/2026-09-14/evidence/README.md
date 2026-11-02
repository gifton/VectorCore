# Reproduction evidence

These eight small clients were compiled against the reviewed package's complete
Release objects and executed by the orchestrator. Observed results are preserved
in [release-results.json](release-results.json). The sources record concrete inputs
for converting findings into regression tests during remediation.

**These are observation programs, not passing correctness tests.** A zero exit
means the program completed; the reported outputs include the defects described
in the review. Some sources also contain explicit boundary modes used during
the original review. The preserved JSON records the normal/default modes only.

## Reproduce on the reviewed revision

Requires the same Apple/Swift environment and a completed `swift test -c release`
build. From the repository root:

```sh
python3 Docs/reviews/2026-09-14/evidence/build_probe.py \
  Docs/reviews/2026-09-14/evidence/ownership_format_probe.swift
/private/tmp/vectorcore-review-replay/ownership_format_probe
```

The helper links existing objects without modifying the package or rebuilding it.
It writes the executable under `/private/tmp/vectorcore-review-replay`. Use another
source from this directory for vector, search, batch, quantization, factorization
or interface observations. Input files must be trusted Swift source.

The review commit is `8b02e75bca7055ae11170873d15c06f1538709b7`. Results after repairs,
on another compiler or on another platform may differ. Hardware-sensitive GEMM
observations are examples of the documented cancellation problem, not universal
bit-pattern predictions. None of these programs is a throughput benchmark.

Other exact-source allocation/pool instrumentation, tool probes, crash logs,
Debug/Release suite logs and lint output remain in the temporary directory linked
from the main report. They are not included as portable production tests here.
