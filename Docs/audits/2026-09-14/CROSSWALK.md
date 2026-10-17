# Original finding crosswalk

[Improvement guide](README.md) · [Complete claim register](CLAIMS.md)

Every canonical label in the initial summary and benchmarking report is mapped below. A single original label can cover several distinct diagnoses; repeated labels also refer to shared causes. The independent validation notes contain the original component-subsection crosswalks, including strengths, documented limits and rejected inferences.

| Original | Validated records | Disposition |
|---|---|---|
| A1 | [OWN-01](CLAIMS.md#own-01), [OWN-02](CLAIMS.md#own-02) | confirmed-reproduced |
| A2 | [API-33](CLAIMS.md#api-33), [OWN-03](CLAIMS.md#own-03) | confirmed-reproduced, confirmed-source |
| A3 | [OWN-04](CLAIMS.md#own-04) | confirmed-reproduced |
| A4 | [API-01](CLAIMS.md#api-01) | confirmed-reproduced |
| A5 | [API-02](CLAIMS.md#api-02), [API-03](CLAIMS.md#api-03) | confirmed-reproduced |
| A6 | [API-04](CLAIMS.md#api-04) | confirmed-reproduced |
| A7 | [API-05](CLAIMS.md#api-05), [API-06](CLAIMS.md#api-06) | confirmed-reproduced |
| A8 | [API-09](CLAIMS.md#api-09), [API-10](CLAIMS.md#api-10) | confirmed-reproduced |
| A9 | [API-11](CLAIMS.md#api-11) | confirmed-reproduced |
| A10 | [API-12](CLAIMS.md#api-12), [API-36](CLAIMS.md#api-36), [OWN-27](CLAIMS.md#own-27) | confirmed-source |
| C1 | [OWN-05](CLAIMS.md#own-05) | confirmed-reproduced |
| C2 | [API-13](CLAIMS.md#api-13) | confirmed-source |
| C3 | [NUM-01](CLAIMS.md#num-01), [NUM-02](CLAIMS.md#num-02) | confirmed-reproduced |
| C4 | [OWN-08](CLAIMS.md#own-08) | confirmed-reproduced |
| C5 | [OWN-06](CLAIMS.md#own-06) | confirmed-reproduced |
| C6 | [NUM-03](CLAIMS.md#num-03), [NUM-04](CLAIMS.md#num-04) | confirmed-reproduced, confirmed-source |
| C7 | [NUM-05](CLAIMS.md#num-05) | confirmed-reproduced |
| C8 | [NUM-06](CLAIMS.md#num-06) | confirmed-reproduced |
| C9 | [NUM-07](CLAIMS.md#num-07) | confirmed-reproduced |
| C10 | [API-14](CLAIMS.md#api-14) | confirmed-reproduced |
| C11 | [API-15](CLAIMS.md#api-15), [API-16](CLAIMS.md#api-16) | confirmed-reproduced |
| C12 | [API-17](CLAIMS.md#api-17), [API-18](CLAIMS.md#api-18) | confirmed-reproduced |
| C13 | [NUM-08](CLAIMS.md#num-08), [NUM-09](CLAIMS.md#num-09) | confirmed-reproduced |
| C14 | [NUM-11](CLAIMS.md#num-11), [NUM-12](CLAIMS.md#num-12), [NUM-13](CLAIMS.md#num-13) | confirmed-reproduced |
| C15 | [API-07](CLAIMS.md#api-07), [API-08](CLAIMS.md#api-08), [API-19](CLAIMS.md#api-19), [API-20](CLAIMS.md#api-20), [API-47](CLAIMS.md#api-47), [NUM-26](CLAIMS.md#num-26) | confirmed-reproduced, qualified |
| C16 | [NUM-14](CLAIMS.md#num-14), [NUM-15](CLAIMS.md#num-15) | confirmed-reproduced, qualified |
| C17 | [OWN-09](CLAIMS.md#own-09), [OWN-11](CLAIMS.md#own-11) | confirmed-reproduced, qualified |
| C18 | [API-21](CLAIMS.md#api-21) | confirmed-source |
| C19 | [NUM-16](CLAIMS.md#num-16), [NUM-17](CLAIMS.md#num-17) | confirmed-reproduced |
| C20 | [SER-01](CLAIMS.md#ser-01) | confirmed-reproduced |
| PERF1 | [OWN-12](CLAIMS.md#own-12), [OWN-13](CLAIMS.md#own-13) | confirmed-reproduced, confirmed-source |
| PERF2 | [API-22](CLAIMS.md#api-22) | qualified |
| PERF3 | [OWN-07](CLAIMS.md#own-07) | confirmed-reproduced |
| PERF4 | [NUM-22](CLAIMS.md#num-22) | confirmed-source |
| PERF5 | [API-23](CLAIMS.md#api-23), [API-45](CLAIMS.md#api-45) | confirmed-source |
| PERF6 | [API-24](CLAIMS.md#api-24) | confirmed-source |
| PERF7 | [API-25](CLAIMS.md#api-25) | confirmed-source |
| PERF8 | [NUM-29](CLAIMS.md#num-29), [OWN-19](CLAIMS.md#own-19) | confirmed-source |
| PERF9 | [TOOL-01](CLAIMS.md#tool-01), [TOOL-02](CLAIMS.md#tool-02), [TOOL-03](CLAIMS.md#tool-03), [TOOL-04](CLAIMS.md#tool-04), [TOOL-05](CLAIMS.md#tool-05), [TOOL-06](CLAIMS.md#tool-06), [TOOL-07](CLAIMS.md#tool-07), [TOOL-08](CLAIMS.md#tool-08) | confirmed-reproduced, confirmed-source, qualified |
| B1 | [TOOL-01](CLAIMS.md#tool-01), [TOOL-02](CLAIMS.md#tool-02) | confirmed-reproduced |
| B2 | [TOOL-03](CLAIMS.md#tool-03) | confirmed-reproduced |
| B3 | [TOOL-04](CLAIMS.md#tool-04) | confirmed-source |
| B4 | [TOOL-05](CLAIMS.md#tool-05) | confirmed-reproduced |
| B5 | [TOOL-06](CLAIMS.md#tool-06) | confirmed-reproduced |
| B6 | [TOOL-07](CLAIMS.md#tool-07) | qualified |
| B7 | [TOOL-08](CLAIMS.md#tool-08) | confirmed-reproduced |

## Component coverage

| Initial component | Second-pass owner / detailed crosswalk |
|---|---|
| Vectors | [Ownership](validation/ownership.md); [numerics](validation/numerics.md); API-19 factory boundary |
| Memory/storage | [Ownership](validation/ownership.md) |
| Execution/concurrency | [Execution/API](validation/execution-api.md) |
| Scalar/SIMD/C kernels | [Numerics](validation/numerics.md); OWN-08 alignment; API-43–46 helpers/tooling claims |
| Search/Top-K | [Execution/API](validation/execution-api.md) |
| Batch/matrix | [Execution/API](validation/execution-api.md); [numerics](validation/numerics.md) statistics/entropy |
| Reduced precision | [Numerics](validation/numerics.md); OWN-02/04/17–19; API-21 diagnostics race |
| Serialization | SER-01–08; TOOL-01/02 nonfinite artifacts; [root crosswalk](validation/tooling-serialization.md) |
| Linear algebra/PCA | [Numerics](validation/numerics.md); API-20 configuration arithmetic |
| UMAP | [Numerics](validation/numerics.md) |
| Downstream interfaces | [Execution/API](validation/execution-api.md); OWN-03 hint mutation |
| Validation/benchmarks/CI | TOOL-01–13; [root crosswalk](validation/tooling-serialization.md) |
