# R05 baseline and migration inventory

All 6,063 files in the prepared archive match immutable base.tar. The binding brief is da4ea6c37730403e5178177cac5f6620cde30eee34101b63673f7c869ab45994. The only existing executable computedScore property callers are two expectations in IntegrationProtocolsTests.testSearchResult_ComputedScore; their explicit Euclidean migration is already approved. No other existing caller migration is necessary.

Both collection producers call the same private searchCapturedVectors helper. Simple captures under its lock and calls the helper after unlock; the default composes a snapshot without claiming atomicity. Scoring selected results in that helper preserves both routes and avoids scoring unselected candidates. Raw distance implementations, ranking, TopK conversion and collection transaction code are read-only except the approved score-construction line/docs.

The initial behavioral test invokes real searchDotProduct with query [1] and candidates [.5], [1], [2]. It asserts existing IDs/distances and expected finite logistic relevance through the old property, so a behavioral failure rather than a missing API compile failure is required. Final filters resolve SearchResultScoreContractTests, IntegrationProtocolsSuite, MetricDispatchContractTests and VectorCollectionTransactionTests.
