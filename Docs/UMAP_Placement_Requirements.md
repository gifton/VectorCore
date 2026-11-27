# Requirements carried forward to UMAP placement

These are accepted requirements for the future item 3.5 contract. Placement is
not implemented by the progress/reference-check cleanup. Existing fit arithmetic
and graph distances remain unchanged.

- Decide exact matches from the original validated vectors, not computed Gram
  distances. Equally sized finite Float rows match when every element compares
  exactly equal numerically: `+0` and `-0` match. Use no epsilon and no normalization
  for this decision. A positive scalar multiple under cosine is an ordinary neighbor
  unless the original elements already compare equal. Nonfinite input is rejected
  by input validation before the match rule.
- Choose the lowest reference index among exact matches and copy its coordinates
  without changing their Float words, including coordinate signed zeros.
- Reuse `KNNGraph.Metric` for the two-sided builder and placement configuration.
  Do not introduce a second Euclidean/cosine selector. Supplied graphs remain
  authoritative under the forthcoming placement contract.
- Carry cooperative cancellation into placement: on entry before validation,
  once per query block even without a handler, and before/after synchronous
  callbacks including completion. Observed cancellation throws `CancellationError`.
  Packing/vendor work has no wall-clock interruption promise. The same lifecycle
  applies to the supplied-graph route.
- Keep `UMAPProgress.Stage` closed through 0.4.x, including pre-release development.
  Placement must reuse existing stage meanings where appropriate or use a separate
  progress type. Its stage names, units and empty-query behavior belong in its own
  contract before implementation.

The placement contract must still resolve how exact matches interact with supplied
graphs and ensure rounded neighbor ranking cannot hide a true original-vector
match. Its input-validation rules, including metric-specific zero-vector handling,
must precede the matching rule. No upstream handoff document is replaced by this note.
