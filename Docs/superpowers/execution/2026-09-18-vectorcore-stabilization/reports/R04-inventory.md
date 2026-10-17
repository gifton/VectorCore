# R04 scope and route inventory

Dispatch base: e0e0e74b5128419482d1e1993d58c2184adc01ba. Base archive SHA256 e860e19254741ceb01e1f3a27e7079b6a813defd0628046edc9087b9c96f5fe5. All 5,891 regular files matched before edits. Frozen v1 has 320 base/321 final code entries; only MinHeap.swift and the new HeapMembershipContractTests.swift differ.

The retained callers.txt search covers Sources and Tests. Production KNearestHeap appears only at its declaration/implementation in MinHeap.swift and a historical comment in TopKSelection.swift. No executable production caller remains. ExecutionOperations.findNearestSequential was deleted by accepted E02 and is absent. Current Operations and TopKSelection construct TopKBuffer in Operations.swift, TopKSelection.swift and Operations/Kernels/TopKSelectionKernels.swift. These routes remain read-only. The legacy public API is tested directly, without reintroducing the deleted route.

## Contracts and independent matrix

The generic comparator is caller-supplied strict weak ordering. Bounded heaps retain the suffix of the input sorted by that comparator. Numeric < retains largest, > retains smallest; no generic NaN rule is imposed. New comparator-equivalent arrivals at a full heap reject, and equal-key removal identity/order remains unspecified.

The numeric matrix enumerates every positional permutation of empty, singleton, four-distinct, and duplicate-value fixtures, both comparator orientations, and distinct capacities nil/0/1/n-1/n/n+1/Int.max. Independent sorted-suffix arrays establish retained multisets and drain order. It checks bulk, streaming, iteration-as-layout-multiset, count bounds, removal on a copy and preservation of the original. Int.max controls use at most four input elements and run only after removal of eager reservation.

Record tests enumerate all24 permutations of four records with duplicate keys, compare keys under both orientations, and separately supply explicit secondary labels to establish deterministic identity. Equivalent newcomers, worse rejection, improved replacement, underfull insertion, zero capacity and unbounded duplicate insertion verify return Bool and count semantics. Reference-valued elements survive copy mutation and release after both heaps leave scope.

The independent nearest oracle explicitly branches on NaN classification, then numeric distance and index. Fixtures include empty/singleton, a six-element mixed NaN/infinity/signed-zero/finite set, shuffled duplicate finite distances plus two NaNs, all-NaN and duplicate-identical records. Every positional permutation is tested at distinct k0/1/n-1/n/n+1/Int.max. Expected membership is the independent sorted prefix. NaNs compare by classification; non-NaNs preserve bits, and zero ties resolve by index. Modern public TopKSelection controls use positional labels and the same independent oracle, without changing modern routing.

Three actual negative-domain children cover empty initializer, bulk empty-array initializer, and KNearestHeap. The baseline bulk case times out at15seconds; final acceptance requires intentional signal rejection, the started marker and no accepted marker. Debug additionally requires the named capacity diagnostic. Release may strip standard precondition text; explicit source guards and actual signals establish that boundary. The parent throws on any timeout after <=2seconds cleanup before reading status/reason. The ordinary parent childProbe function is a no-op and is reported separately from actual children.

## Setup history

An initial combined shell-wrapped approval request was interrupted before execution; no files/process existed. The first executed baseline build failed test compilation (overloaded comparator ternary, mutating call inside #expect macro). It is retained as red/build-red-debug, not behavioral RED. The corrected red2 shell-wrapped request also stalled before approval/execution. Red2 source is retained; there is no invented command exit or runtime result.

The executed red3 uses direct swift test with approved prefix and a canonical checkout-local capture fallback, preserving all returned output in chunk files. Both orientations and NaN eviction fail; negative empty/nearest return normally, and negative bulk hangs within the bounded child. No baseline Int.max eager-reservation test was run. Complete chunk outputs were persisted without truncation; combined logs concatenate those chunks with their preserved line separators.
