# R04 frozen v1 independent source review

**Source verdict: ACCEPT. Final runtime/evidence acceptance remains pending.** No blocking specification or source-test issue found. No builds, test reruns, source changes or delegation performed.

## Freeze and scope

Read approved brief SHA256 `e44c01117ccff56a465cc35b02c54bc3e804d6eac813f2d36a01b4d216d52bfb`, complete patch SHA256 `a5dffcd7c82cad065528e1f32e66196cd8cf518cc2336a36fa5f5fa3f15e0049`, manifest, full frozen MinHeap.swift and all 196 lines of HeapMembershipContractTests.swift. Independently verified all 320 base manifest hashes against exact Git base `e0e0e74b5128419482d1e1993d58c2184adc01ba` and all 321 final canonical source hashes. Exactly the two authorized paths change. Root's broader unowned-archive reconstruction is separate evidence, not repeated here.

## Membership, initialization and ownership

The new initializer delegates through the negative-capacity precondition before assigning input storage, heapifying or invoking the comparator. Empty construction no longer reserves capacity slots. Zero returns empty; nil/full-retention uses the input array and guarded bottom-up heapify; positive truncation streams every input through the existing replacement rule. Empty/singleton inputs avoid invalid heapify indices. Int.max is merely a cap and no longer an immediate allocation request.

Under a strict weak ordering, root is a least retained equivalence class. A full heap replaces it exactly when compare(root,new) is true, so it retains the last capacity elements in comparator order. Equivalent or earlier new elements reject. The repair correctly preserves this rule rather than inverting it; removing arbitrary heap leaves is gone. Full-retention heapify retains all members, while truncated construction literally shares the streaming algorithm. Stable identity/order among equivalent keys is not promised.

sorted() drains a value copy in comparator order; iteration remains the Array's heap layout. Copy mutation and ordinary element lifetimes remain Array semantics. Existing child-index arithmetic is driven by actual materialized count, not the potentially Int.max cap; no new cap-based count arithmetic or unsafe-pointer work was introduced. No universal allocation-success or speed claim is made.

## Legacy nearest policy and live routes

KNearest rejects negative k before initializing its heap and reverses TopKSelection.orderedAscending with smallerIndex. Thus worst-at-root includes NaN after every numeric distance, infinities in numeric order, and larger indices after smaller indices for distance/NaN ties. Draining then reversing gives the specified best-first result without another NaN comparator implementation. Duplicate identical entries remain eligible under capacity and are not deduplicated; equal full-root entries reject through the generic rule. Zero and huge k inherit lazy storage.

Read the shared orderedBefore/orderedAscending implementation and current TopKBuffer route sites in Operations/TopKSelection. Production searches find only KNearestHeap's declaration and a historical explanatory comment, no executable legacy caller. ExecutionOperations.findNearestSequential remains absent. Direct public KNearest tests are therefore the proper obligation; no removed route is restored, and modern TopK controls remain separate.

## Test quality and class coverage

Generic matrices enumerate empty/singleton, distinct four-element and duplicate-key inputs under every permutation and both numeric comparator orientations. Capacities include nil, zero, one, n-1, n, n+1 and Int.max with duplicates removed. The reference independently sorts then takes the suffix; bulk, streaming, iterator multiset, count bounds, removal ordering and copy isolation are checked. Record fixtures distinguish comparator-equivalent keys from a strict key/label secondary order, without inventing stable equivalent-label behavior. Explicit insertion assertions cover underfull, equivalent rejection preserving the old label, worse rejection, better replacement, zero and unbounded duplicates. Weak-reference controls verify retained element survival across replacement of an aliased copy and eventual release.

The nearest oracle is independently implemented in the test: explicit NaN classification, numeric comparisons, then index. It never calls production TopK ordering to form expected values. Every permutation of the bounded fixtures covers NaN positions, numeric/infinity ordering, signed zeros with distinct indices, repeated distances, all-NaN fillers, duplicate identical pairs, empty/singleton and k0/1/n-1/n/>n/Int.max. Numeric results compare bits and NaNs compare classification. A separate explicit NaN→numeric k1 test checks insertion results; the modern public TopKSelection.select control compares IDs and distance words/classification to the independent oracle. Production comments link named tests or follow immediately from code; no generic NaN law is asserted for an invalid caller comparator.

## Bounded negative-domain helper

The helper isolates negative empty/bulk/KNearest construction behind a scenario environment marker and child-only filter. Captures go to files, with a termination handler installed before run. The 15-second execution deadline is followed by at most two one-second cleanup waits; timeout throws before terminationStatus/reason access. Normal completion requires an actual scenario-entry marker, signal termination and absence of accepted marker. Debug additionally requires the domain diagnostic. Release acceptance must combine actual signal records with the unchanged static precondition source, as the brief allows; it must not pretend an optimized-away diagnostic appeared.

The parent childProbe entry is a no-op without its marker. The three scenario arguments are actual separate child obligations, not three additional primary test functions. The helper does not prove successful reaping after every pathological cleanup timeout; it accurately reports failure instead. No unbounded wait/pipe read or status read after timeout occurs.

## Final evidence still required

Await sealed baseline RED, frozen Debug/Release command/log/source/host-versus-loaded-bundle and three-child associations, with setup failures and timeout causes distinct. Confirm the actual final filter resolves HeapMembershipContractTests, TopKNaN contract/wrapper suites and IntegrationProtocolsSuite. The reported preliminary 72-function checkpoint and scanner result are not final evidence in this review. No targeted ASan requirement arises from this value-only change. API-17/API-26 final closure remains with root after runtime adjudication; no downstream or performance acceptance is inferred.
