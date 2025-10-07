# VectorCore Testing Notes

## Test Framework Limitations

### Swift Testing vs XCTest Filtering

**Important:** The VectorCore test suite uses **Swift Testing framework** (`@Suite` annotations), which has different behavior than XCTest:

- ✅ **XCTest**: `swift test --filter` works correctly
- ❌ **Swift Testing**: `--filter` flag is **IGNORED** (tests run anyway or show 0 tests)

### Impact on Batch Testing

The `run_test_batches.sh` script was designed to run tests in filtered batches, but **filtering doesn't work** with Swift Testing tests.

**Current Behavior:**
- Filters are applied to test names
- XCTest tests (MinimalTests.*) may match filters
- Swift Testing tests (ComprehensiveTests.*) ignore filters
- Result: Inconsistent test execution

### Workaround

**Option 1: Run All Tests** (Current)
```bash
# Remove --filter entirely
swift test
```

**Option 2: Use Swift Testing Selection** (Future)
```bash
# Swift Testing supports different selection syntax (when available)
swift test --swift-testing-filter "Vector.*"
```

**Option 3: Split Test Targets**
- Create separate test targets for each batch
- Run each target independently

### Test Suite Mapping

For reference, here are the actual `@Suite` names:

**Batch 1: Core Vector Operations**
- "Vector Construction"
- "Vector Arithmetic"
- "Vector Normalization"
- "Distance Metrics"
- "Vector Entropy"
- "Serialization"
- "Vector Type Factory"
- "Dynamic Vector Tests"
- "Storage Alignment"

**Batch 2: Optimized Vectors**
- "Optimized Vector512"
- "Optimized Vector768"
- "Optimized Vector1536"

**Batch 3: Mixed Precision Core**
- "Mixed Precision Kernels"
- "FP16 Storage Types"
- "Mixed Precision Fuzzing Tests"
- "Mixed Precision Auto-Tuner"
- "Mixed Precision Batch Kernels"
- "Mixed Precision Part 1 - Validation"
- "Mixed Precision Comprehensive Tests"

*(See run_test_batches.sh for complete mapping)*

### Recommendations

1. **Short Term**: Run full test suite without batching
2. **Medium Term**: Use Swift Testing's selection mechanism when available
3. **Long Term**: Consider splitting test targets by functional area

## Current Test Execution

```bash
# Run all tests
swift test

# Run only XCTests (will exclude Swift Testing)
swift test --enable-xctest --disable-swift-testing

# Run only Swift Testing tests
swift test --disable-xctest --enable-swift-testing
```

## Known Issues

- **Index Out of Range**: Fixed in batch operations (see commit notes)
- **Storage Count Mismatch**: Fixed in mixed precision kernels (see commit notes)
- **Filter Matching**: Swift Testing tests cannot be filtered via `--filter` flag
