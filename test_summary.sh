#!/bin/bash

echo "Running test summary..."

test_files=(
    "AlignedValueStorageTests"
    "AlignmentVerificationTests"
    "BatchOperationsTests"
    "BinarySerializationTests"
    "ConcurrencyTests"
    "CoreProtocolsTests"
    "COWDynamicStorageTests"
    "CRC32Tests"
    "CrossPlatformTests"
    "DistanceMetricTests"
    "DynamicVectorTests"
    "ErrorHandlingTests"
    "GenericVectorTests"
    "IntegrationTests"
    "LoggerTests"
    "OptimizedSIMDStorageTests"
    "PerformanceTests"
    "PropertyBasedTests"
    "ProtocolTests"
    "StorageTests"
    "ValueSemanticsTests"
    "VectorFactoryTests"
    "VectorMathTests"
    "VectorTypeTests"
)

total_tests=0
failed_tests=0
passing_suites=0
failing_suites=0

for test in "${test_files[@]}"; do
    echo -n "Testing $test... "
    result=$(swift test --filter "VectorCoreTests.$test" 2>&1 | grep -E "Executed.*tests" | tail -1)
    
    if [[ -n "$result" ]]; then
        tests=$(echo "$result" | sed -n 's/.*Executed \([0-9]*\) tests.*/\1/p')
        failures=$(echo "$result" | sed -n 's/.*with \([0-9]*\) failures.*/\1/p')
        
        if [[ -n "$tests" ]]; then
            total_tests=$((total_tests + tests))
            if [[ "$failures" == "0" ]]; then
                echo "✓ $tests tests passed"
                passing_suites=$((passing_suites + 1))
            else
                echo "✗ $failures/$tests tests failed"
                failed_tests=$((failed_tests + failures))
                failing_suites=$((failing_suites + 1))
            fi
        else
            echo "No tests found"
        fi
    else
        echo "No results"
    fi
done

echo "==========================="
echo "Total test suites: ${#test_files[@]}"
echo "Passing suites: $passing_suites"
echo "Failing suites: $failing_suites"
echo "Total tests: $total_tests"
echo "Failed tests: $failed_tests"
echo "Passed tests: $((total_tests - failed_tests))"

if [[ $total_tests -gt 0 ]]; then
    success_rate=$(awk "BEGIN {printf \"%.2f\", ($total_tests - $failed_tests) * 100 / $total_tests}")
    echo "Success rate: $success_rate%"
fi