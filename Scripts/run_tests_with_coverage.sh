#!/bin/bash

# VectorCore Test Runner with Coverage
# This script runs the test suite and generates coverage reports

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/.build"
COVERAGE_DIR="${PROJECT_DIR}/coverage"

# Parse command line arguments
RUN_EXTENDED=false
VERBOSE=false
PROPERTY_ITERATIONS=100
PERFORMANCE_ITERATIONS=100

while [[ $# -gt 0 ]]; do
    case $1 in
        --extended)
            RUN_EXTENDED=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --property-iterations)
            PROPERTY_ITERATIONS="$2"
            shift 2
            ;;
        --performance-iterations)
            PERFORMANCE_ITERATIONS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --extended              Run extended test suite"
            echo "  --verbose               Enable verbose output"
            echo "  --property-iterations N Set property test iterations (default: 100)"
            echo "  --performance-iterations N Set performance test iterations (default: 100)"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "🧪 VectorCore Test Suite"
echo "========================"
echo ""

# Set environment variables
export VECTORCORE_PROPERTY_ITERATIONS="$PROPERTY_ITERATIONS"
export VECTORCORE_PERF_ITERATIONS="$PERFORMANCE_ITERATIONS"

if [ "$RUN_EXTENDED" = true ]; then
    export VECTORCORE_TEST_EXTENDED=1
    echo "Running extended test suite..."
fi

if [ "$VERBOSE" = true ]; then
    export VECTORCORE_TEST_VERBOSE=1
    echo "Verbose mode enabled..."
fi

# Clean previous coverage data
echo -e "${YELLOW}Cleaning previous coverage data...${NC}"
rm -rf "$COVERAGE_DIR"
mkdir -p "$COVERAGE_DIR"

# Build and run tests with coverage
echo -e "${YELLOW}Running tests with coverage...${NC}"
swift test --enable-code-coverage --parallel

# Check if tests passed
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
else
    echo -e "${RED}❌ Tests failed!${NC}"
    exit 1
fi

# Generate coverage report
echo -e "${YELLOW}Generating coverage report...${NC}"

# Find the test executable
TEST_BINARY=$(find "$BUILD_DIR/debug" -name "VectorCorePackageTests.xctest" -type d | head -n 1)
if [ -z "$TEST_BINARY" ]; then
    # Try alternative location
    TEST_BINARY=$(find "$BUILD_DIR/debug" -name "*PackageTests" -type f | head -n 1)
fi

if [ -z "$TEST_BINARY" ]; then
    echo -e "${RED}Could not find test binary${NC}"
    exit 1
fi

# Find the profdata file
PROF_DATA=$(find "$BUILD_DIR/debug/codecov" -name "default.profdata" | head -n 1)
if [ -z "$PROF_DATA" ]; then
    echo -e "${RED}Could not find coverage data${NC}"
    exit 1
fi

# Export coverage data
echo "Exporting coverage data..."

# Generate LCOV format
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    xcrun llvm-cov export \
        "$TEST_BINARY/Contents/MacOS/VectorCorePackageTests" \
        -instr-profile="$PROF_DATA" \
        -format="lcov" \
        -ignore-filename-regex="Tests|Benchmarks|Scripts" \
        > "$COVERAGE_DIR/coverage.lcov"
else
    # Linux
    llvm-cov export \
        "$TEST_BINARY" \
        -instr-profile="$PROF_DATA" \
        -format="lcov" \
        -ignore-filename-regex="Tests|Benchmarks|Scripts" \
        > "$COVERAGE_DIR/coverage.lcov"
fi

# Generate HTML report
echo "Generating HTML report..."
if command -v genhtml &> /dev/null; then
    genhtml "$COVERAGE_DIR/coverage.lcov" \
        --output-directory "$COVERAGE_DIR/html" \
        --title "VectorCore Coverage Report" \
        --quiet
    echo -e "${GREEN}HTML report generated at: $COVERAGE_DIR/html/index.html${NC}"
fi

# Generate text summary
echo -e "\n${YELLOW}Coverage Summary:${NC}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    xcrun llvm-cov report \
        "$TEST_BINARY/Contents/MacOS/VectorCorePackageTests" \
        -instr-profile="$PROF_DATA" \
        -ignore-filename-regex="Tests|Benchmarks|Scripts" \
        -use-color
else
    llvm-cov report \
        "$TEST_BINARY" \
        -instr-profile="$PROF_DATA" \
        -ignore-filename-regex="Tests|Benchmarks|Scripts"
fi

# Parse coverage percentage
COVERAGE_LINE=$(xcrun llvm-cov report \
    "$TEST_BINARY/Contents/MacOS/VectorCorePackageTests" \
    -instr-profile="$PROF_DATA" \
    -ignore-filename-regex="Tests|Benchmarks|Scripts" \
    | grep "TOTAL" | awk '{print $4}' | sed 's/%//')

echo ""
echo -e "Overall Coverage: ${GREEN}${COVERAGE_LINE}%${NC}"

# Check against threshold
THRESHOLD=85
if (( $(echo "$COVERAGE_LINE < $THRESHOLD" | bc -l) )); then
    echo -e "${RED}❌ Coverage ${COVERAGE_LINE}% is below threshold ${THRESHOLD}%${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Coverage meets threshold${NC}"
fi

# Save coverage metrics
echo "{
  \"timestamp\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\",
  \"coverage\": $COVERAGE_LINE,
  \"threshold\": $THRESHOLD,
  \"propertyIterations\": $PROPERTY_ITERATIONS,
  \"performanceIterations\": $PERFORMANCE_ITERATIONS,
  \"extended\": $RUN_EXTENDED
}" > "$COVERAGE_DIR/metrics.json"

echo ""
echo -e "${GREEN}✅ Test run complete!${NC}"
echo "Coverage report: $COVERAGE_DIR/coverage.lcov"
echo "Metrics saved: $COVERAGE_DIR/metrics.json"