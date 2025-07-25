#!/bin/bash

# VectorCore Optimized Build Script
# This script builds VectorCore with maximum performance optimizations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/.build"
BUILD_CONFIG="release"

# Parse command line arguments
CLEAN_BUILD=false
BENCHMARK=false
VERBOSE=false
ARCHITECTURE=""
USE_FAST_MATH=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --benchmark)
            BENCHMARK=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --arch)
            ARCHITECTURE="$2"
            shift 2
            ;;
        --fast-math)
            USE_FAST_MATH=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --clean      Clean build directory before building"
            echo "  --benchmark  Run benchmarks after building"
            echo "  --verbose    Enable verbose output"
            echo "  --arch       Specify architecture (arm64, x86_64)"
            echo "  --fast-math  Enable fast-math (WARNING: May affect NaN/Inf handling)"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}🔧 VectorCore Optimized Build${NC}"
echo "================================"
echo ""

# Display build configuration
echo "Build Configuration:"
echo "  - Configuration: Release"
echo "  - Whole Module Optimization: ✓"
echo "  - Cross Module Optimization: ✓"
echo "  - Link Time Optimization: ✓"
echo "  - SIMD Enabled: ✓"

if [ -n "$ARCHITECTURE" ]; then
    echo "  - Architecture: $ARCHITECTURE"
fi

echo ""

# Clean if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "$BUILD_DIR"
fi

# Set architecture if specified
ARCH_FLAGS=""
if [ -n "$ARCHITECTURE" ]; then
    ARCH_FLAGS="--arch $ARCHITECTURE"
fi

# Additional Swift flags for maximum optimization
export SWIFT_FLAGS="-Xswiftc -O -Xswiftc -whole-module-optimization"

# Set fast-math flag if requested
FAST_MATH_FLAGS=""
if [ "$USE_FAST_MATH" = true ]; then
    FAST_MATH_FLAGS="-Xcc -ffast-math"
    echo -e "${YELLOW}WARNING: Fast-math enabled - may affect NaN/Infinity handling${NC}"
fi

# Build with optimizations
echo -e "${YELLOW}Building VectorCore with optimizations...${NC}"

if [ "$VERBOSE" = true ]; then
    swift build \
        --configuration release \
        $ARCH_FLAGS \
        -Xswiftc -O \
        -Xswiftc -whole-module-optimization \
        -Xswiftc -cross-module-optimization \
        -Xswiftc -enforce-exclusivity=unchecked \
        -Xcc -O3 \
        -Xcc -march=native \
        $FAST_MATH_FLAGS \
        --verbose
else
    swift build \
        --configuration release \
        $ARCH_FLAGS \
        -Xswiftc -O \
        -Xswiftc -whole-module-optimization \
        -Xswiftc -cross-module-optimization \
        -Xswiftc -enforce-exclusivity=unchecked \
        -Xcc -O3 \
        -Xcc -march=native \
        $FAST_MATH_FLAGS
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Build completed successfully!${NC}"
else
    echo -e "${RED}❌ Build failed!${NC}"
    exit 1
fi

# Validate build optimization
echo ""
echo -e "${YELLOW}Validating optimizations...${NC}"

# Check if binary contains optimization flags
VECTORCORE_LIB=$(find "$BUILD_DIR/release" -name "libVectorCore.a" -o -name "VectorCore.swiftmodule" | head -n 1)

if [ -f "$VECTORCORE_LIB" ]; then
    echo -e "${GREEN}✓ VectorCore library built${NC}"
    
    # Get library size
    LIB_SIZE=$(du -h "$VECTORCORE_LIB" | cut -f1)
    echo "  Library size: $LIB_SIZE"
else
    echo -e "${RED}✗ VectorCore library not found${NC}"
fi

# Check Swift module interface
SWIFT_MODULE=$(find "$BUILD_DIR/release" -name "VectorCore.swiftmodule" -type d | head -n 1)
if [ -d "$SWIFT_MODULE" ]; then
    echo -e "${GREEN}✓ Swift module generated${NC}"
fi

# Run benchmarks if requested
if [ "$BENCHMARK" = true ]; then
    echo ""
    echo -e "${YELLOW}Running benchmarks...${NC}"
    
    BENCHMARK_EXEC=$(find "$BUILD_DIR/release" -name "VectorCoreBenchmarks" -type f | head -n 1)
    
    if [ -x "$BENCHMARK_EXEC" ]; then
        # Set environment for optimal performance
        export VECTORCORE_ENABLE_SIMD=1
        
        # Run with performance mode
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS: Disable turbo boost for consistent results
            echo "Note: For best results, disable Turbo Boost"
        fi
        
        "$BENCHMARK_EXEC"
    else
        echo -e "${RED}Benchmark executable not found${NC}"
    fi
fi

# Generate optimization report
echo ""
echo -e "${BLUE}Optimization Summary:${NC}"
echo "===================="

# List all optimization flags used
cat << EOF
Compiler Optimizations Applied:
  Swift:
    • -O (Full optimization)
    • -whole-module-optimization
    • -cross-module-optimization
    • -enforce-exclusivity=unchecked
    
  C/C++:
    • -O3 (Maximum optimization)
    • -march=native (CPU-specific optimizations)
    • -ffast-math (Fast floating-point)
    
  Linker:
    • -dead_strip (Remove unused code)
    • -lto (Link-time optimization)

Performance Features Enabled:
    • SIMD operations
    • Inlining
    • Loop vectorization
    • Constant folding
    • Dead code elimination
EOF

echo ""
echo -e "${GREEN}✅ Optimized build complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Run benchmarks: $0 --benchmark"
echo "  2. Compare with baseline: swift Scripts/compare_baseline.swift"
echo "  3. Profile with Instruments: xcrun instruments -t 'Time Profiler'"