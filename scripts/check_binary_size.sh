#!/usr/bin/env bash
#
# Binary Size Validation Script
#
# Validates README claim: Binary size <2MB (vs DotCompute 5-10MB)
#
# This script builds the ringkernel library in release mode and measures
# the resulting binary sizes.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Target: <2MB (2097152 bytes)
TARGET_SIZE_BYTES=2097152
TARGET_SIZE_MB=2

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}RingKernel Binary Size Validation${NC}"
echo -e "${BLUE}Target: <${TARGET_SIZE_MB}MB (${TARGET_SIZE_BYTES} bytes)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

# Clean previous builds
echo -e "${YELLOW}Cleaning previous builds...${NC}"
cargo clean 2>/dev/null || true

# Build in release mode
echo -e "${YELLOW}Building ringkernel in release mode...${NC}"
cargo build --release --package ringkernel 2>&1 | tail -5

echo ""
echo -e "${BLUE}Binary Size Analysis:${NC}"
echo -e "${BLUE}-------------------------------------------${NC}"

# Function to format size
format_size() {
    local bytes=$1
    if [ "$bytes" -ge 1048576 ]; then
        echo "$(echo "scale=2; $bytes / 1048576" | bc)MB"
    elif [ "$bytes" -ge 1024 ]; then
        echo "$(echo "scale=2; $bytes / 1024" | bc)KB"
    else
        echo "${bytes}B"
    fi
}

# Function to check and report size
check_size() {
    local file=$1
    local name=$2

    if [ -f "$file" ]; then
        local size
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
        local formatted_size
        formatted_size=$(format_size "$size")

        if [ "$size" -lt "$TARGET_SIZE_BYTES" ]; then
            echo -e "${GREEN}[PASS]${NC} $name: $formatted_size ($size bytes)"
            return 0
        else
            echo -e "${RED}[FAIL]${NC} $name: $formatted_size ($size bytes) - exceeds ${TARGET_SIZE_MB}MB target"
            return 1
        fi
    else
        echo -e "${YELLOW}[SKIP]${NC} $name: not found"
        return 0
    fi
}

# Check library files
passed=0
failed=0
skipped=0

# Check static library (.a)
if check_size "target/release/libringkernel.a" "Static library (libringkernel.a)"; then
    ((passed++))
else
    ((failed++))
fi

# Check dynamic library (.so / .dylib)
if [ -f "target/release/libringkernel.so" ]; then
    if check_size "target/release/libringkernel.so" "Dynamic library (libringkernel.so)"; then
        ((passed++))
    else
        ((failed++))
    fi
elif [ -f "target/release/libringkernel.dylib" ]; then
    if check_size "target/release/libringkernel.dylib" "Dynamic library (libringkernel.dylib)"; then
        ((passed++))
    else
        ((failed++))
    fi
else
    echo -e "${YELLOW}[SKIP]${NC} Dynamic library: not built"
    ((skipped++))
fi

# Check rlib
rlib_file=$(find target/release/deps -name "libringkernel-*.rlib" 2>/dev/null | head -1)
if [ -n "$rlib_file" ] && [ -f "$rlib_file" ]; then
    if check_size "$rlib_file" "Rust library (.rlib)"; then
        ((passed++))
    else
        ((failed++))
    fi
else
    echo -e "${YELLOW}[SKIP]${NC} Rust library (.rlib): not found"
    ((skipped++))
fi

echo ""
echo -e "${BLUE}-------------------------------------------${NC}"
echo -e "${BLUE}Component Breakdown:${NC}"
echo -e "${BLUE}-------------------------------------------${NC}"

# Check individual crate sizes (if available as separate .rlib files)
for crate in ringkernel-core ringkernel-cpu ringkernel-derive ringkernel-codegen; do
    rlib=$(find target/release/deps -name "lib${crate//-/_}-*.rlib" 2>/dev/null | head -1)
    if [ -n "$rlib" ] && [ -f "$rlib" ]; then
        size=$(stat -f%z "$rlib" 2>/dev/null || stat -c%s "$rlib" 2>/dev/null || echo "0")
        formatted=$(format_size "$size")
        echo "  $crate: $formatted"
    fi
done

# Build example binaries to check executable size
echo ""
echo -e "${YELLOW}Building example binaries...${NC}"
cargo build --release --examples 2>&1 | tail -3

echo ""
echo -e "${BLUE}Example Binary Sizes:${NC}"
echo -e "${BLUE}-------------------------------------------${NC}"

for example in hello_kernel vector_add ping_pong; do
    if [ -f "target/release/examples/$example" ]; then
        size=$(stat -f%z "target/release/examples/$example" 2>/dev/null || stat -c%s "target/release/examples/$example" 2>/dev/null || echo "0")
        formatted=$(format_size "$size")

        if [ "$size" -lt "$TARGET_SIZE_BYTES" ]; then
            echo -e "${GREEN}[PASS]${NC} $example: $formatted"
        else
            echo -e "${RED}[FAIL]${NC} $example: $formatted (exceeds target)"
        fi
    fi
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Summary:${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "  Passed: ${GREEN}$passed${NC}"
echo -e "  Failed: ${RED}$failed${NC}"
echo -e "  Skipped: ${YELLOW}$skipped${NC}"
echo ""

if [ "$failed" -eq 0 ]; then
    echo -e "${GREEN}Binary size validation PASSED!${NC}"
    echo -e "${GREEN}All binaries are under the ${TARGET_SIZE_MB}MB target.${NC}"
    exit 0
else
    echo -e "${RED}Binary size validation FAILED!${NC}"
    echo -e "${RED}Some binaries exceed the ${TARGET_SIZE_MB}MB target.${NC}"
    exit 1
fi
