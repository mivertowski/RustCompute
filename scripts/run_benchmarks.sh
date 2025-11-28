#!/usr/bin/env bash
#
# Comprehensive Benchmark Runner Script
#
# Runs all benchmarks and validates performance claims from the README:
# - Serialization: <30ns
# - Message latency: <500µs
# - Startup time: <1ms
# - Binary size: <2MB
#
# Usage:
#   ./scripts/run_benchmarks.sh              # Run all benchmarks
#   ./scripts/run_benchmarks.sh quick        # Quick validation run
#   ./scripts/run_benchmarks.sh claims       # Run claims validation only
#   ./scripts/run_benchmarks.sh <benchmark>  # Run specific benchmark

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Navigate to project root
cd "$(dirname "$0")/.."

# Performance targets from README
declare -A TARGETS=(
    ["serialization"]="<30ns"
    ["message_latency"]="<500µs"
    ["startup_time"]="<1ms"
    ["binary_size"]="<2MB"
)

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_subheader() {
    echo ""
    echo -e "${CYAN}--- $1 ---${NC}"
    echo ""
}

# Available benchmarks
ALL_BENCHMARKS=(
    "serialization"
    "message_queue"
    "startup"
    "latency"
    "memory_layout"
    "hlc"
    "k2k_messaging"
    "claims_validation"
)

# Quick validation benchmarks
QUICK_BENCHMARKS=(
    "claims_validation"
)

# Claims validation benchmarks (comprehensive)
CLAIMS_BENCHMARKS=(
    "claims_validation"
    "serialization"
    "latency"
    "startup"
    "memory_layout"
)

run_benchmark() {
    local bench_name=$1
    local extra_args=${2:-}

    print_subheader "Running benchmark: $bench_name"

    if cargo bench --package ringkernel --bench "$bench_name" -- $extra_args; then
        echo -e "${GREEN}[OK]${NC} $bench_name completed"
    else
        echo -e "${RED}[FAIL]${NC} $bench_name failed"
        return 1
    fi
}

run_all_benchmarks() {
    print_header "Running ALL Benchmarks"

    local passed=0
    local failed=0

    for bench in "${ALL_BENCHMARKS[@]}"; do
        if run_benchmark "$bench"; then
            ((passed++))
        else
            ((failed++))
        fi
    done

    print_header "Benchmark Summary"
    echo -e "  Passed: ${GREEN}$passed${NC}"
    echo -e "  Failed: ${RED}$failed${NC}"

    return $failed
}

run_quick_validation() {
    print_header "Quick Validation Run"

    echo "This runs the comprehensive claims validation benchmark"
    echo "to quickly verify all README performance claims."
    echo ""

    for bench in "${QUICK_BENCHMARKS[@]}"; do
        run_benchmark "$bench" "--quick"
    done
}

run_claims_validation() {
    print_header "Claims Validation Suite"

    echo "Performance Targets:"
    echo -e "  - Serialization: ${CYAN}${TARGETS[serialization]}${NC} (DotCompute: 20-50ns)"
    echo -e "  - Message Latency: ${CYAN}${TARGETS[message_latency]}${NC} (DotCompute: <1ms)"
    echo -e "  - Startup Time: ${CYAN}${TARGETS[startup_time]}${NC} (DotCompute: ~10ms)"
    echo -e "  - Binary Size: ${CYAN}${TARGETS[binary_size]}${NC} (DotCompute: 5-10MB)"
    echo ""

    local passed=0
    local failed=0

    for bench in "${CLAIMS_BENCHMARKS[@]}"; do
        if run_benchmark "$bench"; then
            ((passed++))
        else
            ((failed++))
        fi
    done

    # Also run binary size check
    print_subheader "Binary Size Validation"
    if [ -f "scripts/check_binary_size.sh" ]; then
        if bash scripts/check_binary_size.sh; then
            ((passed++))
        else
            ((failed++))
        fi
    else
        echo -e "${YELLOW}[SKIP]${NC} Binary size script not found"
    fi

    print_header "Claims Validation Summary"
    echo -e "  Passed: ${GREEN}$passed${NC}"
    echo -e "  Failed: ${RED}$failed${NC}"

    return $failed
}

run_specific_benchmark() {
    local bench_name=$1
    shift
    local extra_args="$*"

    # Validate benchmark exists
    local found=false
    for bench in "${ALL_BENCHMARKS[@]}"; do
        if [ "$bench" == "$bench_name" ]; then
            found=true
            break
        fi
    done

    if [ "$found" = false ]; then
        echo -e "${RED}Error: Unknown benchmark '$bench_name'${NC}"
        echo ""
        echo "Available benchmarks:"
        for bench in "${ALL_BENCHMARKS[@]}"; do
            echo "  - $bench"
        done
        exit 1
    fi

    run_benchmark "$bench_name" "$extra_args"
}

show_help() {
    echo "RingKernel Benchmark Suite"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  (none)     Run all benchmarks"
    echo "  quick      Quick validation run (claims_validation only)"
    echo "  claims     Run comprehensive claims validation suite"
    echo "  <name>     Run specific benchmark"
    echo "  list       List available benchmarks"
    echo "  help       Show this help message"
    echo ""
    echo "Available benchmarks:"
    for bench in "${ALL_BENCHMARKS[@]}"; do
        echo "  - $bench"
    done
    echo ""
    echo "Performance targets from README:"
    for key in "${!TARGETS[@]}"; do
        echo "  - $key: ${TARGETS[$key]}"
    done
    echo ""
    echo "Examples:"
    echo "  $0                          # Run all benchmarks"
    echo "  $0 quick                    # Quick validation"
    echo "  $0 claims                   # Full claims validation"
    echo "  $0 serialization            # Run serialization benchmark"
    echo "  $0 hlc --save-baseline      # Run HLC benchmark with baseline"
}

list_benchmarks() {
    echo "Available benchmarks:"
    echo ""
    for bench in "${ALL_BENCHMARKS[@]}"; do
        echo "  $bench"
    done
    echo ""
    echo "Quick validation: ${QUICK_BENCHMARKS[*]}"
    echo "Claims validation: ${CLAIMS_BENCHMARKS[*]}"
}

# Main entry point
main() {
    local command=${1:-all}
    shift 2>/dev/null || true

    case "$command" in
        "all"|"")
            run_all_benchmarks
            ;;
        "quick")
            run_quick_validation
            ;;
        "claims")
            run_claims_validation
            ;;
        "list")
            list_benchmarks
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            run_specific_benchmark "$command" "$@"
            ;;
    esac
}

# Print banner
echo -e "${BLUE}"
echo "  ____  _             _  __                    _ "
echo " |  _ \(_)_ __   __ _| |/ /___ _ __ _ __   ___| |"
echo " | |_) | | '_ \ / _\` | ' // _ \ '__| '_ \ / _ \ |"
echo " |  _ <| | | | | (_| | . \  __/ |  | | | |  __/ |"
echo " |_| \_\_|_| |_|\__, |_|\_\___|_|  |_| |_|\___|_|"
echo "                |___/                            "
echo "               Benchmark Suite                   "
echo -e "${NC}"

main "$@"
