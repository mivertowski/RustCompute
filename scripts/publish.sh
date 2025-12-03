#!/bin/bash
#
# RingKernel Crates Publishing Script
#
# This script publishes all RingKernel crates to crates.io in the correct
# dependency order. It handles the complex dependency graph automatically.
#
# Usage:
#   ./scripts/publish.sh <CRATES_IO_TOKEN>
#   ./scripts/publish.sh --dry-run           # Test without publishing
#   ./scripts/publish.sh <TOKEN> --continue  # Continue from failed publish
#
# The publishing order respects the dependency graph:
#   Tier 1 (no deps):    core, cuda-codegen, wgpu-codegen
#   Tier 2 (core deps):  derive, cpu, cuda, wgpu, metal, codegen, ecosystem, audio-fft
#   Tier 3 (main crate): ringkernel
#   Tier 4 (apps):       wavesim, txmon
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PUBLISH_DELAY=45  # Seconds to wait between publishes for crates.io index
DRY_RUN=false
CONTINUE_MODE=false
SKIP_VERIFY=false

# Parse arguments
TOKEN=""
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --continue)
            CONTINUE_MODE=true
            shift
            ;;
        --skip-verify)
            SKIP_VERIFY=true
            shift
            ;;
        --help|-h)
            echo "RingKernel Crates Publisher"
            echo ""
            echo "Usage: $0 [OPTIONS] [TOKEN]"
            echo ""
            echo "Options:"
            echo "  --dry-run      Perform dry run without publishing"
            echo "  --continue     Skip already-published crates"
            echo "  --skip-verify  Skip initial verification step"
            echo "  --help         Show this help message"
            echo ""
            echo "Get your token from: https://crates.io/settings/tokens"
            exit 0
            ;;
        *)
            if [ -z "$TOKEN" ] && [[ ! "$arg" =~ ^-- ]]; then
                TOKEN="$arg"
            fi
            ;;
    esac
done

# Crates in dependency order (leaves first, main crate last)
# This order ensures each crate's dependencies are published before it
CRATES=(
    # Tier 1: No internal dependencies
    "ringkernel-core"
    "ringkernel-cuda-codegen"
    "ringkernel-wgpu-codegen"

    # Tier 2: Depends only on Tier 1
    "ringkernel-derive"      # depends on: core, cuda-codegen (optional)
    "ringkernel-cpu"         # depends on: core
    "ringkernel-cuda"        # depends on: core
    "ringkernel-wgpu"        # depends on: core
    "ringkernel-metal"       # depends on: core
    "ringkernel-codegen"     # depends on: core
    "ringkernel-ecosystem"   # depends on: core
    "ringkernel-audio-fft"   # depends on: core

    # Tier 3: Main crate (depends on most others)
    "ringkernel"             # depends on: core, derive, cpu, cuda, wgpu, metal, codegen

    # Tier 4: Application crates (depend on main crate)
    "ringkernel-wavesim"     # depends on: ringkernel, core, derive, cuda, cuda-codegen
    "ringkernel-txmon"       # depends on: ringkernel, core, cuda, cuda-codegen
)

# Tier 1 crates can be verified independently
TIER1_CRATES=(
    "ringkernel-core"
    "ringkernel-cuda-codegen"
    "ringkernel-wgpu-codegen"
)

print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}▶${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✖${NC} $1"
}

print_success() {
    echo -e "${GREEN}✔${NC} $1"
}

check_crate_published() {
    local crate=$1
    local version=$2
    # Check if crate version exists on crates.io
    local response=$(curl -s "https://crates.io/api/v1/crates/$crate/$version" 2>/dev/null)
    if echo "$response" | grep -q '"version"'; then
        return 0
    fi
    return 1
}

get_crate_version() {
    local crate=$1
    # Get version from workspace Cargo.toml
    grep -A1 '^\[workspace.package\]' Cargo.toml | grep 'version' | sed 's/.*"\(.*\)".*/\1/' | head -1
}

publish_crate() {
    local crate=$1
    local crate_dir="crates/$crate"

    if [ ! -d "$crate_dir" ]; then
        print_error "Crate directory not found: $crate_dir"
        return 1
    fi

    local version=$(get_crate_version "$crate")

    # Check if already published (for --continue mode)
    if [ "$CONTINUE_MODE" = true ]; then
        if check_crate_published "$crate" "$version"; then
            print_warning "$crate@$version already published, skipping..."
            return 0
        fi
    fi

    print_step "Publishing $crate@$version..."

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] Would publish: $crate@$version"
        # For dry run, just check that the crate builds
        if cargo check -p "$crate" 2>/dev/null; then
            print_success "Build check passed for $crate"
        else
            print_error "Build check failed for $crate"
            return 1
        fi
    else
        if cargo publish -p "$crate" --token "$TOKEN" 2>&1 | sed 's/^/  /'; then
            print_success "$crate@$version published successfully"
        else
            print_error "Failed to publish $crate"
            return 1
        fi
    fi
}

wait_for_index() {
    local crate=$1
    local version=$2
    local seconds=$3

    if [ "$DRY_RUN" = true ]; then
        return 0
    fi

    print_step "Waiting for $crate@$version to appear on crates.io..."

    # First, do a quick wait
    sleep 10

    # Then poll for availability (max wait time)
    local max_wait=$seconds
    local waited=0
    local interval=5

    while [ $waited -lt $max_wait ]; do
        if check_crate_published "$crate" "$version"; then
            print_success "$crate@$version is now available on crates.io"
            # Extra wait for index propagation
            sleep 5
            return 0
        fi
        echo -ne "\r  Waiting... [$waited/$max_wait seconds]"
        sleep $interval
        waited=$((waited + interval))
    done

    echo ""
    print_warning "Timeout waiting for $crate - continuing anyway..."
}

# Main script
print_header "RingKernel Crates Publisher"

VERSION=$(get_crate_version "ringkernel-core")

echo "Configuration:"
echo "  Version:       $VERSION"
echo "  Dry run:       $DRY_RUN"
echo "  Continue mode: $CONTINUE_MODE"
echo "  Crates count:  ${#CRATES[@]}"

# Validate token
if [ "$DRY_RUN" = false ] && [ -z "$TOKEN" ]; then
    echo ""
    print_error "No crates.io token provided!"
    echo ""
    echo "Usage: $0 <CRATES_IO_TOKEN> [--dry-run] [--continue]"
    echo ""
    echo "Options:"
    echo "  --dry-run    Test without publishing"
    echo "  --continue   Skip already-published crates"
    echo ""
    echo "Get your token from: https://crates.io/settings/tokens"
    exit 1
fi

# Change to workspace root
cd "$(dirname "$0")/.."
echo "  Working dir:   $(pwd)"

# Verify all crates exist
print_header "Verifying Crates"
for crate in "${CRATES[@]}"; do
    if [ -d "crates/$crate" ]; then
        print_success "$crate"
    else
        print_error "Missing: $crate"
        exit 1
    fi
done

# Run dry-run verification for Tier 1 crates (they have no internal deps)
if [ "$SKIP_VERIFY" = false ]; then
    print_header "Verifying Tier 1 Crates (cargo publish --dry-run)"
    echo "These crates have no internal dependencies and can be verified independently."
    echo ""

    for crate in "${TIER1_CRATES[@]}"; do
        echo -n "  Checking $crate... "
        if cargo publish -p "$crate" --dry-run --allow-dirty 2>/dev/null; then
            echo -e "${GREEN}OK${NC}"
        else
            echo -e "${RED}FAILED${NC}"
            print_error "Dry run failed for $crate"
            echo ""
            echo "Run for details: cargo publish -p $crate --dry-run --allow-dirty"
            exit 1
        fi
    done
    echo ""
    print_success "Tier 1 crates passed verification"
fi

# Show publish plan
print_header "Publish Plan"
echo "Crates will be published in this order:"
echo ""
echo -e "${CYAN}Tier 1 - No dependencies:${NC}"
for crate in "${TIER1_CRATES[@]}"; do
    echo "  • $crate"
done
echo ""
echo -e "${CYAN}Tier 2 - Depends on core:${NC}"
for crate in ringkernel-derive ringkernel-cpu ringkernel-cuda ringkernel-wgpu ringkernel-metal ringkernel-codegen ringkernel-ecosystem ringkernel-audio-fft; do
    echo "  • $crate"
done
echo ""
echo -e "${CYAN}Tier 3 - Main crate:${NC}"
echo "  • ringkernel"
echo ""
echo -e "${CYAN}Tier 4 - Application crates:${NC}"
echo "  • ringkernel-wavesim"
echo "  • ringkernel-txmon"

# Confirm before publishing
if [ "$DRY_RUN" = false ]; then
    print_header "Ready to Publish"
    echo "This will publish ${#CRATES[@]} crates to crates.io as version $VERSION."
    echo ""
    echo -e "${YELLOW}WARNING: This action cannot be undone!${NC}"
    echo ""
    read -p "Continue? (yes/no) " -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        echo "Aborted. Type 'yes' to confirm."
        exit 0
    fi
fi

# Publish crates
print_header "Publishing Crates"

published=0
skipped=0
failed=0

for i in "${!CRATES[@]}"; do
    crate="${CRATES[$i]}"
    version=$(get_crate_version "$crate")

    echo ""
    echo -e "${BLUE}[$((i+1))/${#CRATES[@]}]${NC} $crate@$version"
    echo "────────────────────────────────────────"

    # Check if already published
    if [ "$CONTINUE_MODE" = true ] && check_crate_published "$crate" "$version"; then
        print_warning "Already published, skipping..."
        skipped=$((skipped + 1))
        continue
    fi

    if publish_crate "$crate"; then
        published=$((published + 1))

        # Wait for index update (except for last crate and dry runs)
        if [ $((i+1)) -lt ${#CRATES[@]} ] && [ "$DRY_RUN" = false ]; then
            wait_for_index "$crate" "$version" $PUBLISH_DELAY
        fi
    else
        failed=$((failed + 1))
        print_error "Publishing stopped due to failure"
        echo ""
        echo "To continue from this point, run:"
        echo "  $0 <TOKEN> --continue"
        exit 1
    fi
done

# Summary
print_header "Summary"
echo -e "Published: ${GREEN}$published${NC}"
echo -e "Skipped:   ${YELLOW}$skipped${NC}"
echo -e "Failed:    ${RED}$failed${NC}"
echo ""

if [ $failed -eq 0 ]; then
    if [ "$DRY_RUN" = true ]; then
        print_success "Dry run completed successfully!"
        echo ""
        echo "To publish for real, run:"
        echo "  $0 <YOUR_CRATES_IO_TOKEN>"
    else
        print_success "All crates published successfully!"
        echo ""
        echo "View your crates at:"
        echo "  https://crates.io/crates/ringkernel"
        echo "  https://crates.io/crates/ringkernel-core"
        echo "  https://crates.io/crates/ringkernel-cuda-codegen"
    fi
else
    print_error "Some crates failed to publish"
    exit 1
fi
