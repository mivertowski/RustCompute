#!/bin/bash
#
# RingKernel Python Package Publishing Script
#
# This script publishes the ringkernel Python package to PyPI using maturin.
#
# Usage:
#   ./scripts/publish-python.sh <PYPI_TOKEN>
#   ./scripts/publish-python.sh --dry-run           # Build without publishing
#   ./scripts/publish-python.sh --test-pypi <TOKEN> # Publish to TestPyPI first
#   ./scripts/publish-python.sh --status            # Check if version is published
#
# Prerequisites:
#   - maturin installed: pip install maturin
#   - PyPI token from: https://pypi.org/manage/account/token/
#
# Features to build (optional, specify with --features):
#   - cuda: CUDA device support
#   - benchmark: Benchmark framework
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
DRY_RUN=false
TEST_PYPI=false
STATUS_ONLY=false
FEATURES=""
PACKAGE_NAME="ringkernel"
CRATE_DIR="crates/ringkernel-python"

# Parse arguments
TOKEN=""
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --test-pypi)
            TEST_PYPI=true
            shift
            ;;
        --status)
            STATUS_ONLY=true
            shift
            ;;
        --features=*)
            FEATURES="${arg#*=}"
            shift
            ;;
        --help|-h)
            echo "RingKernel Python Package Publisher"
            echo ""
            echo "Usage: $0 [OPTIONS] [TOKEN]"
            echo ""
            echo "Options:"
            echo "  --dry-run          Build wheels without publishing"
            echo "  --test-pypi        Publish to TestPyPI instead of PyPI"
            echo "  --status           Check if current version is published"
            echo "  --features=FEAT    Comma-separated features (cuda,benchmark)"
            echo "  --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --dry-run                    # Test build"
            echo "  $0 --test-pypi <TOKEN>          # Publish to TestPyPI"
            echo "  $0 <TOKEN>                      # Publish to PyPI"
            echo "  $0 --features=cuda,benchmark <TOKEN>"
            echo ""
            echo "Get your PyPI token from: https://pypi.org/manage/account/token/"
            echo "Get TestPyPI token from: https://test.pypi.org/manage/account/token/"
            exit 0
            ;;
        *)
            if [ -z "$TOKEN" ] && [[ ! "$arg" =~ ^-- ]]; then
                TOKEN="$arg"
            fi
            ;;
    esac
done

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

get_version() {
    # Get version from pyproject.toml
    grep '^version = ' "$CRATE_DIR/pyproject.toml" | sed 's/version = "\(.*\)"/\1/' | head -1
}

check_pypi_published() {
    local version=$1
    local registry_url="https://pypi.org/pypi/$PACKAGE_NAME/$version/json"

    if [ "$TEST_PYPI" = true ]; then
        registry_url="https://test.pypi.org/pypi/$PACKAGE_NAME/$version/json"
    fi

    local response=$(curl -s -o /dev/null -w "%{http_code}" "$registry_url" 2>/dev/null)
    if [ "$response" = "200" ]; then
        return 0
    fi
    return 1
}

check_maturin() {
    if ! command -v maturin &> /dev/null; then
        print_error "maturin is not installed!"
        echo ""
        echo "Install it with:"
        echo "  pip install maturin"
        echo ""
        echo "Or with pipx:"
        echo "  pipx install maturin"
        exit 1
    fi
    print_success "maturin $(maturin --version | cut -d' ' -f2) found"
}

# Main script
print_header "RingKernel Python Package Publisher"

# Change to workspace root
cd "$(dirname "$0")/.."

# Check prerequisites
print_step "Checking prerequisites..."
check_maturin

# Verify crate directory exists
if [ ! -d "$CRATE_DIR" ]; then
    print_error "Crate directory not found: $CRATE_DIR"
    exit 1
fi
print_success "Found $CRATE_DIR"

VERSION=$(get_version)
if [ -z "$VERSION" ]; then
    print_error "Could not determine version from pyproject.toml"
    exit 1
fi

# Determine registry
REGISTRY="PyPI"
REGISTRY_URL="https://pypi.org/project/$PACKAGE_NAME/"
if [ "$TEST_PYPI" = true ]; then
    REGISTRY="TestPyPI"
    REGISTRY_URL="https://test.pypi.org/project/$PACKAGE_NAME/"
fi

echo ""
echo "Configuration:"
echo "  Package:     $PACKAGE_NAME"
echo "  Version:     $VERSION"
echo "  Registry:    $REGISTRY"
echo "  Features:    ${FEATURES:-none}"
echo "  Dry run:     $DRY_RUN"
echo "  Crate dir:   $CRATE_DIR"

# Status check
if [ "$STATUS_ONLY" = true ]; then
    print_header "Publishing Status"
    echo -n "  $PACKAGE_NAME@$VERSION on $REGISTRY: "
    if check_pypi_published "$VERSION"; then
        echo -e "${GREEN}✔ published${NC}"
        echo ""
        echo "View at: $REGISTRY_URL$VERSION/"
    else
        echo -e "${YELLOW}○ not published${NC}"
    fi
    exit 0
fi

# Check if already published
print_header "Checking Publishing Status"
echo -n "  $PACKAGE_NAME@$VERSION on $REGISTRY: "
if check_pypi_published "$VERSION"; then
    echo -e "${GREEN}✔ already published${NC}"
    echo ""
    print_success "Version $VERSION is already published on $REGISTRY"
    echo ""
    echo "View at: $REGISTRY_URL$VERSION/"
    exit 0
else
    echo -e "${YELLOW}○ not published${NC}"
fi

# Validate token for real publish
if [ "$DRY_RUN" = false ] && [ -z "$TOKEN" ]; then
    echo ""
    print_error "No PyPI token provided!"
    echo ""
    echo "Usage: $0 <PYPI_TOKEN> [--test-pypi]"
    echo ""
    echo "Get your token from:"
    echo "  PyPI:     https://pypi.org/manage/account/token/"
    echo "  TestPyPI: https://test.pypi.org/manage/account/token/"
    exit 1
fi

# Build the package
print_header "Building Package"

cd "$CRATE_DIR"

BUILD_ARGS="--release"
if [ -n "$FEATURES" ]; then
    BUILD_ARGS="$BUILD_ARGS --features $FEATURES"
fi

print_step "Running: maturin build $BUILD_ARGS"
echo ""

if maturin build $BUILD_ARGS; then
    print_success "Build completed successfully"
else
    print_error "Build failed"
    exit 1
fi

# Show built wheels
echo ""
print_step "Built wheels:"
ls -la target/wheels/*.whl 2>/dev/null | sed 's/^/  /'

if [ "$DRY_RUN" = true ]; then
    print_header "Dry Run Complete"
    print_success "Package built successfully (not published)"
    echo ""
    echo "Wheels are in: $CRATE_DIR/target/wheels/"
    echo ""
    echo "To publish for real, run:"
    echo "  $0 <PYPI_TOKEN>"
    exit 0
fi

# Publish
print_header "Publishing to $REGISTRY"

PUBLISH_ARGS="--username __token__ --password $TOKEN"
if [ "$TEST_PYPI" = true ]; then
    PUBLISH_ARGS="$PUBLISH_ARGS --repository-url https://test.pypi.org/legacy/"
fi
if [ -n "$FEATURES" ]; then
    PUBLISH_ARGS="$PUBLISH_ARGS --features $FEATURES"
fi

print_step "Publishing $PACKAGE_NAME@$VERSION to $REGISTRY..."
echo ""

# Hide token in output
if maturin publish $PUBLISH_ARGS 2>&1 | grep -v "$TOKEN"; then
    echo ""
    print_success "Successfully published $PACKAGE_NAME@$VERSION to $REGISTRY!"
else
    echo ""
    print_error "Failed to publish"
    exit 1
fi

# Verify publication
print_header "Verifying Publication"
echo "Waiting for package to appear on $REGISTRY..."
sleep 5

MAX_WAIT=60
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if check_pypi_published "$VERSION"; then
        print_success "Package is now available on $REGISTRY"
        break
    fi
    echo -ne "\r  Waiting... [$WAITED/$MAX_WAIT seconds]"
    sleep 5
    WAITED=$((WAITED + 5))
done

echo ""

# Summary
print_header "Summary"
print_success "Published: $PACKAGE_NAME@$VERSION"
echo ""
echo "Install with:"
if [ "$TEST_PYPI" = true ]; then
    echo "  pip install --index-url https://test.pypi.org/simple/ $PACKAGE_NAME"
else
    echo "  pip install $PACKAGE_NAME"
fi
echo ""
echo "View at: $REGISTRY_URL"
