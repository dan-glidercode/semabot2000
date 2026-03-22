#!/usr/bin/env bash
# ==============================================================================
# SeMaBot2000 Quality Gate
# ==============================================================================
# Enforces code formatting (black), linting (ruff), cognitive complexity
# (flake8), and test coverage (90%+).
# Run this before every commit and at every implementation stage.
#
# Usage:
#   bash scripts/check.sh          # Run all checks
#   bash scripts/check.sh --fix    # Auto-fix formatting and lint issues
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

COVERAGE_THRESHOLD=90
MAX_COGNITIVE_COMPLEXITY=12
SRC_DIR="src/semabot"
TEST_DIR="tests"
FIX_MODE=false

if [[ "${1:-}" == "--fix" ]]; then
    FIX_MODE=true
fi

pass() { echo -e "  ${GREEN}PASS${NC} $1"; }
fail() { echo -e "  ${RED}FAIL${NC} $1"; }
warn() { echo -e "  ${YELLOW}WARN${NC} $1"; }

ERRORS=0

echo ""
echo "============================================================"
echo "  SeMaBot2000 Quality Gate"
echo "============================================================"
echo ""

# --------------------------------------------------------------------------
# 1. Check that source directory exists
# --------------------------------------------------------------------------
echo "[1/5] Checking project structure..."
if [[ ! -d "$SRC_DIR" ]]; then
    warn "Source directory '$SRC_DIR' not found. Skipping code checks."
    echo ""
    echo "  Project not yet scaffolded. Quality gate will enforce once"
    echo "  $SRC_DIR exists. Create it in Phase 0, Task 0.2."
    echo ""
    exit 0
fi
pass "Source directory exists"

# --------------------------------------------------------------------------
# 2. Formatting (black)
# --------------------------------------------------------------------------
echo ""
echo "[2/5] Code formatting (black)..."
if $FIX_MODE; then
    if python -m black "$SRC_DIR" "$TEST_DIR" 2>/dev/null; then
        pass "Formatted with black"
    else
        fail "black formatting failed"
        ERRORS=$((ERRORS + 1))
    fi
else
    if python -m black --check --quiet "$SRC_DIR" "$TEST_DIR" 2>/dev/null; then
        pass "All files formatted correctly"
    else
        fail "Files need formatting. Run: bash scripts/check.sh --fix"
        ERRORS=$((ERRORS + 1))
    fi
fi

# --------------------------------------------------------------------------
# 3. Linting (ruff)
# --------------------------------------------------------------------------
echo ""
echo "[3/5] Linting (ruff)..."
if $FIX_MODE; then
    if python -m ruff check --fix "$SRC_DIR" "$TEST_DIR" 2>/dev/null; then
        pass "Lint issues fixed"
    else
        fail "ruff fix failed — manual fixes needed"
        ERRORS=$((ERRORS + 1))
    fi
else
    if python -m ruff check "$SRC_DIR" "$TEST_DIR" 2>/dev/null; then
        pass "No lint issues"
    else
        fail "Lint issues found. Run: bash scripts/check.sh --fix"
        ERRORS=$((ERRORS + 1))
    fi
fi

# --------------------------------------------------------------------------
# 4. Cognitive complexity (flake8)
# --------------------------------------------------------------------------
echo ""
echo "[4/5] Cognitive complexity (flake8, max=${MAX_COGNITIVE_COMPLEXITY})..."
if python -m flake8 --max-cognitive-complexity="$MAX_COGNITIVE_COMPLEXITY" \
    --select=CCR "$SRC_DIR" 2>/dev/null; then
    pass "All functions within complexity threshold"
else
    fail "Functions exceed cognitive complexity ${MAX_COGNITIVE_COMPLEXITY}. Refactor into smaller functions."
    ERRORS=$((ERRORS + 1))
fi

# --------------------------------------------------------------------------
# 5. Tests + Coverage
# --------------------------------------------------------------------------
echo ""
echo "[5/5] Tests + coverage (pytest, threshold=${COVERAGE_THRESHOLD}%)..."

if [[ ! -d "$TEST_DIR" ]] || [[ -z "$(find "$TEST_DIR" -name 'test_*.py' 2>/dev/null)" ]]; then
    warn "No test files found in '$TEST_DIR'. Skipping coverage check."
else
    COVERAGE_REPORT=$(python -m pytest "$TEST_DIR" \
        --cov="$SRC_DIR" \
        --cov-report=term-missing \
        --cov-fail-under="$COVERAGE_THRESHOLD" \
        -q 2>&1) || true

    if echo "$COVERAGE_REPORT" | grep -q "FAILED\|ERROR"; then
        fail "Tests failed"
        echo "$COVERAGE_REPORT" | tail -20
        ERRORS=$((ERRORS + 1))
    elif echo "$COVERAGE_REPORT" | grep -q "Required test coverage of ${COVERAGE_THRESHOLD}%"; then
        fail "Coverage below ${COVERAGE_THRESHOLD}%"
        echo "$COVERAGE_REPORT" | grep -E "^(TOTAL|Required)" | head -5
        ERRORS=$((ERRORS + 1))
    else
        TOTAL_LINE=$(echo "$COVERAGE_REPORT" | grep "^TOTAL" | head -1)
        if [[ -n "$TOTAL_LINE" ]]; then
            COVERAGE_PCT=$(echo "$TOTAL_LINE" | awk '{print $NF}' | tr -d '%')
            pass "All tests passed, coverage: ${COVERAGE_PCT}%"
        else
            pass "All tests passed"
        fi
    fi
fi

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
echo ""
echo "------------------------------------------------------------"
if [[ $ERRORS -eq 0 ]]; then
    echo -e "  ${GREEN}ALL CHECKS PASSED${NC}"
else
    echo -e "  ${RED}${ERRORS} CHECK(S) FAILED${NC}"
fi
echo "------------------------------------------------------------"
echo ""

exit $ERRORS
