#!/bin/bash
# Run this before every submission to catch issues early

echo "================================================"
echo "  PRE-SUBMISSION VERIFICATION"
echo "================================================"

PASS=0
FAIL=0

check() {
    if eval "$2" > /dev/null 2>&1; then
        echo "  ✓ $1"
        PASS=$((PASS+1))
    else
        echo "  ✗ $1"
        FAIL=$((FAIL+1))
    fi
}

echo ""
echo "── Local file checks ────────────────────────"

# Check inference.py exists
check "inference.py exists" "[ -f inference.py ]"

# Check no sys.exit(1)
check "No sys.exit(1) in inference.py" "! grep -q 'sys.exit(1)' inference.py"

# Check syntax
check "inference.py syntax OK" "python3 -m py_compile inference.py"

# Check line count matches what we expect (not old 372-line version)
LINES=$(wc -l < inference.py)
check "inference.py is new version (<350 lines, got $LINES)" "[ $LINES -lt 350 ]"

# Check key patterns
check "Uses OpenAI client" "grep -q 'from openai import OpenAI' inference.py"
check "Reads HF_TOKEN" "grep -q 'HF_TOKEN' inference.py"
check "Reads API_BASE_URL" "grep -q 'API_BASE_URL' inference.py"
check "Has hf_placeholder fallback" "grep -q 'hf_placeholder' inference.py"
check "No local env import at module level" "! head -50 inference.py | grep -q 'from env'"
check "Has main() function" "grep -q 'def main' inference.py"
check "Has if __name__ block" "grep -q '__name__' inference.py"

echo ""
echo "── Git remote checks ────────────────────────"

GH_LINES=$(git show origin/main:inference.py 2>/dev/null | wc -l)
HF_LINES=$(git show hf/main:inference.py 2>/dev/null | wc -l)
LOCAL_LINES=$(wc -l < inference.py)

echo "  Local:  $LOCAL_LINES lines"
echo "  GitHub: $GH_LINES lines"
echo "  HF:     $HF_LINES lines"

check "GitHub matches local" "[ '$GH_LINES' = '$LOCAL_LINES' ]"
check "HF matches local" "[ '$HF_LINES' = '$LOCAL_LINES' ]"
check "GitHub has no sys.exit(1)" "! git show origin/main:inference.py | grep -q 'sys.exit(1)'"
check "HF has no sys.exit(1)" "! git show hf/main:inference.py | grep -q 'sys.exit(1)'"

echo ""
echo "── Live Space check ─────────────────────────"

SPACE_URL="https://heist-content-mod-openenv.hf.space"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    "$SPACE_URL/reset" \
    -H "Content-Type: application/json" \
    -d '{}' --max-time 15 2>/dev/null)

check "HF Space /reset returns 200 (got $HTTP_CODE)" "[ '$HTTP_CODE' = '200' ]"

echo ""
echo "── openenv validate ─────────────────────────"
check "openenv validate passes" "openenv validate 2>/dev/null | grep -q '\[OK\]'"

echo ""
echo "================================================"
echo "  Passed: $PASS  |  Failed: $FAIL"
echo "================================================"

if [ $FAIL -eq 0 ]; then
    echo "  🎉 ALL CHECKS PASSED — safe to submit!"
else
    echo "  ❌ Fix the failed checks before submitting."
fi
