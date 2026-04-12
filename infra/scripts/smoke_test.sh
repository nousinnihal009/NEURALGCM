#!/bin/bash
# Post-deployment smoke tests.
# Exits non-zero if any test fails (blocks CD pipeline).
set -euo pipefail

API_URL=${API_URL:-"http://localhost:8000"}
echo "════════════════════════════════════════════════════════════"
echo "  Running smoke tests against: $API_URL"
echo "════════════════════════════════════════════════════════════"
PASS=0; FAIL=0

check() {
  local desc="$1"; local cmd="$2"; local expected="$3"
  result=$(eval "$cmd" 2>/dev/null || echo "ERROR")
  if echo "$result" | grep -q "$expected"; then
    echo "  ✓ PASS: $desc"
    PASS=$((PASS+1))
  else
    echo "  ✗ FAIL: $desc (got: $result)"
    FAIL=$((FAIL+1))
  fi
}

echo ""
check "Health endpoint" \
  "curl -sf ${API_URL}/health | python3 -c \"import sys,json; d=json.load(sys.stdin); print(d['status'])\"" \
  "ok"

check "Ready endpoint" \
  "curl -sf ${API_URL}/ready | python3 -c \"import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))\"" \
  "ready"

check "OpenAPI docs accessible" \
  "curl -sf -o /dev/null -w '%{http_code}' ${API_URL}/api/v1/docs" \
  "200"

check "Root info endpoint" \
  "curl -sf -o /dev/null -w '%{http_code}' ${API_URL}/" \
  "200"

check "Forecast submit returns 202" \
  "curl -sf -o /dev/null -w '%{http_code}' -X POST ${API_URL}/api/v1/forecast \
    -H 'Content-Type: application/json' \
    -d '{\"location_name\":\"Test\",\"lat\":13.08,\"lon\":80.27,\"days\":1,\"mode\":\"historical\",\"init_date\":\"2020-06-01\"}'"\
  "202"

check "Metrics endpoint" \
  "curl -sf -o /dev/null -w '%{http_code}' ${API_URL}/metrics" \
  "200"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Results: $PASS passed, $FAIL failed"
echo "════════════════════════════════════════════════════════════"
[ "$FAIL" -eq 0 ] || { echo "  Smoke tests FAILED"; exit 1; }
echo "  All smoke tests PASSED ✓"
