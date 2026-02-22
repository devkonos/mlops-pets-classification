#!/bin/bash
# Smoke tests for deployed service

set -e

API_URL="${1:-http://localhost:8000}"
TIMEOUT=60
ELAPSED=0

echo "=== Cats vs Dogs API - Smoke Tests ==="
echo "API URL: $API_URL"
echo ""

# Wait for API to be ready
echo "[1/6] Waiting for API to be ready..."
while [ $ELAPSED -lt $TIMEOUT ]; do
    if curl -s -f "$API_URL/health" > /dev/null 2>&1; then
        echo "[OK] API is ready"
        break
    fi
    echo "  Waiting... ($ELAPSED/$TIMEOUT seconds)"
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "[FAIL] API did not become ready within $TIMEOUT seconds"
    exit 1
fi

# Test health endpoint
echo ""
echo "[2/6] Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s "$API_URL/health")
echo "Response: $HEALTH_RESPONSE"
if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo "[OK] Health check passed"
else
    echo "[FAIL] Health check failed"
    exit 1
fi

# Test info endpoint
echo ""
echo "[3/6] Testing info endpoint..."
INFO_RESPONSE=$(curl -s "$API_URL/info")
echo "Response: $INFO_RESPONSE"
if echo "$INFO_RESPONSE" | grep -q "model_loaded"; then
    echo "[OK] Info endpoint works"
else
    echo "[FAIL] Info endpoint failed"
    exit 1
fi

# Test root endpoint
echo ""
echo "[4/6] Testing root endpoint..."
ROOT_RESPONSE=$(curl -s "$API_URL/")
echo "Response: $ROOT_RESPONSE"
if echo "$ROOT_RESPONSE" | grep -q "healthy"; then
    echo "[OK] Root endpoint works"
else
    echo "[FAIL] Root endpoint failed"
    exit 1
fi

# Test metrics endpoint (if available)
echo ""
echo "[5/6] Testing metrics endpoint..."
if curl -s -f "$API_URL/metrics" > /dev/null 2>&1; then
    echo "[OK] Metrics endpoint is available"
else
    echo "[WARN] Metrics endpoint not available"
fi

# Performance test
echo ""
echo "[6/6] Testing response times..."
START_TIME=$(date +%s%N)
curl -s "$API_URL/health" > /dev/null
END_TIME=$(date +%s%N)
RESPONSE_TIME=$(( (END_TIME - START_TIME) / 1000000 ))
echo "Health check response time: ${RESPONSE_TIME}ms"
if [ $RESPONSE_TIME -lt 2000 ]; then
    echo "[OK] Response time is good"
else
    echo "[WARN] Response time is high: ${RESPONSE_TIME}ms"
fi

echo ""
echo "=== All Smoke Tests Passed ==="
exit 0
