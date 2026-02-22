#!/bin/bash
# API testing script

API_URL="${1:-http://localhost:8000}"

echo "=========================================="
echo "Testing Cats vs Dogs API"
echo "URL: $API_URL"
echo "=========================================="
echo ""

# Test 1: Health Check
echo "[1/4] Testing /health endpoint..."
curl -s -w "\nStatus: %{http_code}\n" "$API_URL/health" | jq .
echo ""

# Test 2: Root endpoint
echo "[2/4] Testing root endpoint..."
curl -s -w "\nStatus: %{http_code}\n" "$API_URL/" | jq .
echo ""

# Test 3: Info endpoint
echo "[3/4] Testing /info endpoint..."
curl -s -w "\nStatus: %{http_code}\n" "$API_URL/info" | jq .
echo ""

# Test 4: Metrics endpoint
echo "[4/4] Testing /metrics endpoint..."
curl -s -w "\nStatus: %{http_code}\n" "$API_URL/metrics" | head -20
echo ""
echo "... (truncated)"
echo ""

echo "=========================================="
echo "API Tests Completed"
echo "=========================================="
