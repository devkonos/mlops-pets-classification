#!/bin/bash
# Docker entrypoint script for the API

set -e

echo "[STARTUP] Starting Cats vs Dogs Classification API..."
echo "[STARTUP] Python version: $(python --version)"
echo "[STARTUP] Working directory: $(pwd)"
echo "[STARTUP] Directory contents:"
ls -la

# Check if src directory exists
if [ ! -d "src" ]; then
    echo "[ERROR] src directory not found!"
    exit 1
fi

# Create necessary directories
echo "[STARTUP] Creating directories..."
mkdir -p ./models/artifacts ./data ./logs

# Print environment
echo "[STARTUP] Environment:"
echo "  API_HOST: ${API_HOST:-0.0.0.0}"
echo "  API_PORT: ${API_PORT:-8000}"
echo "  ENVIRONMENT: ${ENVIRONMENT:-development}"

# Start the API
echo "[STARTUP] Starting uvicorn server..."
exec python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --log-level info
