#!/bin/bash
# Docker entrypoint script for the API

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

# Test if app can be imported
echo "[STARTUP] Testing app import..."
python -c "from src.api.app import app; print('[OK] App imported successfully')" || {
    echo "[ERROR] Failed to import app!"
    exit 1
}

# Start the API with error handling and logging
echo "[STARTUP] Starting uvicorn server..."
python -m uvicorn src.api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    --access-log \
    2>&1 | tee /tmp/api.log

# If we get here, something went wrong
echo "[ERROR] API exited unexpectedly"
echo "[ERROR] Last 50 lines of log:"
tail -50 /tmp/api.log || true
exit 1
