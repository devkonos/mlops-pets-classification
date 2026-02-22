#!/bin/bash
# Initial project setup script

set -e

echo "=========================================="
echo "Cats vs Dogs MLOps - Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "[1/9] Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"
if ! command -v python3 &> /dev/null; then
    echo "[FAIL] Python 3 is required but not installed"
    exit 1
fi

# Create virtual environment
echo ""
echo "[2/9] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "[OK] Virtual environment created"
else
    echo "[OK] Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "[3/9] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "[4/9] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo ""
echo "[5/9] Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "[6/9] Creating data directories..."
mkdir -p data/raw data/processed data/splits
mkdir -p models/artifacts
mkdir -p logs
mkdir -p mlruns
echo "[OK] Directories created"

# DVC initialization
echo ""
echo "[7/9] Initializing DVC..."
if ! dvc status > /dev/null 2>&1; then
    dvc init --no-scm 2>/dev/null || true
    echo "[OK] DVC initialized"
else
    echo "[OK] DVC already initialized"
fi

# Install development dependencies (optional)
echo ""
read -p "Install development dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing dev dependencies..."
    pip install pytest pytest-cov flake8 black jupyter notebook
    echo "[OK] Dev dependencies installed"
fi

# Setup pre-commit hooks (optional)
echo ""
read -p "Setup pre-commit hooks? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install pre-commit
    pre-commit install 2>/dev/null || true
    echo "[OK] Pre-commit hooks installed"
fi

# Download dataset (optional)
echo ""
echo "[8/9] Dataset Setup"
read -p "Download Cats vs Dogs dataset from Kaggle? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Make sure Kaggle API credentials are configured at ~/.kaggle/kaggle.json"
    python src/data/download_data.py
    echo "[OK] Dataset download initiated"
else
    echo "[SKIP] Skipping dataset download"
fi

# Run quick tests
echo ""
echo "[9/9] Running quick validation tests..."
python -m pytest tests/ -v -k "config" --tb=short 2>/dev/null || true

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Download data: python src/data/download_data.py"
echo "  3. Train model: make train"
echo "  4. Start API: make api-server"
echo "  5. View dashboard: make mlflow-ui"
echo ""
