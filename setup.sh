#!/bin/bash

# RL Crypto Trading System - Setup Script
# 1H Timeframe Trading System

echo "=========================================="
echo "RL Crypto Trading System - Setup"
echo "Version: 2.0.0 | Timeframe: 1H Only"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment (optional but recommended)
read -p "Create virtual environment? (y/n): " create_venv
if [ "$create_venv" = "y" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Virtual environment activated"
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data/npz
mkdir -p logs
mkdir -p checkpoints
mkdir -p results
echo "Directories created"

# Check for TA-Lib
echo ""
echo "Checking TA-Lib installation..."
if python -c "import talib" 2>/dev/null; then
    echo "✓ TA-Lib is installed"
else
    echo "⚠ TA-Lib not found. Please install it manually:"
    echo "  Ubuntu/Debian: sudo apt-get install ta-lib"
    echo "  macOS: brew install ta-lib"
    echo "  Then: pip install TA-Lib"
fi

# Check for CUDA
echo ""
echo "Checking CUDA availability..."
if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    echo "PyTorch CUDA check completed"
else
    echo "⚠ Could not check CUDA availability"
fi

echo ""
echo "=========================================="
echo "Setup completed!"
echo "=========================================="
echo ""
echo "To start the backend server:"
echo "  cd server"
echo "  python main.py"
echo ""
echo "To download data:"
echo "  cd server"
echo "  python download_data.py --symbols BTC/USDT ETH/USDT"
echo ""
echo "API Documentation will be available at:"
echo "  http://localhost:8000/docs"
echo ""
echo "=========================================="
