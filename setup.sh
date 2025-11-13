#!/bin/bash

# ============================================================================
# CRYPPO Setup Script for Linux / macOS
# Compatible with Python 3.10-3.12
# ============================================================================

echo "============================================================"
echo "CRYPPO - Linux/macOS Setup Script"
echo "============================================================"
echo ""
echo "This script will:"
echo "  1. Remove old virtual environment (if exists)"
echo "  2. Create fresh Python virtual environment"
echo "  3. Install PyTorch with CUDA 12.6 support (Linux) or CPU (macOS)"
echo "  4. Install all Python dependencies"
echo "  5. Install frontend Node.js dependencies"
echo "  6. Create necessary directories"
echo ""
echo "Requirements:"
echo "  - Python 3.10-3.12 installed"
echo "  - Node.js 18+ installed"
echo "  - NVIDIA GPU with CUDA 12.6 (optional, for GPU training on Linux)"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# ============================================================================
# 1. Check Python Version
# ============================================================================
echo ""
echo "[1/7] Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found!"
    echo "Please install Python 3.10-3.12 first."
    exit 1
fi

python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# ============================================================================
# 2. Clean Old Virtual Environment
# ============================================================================
echo ""
echo "[2/7] Cleaning old virtual environment..."
if [ -d "venv" ]; then
    echo "Removing old venv directory..."
    rm -rf venv
    echo "Old venv removed successfully!"
else
    echo "No old venv found, proceeding with fresh installation."
fi

# Clean Python cache
echo "Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null
echo "Python cache cleaned!"

# ============================================================================
# 3. Create Fresh Virtual Environment
# ============================================================================
echo ""
echo "[3/7] Creating fresh virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment!"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment!"
    exit 1
fi

echo "Virtual environment created and activated!"

# ============================================================================
# 4. Upgrade pip
# ============================================================================
echo ""
echo "[4/7] Upgrading pip..."
python -m pip install --upgrade pip
echo "pip upgraded successfully!"

# ============================================================================
# 5. Install PyTorch
# ============================================================================
echo ""
echo "[5/7] Installing PyTorch..."

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux - Installing PyTorch with CUDA 12.6 support..."
    echo "This may take a few minutes..."
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    if [ $? -ne 0 ]; then
        echo "WARNING: PyTorch CUDA installation failed!"
        echo "Trying CPU-only version..."
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS - Installing PyTorch CPU version..."
    pip3 install torch torchvision
else
    echo "Unknown OS - Installing PyTorch CPU version..."
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully!')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch installation verification failed!"
    exit 1
fi

# ============================================================================
# 6. Install Python Dependencies
# ============================================================================
echo ""
echo "[6/7] Installing Python dependencies..."
echo "This may take several minutes..."

pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "WARNING: Some packages failed to install."
    echo "Continuing anyway..."
fi

# Verify critical packages
echo ""
echo "Verifying critical packages:"
python -c "import websockets; print(f'  ✓ websockets: {websockets.__version__}')" 2>/dev/null || echo "  ✗ ERROR: websockets not installed!"
python -c "import fastapi; print(f'  ✓ fastapi: {fastapi.__version__}')" 2>/dev/null || echo "  ✗ ERROR: fastapi not installed!"
python -c "import ccxt; print(f'  ✓ ccxt: {ccxt.__version__}')" 2>/dev/null || echo "  ✗ ERROR: ccxt not installed!"
python -c "import gymnasium; print(f'  ✓ gymnasium: {gymnasium.__version__}')" 2>/dev/null || echo "  ✗ ERROR: gymnasium not installed!"
python -c "import pandas; print(f'  ✓ pandas: {pandas.__version__}')" 2>/dev/null || echo "  ✗ ERROR: pandas not installed!"

# Check TA-Lib (optional)
echo ""
if python -c "import talib" 2>/dev/null; then
    python -c "import talib; print(f'  ✓ TA-Lib: {talib.__version__}')"
else
    echo "  ⚠ TA-Lib not installed (optional)"
    echo "    To install:"
    echo "    Ubuntu/Debian: sudo apt-get install ta-lib && pip install TA-Lib"
    echo "    macOS: brew install ta-lib && pip install TA-Lib"
fi

echo ""
echo "Python dependencies installed!"

# ============================================================================
# 7. Install Frontend Dependencies
# ============================================================================
echo ""
echo "[7/7] Installing frontend dependencies..."

if ! command -v node &> /dev/null; then
    echo "WARNING: Node.js not found!"
    echo "Frontend will not work without Node.js 18+"
    echo "Download from: https://nodejs.org/"
else
    echo "Node version: $(node --version)"
    echo "NPM version: $(npm --version)"
    echo ""
    echo "Installing frontend packages..."
    cd client
    if [ -d "node_modules" ]; then
        rm -rf node_modules
    fi
    if [ -f "package-lock.json" ]; then
        rm -f package-lock.json
    fi
    npm install
    if [ $? -ne 0 ]; then
        echo "WARNING: Frontend installation had issues."
        echo "Try running 'npm install' manually in the client directory."
    else
        echo "Frontend dependencies installed successfully!"
    fi
    cd ..
fi

# ============================================================================
# 8. Create Necessary Directories
# ============================================================================
echo ""
echo "Creating necessary directories..."
mkdir -p data
mkdir -p logs
mkdir -p checkpoints
mkdir -p server/checkpoints
mkdir -p server/logs
echo "Directories created!"

# ============================================================================
# Setup Complete!
# ============================================================================
echo ""
echo "============================================================"
echo "Setup completed successfully!"
echo "============================================================"
echo ""
echo "To start using CRYPPO:"
echo ""
echo "  1. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Start backend server:"
echo "     cd server"
echo "     python main.py"
echo ""
echo "  3. Start frontend (in new terminal):"
echo "     cd client"
echo "     npm run dev"
echo ""
echo "  4. Access the application:"
echo "     Frontend: http://localhost:3000"
echo "     API Docs: http://localhost:8000/docs"
echo ""
echo "For CLI-only usage (testing):"
echo "  cd server"
echo "  python -m src.training.train_d3qn --episodes 10 --sample-data"
echo "  python test_backtest.py --sample-data"
echo ""
echo "============================================================"
echo "Check logs/ directory for error messages if issues occur."
echo "============================================================"
echo ""
