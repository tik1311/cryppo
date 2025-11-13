@echo off
REM ============================================================================
REM CRYPPO Setup Script for Windows 11
REM Compatible with Python 3.10-3.12
REM ============================================================================

echo ============================================================
echo CRYPPO - Windows 11 Setup Script
echo ============================================================
echo.
echo This script will:
echo   1. Remove old virtual environment (if exists)
echo   2. Create fresh Python virtual environment
echo   3. Install PyTorch with CUDA 12.6 support
echo   4. Install all Python dependencies
echo   5. Install frontend Node.js dependencies
echo   6. Create necessary directories
echo.
echo Requirements:
echo   - Python 3.10-3.12 installed and in PATH
echo   - Node.js 18+ installed and in PATH
echo   - NVIDIA GPU with CUDA 12.6 (optional, for GPU training)
echo.
pause

REM ============================================================================
REM 1. Check Python Version
REM ============================================================================
echo.
echo [1/7] Checking Python version...
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.10-3.12
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM ============================================================================
REM 2. Clean Old Virtual Environment
REM ============================================================================
echo.
echo [2/7] Cleaning old virtual environment...
if exist venv (
    echo Removing old venv directory...
    rmdir /s /q venv
    echo Old venv removed successfully!
) else (
    echo No old venv found, proceeding with fresh installation.
)

REM Also clean any Python cache
if exist __pycache__ rmdir /s /q __pycache__
if exist server\__pycache__ rmdir /s /q server\__pycache__
if exist server\src\__pycache__ rmdir /s /q server\src\__pycache__
for /d /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
for /d /r %%d in (*.egg-info) do @if exist "%%d" rmdir /s /q "%%d"

echo Python cache cleaned!

REM ============================================================================
REM 3. Create Fresh Virtual Environment
REM ============================================================================
echo.
echo [3/7] Creating fresh virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment!
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)

echo Virtual environment created and activated!

REM ============================================================================
REM 4. Upgrade pip
REM ============================================================================
echo.
echo [4/7] Upgrading pip...
python -m pip install --upgrade pip
echo pip upgraded successfully!

REM ============================================================================
REM 5. Install PyTorch
REM ============================================================================
echo.
echo [5/7] Installing PyTorch...
echo.
echo IMPORTANT: PyTorch CUDA requires Microsoft Visual C++ Redistributable
echo If you don't have it installed, download from:
echo https://aka.ms/vs/17/release/vc_redist.x64.exe
echo.
echo Choose PyTorch version:
echo   1. CUDA 12.6 (for NVIDIA GPU - requires CUDA Toolkit)
echo   2. CPU only (recommended if you don't have CUDA or get DLL errors)
echo.
set /p pytorch_choice="Enter your choice (1 or 2, default=2): "

if "%pytorch_choice%"=="" set pytorch_choice=2
if "%pytorch_choice%"=="1" (
    echo.
    echo Installing PyTorch with CUDA 12.6 support...
    echo This may take a few minutes...
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    goto verify_torch
)

echo.
echo Installing PyTorch CPU-only version...
echo This may take a few minutes...
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

:verify_torch
echo.
echo Verifying PyTorch installation...
python -c "import torch; print(f'✓ PyTorch {torch.__version__} installed successfully!')" 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: PyTorch DLL initialization failed!
    echo This usually means:
    echo   1. Missing Visual C++ Redistributable
    echo   2. Missing CUDA libraries (if using CUDA version)
    echo.
    echo SOLUTION: Installing CPU-only version instead...
    pip3 uninstall -y torch torchvision
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    echo.
    echo Verifying CPU version...
    python -c "import torch; print(f'✓ PyTorch {torch.__version__} (CPU) installed successfully!')"
    if errorlevel 1 (
        echo ERROR: PyTorch installation failed completely!
        echo Please install Visual C++ Redistributable and try again.
        pause
        exit /b 1
    )
) else (
    python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"
)

echo PyTorch installation verified!

REM ============================================================================
REM 6. Install Python Dependencies
REM ============================================================================
echo.
echo [6/7] Installing Python dependencies...
echo This may take several minutes...

REM Install without TA-Lib first (it requires binary)
pip install -r requirements.txt
if errorlevel 1 (
    echo WARNING: Some packages failed to install.
    echo Continuing anyway...
)

REM Verify critical packages
echo.
echo Verifying critical packages:
python -c "import websockets; print(f'  websockets: {websockets.__version__}')" || echo   ERROR: websockets not installed!
python -c "import fastapi; print(f'  fastapi: {fastapi.__version__}')" || echo   ERROR: fastapi not installed!
python -c "import ccxt; print(f'  ccxt: {ccxt.__version__}')" || echo   ERROR: ccxt not installed!
python -c "import gymnasium; print(f'  gymnasium: {gymnasium.__version__}')" || echo   ERROR: gymnasium not installed!
python -c "import pandas; print(f'  pandas: {pandas.__version__}')" || echo   ERROR: pandas not installed!

echo.
echo Python dependencies installed!

REM ============================================================================
REM 7. Install Frontend Dependencies
REM ============================================================================
echo.
echo [7/7] Installing frontend dependencies...
echo Checking Node.js...
node --version
if errorlevel 1 (
    echo WARNING: Node.js not found!
    echo Frontend will not work without Node.js 18+
    echo Download from: https://nodejs.org/
) else (
    echo Installing frontend packages...
    cd client
    if exist node_modules rmdir /s /q node_modules
    if exist package-lock.json del /f package-lock.json
    call npm install
    if errorlevel 1 (
        echo WARNING: Frontend installation had issues.
        echo Try running 'npm install' manually in the client directory.
    ) else (
        echo Frontend dependencies installed successfully!
    )
    cd ..
)

REM ============================================================================
REM 8. Create Necessary Directories
REM ============================================================================
echo.
echo Creating necessary directories...
if not exist data mkdir data
if not exist logs mkdir logs
if not exist checkpoints mkdir checkpoints
if not exist server\checkpoints mkdir server\checkpoints
if not exist server\logs mkdir server\logs
echo Directories created!

REM ============================================================================
REM Setup Complete!
REM ============================================================================
echo.
echo ============================================================
echo Setup completed successfully!
echo ============================================================
echo.
echo To start using CRYPPO:
echo.
echo   1. Activate virtual environment:
echo      venv\Scripts\activate
echo.
echo   2. Start backend server:
echo      cd server
echo      python main.py
echo.
echo   3. Start frontend (in new terminal):
echo      cd client
echo      npm run dev
echo.
echo   4. Access the application:
echo      Frontend: http://localhost:3000
echo      API Docs: http://localhost:8000/docs
echo.
echo For CLI-only usage (testing):
echo   cd server
echo   python -m src.training.train_d3qn --episodes 10 --sample-data
echo   python test_backtest.py --sample-data
echo.
echo ============================================================
echo Check logs/ directory for error messages if issues occur.
echo ============================================================
echo.
pause
