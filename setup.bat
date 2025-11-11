@echo off
REM RL Crypto Trading System - Setup Script (Windows)
REM 1H Timeframe Trading System

echo ==========================================
echo RL Crypto Trading System - Setup
echo Version: 2.0.0 ^| Timeframe: 1H Only
echo ==========================================
echo.

REM Check Python
echo Checking Python version...
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.10 or higher.
    pause
    exit /b 1
)

REM Create virtual environment (optional)
set /p create_venv="Create virtual environment? (y/n): "
if "%create_venv%"=="y" (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Virtual environment activated
)

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

REM Create directories
echo.
echo Creating directories...
if not exist "data\npz" mkdir data\npz
if not exist "logs" mkdir logs
if not exist "checkpoints" mkdir checkpoints
if not exist "results" mkdir results
echo Directories created

REM Check TA-Lib
echo.
echo Checking TA-Lib installation...
python -c "import talib" 2>nul
if errorlevel 1 (
    echo WARNING: TA-Lib not found!
    echo Please install TA-Lib manually:
    echo   Download from: https://github.com/mrjbq7/ta-lib#windows
    echo   Then: pip install TA-Lib
)

REM Check CUDA
echo.
echo Checking CUDA availability...
python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    echo WARNING: Could not check CUDA availability
)

echo.
echo ==========================================
echo Setup completed!
echo ==========================================
echo.
echo To start the backend server:
echo   cd server
echo   python main.py
echo.
echo To download data:
echo   cd server
echo   python download_data.py --symbols BTC/USDT ETH/USDT
echo.
echo API Documentation will be available at:
echo   http://localhost:8000/docs
echo.
echo ==========================================
pause
