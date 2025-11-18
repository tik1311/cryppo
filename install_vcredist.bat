@echo off
REM ============================================================================
REM Automated Visual C++ Redistributable Installer
REM Fixes PyTorch DLL Error for GPU Support
REM ============================================================================

echo ============================================================
echo Visual C++ Redistributable Installer
echo Required for PyTorch GPU (CUDA) Support
echo ============================================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script must be run as Administrator!
    echo.
    echo Right-click this file and select "Run as administrator"
    pause
    exit /b 1
)

echo [1/3] Downloading Visual C++ Redistributable...
echo Download URL: https://aka.ms/vs/17/release/vc_redist.x64.exe
echo.

REM Create temp directory
if not exist "%TEMP%\cryppo" mkdir "%TEMP%\cryppo"

REM Download using PowerShell
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile '%TEMP%\cryppo\vc_redist.x64.exe'}"

if not exist "%TEMP%\cryppo\vc_redist.x64.exe" (
    echo ERROR: Download failed!
    echo Please download manually from: https://aka.ms/vs/17/release/vc_redist.x64.exe
    pause
    exit /b 1
)

echo Download completed!
echo.

echo [2/3] Installing Visual C++ Redistributable...
echo This may take a few minutes...
echo.

REM Install silently
"%TEMP%\cryppo\vc_redist.x64.exe" /install /quiet /norestart

if errorlevel 1 (
    echo WARNING: Installation may have failed or already installed
    echo Continuing anyway...
)

echo.
echo [3/3] Cleaning up...
del "%TEMP%\cryppo\vc_redist.x64.exe"
rmdir "%TEMP%\cryppo"

echo.
echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo IMPORTANT: You must RESTART YOUR COMPUTER for changes to take effect!
echo.
echo After restart:
echo   1. Activate venv: venv\Scripts\activate
echo   2. Install PyTorch GPU: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
echo   3. Verify: python -c "import torch; print(torch.cuda.is_available())"
echo.
pause
