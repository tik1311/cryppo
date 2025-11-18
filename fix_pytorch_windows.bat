@echo off
REM ============================================================================
REM Quick Fix: PyTorch DLL Error on Windows
REM This script switches from CUDA to CPU-only PyTorch
REM ============================================================================

echo ============================================================
echo PyTorch DLL Error - Quick Fix
echo ============================================================
echo.
echo This script will:
echo   1. Uninstall PyTorch CUDA version (causing DLL errors)
echo   2. Install PyTorch CPU-only version
echo   3. Verify the installation works
echo.
echo Note: Training will be slower on CPU, but everything will work!
echo.
pause

REM Check if venv is activated
python -c "import sys; exit(0 if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 1)" 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: Virtual environment not activated!
    echo Please run: venv\Scripts\activate
    echo Then run this script again.
    pause
    exit /b 1
)

echo.
echo [1/3] Uninstalling PyTorch CUDA version...
pip3 uninstall -y torch torchvision torchaudio
if errorlevel 1 (
    echo WARNING: Uninstall had issues, continuing anyway...
)

echo.
echo [2/3] Installing PyTorch CPU-only version...
echo This may take a few minutes...
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch CPU version!
    pause
    exit /b 1
)

echo.
echo [3/3] Verifying installation...
python -c "import torch; print(f'✓ PyTorch {torch.__version__} installed successfully!')"
if errorlevel 1 (
    echo ERROR: PyTorch still has issues!
    echo.
    echo This might be due to missing Visual C++ Redistributable.
    echo Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo.
    pause
    exit /b 1
)

python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()} (CPU mode)')"
python -c "import torch; print(f'✓ Device: {torch.device(\"cpu\")}')"

echo.
echo ============================================================
echo PyTorch CPU version installed successfully!
echo ============================================================
echo.
echo What changed:
echo   - PyTorch now uses CPU instead of CUDA
echo   - No more DLL errors
echo   - Training will work (just slower)
echo.
echo You can now:
echo   1. Restart the server: python main.py
echo   2. Start training through the UI
echo.
echo Training speed:
echo   - CPU: ~5-10 minutes for 10 episodes (testing)
echo   - GPU: ~30-60 seconds for 10 episodes
echo.
echo If you want GPU support later:
echo   1. Install Visual C++ Redistributable
echo   2. Install CUDA Toolkit 12.6
echo   3. Run: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
echo.
pause
