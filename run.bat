@echo off
title Drug Safety Intelligence Platform Launcher
color 0B

echo =================================================================
echo             🧬 DRUG SAFETY INTELLIGENCE PLATFORM 🧬
echo       FDA FAERS (2015-2026) * Multi-Model AI Predictions
echo =================================================================
echo.

:: 1. Verify Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0C
    echo [ERROR] Python was not found on your system path.
    echo Please install Python 3.9 or higher and check "Add Python to PATH" during setup.
    echo.
    pause
    exit /b
)

:: 2. Check and activate/create virtual environment
if not exist "venv" (
    echo [INFO] Virtual environment (venv) not found. Initializing setup...
    echo [INFO] Creating Python virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        color 0C
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b
    )
    echo [INFO] Activating virtual environment...
    call .\venv\Scripts\activate
    
    echo [INFO] Installing required dependencies (this may take ~1 minute)...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        color 0C
        echo [ERROR] Failed to install requirements. Please review requirements.txt.
        pause
        exit /b
    )
    echo [INFO] Dependencies installed successfully!
    echo.
) else (
    echo [INFO] Activating existing virtual environment (venv)...
    call .\venv\Scripts\activate
)

:: 3. Launch Streamlit Application
echo.
echo =================================================================
echo [LAUNCHING] Starting Streamlit Server...
echo =================================================================
echo.

streamlit run app.py

if %errorlevel% neq 0 (
    color 0C
    echo.
    echo [ERROR] Streamlit server crashed or exited with error code %errorlevel%.
    pause
)
