@echo off
REM Build script for Goku-ELG documentation on Windows

echo ======================================
echo Goku-ELG Documentation Build Script
echo ======================================
echo.

REM Check if we're in the docs directory
if not exist "Makefile" (
    echo [Error] Please run this script from the docs\ directory
    exit /b 1
)

REM Check for Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [Error] Python is not installed or not in PATH
    exit /b 1
)
echo [OK] Python found

REM Check if virtual environment exists or create one
if not exist "venv" (
    echo [Info] Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
)

REM Activate virtual environment
echo [Info] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade requirements
echo [Info] Installing documentation requirements...
python -m pip install --upgrade pip >nul 2>nul
pip install -r requirements.txt >nul 2>nul
echo [OK] Requirements installed

REM Clean previous builds
echo [Info] Cleaning previous builds...
call make.bat clean >nul 2>nul
echo [OK] Previous builds cleaned

REM Build HTML documentation
echo.
echo Building HTML documentation...
call make.bat html
if %ERRORLEVEL% EQU 0 (
    echo [OK] HTML documentation built successfully!
    echo.
    echo Documentation is available at: build\html\index.html
) else (
    echo [Error] HTML build failed
    exit /b 1
)

REM Ask if user wants to build PDF
echo.
set /p BUILD_PDF="Do you want to build PDF documentation? (y/n): "
if /i "%BUILD_PDF%"=="y" (
    echo [Info] Building PDF documentation...
    call make.bat latexpdf
    if %ERRORLEVEL% EQU 0 (
        echo [OK] PDF documentation built successfully!
        echo PDF is available at: build\latex\goku-elg.pdf
    ) else (
        echo [Warning] PDF build failed (this requires LaTeX installation)
    )
)

echo.
echo ======================================
echo Build complete!
echo ======================================
pause
