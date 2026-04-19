@echo off
REM ============================================================================
REM ViT Cardiac Pathology Project Setup Script (Windows)
REM ============================================================================
REM This script creates the complete project structure for the Vision Transformer
REM cardiac pathology detection research project.
REM ============================================================================

setlocal enabledelayedexpansion

REM Set the project root directory
set PROJECT_ROOT=E:\PORTFOLIO\MEDICAL IMAGING\vision_transformer

echo.
echo ============================================================================
echo  ViT Cardiac Pathology Project Setup
echo ============================================================================
echo.
echo Project Root: %PROJECT_ROOT%
echo.

REM Check if project directory exists
if not exist "%PROJECT_ROOT%" (
    echo ERROR: Project directory does not exist!
    echo Please create: %PROJECT_ROOT%
    pause
    exit /b 1
)

echo [1/4] Creating data directory structure...
mkdir "%PROJECT_ROOT%\data\raw\images" 2>nul
mkdir "%PROJECT_ROOT%\data\processed\train\images" 2>nul
mkdir "%PROJECT_ROOT%\data\processed\val\images" 2>nul
mkdir "%PROJECT_ROOT%\data\processed\test\images" 2>nul
mkdir "%PROJECT_ROOT%\data\splits" 2>nul
mkdir "%PROJECT_ROOT%\data\cache" 2>nul
echo [✓] Data directories created

echo.
echo [2/4] Creating source code directory structure...
mkdir "%PROJECT_ROOT%\src\data" 2>nul
mkdir "%PROJECT_ROOT%\src\models" 2>nul
mkdir "%PROJECT_ROOT%\src\training" 2>nul
mkdir "%PROJECT_ROOT%\src\interpretability" 2>nul
mkdir "%PROJECT_ROOT%\src\utils" 2>nul
echo [✓] Source directories created

echo.
echo [3/4] Creating auxiliary directories...
mkdir "%PROJECT_ROOT%\notebooks" 2>nul
mkdir "%PROJECT_ROOT%\scripts" 2>nul
mkdir "%PROJECT_ROOT%\results\checkpoints" 2>nul
mkdir "%PROJECT_ROOT%\results\attention_maps" 2>nul
mkdir "%PROJECT_ROOT%\results\analysis" 2>nul
mkdir "%PROJECT_ROOT%\results\figures" 2>nul
mkdir "%PROJECT_ROOT%\tests" 2>nul
echo [✓] Auxiliary directories created

echo.
echo [4/4] Creating __init__.py files for Python packages...
type nul > "%PROJECT_ROOT%\src\__init__.py"
type nul > "%PROJECT_ROOT%\src\data\__init__.py"
type nul > "%PROJECT_ROOT%\src\models\__init__.py"
type nul > "%PROJECT_ROOT%\src\training\__init__.py"
type nul > "%PROJECT_ROOT%\src\interpretability\__init__.py"
type nul > "%PROJECT_ROOT%\src\utils\__init__.py"
type nul > "%PROJECT_ROOT%\tests\__init__.py"
echo [✓] __init__.py files created

echo.
echo ============================================================================
echo  PROJECT STRUCTURE CREATED SUCCESSFULLY!
echo ============================================================================
echo.
echo Next Steps:
echo.
echo 1. MOVE YOUR DATA:
echo    - Copy your CheXchoNet images folder to:
echo      %PROJECT_ROOT%\data\raw\images\
echo    - Copy metadata.csv to:
echo      %PROJECT_ROOT%\data\raw\
echo    - Copy metadata.txt to:
echo      %PROJECT_ROOT%\data\raw\
echo.
echo 2. VERIFY DATA:
echo    - Check that all files are in correct locations
echo    - Run data verification script (coming next)
echo.
echo 3. SET UP PYTHON ENVIRONMENT:
echo    a) Open Command Prompt in project directory
echo    b) Create virtual environment: python -m venv venv
echo    c) Activate it: venv\Scripts\activate
echo    d) Install dependencies: pip install -r requirements.txt
echo.
echo 4. OPEN IN PYCHARM:
echo    - File > Open > Select %PROJECT_ROOT%
echo    - Configure Python interpreter to use venv\Scripts\python.exe
echo.
echo ============================================================================
echo.
pause
