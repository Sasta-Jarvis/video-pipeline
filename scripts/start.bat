@echo off
REM Sets up (on first run) and launches the video pipeline. Windows.
setlocal

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
cd /d "%PROJECT_DIR%"

set "VENV_DIR=%PROJECT_DIR%\.venv"

if not exist "%VENV_DIR%" (
    echo Creating virtual environment at %VENV_DIR% ...
    python -m venv "%VENV_DIR%"
)

call "%VENV_DIR%\Scripts\activate.bat"

echo Installing/upgrading dependencies...
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo.
python scripts\inspect_env.py
if errorlevel 1 (
    echo.
    echo Environment check failed ^(see above^). Fix ffmpeg/Python and re-run.
    exit /b 1
)

if not exist "config.yaml" (
    echo.
    echo No config.yaml found — copying config.example.yaml -^> config.yaml
    copy config.example.yaml config.yaml
)

echo.
echo Starting video pipeline. Press Ctrl+C to stop.
echo.
set "PYTHONPATH=%PROJECT_DIR%\src;%PYTHONPATH%"
python -m video_pipeline.main --config config.yaml
