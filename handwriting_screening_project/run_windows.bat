@echo off
REM One-click setup + launch for Windows. Double-click this file.
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
  echo Creating virtual environment...
  python -m venv venv || (echo Python not found. Install Python 3.11+ from python.org and tick "Add to PATH". & pause & exit /b 1)
  venv\Scripts\python -m pip install --upgrade pip
  venv\Scripts\pip install -r requirements.txt || (echo Install failed. & pause & exit /b 1)
)
venv\Scripts\python check_setup.py
echo.
echo Starting the app - your browser will open at http://localhost:8501  (Ctrl+C to stop)
venv\Scripts\streamlit run app.py
pause
