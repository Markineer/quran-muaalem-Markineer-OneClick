@echo off
setlocal EnableDelayedExpansion

title Quran Muaalem - Starting...

cls
echo.
echo ========================================================================
echo.
echo                         QURAN MUAALEM
echo                    Intelligent Quran Teacher
echo.
echo ========================================================================
echo.

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

echo [1/5] Checking prerequisites...

where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed!
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo   [OK] Python %PYTHON_VERSION% found

where node >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js is not installed!
    echo Please install Node.js 18+ from https://nodejs.org
    pause
    exit /b 1
)

for /f "tokens=1" %%i in ('node --version 2^>^&1') do set NODE_VERSION=%%i
echo   [OK] Node.js %NODE_VERSION% found

where npm >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] npm is not installed!
    pause
    exit /b 1
)
echo   [OK] npm found
echo.

echo [2/5] Setting up Python environment...

if not exist "venv" (
    echo   Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
)

call venv\Scripts\activate.bat

echo   Upgrading pip...
python -m pip install --upgrade pip -q

echo   [OK] Virtual environment ready
echo.

echo [3/5] Installing Python dependencies...

python -c "import torch; import fastapi; import quran_transcript" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo   Installing packages - this may take a few minutes...
    pip install -r requirements.txt -q
    pip install fastapi uvicorn python-multipart websockets -q
    pip install -e . -q 2>nul
    echo   [OK] Python dependencies installed
) else (
    echo   [OK] Python dependencies already installed
)
echo.

echo [4/5] Installing frontend dependencies...

cd frontend

if not exist "node_modules" (
    echo   Installing npm packages...
    call npm install --silent
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install npm packages
        cd ..
        pause
        exit /b 1
    )
    echo   [OK] Frontend dependencies installed
) else (
    echo   [OK] Frontend dependencies already installed
)

cd ..
echo.

echo [5/5] Starting Quran Muaalem...
echo.

if not exist "logs" mkdir logs

echo   Starting backend server...
start /B cmd /c "cd /d "%PROJECT_DIR%backend" && "%PROJECT_DIR%venv\Scripts\python.exe" main.py > "%PROJECT_DIR%logs\backend.log" 2>&1"

timeout /t 3 /nobreak >nul

echo   Starting frontend server...
start /B cmd /c "cd /d "%PROJECT_DIR%frontend" && npm run dev > "%PROJECT_DIR%logs\frontend.log" 2>&1"

timeout /t 5 /nobreak >nul

echo.
echo ========================================================================
echo.
echo              Quran Muaalem is running!
echo.
echo ========================================================================
echo.
echo   Frontend:  http://localhost:3000
echo   Backend:   http://localhost:8000
echo.
echo   Opening browser...
echo.

timeout /t 2 /nobreak >nul
start http://localhost:3000

echo.
echo   Press any key to stop the servers and exit...
pause >nul

echo.
echo   Stopping servers...

taskkill /F /IM "python.exe" /FI "WINDOWTITLE eq *main.py*" >nul 2>&1
taskkill /F /IM "node.exe" /FI "WINDOWTITLE eq *npm*" >nul 2>&1

for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":3000" ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)

echo   [OK] Servers stopped. Goodbye!
echo.

endlocal
exit /b 0
