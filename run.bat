@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: ═══════════════════════════════════════════════════════════════════════════
::                    المعلم القرآني - QURAN MUAALEM
::                    Intelligent Quran Recitation Teacher
:: ═══════════════════════════════════════════════════════════════════════════

title Quran Muaalem - Starting...

:: Colors
set "GREEN=[92m"
set "YELLOW=[93m"
set "CYAN=[96m"
set "RED=[91m"
set "GOLD=[33m"
set "RESET=[0m"

cls
echo.
echo %GOLD%═══════════════════════════════════════════════════════════════════════════%RESET%
echo %GOLD%                                                                           %RESET%
echo %GOLD%                        المعلم القرآني                                    %RESET%
echo %GOLD%                      QURAN MUAALEM                                        %RESET%
echo %GOLD%                Intelligent Quran Teacher                                  %RESET%
echo %GOLD%                                                                           %RESET%
echo %GOLD%═══════════════════════════════════════════════════════════════════════════%RESET%
echo.

:: Get the directory where this script is located
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 1: Check Prerequisites
:: ─────────────────────────────────────────────────────────────────────────────
echo %CYAN%[1/5]%RESET% Checking prerequisites...

:: Check Python
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo %RED%[ERROR]%RESET% Python is not installed!
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

:: Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo   %GREEN%✓%RESET% Python %PYTHON_VERSION% found

:: Check Node.js
where node >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo %RED%[ERROR]%RESET% Node.js is not installed!
    echo Please install Node.js 18+ from https://nodejs.org
    pause
    exit /b 1
)

:: Get Node version
for /f "tokens=1" %%i in ('node --version 2^>^&1') do set NODE_VERSION=%%i
echo   %GREEN%✓%RESET% Node.js %NODE_VERSION% found

:: Check npm
where npm >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo %RED%[ERROR]%RESET% npm is not installed!
    pause
    exit /b 1
)
echo   %GREEN%✓%RESET% npm found
echo.

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 2: Setup Python Virtual Environment
:: ─────────────────────────────────────────────────────────────────────────────
echo %CYAN%[2/5]%RESET% Setting up Python environment...

if not exist "venv" (
    echo   Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo %RED%[ERROR]%RESET% Failed to create virtual environment
        pause
        exit /b 1
    )
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Upgrade pip
echo   Upgrading pip...
python -m pip install --upgrade pip -q

echo   %GREEN%✓%RESET% Virtual environment ready
echo.

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 3: Install Python Dependencies
:: ─────────────────────────────────────────────────────────────────────────────
echo %CYAN%[3/5]%RESET% Installing Python dependencies...

:: Check if dependencies are already installed
python -c "import torch; import fastapi; import quran_transcript" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo   Installing packages (this may take a few minutes)...

    :: Install main requirements
    pip install -r requirements.txt -q
    if %ERRORLEVEL% NEQ 0 (
        echo %YELLOW%[WARN]%RESET% Some packages may have failed, continuing...
    )

    :: Install backend requirements
    pip install fastapi uvicorn python-multipart websockets -q

    :: Install the project in editable mode
    pip install -e . -q 2>nul

    echo   %GREEN%✓%RESET% Python dependencies installed
) else (
    echo   %GREEN%✓%RESET% Python dependencies already installed
)
echo.

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 4: Install Frontend Dependencies
:: ─────────────────────────────────────────────────────────────────────────────
echo %CYAN%[4/5]%RESET% Installing frontend dependencies...

cd frontend

if not exist "node_modules" (
    echo   Installing npm packages...
    call npm install --silent
    if %ERRORLEVEL% NEQ 0 (
        echo %RED%[ERROR]%RESET% Failed to install npm packages
        cd ..
        pause
        exit /b 1
    )
    echo   %GREEN%✓%RESET% Frontend dependencies installed
) else (
    echo   %GREEN%✓%RESET% Frontend dependencies already installed
)

cd ..
echo.

:: ─────────────────────────────────────────────────────────────────────────────
:: STEP 5: Start the Application
:: ─────────────────────────────────────────────────────────────────────────────
echo %CYAN%[5/5]%RESET% Starting Quran Muaalem...
echo.

:: Kill any existing processes on our ports
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3000" ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)

:: Start Backend Server
echo   Starting backend server...
start /B cmd /c "cd /d "%PROJECT_DIR%backend" && call "%PROJECT_DIR%venv\Scripts\activate.bat" && python main.py > "%PROJECT_DIR%logs\backend.log" 2>&1"

:: Wait for backend to start
timeout /t 3 /nobreak >nul

:: Start Frontend Server
echo   Starting frontend server...
start /B cmd /c "cd /d "%PROJECT_DIR%frontend" && npm run dev > "%PROJECT_DIR%logs\frontend.log" 2>&1"

:: Wait for frontend to start
timeout /t 5 /nobreak >nul

echo.
echo %GREEN%═══════════════════════════════════════════════════════════════════════════%RESET%
echo %GREEN%                                                                           %RESET%
echo %GREEN%                    ✓ Quran Muaalem is running!                           %RESET%
echo %GREEN%                                                                           %RESET%
echo %GREEN%═══════════════════════════════════════════════════════════════════════════%RESET%
echo.
echo   %CYAN%Frontend:%RESET%  http://localhost:3000
echo   %CYAN%Backend:%RESET%   http://localhost:8000
echo.
echo   %YELLOW%Opening browser...%RESET%
echo.

:: Open the frontend in default browser
timeout /t 2 /nobreak >nul
start http://localhost:3000

echo.
echo   Press any key to stop the servers and exit...
pause >nul

:: Cleanup - Kill the servers
echo.
echo   Stopping servers...

for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3000" ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)

echo   %GREEN%✓%RESET% Servers stopped. Goodbye!
echo.

endlocal
exit /b 0
