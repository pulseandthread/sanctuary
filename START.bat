@echo off
title Sanctuary
color 0A

echo.
echo  Starting Sanctuary...
echo.

:: Check .env exists
if not exist ".env" (
    echo  ERROR: No .env file found!
    echo  Please run setup.bat first.
    pause
    exit /b 1
)

:: Check venv exists
if not exist "venv\Scripts\activate" (
    echo  ERROR: Virtual environment not found!
    echo  Please run setup.bat first.
    pause
    exit /b 1
)

:: Activate and run
call venv\Scripts\activate
echo  Opening Sanctuary in your browser...
start http://localhost:5000
python app.py
pause
