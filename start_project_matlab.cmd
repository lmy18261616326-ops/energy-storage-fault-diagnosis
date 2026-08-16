@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "PROJECT_TEMP=%PROJECT_ROOT%work\temp"
set "PROJECT_PARALLEL=%PROJECT_ROOT%work\parallel_jobs"

if not exist "%PROJECT_TEMP%" mkdir "%PROJECT_TEMP%"
if not exist "%PROJECT_PARALLEL%" mkdir "%PROJECT_PARALLEL%"

set "TEMP=%PROJECT_TEMP%"
set "TMP=%PROJECT_TEMP%"

where matlab >nul 2>&1
if errorlevel 1 (
    echo MATLAB was not found on PATH.
    echo Add the MATLAB bin directory to PATH and retry.
    pause
    exit /b 1
)

start "" matlab -sd "%PROJECT_ROOT%"
