@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title MNIST SNN Paper Demo - Local Online Backpropagation

rem ===================================================================
rem  MNIST 脉冲 SNN 论文演示 —— 双击即训练（无需联网，数据随包自带）
rem ===================================================================

rem ---- locate Python ----
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY ( where py >nul 2>nul && set "PY=py" )
if not defined PY (
  echo [ERROR] Python 3.9+ was not found on this machine.
  echo         Install it from https://www.python.org/downloads/ and
  echo         tick "Add python.exe to PATH" during setup, then run again.
  echo.
  pause
  exit /b 1
)

rem ---- ensure numpy / matplotlib ----
%PY% -c "import numpy, matplotlib" >nul 2>nul
if errorlevel 1 (
  echo [INFO] Installing dependencies ^(numpy, matplotlib^) ...
  %PY% -m pip install --quiet numpy matplotlib
  if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Run this manually:
    echo         %PY% -m pip install numpy matplotlib
    pause
    exit /b 1
  )
)

rem ---- run the demo ----
echo.
echo Starting MNIST SNN demo ...
%PY% mnist_demo_train.py %*
if errorlevel 1 (
  echo.
  echo [DEMO] Exited with an error ^(see messages above^).
  pause
)
endlocal