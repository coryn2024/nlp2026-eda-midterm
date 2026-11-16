@echo off
setlocal
set "B_PYTHON=E:\Anaconda_envs\envs\myenv1\python.exe"
if defined EDA_PYTHON set "B_PYTHON=%EDA_PYTHON%"
if not exist "%B_PYTHON%" (
    echo Python not found: "%B_PYTHON%"
    exit /b 1
)
"%B_PYTHON%" -B -X utf8 -u "%~dp0train_cnn.py" %*
exit /b %ERRORLEVEL%
