@echo off
rem ==========================================================================
rem  Sirio OCR - avvio con un doppio clic (Windows 10/11)
rem  Al primo avvio scarica automaticamente Python e tutte le dipendenze.
rem ==========================================================================
setlocal
chcp 65001 >nul
title Sirio OCR
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0launcher\avvia.ps1" %*
if errorlevel 1 (
    echo.
    echo    Premi un tasto per chiudere questa finestra.
    pause >nul
    exit /b 1
)
endlocal
