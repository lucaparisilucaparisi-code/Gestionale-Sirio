@echo off
rem ==========================================================================
rem  Sirio OCR - avvio con un doppio clic (Windows 10/11)
rem  Al primo avvio scarica automaticamente Python e tutte le dipendenze.
rem ==========================================================================
setlocal
chcp 65001 >nul
title Sirio OCR
cd /d "%~dp0"
rem Avviato dall'anteprima del file ZIP: Esplora file copia solo questo file.
if not exist "%~dp0launcher\avvia.ps1" (
    echo.
    echo    Sirio OCR non puo' essere avviato dall'interno del file ZIP.
    echo    Estrai prima tutti i file ^(clic destro sul file ZIP - Estrai tutto^),
    echo    poi fai doppio clic su "Avvia Sirio OCR.bat" nella cartella estratta.
    echo.
    echo    Premi un tasto per chiudere questa finestra.
    pause >nul
    exit /b 1
)
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0launcher\avvia.ps1" %*
rem Qualsiasi codice diverso da 0 (anche negativo, es. errori di avvio di PowerShell)
if not "%errorlevel%"=="0" (
    echo.
    echo    Premi un tasto per chiudere questa finestra.
    pause >nul
    exit /b 1
)
endlocal
