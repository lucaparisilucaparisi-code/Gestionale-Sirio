# Sirio OCR - avvio per Windows.
#
# Al primo avvio scarica automaticamente uv (gestore Python di Astral), che a sua
# volta installa Python 3.12 e tutte le dipendenze nella cartella ".runtime" del
# programma. Nessun privilegio di amministratore richiesto. Agli avvii successivi
# controlla in pochi secondi che sia tutto aggiornato e apre l'applicazione.
#
# Variabili d'ambiente facoltative:
#   SIRIO_SENZA_OFFLINE=1   non installa il motore OCR offline (PyTorch e modello TrOCR, ~2 GB)
#   SIRIO_REINSTALLA=1      forza la reinstallazione delle dipendenze

param(
    [switch]$SoloInstalla   # installa/aggiorna senza aprire l'applicazione
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest molto più veloce

$Root    = Split-Path -Parent $PSScriptRoot
$Runtime = Join-Path $Root '.runtime'
$UvDir   = Join-Path $Runtime 'uv'
$Venv    = Join-Path $Runtime 'venv'
$Stamp   = Join-Path $Runtime 'installazione.ok'
$LogFile = Join-Path $Runtime 'avvio.log'

New-Item -ItemType Directory -Force -Path $Runtime | Out-Null
try { Start-Transcript -Path $LogFile -Force | Out-Null } catch { }

$env:UV_PYTHON_INSTALL_DIR  = Join-Path $Runtime 'python'
$env:UV_PROJECT_ENVIRONMENT = $Venv
$env:UV_CACHE_DIR           = Join-Path $Runtime 'cache'
$env:UV_PYTHON_PREFERENCE   = 'only-managed'
$env:UV_LINK_MODE           = 'copy'
$env:PYTHONUTF8             = '1'

function Write-Titolo {
    Clear-Host
    Write-Host ''
    Write-Host '   *  S I R I O   O C R' -ForegroundColor Cyan
    Write-Host '      Lettura automatica dei fogli firma e rendicontazione Excel' -ForegroundColor DarkGray
    Write-Host ''
}

function Write-Passo([string]$Testo) {
    Write-Host ('   » ' + $Testo) -ForegroundColor White
}

function Write-Nota([string]$Testo) {
    Write-Host ('     ' + $Testo) -ForegroundColor DarkGray
}

function Get-Impronta {
    # Impronta dei file che definiscono le dipendenze: se cambiano si reinstalla.
    $testo = ''
    foreach ($f in @('pyproject.toml', 'uv.lock')) {
        $p = Join-Path $Root $f
        if (Test-Path $p) { $testo += (Get-FileHash -Algorithm SHA256 -Path $p).Hash }
    }
    if ($env:SIRIO_SENZA_OFFLINE -eq '1') { $testo += '-senza-offline' }
    return $testo
}

function Find-Uv {
    $locale = Join-Path $UvDir 'uv.exe'
    if (Test-Path $locale) { return $locale }
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Install-Uv {
    Write-Passo 'Scarico il gestore dei componenti (uv)...'
    New-Item -ItemType Directory -Force -Path $UvDir | Out-Null
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    } catch { }

    # 1) Installer ufficiale di Astral, eseguito in un processo separato e
    #    limitato alla cartella del programma (non modifica il PATH di sistema).
    try {
        $env:UV_INSTALL_DIR = $UvDir
        $env:UV_NO_MODIFY_PATH = '1'
        $env:INSTALLER_NO_MODIFY_PATH = '1'
        & powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" | Out-Null
    } catch {
        Write-Nota ('Installer non disponibile (' + $_.Exception.Message + '), provo il download diretto...')
    } finally {
        Remove-Item Env:UV_INSTALL_DIR -ErrorAction SilentlyContinue
    }
    $uv = Join-Path $UvDir 'uv.exe'
    if (Test-Path $uv) { return $uv }

    # 2) Archivio dalla pagina delle release su GitHub.
    $arch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'aarch64' } else { 'x86_64' }
    $zip = Join-Path $Runtime 'uv.zip'
    $tmp = Join-Path $Runtime 'uv-estratto'
    Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/astral-sh/uv/releases/latest/download/uv-$arch-pc-windows-msvc.zip" -OutFile $zip
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    Get-ChildItem -Path $tmp -Recurse -Filter 'uv*.exe' | ForEach-Object { Copy-Item $_.FullName -Destination $UvDir -Force }
    Remove-Item -Recurse -Force $tmp, $zip -ErrorAction SilentlyContinue
    if (Test-Path $uv) { return $uv }
    return $null
}

function Install-ConPython {
    # Ripiego senza uv: Python di sistema (>= 3.10) + venv + pip.
    $candidati = @(
        @('py', '-3.12'), @('py', '-3.13'), @('py', '-3.11'), @('py', '-3.10'), @('python'), @('python3')
    )
    $python = $null
    foreach ($c in $candidati) {
        $exe = Get-Command $c[0] -ErrorAction SilentlyContinue
        if (-not $exe) { continue }
        $argsVer = @($c | Select-Object -Skip 1) + @('-c', 'import sys; print(sys.version_info >= (3, 10) and sys.version_info < (3, 14))')
        try {
            $ok = & $exe.Source @argsVer 2>$null
            if ($ok -match 'True') { $python = $c; break }
        } catch { }
    }
    if (-not $python) {
        throw 'Impossibile scaricare i componenti e nessun Python 3.10-3.13 trovato. Verifica la connessione a Internet (o installa Python da python.org) e riprova.'
    }
    Write-Passo 'Creo l''ambiente Python...'
    $pyArgs = @($python | Select-Object -Skip 1)
    & (Get-Command $python[0]).Source @pyArgs -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw 'Creazione dell''ambiente Python non riuscita.' }
    $pip = Join-Path $Venv 'Scripts\python.exe'
    $req = if ($env:SIRIO_SENZA_OFFLINE -eq '1') { 'requisiti-base.txt' } else { 'requisiti.txt' }
    Write-Passo 'Installo le dipendenze (alcuni minuti al primo avvio)...'
    & $pip -m pip install --disable-pip-version-check --upgrade pip | Out-Null
    & $pip -m pip install --disable-pip-version-check -r (Join-Path $PSScriptRoot $req)
    if ($LASTEXITCODE -ne 0) { throw 'Installazione delle dipendenze non riuscita (vedi messaggi sopra).' }
}

function Install-Dipendenze {
    $uv = Find-Uv
    if (-not $uv) {
        try { $uv = Install-Uv } catch { Write-Nota ('Download di uv non riuscito: ' + $_.Exception.Message) }
    }
    if ($uv) {
        Write-Passo 'Installo Python 3.12 e le dipendenze (alcuni minuti solo al primo avvio)...'
        $argsSync = @('sync', '--project', $Root, '--python', '3.12', '--frozen')
        if ($env:SIRIO_SENZA_OFFLINE -ne '1') { $argsSync += @('--extra', 'offline') }
        & $uv @argsSync
        if ($LASTEXITCODE -ne 0) {
            Write-Nota 'Nuovo tentativo con risoluzione aggiornata delle dipendenze...'
            $argsSync = $argsSync | Where-Object { $_ -ne '--frozen' }
            & $uv @argsSync
        }
        if ($LASTEXITCODE -ne 0) { throw 'Installazione delle dipendenze non riuscita (vedi messaggi sopra).' }
    } else {
        Install-ConPython
    }
}

function New-CollegamentoDesktop {
    try {
        $desktop = [Environment]::GetFolderPath('Desktop')
        if (-not $desktop) { return }
        $lnk = Join-Path $desktop 'Sirio OCR.lnk'
        if (Test-Path $lnk) { return }
        $shell = New-Object -ComObject WScript.Shell
        $sc = $shell.CreateShortcut($lnk)
        $sc.TargetPath = Join-Path $Root 'Avvia Sirio OCR.bat'
        $sc.WorkingDirectory = $Root
        $sc.IconLocation = (Join-Path $PSScriptRoot 'sirio.ico') + ',0'
        $sc.WindowStyle = 7   # finestra di avvio ridotta a icona
        $sc.Description = 'Sirio OCR - fogli firma e rendicontazione Excel'
        $sc.Save()
        Write-Nota 'Creato il collegamento "Sirio OCR" sul desktop.'
    } catch {
        Write-Nota 'Collegamento sul desktop non creato (non indispensabile).'
    }
}

function Save-ModelloOffline {
    # Modello del motore offline scaricato subito (con l'avanzamento): il primo foglio
    # non deve attenderlo. Un errore qui non impedisce l'avvio: il motore lo scarica
    # comunque al primo utilizzo.
    if ($env:SIRIO_SENZA_OFFLINE -eq '1') { return }
    $python = Join-Path $Venv 'Scripts\python.exe'
    if (-not (Test-Path $python)) { return }
    Write-Host ''
    Write-Passo 'Scarico il modello di riconoscimento della scrittura (circa 1,3 GB, solo la prima volta)...'
    $ErrorActionPreference = 'Continue'   # i messaggi su stderr non devono interrompere lo script
    $esito = 1
    Push-Location $Root
    try {
        & $python -m sirio --scarica-modello
        $esito = $LASTEXITCODE
    } catch {
        $esito = 1
    } finally {
        Pop-Location
    }
    if ($esito -ne 0) {
        Write-Nota 'Download del modello non completato: verrà scaricato automaticamente al primo utilizzo.'
    }
}

try {
    Write-Titolo
    $pythonw = Join-Path $Venv 'Scripts\pythonw.exe'
    $impronta = Get-Impronta
    $installato = (Test-Path $pythonw) -and (Test-Path $Stamp) -and ((Get-Content $Stamp -Raw).Trim() -eq $impronta)

    if ($env:SIRIO_REINSTALLA -eq '1' -or -not $installato) {
        $primo = -not (Test-Path $pythonw)
        if ($primo) {
            Write-Passo 'Primo avvio: preparo il programma. Serve la connessione a Internet.'
            Write-Nota 'Lo scaricamento avviene una sola volta (circa 2 GB con il motore offline, modello incluso).'
            Write-Host ''
        } else {
            Write-Passo 'Aggiornamento dei componenti...'
        }
        Install-Dipendenze
        Set-Content -Path $Stamp -Value $impronta -Encoding ASCII
        New-CollegamentoDesktop
        Save-ModelloOffline
        Write-Host ''
        Write-Passo 'Installazione completata.'
    }

    if ($SoloInstalla) { exit 0 }

    Write-Passo 'Avvio di Sirio OCR...'
    Start-Process -FilePath $pythonw -ArgumentList @('-m', 'sirio') -WorkingDirectory $Root
    Start-Sleep -Milliseconds 600
    exit 0
} catch {
    Write-Host ''
    Write-Host ('   X  ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host ('      Dettagli nel file: ' + $LogFile) -ForegroundColor DarkGray
    exit 1
} finally {
    try { Stop-Transcript | Out-Null } catch { }
}
