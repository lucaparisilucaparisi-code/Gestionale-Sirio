# Sirio OCR - avvio per Windows.
#
# Al primo avvio scarica automaticamente uv (gestore Python di Astral), che a sua
# volta installa Python 3.12 e tutte le dipendenze in %LOCALAPPDATA%\SirioOCR\runtime:
# una cartella locale, mai sincronizzata con OneDrive e comune a tutte le copie
# estratte del programma. Nessun privilegio di amministratore richiesto. Agli avvii
# successivi controlla in pochi secondi che sia tutto aggiornato e apre l'applicazione.
#
# Variabili d'ambiente facoltative:
#   SIRIO_SENZA_OFFLINE=1   non installa il motore OCR offline (PyTorch e modello TrOCR, ~1,5 GB)
#   SIRIO_REINSTALLA=1      forza la reinstallazione delle dipendenze

param(
    [switch]$SoloInstalla   # installa/aggiorna senza aprire l'applicazione
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest molto più veloce

$Root    = Split-Path -Parent $PSScriptRoot
# Componenti fuori dalla cartella del programma, che spesso è in Documenti: con il backup
# di OneDrive verrebbero sincronizzati decine di migliaia di file e alcuni GB.
$Runtime = if ($env:LOCALAPPDATA) { Join-Path $env:LOCALAPPDATA 'SirioOCR\runtime' } else { Join-Path $Root '.runtime' }
$RuntimePrecedente = Join-Path $Root '.runtime'   # posizione usata dalle versioni precedenti
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
# Certificati di Windows invece di quelli inclusi in uv: necessari con gli antivirus che
# analizzano le connessioni HTTPS e con i proxy aziendali.
$env:UV_SYSTEM_CERTS = '1'
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

function Write-Avviso([string]$Testo) {
    Write-Host ('     ' + $Testo) -ForegroundColor Yellow
}

function Wait-Tasto {
    # Lascia leggere gli avvisi prima che la finestra si chiuda.
    if ($SoloInstalla) { return }
    Write-Host ''
    Write-Host '     Premi un tasto per continuare...' -ForegroundColor DarkGray
    try { $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown') } catch { Start-Sleep -Seconds 15 }
}

function Get-Impronta {
    # Impronta dei file che definiscono le dipendenze: se cambiano si reinstalla.
    $testo = ''
    foreach ($f in @('pyproject.toml', 'uv.lock')) {
        $p = Join-Path $Root $f
        if (Test-Path $p) { $testo += (Get-FileHash -Algorithm SHA256 -Path $p).Hash }
    }
    return $testo
}

function Test-InUso([string]$Cartella) {
    # True se Python è in esecuzione da questa cartella (Sirio OCR aperto).
    $prefisso = $Cartella.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
    $processi = @(Get-Process -Name 'python', 'pythonw' -ErrorAction SilentlyContinue | Where-Object {
        $_.Path -and $_.Path.StartsWith($prefisso, [StringComparison]::OrdinalIgnoreCase) })
    return ($processi.Count -gt 0)
}

function Wait-SirioChiuso {
    # I file in uso non si possono sostituire: per aggiornare i componenti Sirio OCR non
    # deve essere in esecuzione (attesa massima 5 minuti).
    if (-not (Test-InUso $Runtime)) { return }
    Write-Passo 'Sirio OCR è aperto: chiudi la sua finestra per continuare l''aggiornamento.'
    Write-Nota 'L''aggiornamento riprende da solo poco dopo la chiusura (al massimo un paio di minuti).'
    $limite = (Get-Date).AddMinutes(5)
    while ((Test-InUso $Runtime) -and (Get-Date) -lt $limite) { Start-Sleep -Seconds 2 }
}

function Invoke-Registrato([string]$Exe, [string[]]$Argomenti) {
    # Esegue un programma esterno mostrandone l'output e copiandolo nel registro: la
    # trascrizione di Windows PowerShell non registra ciò che i programmi esterni scrivono
    # direttamente sulla console. Le righe su stderr non sono errori dello script.
    # Restituisce il codice d'uscita.
    $ErrorActionPreference = 'Continue'
    $global:LASTEXITCODE = 1
    & $Exe @Argomenti 2>&1 | ForEach-Object { Write-Host "$_" }
    return $global:LASTEXITCODE
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
    #    limitato alla cartella dei componenti (non modifica il PATH di sistema).
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

function Set-CertificatiSistema([string]$Uv) {
    # Le versioni di uv che non conoscono UV_SYSTEM_CERTS usano UV_NATIVE_TLS (nelle versioni
    # recenti è deprecata e produce un avviso: si imposta solo se serve).
    $ErrorActionPreference = 'Continue'
    $aiuto = (& $Uv sync --help 2>&1 | Out-String)
    if ($aiuto -notmatch 'system-certs') { $env:UV_NATIVE_TLS = '1' }
}

# Esito del motore offline restituito dalle funzioni d'installazione: 'ok', 'non-richiesto'
# (SIRIO_SENZA_OFFLINE=1), 'non-supportato' (nessun pacchetto per questo computer) oppure
# 'non-riuscito' (errore temporaneo: si riprova al prossimo avvio). Senza i componenti di
# base si genera un errore.

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
    & (Get-Command $python[0]).Source @pyArgs -m venv $Venv | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Creazione dell''ambiente Python non riuscita.' }
    $pip = Join-Path $Venv 'Scripts\python.exe'
    $offline = $env:SIRIO_SENZA_OFFLINE -ne '1'
    $req = if ($offline) { 'requisiti.txt' } else { 'requisiti-base.txt' }
    Write-Passo 'Installo le dipendenze (alcuni minuti al primo avvio)...'
    & $pip -m pip install --disable-pip-version-check --upgrade pip | Out-Null
    $pipArgs = @('-m', 'pip', 'install', '--disable-pip-version-check', '-r')
    if ((Invoke-Registrato $pip ($pipArgs + @(Join-Path $PSScriptRoot $req))) -eq 0) {
        if ($offline) { return 'ok' } else { return 'non-richiesto' }
    }
    if (-not $offline) { throw 'Installazione delle dipendenze non riuscita (vedi messaggi sopra).' }
    Write-Nota 'Installo i soli componenti di base, senza il motore offline...'
    if ((Invoke-Registrato $pip ($pipArgs + @(Join-Path $PSScriptRoot 'requisiti-base.txt'))) -ne 0) {
        throw 'Installazione delle dipendenze non riuscita (vedi messaggi sopra).'
    }
    return 'non-riuscito'
}

function Sync-Dipendenze([string]$Uv, [string[]]$Opzioni = @()) {
    # uv sync con il lockfile; se non riesce, nuovo tentativo con risoluzione aggiornata.
    $base = @('sync', '--project', $Root, '--python', '3.12')
    if ((Invoke-Registrato $Uv ($base + @('--frozen') + $Opzioni)) -eq 0) { return 0 }
    Write-Nota 'Nuovo tentativo con risoluzione aggiornata delle dipendenze...'
    return (Invoke-Registrato $Uv ($base + $Opzioni))
}

function Install-Dipendenze {
    $uv = Find-Uv
    if (-not $uv) {
        try { $uv = Install-Uv } catch { Write-Nota ('Download di uv non riuscito: ' + $_.Exception.Message) }
    }
    if (-not $uv) { return (Install-ConPython) }
    Set-CertificatiSistema $uv
    Write-Passo 'Installo Python 3.12 e le dipendenze (alcuni minuti solo al primo avvio)...'
    if ($env:SIRIO_SENZA_OFFLINE -eq '1') {
        if ((Sync-Dipendenze $uv @()) -ne 0) { throw 'Installazione delle dipendenze non riuscita (vedi messaggi sopra).' }
        return 'non-richiesto'
    }
    if ((Sync-Dipendenze $uv @('--extra', 'offline')) -eq 0) { return 'ok' }
    # Motore offline non installabile: si installano i componenti di base (motore Claude),
    # senza rimuovere quelli del motore offline eventualmente già presenti.
    Write-Nota 'Installo i soli componenti di base, senza il motore offline...'
    if ((Invoke-Registrato $uv @('sync', '--project', $Root, '--python', '3.12', '--frozen', '--inexact')) -ne 0) {
        throw 'Installazione delle dipendenze non riuscita (vedi messaggi sopra).'
    }
    # Nessun pacchetto del motore offline per questo computer oppure errore temporaneo
    # (rete, disco)? Verifica senza scaricare nulla.
    $ErrorActionPreference = 'Continue'
    & $uv sync --project $Root --python 3.12 --frozen --extra offline --dry-run --offline 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { return 'non-supportato' }
    return 'non-riuscito'
}

function Test-RuntimeVisualC {
    # PyTorch richiede il "Microsoft Visual C++ Redistributable" (msvcp140.dll), che manca su
    # alcune installazioni di Windows e non è incluso né in Python né in PyTorch.
    if (Test-Path (Join-Path $env:SystemRoot 'System32\msvcp140.dll')) { return $true }
    $python = Join-Path $Venv 'Scripts\python.exe'
    if (-not (Test-Path $python)) { return $true }
    $ErrorActionPreference = 'Continue'
    & $python -c 'import torch' 2>&1 | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function New-CollegamentoDesktop([switch]$SoloSeEsiste) {
    # Crea il collegamento sul desktop, oppure lo corregge se apre un'altra copia del
    # programma (es. una versione precedente estratta altrove) o una finestra ridotta a
    # icona. Con -SoloSeEsiste non ricrea un collegamento eliminato dall'utente.
    try {
        $desktop = [Environment]::GetFolderPath('Desktop')
        if (-not $desktop) { return }
        $lnk = Join-Path $desktop 'Sirio OCR.lnk'
        $esiste = Test-Path -LiteralPath $lnk
        if ($SoloSeEsiste -and -not $esiste) { return }
        $bat = Join-Path $Root 'Avvia Sirio OCR.bat'
        $shell = New-Object -ComObject WScript.Shell
        $sc = $shell.CreateShortcut($lnk)   # legge il collegamento esistente
        if ($esiste -and $sc.TargetPath -eq $bat -and $sc.WindowStyle -eq 1) { return }
        $sc.TargetPath = $bat
        $sc.WorkingDirectory = $Root
        $sc.IconLocation = (Join-Path $PSScriptRoot 'sirio.ico') + ',0'
        $sc.WindowStyle = 1   # finestra normale: avanzamento ed eventuali errori restano visibili
        $sc.Description = 'Sirio OCR - fogli firma e rendicontazione Excel'
        $sc.Save()
        if ($esiste) { Write-Nota 'Aggiornato il collegamento "Sirio OCR" sul desktop.' }
        else { Write-Nota 'Creato il collegamento "Sirio OCR" sul desktop.' }
    } catch {
        if (-not $SoloSeEsiste) { Write-Nota 'Collegamento sul desktop non creato (non indispensabile).' }
    }
}

function Remove-RuntimePrecedente {
    # Le versioni precedenti installavano i componenti nella cartella ".runtime" del
    # programma: non serve più e occupa alcuni GB (spesso sincronizzati con OneDrive).
    if ($RuntimePrecedente -eq $Runtime -or -not (Test-Path -LiteralPath $RuntimePrecedente)) { return }
    if (Test-InUso $RuntimePrecedente) { return }   # versione precedente ancora aperta
    try {
        Remove-Item -LiteralPath $RuntimePrecedente -Recurse -Force -ErrorAction Stop
        Write-Nota 'Eliminati i componenti della versione precedente (cartella ".runtime" del programma).'
    } catch {
        Write-Nota ('La cartella "' + $RuntimePrecedente + '" non serve più e si può eliminare.')
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
    $hash = Get-Impronta
    $impronta = if ($env:SIRIO_SENZA_OFFLINE -eq '1') { $hash + '-senza-offline' } else { $hash }
    # Installazione riuscita senza motore offline perché non disponibile su questo computer:
    # non si ritenta a ogni avvio, solo quando cambiano le dipendenze.
    $improntaBase = $hash + '-offline-non-supportato'
    $registrata = if (Test-Path $Stamp) { "$(Get-Content $Stamp -Raw)".Trim() } else { '' }
    $installato = (Test-Path $pythonw) -and ($registrata -eq $impronta -or $registrata -eq $improntaBase)

    if ($env:SIRIO_REINSTALLA -eq '1' -or -not $installato) {
        $primo = -not (Test-Path $pythonw)
        if ($primo) {
            Write-Passo 'Primo avvio: preparo il programma. Serve la connessione a Internet.'
            Write-Nota 'Lo scaricamento avviene una sola volta (circa 1,7 GB con il motore offline, modello incluso);'
            Write-Nota 'servono circa 4 GB di spazio libero.'
            Write-Host ''
        } else {
            Write-Passo 'Aggiornamento dei componenti...'
            Wait-SirioChiuso
        }
        $esito = Install-Dipendenze
        $avvisi = $false
        if ($esito -eq 'non-supportato') {
            Set-Content -Path $Stamp -Value $improntaBase -Encoding ASCII
            Write-Host ''
            Write-Avviso 'Il motore offline non è disponibile su questo computer (richiede Windows a 64 bit):'
            Write-Avviso 'si può usare il motore Claude, da scegliere in Impostazioni.'
            $avvisi = $true
        } elseif ($esito -eq 'non-riuscito') {
            Remove-Item -LiteralPath $Stamp -Force -ErrorAction SilentlyContinue
            Write-Host ''
            Write-Avviso 'Il motore offline non è stato installato (vedi messaggi sopra): l''installazione verrà'
            Write-Avviso 'ritentata al prossimo avvio. Nel frattempo si può usare il motore Claude (Impostazioni).'
            $avvisi = $true
        } else {
            Set-Content -Path $Stamp -Value $impronta -Encoding ASCII
        }
        New-CollegamentoDesktop
        if ($esito -eq 'ok') {
            Save-ModelloOffline
            if (-not (Test-RuntimeVisualC)) {
                $vcArch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'arm64' } else { 'x64' }
                Write-Host ''
                Write-Avviso 'Attenzione: il motore offline richiede il componente di Windows "Microsoft Visual C++'
                Write-Avviso 'Redistributable", che non è installato su questo computer. Scaricalo e installalo da'
                Write-Avviso ('   https://aka.ms/vs/17/release/vc_redist.' + $vcArch + '.exe')
                Write-Avviso '(se serve, con l''aiuto dell''assistenza informatica), poi riavvia Sirio OCR.'
                Write-Avviso 'Nel frattempo si può usare il motore Claude (Impostazioni).'
                $avvisi = $true
            }
        }
        Write-Host ''
        Write-Passo 'Installazione completata.'
        if ($avvisi) { Wait-Tasto }
    } else {
        New-CollegamentoDesktop -SoloSeEsiste
    }
    Remove-RuntimePrecedente

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
