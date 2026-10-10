#!/usr/bin/env bash
# Sirio OCR - avvio per macOS e Linux.
#
# Al primo avvio scarica automaticamente uv (gestore Python di Astral), che a sua
# volta installa Python 3.12 e tutte le dipendenze nella cartella ".runtime" del
# programma. Nessun privilegio di amministratore richiesto.
#
# Variabili d'ambiente facoltative:
#   SIRIO_SENZA_OFFLINE=1   non installa il motore OCR offline (PyTorch e modello TrOCR, ~1,5 GB)
#   SIRIO_REINSTALLA=1      forza la reinstallazione delle dipendenze
# Opzioni: --solo-installa  installa/aggiorna senza aprire l'applicazione
#
# Il motore offline richiede Linux (x86_64/aarch64) o un Mac con processore Apple e
# macOS 14 o successivo: altrove (es. Mac con processore Intel) si installano solo i
# componenti di base, sufficienti per il motore Claude.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="$ROOT/.runtime"
UV_DIR="$RUNTIME/uv"
VENV="$RUNTIME/venv"
STAMP="$RUNTIME/installazione.ok"
LOG="$RUNTIME/avvio.log"
mkdir -p "$RUNTIME"

export UV_PYTHON_INSTALL_DIR="$RUNTIME/python"
export UV_PROJECT_ENVIRONMENT="$VENV"
export UV_CACHE_DIR="$RUNTIME/cache"
export UV_PYTHON_PREFERENCE="only-managed"
# Certificati del sistema operativo invece di quelli inclusi in uv: necessari con gli
# antivirus che analizzano le connessioni HTTPS e con i proxy aziendali.
export UV_SYSTEM_CERTS=1
export PYTHONUTF8=1

SOLO_INSTALLA=0
for arg in "$@"; do
    [ "$arg" = "--solo-installa" ] && SOLO_INSTALLA=1
done

if [ -t 1 ]; then
    C_TIT=$'\033[1;36m'; C_DIM=$'\033[2m'; C_AVV=$'\033[1;33m'; C_ERR=$'\033[1;31m'; C_OFF=$'\033[0m'
else
    C_TIT=""; C_DIM=""; C_AVV=""; C_ERR=""; C_OFF=""
fi

# Registro dell'avvio (riscritto a ogni avvio): vi si copia l'output di uv e pip.
if ! { printf '=== Sirio OCR - avvio del %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" >"$LOG"; } 2>/dev/null; then
    LOG=/dev/null
fi

passo()  { printf '   » %s\n' "$1"; }
nota()   { printf '     %s%s%s\n' "$C_DIM" "$1" "$C_OFF"; }
avviso() { printf '     %s%s%s\n' "$C_AVV" "$1" "$C_OFF"; }
errore() {
    printf 'ERRORE: %s\n' "$1" >>"$LOG" 2>/dev/null || true
    printf '\n   %sX  %s%s\n' "$C_ERR" "$1" "$C_OFF" >&2
    [ "$LOG" = /dev/null ] || printf '      Dettagli nel file: %s\n' "$LOG" >&2
    exit 1
}

esegui() {  # esegue un comando mostrandone l'output e copiandolo nel registro
    local rc=0
    printf '\n$ %s\n' "$*" >>"$LOG" 2>/dev/null || true
    "$@" 2>&1 | tee -a "$LOG" || rc=$?
    return "$rc"
}

printf '\n   %s*  S I R I O   O C R%s\n' "$C_TIT" "$C_OFF"
printf '      %sLettura automatica dei fogli firma e rendicontazione Excel%s\n\n' "$C_DIM" "$C_OFF"

scarica() {  # scarica <url> <destinazione>
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 3 "$1" -o "$2"
    elif command -v wget >/dev/null 2>&1; then
        wget -q "$1" -O "$2"
    else
        return 1
    fi
}

impronta() {  # impronta dei file che definiscono le dipendenze: se cambiano si reinstalla
    local testo=""
    for f in pyproject.toml uv.lock; do
        if [ -f "$ROOT/$f" ]; then
            if command -v shasum >/dev/null 2>&1; then
                testo+="$(shasum -a 256 "$ROOT/$f" | cut -d' ' -f1)"
            else
                testo+="$(sha256sum "$ROOT/$f" | cut -d' ' -f1)"
            fi
        fi
    done
    printf '%s' "$testo"
}

trova_uv() {
    if [ -x "$UV_DIR/uv" ]; then echo "$UV_DIR/uv"; return 0; fi
    if command -v uv >/dev/null 2>&1; then command -v uv; return 0; fi
    return 1
}

installa_uv() {
    passo "Scarico il gestore dei componenti (uv)..."
    mkdir -p "$UV_DIR"
    local script="$RUNTIME/uv-install.sh"
    if scarica "https://astral.sh/uv/install.sh" "$script"; then
        env UV_INSTALL_DIR="$UV_DIR" UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1 sh "$script" >>"$LOG" 2>&1 || true
        rm -f "$script"
    fi
    if [ -x "$UV_DIR/uv" ]; then return 0; fi

    # Ripiego: archivio dalla pagina delle release su GitHub.
    local os arch
    case "$(uname -s)" in
        Darwin) os="apple-darwin" ;;
        Linux)  os="unknown-linux-gnu" ;;
        *) return 1 ;;
    esac
    case "$(uname -m)" in
        arm64|aarch64) arch="aarch64" ;;
        x86_64|amd64)  arch="x86_64" ;;
        *) return 1 ;;
    esac
    local tgz="$RUNTIME/uv.tar.gz"
    scarica "https://github.com/astral-sh/uv/releases/latest/download/uv-$arch-$os.tar.gz" "$tgz" || return 1
    tar -xzf "$tgz" -C "$UV_DIR" --strip-components=1 || return 1
    rm -f "$tgz"
    [ -x "$UV_DIR/uv" ]
}

certificati_di_sistema() {  # certificati_di_sistema <uv>
    # Le versioni di uv che non conoscono UV_SYSTEM_CERTS usano UV_NATIVE_TLS (nelle
    # versioni recenti e' deprecata e produce un avviso: si imposta solo se serve).
    local aiuto
    aiuto="$("$1" sync --help 2>/dev/null || true)"
    case "$aiuto" in
        *--system-certs*) ;;
        *) export UV_NATIVE_TLS=1 ;;
    esac
}

# Esito del motore offline dopo l'installazione: installato, non-richiesto
# (SIRIO_SENZA_OFFLINE=1), non-supportato (nessun pacchetto per questo computer) o
# non-riuscito (errore temporaneo: si riprova al prossimo avvio).
OFFLINE_ESITO="installato"

installa_con_python() {
    # Ripiego senza uv: Python di sistema (>= 3.10, < 3.14) + venv + pip.
    local py=""
    for c in python3.12 python3.13 python3.11 python3.10 python3; do
        if command -v "$c" >/dev/null 2>&1 && \
           "$c" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)' 2>/dev/null; then
            py="$c"; break
        fi
    done
    [ -n "$py" ] || errore "Impossibile scaricare i componenti e nessun Python 3.10-3.13 trovato. Verifica la connessione a Internet e riprova."
    passo "Creo l'ambiente Python..."
    "$py" -m venv "$VENV" || errore "Creazione dell'ambiente Python non riuscita."
    local req="requisiti.txt"
    if [ "${SIRIO_SENZA_OFFLINE:-0}" = "1" ]; then req="requisiti-base.txt"; OFFLINE_ESITO="non-richiesto"; fi
    passo "Installo le dipendenze (alcuni minuti al primo avvio)..."
    "$VENV/bin/python" -m pip install --disable-pip-version-check --upgrade pip >>"$LOG" 2>&1 || true
    esegui "$VENV/bin/python" -m pip install --disable-pip-version-check -r "$ROOT/launcher/$req" && return 0
    [ "$req" = "requisiti.txt" ] || errore "Installazione delle dipendenze non riuscita (vedi messaggi sopra)."
    nota "Installo i soli componenti di base, senza il motore offline..."
    esegui "$VENV/bin/python" -m pip install --disable-pip-version-check -r "$ROOT/launcher/requisiti-base.txt" \
        || errore "Installazione delle dipendenze non riuscita (vedi messaggi sopra)."
    OFFLINE_ESITO="non-riuscito"
}

sincronizza() {  # sincronizza <uv> [opzioni]: uv sync, con un nuovo tentativo senza --frozen
    local uv="$1"; shift
    esegui "$uv" sync --project "$ROOT" --python 3.12 --frozen "$@" && return 0
    nota "Nuovo tentativo con risoluzione aggiornata delle dipendenze..."
    esegui "$uv" sync --project "$ROOT" --python 3.12 "$@"
}

installa_dipendenze() {
    local uv
    if ! uv="$(trova_uv)"; then
        if installa_uv; then uv="$UV_DIR/uv"; else uv=""; nota "Download di uv non riuscito."; fi
    fi
    if [ -z "$uv" ]; then
        installa_con_python
        return 0
    fi
    certificati_di_sistema "$uv"
    passo "Installo Python 3.12 e le dipendenze (alcuni minuti solo al primo avvio)..."
    if [ "${SIRIO_SENZA_OFFLINE:-0}" = "1" ]; then
        OFFLINE_ESITO="non-richiesto"
        sincronizza "$uv" || errore "Installazione delle dipendenze non riuscita (vedi messaggi sopra)."
        return 0
    fi
    sincronizza "$uv" --extra offline && return 0
    # Motore offline non installabile: si installano i componenti di base (motore Claude),
    # senza rimuovere quelli del motore offline eventualmente gia' presenti.
    nota "Installo i soli componenti di base, senza il motore offline..."
    esegui "$uv" sync --project "$ROOT" --python 3.12 --frozen --inexact \
        || errore "Installazione delle dipendenze non riuscita (vedi messaggi sopra)."
    # Nessun pacchetto del motore offline per questo computer (es. PyTorch per i Mac con
    # processore Intel) oppure errore temporaneo (rete, disco)? Verifica senza scaricare nulla.
    if "$uv" sync --project "$ROOT" --python 3.12 --frozen --extra offline --dry-run --offline >>"$LOG" 2>&1; then
        OFFLINE_ESITO="non-riuscito"
    else
        OFFLINE_ESITO="non-supportato"
    fi
}

scarica_modello() {
    # Modello del motore offline scaricato subito (con l'avanzamento): il primo foglio
    # non deve attenderlo. Un errore qui non impedisce l'avvio: il motore lo scarica
    # comunque al primo utilizzo.
    [ "${SIRIO_SENZA_OFFLINE:-0}" = "1" ] && return 0
    echo
    passo "Scarico il modello di riconoscimento della scrittura (circa 1,3 GB, solo la prima volta)..."
    if ! (cd "$ROOT" && "$PY" -m sirio --scarica-modello); then
        nota "Download del modello non completato: verrà scaricato automaticamente al primo utilizzo."
    fi
}

PY="$VENV/bin/python"
IMPRONTA_FILE="$(impronta)"
IMPRONTA="$IMPRONTA_FILE"
[ "${SIRIO_SENZA_OFFLINE:-0}" = "1" ] && IMPRONTA+="-senza-offline"
# Installazione riuscita senza motore offline perche' non disponibile su questo computer:
# non si ritenta a ogni avvio, solo quando cambiano le dipendenze.
IMPRONTA_BASE="$IMPRONTA_FILE-offline-non-supportato"
REGISTRATA="$(cat "$STAMP" 2>/dev/null || true)"
if [ "${SIRIO_REINSTALLA:-0}" = "1" ] || [ ! -x "$PY" ] || \
   { [ "$REGISTRATA" != "$IMPRONTA" ] && [ "$REGISTRATA" != "$IMPRONTA_BASE" ]; }; then
    if [ ! -x "$PY" ]; then
        passo "Primo avvio: preparo il programma. Serve la connessione a Internet."
        nota "Lo scaricamento avviene una sola volta (circa 1,7 GB con il motore offline, modello incluso);"
        nota "servono circa 4 GB di spazio libero."
        echo
    else
        passo "Aggiornamento dei componenti..."
    fi
    installa_dipendenze
    case "$OFFLINE_ESITO" in
        non-supportato)
            printf '%s' "$IMPRONTA_BASE" > "$STAMP"
            echo
            avviso "Il motore offline non è disponibile su questo computer (richiede un Mac con processore Apple"
            avviso "e macOS 14 o successivo, oppure Linux a 64 bit): si può usare il motore Claude (Impostazioni)."
            ;;
        non-riuscito)
            rm -f "$STAMP"
            echo
            avviso "Il motore offline non è stato installato (vedi messaggi sopra): l'installazione verrà"
            avviso "ritentata al prossimo avvio. Nel frattempo si può usare il motore Claude (Impostazioni)."
            ;;
        *)
            printf '%s' "$IMPRONTA" > "$STAMP"
            scarica_modello
            ;;
    esac
    echo
    passo "Installazione completata."
fi

[ "$SOLO_INSTALLA" = "1" ] && exit 0

passo "Avvio di Sirio OCR... (puoi ridurre a icona questa finestra)"
cd "$ROOT"
exec "$PY" -m sirio
