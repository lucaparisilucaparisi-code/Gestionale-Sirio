#!/usr/bin/env bash
# Sirio OCR - avvio per macOS e Linux.
#
# Al primo avvio scarica automaticamente uv (gestore Python di Astral), che a sua
# volta installa Python 3.12 e tutte le dipendenze nella cartella ".runtime" del
# programma. Nessun privilegio di amministratore richiesto.
#
# Variabili d'ambiente facoltative:
#   SIRIO_SENZA_OFFLINE=1   non installa il motore OCR offline (PyTorch e modello TrOCR, ~2 GB)
#   SIRIO_REINSTALLA=1      forza la reinstallazione delle dipendenze
# Opzioni: --solo-installa  installa/aggiorna senza aprire l'applicazione
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
export PYTHONUTF8=1

SOLO_INSTALLA=0
for arg in "$@"; do
    [ "$arg" = "--solo-installa" ] && SOLO_INSTALLA=1
done

if [ -t 1 ]; then
    C_TIT=$'\033[1;36m'; C_DIM=$'\033[2m'; C_ERR=$'\033[1;31m'; C_OFF=$'\033[0m'
else
    C_TIT=""; C_DIM=""; C_ERR=""; C_OFF=""
fi
passo() { printf '   » %s\n' "$1"; }
nota()  { printf '     %s%s%s\n' "$C_DIM" "$1" "$C_OFF"; }
errore() {
    printf '\n   %sX  %s%s\n' "$C_ERR" "$1" "$C_OFF" >&2
    printf '      Dettagli nel file: %s\n' "$LOG" >&2
    exit 1
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

impronta() {
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
    [ "${SIRIO_SENZA_OFFLINE:-0}" = "1" ] && testo+="-senza-offline"
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
    [ "${SIRIO_SENZA_OFFLINE:-0}" = "1" ] && req="requisiti-base.txt"
    passo "Installo le dipendenze (alcuni minuti al primo avvio)..."
    "$VENV/bin/python" -m pip install --disable-pip-version-check --upgrade pip >>"$LOG" 2>&1 || true
    "$VENV/bin/python" -m pip install --disable-pip-version-check -r "$ROOT/launcher/$req" \
        || errore "Installazione delle dipendenze non riuscita (vedi messaggi sopra)."
}

installa_dipendenze() {
    local uv
    if ! uv="$(trova_uv)"; then
        if installa_uv; then uv="$UV_DIR/uv"; else uv=""; nota "Download di uv non riuscito."; fi
    fi
    if [ -n "$uv" ]; then
        passo "Installo Python 3.12 e le dipendenze (alcuni minuti solo al primo avvio)..."
        local extra=(--extra offline)
        [ "${SIRIO_SENZA_OFFLINE:-0}" = "1" ] && extra=()
        if ! "$uv" sync --project "$ROOT" --python 3.12 --frozen ${extra[@]+"${extra[@]}"}; then
            nota "Nuovo tentativo con risoluzione aggiornata delle dipendenze..."
            "$uv" sync --project "$ROOT" --python 3.12 ${extra[@]+"${extra[@]}"} \
                || errore "Installazione delle dipendenze non riuscita (vedi messaggi sopra)."
        fi
    else
        installa_con_python
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
IMPRONTA="$(impronta)"
if [ "${SIRIO_REINSTALLA:-0}" = "1" ] || [ ! -x "$PY" ] || [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$IMPRONTA" ]; then
    if [ ! -x "$PY" ]; then
        passo "Primo avvio: preparo il programma. Serve la connessione a Internet."
        nota "Lo scaricamento avviene una sola volta (circa 2 GB con il motore offline, modello incluso)."
        echo
    else
        passo "Aggiornamento dei componenti..."
    fi
    installa_dipendenze
    printf '%s' "$IMPRONTA" > "$STAMP"
    scarica_modello
    echo
    passo "Installazione completata."
fi

[ "$SOLO_INSTALLA" = "1" ] && exit 0

passo "Avvio di Sirio OCR... (puoi ridurre a icona questa finestra)"
cd "$ROOT"
exec "$PY" -m sirio
