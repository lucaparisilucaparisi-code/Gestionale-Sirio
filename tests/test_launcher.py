"""Test dei programmi di avvio (launcher/, file "Avvia Sirio OCR") e del ciclo di vita del server."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from sirio import app as sirio_app

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launcher"


# ==========================================================================
# Ciclo di vita: piu' finestre, finestre ridotte a icona
# ==========================================================================

class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _simulate(lc: sirio_app.Lifecycle, clock: Clock, events: dict[float, list[str]], until: float,
              step: float = 0.5) -> tuple[float, str] | None:
    """Fa avanzare l'orologio come il ciclo di main() (un controllo ogni 0,5 s) e applica gli
    eventi; restituisce (istante, motivo) del primo arresto, oppure None."""
    pending = sorted(events.items())
    while clock.t <= until:
        while pending and pending[0][0] <= clock.t:
            for name in pending.pop(0)[1]:
                getattr(lc, name)()
        reason = lc.check()
        if reason:
            return clock.t, reason
        clock.t += step
    return None


@pytest.mark.parametrize("interval", [20.0, sirio_app.HEARTBEAT_MAX_GAP])
def test_lifecycle_second_window_keeps_server_alive_after_bye(interval: float) -> None:
    # Due finestre sullo stesso server (secondo doppio clic sull'icona): la prima viene
    # chiusa (bye) mentre la seconda, eventualmente ridotta a icona, invia heartbeat ogni
    # `interval` secondi. In nessuna fase il server deve arrestarsi.
    for phase in range(int(interval)):
        clock = Clock()
        lc = sirio_app.Lifecycle(auto_shutdown=True, clock=clock)
        events: dict[float, list[str]] = {}
        t = float(phase)
        while t <= 600:
            events.setdefault(t, []).append("heartbeat")
            t += interval
        events.setdefault(100.0, []).append("bye")
        assert _simulate(lc, clock, events, until=600) is None, f"arresto con la finestra aperta (fase {phase})"


@pytest.mark.parametrize("interval", [20.0, sirio_app.HEARTBEAT_MAX_GAP])
def test_lifecycle_window_handed_to_running_browser(interval: float) -> None:
    # Il processo della finestra termina subito (il browser l'ha passata a un processo gia'
    # attivo dello stesso profilo) ma la finestra continua a inviare heartbeat.
    for phase in range(0, int(interval), 3):
        clock = Clock()
        lc = sirio_app.Lifecycle(auto_shutdown=True, clock=clock)
        events: dict[float, list[str]] = {1.0: ["window_exited"]}
        t = float(phase)
        while t <= 600:
            events.setdefault(t, []).append("heartbeat")
            t += interval
        assert _simulate(lc, clock, events, until=600) is None, f"arresto con la finestra aperta (fase {phase})"


def test_lifecycle_stops_after_last_window_closes() -> None:
    assert sirio_app.BYE_GRACE > sirio_app.HEARTBEAT_MAX_GAP + 10
    assert sirio_app.WINDOW_GRACE > sirio_app.HEARTBEAT_MAX_GAP + 10
    # ultima finestra chiusa: arresto dopo l'attesa, senza aspettare i 15 minuti di inattivita'
    clock = Clock()
    lc = sirio_app.Lifecycle(auto_shutdown=True, clock=clock)
    events = {float(t): ["heartbeat"] for t in range(0, 101, 20)}
    events[110.0] = ["bye"]
    stop = _simulate(lc, clock, events, until=1000)
    assert stop is not None and stop[1] == "finestra chiusa"
    assert 110 + sirio_app.BYE_GRACE <= stop[0] <= 110 + sirio_app.BYE_GRACE + 1
    # processo della finestra terminato e nessun heartbeat successivo
    clock = Clock()
    lc = sirio_app.Lifecycle(auto_shutdown=True, clock=clock)
    stop = _simulate(lc, clock, {0.0: ["heartbeat"], 20.0: ["heartbeat", "window_exited"]}, until=1000)
    assert stop is not None and stop[1] == "processo della finestra terminato"
    assert 20 + sirio_app.WINDOW_GRACE <= stop[0] <= 20 + sirio_app.WINDOW_GRACE + 1


def test_lifecycle_grace_exceeds_web_heartbeat_interval() -> None:
    js = (ROOT / "sirio" / "web" / "app.js").read_text(encoding="utf-8")
    intervals = [int(ms) / 1000 for ms in re.findall(r"setInterval\(\s*beat\s*,\s*(\d+)\s*\)", js)]
    for seconds in intervals:
        assert seconds <= sirio_app.HEARTBEAT_MAX_GAP < sirio_app.BYE_GRACE


# ==========================================================================
# launcher/avvia.sh (macOS e Linux) con un uv finto
# ==========================================================================

FAKE_UV = r"""#!/usr/bin/env bash
# uv finto: registra argomenti e variabili UV_*, simula sync e dry-run.
printf '%s\n' "$*" >> "$FAKE_LOG/uv-args"
env | grep '^UV_' | sort > "$FAKE_LOG/uv-env"
case " $* " in
  *" --help "*)
    if [ "${FAKE_UV_OLD:-0}" = 1 ]; then echo "      --native-tls  Whether to load TLS certificates"
    else echo "      --system-certs  Whether to load TLS certificates from the platform's native store"; fi
    exit 0 ;;
  *" --dry-run "*)
    echo "dry-run: verifica" >&2
    exit "${FAKE_UV_DRYRUN_RC:-0}" ;;
  *" --extra offline "*)
    if [ "${FAKE_UV_OFFLINE_RC:-0}" != 0 ]; then
      echo "error: Distribution torch==2.14.1 can't be installed for the current platform" >&2
      exit "$FAKE_UV_OFFLINE_RC"
    fi ;;
esac
mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"
cat > "$UV_PROJECT_ENVIRONMENT/bin/python" <<'EOF'
#!/usr/bin/env bash
printf 'python %s\n' "$*" >> "$FAKE_LOG/python-args"
EOF
chmod +x "$UV_PROJECT_ENVIRONMENT/bin/python"
echo "Installed 3 packages" >&2
exit 0
"""


@pytest.fixture
def sh_launcher(tmp_path: Path):
    if shutil.which("bash") is None or sys.platform.startswith("win"):
        pytest.skip("bash non disponibile")
    prog = tmp_path / "programma"
    (prog / "launcher").mkdir(parents=True)
    shutil.copy2(LAUNCHER / "avvia.sh", prog / "launcher" / "avvia.sh")
    (prog / "pyproject.toml").write_text('[project]\nname = "prova"\n', encoding="utf-8")
    (prog / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    (fakebin / "uv").write_text(FAKE_UV, encoding="utf-8")
    (fakebin / "uv").chmod(0o755)
    logs = tmp_path / "registro"
    logs.mkdir()

    def run(**extra: str) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("UV_", "SIRIO_", "FAKE_")) and k not in ("SSL_CERT_FILE", "PATH")}
        env.update({"PATH": f"{fakebin}{os.pathsep}/usr/bin{os.pathsep}/bin", "FAKE_LOG": str(logs), **extra})
        for name in ("uv-args", "python-args"):
            (logs / name).unlink(missing_ok=True)
        return subprocess.run(["bash", str(prog / "launcher" / "avvia.sh"), "--solo-installa"], env=env,
                              capture_output=True, text=True, timeout=120, check=False)

    def read(name: str) -> str:
        path = logs / name
        return path.read_text(encoding="utf-8") if path.exists() else ""

    return types.SimpleNamespace(prog=prog, run=run, read=read, runtime=prog / ".runtime")


def test_sh_launcher_installs_with_system_certificates(sh_launcher) -> None:
    res = sh_launcher.run()
    assert res.returncode == 0, res.stdout + res.stderr
    env = sh_launcher.read("uv-env")
    assert "UV_SYSTEM_CERTS=1" in env
    assert "UV_NATIVE_TLS" not in env            # uv recente: nessun avviso di deprecazione
    args = sh_launcher.read("uv-args").splitlines()
    assert any("--frozen --extra offline" in a for a in args)
    assert "-m sirio --scarica-modello" in sh_launcher.read("python-args")
    stamp = (sh_launcher.runtime / "installazione.ok").read_text(encoding="utf-8")
    assert re.fullmatch(r"[0-9a-f]{128}", stamp)
    # l'output di uv (anche su stderr) finisce nel registro dell'avvio
    log = (sh_launcher.runtime / "avvio.log").read_text(encoding="utf-8")
    assert "Installed 3 packages" in log and "sync --project" in log
    # secondo avvio: nulla da installare
    res = sh_launcher.run()
    assert res.returncode == 0 and sh_launcher.read("uv-args") == ""


def test_sh_launcher_old_uv_uses_native_tls(sh_launcher) -> None:
    res = sh_launcher.run(FAKE_UV_OLD="1")
    assert res.returncode == 0, res.stdout + res.stderr
    env = sh_launcher.read("uv-env")
    assert "UV_SYSTEM_CERTS=1" in env and "UV_NATIVE_TLS=1" in env


def test_sh_launcher_falls_back_without_offline_engine_on_unsupported_platform(sh_launcher) -> None:
    # es. Mac con processore Intel: PyTorch non ha pacchetti per questo computer
    res = sh_launcher.run(FAKE_UV_OFFLINE_RC="1", FAKE_UV_DRYRUN_RC="2")
    assert res.returncode == 0, res.stdout + res.stderr
    args = sh_launcher.read("uv-args").splitlines()
    offline = [a for a in args if "--extra offline" in a and "--dry-run" not in a]
    assert len(offline) == 2                                   # lockfile, poi risoluzione aggiornata
    assert any(a.endswith("--frozen --inexact") for a in args)    # componenti di base
    assert "Il motore offline non è disponibile su questo computer" in res.stdout
    assert "motore Claude" in res.stdout and "Installazione completata" in res.stdout
    assert sh_launcher.read("python-args") == ""               # nessun download del modello
    stamp = (sh_launcher.runtime / "installazione.ok").read_text(encoding="utf-8")
    assert stamp.endswith("-offline-non-supportato")
    log = (sh_launcher.runtime / "avvio.log").read_text(encoding="utf-8")
    assert "can't be installed for the current platform" in log
    # gli avvii successivi non ritentano l'installazione
    res = sh_launcher.run(FAKE_UV_OFFLINE_RC="1", FAKE_UV_DRYRUN_RC="2")
    assert res.returncode == 0 and sh_launcher.read("uv-args") == ""
    # ...finche' non cambiano le dipendenze
    (sh_launcher.prog / "uv.lock").write_text("version = 2\n", encoding="utf-8")
    res = sh_launcher.run()
    assert res.returncode == 0
    assert any("--frozen --extra offline" in a for a in sh_launcher.read("uv-args").splitlines())
    assert not (sh_launcher.runtime / "installazione.ok").read_text(encoding="utf-8").endswith("non-supportato")


def test_sh_launcher_retries_offline_engine_after_temporary_error(sh_launcher) -> None:
    res = sh_launcher.run(FAKE_UV_OFFLINE_RC="1", FAKE_UV_DRYRUN_RC="0")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "ritentata al prossimo avvio" in res.stdout
    assert not (sh_launcher.runtime / "installazione.ok").exists()
    res = sh_launcher.run()
    assert res.returncode == 0
    assert any("--frozen --extra offline" in a for a in sh_launcher.read("uv-args").splitlines())
    assert (sh_launcher.runtime / "installazione.ok").exists()


def test_sh_launcher_error_when_base_install_fails(sh_launcher, tmp_path: Path) -> None:
    broken = tmp_path / "bin" / "uv"
    broken.write_text('#!/usr/bin/env bash\necho "error: rete non raggiungibile" >&2\nexit 1\n', encoding="utf-8")
    res = sh_launcher.run()
    assert res.returncode == 1
    assert "Installazione delle dipendenze non riuscita" in res.stderr
    log = (sh_launcher.runtime / "avvio.log").read_text(encoding="utf-8")
    assert "rete non raggiungibile" in log and "Installazione delle dipendenze non riuscita" in log


# ==========================================================================
# launcher/avvia.ps1 e "Avvia Sirio OCR.bat" (Windows: controlli statici)
# ==========================================================================

def _ps1() -> str:
    raw = (LAUNCHER / "avvia.ps1").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and raw.count(b"\r\n") == raw.count(b"\n")   # UTF-8 con BOM, CRLF
    return raw.decode("utf-8-sig")


def _function(ps1: str, name: str) -> str:
    match = re.search(rf"^function {name}\b.*?^\}}", ps1, re.DOTALL | re.MULTILINE)
    assert match, name
    return match.group(0)


def test_ps1_uses_system_certificates() -> None:
    ps1 = _ps1()
    assert re.search(r"^\$env:UV_SYSTEM_CERTS\s*=\s*'1'", ps1, re.MULTILINE)
    certs = _function(ps1, "Set-CertificatiSistema")
    assert "system-certs" in certs and "$env:UV_NATIVE_TLS = '1'" in certs


def test_ps1_runtime_outside_program_folder() -> None:
    ps1 = _ps1()
    assert re.search(r"^\$Runtime\s*=.*Join-Path \$env:LOCALAPPDATA 'SirioOCR\\runtime'", ps1, re.MULTILINE)
    assert "$LogFile = Join-Path $Runtime 'avvio.log'" in ps1
    # la cartella ".runtime" delle versioni precedenti si elimina solo se non e' in uso
    old = _function(ps1, "Remove-RuntimePrecedente")
    assert "Test-InUso $RuntimePrecedente" in old and "Remove-Item -LiteralPath $RuntimePrecedente" in old


def test_ps1_falls_back_without_offline_engine() -> None:
    ps1 = _ps1()
    deps = _function(ps1, "Install-Dipendenze")
    assert "'--inexact'" in deps and "--dry-run --offline" in deps
    assert "'non-supportato'" in deps and "'non-riuscito'" in deps
    assert "'-offline-non-supportato'" in ps1
    assert "$registrata -eq $improntaBase" in ps1
    # pip: ripiego sui requisiti di base
    assert "requisiti-base.txt" in _function(ps1, "Install-ConPython")


def test_ps1_logs_native_output_and_checks_vc_runtime() -> None:
    ps1 = _ps1()
    run = _function(ps1, "Invoke-Registrato")
    assert "2>&1 | ForEach-Object { Write-Host" in run and "$ErrorActionPreference = 'Continue'" in run
    deps = _function(ps1, "Install-Dipendenze") + _function(ps1, "Sync-Dipendenze")
    assert "& $uv @argsSync" not in deps and "Invoke-Registrato $Uv" in deps
    vc = _function(ps1, "Test-RuntimeVisualC")
    assert "msvcp140.dll" in vc and "import torch" in vc
    assert "vc_redist." in ps1


def test_ps1_desktop_shortcut_is_repaired() -> None:
    ps1 = _ps1()
    sc = _function(ps1, "New-CollegamentoDesktop")
    assert "if (Test-Path $lnk) { return }" not in sc
    assert "$sc.TargetPath -eq $bat" in sc and "$sc.WindowStyle = 1" in sc
    assert "New-CollegamentoDesktop -SoloSeEsiste" in ps1


def _pwsh() -> str:
    exe = os.environ.get("SIRIO_PWSH") or shutil.which("pwsh") or shutil.which("powershell")
    if not exe:
        pytest.skip("PowerShell non disponibile (percorso facoltativo in SIRIO_PWSH)")
    return exe


def test_ps1_parses_with_powershell() -> None:
    exe = _pwsh()
    script = ("$t = $null; $e = $null; "
              "[void][System.Management.Automation.Language.Parser]::ParseFile($env:SIRIO_PS1, [ref]$t, [ref]$e); "
              "$e | ForEach-Object { $_.ToString() }; exit $e.Count")
    res = subprocess.run([exe, "-NoProfile", "-NonInteractive", "-Command", script],
                         env={**os.environ, "SIRIO_PS1": str(LAUNCHER / "avvia.ps1")},
                         capture_output=True, text=True, timeout=120, check=False)
    assert res.returncode == 0, res.stdout + res.stderr


FAKE_UV_WIN = FAKE_UV.replace('"$UV_PROJECT_ENVIRONMENT/bin"', '"$UV_PROJECT_ENVIRONMENT/Scripts"').replace(
    'cat > "$UV_PROJECT_ENVIRONMENT/bin/python" <<\'EOF\'', 'for n in python.exe pythonw.exe; do\n'
    'cat > "$UV_PROJECT_ENVIRONMENT/Scripts/$n" <<\'EOF\'').replace(
    'chmod +x "$UV_PROJECT_ENVIRONMENT/bin/python"', 'chmod +x "$UV_PROJECT_ENVIRONMENT/Scripts/$n"\ndone')


def test_ps1_install_flow_with_fake_uv(tmp_path: Path) -> None:
    # Esecuzione reale di avvia.ps1 (PowerShell 7 su macOS/Linux) con un uv finto.
    exe = _pwsh()
    if sys.platform.startswith("win") or shutil.which("bash") is None:
        pytest.skip("uv finto scritto in bash")
    assert "Scripts/$n" in FAKE_UV_WIN
    prog = tmp_path / "programma"
    (prog / "launcher").mkdir(parents=True)
    shutil.copy2(LAUNCHER / "avvia.ps1", prog / "launcher" / "avvia.ps1")
    (prog / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (prog / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (prog / ".runtime" / "venv").mkdir(parents=True)          # componenti della versione precedente
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    (fakebin / "uv").write_text(FAKE_UV_WIN, encoding="utf-8")
    (fakebin / "uv").chmod(0o755)
    logs = tmp_path / "registro"
    logs.mkdir()
    sysroot = tmp_path / "windows"
    (sysroot / "System32").mkdir(parents=True)
    (sysroot / "System32" / "msvcp140.dll").write_bytes(b"")
    runtime = tmp_path / "localappdata" / "SirioOCR" / "runtime"

    def run(**extra: str) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("UV_", "SIRIO_", "FAKE_")) and k not in ("SSL_CERT_FILE", "PATH")}
        env.update({"PATH": f"{fakebin}{os.pathsep}/usr/bin{os.pathsep}/bin", "FAKE_LOG": str(logs),
                    "LOCALAPPDATA": str(tmp_path / "localappdata"), "SystemRoot": str(sysroot), **extra})
        (logs / "uv-args").unlink(missing_ok=True)
        return subprocess.run([exe, "-NoProfile", "-NonInteractive", "-File", str(prog / "launcher" / "avvia.ps1"),
                               "-SoloInstalla"], env=env, capture_output=True, text=True, timeout=120, check=False)

    res = run(FAKE_UV_OFFLINE_RC="1", FAKE_UV_DRYRUN_RC="2")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "Il motore offline non è disponibile su questo computer" in res.stdout
    assert (runtime / "installazione.ok").read_text(encoding="ascii").strip().endswith("-offline-non-supportato")
    assert "UV_SYSTEM_CERTS=1" in (logs / "uv-env").read_text(encoding="utf-8")
    assert not (prog / ".runtime").exists()                     # vecchia cartella eliminata
    assert "can't be installed" in (runtime / "avvio.log").read_text(encoding="utf-8")
    res = run(FAKE_UV_OFFLINE_RC="1", FAKE_UV_DRYRUN_RC="2")
    assert res.returncode == 0 and not (logs / "uv-args").exists()   # nessun nuovo tentativo
    (prog / "uv.lock").write_text("version = 2\n", encoding="utf-8")
    res = run()
    assert res.returncode == 0, res.stdout + res.stderr
    assert "--frozen --extra offline" in (logs / "uv-args").read_text(encoding="utf-8")
    assert "-m sirio --scarica-modello" in (logs / "python-args").read_text(encoding="utf-8")
    assert not (runtime / "installazione.ok").read_text(encoding="ascii").strip().endswith("non-supportato")


def test_bat_handles_missing_script_and_negative_exit_codes() -> None:
    raw = (ROOT / "Avvia Sirio OCR.bat").read_bytes()
    raw.decode("ascii")
    assert raw.count(b"\r\n") == raw.count(b"\n")
    bat = raw.decode("ascii")
    check = bat.index('if not exist "%~dp0launcher\\avvia.ps1"')
    assert check < bat.index("powershell.exe")
    assert "Estrai tutto" in bat
    assert "if errorlevel 1" not in bat
    assert 'if not "%errorlevel%"=="0" (' in bat


# ==========================================================================
# Guida (README)
# ==========================================================================

def test_readme_install_and_privacy_notes() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "ad esempio in *Documenti*" not in readme
    assert "OneDrive" in readme and "SIRIO_EXPORT_DIR" in readme
    assert "1,7 GB" in readme and "4 GB" in readme
    assert r"%LOCALAPPDATA%\SirioOCR\runtime" in readme
    assert "Apri comunque" in readme
    assert "macOS 14" in readme
    assert "Visual C++" in readme
