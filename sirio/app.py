"""Avvio di Sirio OCR: server locale, finestra dell'applicazione e ciclo di vita.

``python -m sirio [--porta N] [--senza-finestra] [--cartella-dati DIR] [--log-level LIVELLO]``

1. istanza singola: un lucchetto su ``data_dir()/istanza.lock`` e il file
   ``istanza.json`` ``{port, pid, token}``; se un'istanza e' gia' attiva si apre
   solo una nuova finestra verso di essa e si esce;
2. uvicorn su ``127.0.0.1:<porta libera>`` in un thread, token casuale;
3. finestra: Microsoft Edge / Google Chrome / Chromium in modalita' applicazione
   (profilo dedicato in ``data_dir()/finestra``), altrimenti il browser predefinito;
4. chiusura automatica: processo della finestra terminato e nessun heartbeat da
   10 s, oppure ``/api/bye`` senza heartbeat successivi per 8 s, oppure nessun
   heartbeat per 15 minuti (``--senza-finestra`` disattiva la chiusura automatica);
5. registro in ``logs_dir()/sirio.log`` (con rotazione), eccezioni non gestite incluse.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser
from collections.abc import Callable
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from sirio import __version__, config

log = logging.getLogger("sirio.app")

HOST = "127.0.0.1"
WINDOW_SIZE = (1480, 940)
WINDOW_GRACE = 10.0          # finestra chiusa: attesa di eventuali heartbeat (s)
BYE_GRACE = 8.0              # dopo /api/bye (s)
IDLE_TIMEOUT = 15 * 60.0     # nessun heartbeat (s)
SUSPEND_GAP = 30.0           # pausa del ciclo oltre la quale si presume una sospensione del sistema
INSTANCE_WAIT = 30.0         # attesa dell'istanza in avvio (s)
SERVER_START_TIMEOUT = 20.0
PROCESSOR_STOP_TIMEOUT = 5.0

INSTANCE_FILE = "istanza.json"
LOCK_FILE = "istanza.lock"
PROFILE_DIR = "finestra"
_LOG_MARK = "_sirio_handler"


# ==========================================================================
# Registro
# ==========================================================================

def setup_logging(level: str = "INFO") -> Path | None:
    """Registro su ``logs_dir()/sirio.log`` (rotazione 2 MB x 5) e su stderr se disponibile."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _LOG_MARK, False):
            root.removeHandler(handler)
            handler.close()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s")
    log_file: Path | None = None
    try:
        log_file = config.logs_dir() / "sirio.log"
        fh = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
        fh.setFormatter(fmt)
        setattr(fh, _LOG_MARK, True)
        root.addHandler(fh)
    except OSError:
        log_file = None
    if sys.stderr is not None:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        setattr(sh, _LOG_MARK, True)
        root.addHandler(sh)
    for noisy in ("uvicorn.access", "httpx", "httpcore", "PIL", "urllib3", "multipart", "python_multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    def excepthook(exc_type, exc, tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logging.getLogger("sirio").critical("Eccezione non gestita", exc_info=(exc_type, exc, tb))

    def thread_excepthook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is SystemExit:
            return
        name = args.thread.name if args.thread else "?"
        logging.getLogger("sirio").critical(
            "Eccezione non gestita nel thread %s", name, exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
        )

    sys.excepthook = excepthook
    threading.excepthook = thread_excepthook
    return log_file


def show_error(message: str) -> None:
    """Messaggio d'errore visibile anche senza console (Windows: finestra di dialogo)."""
    if sys.stderr is not None:
        try:
            print(f"Sirio OCR: {message}", file=sys.stderr)
        except Exception:  # noqa: BLE001
            pass
    if sys.platform.startswith("win"):
        try:
            import ctypes  # noqa: PLC0415

            # MB_OK | MB_ICONERROR | MB_TOPMOST
            ctypes.windll.user32.MessageBoxW(None, message, "Sirio OCR", 0x0 | 0x10 | 0x40000)
        except Exception:  # noqa: BLE001
            pass


# ==========================================================================
# Istanza singola
# ==========================================================================

class InstanceLock:
    """Lucchetto esclusivo su file, mantenuto per tutta la vita del processo."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fh: Any = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")  # noqa: SIM115 - resta aperto finche' si detiene il lucchetto
        try:
            if os.name == "nt":
                import msvcrt  # noqa: PLC0415

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl  # noqa: PLC0415

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt  # noqa: PLC0415

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl  # noqa: PLC0415

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            fh.close()


def _instance_path() -> Path:
    return config.data_dir() / INSTANCE_FILE


def read_instance() -> dict | None:
    try:
        info = json.loads(_instance_path().read_text(encoding="utf-8"))
        if isinstance(info, dict) and isinstance(info.get("port"), int):
            return info
    except (OSError, ValueError):
        pass
    return None


def write_instance(info: dict) -> None:
    path = _instance_path()
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".istanza.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(info, fh)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def remove_instance(token: str) -> None:
    info = read_instance()
    if info is not None and info.get("token") != token:
        return  # appartiene a un'altra istanza
    try:
        _instance_path().unlink()
    except OSError:
        pass


def instance_url(port: int) -> str:
    return f"http://{HOST}:{port}/"


def probe_instance(port: int, timeout: float = 1.5) -> bool:
    """True se all'indirizzo risponde un server di Sirio OCR."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # mai tramite proxy
    try:
        with opener.open(f"http://{HOST}:{port}/api/health", timeout=timeout) as resp:
            data = json.loads(resp.read(4096).decode("utf-8"))
        return bool(data.get("ok"))
    except (OSError, ValueError):
        return False


def find_running_instance() -> dict | None:
    info = read_instance()
    if info and probe_instance(int(info["port"])):
        return info
    return None


# ==========================================================================
# Finestra dell'applicazione
# ==========================================================================

def _windows_registry_paths(exe: str) -> list[Path]:
    paths: list[Path] = []
    try:
        import winreg  # noqa: PLC0415
    except ImportError:
        return paths
    key_path = rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe}"
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, key_path) as key:
                value, _ = winreg.QueryValueEx(key, None)
                if value:
                    paths.append(Path(str(value).strip('"')))
        except OSError:
            continue
    return paths


def browser_candidates() -> list[tuple[str, Path]]:
    """Possibili eseguibili (nome, percorso) in ordine di preferenza: Edge, Chrome, Chromium."""
    out: list[tuple[str, Path]] = []
    override = os.environ.get("SIRIO_BROWSER", "").strip()
    if override:
        out.append(("personalizzato", Path(override)))
    if sys.platform.startswith("win"):
        roots = [os.environ.get(k) for k in ("ProgramFiles(x86)", "ProgramFiles", "ProgramW6432", "LOCALAPPDATA")]
        roots = [r for r in dict.fromkeys(roots) if r]
        for name, rel, exe in (
            ("Microsoft Edge", ("Microsoft", "Edge", "Application"), "msedge.exe"),
            ("Google Chrome", ("Google", "Chrome", "Application"), "chrome.exe"),
            ("Chromium", ("Chromium", "Application"), "chrome.exe"),
        ):
            out.extend((name, Path(root, *rel, exe)) for root in roots)
            if exe == "msedge.exe" or name == "Google Chrome":
                out.extend((name, p) for p in _windows_registry_paths(exe))
    elif sys.platform == "darwin":
        for base in (Path("/Applications"), Path.home() / "Applications"):
            out.append(("Microsoft Edge", base / "Microsoft Edge.app/Contents/MacOS/Microsoft Edge"))
            out.append(("Google Chrome", base / "Google Chrome.app/Contents/MacOS/Google Chrome"))
            out.append(("Chromium", base / "Chromium.app/Contents/MacOS/Chromium"))
    else:
        for name, exe in (
            ("Microsoft Edge", "microsoft-edge"),
            ("Microsoft Edge", "microsoft-edge-stable"),
            ("Google Chrome", "google-chrome"),
            ("Google Chrome", "google-chrome-stable"),
            ("Chromium", "chromium"),
            ("Chromium", "chromium-browser"),
        ):
            found = shutil.which(exe)
            if found:
                out.append((name, Path(found)))
    return out


def find_browser() -> tuple[str, Path] | None:
    for name, path in browser_candidates():
        try:
            if path.is_file():
                return name, path
        except OSError:
            continue
    return None


def window_command(browser: Path, url: str, profile_dir: Path) -> list[str]:
    return [
        str(browser),
        f"--app={url}",
        f"--user-data-dir={profile_dir}",
        f"--window-size={WINDOW_SIZE[0]},{WINDOW_SIZE[1]}",
        "--no-first-run",
        "--no-default-browser-check",
    ]


def open_window(url: str, profile_dir: Path) -> subprocess.Popen | None:
    """Apre la finestra dell'applicazione. Restituisce il processo del browser in modalita'
    app, oppure None se e' stato usato il browser predefinito (o nessun browser)."""
    found = find_browser()
    if found is not None:
        name, exe = found
        try:
            profile_dir.mkdir(parents=True, exist_ok=True)
            kwargs: dict[str, Any] = {
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "close_fds": True,
            }
            if sys.platform.startswith("win"):
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
            proc = subprocess.Popen(window_command(exe, url, profile_dir), **kwargs)  # noqa: S603
            log.info("Finestra aperta con %s (%s), pid %d", name, exe, proc.pid)
            return proc
        except OSError:
            log.warning("Avvio di %s non riuscito, uso il browser predefinito", name, exc_info=True)
    else:
        log.info("Edge/Chrome/Chromium non trovati: uso il browser predefinito")
    try:
        if webbrowser.open(url, new=1):
            return None
    except Exception:  # noqa: BLE001
        log.warning("Apertura del browser predefinito non riuscita", exc_info=True)
    log.error("Nessun browser disponibile: aprire manualmente %s", url)
    if sys.stderr is not None:
        try:
            print(f"Aprire nel browser: {url}", file=sys.stderr)
        except Exception:  # noqa: BLE001
            pass
    return None


# ==========================================================================
# Ciclo di vita
# ==========================================================================

class Lifecycle:
    """Decide quando arrestare l'applicazione (vedi docstring del modulo)."""

    def __init__(self, auto_shutdown: bool = True, clock: Callable[[], float] = time.monotonic):
        self.auto_shutdown = auto_shutdown
        self._clock = clock
        self._lock = threading.Lock()
        now = clock()
        self.last_heartbeat = now
        self.bye_at: float | None = None
        self.window_exited_at: float | None = None
        self.stop_event = threading.Event()
        self.reason = ""

    def heartbeat(self) -> None:
        with self._lock:
            self.last_heartbeat = self._clock()
            self.bye_at = None

    def bye(self) -> None:
        with self._lock:
            self.bye_at = self._clock()

    def window_exited(self) -> None:
        with self._lock:
            if self.window_exited_at is None:
                self.window_exited_at = self._clock()

    def resumed(self) -> None:
        """Dopo una sospensione del sistema i tempi d'attesa ripartono da zero."""
        with self._lock:
            self.last_heartbeat = self._clock()
            self.bye_at = None

    def request_stop(self, reason: str) -> None:
        with self._lock:
            if not self.reason:
                self.reason = reason
        self.stop_event.set()

    def check(self, now: float | None = None) -> str | None:
        """Motivo dell'arresto automatico, oppure None."""
        if not self.auto_shutdown:
            return None
        now = self._clock() if now is None else now
        with self._lock:
            if self.bye_at is not None and self.last_heartbeat <= self.bye_at and now - self.bye_at >= BYE_GRACE:
                return "finestra chiusa"
            if self.window_exited_at is not None and \
                    now - max(self.last_heartbeat, self.window_exited_at) >= WINDOW_GRACE:
                return "processo della finestra terminato"
            if now - self.last_heartbeat >= IDLE_TIMEOUT:
                return "nessuna attività da 15 minuti"
        return None


# ==========================================================================
# Server
# ==========================================================================

def bind_socket(port: int) -> socket.socket:
    """Socket TCP in ascolto su 127.0.0.1 (porta 0 = libera scelta dal sistema)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:  # Windows: nessun altro processo puo' agganciarsi alla stessa porta
            sock.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        sock.bind((HOST, port))
        sock.set_inheritable(True)
    except OSError:
        sock.close()
        raise
    return sock


class ServerThread:
    """uvicorn in un thread, su un socket gia' aperto."""

    def __init__(self, app: Any, sock: socket.socket):
        import uvicorn  # noqa: PLC0415

        self.sock = sock
        self.port = sock.getsockname()[1]
        cfg = uvicorn.Config(
            app,
            host=HOST,
            port=self.port,
            log_config=None,
            access_log=False,
            lifespan="off",
            ws="none",
            timeout_graceful_shutdown=3,
        )
        self.server = uvicorn.Server(cfg)
        self.thread = threading.Thread(target=self._run, name="sirio-http", daemon=True)
        self.error: BaseException | None = None

    def _run(self) -> None:
        try:
            self.server.run(sockets=[self.sock])
        except BaseException as exc:  # noqa: BLE001
            self.error = exc
            log.exception("Il server HTTP si è arrestato per un errore")

    def start(self, timeout: float = SERVER_START_TIMEOUT) -> None:
        self.thread.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.server.started:
                return
            if not self.thread.is_alive():
                break
            time.sleep(0.05)
        raise RuntimeError("Il server interno non si è avviato.") from self.error

    def stop(self, timeout: float = 10.0) -> None:
        self.server.should_exit = True
        self.thread.join(timeout)
        if self.thread.is_alive():
            self.server.force_exit = True
            self.thread.join(2.0)
        try:
            self.sock.close()
        except OSError:
            pass

    def is_alive(self) -> bool:
        return self.thread.is_alive()


# ==========================================================================
# main
# ==========================================================================

def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m sirio",
        description="Sirio OCR: lettura dei fogli firma e rendicontazione Excel.",
    )
    parser.add_argument("--porta", type=int, default=0,
                        help="porta del server locale (predefinita: una porta libera)")
    parser.add_argument("--senza-finestra", action="store_true",
                        help="non aprire la finestra e non chiudere automaticamente il server")
    parser.add_argument("--cartella-dati", metavar="CARTELLA",
                        help="cartella dei dati (documenti, impostazioni, registro)")
    parser.add_argument("--log-level", default="INFO", type=str.upper,
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                        help="livello di dettaglio del registro (predefinito: INFO)")
    args = parser.parse_args(argv)
    if not 0 <= args.porta <= 65535:
        parser.error("la porta deve essere compresa tra 1 e 65535")
    return args


def _say(text: str) -> None:
    if sys.stdout is not None:
        try:
            print(text, flush=True)
        except Exception:  # noqa: BLE001
            pass


def _delegate_to(info: dict, with_window: bool) -> int:
    url = instance_url(int(info["port"]))
    log.info("Sirio OCR è già in esecuzione (pid %s, %s)", info.get("pid"), url)
    if with_window:
        open_window(url, config.data_dir() / PROFILE_DIR)
    else:
        _say(f"Sirio OCR è già in esecuzione: {url}")
    return 0


def _install_signal_handlers(lifecycle: Lifecycle) -> None:
    if threading.current_thread() is not threading.main_thread():
        return

    def handler(signum, _frame) -> None:
        lifecycle.request_stop(f"segnale {signum}")

    for name in ("SIGTERM", "SIGBREAK", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError):
                pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cartella_dati:
        os.environ["SIRIO_DATA_DIR"] = str(Path(args.cartella_dati).expanduser().resolve())

    try:
        data_dir = config.data_dir()
    except OSError as exc:
        show_error(f"Impossibile creare la cartella dei dati: {exc}")
        return 1
    log_file = setup_logging(args.log_level)
    log.info("Avvio di Sirio OCR %s (Python %s, %s) - dati in %s", __version__,
             sys.version.split()[0], sys.platform, data_dir)

    with_window = not args.senza_finestra
    lock = InstanceLock(data_dir / LOCK_FILE)
    try:
        acquired = lock.acquire()
    except OSError as exc:
        log.critical("Lucchetto dell'istanza non disponibile", exc_info=True)
        show_error(f"Impossibile avviare Sirio OCR: la cartella dei dati non è scrivibile ({exc}).")
        return 1
    if not acquired:
        deadline = time.monotonic() + INSTANCE_WAIT
        while time.monotonic() < deadline:
            info = find_running_instance()
            if info is not None:
                return _delegate_to(info, with_window)
            time.sleep(0.5)
        show_error(
            "Sirio OCR risulta già in esecuzione ma non risponde.\n\n"
            "Attendere qualche secondo e riprovare; se il problema persiste riavviare il computer."
        )
        return 1

    existing = find_running_instance()   # istanza senza lucchetto (es. versione precedente)
    if existing is not None:
        lock.release()
        return _delegate_to(existing, with_window)

    token = secrets.token_urlsafe(24)
    lifecycle = Lifecycle(auto_shutdown=with_window)
    processor = None
    server: ServerThread | None = None
    window: subprocess.Popen | None = None
    try:
        from sirio.processing import Processor  # noqa: PLC0415
        from sirio.server import create_app  # noqa: PLC0415
        from sirio.store import DocumentStore  # noqa: PLC0415

        store = DocumentStore(config.documents_dir())
        processor = Processor(store, config.load_settings)
        app = create_app(
            store,
            processor,
            token,
            on_shutdown=lambda: lifecycle.request_stop("richiesta dall'interfaccia"),
            on_heartbeat=lifecycle.heartbeat,
            on_bye=lifecycle.bye,
        )
        try:
            sock = bind_socket(args.porta)
        except OSError as exc:
            if args.porta:
                raise RuntimeError(f"La porta {args.porta} non è disponibile ({exc.strerror or exc}).") from exc
            raise RuntimeError(f"Impossibile aprire il server locale ({exc.strerror or exc}).") from exc
        server = ServerThread(app, sock)
        server.start()
        url = instance_url(server.port)
        write_instance({
            "port": server.port,
            "pid": os.getpid(),
            "token": token,
            "version": __version__,
            "url": url,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        processor.start()
        log.info("Server in ascolto su %s", url)
        _say(f"Sirio OCR {__version__} in esecuzione su {url}")
        if args.senza_finestra:
            _say(f"Token per le API (header X-Sirio-Token): {token}")
            _say("Chiusura automatica disattivata: premere Ctrl+C per terminare.")
        if log_file:
            _say(f"Registro: {log_file}")
        if with_window:
            window = open_window(url, data_dir / PROFILE_DIR)
    except Exception as exc:  # noqa: BLE001
        log.critical("Avvio non riuscito", exc_info=True)
        show_error(f"Impossibile avviare Sirio OCR.\n\n{exc}\n\nDettagli nel registro: "
                   f"{log_file or config.data_dir()}")
        _shutdown(processor, server, token, lock)
        return 1

    _install_signal_handlers(lifecycle)
    last_tick = time.monotonic()
    try:
        while not lifecycle.stop_event.wait(0.5):
            now = time.monotonic()
            if now - last_tick > SUSPEND_GAP:
                log.info("Ripresa dopo una sospensione del sistema (%.0f s)", now - last_tick)
                lifecycle.resumed()
            last_tick = now
            if window is not None and lifecycle.window_exited_at is None and window.poll() is not None:
                log.info("Il processo della finestra è terminato (codice %s)", window.returncode)
                lifecycle.window_exited()
            if not server.is_alive():
                lifecycle.request_stop("server interno arrestato")
                break
            reason = lifecycle.check(now)
            if reason:
                lifecycle.request_stop(reason)
    except KeyboardInterrupt:
        lifecycle.request_stop("interruzione da tastiera")

    log.info("Arresto di Sirio OCR: %s", lifecycle.reason or "richiesto")
    if window is not None and window.poll() is None and sys.platform == "darwin" \
            and lifecycle.reason == "finestra chiusa":
        # su macOS il browser resta attivo senza finestre: si chiude il profilo dedicato
        try:
            window.terminate()
        except OSError:
            pass
    _shutdown(processor, server, token, lock)
    log.info("Sirio OCR terminato")
    return 0


def _shutdown(processor: Any, server: ServerThread | None, token: str, lock: InstanceLock) -> None:
    if processor is not None:
        try:
            # le letture ancora in corso dopo l'attesa riprendono al prossimo avvio
            processor.stop(timeout=PROCESSOR_STOP_TIMEOUT)
        except Exception:  # noqa: BLE001
            log.exception("Arresto della coda di elaborazione non riuscito")
    if server is not None:
        try:
            server.stop()
        except Exception:  # noqa: BLE001
            log.exception("Arresto del server non riuscito")
    remove_instance(token)
    lock.release()


__all__ = [
    "InstanceLock",
    "Lifecycle",
    "browser_candidates",
    "find_browser",
    "main",
    "open_window",
    "parse_args",
    "window_command",
]
