"""Coda di elaborazione dei documenti (thread).

Un thread "smistatore" preleva i documenti in coda e avvia un thread per
documento, fino al limite di concorrenza (``settings.concorrenza`` per Claude,
1 per il motore locale). Ciclo di un documento::

    in_coda -> in_lavorazione -> engine.extract (avanzamento salvato al massimo
    4 volte al secondo) -> anagrafica (nomi letti male ricondotti a quelli gia'
    confermati, vedi ``sirio.anagrafica``) -> apply_extraction -> validate_document
            -> completato | scartato (non e' un foglio firma) | errore

Se il motore scelto non e' disponibile (es. manca la chiave API) i documenti
restano "in_coda" con un messaggio esplicativo finche' ``kick()`` non li
sblocca. Ogni ``enqueue`` incrementa la *generazione* del documento: i
risultati di un'elaborazione superata (documento rimesso in coda o eliminato
nel frattempo) vengono scartati.
"""

from __future__ import annotations

import itertools
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable

from sirio import config
from sirio.anagrafica import Anagrafica, anagrafica_predefinita, applica_al_risultato
from sirio.config import Settings
from sirio.engines.base import EngineError, OCREngine, PageInput, get_engine
from sirio.models import Document, ExtractionResult
from sirio.store import STATUS_MESSAGE_QUEUED, DocumentNotFound, DocumentStore
from sirio.validation import validate_document
from sirio.vision.grid import TableGrid, detect_grid

log = logging.getLogger(__name__)

PROGRESS_INTERVAL = 0.25       # al massimo 4 salvataggi al secondo per documento
READY_TTL = 15.0               # validita' della verifica di disponibilita' del motore
BLOCKED_RECHECK = 60.0         # motore non pronto: nuova verifica automatica ogni minuto
IDLE_RECHECK = 60.0            # a riposo: la disponibilita' mostrata viene rinfrescata ogni minuto
MAX_CONCURRENCY = 8
STOP_TIMEOUT = 15.0

WAITING_NO_KEY = "In attesa: inserisci la chiave API nelle Impostazioni"
MESSAGE_NO_KEY = "Chiave API di Anthropic non configurata: inseriscila nelle Impostazioni per usare Claude."
MESSAGE_CHECKING = "Verifica del motore di lettura in corso…"
GENERIC_ERROR = (
    "Errore imprevisto durante la lettura del foglio. Riprova con «Rielabora»; "
    "se il problema persiste consulta il registro dell'applicazione (sirio.log)."
)
MISSING_PAGE_ERROR = (
    "L'immagine della pagina non è più disponibile: eliminare il documento e importare di nuovo il file."
)

_ENGINE_LABELS = {"claude": "Claude", "locale": "locale"}


_PROPER_NOUNS = {"Claude", "Anthropic", "Internet", "Windows", "Python", "Microsoft", "Google", "Hugging", "Edge",
                 "Chrome", "Sirio", "Napoli", "Excel"}


def _lower_first(text: str) -> str:
    """Minuscola iniziale per comporre frasi ("Chiave mancante" -> "chiave mancante"),
    lasciando intatti sigle e nomi propri ("PyTorch", "API", "Claude")."""
    first = text.split(" ", 1)[0].strip(",.:;")
    single = len(first) == 1 and first.isalpha() and first.isupper()          # "È", "A", "I"
    word = len(first) >= 2 and first[0].isupper() and first[1:].islower() and first not in _PROPER_NOUNS
    if single or word:
        return text[0].lower() + text[1:]
    return text


def waiting_message(kind: str, message: str) -> str:
    """Messaggio mostrato sui documenti in attesa che il motore diventi disponibile."""
    if kind == "claude" and not config.get_api_key():
        return WAITING_NO_KEY
    text = (message or "").strip().rstrip(".")
    if not text:
        return f"In attesa: il motore di lettura {_ENGINE_LABELS.get(kind, kind)} non è disponibile"
    return f"In attesa: {_lower_first(text)}"


class _ProgressSink:
    """Callback di avanzamento passata al motore: salva al massimo ogni ``interval`` secondi
    (l'ultimo valore viene comunque salvato con un timer) e non solleva mai eccezioni."""

    def __init__(self, processor: Processor, doc_id: str, gen: int, interval: float = PROGRESS_INTERVAL):
        self._processor = processor
        self._doc_id = doc_id
        self._gen = gen
        self._interval = interval
        self._lock = threading.Lock()
        self._pending: tuple[float, str] | None = None
        self._last = 0.0
        self._timer: threading.Timer | None = None
        self._closed = False

    def __call__(self, fraction: float, message: str = "") -> None:
        try:
            value = float(fraction)
            if math.isnan(value):
                value = 0.0
        except (TypeError, ValueError):
            value = 0.0
        value = min(1.0, max(0.0, value))
        with self._lock:
            if self._closed:
                return
            self._pending = (value, str(message or "").strip())
            wait = self._interval - (time.monotonic() - self._last)
            if wait <= 0:
                self._flush_locked()
            elif self._timer is None:
                self._timer = threading.Timer(wait, self._on_timer)
                self._timer.daemon = True
                self._timer.start()

    def _on_timer(self) -> None:
        with self._lock:
            self._timer = None
            if not self._closed and self._pending is not None:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if self._pending is None:
            return
        fraction, message = self._pending
        self._pending = None
        self._last = time.monotonic()
        progress = round(0.05 + 0.9 * fraction, 4)

        def apply(doc: Document) -> bool | None:
            if doc.status != "in_lavorazione":
                return False
            new_progress = max(doc.progress, progress)
            new_message = message[:300] if message else doc.status_message
            if new_progress == doc.progress and new_message == doc.status_message:
                return False
            doc.progress = new_progress
            doc.status_message = new_message
            return None

        try:
            self._processor._guarded_update(self._doc_id, self._gen, apply)
        except Exception:  # noqa: BLE001 - l'avanzamento non deve mai interrompere la lettura
            log.debug("Salvataggio dell'avanzamento non riuscito per %s", self._doc_id, exc_info=True)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._pending = None
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


class Processor:
    """Coda di elaborazione con thread di lavoro e controllo della disponibilita' del motore."""

    def __init__(
        self,
        store: DocumentStore,
        settings_provider: Callable[[], Settings],
        engine_factory: Callable[[Settings], OCREngine] = get_engine,
        anagrafica: Anagrafica | None = None,
    ):
        self.store = store
        self._settings_provider = settings_provider
        self._engine_factory = engine_factory
        self._anagrafica = anagrafica

        # coda e lavori in corso
        self._cond = threading.Condition(threading.RLock())
        self._queue: deque[str] = deque()
        self._queued: set[str] = set()
        self._jobs: dict[int, tuple[str, int]] = {}
        self._job_ids = itertools.count(1)
        self._limit = 1
        self._blocked = False
        self._blocked_since = 0.0
        self._refresh = True
        self._running = False
        self._stopping = False
        self._dispatcher: threading.Thread | None = None

        # generazioni dei documenti (risultati superati vengono scartati)
        self._doc_lock = threading.RLock()
        self._gen: dict[str, int] = {}

        # motori e disponibilita'
        self._engine_lock = threading.RLock()
        self._engines: dict[str, OCREngine] = {}
        self._availability: dict[str, tuple[float, bool, str]] = {}
        self._ready: tuple[bool, str, str] | None = None   # (pronto, messaggio, motore)
        self._ready_at = 0.0
        self._waiting = ""

    # ================================================================ ciclo di vita
    def start(self) -> None:
        """Avvia lo smistatore e rimette in coda i documenti rimasti "in_coda"/"in_lavorazione"."""
        with self._cond:
            if self._running:
                return
            self._running = True
            self._stopping = False
            self._refresh = True
        recovered = 0
        for doc in self.store.list(copy=False):
            if doc.status in ("in_coda", "in_lavorazione"):
                self._enqueue(doc.id, recovered=True)
                recovered += 1
        if recovered:
            log.info("Ripresi %d documenti in attesa di elaborazione", recovered)
        thread = threading.Thread(target=self._dispatch_loop, name="sirio-coda", daemon=True)
        with self._cond:
            self._dispatcher = thread
        thread.start()

    def stop(self, timeout: float = STOP_TIMEOUT) -> None:
        """Arresta lo smistatore e attende (entro ``timeout``) le letture in corso.

        Le letture che non terminano in tempo restano "in_lavorazione" su disco e
        vengono riprese al prossimo avvio."""
        with self._cond:
            if not self._running:
                return
            self._stopping = True
            self._cond.notify_all()
            dispatcher = self._dispatcher
        deadline = time.monotonic() + max(0.0, timeout)
        if dispatcher is not None and dispatcher is not threading.current_thread():
            dispatcher.join(max(0.1, deadline - time.monotonic()))
        with self._cond:
            while self._jobs:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(min(remaining, 0.5))
            pending = len(self._jobs)
            self._running = False
            self._dispatcher = None
        if pending:
            log.info("Arresto: %d letture ancora in corso verranno riprese al prossimo avvio", pending)

    # ================================================================ coda
    def enqueue(self, doc_id: str) -> None:
        """Mette (o rimette) in coda un documento. Un'eventuale lettura in corso dello
        stesso documento viene superata: il suo risultato sara' scartato."""
        self._enqueue(doc_id)

    def _enqueue(self, doc_id: str, recovered: bool = False) -> None:
        with self._cond:
            message = self._waiting if (self._ready is not None and not self._ready[0]) else STATUS_MESSAGE_QUEUED

        def reset(doc: Document) -> bool | None:
            if (
                doc.status == "in_coda"
                and doc.progress == 0.0
                and doc.error is None
                and doc.status_message == message
            ):
                return False
            doc.status = "in_coda"
            doc.progress = 0.0
            doc.error = None
            doc.status_message = message
            return None

        with self._doc_lock:
            if not self.store.exists(doc_id):
                return
            self._gen[doc_id] = self._gen.get(doc_id, 0) + 1
            if self.store.update(doc_id, reset) is None:
                return
        with self._cond:
            if doc_id not in self._queued:
                self._queue.append(doc_id)
                self._queued.add(doc_id)
            if not recovered:
                self._blocked = False   # nuovo lavoro: si riverifica il motore
            self._cond.notify_all()

    def discard(self, doc_id: str) -> None:
        """Toglie il documento dalla coda e invalida un'eventuale lettura in corso
        (da chiamare quando il documento viene eliminato)."""
        with self._doc_lock:
            self._gen[doc_id] = self._gen.get(doc_id, 0) + 1
        with self._cond:
            if doc_id in self._queued:
                self._queued.discard(doc_id)
                try:
                    self._queue.remove(doc_id)
                except ValueError:
                    pass

    def discard_all(self) -> None:
        """Svuota la coda e invalida tutte le letture in corso."""
        with self._cond:
            ids = set(self._queued) | {doc_id for doc_id, _ in self._jobs.values()}
            self._queue.clear()
            self._queued.clear()
        with self._doc_lock:
            for doc_id in ids | set(self._gen):
                self._gen[doc_id] = self._gen.get(doc_id, 0) + 1

    def kick(self) -> None:
        """Riverifica subito il motore (es. dopo il salvataggio della chiave API o un
        cambio di impostazioni) e riprende i documenti in attesa."""
        with self._engine_lock:
            self._engines.clear()
            self._availability.clear()
        with self._cond:
            self._blocked = False
            self._refresh = True
            self._cond.notify_all()

    # ================================================================ stato
    def status(self) -> dict:
        counts = self.store.counts()
        now = time.monotonic()
        with self._cond:
            ready = self._ready
            active = len(self._jobs)
            limit = self._limit
            if self._running and (ready is None or now - self._ready_at > IDLE_RECHECK):
                self._refresh = True
                self._cond.notify_all()
        if ready is None:
            pronto, message, engine = False, MESSAGE_CHECKING, self._settings().engine
        else:
            pronto, message, engine = ready
        if not pronto:
            messaggio = message or MESSAGE_CHECKING
        elif counts["in_lavorazione"] or counts["in_coda"]:
            messaggio = (
                f"Lettura in corso: {counts['in_lavorazione']} in lavorazione, "
                f"{counts['in_coda']} in coda"
            )
        else:
            messaggio = "Pronto"
        return {
            "in_coda": counts["in_coda"],
            "in_lavorazione": counts["in_lavorazione"],
            "completati": counts["completato"],
            "errori": counts["errore"],
            "scartati": counts["scartato"],
            "motore_pronto": bool(pronto),
            "messaggio": messaggio,
            "motore": engine,
            "concorrenza": limit,
            "attivi": active,
        }

    def engine_status(self, kind: str) -> tuple[bool, str]:
        """Disponibilita' di un motore ("claude" | "locale") con le impostazioni correnti."""
        settings = self._settings()
        try:
            settings = settings.model_copy(update={"engine": kind})
        except Exception:  # noqa: BLE001
            return False, "Motore di lettura sconosciuto."
        _engine, ok, message = self._prepare(settings)
        if not ok and kind == "claude" and not config.get_api_key():
            return False, MESSAGE_NO_KEY
        return ok, message

    # ================================================================ anagrafica
    @property
    def anagrafica(self) -> Anagrafica:
        """Anagrafica usata per correggere i nomi letti (predefinita: quella della cartella dei dati)."""
        return self._anagrafica if self._anagrafica is not None else anagrafica_predefinita()

    def _apply_registry(self, doc_id: str, result: ExtractionResult) -> None:
        """Riconduce i nomi letti a quelli gia' confermati; un problema dell'anagrafica non
        deve mai far fallire la lettura (il risultato resta quello del motore)."""
        try:
            decisions = applica_al_risultato(result, self.anagrafica)
        except Exception:  # noqa: BLE001
            log.exception("Applicazione dell'anagrafica non riuscita per %s", doc_id)
            return
        changed = [d.campo for d in decisions if d.esito in ("sostituito", "dedotto")]
        if changed:
            log.info("Documento %s: anagrafica applicata a %s", doc_id, ", ".join(changed))

    # ================================================================ motore
    def _settings(self) -> Settings:
        try:
            settings = self._settings_provider()
            if isinstance(settings, Settings):
                return settings
        except Exception:  # noqa: BLE001
            log.exception("Lettura delle impostazioni non riuscita: uso i valori predefiniti")
        return Settings()

    def _prepare(self, settings: Settings) -> tuple[OCREngine | None, bool, str]:
        """(motore, pronto, messaggio) con cache: il motore viene ricostruito solo se
        cambiano le impostazioni (o dopo ``kick``)."""
        signature = settings.model_dump_json()
        label = _ENGINE_LABELS.get(settings.engine, settings.engine)
        with self._engine_lock:
            engine = self._engines.get(signature)
            if engine is None:
                try:
                    engine = self._engine_factory(settings)
                except EngineError as exc:
                    return None, False, str(exc) or f"Il motore di lettura {label} non è disponibile."
                except ImportError:
                    log.exception("Componenti del motore %s mancanti", settings.engine)
                    return None, False, (
                        f"Il motore di lettura {label} non è installato correttamente: "
                        "riavviare il programma dal collegamento sul desktop per completare l'installazione."
                    )
                except Exception:  # noqa: BLE001
                    log.exception("Creazione del motore %s non riuscita", settings.engine)
                    return None, False, (
                        f"Il motore di lettura {label} non è disponibile per un errore interno "
                        "(dettagli nel registro sirio.log)."
                    )
                if len(self._engines) >= 4:
                    self._engines.clear()
                    self._availability.clear()
                self._engines[signature] = engine
            cached = self._availability.get(signature)
            if cached is not None and time.monotonic() - cached[0] < READY_TTL:
                return engine, cached[1], cached[2]
            try:
                ok, message = engine.is_available()
                ok, message = bool(ok), str(message or "")
            except EngineError as exc:
                ok, message = False, str(exc)
            except Exception:  # noqa: BLE001
                log.exception("Verifica del motore %s non riuscita", settings.engine)
                ok, message = False, (
                    f"Impossibile verificare il motore di lettura {label} (dettagli nel registro sirio.log)."
                )
            self._availability[signature] = (time.monotonic(), ok, message)
            return engine, ok, message

    @staticmethod
    def _concurrency(settings: Settings, engine: OCREngine | None) -> int:
        kind = getattr(engine, "name", None) or settings.engine
        if kind == "claude":
            try:
                return max(1, min(MAX_CONCURRENCY, int(settings.concorrenza)))
            except (TypeError, ValueError):
                return 1
        return 1

    def _publish_readiness(self, ready: bool, message: str, kind: str) -> None:
        waiting = "" if ready else waiting_message(kind, message)
        state = (ready, message if not ready else "", kind)
        with self._cond:
            changed = self._ready != state or self._waiting != waiting
            self._ready = state
            self._ready_at = time.monotonic()
            self._waiting = waiting
            queued = list(self._queue)
        if not changed:
            return
        if ready:
            log.info("Motore di lettura «%s» pronto", kind)
        else:
            log.info("Motore di lettura «%s» non disponibile: %s", kind, message)
        target = waiting or STATUS_MESSAGE_QUEUED

        def set_message(doc: Document) -> bool | None:
            if doc.status != "in_coda" or doc.status_message == target:
                return False
            doc.status_message = target
            return None

        for doc_id in queued:
            try:
                self.store.update(doc_id, set_message)
            except Exception:  # noqa: BLE001
                log.debug("Aggiornamento del messaggio di attesa non riuscito", exc_info=True)
        self.store.touch()

    # ================================================================ smistatore
    def _has_work_locked(self) -> bool:
        if self._refresh:
            return True
        return bool(self._queue) and not self._blocked and len(self._jobs) < self._limit

    def _dispatch_loop(self) -> None:
        try:
            while True:
                with self._cond:
                    while not self._stopping and not self._has_work_locked():
                        self._cond.wait(5.0)
                        if self._blocked and time.monotonic() - self._blocked_since > BLOCKED_RECHECK:
                            self._blocked = False
                            self._refresh = True
                    if self._stopping:
                        return
                    self._refresh = False
                settings = self._settings()
                engine, ready, message = self._prepare(settings)
                kind = getattr(engine, "name", None) or settings.engine
                self._publish_readiness(ready, message, kind)
                limit = self._concurrency(settings, engine)
                with self._cond:
                    self._limit = limit
                    if self._stopping:
                        return
                    if not ready or engine is None:
                        if not self._blocked:
                            self._blocked = True
                            self._blocked_since = time.monotonic()
                        continue
                    self._blocked = False
                    while self._queue and len(self._jobs) < limit:
                        doc_id = self._queue.popleft()
                        self._queued.discard(doc_id)
                        if not self.store.exists(doc_id):
                            continue
                        with self._doc_lock:
                            gen = self._gen.get(doc_id, 0)
                        job_id = next(self._job_ids)
                        self._jobs[job_id] = (doc_id, gen)
                        worker = threading.Thread(
                            target=self._run_job,
                            args=(job_id, doc_id, gen, engine),
                            name=f"sirio-lettura-{doc_id}",
                            daemon=True,
                        )
                        worker.start()
        except Exception:  # noqa: BLE001 - lo smistatore non deve morire in silenzio
            log.exception("Errore nello smistatore della coda di elaborazione")
            with self._cond:
                self._running = False

    # ================================================================ lavoro
    def _guarded_update(self, doc_id: str, gen: int, mutate: Callable[[Document], object]) -> Document | None:
        """Aggiorna il documento solo se la lettura ``gen`` e' ancora quella valida."""
        with self._doc_lock:
            if self._gen.get(doc_id, 0) != gen:
                return None
            return self.store.update(doc_id, mutate)

    def _run_job(self, job_id: int, doc_id: str, gen: int, engine: OCREngine) -> None:
        try:
            self._process(doc_id, gen, engine)
        except Exception:  # noqa: BLE001
            log.exception("Errore non gestito nell'elaborazione del documento %s", doc_id)
        finally:
            with self._cond:
                self._jobs.pop(job_id, None)
                self._cond.notify_all()

    def _load_input(self, doc: Document) -> PageInput:
        image = self.store.load_page(doc.id)
        h, w = image.shape[:2]
        grid: TableGrid | None = self.store.load_grid(doc.id)
        if grid is not None and (grid.width, grid.height) != (w, h):
            grid = grid.scaled(w / float(grid.width), h / float(grid.height))
            grid.width, grid.height = w, h
        if grid is None:
            grid = detect_grid(image)
            try:
                self.store.save_grid(doc.id, grid)
            except (OSError, DocumentNotFound):
                log.debug("Griglia non salvata per %s", doc.id, exc_info=True)
        return PageInput(image=image, grid=grid, source_file=doc.source_file, page=doc.source_page)

    def _process(self, doc_id: str, gen: int, engine: OCREngine) -> None:
        started = time.monotonic()
        engine_name = str(getattr(engine, "name", "") or "")

        def begin(doc: Document) -> None:
            doc.status = "in_lavorazione"
            doc.progress = 0.02
            doc.error = None
            doc.status_message = "Preparazione della pagina…"
            if engine_name:
                doc.engine = engine_name

        doc = self._guarded_update(doc_id, gen, begin)
        if doc is None:
            return
        log.info("Lettura di «%s» pag. %d (%s) con il motore %s", doc.source_file, doc.source_page, doc_id,
                 engine_name or "?")
        sink = _ProgressSink(self, doc_id, gen)
        try:
            page = self._load_input(doc)
            sink(0.0, "Lettura del foglio in corso…")
            result = engine.extract(page, sink)
            if not isinstance(result, ExtractionResult):
                raise TypeError(f"Il motore ha restituito {type(result).__name__} invece di ExtractionResult")
        except EngineError as exc:
            sink.close()
            self._fail(doc_id, gen, str(exc).strip() or GENERIC_ERROR)
            return
        except (FileNotFoundError, DocumentNotFound):
            sink.close()
            if self.store.exists(doc_id):
                self._fail(doc_id, gen, MISSING_PAGE_ERROR)
            return
        except Exception:  # noqa: BLE001
            sink.close()
            log.exception("Lettura del documento %s non riuscita", doc_id)
            self._fail(doc_id, gen, GENERIC_ERROR)
            return
        sink.close()
        self._apply_registry(doc_id, result)
        elapsed = round(time.monotonic() - started, 2)

        def finish(doc: Document) -> None:
            doc.apply_extraction(result)
            if not doc.engine:
                doc.engine = engine_name or None
            if doc.usage.seconds <= 0:
                doc.usage.seconds = elapsed
            # una nuova lettura sostituisce i dati: le correzioni precedenti non valgono piu'
            doc.user_edited = []
            doc.ocr_originali = {}
            doc.user_verified = False
            validate_document(doc)
            doc.error = None
            doc.progress = 1.0
            if doc.is_foglio_firma:
                doc.status = "completato"
                doc.status_message = "Lettura completata"
            else:
                doc.status = "scartato"
                doc.status_message = "Pagina non riconosciuta come foglio firma"

        try:
            done = self._guarded_update(doc_id, gen, finish)
        except Exception:  # noqa: BLE001
            log.exception("Registrazione del risultato non riuscita per %s", doc_id)
            self._fail(doc_id, gen, GENERIC_ERROR)
            return
        if done is not None:
            log.info("Documento %s: %s in %.1f s (%s)", doc_id, done.status, elapsed, done.totals.stato)
            # le correzioni precedenti sono state sostituite dalla nuova lettura: quanto
            # appreso da questo documento resta nell'anagrafica ma non e' piu' legato a esso
            try:
                self.anagrafica.scollega_documento(doc_id)
            except Exception:  # noqa: BLE001
                log.exception("Aggiornamento dell'anagrafica non riuscito per %s", doc_id)

    def _fail(self, doc_id: str, gen: int, message: str) -> None:
        def mark(doc: Document) -> None:
            doc.status = "errore"
            doc.error = message
            doc.progress = 0.0
            doc.status_message = "Lettura non riuscita"

        try:
            if self._guarded_update(doc_id, gen, mark) is not None:
                log.warning("Documento %s non elaborato: %s", doc_id, message)
        except Exception:  # noqa: BLE001
            log.exception("Impossibile registrare l'errore del documento %s", doc_id)


__all__ = ["Processor", "WAITING_NO_KEY", "waiting_message"]
