"""Archivio dei documenti su disco.

Struttura: ``<root>/<id>/`` con

* ``page.jpg``  pagina normalizzata (raddrizzata), JPEG q92, lato lungo <= 3600 px;
* ``thumb.jpg`` miniatura (lato lungo <= 480 px);
* ``grid.json`` geometria della tabella (``TableGrid.to_dict``) nelle coordinate di ``page.jpg``;
* ``preparazione.json`` correzioni applicate alla pagina importata (``normalize_page_info``:
  verso, inclinazione, prospettiva, raddrizzamento a maglia), per spiegarle all'utente;
* ``doc.json``  il ``Document`` (scritto per ultimo: una cartella senza ``doc.json``
  e' un'importazione interrotta e viene eliminata all'avvio).

Importazioni in sospeso: prima di importare le pagine di un file, ``add_file`` ne salva
una copia in ``<root>/.import-<sha256>.bin`` con ``.import-<sha256>.json`` (nome del
file), eliminati alla fine dell'importazione (anche se non riuscita: l'errore e'
segnalato all'utente). Se il programma si chiude a meta' di un PDF di piu' pagine i due
file restano e ``resume_pending_imports`` (chiamato all'avvio) completa l'importazione:
le pagine gia' presenti sono riconosciute (stesso sha256 e numero di pagina), quelle
mancanti vengono aggiunte e messe in coda.

Tutte le scritture sono atomiche (file temporaneo + ``os.replace``, ritentato se la
destinazione e' bloccata per un istante: ``sirio.fsutil.replace_with_retry``). I documenti
in memoria non vengono mai modificati sul posto: ``get``/``list`` restituiscono
copie (salvo ``list(copy=False)``, in sola lettura) e ``save``/``update``
sostituiscono la versione corrente. ``revision`` cresce a ogni modifica e serve
all'interfaccia per sapere se lo stato e' cambiato.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pydantic import ValidationError

from sirio import pdf_io
from sirio.fsutil import replace_with_retry
from sirio.models import Document
from sirio.vision.grid import TableGrid, detect_grid
from sirio.vision.preprocess import normalize_page_info

log = logging.getLogger(__name__)

RENDER_DPI = 300
PAGE_MAX_SIDE = 3600
PAGE_JPEG_QUALITY = 92
THUMB_MAX_SIDE = 480
THUMB_JPEG_QUALITY = 82

PAGE_FILE = "page.jpg"
THUMB_FILE = "thumb.jpg"
GRID_FILE = "grid.json"
PREP_FILE = "preparazione.json"
DOC_FILE = "doc.json"

PENDING_PREFIX = ".import-"     # importazioni in sospeso: .import-<sha256>.bin / .json
MAX_RESUME_ATTEMPTS = 3        # riprese di un'importazione in sospeso prima di abbandonarla

_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")

STATUS_MESSAGE_QUEUED = "In coda"


class DocumentNotFound(KeyError):
    """Il documento richiesto non esiste (o e' stato eliminato)."""

    def __str__(self) -> str:  # KeyError mostrerebbe la rappresentazione con apici
        return str(self.args[0]) if self.args else "Documento non trovato."


class PartialImportError(ValueError):
    """Importazione interrotta da una pagina illeggibile: le pagine precedenti sono state
    importate e sono disponibili in ``documents``."""

    def __init__(self, message: str, documents: list[Document]):
        super().__init__(message)
        self.documents = documents


def now_iso() -> str:
    """Data e ora locali in ISO 8601 con fuso orario e millisecondi."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _display_filename(filename: str) -> str:
    name = os.path.basename(str(filename or "").replace("\\", "/")).strip()
    # caratteri di controllo esclusi (il nome viene mostrato e scritto nell'Excel)
    name = "".join(ch for ch in name if ch >= " " and ch != "\x7f")
    return name[:255] or "documento"


def _atomic_write_bytes(path: Path, data: bytes, durable: bool = True) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            if durable:
                fh.flush()
                os.fsync(fh.fileno())
        replace_with_retry(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _rmtree(path: Path) -> None:
    for attempt in range(5):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == 4:
                log.warning("Impossibile eliminare la cartella %s", path, exc_info=True)
                return
            time.sleep(0.1 * (attempt + 1))


def _sort_key(doc: Document) -> tuple[str, str, int]:
    return (doc.created_at, doc.source_file.casefold(), doc.source_page)


class DocumentStore:
    """Archivio thread-safe dei documenti (una cartella per pagina importata)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._docs: dict[str, Document] = {}
        self._revision = 1
        # un'importazione per volta dello stesso file (deduplicazione affidabile)
        self._import_locks: dict[str, threading.Lock] = {}
        self._load_all()

    # ------------------------------------------------------------------ stato
    @property
    def revision(self) -> int:
        return self._revision

    def touch(self) -> int:
        """Segnala un cambiamento di stato non legato ai documenti (es. motore pronto)."""
        with self._lock:
            self._revision += 1
            return self._revision

    def _bump(self) -> None:
        self._revision += 1

    # -------------------------------------------------------------- caricamento
    def _load_all(self) -> None:
        loaded = 0
        for entry in sorted(self.root.iterdir()) if self.root.exists() else []:
            if not entry.is_dir() or not _ID_RE.match(entry.name):
                continue
            doc_file = entry / DOC_FILE
            if not doc_file.exists():
                log.info("Importazione incompleta eliminata: %s", entry.name)
                _rmtree(entry)
                continue
            for tmp in entry.glob(".*.tmp"):
                try:
                    tmp.unlink()
                except OSError:
                    pass
            try:
                doc = Document.model_validate_json(doc_file.read_bytes())
            except (OSError, ValidationError, ValueError):
                log.warning("Documento illeggibile ignorato: %s", doc_file, exc_info=True)
                continue
            if doc.id != entry.name:
                log.warning("Identificativo incoerente in %s: uso il nome della cartella", doc_file)
                doc.id = entry.name
            self._docs[doc.id] = doc
            loaded += 1
        if loaded:
            log.info("Archivio: %d documenti caricati da %s", loaded, self.root)
        # importazioni in sospeso con la copia scritta a meta' (nessun .json): non recuperabili.
        # I temporanei di _atomic_write_bytes si chiamano ".<nome>.<casuale>.tmp", cioe'
        # "..import-<sha256>.bin.<casuale>.tmp"
        for tmp in self.root.glob(f".{PENDING_PREFIX}*.tmp") if self.root.exists() else []:
            try:
                tmp.unlink()
            except OSError:
                pass
        for data_path in self.root.glob(f"{PENDING_PREFIX}*.bin") if self.root.exists() else []:
            sha = data_path.name[len(PENDING_PREFIX):-len(".bin")]
            if _SHA_RE.match(sha) and not self._pending_paths(sha)[1].exists():
                self._drop_pending_import(sha)

    # ------------------------------------------------------------------ percorsi
    def _dir(self, doc_id: str) -> Path:
        if not isinstance(doc_id, str) or not _ID_RE.match(doc_id):
            raise DocumentNotFound("Identificativo del documento non valido.")
        return self.root / doc_id

    def page_path(self, doc_id: str) -> Path:
        return self._dir(doc_id) / PAGE_FILE

    def thumb_path(self, doc_id: str) -> Path:
        return self._dir(doc_id) / THUMB_FILE

    def grid_path(self, doc_id: str) -> Path:
        return self._dir(doc_id) / GRID_FILE

    # ---------------------------------------------------------------- lettura
    def list(self, copy: bool = True) -> list[Document]:
        """Documenti ordinati per data di creazione, nome del file e pagina.

        Con ``copy=False`` restituisce le istanze interne: vanno usate in sola lettura.
        """
        with self._lock:
            docs = sorted(self._docs.values(), key=_sort_key)
            if copy:
                return [d.model_copy(deep=True) for d in docs]
            return docs

    def get(self, doc_id: str) -> Document | None:
        with self._lock:
            doc = self._docs.get(doc_id)
            return doc.model_copy(deep=True) if doc is not None else None

    def exists(self, doc_id: str) -> bool:
        with self._lock:
            return doc_id in self._docs

    def counts(self) -> dict[str, int]:
        """Numero di documenti per stato (senza copie)."""
        out = {"in_coda": 0, "in_lavorazione": 0, "completato": 0, "errore": 0, "scartato": 0}
        with self._lock:
            for d in self._docs.values():
                out[d.status] = out.get(d.status, 0) + 1
        return out

    def load_page(self, doc_id: str) -> np.ndarray:
        """Immagine BGR della pagina normalizzata. ``FileNotFoundError`` se manca."""
        path = self.page_path(doc_id)
        data = path.read_bytes()
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("L'immagine della pagina è danneggiata: reimportare il file.")
        return img

    def load_grid(self, doc_id: str) -> TableGrid | None:
        try:
            raw = json.loads(self.grid_path(doc_id).read_text(encoding="utf-8"))
            return TableGrid.from_dict(raw)
        except (OSError, ValueError, TypeError, DocumentNotFound):
            return None

    def load_preparation(self, doc_id: str) -> dict | None:
        """Correzioni applicate alla pagina all'importazione (``normalize_page_info``), se note."""
        try:
            raw = json.loads((self._dir(doc_id) / PREP_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, DocumentNotFound):
            return None
        return raw if isinstance(raw, dict) else None

    def save_grid(self, doc_id: str, grid: TableGrid) -> None:
        path = self.grid_path(doc_id)
        if not path.parent.exists():
            raise DocumentNotFound("Documento non trovato.")
        _atomic_write_bytes(path, json.dumps(grid.to_dict()).encode("utf-8"))

    # --------------------------------------------------------------- scrittura
    def _write_doc(self, doc: Document) -> None:
        folder = self._dir(doc.id)
        if not folder.exists():
            raise DocumentNotFound("Documento non trovato: è stato eliminato.")
        # Gli aggiornamenti di avanzamento (frequenti) non richiedono fsync: il risultato
        # finale si', e in caso di interruzione il documento viene comunque rielaborato.
        _atomic_write_bytes(
            folder / DOC_FILE,
            doc.model_dump_json(indent=1).encode("utf-8"),
            durable=doc.status != "in_lavorazione",
        )

    def save(self, doc: Document) -> None:
        """Salva (scrittura atomica) e aggiorna ``updated_at``.

        Solleva ``DocumentNotFound`` se il documento non esiste piu'.
        """
        with self._lock:
            if doc.id not in self._docs:
                raise DocumentNotFound("Documento non trovato: è stato eliminato.")
            doc.updated_at = now_iso()
            stored = doc.model_copy(deep=True)
            self._write_doc(stored)
            self._docs[doc.id] = stored
            self._bump()

    def update(self, doc_id: str, mutate: Callable[[Document], object]) -> Document | None:
        """Modifica atomica: applica ``mutate`` all'ultima versione e salva.

        Restituisce una copia del documento aggiornato, oppure ``None`` se il documento
        non esiste. Se ``mutate`` restituisce esattamente ``False`` non salva nulla.
        Le eccezioni sollevate da ``mutate`` si propagano senza modificare il documento.
        """
        with self._lock:
            current = self._docs.get(doc_id)
            if current is None:
                return None
            doc = current.model_copy(deep=True)
            if mutate(doc) is False:
                return doc
            if doc.id != doc_id:
                raise ValueError("L'identificativo del documento non può essere modificato.")
            self.save(doc)
            return doc.model_copy(deep=True)

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self._docs:
                return False
            del self._docs[doc_id]
            self._bump()
            folder = self._dir(doc_id)
        _rmtree(folder)
        return True

    def clear(self) -> None:
        with self._lock:
            ids = list(self._docs)
            self._docs.clear()
            self._bump()
        for doc_id in ids:
            _rmtree(self.root / doc_id)

    # -------------------------------------------------------------- importazione
    def _import_lock(self, sha: str) -> threading.Lock:
        with self._lock:
            lock = self._import_locks.get(sha)
            if lock is None:
                lock = self._import_locks[sha] = threading.Lock()
            return lock

    def _existing_pages(self, sha: str) -> dict[int, Document]:
        with self._lock:
            return {d.source_page: d for d in self._docs.values() if d.source_sha256 == sha}

    # ------------------------------------------------------- importazioni in sospeso
    def _pending_paths(self, sha: str) -> tuple[Path, Path]:
        return self.root / f"{PENDING_PREFIX}{sha}.bin", self.root / f"{PENDING_PREFIX}{sha}.json"

    def _save_pending_import(self, sha: str, name: str, data: bytes) -> None:
        """Copia del file da importare: se il programma si chiude prima della fine,
        ``resume_pending_imports`` completa l'importazione al prossimo avvio."""
        data_path, meta_path = self._pending_paths(sha)
        try:
            if meta_path.exists() and data_path.is_file() and data_path.stat().st_size == len(data):
                return   # ripresa di un'importazione gia' in sospeso
            _atomic_write_bytes(data_path, data)
            meta = {"file": name, "sha256": sha, "creato": now_iso()}
            _atomic_write_bytes(meta_path, json.dumps(meta, ensure_ascii=False).encode("utf-8"))
        except OSError:
            # l'importazione prosegue comunque: manca solo la ripresa dopo un'interruzione
            log.warning("Impossibile salvare la copia di «%s» per la ripresa dell'importazione", name, exc_info=True)
            self._drop_pending_import(sha)

    def _drop_pending_import(self, sha: str) -> None:
        for path in reversed(self._pending_paths(sha)):   # prima il .json: senza, il .bin e' ignorato
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                log.warning("Impossibile eliminare %s", path, exc_info=True)

    def pending_imports(self) -> list[str]:
        """Nomi dei file la cui importazione e' rimasta in sospeso."""
        out = []
        for meta_path in sorted(self.root.glob(f"{PENDING_PREFIX}*.json")):
            try:
                out.append(str(json.loads(meta_path.read_text(encoding="utf-8")).get("file") or meta_path.name))
            except (OSError, ValueError, AttributeError):
                out.append(meta_path.name)
        return out

    def resume_pending_imports(self, on_document: Callable[[Document], None] | None = None) -> list[Document]:
        """Completa le importazioni interrotte (es. programma chiuso a meta' di un PDF).

        Per ogni file in sospeso si ripete ``add_file`` sulla copia salvata: le pagine gia'
        importate non vengono duplicate, quelle mancanti sono aggiunte e notificate a
        ``on_document`` (la stessa funzione usata dal caricamento, che le mette in coda).
        Restituisce i documenti nuovi. Un file non recuperabile viene scartato (registro).
        """
        added: list[Document] = []

        def collect(doc: Document) -> None:
            added.append(doc)
            if on_document is not None:
                on_document(doc)

        for meta_path in sorted(self.root.glob(f"{PENDING_PREFIX}*.json")):
            sha = meta_path.name[len(PENDING_PREFIX):-len(".json")]
            if not _SHA_RE.match(sha):
                continue
            data_path = self._pending_paths(sha)[0]
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                name = _display_filename(str(meta.get("file") or ""))
                data = data_path.read_bytes()
            except (OSError, ValueError, AttributeError):
                log.warning("Importazione in sospeso non recuperabile: %s", meta_path.name, exc_info=True)
                self._drop_pending_import(sha)
                continue
            if hashlib.sha256(data).hexdigest() != sha:
                log.warning("Copia dell'importazione in sospeso di «%s» danneggiata: scartata", name)
                self._drop_pending_import(sha)
                continue
            # tentativi contati prima di riprendere: un file che fa chiudere il programma
            # (es. memoria esaurita) non deve bloccarne ogni avvio successivo
            tentativi = meta.get("tentativi")
            tentativi = tentativi if isinstance(tentativi, int) and tentativi >= 0 else 0
            if tentativi >= MAX_RESUME_ATTEMPTS:
                log.warning("Importazione in sospeso di «%s» abbandonata: già ripresa %d volte senza "
                            "completarla (le pagine importate restano)", name, tentativi)
                self._drop_pending_import(sha)
                continue
            try:
                _atomic_write_bytes(meta_path, json.dumps({**meta, "tentativi": tentativi + 1},
                                                          ensure_ascii=False).encode("utf-8"))
            except OSError:
                log.warning("Importazione in sospeso di «%s» non riprendibile", name, exc_info=True)
                self._drop_pending_import(sha)
                continue
            before = len(added)
            try:
                self.add_file(name, data, collect)
            except Exception:  # noqa: BLE001 - un file non recuperabile non blocca gli altri
                log.warning("Ripresa dell'importazione di «%s» non riuscita", name, exc_info=True)
            # (una chiusura del programma durante la ripresa lascia la copia: nuovo tentativo al
            # prossimo avvio, fino a MAX_RESUME_ATTEMPTS)
            self._drop_pending_import(sha)
            log.info("Ripresa l'importazione interrotta di «%s»: %d pagine aggiunte", name, len(added) - before)
        return added

    def _new_id(self) -> str:
        while True:
            doc_id = uuid.uuid4().hex[:12]
            if doc_id not in self._docs and not (self.root / doc_id).exists():
                return doc_id

    def add_file(
        self,
        filename: str,
        data: bytes,
        on_document: Callable[[Document], None] | None = None,
    ) -> list[Document]:
        """Importa un PDF o un'immagine: un ``Document`` (stato "in_coda") per pagina.

        Le pagine gia' presenti (stesso sha256 del file e stessa pagina) non vengono
        duplicate: si restituisce il documento esistente. ``on_document`` viene chiamata
        per ogni documento *nuovo* appena salvato (es. per metterlo subito in coda).

        Solleva ``ValueError`` (messaggio in italiano) se il file non e' leggibile e
        ``PartialImportError`` se una pagina intermedia e' danneggiata (le pagine
        precedenti restano importate).
        """
        name = _display_filename(filename)
        if not data:
            raise ValueError(f"Il file «{name}» è vuoto.")
        sha = hashlib.sha256(data).hexdigest()
        with self._import_lock(sha):
            n_pages = pdf_io.count_pages(data, name)
            existing = self._existing_pages(sha)
            if all(p in existing for p in range(1, n_pages + 1)):
                return [existing[p].model_copy(deep=True) for p in range(1, n_pages + 1)]

            # copia per la ripresa: se il programma si chiude a meta', le pagine mancanti
            # vengono importate al prossimo avvio (resume_pending_imports)
            self._save_pending_import(sha, name, data)
            results: list[Document] = []
            added = 0
            try:
                for index, image in pdf_io.iter_pages(data, name, dpi=RENDER_DPI):
                    page_no = index + 1
                    current = self._existing_pages(sha).get(page_no)
                    if current is not None:
                        results.append(current.model_copy(deep=True))
                        continue
                    doc = self._import_page(image, name, page_no, n_pages, sha)
                    added += 1
                    results.append(doc)
                    if on_document is not None:
                        try:
                            on_document(doc.model_copy(deep=True))
                        except Exception:  # noqa: BLE001 - il chiamante non deve interrompere l'importazione
                            log.exception("Errore nella notifica del documento importato %s", doc.id)
            except ValueError as exc:
                self._drop_pending_import(sha)   # errore segnalato all'utente: nessuna ripresa
                if added:
                    raise PartialImportError(
                        f"{exc} Sono state importate le prime {len(results)} pagine su {n_pages}.",
                        results,
                    ) from exc
                raise
            except Exception:
                self._drop_pending_import(sha)
                raise
            self._drop_pending_import(sha)
            if added:
                log.info("Importato «%s»: %d pagine nuove su %d", name, added, n_pages)
            return results

    def _import_page(self, image: np.ndarray, name: str, page_no: int, n_pages: int, sha: str) -> Document:
        prep: dict = {}
        try:
            page, prep = normalize_page_info(image)
        except Exception:  # noqa: BLE001 - si conserva la pagina cosi' com'e'
            log.warning("Normalizzazione non riuscita per «%s» pag. %d", name, page_no, exc_info=True)
            page = image
        # la griglia si cerca sulla pagina gia' raddrizzata e ridimensionata, quella salvata:
        # le coordinate di grid.json valgono per page.jpg (riquadri dell'interfaccia e motore)
        page = pdf_io.resize_max(page, max_side=PAGE_MAX_SIDE)
        grid = detect_grid(page)
        page_jpeg = pdf_io.encode_jpeg(page, quality=PAGE_JPEG_QUALITY)
        thumb_jpeg = pdf_io.encode_jpeg(page, quality=THUMB_JPEG_QUALITY, max_side=THUMB_MAX_SIDE)

        with self._lock:
            doc_id = self._new_id()
            folder = self.root / doc_id
            folder.mkdir(parents=True)
        try:
            _atomic_write_bytes(folder / PAGE_FILE, page_jpeg)
            _atomic_write_bytes(folder / THUMB_FILE, thumb_jpeg)
            _atomic_write_bytes(folder / GRID_FILE, json.dumps(grid.to_dict()).encode("utf-8"))
            if prep:
                _atomic_write_bytes(folder / PREP_FILE, json.dumps(prep).encode("utf-8"))
            stamp = now_iso()
            doc = Document(
                id=doc_id,
                source_file=name,
                source_page=page_no,
                page_count=n_pages,
                source_sha256=sha,
                created_at=stamp,
                updated_at=stamp,
                status="in_coda",
                progress=0.0,
                status_message=STATUS_MESSAGE_QUEUED,
            )
            with self._lock:
                self._write_doc(doc)
                self._docs[doc_id] = doc.model_copy(deep=True)
                self._bump()
        except BaseException:
            _rmtree(folder)
            raise
        return doc


__all__ = ["DocumentNotFound", "DocumentStore", "PartialImportError", "now_iso"]
