"""API HTTP (FastAPI) e file statici dell'interfaccia.

Sicurezza (il server ascolta solo su 127.0.0.1):

* ogni richiesta deve avere un header ``Host`` locale (127.0.0.1 / localhost / [::1]):
  protezione dal *DNS rebinding*;
* ogni richiesta ``/api/*`` (tranne ``/api/health``) deve presentare il token della
  sessione: header ``X-Sirio-Token`` oppure parametro ``?t=<token>``. Per le sole
  richieste di lettura (GET/HEAD: immagini, download) e per ``heartbeat``/``bye``
  (inviati anche con ``navigator.sendBeacon``) vale anche il cookie ``sirio_token``
  (HttpOnly, SameSite=Strict) impostato quando viene servita la pagina principale;
* nessun CORS: le pagine di altri siti non possono leggere le risposte.

Tutti gli errori hanno la forma ``{"detail": "messaggio in italiano"}``.
"""

from __future__ import annotations

import logging
import math
import os
import re
import secrets
import subprocess
import sys
import threading
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import cv2
import numpy as np
from fastapi import Body, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import HTTPConnection
from starlette.staticfiles import StaticFiles

from sirio import __version__, calendario, config, pdf_io
from sirio import validation as val
from sirio.models import DAY_FIELDS, HEADER_FIELDS, DayRow, Document, Header
from sirio.processing import Processor
from sirio.store import DocumentNotFound, DocumentStore, PartialImportError

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web"
TOKEN_PLACEHOLDER = "__SIRIO_TOKEN__"
TOKEN_HEADER = "x-sirio-token"
TOKEN_COOKIE = "sirio_token"
TOKEN_QUERY = ("t", "token")

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_UPLOAD_FILES = 5000
XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
_COOKIE_POST_PATHS = {"/api/heartbeat", "/api/bye"}
_IMAGE_CACHE = "private, max-age=86400"

FORBIDDEN_HOST = "Richiesta rifiutata: Sirio OCR accetta solo connessioni da questo computer."
FORBIDDEN_TOKEN = (
    "Accesso non autorizzato: la sessione non è valida o è scaduta. "
    "Chiudere la finestra e riaprire Sirio OCR."
)

_STATUS_TEXT = {
    400: "Richiesta non valida.",
    401: "Accesso non autorizzato.",
    403: "Accesso negato.",
    404: "Risorsa non trovata.",
    405: "Metodo non consentito per questa risorsa.",
    409: "Operazione non possibile nello stato attuale.",
    413: "Il contenuto inviato è troppo grande.",
    415: "Formato del contenuto non supportato.",
    422: "Dati non validi.",
    500: "Errore interno del server.",
    503: "Servizio momentaneamente non disponibile.",
}

_TIME_FIELDS = ("prog_entrata", "prog_uscita", "eff_entrata", "eff_uscita")
_ROW_BOOL_FIELDS = ("assenza_alunno", "assenza_operatore", "firma")
_HEADER_TEXT_FIELDS = (
    "anno_scolastico", "lotto", "municipalita", "ente", "istituto", "operatore", "alunno", "data_compilazione",
)
_HEADER_HOURS_FIELDS = ("ore_pei", "totale_mensile_dichiarato")
_HEADER_BOOL_FIELDS = ("firma_coordinatore", "timbro_referente")
_DASH_RE = re.compile(r"^[\s\-–—−_]+$")
_TRUE_WORDS = {"true", "1", "si", "sì", "s", "x", "yes", "vero", "presente", "on"}
_FALSE_WORDS = {"false", "0", "no", "n", "falso", "assente", "off", ""}
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
_FILENAME_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')


class ApiError(Exception):
    """Errore da restituire al client con un messaggio in italiano."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


# ==========================================================================
# Utilita'
# ==========================================================================

def _json(data: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(data, status_code=status_code)


def _host_is_local(host_header: str | None) -> bool:
    if not host_header:
        return False
    host = host_header.strip().lower()
    if host.startswith("["):
        end = host.find("]")
        if end < 0:
            return False
        name, rest = host[1:end], host[end + 1:]
        if rest and not re.fullmatch(r":\d{1,5}", rest):
            return False
    else:
        name, _, port = host.partition(":")
        if port and not port.isdigit():
            return False
    return name in _LOCAL_HOSTS


def _round(x: float | None, digits: int = 2) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, digits)


def _short(text: str, limit: int = 60) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def open_path(path: Path) -> None:
    """Apre un file o una cartella con il programma predefinito del sistema."""
    target = str(path)
    if sys.platform.startswith("win"):
        os.startfile(target)  # type: ignore[attr-defined]  # noqa: S606
        return
    command = ["open", target] if sys.platform == "darwin" else ["xdg-open", target]
    subprocess.Popen(  # noqa: S603
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


# --------------------------------------------------------------------------
# Riepiloghi
# --------------------------------------------------------------------------

def doc_summary(doc: Document) -> dict:
    """``DocSummary`` (vedi docs/ARCHITETTURA.md, 4.10)."""
    h, t = doc.header, doc.totals
    return {
        "id": doc.id,
        "display_name": doc.display_name(),
        "source_file": doc.source_file,
        "source_page": doc.source_page,
        "page_count": doc.page_count,
        "status": doc.status,
        "progress": _round(doc.progress, 4) or 0.0,
        "status_message": doc.status_message,
        "error": doc.error,
        "is_foglio_firma": doc.is_foglio_firma,
        "engine": doc.engine,
        "model": doc.model,
        "user_verified": doc.user_verified,
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
        "header": {
            "operatore": h.operatore,
            "alunno": h.alunno,
            "istituto": h.istituto,
            "ente": h.ente,
            "mese": h.mese,
            "anno": h.anno,
            "ore_pei": h.ore_pei,
        },
        "totals": {
            "ore_riconosciute": _round(t.ore_riconosciute),
            "ore_dichiarate": _round(t.ore_dichiarate),
            "ore_calcolate": _round(t.ore_calcolate),
            "totale_mensile_dichiarato": _round(t.totale_mensile_dichiarato),
            "differenza_totale": _round(t.differenza_totale),
            "giorni_lavorati": t.giorni_lavorati,
            "n_errori": t.n_errori,
            "n_attenzioni": t.n_attenzioni,
            "n_info": t.n_info,
            "campi_incerti": t.campi_incerti,
            "campi_illeggibili": t.campi_illeggibili,
            "stato": t.stato,
        },
        "cost_usd": _round(doc.usage.cost_usd, 4) or 0.0,
        "n_modifiche": len(doc.user_edited),
        "thumb_url": f"/api/documents/{doc.id}/thumb",
        "image_url": f"/api/documents/{doc.id}/image",
    }


def _exportable(doc: Document) -> bool:
    return doc.status == "completato" and doc.is_foglio_firma


def compute_kpi(docs: list[Document]) -> dict:
    """Indicatori per la dashboard.

    ``errori``/``attenzioni`` = numero di anomalie di quella gravita', ``illeggibili``/``incerti``
    = numero di campi, ``da_verificare`` = fogli completati con stato diverso da OK non ancora
    confermati dall'utente, ``verificati`` = fogli confermati.
    """
    done = [d for d in docs if _exportable(d)]
    return {
        "documenti": len(docs),
        "completati": len(done),
        "ore_totali": _round(sum(d.totals.ore_riconosciute for d in done)) or 0.0,
        "errori": sum(d.totals.n_errori for d in done),
        "attenzioni": sum(d.totals.n_attenzioni for d in done),
        "illeggibili": sum(d.totals.campi_illeggibili for d in done),
        "incerti": sum(d.totals.campi_incerti for d in done),
        "da_verificare": sum(1 for d in done if d.totals.stato != "ok" and not d.user_verified),
        "verificati": sum(1 for d in done if d.user_verified),
        "documenti_con_errori": sum(1 for d in done if d.totals.stato == "errori"),
        "in_coda": sum(1 for d in docs if d.status == "in_coda"),
        "in_lavorazione": sum(1 for d in docs if d.status == "in_lavorazione"),
        "errori_elaborazione": sum(1 for d in docs if d.status == "errore"),
        "scartati": sum(1 for d in docs if d.status == "scartato"),
        "costo_usd": _round(sum(d.usage.cost_usd for d in docs), 4) or 0.0,
    }


def _field_state(doc: Document, obj: Header | DayRow, campo: str, key: str) -> str:
    stato = val.stato_campo(obj, campo)
    if stato:
        return stato
    return "corretto" if key in doc.user_edited else ""


def _period(doc: Document) -> tuple[int, int] | None:
    m, a = doc.header.mese, doc.header.anno
    if isinstance(m, int) and isinstance(a, int) and 1 <= m <= 12 and 2000 <= a <= 2100:
        return m, a
    return None


def day_info(doc: Document) -> list[dict]:
    """Per ogni giorno: calendario (tipo di giorno, festivita') e valori calcolati."""
    period = _period(doc)
    out = []
    for row in doc.rows:
        g = row.giorno
        tipo = nome = festa = None
        if period:
            mese, anno = period
            tipo = calendario.tipo_giorno(anno, mese, g)
            nome = calendario.nome_giorno(anno, mese, g, breve=True)
            festa = calendario.nome_festivita(anno, mese, g)
        out.append({
            "giorno": g,
            "tipo_giorno": tipo,
            "giorno_settimana": nome,
            "festivita": festa,
            "ore_programmate": _round(val.hours_between(row.prog_entrata, row.prog_uscita)),
            "ore_calcolate": _round(val.hours_between(row.eff_entrata, row.eff_uscita)),
            "ore_riconosciute": _round(val.ore_riconosciute(row)),
            "esito": val.esito_riga(row, doc.anomalies) if doc.status in ("completato", "scartato") else "",
            "stati": {
                campo: s for campo in DAY_FIELDS
                if (s := _field_state(doc, row, campo, f"rows.{g}.{campo}"))
            },
        })
    return out


def document_payload(doc: Document, grid: dict | None) -> dict:
    data = doc.model_dump(mode="json")
    data.update({
        "display_name": doc.display_name(),
        "grid": grid,
        "image_url": f"/api/documents/{doc.id}/image",
        "thumb_url": f"/api/documents/{doc.id}/thumb",
        "giorni": day_info(doc),
        "stati_intestazione": {
            campo: s for campo in HEADER_FIELDS
            if (s := _field_state(doc, doc.header, campo, f"header.{campo}"))
        },
    })
    return data


def preview_payload(docs: list[Document], giorni_vuoti: bool = False) -> dict:
    """Dati per l'anteprima dell'Excel (Riepilogo, Dettaglio giornaliero, Anomalie)."""
    dettaglio: list[dict] = []
    anomalie: list[dict] = []
    for doc in docs:
        h = doc.header
        period = _period(doc)
        info = {d["giorno"]: d for d in day_info(doc)}
        for row in doc.rows:
            g = row.giorno
            if period and calendario.tipo_giorno(period[1], period[0], g) == "inesistente" and not row.has_content():
                continue
            if not giorni_vuoti and not row.has_content():
                continue
            calc = info[g]
            dich = row.ore_dichiarate
            diff = None
            if dich is not None and calc["ore_calcolate"] is not None:
                diff = _round(dich - calc["ore_calcolate"])
            data_iso = f"{period[1]:04d}-{period[0]:02d}-{g:02d}" if period and calc["tipo_giorno"] != "inesistente" else None
            dettaglio.append({
                "doc_id": doc.id,
                "documento": doc.display_name(),
                "operatore": h.operatore,
                "alunno": h.alunno,
                "istituto": h.istituto,
                "data": data_iso,
                "giorno": g,
                "giorno_settimana": calc["giorno_settimana"],
                "tipo_giorno": calc["tipo_giorno"],
                "festivita": calc["festivita"],
                "prog_entrata": row.prog_entrata,
                "prog_uscita": row.prog_uscita,
                "eff_entrata": row.eff_entrata,
                "eff_uscita": row.eff_uscita,
                "ore_programmate": calc["ore_programmate"],
                "ore_calcolate": calc["ore_calcolate"],
                "ore_dichiarate": dich,
                "ore_riconosciute": calc["ore_riconosciute"],
                "differenza": diff,
                "assenza_alunno": row.assenza_alunno,
                "assenza_operatore": row.assenza_operatore,
                "firma": row.firma,
                "note": row.note,
                "trattino_effettivo": row.trattino_effettivo,
                "esito": calc["esito"],
                "anomalie": "; ".join(a.messaggio for a in val.anomalies_for_day(doc.anomalies, g)),
                "stati": calc["stati"],
            })
        for a in doc.anomalies:
            anomalie.append({"doc_id": doc.id, "documento": doc.display_name(), **a.model_dump(mode="json")})
    return {
        "documenti": [doc_summary(d) for d in docs],
        "dettaglio": dettaglio,
        "anomalie": anomalie,
        "kpi": compute_kpi(docs),
    }


# --------------------------------------------------------------------------
# Modifica di un documento (PUT)
# --------------------------------------------------------------------------

def _as_original(value: Any) -> str | None:
    """Valore OCR originale come testo (per ``ocr_originali``)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _text(value: Any, limit: int, what: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ApiError(422, f"{what}: valore non valido.")
    text = " ".join(str(value).split())
    if not text:
        return None
    if len(text) > limit:
        raise ApiError(422, f"{what}: testo troppo lungo (massimo {limit} caratteri).")
    return text


def _bool(value: Any, what: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    raise ApiError(422, f"{what}: indicare sì o no.")


def _hours(value: Any, what: str) -> float | None:
    if value is None or (isinstance(value, str) and (not value.strip() or _DASH_RE.match(value))):
        return None
    if isinstance(value, bool):
        raise ApiError(422, f"{what}: valore delle ore non valido.")
    if isinstance(value, int):
        value = float(value)
    parsed = val.parse_hours(value)
    if parsed is None or parsed < 0 or parsed > 744:
        raise ApiError(422, f"{what}: valore delle ore non valido («{_short(value)}»). Esempi validi: 3, 1,5, 2:30.")
    return round(float(parsed), 4)


def _time(value: Any, what: str) -> tuple[str | None, bool]:
    """(orario normalizzato o testo originale se non interpretabile, trattino)."""
    if value is None:
        return None, False
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ApiError(422, f"{what}: orario non valido.")
    text = str(value).strip()
    if not text:
        return None, False
    if _DASH_RE.match(text):
        return None, True
    if len(text) > 20:
        raise ApiError(422, f"{what}: orario non valido («{_short(text)}»).")
    # un orario non interpretabile resta com'e': la validazione lo segnala (E03)
    return val.normalize_time(text) or text, False


def _month(value: Any) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ApiError(422, "Mese: valore non valido.")
    if isinstance(value, str):
        word = value.strip().lower()
        for i, name in enumerate(calendario.MESI, start=1):
            if word in (name, name[:3]):
                return i
        value = word
    try:
        month = int(float(value))
    except (TypeError, ValueError):
        raise ApiError(422, f"Mese: valore non valido («{_short(value)}»), indicare un numero da 1 a 12.") from None
    if not 1 <= month <= 12:
        raise ApiError(422, f"Mese: valore non valido ({month}), indicare un numero da 1 a 12.")
    return month


def _year(value: Any) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ApiError(422, "Anno: valore non valido.")
    try:
        year = int(float(str(value).strip()))
    except (TypeError, ValueError):
        raise ApiError(422, f"Anno: valore non valido («{_short(value)}»).") from None
    if 0 <= year <= 99:
        year += 2000
    if not 2000 <= year <= 2100:
        raise ApiError(422, f"Anno: valore non valido ({year}).")
    return year


def _sostituzione(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "SI" if value else "NO"
    word = str(value).strip().lower()
    if not word:
        return None
    if word in ("si", "sì", "s", "true", "x", "yes"):
        return "SI"
    if word in ("no", "n", "false"):
        return "NO"
    raise ApiError(422, "Sostituzione: indicare SI o NO.")


def _flag_list(value: Any, allowed: tuple[str, ...]) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ApiError(422, "Elenco dei campi incerti/illeggibili non valido.")
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item in allowed and item not in out:
            out.append(item)
    return out


def _normalize_header_value(campo: str, value: Any) -> Any:
    what = val.etichetta_campo(campo)
    if campo in _HEADER_TEXT_FIELDS:
        return _text(value, 200, what)
    if campo == "mese":
        return _month(value)
    if campo == "anno":
        return _year(value)
    if campo in _HEADER_HOURS_FIELDS:
        return _hours(value, what)
    if campo == "sostituzione":
        return _sostituzione(value)
    if campo in _HEADER_BOOL_FIELDS:
        return _bool(value, what)
    raise ApiError(422, f"Campo d'intestazione sconosciuto: {campo}.")


def _track(doc: Document, obj: Header | DayRow, campo: str, key: str, old: Any, new: Any) -> None:
    """Registra la modifica di un campo (o il ripristino del valore OCR originale)."""
    if old == new:
        return
    setattr(obj, campo, new)
    if campo in obj.incerti:
        obj.incerti.remove(campo)
    if campo in obj.illeggibili:
        obj.illeggibili.remove(campo)
    if key not in doc.ocr_originali and key not in doc.user_edited:
        doc.ocr_originali[key] = _as_original(old)
    original = doc.ocr_originali.get(key)
    if original is not None and _as_original(new) == original:
        # l'utente ha ripristinato il valore letto: non e' piu' una correzione (per i campi
        # letti vuoti/illeggibili la cancellazione resta invece registrata come modifica)
        doc.ocr_originali.pop(key, None)
        if key in doc.user_edited:
            doc.user_edited.remove(key)
        return
    if key not in doc.user_edited:
        doc.user_edited.append(key)


def _merge_header(doc: Document, payload: dict) -> None:
    if not isinstance(payload, dict):
        raise ApiError(422, "Intestazione non valida.")
    header = doc.header
    flags = {k: _flag_list(payload.get(k), HEADER_FIELDS) for k in ("incerti", "illeggibili") if k in payload}
    if flags.get("incerti") is not None:
        header.incerti = flags["incerti"]
    if flags.get("illeggibili") is not None:
        header.illeggibili = flags["illeggibili"]
    for campo in HEADER_FIELDS:
        if campo not in payload:
            continue
        new = _normalize_header_value(campo, payload[campo])
        _track(doc, header, campo, f"header.{campo}", getattr(header, campo), new)


def _merge_row(doc: Document, payload: dict) -> None:
    if not isinstance(payload, dict):
        raise ApiError(422, "Riga giornaliera non valida.")
    try:
        g = int(payload.get("giorno"))
    except (TypeError, ValueError):
        raise ApiError(422, "Riga giornaliera senza un giorno valido (1-31).") from None
    if not 1 <= g <= 31:
        raise ApiError(422, f"Giorno non valido: {g} (ammessi 1-31).")
    row = next((r for r in doc.rows if r.giorno == g), None)
    if row is None:
        row = DayRow(giorno=g)
        doc.rows.append(row)
        doc.rows.sort(key=lambda r: r.giorno)
    prefix = f"Giorno {g}"
    for k in ("incerti", "illeggibili"):
        if k in payload:
            flags = _flag_list(payload.get(k), DAY_FIELDS)
            if flags is not None:
                setattr(row, k, flags)
    dash_typed = False
    for campo in DAY_FIELDS:
        if campo not in payload:
            continue
        what = f"{prefix}, {val.etichetta_campo(campo).lower()}"
        value = payload[campo]
        if campo in _TIME_FIELDS:
            new, dash = _time(value, what)
            if dash and campo.startswith("eff_"):
                dash_typed = True
        elif campo == "ore_dichiarate":
            new = _hours(value, what)
        elif campo in _ROW_BOOL_FIELDS:
            new = _bool(value, what)
        else:  # note
            new = _text(value, 500, what)
        _track(doc, row, campo, f"rows.{g}.{campo}", getattr(row, campo), new)
    if "trattino_effettivo" in payload:
        row.trattino_effettivo = _bool(payload["trattino_effettivo"], f"{prefix}, trattino")
    elif dash_typed:
        row.trattino_effettivo = True
    elif (row.eff_entrata or row.eff_uscita) and any(c in payload for c in ("eff_entrata", "eff_uscita")):
        row.trattino_effettivo = False


def apply_document_update(doc: Document, payload: dict) -> None:
    """Applica una modifica dell'utente (vedi ``PUT /api/documents/{id}``) e rivalida."""
    if doc.status in ("in_coda", "in_lavorazione"):
        raise ApiError(
            409,
            "Il documento è in elaborazione: attendere il termine della lettura prima di modificarlo.",
        )
    if payload.get("header") is not None:
        _merge_header(doc, payload["header"])
    rows = payload.get("rows")
    if rows is not None:
        if not isinstance(rows, list):
            raise ApiError(422, "Elenco delle righe giornaliere non valido.")
        for item in rows:
            _merge_row(doc, item)
    if payload.get("is_foglio_firma") is not None:
        is_ff = _bool(payload["is_foglio_firma"], "Foglio firma")
        doc.is_foglio_firma = is_ff
        if is_ff and doc.status == "scartato":
            doc.status = "completato"
            doc.status_message = "Riconosciuto manualmente come foglio firma"
        elif not is_ff and doc.status == "completato":
            doc.status = "scartato"
            doc.status_message = "Escluso manualmente: non è un foglio firma"
    if payload.get("user_verified") is not None:
        doc.user_verified = _bool(payload["user_verified"], "Conferma documento")
    val.validate_document(doc)


# --------------------------------------------------------------------------
# File di esportazione
# --------------------------------------------------------------------------

def safe_export_name(name: str | None) -> str | None:
    """Nome di file ``.xlsx`` sicuro (senza percorso e caratteri non ammessi), o None."""
    if not name:
        return None
    base = os.path.basename(str(name).replace("\\", "/"))
    base = _FILENAME_BAD_CHARS.sub("_", base).strip().strip(".").strip()
    if base.lower().endswith(".xlsx"):
        base = base[:-5]
    stem = base.strip(" .")[:150].strip(" .")
    if not stem:
        return None
    if stem.split(".")[0].upper() in _WINDOWS_RESERVED:
        stem = f"_{stem}"
    return f"{stem}.xlsx"


def _unique_path(folder: Path, filename: str) -> Path:
    path = folder / filename
    if not path.exists():
        return path
    stem = filename[:-5]
    for i in range(2, 1000):
        candidate = folder / f"{stem} ({i}).xlsx"
        if not candidate.exists():
            return candidate
    return folder / f"{stem}_{secrets.token_hex(3)}.xlsx"


def resolve_export_file(filename: str) -> Path:
    """Percorso di un file della cartella export; ``ApiError`` se il nome non e' ammesso
    (percorsi, file nascosti, estensioni diverse da .xlsx) o il file non esiste."""
    if (
        not filename
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
        or filename.startswith(".")
        or filename != os.path.basename(filename)
        or not filename.lower().endswith(".xlsx")
        or ":" in filename
    ):
        raise ApiError(400, "Nome del file non valido.")
    base = config.export_dir().resolve()
    path = (base / filename).resolve()
    if path.parent != base or not path.is_file():
        raise ApiError(404, "File non trovato nella cartella delle esportazioni.")
    return path


def _export_entry(path: Path) -> dict:
    st = path.stat()
    return {
        "filename": path.name,
        "path": str(path),
        "size": st.st_size,
        "created_at": datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds"),
        "url": f"/api/exports/{quote(path.name)}",
    }


# --------------------------------------------------------------------------
# Messaggi di validazione in italiano
# --------------------------------------------------------------------------

_PYDANTIC_IT = {
    "missing": "campo obbligatorio mancante",
    "int_parsing": "deve essere un numero intero",
    "int_type": "deve essere un numero intero",
    "int_from_float": "deve essere un numero intero",
    "float_parsing": "deve essere un numero",
    "float_type": "deve essere un numero",
    "bool_parsing": "deve essere vero o falso",
    "bool_type": "deve essere vero o falso",
    "string_type": "deve essere un testo",
    "list_type": "deve essere un elenco",
    "dict_type": "deve essere un oggetto",
    "model_type": "deve essere un oggetto",
    "literal_error": "valore non ammesso",
    "enum": "valore non ammesso",
    "greater_than_equal": "valore troppo piccolo",
    "greater_than": "valore troppo piccolo",
    "less_than_equal": "valore troppo grande",
    "less_than": "valore troppo grande",
    "json_invalid": "JSON non valido",
    "extra_forbidden": "campo non previsto",
}


def validation_message(errors: list[dict]) -> str:
    parts = []
    for err in errors[:3]:
        loc = [str(x) for x in err.get("loc", ()) if x not in ("body", "query", "path")]
        where = ".".join(loc)
        msg = _PYDANTIC_IT.get(str(err.get("type", "")), "valore non valido")
        parts.append(f"«{where}»: {msg}" if where else msg)
    return "Dati non validi: " + "; ".join(parts) + "." if parts else "Dati non validi."


# ==========================================================================
# Middleware: host locale e token
# ==========================================================================

class _GuardMiddleware:
    def __init__(self, app: Any, token: str):
        self.app = app
        self.token = token

    def _token_ok(self, conn: HTTPConnection, method: str, path: str) -> bool:
        candidates = [conn.headers.get(TOKEN_HEADER)]
        candidates += [conn.query_params.get(name) for name in TOKEN_QUERY]
        if method in ("GET", "HEAD") or (method == "POST" and path in _COOKIE_POST_PATHS):
            candidates.append(conn.cookies.get(TOKEN_COOKIE))
        return any(c and secrets.compare_digest(c.encode(), self.token.encode()) for c in candidates)

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        conn = HTTPConnection(scope)
        path = scope.get("path", "")
        method = scope.get("method", "GET").upper()
        if not _host_is_local(conn.headers.get("host")):
            log.warning("Richiesta rifiutata: host non locale %r", conn.headers.get("host"))
            await _json({"detail": FORBIDDEN_HOST}, 403)(scope, receive, send)
            return
        if path.startswith("/api/") and path != "/api/health" and not self._token_ok(conn, method, path):
            await _json({"detail": FORBIDDEN_TOKEN}, 403)(scope, receive, send)
            return

        async def send_with_headers(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                names = {k.lower() for k, _ in headers}
                extra = [(b"x-content-type-options", b"nosniff"), (b"referrer-policy", b"no-referrer")]
                if path.startswith("/api/") and b"cache-control" not in names:
                    extra.append((b"cache-control", b"no-store"))
                message = {**message, "headers": headers + [h for h in extra if h[0] not in names]}
            await send(message)

        await self.app(scope, receive, send_with_headers)


class _NoCacheStaticFiles(StaticFiles):
    def file_response(self, full_path, stat_result, scope, status_code: int = 200) -> Response:  # type: ignore[override]
        response = super().file_response(full_path, stat_result, scope, status_code)
        response.headers["Cache-Control"] = "no-cache"
        return response


# ==========================================================================
# Applicazione
# ==========================================================================

class _PageCache:
    """Piccola cache LRU delle pagine decodificate (ritagli e anteprime ridotte)."""

    def __init__(self, size: int = 3):
        self._size = size
        self._lock = threading.Lock()
        self._items: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()

    def get(self, store: DocumentStore, doc_id: str) -> np.ndarray:
        path = store.page_path(doc_id)
        key = (doc_id, path.stat().st_mtime_ns)
        with self._lock:
            img = self._items.get(key)
            if img is not None:
                self._items.move_to_end(key)
                return img
        img = store.load_page(doc_id)
        with self._lock:
            for k in [k for k in self._items if k[0] == doc_id]:
                del self._items[k]
            self._items[key] = img
            while len(self._items) > self._size:
                self._items.popitem(last=False)
        return img

    def drop(self, doc_id: str | None = None) -> None:
        with self._lock:
            if doc_id is None:
                self._items.clear()
            else:
                for k in [k for k in self._items if k[0] == doc_id]:
                    del self._items[k]


class ExportRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    ids: list[str] | None = None
    options: dict[str, Any] | None = None
    filename: str | None = None


class OpenRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    target: Literal["export_dir", "file"]
    filename: str | None = None


class KeyTestRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    api_key: str | None = None
    model: str | None = None


def _call(callback: Callable[[], Any] | None, what: str) -> None:
    if callback is None:
        return
    try:
        callback()
    except Exception:  # noqa: BLE001
        log.exception("Errore nella gestione di %s", what)


def create_app(
    store: DocumentStore,
    processor: Processor,
    token: str,
    on_shutdown: Callable[[], Any] | None = None,
    on_heartbeat: Callable[[], Any] | None = None,
    on_bye: Callable[[], Any] | None = None,
) -> FastAPI:
    """Crea l'applicazione FastAPI (vedi docs/ARCHITETTURA.md, 4.10)."""
    if not token:
        raise ValueError("Token di sessione mancante.")
    app = FastAPI(title="Sirio OCR", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(_GuardMiddleware, token=token)
    app.mount("/static", _NoCacheStaticFiles(directory=WEB_DIR, check_dir=False), name="static")
    pages = _PageCache()

    # ------------------------------------------------------------ errori
    @app.exception_handler(ApiError)
    async def _api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return _json({"detail": exc.detail}, exc.status_code)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else ""
        # i messaggi standard (in inglese) di Starlette vengono tradotti
        if not detail or detail.isascii() and detail.lower().rstrip(".") in {
            "not found", "method not allowed", "bad request", "forbidden", "unauthorized",
            "internal server error", "request entity too large", "unsupported media type",
        }:
            detail = _STATUS_TEXT.get(exc.status_code, "Errore nella richiesta.")
        return _json({"detail": detail}, exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return _json({"detail": validation_message(list(exc.errors()))}, 422)

    @app.exception_handler(Exception)
    async def _unexpected(_request: Request, exc: Exception) -> JSONResponse:
        log.error("Errore non gestito", exc_info=exc)
        return _json({"detail": "Errore interno: l'operazione non è riuscita. Dettagli nel registro sirio.log."}, 500)

    def _doc_or_404(doc_id: str) -> Document:
        doc = store.get(doc_id)
        if doc is None:
            raise ApiError(404, "Documento non trovato: potrebbe essere stato eliminato.")
        return doc

    # ------------------------------------------------------------ pagina
    def _index(request: Request) -> Response:
        index = WEB_DIR / "index.html"
        try:
            html = index.read_text(encoding="utf-8")
        except OSError:
            return HTMLResponse(
                "<!doctype html><html lang='it'><meta charset='utf-8'><title>Sirio OCR</title>"
                "<body style='font-family:sans-serif;padding:2rem'><h1>Sirio OCR</h1>"
                "<p>L'interfaccia non è installata correttamente (manca <code>index.html</code>). "
                "Reinstallare il programma.</p></body></html>",
                status_code=503,
            )
        response = HTMLResponse(
            html.replace(TOKEN_PLACEHOLDER, token),
            headers={"Cache-Control": "no-store", "Content-Security-Policy": "frame-ancestors 'none'"},
        )
        response.set_cookie(TOKEN_COOKIE, token, httponly=True, samesite="strict", path="/")
        return response

    app.add_api_route("/", _index, methods=["GET"], include_in_schema=False)
    app.add_api_route("/index.html", _index, methods=["GET"], include_in_schema=False)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        for name, media in (("sirio.ico", "image/x-icon"), ("favicon.svg", "image/svg+xml")):
            path = WEB_DIR / "assets" / name
            if path.is_file():
                return FileResponse(path, media_type=media, headers={"Cache-Control": "no-cache"})
        return Response(status_code=204)

    # ------------------------------------------------------------ stato
    @app.get("/api/health")
    def health() -> JSONResponse:
        return _json({"ok": True, "version": __version__})

    @app.get("/api/state")
    def state(since: str | None = None) -> JSONResponse:
        revision = store.revision
        if since is not None and since.strip().lstrip("-").isdigit() and int(since) == revision:
            return _json({"revision": revision, "unchanged": True})
        docs = store.list(copy=False)
        return _json({
            "revision": revision,
            "documents": [doc_summary(d) for d in docs],
            "queue": processor.status(),
            "kpi": compute_kpi(docs),
        })

    # ------------------------------------------------------------ importazione
    @app.post("/api/upload")
    async def upload(request: Request) -> JSONResponse:
        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" not in content_type.lower():
            raise ApiError(415, "Caricamento non valido: inviare i file come multipart/form-data nel campo «files».")
        try:
            form = await request.form(max_files=MAX_UPLOAD_FILES, max_fields=MAX_UPLOAD_FILES)
        except Exception as exc:  # noqa: BLE001 - errori del parser multipart
            log.warning("Caricamento non leggibile: %s", exc)
            raise ApiError(400, "Caricamento non riuscito: i dati ricevuti sono incompleti o non validi.") from exc
        documents: dict[str, dict] = {}
        errors: list[dict] = []
        duplicates: list[dict] = []
        try:
            items = [f for key in ("files", "file") for f in form.getlist(key)]
            uploads = [f for f in items if isinstance(f, UploadFile)]
            if not uploads:
                raise ApiError(400, "Nessun file ricevuto: selezionare uno o più PDF o immagini.")
            for up in uploads:
                name = os.path.basename((up.filename or "").replace("\\", "/")) or "file senza nome"
                if not pdf_io.is_supported(name):
                    ext = pdf_io.file_extension(name)
                    errors.append({
                        "file": name,
                        "message": (f"Formato «{ext}» non supportato" if ext else "Formato non riconosciuto")
                        + ": sono accettati PDF e immagini JPG, PNG, TIFF, BMP o WEBP.",
                    })
                    continue
                size = up.size
                if size is not None and size > MAX_UPLOAD_BYTES:
                    errors.append({"file": name, "message": f"Il file «{name}» supera il limite di 200 MB."})
                    continue
                data = await up.read(MAX_UPLOAD_BYTES + 1)
                if len(data) > MAX_UPLOAD_BYTES:
                    errors.append({"file": name, "message": f"Il file «{name}» supera il limite di 200 MB."})
                    continue
                created: list[str] = []

                def _on_new(doc: Document, _created: list[str] = created) -> None:
                    _created.append(doc.id)
                    processor.enqueue(doc.id)

                try:
                    docs = await run_in_threadpool(store.add_file, name, data, _on_new)
                except PartialImportError as exc:
                    docs = exc.documents
                    errors.append({"file": name, "message": str(exc)})
                except ValueError as exc:
                    errors.append({"file": name, "message": str(exc)})
                    continue
                except Exception:  # noqa: BLE001
                    log.exception("Importazione di «%s» non riuscita", name)
                    errors.append({
                        "file": name,
                        "message": f"Impossibile importare «{name}»: errore imprevisto (dettagli nel registro sirio.log).",
                    })
                    continue
                finally:
                    del data
                if docs and not created:
                    duplicates.append({
                        "file": name,
                        "message": f"«{name}» era già stato importato: nessun duplicato creato.",
                    })
                for d in docs:
                    current = store.get(d.id) or d
                    documents[d.id] = doc_summary(current)
        finally:
            await form.close()
        return _json({"documents": list(documents.values()), "errors": errors, "duplicates": duplicates})

    # ------------------------------------------------------------ documenti
    @app.get("/api/documents")
    def list_documents() -> JSONResponse:
        return _json([doc_summary(d) for d in store.list(copy=False)])

    @app.delete("/api/documents")
    def delete_all() -> JSONResponse:
        processor.discard_all()
        store.clear()
        pages.drop()
        return _json({"ok": True})

    @app.get("/api/documents/{doc_id}")
    def get_document(doc_id: str) -> JSONResponse:
        doc = _doc_or_404(doc_id)
        grid = store.load_grid(doc_id)
        return _json(document_payload(doc, grid.to_dict() if grid else None))

    @app.put("/api/documents/{doc_id}")
    def put_document(doc_id: str, payload: dict[str, Any] = Body(...)) -> JSONResponse:  # noqa: B008
        _doc_or_404(doc_id)
        updated = store.update(doc_id, lambda d: apply_document_update(d, payload))
        if updated is None:
            raise ApiError(404, "Documento non trovato: potrebbe essere stato eliminato.")
        grid = store.load_grid(doc_id)
        return _json(document_payload(updated, grid.to_dict() if grid else None))

    @app.post("/api/documents/{doc_id}/reprocess")
    def reprocess(doc_id: str) -> JSONResponse:
        _doc_or_404(doc_id)
        processor.enqueue(doc_id)
        return _json(doc_summary(_doc_or_404(doc_id)))

    @app.delete("/api/documents/{doc_id}")
    def delete_document(doc_id: str) -> JSONResponse:
        processor.discard(doc_id)
        if not store.delete(doc_id):
            raise ApiError(404, "Documento non trovato: potrebbe essere già stato eliminato.")
        pages.drop(doc_id)
        return _json({"ok": True})

    @app.get("/api/documents/{doc_id}/image")
    def image(doc_id: str, max_px: int | None = Query(None, alias="max")) -> Response:  # noqa: B008
        _doc_or_404(doc_id)
        path = store.page_path(doc_id)
        if not path.is_file():
            raise ApiError(404, "Immagine della pagina non disponibile.")
        if max_px is None:
            return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": _IMAGE_CACHE})
        if not 32 <= max_px <= 10000:
            raise ApiError(422, "Dimensione richiesta non valida (ammessa tra 32 e 10000 pixel).")
        try:
            img = pages.get(store, doc_id)
        except (FileNotFoundError, DocumentNotFound):
            raise ApiError(404, "Immagine della pagina non disponibile.") from None
        if max(img.shape[:2]) <= max_px:
            return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": _IMAGE_CACHE})
        data = pdf_io.encode_jpeg(img, quality=88, max_side=max_px)
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": _IMAGE_CACHE})

    @app.get("/api/documents/{doc_id}/thumb")
    def thumb(doc_id: str) -> Response:
        _doc_or_404(doc_id)
        path = store.thumb_path(doc_id)
        if not path.is_file():
            raise ApiError(404, "Miniatura non disponibile.")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": _IMAGE_CACHE})

    @app.get("/api/documents/{doc_id}/crop")
    def crop(doc_id: str, x0: int, y0: int, x1: int, y1: int, scale: float = 2.0) -> Response:
        _doc_or_404(doc_id)
        if not math.isfinite(scale) or not 0.1 <= scale <= 6.0:
            raise ApiError(422, "Fattore di ingrandimento non valido (ammesso tra 0,1 e 6).")
        try:
            img = pages.get(store, doc_id)
        except (FileNotFoundError, DocumentNotFound):
            raise ApiError(404, "Immagine della pagina non disponibile.") from None
        h, w = img.shape[:2]
        xa, xb = sorted((x0, x1))
        ya, yb = sorted((y0, y1))
        xa, xb = max(0, xa), min(w, xb)
        ya, yb = max(0, ya), min(h, yb)
        if xb - xa < 1 or yb - ya < 1:
            raise ApiError(422, "Area di ritaglio non valida: è fuori dalla pagina o vuota.")
        part = img[ya:yb, xa:xb]
        # limite all'immagine prodotta (al massimo 4000 px per lato)
        scale = min(scale, 4000.0 / max(xb - xa, yb - ya))
        if abs(scale - 1.0) > 1e-3:
            interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            part = cv2.resize(
                part,
                (max(1, round((xb - xa) * scale)), max(1, round((yb - ya) * scale))),
                interpolation=interp,
            )
        data = pdf_io.encode_jpeg(part, quality=90)
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": _IMAGE_CACHE})

    @app.get("/api/preview")
    def preview(ids: str | None = None, giorni_vuoti: bool = False) -> JSONResponse:
        docs = store.list(copy=False)
        if ids:
            wanted = [i for i in ids.split(",") if i.strip()]
            by_id = {d.id: d for d in docs}
            docs = [by_id[i.strip()] for i in wanted if i.strip() in by_id]
        else:
            docs = [d for d in docs if _exportable(d)]
        return _json(preview_payload(docs, giorni_vuoti=giorni_vuoti))

    # ------------------------------------------------------------ impostazioni
    def _engines() -> dict:
        out = {}
        for kind in ("locale", "claude"):
            try:
                ok, message = processor.engine_status(kind)
            except Exception:  # noqa: BLE001
                log.exception("Verifica del motore %s non riuscita", kind)
                ok, message = False, "Impossibile verificare il motore (dettagli nel registro sirio.log)."
            out[kind] = {"available": ok, "message": message or ("Pronto" if ok else "Non disponibile")}
        return out

    def _settings_payload() -> dict:
        settings = config.load_settings()
        return {
            "settings": settings.model_dump(mode="json"),
            "api_key_set": bool(config.get_api_key()),
            "api_key_hint": config.api_key_hint(),
            "api_key_source": config.api_key_source(),
            "engines": _engines(),
            "models": config.CLAUDE_MODELS,
            "data_dir": str(config.data_dir()),
            "export_dir": str(config.export_dir()),
            "version": __version__,
        }

    @app.get("/api/settings")
    def get_settings() -> JSONResponse:
        return _json(_settings_payload())

    @app.put("/api/settings")
    def put_settings(payload: dict[str, Any] = Body(...)) -> JSONResponse:  # noqa: B008
        data = dict(payload)
        if isinstance(data.get("settings"), dict):  # accetta anche {"settings": {...}, "api_key": ...}
            data = {**data.pop("settings"), **data}
        api_key = data.pop("api_key", None)
        current = config.load_settings()
        updates = {k: v for k, v in data.items() if k in config.Settings.model_fields}
        if "concorrenza" in updates:
            c = updates["concorrenza"]
            if isinstance(c, bool) or not isinstance(c, (int, float, str)) or not str(c).strip().isdigit() \
                    or not 1 <= int(str(c).strip()) <= 8:
                raise ApiError(422, "Concorrenza non valida: indicare un numero da 1 a 8.")
            updates["concorrenza"] = int(str(c).strip())
        if "claude_model" in updates and updates["claude_model"] not in {m["id"] for m in config.CLAUDE_MODELS}:
            raise ApiError(422, "Modello Claude non riconosciuto.")
        try:
            new = config.Settings.model_validate({**current.model_dump(), **updates})
        except ValidationError as exc:
            campi = ", ".join(sorted({str(e["loc"][0]) for e in exc.errors() if e.get("loc")}))
            raise ApiError(422, f"Impostazioni non valide: controllare {campi or 'i valori inseriti'}.") from None
        key: str | None = None
        if api_key is not None:
            if not isinstance(api_key, str):
                raise ApiError(422, "Chiave API non valida.")
            key = api_key.strip()
            if key and (len(key) < 20 or len(key) > 400 or not key.isascii() or any(ch.isspace() for ch in key)):
                raise ApiError(
                    422,
                    "La chiave API non sembra valida: copiarla per intero dalla console di Anthropic "
                    "(inizia con «sk-ant-»).",
                )
        try:
            config.save_settings(new)
            if api_key is not None:
                config.set_api_key(key or None)
        except OSError as exc:
            log.exception("Salvataggio delle impostazioni non riuscito")
            raise ApiError(500, f"Impossibile salvare le impostazioni: {exc.strerror or exc}.") from None
        processor.kick()
        return _json(_settings_payload())

    @app.post("/api/settings/test")
    def test_settings(body: KeyTestRequest | None = Body(None)) -> JSONResponse:  # noqa: B008
        settings = config.load_settings()
        key = (body.api_key or "").strip() if body and body.api_key else (config.get_api_key() or "")
        model = (body.model if body and body.model else None) or settings.claude_model
        if not key:
            return _json({"ok": False, "message": "Nessuna chiave API configurata: inserirla e premere «Verifica»."})
        try:
            from sirio.engines.claude_engine import test_api_key  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            log.exception("Motore Claude non importabile")
            return _json({"ok": False, "message": "Il motore Claude non è disponibile in questa installazione."})
        try:
            ok, message = test_api_key(key, model)
        except Exception:  # noqa: BLE001
            log.exception("Verifica della chiave API non riuscita")
            return _json({
                "ok": False,
                "message": "Verifica non riuscita per un errore imprevisto (dettagli nel registro sirio.log).",
            })
        return _json({"ok": bool(ok), "message": str(message or ("Chiave valida." if ok else "Chiave non valida."))})

    # ------------------------------------------------------------ esportazione
    @app.post("/api/export")
    def export(body: ExportRequest | None = Body(None)) -> JSONResponse:  # noqa: B008
        try:
            from sirio import excel_export  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            log.exception("Modulo di esportazione non importabile")
            raise ApiError(503, "La generazione dell'Excel non è disponibile in questa installazione.") from None
        body = body or ExportRequest()
        all_docs = store.list()
        if body.ids:
            by_id = {d.id: d for d in all_docs}
            docs = [by_id[i] for i in dict.fromkeys(body.ids) if i in by_id]
            if not docs:
                raise ApiError(404, "Nessuno dei documenti selezionati è disponibile.")
        else:
            docs = [d for d in all_docs if _exportable(d)]
            if not docs:
                raise ApiError(
                    400,
                    "Nessun foglio firma completato da esportare: attendere la fine della lettura dei documenti.",
                )
        settings = config.load_settings()
        options_data = {
            "fogli_per_documento": settings.export_fogli_per_documento,
            "giorni_vuoti": settings.export_giorni_vuoti,
            **(body.options or {}),
        }
        try:
            options = excel_export.ExportOptions.model_validate(options_data)
        except ValidationError:
            raise ApiError(422, "Opzioni di esportazione non valide.") from None
        folder = config.export_dir()
        filename = safe_export_name(body.filename) or safe_export_name(excel_export.default_filename(docs))
        path = _unique_path(folder, filename or "Rendicontazione.xlsx")
        try:
            written = Path(excel_export.export_workbook(docs, path, options))
        except ValueError as exc:
            raise ApiError(400, str(exc)) from None
        except OSError as exc:
            log.exception("Scrittura dell'Excel non riuscita")
            raise ApiError(500, str(exc) or "Impossibile scrivere il file Excel.") from None
        try:
            written.resolve().relative_to(folder.resolve())
        except ValueError:
            log.error("File esportato fuori dalla cartella export: %s", written)
            raise ApiError(500, "Il file Excel è stato scritto in una cartella inattesa.") from None
        entry = _export_entry(written)
        entry["documenti"] = sum(1 for d in docs if _exportable(d))
        return _json(entry)

    @app.get("/api/exports")
    def list_exports() -> JSONResponse:
        folder = config.export_dir()
        entries = []
        for path in folder.glob("*.xlsx"):
            if path.name.startswith((".", "~$")) or not path.is_file():
                continue
            try:
                entries.append(_export_entry(path))
            except OSError:
                continue
        entries.sort(key=lambda e: e["created_at"], reverse=True)
        return _json(entries)

    @app.get("/api/exports/{filename}")
    def download_export(filename: str) -> Response:
        path = resolve_export_file(filename)
        return FileResponse(path, media_type=XLSX_MEDIA_TYPE, filename=path.name,
                            headers={"Cache-Control": "no-store"})

    @app.post("/api/open")
    def open_target(body: OpenRequest) -> JSONResponse:
        if body.target == "export_dir":
            path = config.export_dir()
        else:
            if not body.filename:
                raise ApiError(422, "Indicare il file da aprire.")
            path = resolve_export_file(body.filename)
        try:
            open_path(path)
        except FileNotFoundError:
            raise ApiError(
                500, "Impossibile aprire: nessun programma predefinito disponibile per questo tipo di file."
            ) from None
        except OSError as exc:
            log.warning("Apertura di %s non riuscita: %s", path, exc)
            raise ApiError(500, f"Impossibile aprire «{path.name}»: {exc.strerror or exc}.") from None
        return _json({"ok": True, "path": str(path)})

    # ------------------------------------------------------------ ciclo di vita
    @app.post("/api/heartbeat")
    def heartbeat() -> JSONResponse:
        _call(on_heartbeat, "heartbeat")
        return _json({"ok": True})

    @app.post("/api/bye")
    def bye() -> JSONResponse:
        _call(on_bye, "chiusura della finestra")
        return _json({"ok": True})

    @app.post("/api/shutdown")
    def shutdown() -> JSONResponse:
        if on_shutdown is None:
            raise ApiError(404, "Funzione non disponibile.")
        threading.Timer(0.2, _call, args=(on_shutdown, "arresto")).start()
        return _json({"ok": True})

    return app


__all__ = [
    "MAX_UPLOAD_BYTES",
    "WEB_DIR",
    "apply_document_update",
    "compute_kpi",
    "create_app",
    "doc_summary",
    "open_path",
    "resolve_export_file",
    "safe_export_name",
]
