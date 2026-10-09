"""Motore OCR Claude Vision (cloud, facoltativo, a pagamento, massima precisione).

Pipeline di una pagina
----------------------
1. **Immagini**: dalla pagina normalizzata e dalla ``TableGrid`` si ricavano la
   pagina intera, l'intestazione del modulo e due ritagli ad alta risoluzione
   della tabella (intestazione + giorni 1-16; giorni 16-31 + totale + pie' di
   pagina). Ogni immagine e' ridotta entro i limiti dell'API (lato lungo
   <= 2576 px e <= 3,75 megapixel) e codificata in JPEG (qualita' 90). Se la
   griglia non e' stata individuata si usano ritagli proporzionali generosi.
2. **Lettura completa**: una richiesta in streaming con output strutturato
   (``prompts.SCHEMA``), thinking adattivo (parametro omesso), effort dalle
   impostazioni, fallback lato server per Opus 5.5 / Sonnet 5.5.
3. **Normalizzazione**: il JSON viene validato (pydantic) e convertito in
   ``Header``/``DayRow``: orari "HH:MM", ore decimali, testi ripuliti, mese e
   anno da "MM/AAAA", sempre 31 righe; i valori scritti ma non interpretabili
   restano come scritti (li segnala la validazione) e sono marcati incerti.
4. **Verifica incrociata** (facoltativa): ``validation.validate`` sul primo
   risultato; le righe con anomalie di coerenza, campi incerti o illeggibili
   (e, se il totale mensile non torna, tutte le righe con ore) vengono rilette
   con una seconda richiesta che contiene solo le strisce di riga ingrandite.
   Regole di fusione (``merge_row``):

   * illeggibile in entrambe le letture -> ``illeggibili`` (valore ``None``);
   * illeggibile solo nella rilettura -> resta il valore della prima, incerto;
   * valori concordi -> il campo esce da ``incerti`` (resta incerto solo se la
     rilettura lo segnala ancora *e* la riga risulta incoerente);
   * valori discordi (o illeggibile solo nella prima) -> vale la rilettura
     mirata; il campo resta incerto, salvo quando la rilettura non lo segnala
     come dubbio e la riga risultante e' coerente (nessuna anomalia
     E01/E03/W01/W02/W05/W11); se una sola delle due letture vede scrittura
     nella cella, il campo resta comunque incerto.

   Il totale mensile, quando riletto, segue le stesse regole (coerente = pari
   alla somma delle ore giornaliere). Un errore nella verifica non fa fallire
   il documento: resta la prima lettura, con una nota.
5. **Consumi**: token e costo stimato (prezzi di ``config.CLAUDE_MODELS``,
   cache dei prompt inclusa) sommati su tutte le richieste, piu' i secondi.

Il motore e' privo di stato per documento ed e' quindi utilizzabile da piu'
thread contemporaneamente. La chiave API non viene mai registrata nei log.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import re
import threading
import time
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Any

import anthropic
import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from sirio.config import CLAUDE_MODELS, model_info
from sirio.engines import prompts
from sirio.engines.base import EngineError, PageInput, ProgressFn, no_progress
from sirio.models import DAY_FIELDS, HEADER_FIELDS, DayRow, ExtractionResult, Header, Usage
from sirio.pdf_io import encode_jpeg
from sirio.validation import (
    E01_ORE_NON_COERENTI,
    E02_TOTALE_MENSILE_DIVERSO,
    E03_ORARIO_NON_VALIDO,
    E05_CAMPO_ILLEGGIBILE,
    TOLLERANZA,
    W01_FIRMA_MANCANTE,
    W02_ORE_SENZA_ORARIO,
    W04_CAMPO_INCERTO,
    W05_ASSENZA_OPERATORE_CON_ORE,
    W11_DOPPIA_ASSENZA,
    etichetta_campo,
    normalize_time,
    ore_riconosciute,
    parse_hours,
    validate,
)
from sirio.vision.grid import N_DAYS, TableGrid, template_grid

log = logging.getLogger(__name__)

# --- Parametri dell'API ---------------------------------------------------------

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "high"
EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
# Fallback lato server ("default") solo per questi modelli (NON per Haiku 5.5).
FALLBACK_MODELS: frozenset[str] = frozenset({"claude-opus-5-5", "claude-sonnet-5-5"})
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOKENS = 32000
MAX_TOKENS_RETRY = 64000
MAX_RETRIES = 4
REQUEST_TIMEOUT = 300.0
CONNECT_TIMEOUT = 15.0

# --- Limiti delle immagini -------------------------------------------------------

MAX_LONG_EDGE = 2576
MAX_PIXELS = 3_750_000
JPEG_QUALITY = 90
STRIP_UPSCALE = 2.0            # ingrandimento delle strisce di riga nella verifica

# Codici che rendono una riga meritevole di rilettura mirata.
VERIFY_CODES: frozenset[str] = frozenset(
    {
        E01_ORE_NON_COERENTI,
        E03_ORARIO_NON_VALIDO,
        E05_CAMPO_ILLEGGIBILE,
        W01_FIRMA_MANCANTE,
        W02_ORE_SENZA_ORARIO,
        W04_CAMPO_INCERTO,
        W05_ASSENZA_OPERATORE_CON_ORE,
        W11_DOPPIA_ASSENZA,
    }
)
# Codici di (in)coerenza di una riga usati per decidere la fusione.
COHERENCE_CODES: frozenset[str] = frozenset(
    {
        E01_ORE_NON_COERENTI,
        E03_ORARIO_NON_VALIDO,
        W01_FIRMA_MANCANTE,
        W02_ORE_SENZA_ORARIO,
        W05_ASSENZA_OPERATORE_CON_ORE,
        W11_DOPPIA_ASSENZA,
    }
)

_TIME_FIELDS = ("prog_entrata", "prog_uscita", "eff_entrata", "eff_uscita")
_ROW_BOOL_FIELDS = frozenset({"assenza_alunno", "assenza_operatore", "firma"})
_HEADER_BOOL_FIELDS = frozenset({"firma_coordinatore", "timbro_referente"})
_TOTAL = "totale_mensile_dichiarato"

_MSG_NO_KEY = "Inserisci la chiave API di Anthropic nelle Impostazioni."
_MSG_REFUSAL = (
    "Claude ha rifiutato di leggere questa pagina (blocco di sicurezza del servizio). "
    "Riprova più tardi oppure elabora il documento con il motore locale."
)
_MSG_TRUNCATED = (
    "La risposta di Claude è stata interrotta per eccesso di lunghezza anche al secondo tentativo. "
    "Riprova con un livello di accuratezza (effort) più basso nelle Impostazioni."
)
_MSG_BAD_JSON = (
    "Claude ha restituito una risposta non interpretabile per questa pagina. "
    "Riprova a elaborare il documento."
)


# ==========================================================================
# Errori dell'SDK -> messaggi in italiano
# ==========================================================================

_KEY_RE = re.compile(r"sk-ant-[A-Za-z0-9_\-]+")


def _error_detail(exc: BaseException) -> str:
    """Messaggio sintetico dell'errore restituito dal servizio (senza segreti)."""
    detail = ""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            detail = err["message"]
        elif isinstance(body.get("message"), str):
            detail = body["message"]
    if not detail:
        detail = str(getattr(exc, "message", "") or exc or "")
    detail = _KEY_RE.sub("sk-ant-…", " ".join(detail.split()))
    return detail[:300].rstrip()


def map_api_error(exc: BaseException, model: str | None = None) -> EngineError:
    """Converte un'eccezione dell'SDK Anthropic in ``EngineError`` con un messaggio
    comprensibile (gli specifici prima dei generici)."""
    detail = _error_detail(exc)
    status = getattr(exc, "status_code", None)
    log.warning("Errore del servizio Claude: %s (codice %s): %s", type(exc).__name__, status, detail)
    if isinstance(exc, anthropic.AuthenticationError):
        return EngineError(
            "Chiave API di Anthropic non valida o revocata: controllala o sostituiscila nelle Impostazioni."
        )
    if isinstance(exc, anthropic.PermissionDeniedError):
        return EngineError(
            "La chiave API non ha i permessi per usare Claude (accesso negato). "
            "Verifica l'account e l'area di lavoro su console.anthropic.com."
        )
    if isinstance(exc, anthropic.NotFoundError):
        nome = f"«{model}» " if model else ""
        return EngineError(
            f"Il modello {nome}non è disponibile per questo account: scegline un altro nelle Impostazioni."
        )
    if isinstance(exc, anthropic.RateLimitError):
        return EngineError(
            "Limite di utilizzo dell'account Anthropic raggiunto. Riprova tra qualche minuto oppure riduci "
            "i documenti elaborati in parallelo nelle Impostazioni."
        )
    too_large = getattr(anthropic, "RequestTooLargeError", None)
    if (too_large is not None and isinstance(exc, too_large)) or status == 413:
        return EngineError(
            "Le immagini di questa pagina sono troppo grandi per il servizio Claude: "
            "riprova con una scansione a risoluzione più bassa."
        )
    if isinstance(exc, anthropic.BadRequestError):
        testo = f"{detail} {exc}".lower()
        if "credit balance" in testo or "credit_balance" in testo:
            return EngineError(
                "Credito Anthropic esaurito: ricarica il credito su console.anthropic.com "
                "(sezione Billing) e poi riprova."
            )
        return EngineError(f"Richiesta non accettata dal servizio Claude: {detail or 'errore 400'}.")
    if isinstance(exc, anthropic.APIStatusError):
        if status is not None and (status >= 500 or status == 529):
            return EngineError(
                "Il servizio Claude è momentaneamente sovraccarico o non disponibile. "
                "Riprova tra qualche minuto."
            )
        return EngineError(f"Errore del servizio Claude (codice {status}): {detail or 'nessun dettaglio'}.")
    if isinstance(exc, anthropic.APITimeoutError):
        return EngineError(
            "Il servizio Claude non ha risposto in tempo. Controlla la connessione a Internet e riprova."
        )
    if isinstance(exc, anthropic.APIConnectionError):
        return EngineError(
            "Connessione a Internet assente o servizio Claude non raggiungibile: controlla la connessione "
            "(ed eventuali proxy o firewall) e riprova."
        )
    return EngineError(f"Errore del servizio Claude: {detail or type(exc).__name__}.")


# ==========================================================================
# Prezzi e consumi
# ==========================================================================

def _prices(model_id: str | None, fallback_model: str = DEFAULT_MODEL) -> tuple[float, float]:
    """(input, output) in USD per milione di token; per modelli non in elenco si
    usa il modello della stessa famiglia (stima)."""
    info = model_info(model_id or "")
    if info is None and model_id:
        for family in ("opus", "sonnet", "haiku"):
            if family in model_id:
                info = next((m for m in CLAUDE_MODELS if family in m["id"]), None)
                break
    if info is None:
        info = model_info(fallback_model) or CLAUDE_MODELS[0]
    return float(info["input"]), float(info["output"])


def _tok(obj: Any, name: str) -> int:
    value = getattr(obj, name, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


class _UsageMeter:
    """Somma token e costo stimato di tutte le risposte di un documento."""

    def __init__(self, model: str):
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.served_by: set[str] = set()

    def _add(self, usage: Any, model_id: str | None) -> None:
        inp = _tok(usage, "input_tokens")
        out = _tok(usage, "output_tokens")
        created = _tok(usage, "cache_creation_input_tokens")
        read = _tok(usage, "cache_read_input_tokens")
        price_in, price_out = _prices(model_id, self.model)
        self.input_tokens += inp + created + read
        self.output_tokens += out
        self.cost_usd += ((inp + 1.25 * created + 0.1 * read) * price_in + out * price_out) / 1e6

    def add(self, msg: Any) -> None:
        served = getattr(msg, "model", None)
        if isinstance(served, str) and served:
            self.served_by.add(served)
        usage = getattr(msg, "usage", None)
        if usage is None:
            return
        # Con il fallback lato server ``usage.iterations`` riporta ogni tentativo
        # (anche quello rifiutato) ed e' la fonte corretta; altrimenti il totale.
        iterations = getattr(usage, "iterations", None)
        attempts = [
            it for it in iterations if getattr(it, "type", None) in ("message", "fallback_message")
        ] if isinstance(iterations, (list, tuple)) else []
        if attempts:
            for it in attempts:
                model_id = getattr(it, "model", None)
                self._add(it, model_id if isinstance(model_id, str) else self.model)
        else:
            self._add(usage, served if isinstance(served, str) and served else self.model)

    def usage(self, seconds: float) -> Usage:
        return Usage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost_usd=round(self.cost_usd, 6),
            seconds=round(max(0.0, seconds), 2),
        )


# ==========================================================================
# Immagini
# ==========================================================================

Box = tuple[int, int, int, int]


def fit_to_limits(img: np.ndarray, max_upscale: float = 1.0) -> np.ndarray:
    """Ridimensiona entro i limiti dell'API (lato lungo e megapixel); ingrandisce
    al massimo di ``max_upscale`` (1.0 = mai)."""
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Immagine vuota.")
    scale = min(float(max_upscale), MAX_LONG_EDGE / float(max(h, w)), math.sqrt(MAX_PIXELS / float(h * w)))
    if abs(scale - 1.0) < 1e-3 and max(h, w) <= MAX_LONG_EDGE and h * w <= MAX_PIXELS:
        return img
    nw = max(1, int(math.floor(w * scale)))
    nh = max(1, int(math.floor(h * scale)))
    while nw * nh > MAX_PIXELS or max(nw, nh) > MAX_LONG_EDGE:   # difesa contro gli arrotondamenti
        nw, nh = max(1, nw - 1), max(1, nh - 1)
    if (nw, nh) == (w, h):
        return img
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img, (nw, nh), interpolation=interp)


def _clip_box(box: tuple[float, float, float, float], w: int, h: int) -> Box:
    x0, y0, x1, y1 = box
    xa = max(0, min(w - 1, int(math.floor(min(x0, x1)))))
    ya = max(0, min(h - 1, int(math.floor(min(y0, y1)))))
    xb = max(xa + 1, min(w, int(math.ceil(max(x0, x1)))))
    yb = max(ya + 1, min(h, int(math.ceil(max(y0, y1)))))
    return xa, ya, xb, yb


def _crop(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = _clip_box(box, w, h)
    return np.ascontiguousarray(img[y0:y1, x0:x1])


def _as_bgr(img: Any) -> np.ndarray:
    if not isinstance(img, np.ndarray) or img.size == 0 or img.ndim not in (2, 3):
        raise EngineError("Immagine della pagina non valida: impossibile inviarla a Claude.")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _grid_usable(grid: TableGrid | None, w: int, h: int) -> bool:
    """True se la griglia e' stata rilevata davvero e ha una geometria sensata."""
    if grid is None or not grid.detected:
        return False
    try:
        if len(grid.col_x) != 11 or len(grid.row_y) != N_DAYS + 1:
            return False
        if any(b <= a for a, b in zip(grid.row_y, grid.row_y[1:], strict=False)):
            return False
        if any(b <= a for a, b in zip(grid.col_x, grid.col_x[1:], strict=False)):
            return False
        rh = (grid.row_y[-1] - grid.row_y[0]) / float(N_DAYS)
        return rh >= 0.004 * h and grid.col_x[-1] - grid.col_x[0] >= 0.3 * w and grid.row_y[-1] <= h + rh
    except (TypeError, ValueError, AttributeError):
        return False


def _page_and_grid(page: PageInput) -> tuple[np.ndarray, TableGrid, bool]:
    """Immagine BGR, griglia nelle sue coordinate e se la griglia e' affidabile."""
    img = _as_bgr(page.image)
    h, w = img.shape[:2]
    grid = page.grid
    if grid is not None and (grid.width != w or grid.height != h) and grid.width > 0 and grid.height > 0:
        grid = grid.scaled(w / float(grid.width), h / float(grid.height))
        grid.width, grid.height = w, h
    ok = _grid_usable(grid, w, h)
    if grid is None or not ok:
        grid = template_grid(w, h)
    return img, grid, ok


def _row_h(grid: TableGrid) -> float:
    return (grid.row_y[-1] - grid.row_y[0]) / float(N_DAYS)


def first_pass_crops(img: np.ndarray, grid: TableGrid, detected: bool) -> list[tuple[str, np.ndarray]]:
    """Immagini della lettura completa: pagina intera, intestazione del modulo,
    tabella giorni 1-16 (con intestazione) e giorni 16-31 con totale e pie' di pagina."""
    h, w = img.shape[:2]
    if detected:
        rh = _row_h(grid)
        tx0 = grid.col_x[0] - 0.03 * w
        tx1 = grid.col_x[-1] + 0.03 * w
        intestazione = (0, 0, w, grid.header_top + 0.3 * rh)
        alto = (tx0, grid.header_top - 0.4 * rh, tx1, grid.row_y[16] + 0.3 * rh)
        fondo_tabella = grid.total_row[1] if grid.total_row else grid.row_y[-1] + 0.9 * rh
        basso = (0, grid.row_y[15] - 0.3 * rh, w, fondo_tabella + 3.2 * rh)
    else:
        intestazione = (0, 0, w, 0.22 * h)
        alto = (0, 0.13 * h, w, 0.62 * h)
        basso = (0, 0.52 * h, w, h)
    return [
        ("pagina intera (per orientarsi)", img),
        ("intestazione del modulo, ingrandita (anno scolastico, lotto, municipalità, ente, istituto, "
         "operatore, alunno, mese/anno, ore da PEI, sostituzione)", _crop(img, intestazione)),
        ("tabella: intestazione delle colonne e giorni da 1 a 16, ad alta risoluzione", _crop(img, alto)),
        ("tabella: giorni da 16 a 31, riga \"Totale ore effettive mensili\" con la firma del coordinatore, "
         "data \"Napoli, __/__/____\" e spazio per timbro e firma del referente scolastico, ad alta risoluzione",
         _crop(img, basso)),
    ]


def verification_crops(img: np.ndarray, grid: TableGrid, detected: bool, days: list[int],
                       include_total: bool) -> list[tuple[str, np.ndarray]]:
    """Strisce della verifica incrociata: intestazione della tabella, una striscia
    per giorno (con margine verticale) e, se richiesta, la riga del totale."""
    h, w = img.shape[:2]
    rh = _row_h(grid)
    vpad = (0.35 if detected else 1.0) * rh
    tx0 = grid.col_x[0] - 0.015 * w
    tx1 = grid.col_x[-1] + 0.015 * w
    out: list[tuple[str, np.ndarray]] = [
        ("intestazione della tabella (nomi delle colonne)",
         _crop(img, (tx0, grid.header_top - 0.15 * rh, tx1, grid.row_y[0] + 0.1 * rh))),
    ]
    for d in days:
        out.append((f"giorno {d}", _crop(img, (tx0, grid.row_y[d - 1] - vpad, tx1, grid.row_y[d] + vpad))))
    if include_total:
        if grid.total_row:
            y0, y1 = grid.total_row[0] - 0.25 * rh, grid.total_row[1] + 0.4 * rh
        else:
            y0, y1 = grid.row_y[-1] - 0.25 * rh, grid.row_y[-1] + 1.4 * rh
        out.append(("riga \"Totale ore effettive mensili\"", _crop(img, (tx0, y0, tx1, y1))))
    return out


def _image_block(img: np.ndarray, max_upscale: float = 1.0) -> dict[str, Any]:
    fitted = fit_to_limits(img, max_upscale)
    data = encode_jpeg(fitted, quality=JPEG_QUALITY)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(data).decode("ascii"),
        },
    }


# ==========================================================================
# Normalizzazione della risposta
# ==========================================================================

_DASH_RE = re.compile(r"^[\s\-–—−_~=.]+$")
_NULL_WORDS = {"null", "none", "n/a", "nd", "n.d."}


def _clean(value: Any, max_len: int = 200) -> str | None:
    """Testo ripulito (spazi compressi) oppure None se vuoto."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    s = unicodedata.normalize("NFKC", str(value))
    s = "".join(ch for ch in s if ch.isprintable() or ch.isspace())
    s = re.sub(r"\s+", " ", s).strip()
    if not s or s.lower() in _NULL_WORDS:
        return None
    return s[:max_len].rstrip()


def _is_dash(s: str) -> bool:
    return bool(_DASH_RE.match(s)) and any(ch in s for ch in "-–—−_~")


def _upper(value: Any, max_len: int = 200) -> str | None:
    s = _clean(value, max_len)
    if s is None or _is_dash(s):
        return None
    return s.upper()


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"true", "si", "sì", "x", "1", "yes"}
    return False


def _flag_list(values: Iterable[Any], allowed: Iterable[str]) -> list[str]:
    """Nomi di campo ammessi, senza duplicati (l'output strutturato non garantisce
    le maiuscole dei valori enum)."""
    ok = {a.lower(): a for a in allowed}
    out: list[str] = []
    for v in values or []:
        name = ok.get(v.strip().lower()) if isinstance(v, str) else None
        if name is not None and name not in out:
            out.append(name)
    return out


def _norm_time(value: Any) -> tuple[str | None, bool, bool]:
    """(orario "HH:MM" o testo come scritto, trattino?, non interpretabile?)."""
    s = _clean(value, 20)
    if s is None:
        return None, False, False
    if _is_dash(s):
        return None, True, False
    t = normalize_time(s)
    if t is not None:
        return t, False, False
    return s, False, True


def _norm_hours(value: Any) -> tuple[float | None, bool]:
    """(ore decimali, scritto ma non interpretabile?)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        v = parse_hours(float(value))
        return (None, True) if v is None else (round(v, 2), False)
    s = _clean(value, 20)
    if s is None or _is_dash(s):
        return None, False
    v = parse_hours(s)
    return (None, True) if v is None else (round(v, 2), False)


def _parse_number(value: Any, max_value: float) -> tuple[float | None, bool]:
    """Numero di ore anche oltre le 24 (totale mensile, ore PEI settimanali):
    (valore, scritto ma non interpretabile?)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        v = float(value)
        return (round(v, 2), False) if math.isfinite(v) and 0 <= v <= max_value else (None, True)
    s = _clean(value, 30)
    if s is None or _is_dash(s):
        return None, False
    t = s.lower()
    t = re.sub(r"\b(?:ore|ora|h|hh|tot\.?|totale)\b", " ", t)
    t = re.sub(r"(?<=\d)\s*(?:ore|h)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" .;:")
    v: float | None
    decimale = re.fullmatch(r"(\d{1,3})(?:\s*[.,]\s*(\d{1,2}))?", t)
    ore_minuti = re.fullmatch(r"(\d{1,3})\s*[:h]\s*(\d{2})", t)
    if decimale:
        v = float(f"{decimale.group(1)}.{decimale.group(2) or 0}")
    elif ore_minuti and int(ore_minuti.group(2)) < 60:
        v = int(ore_minuti.group(1)) + int(ore_minuti.group(2)) / 60.0
    else:
        v = parse_hours(t)
    if v is None or not 0 <= v <= max_value:
        return None, True
    return round(v, 2), False


_MESI = {
    "gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4, "maggio": 5, "giugno": 6, "luglio": 7,
    "agosto": 8, "settembre": 9, "ottobre": 10, "novembre": 11, "dicembre": 12,
    "gen": 1, "feb": 2, "mar": 3, "apr": 4, "mag": 5, "giu": 6, "lug": 7, "ago": 8, "set": 9,
    "sett": 9, "ott": 10, "nov": 11, "dic": 12,
}


def _year(text: str) -> int | None:
    y = int(text)
    if len(text) == 2:
        y += 2000
    return y if 2000 <= y <= 2100 else None


def _parse_mese_anno(value: Any) -> tuple[int | None, int | None, bool]:
    """Mese e anno da "02/2026", "2-26", "febbraio 2026"...; terzo valore: True se
    il testo c'era (anche se non interpretabile)."""
    s = _clean(value, 40)
    if s is None or _is_dash(s):
        return None, None, False
    t = s.lower()
    m = re.search(r"(?<!\d)(\d{1,2})\s*[/\-.\\ ]\s*(\d{4}|\d{2})(?!\d)", t)
    if m:
        mese = int(m.group(1))
        return (mese if 1 <= mese <= 12 else None), _year(m.group(2)), True
    mese = None
    for nome, numero in sorted(_MESI.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{nome}\b", t):
            mese = numero
            break
    ya = re.search(r"(?<!\d)(\d{4})(?!\d)", t)
    return mese, (_year(ya.group(1)) if ya else None), True


def _parse_data(value: Any) -> tuple[str | None, bool]:
    """Data "GG/MM/AAAA" normalizzata; (valore, dubbia?). Testo non interpretabile
    restituito com'e' (marcato incerto); modulo non compilato -> None."""
    s = _clean(value, 40)
    if s is None or re.fullmatch(r"[\s_/.\-–]*", s):
        return None, False
    m = re.search(r"(?<!\d)(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{4}|\d{2})(?!\d)", s)
    if m:
        y = _year(m.group(3))
        try:
            if y is not None:
                d = date(y, int(m.group(2)), int(m.group(1)))
                return f"{d.day:02d}/{d.month:02d}/{d.year:04d}", False
        except ValueError:
            pass
    return s, True


def _anno_scolastico(value: Any) -> str | None:
    s = _clean(value, 40)
    if s is None or _is_dash(s):
        return None
    m = re.search(r"(20\d{2})\s*[/\-–]\s*(20\d{2}|\d{2})(?!\d)", s)
    if m:
        a, b = m.group(1), m.group(2)
        if len(b) == 2:
            b = a[:2] + b
        return f"{a}/{b}"
    return s


def _short_value(value: Any) -> str | None:
    s = _clean(value, 60)
    if s is None or _is_dash(s):
        return None
    return s.strip(" -–:;,.").upper() or None


def _sostituzione(value: Any) -> str | None:
    s = _clean(value, 10)
    if s is None:
        return None
    t = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper().strip(" .")
    return t if t in ("SI", "NO") else None


def _confidence(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    if 1.0 < v <= 100.0:          # percentuale
        v /= 100.0
    return round(min(1.0, max(0.0, v)), 3)


class _Raw(BaseModel):
    model_config = ConfigDict(extra="ignore")

    @field_validator("incerti", "illeggibili", mode="before", check_fields=False)
    @classmethod
    def _as_list(cls, v: Any) -> list[Any]:
        return list(v) if isinstance(v, (list, tuple)) else []


class _RawRow(_Raw):
    giorno: int | None = None
    prog_entrata: Any = None
    prog_uscita: Any = None
    eff_entrata: Any = None
    eff_uscita: Any = None
    ore_dichiarate: Any = None
    assenza_alunno: Any = False
    assenza_operatore: Any = False
    firma: Any = False
    note: Any = None
    trattino_effettivo: Any = False
    incerti: list[Any] = Field(default_factory=list)
    illeggibili: list[Any] = Field(default_factory=list)

    @field_validator("giorno", mode="before")
    @classmethod
    def _giorno(cls, v: Any) -> int | None:
        if isinstance(v, bool):
            return None
        try:
            g = int(str(v).strip()) if not isinstance(v, (int, float)) else int(v)
        except (TypeError, ValueError):
            return None
        return g if 1 <= g <= N_DAYS else None


def _dict_items(v: Any) -> list[Any]:
    return [x for x in v if isinstance(x, dict)] if isinstance(v, (list, tuple)) else []


class _RawHeader(_Raw):
    anno_scolastico: Any = None
    lotto: Any = None
    municipalita: Any = None
    ente: Any = None
    istituto: Any = None
    operatore: Any = None
    alunno: Any = None
    mese_anno: Any = None
    mese: Any = None
    anno: Any = None
    ore_pei: Any = None
    sostituzione: Any = None
    data_compilazione: Any = None
    firma_coordinatore: Any = False
    timbro_referente: Any = False
    totale_mensile_dichiarato: Any = None
    incerti: list[Any] = Field(default_factory=list)
    illeggibili: list[Any] = Field(default_factory=list)


class _RawResult(_Raw):
    is_foglio_firma: Any = True
    header: _RawHeader = Field(default_factory=_RawHeader)
    rows: list[_RawRow] = Field(default_factory=list)
    confidence: Any = None
    ocr_notes: Any = None

    @field_validator("header", mode="before")
    @classmethod
    def _header(cls, v: Any) -> Any:
        return v if isinstance(v, dict) else {}

    @field_validator("rows", mode="before")
    @classmethod
    def _rows(cls, v: Any) -> list[Any]:
        return _dict_items(v)


class _RawVerify(_Raw):
    rows: list[_RawRow] = Field(default_factory=list)
    totale_mensile_dichiarato: Any = None
    totale_incerto: Any = False
    totale_illeggibile: Any = False
    ocr_notes: Any = None

    @field_validator("rows", mode="before")
    @classmethod
    def _rows(cls, v: Any) -> list[Any]:
        return _dict_items(v)


def _finish_flags(values: dict[str, Any], incerti: list[str], illeggibili: list[str],
                  bool_fields: frozenset[str]) -> tuple[list[str], list[str]]:
    """Applica la regola "illeggibile => valore None" (per i campi booleani un
    'illeggibile' diventa 'incerto') e rende disgiunti i due elenchi."""
    ill: list[str] = []
    inc = list(incerti)
    for f in illeggibili:
        if f in bool_fields:
            inc.append(f)
        else:
            values[f] = None
            ill.append(f)
    inc_out: list[str] = []
    for f in inc:
        if f not in ill and f not in inc_out:
            inc_out.append(f)
    return inc_out, ill


def normalize_row(raw: _RawRow | dict, giorno: int) -> DayRow:
    """Riga del modello -> ``DayRow`` normalizzata."""
    if isinstance(raw, dict):
        raw = _RawRow.model_validate(raw)
    incerti = _flag_list(raw.incerti, DAY_FIELDS)
    illeggibili = _flag_list(raw.illeggibili, DAY_FIELDS)
    values: dict[str, Any] = {}
    trattino = _bool(raw.trattino_effettivo)
    for f in _TIME_FIELDS:
        v, dash, bad = _norm_time(getattr(raw, f))
        if dash and f.startswith("eff_"):
            trattino = True
        if bad and f not in illeggibili:
            incerti.append(f)          # trascritto com'e': lo segnalera' la validazione (E03)
        values[f] = v
    ore, bad = _norm_hours(raw.ore_dichiarate)
    if bad and "ore_dichiarate" not in illeggibili:
        illeggibili.append("ore_dichiarate")
    values["ore_dichiarate"] = ore
    for f in _ROW_BOOL_FIELDS:
        values[f] = _bool(getattr(raw, f))
    values["note"] = _upper(raw.note, 200)
    incerti, illeggibili = _finish_flags(values, incerti, illeggibili, _ROW_BOOL_FIELDS)
    return DayRow(giorno=giorno, trattino_effettivo=trattino, incerti=incerti, illeggibili=illeggibili, **values)


def _rows_by_day(raw_rows: list[_RawRow], wanted: list[int] | None = None) -> dict[int, DayRow]:
    """Righe normalizzate per giorno. Senza numero di giorno valido si usa la
    posizione, se il numero di righe coincide con quello atteso."""
    order = wanted if wanted is not None else list(range(1, N_DAYS + 1))
    allowed = set(order)
    out: dict[int, DayRow] = {}
    positional = len(raw_rows) == len(order)
    for idx, raw in enumerate(raw_rows):
        g = raw.giorno if raw.giorno in allowed else None
        if g is None and positional:
            g = order[idx]
        if g is None:
            continue
        row = normalize_row(raw, g)
        prev = out.get(g)
        if prev is None or (not prev.has_content() and row.has_content()):
            out[g] = row
    return out


def normalize_header(raw: _RawHeader | dict) -> Header:
    """Intestazione del modello -> ``Header`` normalizzata."""
    if isinstance(raw, dict):
        raw = _RawHeader.model_validate(raw)

    def expand(flags: list[Any]) -> list[str]:
        out: list[str] = []
        for f in flags:
            if f == "mese_anno":
                out += ["mese", "anno"]
            elif isinstance(f, str):
                out.append(f)
        return _flag_list(out, HEADER_FIELDS)

    incerti = expand(raw.incerti)
    illeggibili = expand(raw.illeggibili)
    values: dict[str, Any] = {
        "anno_scolastico": _anno_scolastico(raw.anno_scolastico),
        "lotto": _short_value(raw.lotto),
        "municipalita": _short_value(raw.municipalita),
        "ente": _upper(raw.ente, 120),
        "istituto": _upper(raw.istituto, 120),
        "operatore": _upper(raw.operatore, 120),
        "alunno": _upper(raw.alunno, 120),
        "sostituzione": _sostituzione(raw.sostituzione),
        "firma_coordinatore": _bool(raw.firma_coordinatore),
        "timbro_referente": _bool(raw.timbro_referente),
    }
    source = raw.mese_anno
    if _clean(source) is None and (raw.mese is not None or raw.anno is not None):
        source = f"{raw.mese}/{raw.anno}"
    mese, anno, scritto = _parse_mese_anno(source)
    values["mese"], values["anno"] = mese, anno
    if scritto:
        for f, v in (("mese", mese), ("anno", anno)):
            if v is None and f not in illeggibili:
                illeggibili.append(f)
    for f, limit in (("ore_pei", 60.0), (_TOTAL, 744.0)):
        v, bad = _parse_number(getattr(raw, f), limit)
        values[f] = v
        if bad and f not in illeggibili:
            illeggibili.append(f)
    data, dubbia = _parse_data(raw.data_compilazione)
    values["data_compilazione"] = data
    if dubbia:
        incerti.append("data_compilazione")
    incerti, illeggibili = _finish_flags(values, incerti, illeggibili, _HEADER_BOOL_FIELDS)
    return Header(incerti=incerti, illeggibili=illeggibili, **values)


def _notes(value: Any, max_len: int = 1500) -> str | None:
    return _clean(value, max_len)


def parse_extraction(data: dict) -> ExtractionResult:
    """JSON della lettura completa -> ``ExtractionResult`` (sempre 31 righe)."""
    try:
        raw = _RawResult.model_validate(data)
    except ValidationError as exc:
        log.warning("Risposta di Claude non conforme allo schema: %s", exc.errors()[:3])
        raise EngineError(_MSG_BAD_JSON) from exc
    is_form = _bool(raw.is_foglio_firma) if raw.is_foglio_firma is not None else True
    by_day = _rows_by_day(raw.rows)
    rows = [by_day.get(g, DayRow(giorno=g)) for g in range(1, N_DAYS + 1)]
    return ExtractionResult(
        is_foglio_firma=is_form,
        header=normalize_header(raw.header),
        rows=rows,
        ocr_notes=_notes(raw.ocr_notes),
        confidence=_confidence(raw.confidence),
        engine="claude",
    )


@dataclass
class Reread:
    """Esito normalizzato della verifica incrociata."""

    rows: dict[int, DayRow]
    totale: float | None = None
    totale_incerto: bool = False
    totale_illeggibile: bool = False
    totale_richiesto: bool = False
    notes: str | None = None


def parse_verification(data: dict, days: list[int], include_total: bool) -> Reread:
    try:
        raw = _RawVerify.model_validate(data)
    except ValidationError as exc:
        log.warning("Risposta di verifica non conforme allo schema: %s", exc.errors()[:3])
        raise EngineError(_MSG_BAD_JSON) from exc
    out = Reread(rows=_rows_by_day(raw.rows, days), totale_richiesto=include_total, notes=_notes(raw.ocr_notes, 600))
    if include_total:
        v, bad = _parse_number(raw.totale_mensile_dichiarato, 744.0)
        out.totale = v
        out.totale_illeggibile = _bool(raw.totale_illeggibile) or bad
        out.totale_incerto = _bool(raw.totale_incerto) and not out.totale_illeggibile
        if out.totale_illeggibile:
            out.totale = None
    return out


# ==========================================================================
# Verifica incrociata: selezione e fusione
# ==========================================================================

@dataclass
class VerificationPlan:
    days: list[int]
    include_total: bool

    @property
    def needed(self) -> bool:
        return bool(self.days) or self.include_total


def plan_verification(header: Header, rows: list[DayRow]) -> VerificationPlan:
    """Righe da rileggere: anomalie di coerenza/lettura, campi incerti o
    illeggibili; con E02 (totale mensile diverso) anche tutte le righe con ore."""
    anomalies, _ = validate(header, rows)
    days: set[int] = set()
    for a in anomalies:
        if a.giorno is not None and a.codice in VERIFY_CODES:
            days.add(a.giorno)
    for r in rows:
        if r.incerti or r.illeggibili:
            days.add(r.giorno)
    include_total = any(a.codice == E02_TOTALE_MENSILE_DIVERSO for a in anomalies)
    if include_total:
        for r in rows:
            if r.ore_dichiarate is not None or ore_riconosciute(r) > 0:
                days.add(r.giorno)
    if _TOTAL in header.incerti or _TOTAL in header.illeggibili:
        include_total = True
    return VerificationPlan(days=sorted(d for d in days if 1 <= d <= N_DAYS), include_total=include_total)


def _same(field: str, a: Any, b: Any) -> bool:
    if field == "ore_dichiarate":
        if a is None or b is None:
            return a is None and b is None
        return abs(float(a) - float(b)) <= TOLLERANZA
    if field in _TIME_FIELDS:
        na = normalize_time(a) if isinstance(a, str) else None
        nb = normalize_time(b) if isinstance(b, str) else None
        if na is not None or nb is not None:
            return na == nb
        return (a or None) == (b or None)
    if field == "note":
        def key(s: Any) -> str:
            return re.sub(r"[\W_]+", " ", str(s or "")).strip().casefold()
        return key(a) == key(b)
    return bool(a) == bool(b) if field in _ROW_BOOL_FIELDS else a == b


def row_consistent(header: Header, row: DayRow) -> bool:
    """True se la riga non presenta anomalie di coerenza (E01/E03/W01/W02/W05/W11)."""
    anomalies, _ = validate(header, [row])
    return not any(a.giorno == row.giorno and a.codice in COHERENCE_CODES for a in anomalies)


def _fmt(field: str, value: Any) -> str:
    if value is None:
        return "vuoto"
    if isinstance(value, bool):
        return "sì" if value else "no"
    if isinstance(value, float):
        return f"{value:g}".replace(".", ",")
    return str(value)


def _present(value: Any) -> bool:
    return value is not None and value is not False and value != ""


def merge_row(header: Header, first: DayRow, reread: DayRow) -> tuple[DayRow, list[str]]:
    """Fonde la prima lettura con la rilettura mirata (regole nel docstring del
    modulo). Restituisce la riga fusa e l'elenco dei valori cambiati."""
    merged = first.model_copy(deep=True)
    incerti: set[str] = set()
    illeggibili: set[str] = set()
    taken: dict[str, bool] = {}          # campo preso dalla rilettura -> da segnalare comunque come incerto
    agreed_flagged: list[str] = []       # concordi ma ancora incerti nella rilettura
    for f in DAY_FIELDS:
        a_ill, b_ill = f in first.illeggibili, f in reread.illeggibili
        b_unc = f in reread.incerti
        av, bv = getattr(first, f), getattr(reread, f)
        if a_ill and b_ill:
            setattr(merged, f, False if f in _ROW_BOOL_FIELDS else None)
            illeggibili.add(f)
        elif b_ill:
            incerti.add(f)               # solo la prima lettura ha un valore
        elif not a_ill and _same(f, av, bv):
            if b_unc:
                agreed_flagged.append(f)
        else:
            setattr(merged, f, bv)
            # una lettura vede scrittura e l'altra no: resta sempre da verificare
            taken[f] = b_unc or (a_ill or _present(av)) != _present(bv)
    if all(f in taken or _same(f, getattr(first, f), getattr(reread, f)) for f in ("eff_entrata", "eff_uscita")):
        merged.trattino_effettivo = reread.trattino_effettivo
    consistent = row_consistent(header, merged.model_copy(update={"incerti": [], "illeggibili": []}))
    for f, flagged in taken.items():
        if flagged or not consistent:
            incerti.add(f)
    if not consistent:
        incerti.update(agreed_flagged)
    merged.illeggibili = [f for f in DAY_FIELDS if f in illeggibili]
    merged.incerti = [f for f in DAY_FIELDS if f in incerti and f not in illeggibili]
    for f in merged.illeggibili:
        merged.confidenza.pop(f, None)
    def before(f: str) -> str:
        return "illeggibile" if f in first.illeggibili else _fmt(f, getattr(first, f))

    def after(f: str) -> str:
        return "illeggibile" if f in merged.illeggibili else _fmt(f, getattr(merged, f))

    changes = [
        f"{etichetta_campo(f).lower()} {before(f)} → {after(f)}"
        for f in DAY_FIELDS
        if not _same(f, getattr(first, f), getattr(merged, f)) or before(f) != after(f)
    ]
    return merged, changes


def merge_verification(header: Header, rows: list[DayRow], reread: Reread) -> tuple[Header, list[DayRow], list[str]]:
    """Applica la rilettura a righe e totale; restituisce (intestazione, righe, note)."""
    by_day = {r.giorno: r for r in rows}
    notes: list[str] = []
    for g, rr in sorted(reread.rows.items()):
        first = by_day.get(g)
        if first is None:
            continue
        if first.has_content() and not rr.has_content() and not rr.incerti:
            # striscia vuota per una riga compilata: piu' probabile un ritaglio
            # fuori posto che una cella vuota; resta la prima lettura
            notes.append(f"giorno {g}: rilettura senza dati, mantenuta la prima lettura")
            continue
        merged, changes = merge_row(header, first, rr)
        by_day[g] = merged
        if changes:
            notes.append(f"giorno {g}: " + ", ".join(changes))
    new_rows = [by_day.get(g, DayRow(giorno=g)) for g in range(1, N_DAYS + 1)]
    new_header = header.model_copy(deep=True)
    if reread.totale_richiesto:
        a = header.totale_mensile_dichiarato
        a_ill = _TOTAL in header.illeggibili
        b, b_ill, b_unc = reread.totale, reread.totale_illeggibile, reread.totale_incerto
        inc = set(new_header.incerti) - {_TOTAL}
        ill = set(new_header.illeggibili) - {_TOTAL}
        if a_ill and b_ill:
            new_header.totale_mensile_dichiarato = None
            ill.add(_TOTAL)
        elif b_ill:
            inc.add(_TOTAL)
        else:
            _, totals = validate(header, new_rows)
            consistent = b is not None and abs(b - totals.ore_dichiarate) <= TOLLERANZA
            if not a_ill and _same("ore_dichiarate", a, b):
                if b_unc and not consistent:
                    inc.add(_TOTAL)
            else:
                new_header.totale_mensile_dichiarato = b
                if b_unc or not consistent:
                    inc.add(_TOTAL)
                if not _same("ore_dichiarate", a, b):
                    notes.append(f"totale mensile {_fmt(_TOTAL, a)} → {_fmt(_TOTAL, b)}")
        new_header.incerti = [f for f in HEADER_FIELDS if f in inc and f not in ill]
        new_header.illeggibili = [f for f in HEADER_FIELDS if f in ill]
    return new_header, new_rows, notes


# ==========================================================================
# Lettura della risposta
# ==========================================================================

def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t


def message_json(msg: Any) -> dict:
    """Oggetto JSON della risposta. Si seleziona il testo per tipo di blocco (i
    blocchi di thinking possono precedere il testo); dopo un fallback lato
    server a meta' risposta il testo puo' essere spezzato in piu' blocchi."""
    blocks = list(getattr(msg, "content", None) or [])
    texts = [b.text for b in blocks if getattr(b, "type", None) == "text" and isinstance(getattr(b, "text", None), str)]
    last_fb = max((i for i, b in enumerate(blocks) if getattr(b, "type", None) == "fallback"), default=-1)
    after = [
        b.text for b in blocks[last_fb + 1:]
        if getattr(b, "type", None) == "text" and isinstance(getattr(b, "text", None), str)
    ]
    ordered = [texts[0]] if texts else []
    ordered.append("".join(texts))
    if after:
        ordered += [after[0], "".join(after)]
    ordered += texts[1:]
    candidates: list[str] = []
    for c in ordered:
        if c and c not in candidates:
            candidates.append(c)
    for c in candidates:
        t = _strip_fences(c)
        for attempt in (t, t[t.find("{"): t.rfind("}") + 1] if "{" in t and "}" in t else ""):
            if not attempt:
                continue
            try:
                obj = json.loads(attempt)
            except (ValueError, TypeError):
                continue
            if isinstance(obj, dict):
                return obj
    raise EngineError(_MSG_BAD_JSON)


def _safe_progress(progress: ProgressFn, fraction: float, message: str) -> None:
    try:
        progress(max(0.0, min(1.0, fraction)), message)
    except Exception:  # noqa: BLE001 - l'avanzamento non deve interrompere la lettura
        log.debug("Callback di avanzamento non riuscita", exc_info=True)


def _short_label(model: str) -> str:
    info = model_info(model)
    if info:
        return str(info["label"]).split(" — ")[0]
    return model


# ==========================================================================
# Motore
# ==========================================================================

class ClaudeEngine:
    """Motore OCR basato su Claude Vision (vedi docstring del modulo)."""

    name = "claude"

    def __init__(self, api_key: str | None, model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT,
                 verify: bool = True, client: Any = None):
        self.api_key = (api_key or "").strip() or None
        self.model = (model or DEFAULT_MODEL).strip()
        self.effort = effort if effort in EFFORTS else DEFAULT_EFFORT
        self.verify = bool(verify)
        self._client = client
        self._lock = threading.Lock()

    # ------------------------------------------------------------- stato
    def is_available(self) -> tuple[bool, str]:
        if self._client is None and not self.api_key:
            return False, _MSG_NO_KEY
        return True, f"Claude Vision pronto ({_short_label(self.model)})."

    def _get_client(self) -> Any:
        with self._lock:
            if self._client is None:
                if not self.api_key:
                    raise EngineError(_MSG_NO_KEY)
                self._client = anthropic.Anthropic(
                    api_key=self.api_key,
                    max_retries=MAX_RETRIES,
                    timeout=anthropic.Timeout(REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT),
                )
            return self._client

    # ---------------------------------------------------------- richieste
    def _request_args(self, content: list[dict], schema: dict, max_tokens: int) -> tuple[Any, dict[str, Any]]:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": prompts.SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": content}],
            "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        }
        if self.model in FALLBACK_MODELS:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
            return client.beta.messages.stream, kwargs
        return client.messages.stream, kwargs

    def _stream(self, content: list[dict], schema: dict, max_tokens: int, progress: ProgressFn,
                span: tuple[float, float], message: str, expected_chars: int) -> Any:
        stream_fn, kwargs = self._request_args(content, schema, max_tokens)
        p0, p1 = span
        try:
            with stream_fn(**kwargs) as stream:
                start = last = time.monotonic()
                chars = 0
                try:
                    events = iter(stream)
                except TypeError:
                    events = iter(())
                for event in events:
                    if getattr(event, "type", None) == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if getattr(delta, "type", None) == "text_delta":
                            chars += len(getattr(delta, "text", "") or "")
                    now = time.monotonic()
                    if now - last >= 0.5:
                        last = now
                        frac = max(min(1.0, chars / float(max(1, expected_chars))), min(0.5, (now - start) / 240.0))
                        _safe_progress(progress, p0 + (p1 - p0) * min(0.98, frac), message)
                return stream.get_final_message()
        except anthropic.AnthropicError as exc:
            raise map_api_error(exc, self.model) from exc

    def _call(self, content: list[dict], schema: dict, meter: _UsageMeter, progress: ProgressFn,
              span: tuple[float, float], message: str, expected_chars: int) -> dict:
        """Una richiesta completa: refusal -> errore; max_tokens -> un nuovo tentativo
        con limite doppio; poi il JSON della risposta."""
        for attempt, max_tokens in enumerate((MAX_TOKENS, MAX_TOKENS_RETRY)):
            msg = self._stream(content, schema, max_tokens, progress, span, message, expected_chars)
            meter.add(msg)
            stop = getattr(msg, "stop_reason", None)
            if stop == "refusal":
                details = getattr(msg, "stop_details", None)
                log.warning("Richiesta rifiutata da Claude (categoria: %s)", getattr(details, "category", None))
                raise EngineError(_MSG_REFUSAL)
            if stop == "max_tokens":
                log.warning("Risposta di Claude troncata (max_tokens=%d, tentativo %d)", max_tokens, attempt + 1)
                continue
            return message_json(msg)
        raise EngineError(_MSG_TRUNCATED)

    # ------------------------------------------------------------ lettura
    def extract(self, page: PageInput, progress: ProgressFn | None = None) -> ExtractionResult:
        progress = progress or no_progress
        ok, why = self.is_available()
        if not ok:
            raise EngineError(why)
        started = time.monotonic()
        meter = _UsageMeter(self.model)
        label = _short_label(self.model)

        _safe_progress(progress, 0.05, "Preparazione delle immagini…")
        img, grid, detected = _page_and_grid(page)
        crops = first_pass_crops(img, grid, detected)
        content: list[dict] = [_image_block(c) for _, c in crops]
        content.append({"type": "text", "text": prompts.first_pass_text([lbl for lbl, _ in crops], detected)})

        _safe_progress(progress, 0.15, f"Invio a {label}…")
        data = self._call(content, prompts.SCHEMA, meter, progress, (0.15, 0.68),
                          "Lettura della scrittura a mano in corso…", 9000)
        result = parse_extraction(data)
        _safe_progress(progress, 0.7, "Lettura completata")

        notes: list[str] = [result.ocr_notes] if result.ocr_notes else []
        if self.verify and result.is_foglio_firma:
            try:
                extra = self._verify(img, grid, detected, result, meter, progress)
                notes += extra
            except EngineError as exc:
                log.warning("Verifica incrociata non riuscita: %s", exc)
                motivo = re.split(r"(?<=[.!?])\s+", str(exc).strip())[0].rstrip(".") or "errore del servizio"
                notes.append(f"Verifica incrociata non eseguita: {motivo}. Restano i valori della prima lettura.")
            except Exception:  # noqa: BLE001 - la prima lettura (gia' pagata) non va persa
                log.exception("Errore interno nella verifica incrociata")
                notes.append("Verifica incrociata non eseguita per un errore interno: "
                             "restano i valori della prima lettura.")
        fallback_models = sorted(m for m in meter.served_by if m != self.model)
        if fallback_models:
            notes.append(f"Parte della lettura è stata eseguita dal modello di riserva {', '.join(fallback_models)}.")

        result.ocr_notes = " ".join(notes).strip() or None
        result.engine = "claude"
        result.model = self.model
        result.usage = meter.usage(time.monotonic() - started)
        _safe_progress(progress, 1.0, "Completato")
        return result

    def _verify(self, img: np.ndarray, grid: TableGrid, detected: bool, result: ExtractionResult,
                meter: _UsageMeter, progress: ProgressFn) -> list[str]:
        """Seconda richiesta mirata sulle righe dubbie; aggiorna ``result`` e
        restituisce le note da aggiungere."""
        plan = plan_verification(result.header, result.rows)
        if not plan.needed:
            _safe_progress(progress, 0.95, "Verifica incrociata: nessuna riga da ricontrollare")
            return []
        n = len(plan.days)
        parts = [f"di {n} righe" if n != 1 else "di 1 riga"] if n else []
        if plan.include_total:
            parts.append("del totale mensile")
        what = " e ".join(parts)
        _safe_progress(progress, 0.75, f"Verifica incrociata: rilettura {what}…")
        crops = verification_crops(img, grid, detected, plan.days, plan.include_total)
        content: list[dict] = [_image_block(c, STRIP_UPSCALE) for _, c in crops]
        content.append({
            "type": "text",
            "text": prompts.verification_text(plan.days, [lbl for lbl, _ in crops], plan.include_total),
        })
        data = self._call(content, prompts.VERIFY_SCHEMA, meter, progress, (0.75, 0.93),
                          "Verifica incrociata in corso…", 330 * max(1, n) + 300)
        reread = parse_verification(data, plan.days, plan.include_total)
        _safe_progress(progress, 0.95, "Unione dei risultati della verifica…")
        header, rows, changes = merge_verification(result.header, result.rows, reread)
        result.header, result.rows = header, rows
        notes: list[str] = []
        if changes:
            notes.append(f"Verifica incrociata: valori aggiornati dopo la rilettura mirata ({'; '.join(changes)}).")
        else:
            notes.append(f"Verifica incrociata: rilettura {what} senza valori cambiati.")
        if reread.notes:
            notes.append(reread.notes)
        return notes


# ==========================================================================
# Verifica della chiave
# ==========================================================================

def test_api_key(api_key: str, model: str, *, client: Any = None) -> tuple[bool, str]:
    """Verifica chiave e modello senza consumare token (``models.retrieve``)."""
    key = (api_key or "").strip()
    model = (model or DEFAULT_MODEL).strip()
    if not key and client is None:
        return False, "Inserisci la chiave API di Anthropic."
    own = client is None
    if own:
        client = anthropic.Anthropic(api_key=key, max_retries=1, timeout=anthropic.Timeout(30.0, connect=10.0))
    try:
        info = client.models.retrieve(model)
    except anthropic.NotFoundError:
        return False, (
            f"La chiave API è valida, ma il modello «{model}» non è disponibile per questo account: "
            "scegline un altro nelle Impostazioni."
        )
    except anthropic.AnthropicError as exc:
        return False, str(map_api_error(exc, model))
    finally:
        if own:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                log.debug("Chiusura del client non riuscita", exc_info=True)
    nome = getattr(info, "display_name", None) or _short_label(model)
    return True, f"Chiave API valida: modello {nome} disponibile."


__all__ = [
    "ClaudeEngine",
    "DEFAULT_MODEL",
    "MAX_LONG_EDGE",
    "MAX_PIXELS",
    "Reread",
    "VerificationPlan",
    "first_pass_crops",
    "fit_to_limits",
    "map_api_error",
    "merge_row",
    "merge_verification",
    "message_json",
    "normalize_header",
    "normalize_row",
    "parse_extraction",
    "parse_verification",
    "plan_verification",
    "row_consistent",
    "test_api_key",
    "verification_crops",
]
