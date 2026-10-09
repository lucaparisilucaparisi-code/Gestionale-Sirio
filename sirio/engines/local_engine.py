"""Motore OCR locale (offline): OpenCV + TrOCR, solo CPU.

Funziona senza Internet (dopo il primo download del modello) e senza chiave API.

Pipeline di una pagina
----------------------
1. **Inchiostro** (``sirio.vision.ink``): ogni cella dei 31 giorni viene
   classificata come vuota, trattino, crocetta (assenze), firma o scrittura.
2. **Ritagli puliti**: le celle con scrittura vengono ritagliate togliendo le
   linee della griglia e i tratti che sconfinano dalle celle vicine; i campi
   dell'intestazione vengono cercati sulle righe di risposta a destra delle
   etichette stampate (posizioni misurate sul modulo reale, affinate con le
   linee trovate nella scansione).
3. **TrOCR** (``VisionEncoderDecoderModel``): ogni ritaglio e' codificato una
   sola volta; la decodifica e' *vincolata* al lessico dei valori ammessi
   (orari HH:MM, ore 0,5-8, mese/anno, numeri, note frequenti...) e affiancata
   da una lettura libera che misura quanto la scrittura si discosti dal
   lessico. La cache della cross-attention e' condivisa fra le ipotesi del
   beam search: molto piu' veloce di ``generate`` su CPU.
4. **Riconciliazione**: per ogni riga si sceglie la combinazione di orari e ore
   piu' probabile *e* coerente (programmato ~ effettivo, uscita - entrata =
   ore); i campi scelti contro la lettura migliore o con bassa probabilita'
   finiscono in ``incerti``; scrittura presente ma non interpretabile ->
   ``illeggibili`` (valore ``None``). Una cella poco leggibile il cui valore e'
   determinato dal resto della riga (es. uscita = entrata + ore) viene
   compilata ma resta "incerta" ed e' elencata in ``ocr_notes``.

Precisione misurata (foglio reale d'esempio, CPU a 4 core)
---------------------------------------------------------
Modello predefinito ``microsoft/trocr-base-handwritten`` (decoder quantizzato
int8): orari 69/72 (tutti gli errori segnalati come incerti), ore 17/17,
assenze 62/62, firme 31/31, trattini 31/31, note 31/31, intestazione (lotto,
municipalita', mese/anno, ore PEI, totale, firma coordinatore, timbro) 10/10;
i nomi in stampatello sono letti solo in parte e vanno sempre verificati.
Circa 90 secondi per pagina (~100 campi scritti). A confronto, con la stessa
pipeline: ``trocr-small`` 33 s/pagina ma orari 63/72 e note 28/31 (richiede
anche ``sentencepiece`` e ``protobuf``); ``trocr-large`` 177 s/pagina, orari
64/72, note 28/31.
"""

from __future__ import annotations

import calendar
import errno
import fnmatch
import functools
import importlib
import importlib.util
import itertools
import logging
import math
import os
import re
import threading
import time
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import cv2
import numpy as np

from sirio.engines.base import EngineError, PageInput, ProgressFn, no_progress
from sirio.models import DayRow, ExtractionResult, Header, Usage
from sirio.validation import hours_between, normalize_time, parse_hours, parse_time
from sirio.vision import ink
from sirio.vision.preprocess import to_gray

if TYPE_CHECKING:
    from sirio.vision.grid import TableGrid

log = logging.getLogger(__name__)

Box = tuple[int, int, int, int]

# ==========================================================================
# Modelli
# ==========================================================================

DEFAULT_MODEL = "microsoft/trocr-base-handwritten"

# Dimensione indicativa del download (MB) e nome breve per i messaggi.
KNOWN_MODELS: dict[str, dict[str, Any]] = {
    "microsoft/trocr-small-handwritten": {"label": "TrOCR small", "mb": 250},
    "microsoft/trocr-base-handwritten": {"label": "TrOCR base", "mb": 1340},
    "microsoft/trocr-large-handwritten": {"label": "TrOCR large", "mb": 2240},
}

_TOKENIZER_FILES = ("vocab.json", "tokenizer.json", "sentencepiece.bpe.model", "spiece.model")
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
_MISSING_MSG = (
    "Componenti offline non installati: il motore locale richiede i pacchetti "
    "«torch» e «transformers». Si installano con il programma di avvio (opzione "
    "offline) oppure si può usare il motore Claude."
)


def _model_label(model_name: str) -> str:
    info = KNOWN_MODELS.get(model_name)
    if info:
        return str(info["label"])
    return Path(model_name).name or model_name


def _fmt_mb(n_bytes: float) -> str:
    """Megabyte con il separatore delle migliaia italiano (es. "1.334 MB")."""
    return f"{n_bytes / 1e6:,.0f} MB".replace(",", ".")


def _fmt_size(mb: float) -> str:
    if mb >= 1000:
        return f"{mb / 1000:.1f} GB".replace(".", ",")
    return f"{mb:.0f} MB"


# ==========================================================================
# Lessici dei valori ammessi
# ==========================================================================

TIME_MIN = 6 * 60          # 06:00
TIME_MAX = 20 * 60         # 20:00
TIME_STEP = 15             # minuti ammessi: 00, 15, 30, 45
HOURS_MAX_DAY = 8.0

MESI_NOMI = (
    "GENNAIO", "FEBBRAIO", "MARZO", "APRILE", "MAGGIO", "GIUGNO",
    "LUGLIO", "AGOSTO", "SETTEMBRE", "OTTOBRE", "NOVEMBRE", "DICEMBRE",
)
_ROMANI = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")

# Note ricorrenti sui fogli firma: (valore canonico, varianti scritte).
NOTE_FREQUENTI: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("104", ("104", "L104", "L.104", "LEGGE 104", "PERMESSO 104", "PERM. 104")),
    ("PONTE DI CARNEVALE", ("PONTE DI CARNEVALE", "PONTE CARNEVALE")),
    ("CARNEVALE", ("CARNEVALE",)),
    ("PONTE", ("PONTE",)),
    ("FESTIVO", ("FESTIVO",)),
    ("FESTA", ("FESTA",)),
    ("SCIOPERO", ("SCIOPERO",)),
    ("ASSEMBLEA SINDACALE", ("ASSEMBLEA SINDACALE",)),
    ("ASSEMBLEA", ("ASSEMBLEA",)),
    ("MALATTIA", ("MALATTIA",)),
    ("FERIE", ("FERIE",)),
    ("PERMESSO", ("PERMESSO",)),
    ("CHIUSURA SCUOLA", ("CHIUSURA SCUOLA", "SCUOLA CHIUSA")),
    ("USCITA DIDATTICA", ("USCITA DIDATTICA",)),
    ("GITA SCOLASTICA", ("GITA SCOLASTICA", "GITA")),
    ("ELEZIONI", ("ELEZIONI",)),
    ("VACANZE", ("VACANZE",)),
    ("VACANZE NATALIZIE", ("VACANZE NATALIZIE", "VACANZE DI NATALE")),
    ("VACANZE PASQUALI", ("VACANZE PASQUALI", "VACANZE DI PASQUA")),
    ("ALUNNO ASSENTE", ("ALUNNO ASSENTE",)),
    ("ASSENTE", ("ASSENTE",)),
    ("RECUPERO", ("RECUPERO",)),
    ("SOSTITUZIONE", ("SOSTITUZIONE",)),
    ("FORMAZIONE", ("FORMAZIONE", "CORSO DI FORMAZIONE")),
    ("GLO", ("GLO", "GLHO", "INCONTRO GLO", "RIUNIONE GLO")),
    ("RIUNIONE", ("RIUNIONE",)),
)

_NOTE_CANONICHE = frozenset(canon for canon, _forms in NOTE_FREQUENTI)

NAME_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "ÀÈÉÌÒÙàèéìòù0123456789 '.-/,"
)
NOTE_CHARS = NAME_CHARS | frozenset("()")


def _key(text: str) -> str:
    """Forma di confronto: senza spazi, maiuscola, Unicode NFKC."""
    return "".join(unicodedata.normalize("NFKC", text).split()).upper()


# Segni spuri tollerati prima/dopo il valore (residui di linee, punti, apostrofi).
_JUNK_LEAD = ".'`\"-"
_JUNK_TRAIL = ".,'`\"-:;"
_MAX_JUNK = 2


@dataclass(frozen=True, eq=False)
class Lexicon:
    """Insieme finito di scritture ammesse, ognuna con il suo valore canonico.

    Le scritture sono confrontate senza spazi e in maiuscolo (``_key``): la
    decodifica vincolata ammette solo i token che mantengono il testo un
    prefisso di qualche scrittura del lessico. Sono tollerati fino a due segni
    spuri (punti, apostrofi, trattini) prima e dopo la scrittura.
    """

    name: str
    forms: dict[str, Any]
    prefixes: frozenset[str]
    alphabet: frozenset[str]
    max_chars: int
    numeric: bool = False

    @classmethod
    def build(cls, name: str, pairs: Iterable[tuple[str, Any]], numeric: bool = False) -> Lexicon:
        forms: dict[str, Any] = {}
        for text, value in pairs:
            k = _key(text)
            if k and k not in forms:
                forms[k] = value
        if not forms:
            raise ValueError(f"Lessico vuoto: {name}")
        prefixes = frozenset(k[:i] for k in forms for i in range(len(k) + 1))
        alphabet = frozenset("".join(forms)) | frozenset(_JUNK_LEAD) | frozenset(_JUNK_TRAIL)
        return cls(name, forms, prefixes, alphabet, max(len(k) for k in forms) + 2 * _MAX_JUNK, numeric)

    def signature(self, key: str) -> str:
        """Contenuto essenziale di una scrittura (senza separatori e segni spuri;
        nei lessici numerici le lettere confuse con cifre valgono come cifre)."""
        if self.numeric:
            return "".join(c for c in key.translate(_ALIAS_TABLE) if c.isdigit())
        return "".join(c for c in key if c.isalnum())

    def is_prefix(self, key: str) -> bool:
        """True se ``key`` puo' ancora diventare una scrittura ammessa."""
        core = key.lstrip(_JUNK_LEAD)
        if len(key) - len(core) > _MAX_JUNK:
            return False
        if core in self.prefixes:
            return True
        body = core.rstrip(_JUNK_TRAIL)
        return len(core) - len(body) <= _MAX_JUNK and body in self.forms

    def complete(self, key: str) -> Any:
        """Valore canonico se ``key`` e' una scrittura completa (``None`` altrimenti)."""
        core = key.lstrip(_JUNK_LEAD)
        if len(key) - len(core) > _MAX_JUNK:
            return None
        if core in self.forms:
            return self.forms[core]
        body = core.rstrip(_JUNK_TRAIL)
        if len(core) - len(body) > _MAX_JUNK:
            return None
        return self.forms.get(body)

    def value(self, text: str | None) -> Any:
        """Valore canonico di una scrittura (``None`` se non appartiene al lessico)."""
        if not text:
            return None
        return self.complete(_key(text))


# Confusioni tipiche della lettura delle cifre scritte a mano.
_DIGIT_ALIASES = {"1": ("1", "I", "L"), "0": ("0", "O")}
_ALIAS_TABLE = str.maketrans({"I": "1", "L": "1", "O": "0", "U": "11"})


def _digit_variants(s: str) -> list[str]:
    """Varianti di una sequenza di cifre con le confusioni piu' comuni ("11" -> "1I", "LL", "U"...)."""
    out = ["".join(p) for p in itertools.product(*(_DIGIT_ALIASES.get(c, (c,)) for c in s))]
    if s == "11":
        out.append("U")
    return out


def _num_forms(x: float) -> list[str]:
    """Scritture di un numero di ore: "3", "1,5", "1.5"."""
    if abs(x - round(x)) < 1e-9:
        return [str(int(round(x)))]
    whole = int(math.floor(x))
    frac = f"{x - whole:.2f}".rstrip("0").split(".")[1]
    head = str(whole) if whole else "0"
    forms = [f"{head},{frac}", f"{head}.{frac}"]
    if not whole:
        forms += [f",{frac}"]
    return forms


_TIME_SEPARATORS = (":", ".", ",", ";", "'", "")


@functools.cache
def time_lexicon() -> Lexicon:
    """Orari 06:00-20:00 a passi di 15 minuti ("8:00", "08.30", "1100", "8"...)."""
    pairs: list[tuple[str, str]] = []
    for minutes in range(TIME_MIN, TIME_MAX + 1, TIME_STEP):
        h, m = divmod(minutes, 60)
        value = f"{h:02d}:{m:02d}"
        mins = _digit_variants(f"{m:02d}")
        if m == 0:
            mins += ["000", "0"]
        for hh_digits in dict.fromkeys((str(h), f"{h:02d}")):
            for hh in _digit_variants(hh_digits):
                for sep in _TIME_SEPARATORS:
                    for mm in mins:
                        if sep or len(mm) == 2:
                            pairs.append((f"{hh}{sep}{mm}", value))
                if m == 0:
                    pairs.append((hh, value))
    return Lexicon.build("orari", pairs, numeric=True)


@functools.cache
def hours_lexicon() -> Lexicon:
    """Ore giornaliere 0-8 a passi di mezz'ora ("3", "1,5", "1.5", "3h")."""
    pairs: list[tuple[str, float]] = []
    for half in range(int(HOURS_MAX_DAY * 2) + 1):
        v = half / 2
        for f in _num_forms(v):
            for variant in _digit_variants(f):
                pairs.append((variant, v))
                pairs.append((variant + "h", v))
    return Lexicon.build("ore", pairs, numeric=True)


@functools.cache
def total_lexicon() -> Lexicon:
    """Totale mensile 0-300 ore a passi di mezz'ora."""
    pairs: list[tuple[str, float]] = []
    for half in range(601):
        v = half / 2
        for f in _num_forms(v):
            pairs.append((f, v))
    return Lexicon.build("totale", pairs, numeric=True)


@functools.cache
def pei_lexicon() -> Lexicon:
    """Ore settimanali da PEI 1-40 a passi di mezz'ora."""
    pairs: list[tuple[str, float]] = []
    for half in range(2, 81):
        v = half / 2
        for f in _num_forms(v):
            pairs.append((f, v))
            pairs.append((f + "h", v))
    return Lexicon.build("ore_pei", pairs, numeric=True)


@functools.cache
def small_int_lexicon(maximum: int, roman: bool = False) -> Lexicon:
    """Numeri interi 1..maximum (lotto, municipalita'), anche "01" e romani."""
    pairs: list[tuple[str, str]] = []
    for n in range(1, maximum + 1):
        pairs += [(str(n), str(n)), (f"{n:02d}", str(n))]
        if roman and n <= len(_ROMANI):
            pairs.append((_ROMANI[n - 1], str(n)))
    return Lexicon.build(f"interi_{maximum}_{int(roman)}", pairs, numeric=not roman)


def _years_around(year: int | None) -> tuple[int, ...]:
    y = year or date.today().year
    return tuple(range(y - 2, y + 2))


_MONTH_SEPARATORS = ("/", "-", ".", ",", "|", "1") + tuple(a + b for a in "/.,-" for b in "/.,-" if a != b)


@functools.cache
def month_year_lexicon(years: tuple[int, ...]) -> Lexicon:
    """Mese/anno di riferimento ("02/2026", "2/26", "febbraio 2026"...)."""
    pairs: list[tuple[str, tuple[int, int]]] = []
    for y in years:
        for m in range(1, 13):
            v = (m, y)
            for mm in dict.fromkeys((str(m), f"{m:02d}")):
                for yy in (str(y), f"{y % 100:02d}"):
                    for sep in _MONTH_SEPARATORS:
                        pairs.append((f"{mm}{sep}{yy}", v))
            pairs.append((f"{MESI_NOMI[m - 1]} {y}", v))
    return Lexicon.build(f"mese_anno_{years[0]}_{years[-1]}", pairs)


@functools.cache
def date_lexicon(years: tuple[int, ...]) -> Lexicon:
    """Date gg/mm/aaaa (data di compilazione "Napoli, __/__/____")."""
    pairs: list[tuple[str, str]] = []
    for y in years:
        for m in range(1, 13):
            for d in range(1, calendar.monthrange(y, m)[1] + 1):
                v = f"{d:02d}/{m:02d}/{y}"
                for dd in dict.fromkeys((str(d), f"{d:02d}")):
                    for mm in dict.fromkeys((str(m), f"{m:02d}")):
                        for yy in (str(y), f"{y % 100:02d}"):
                            for sep in ("/", "."):
                                pairs.append((f"{dd}{sep}{mm}{sep}{yy}", v))
    return Lexicon.build(f"date_{years[0]}_{years[-1]}", pairs, numeric=True)


@functools.cache
def notes_lexicon() -> Lexicon:
    pairs = [(form, canon) for canon, forms in NOTE_FREQUENTI for form in forms]
    return Lexicon.build("note", pairs)


# ==========================================================================
# Richieste di lettura e risultati del riconoscitore
# ==========================================================================

@dataclass
class ReadRequest:
    """Un ritaglio da leggere con TrOCR."""

    key: str                          # es. "rows.5.eff_uscita", "header.operatore"
    image: np.ndarray                 # ritaglio pulito, grigi uint8 (inchiostro scuro su bianco)
    lexicon: Lexicon | None = None    # decodifica vincolata (None = solo lettura libera)
    beams: int = 6                    # ampiezza del beam vincolato
    free: bool = True                 # eseguire anche la lettura libera
    free_beams: int = 1
    charset: frozenset[str] | None = None   # caratteri ammessi nella lettura libera (None = tutti)
    max_tokens: int = 10              # lunghezza massima della lettura libera


@dataclass
class Reading:
    """Esito della lettura di un ritaglio."""

    candidates: list[tuple[Any, float]] = field(default_factory=list)  # (valore, log-prob) per valore, decrescenti
    free_text: str = ""
    free_logprob: float = -math.inf
    free_tokens: int = 0

    @property
    def free_confidence(self) -> float:
        """Probabilita' media per token della lettura libera (0..1)."""
        if not self.free_text or self.free_tokens <= 0 or not math.isfinite(self.free_logprob):
            return 0.0
        return float(math.exp(self.free_logprob / (self.free_tokens + 1)))


class Recognizer(Protocol):
    def read(self, requests: Sequence[ReadRequest],
             progress: Callable[[int, int], None] | None = None) -> list[Reading]:
        ...


def _logsumexp(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return -math.inf
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


# ==========================================================================
# Riconoscitore TrOCR con decodifica vincolata
# ==========================================================================

class _Hyp:
    __slots__ = ("ids", "key", "score")

    def __init__(self, ids: tuple[int, ...], key: str, score: float):
        self.ids = ids
        self.key = key
        self.score = score


class TrOCRRecognizer:
    """TrOCR su CPU con beam search vincolato a un lessico.

    Per ogni ritaglio: l'encoder gira una volta (a lotti), il decoder elabora
    il token iniziale una volta sola e le chiavi/valori della cross-attention
    vengono riusati da tutte le ipotesi (``EncoderDecoderCache``).
    """

    ENCODER_BATCH = 8

    def __init__(self, model: Any, processor: Any, model_name: str, device: str = "cpu"):
        import torch  # noqa: PLC0415

        self.torch = torch
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.model_name = model_name
        self.device = torch.device(device)
        self.decoder = model.decoder
        self._enc_proj = getattr(model, "enc_to_dec_proj", None)
        gen = getattr(model, "generation_config", None)
        cfg = model.config
        start = getattr(gen, "decoder_start_token_id", None)
        if start is None:
            start = getattr(cfg, "decoder_start_token_id", None)
        eos = getattr(gen, "eos_token_id", None)
        if eos is None:
            eos = self.tokenizer.eos_token_id
        if isinstance(eos, (list, tuple)):
            eos = eos[0]
        if start is None:
            start = eos
        self.start_id = int(start)
        self.eos_id = int(eos)
        vocab = int(self.decoder.config.vocab_size)
        n_tok = min(vocab, len(self.tokenizer))
        texts = self.tokenizer.batch_decode([[i] for i in range(n_tok)], skip_special_tokens=False,
                                            clean_up_tokenization_spaces=False)
        texts += [""] * (vocab - n_tok)
        special = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        for i in special:
            if 0 <= i < vocab:
                texts[i] = ""
        self.vocab = vocab
        self.tok_text: list[str] = texts
        self.tok_key: list[str] = [_key(t) for t in texts]
        self._lex_tokens: dict[str, list[tuple[int, str]]] = {}
        self._allowed: dict[tuple[str, str], Any] = {}
        self._charset_ids: dict[frozenset[str] | None, Any] = {}
        self._fast = True
        self.lock = threading.Lock()

    # ------------------------------------------------------------- vincoli
    def _lexicon_tokens(self, lex: Lexicon) -> list[tuple[int, str]]:
        toks = self._lex_tokens.get(lex.name)
        if toks is None:
            alpha = lex.alphabet
            toks = [(i, k) for i, k in enumerate(self.tok_key) if k and set(k) <= alpha]
            self._lex_tokens[lex.name] = toks
        return toks

    def _allowed_for(self, lex: Lexicon, prefix: str) -> Any:
        ck = (lex.name, prefix)
        ids = self._allowed.get(ck)
        if ids is None:
            sel = [i for i, k in self._lexicon_tokens(lex) if lex.is_prefix(prefix + k)]
            ids = self.torch.tensor(sel, dtype=self.torch.long)
            self._allowed[ck] = ids
        return ids

    def _charset_for(self, charset: frozenset[str] | None) -> Any:
        ids = self._charset_ids.get(charset)
        if ids is None:
            if charset is None:
                sel = [i for i, t in enumerate(self.tok_text) if t]
            else:
                sel = [i for i, t in enumerate(self.tok_text) if t and set(t) <= charset]
            sel = [i for i in sel if i != self.eos_id]
            ids = self.torch.tensor(sel, dtype=self.torch.long)
            self._charset_ids[charset] = ids
        return ids

    # ------------------------------------------------------------ encoder
    def _pixel_values(self, images: Sequence[np.ndarray]) -> Any:
        from PIL import Image  # noqa: PLC0415

        pil = [Image.fromarray(_to_rgb(img)) for img in images]
        return self.processor(images=pil, return_tensors="pt").pixel_values.to(self.device)

    def encode(self, images: Sequence[np.ndarray]) -> Any:
        pv = self._pixel_values(images)
        out = self.model.encoder(pixel_values=pv)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        if self._enc_proj is not None:
            hidden = self._enc_proj(hidden)
        return hidden

    # ------------------------------------------------------------ decoder
    def _caches(self) -> tuple[Any, Any, Any]:
        from transformers.cache_utils import (  # noqa: PLC0415
            DynamicCache,
            EncoderDecoderCache,
        )

        return DynamicCache, EncoderDecoderCache, self.decoder.config

    def _start(self, enc1: Any) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]], Any]:
        """Elabora il token iniziale: (chiavi/valori self-attn, cross-attn, log-prob)."""
        torch = self.torch
        dyn, encdec, cfg = self._caches()
        cache = encdec(dyn(config=cfg), dyn(config=cfg))
        ids = torch.tensor([[self.start_id]], dtype=torch.long, device=self.device)
        out = self.decoder(input_ids=ids, encoder_hidden_states=enc1, past_key_values=cache, use_cache=True)
        lp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)[0]
        self_kv = [(layer.keys, layer.values) for layer in cache.self_attention_cache.layers]
        cross_kv = [(layer.keys, layer.values) for layer in cache.cross_attention_cache.layers]
        if not self_kv or not cross_kv or len(self_kv) != len(cross_kv):
            raise RuntimeError("cache del decoder in formato inatteso")
        return self_kv, cross_kv, lp

    def _make_cache(self, self_kv: list[tuple[Any, Any]], cross_kv: list[tuple[Any, Any]], width: int) -> Any:
        """Cache per ``width`` ipotesi: le chiavi/valori della cross-attention (uguali
        per tutte) vengono copiati una volta sola, non ricalcolati a ogni passo."""
        dyn, encdec, _ = self._caches()
        cross = [(k.expand(width, -1, -1, -1), v.expand(width, -1, -1, -1)) for k, v in cross_kv]
        selfc = [(k.expand(width, -1, -1, -1), v.expand(width, -1, -1, -1)) for k, v in self_kv]
        cache = encdec(dyn(selfc), dyn(cross))
        for i in range(len(cross)):
            cache.is_updated[i] = True
        return cache

    def _search(self, enc1: Any, start: tuple[Any, Any, Any], width: int, max_steps: int,
                lexicon: Lexicon | None, charset: frozenset[str] | None,
                length_norm: bool) -> list[tuple[str, tuple[int, ...], float]]:
        """Beam search (vincolato al lessico o al set di caratteri).

        Restituisce le ipotesi concluse: (chiave del testo, token, log-prob totale)."""
        torch = self.torch
        self_kv, cross_kv, lp0 = start
        live = [_Hyp((), "", 0.0)]
        lps = lp0.unsqueeze(0)
        finished: list[tuple[str, tuple[int, ...], float]] = []
        cache = None
        enc_view = None
        eos = self.eos_id
        static_ids = None if lexicon is not None else self._charset_for(charset)
        for step in range(max_steps + 1):
            cand_scores = []
            cand_parent = []
            cand_tok = []
            for i, h in enumerate(live):
                if not math.isfinite(h.score):
                    continue
                if lexicon is not None:
                    allowed = self._allowed_for(lexicon, h.key)
                    eos_ok = lexicon.complete(h.key) is not None
                else:
                    allowed = static_ids
                    eos_ok = bool(h.key)
                if eos_ok:
                    finished.append((h.key, h.ids, h.score + float(lps[i, eos])))
                if allowed.numel() and step < max_steps:
                    cand_scores.append(lps[i, allowed] + h.score)
                    cand_parent.append(torch.full((allowed.numel(),), i, dtype=torch.long))
                    cand_tok.append(allowed)
            if not cand_scores:
                break
            scores = torch.cat(cand_scores)
            parents = torch.cat(cand_parent)
            toks = torch.cat(cand_tok)
            k = min(width * 4 if lexicon is not None else width, int(scores.numel()))
            top_s, top_i = torch.topk(scores, k)
            best_live = float(top_s[0])
            if len(finished) >= width:
                if lexicon is not None:
                    # conta i valori distinti gia' conclusi (le varianti di scrittura non contano)
                    best_by_value: dict[Any, float] = {}
                    for fk, _fi, fs in finished:
                        fv = lexicon.complete(fk)
                        best_by_value[fv] = max(fs, best_by_value.get(fv, -math.inf))
                    ranked = sorted(best_by_value.values(), reverse=True)
                else:
                    ranked = sorted((self._rank(f, length_norm) for f in finished), reverse=True)
                bound = best_live if not length_norm else best_live / (step + 2)
                if len(ranked) >= width and ranked[width - 1] >= bound:
                    break
            new_live: list[_Hyp] = []
            sel_parents: list[int] = []
            sel_tokens: list[int] = []
            seen: set[str] = set()
            for s, j in zip(top_s.tolist(), top_i.tolist()):
                if not math.isfinite(s):
                    continue
                p = int(parents[j])
                t = int(toks[j])
                h = live[p]
                key = h.key + self.tok_key[t]
                if lexicon is not None:
                    # una sola ipotesi per "contenuto": le varianti (8:00 / 8.00 / 8,00)
                    # non devono occupare tutto il beam a scapito di valori diversi
                    sig = lexicon.signature(key)
                    if sig in seen:
                        continue
                    seen.add(sig)
                new_live.append(_Hyp(h.ids + (t,), key, s))
                sel_parents.append(p)
                sel_tokens.append(t)
                if len(new_live) >= width:
                    break
            if not new_live:
                break
            # larghezza fissa: le posizioni libere ripetono l'ultima ipotesi (punteggio -inf)
            while len(new_live) < width:
                new_live.append(_Hyp(new_live[-1].ids, new_live[-1].key, -math.inf))
                sel_parents.append(sel_parents[-1])
                sel_tokens.append(sel_tokens[-1])
            idx = torch.tensor(sel_parents, dtype=torch.long, device=self.device)
            if cache is None:
                cache = self._make_cache([(k_.index_select(0, idx[:1]), v_.index_select(0, idx[:1]))
                                          for k_, v_ in self_kv], cross_kv, width)
                enc_view = enc1.expand(width, -1, -1)
            else:
                cache.self_attention_cache.batch_select_indices(idx)
            inp = torch.tensor(sel_tokens, dtype=torch.long, device=self.device).unsqueeze(1)
            out = self.decoder(input_ids=inp, encoder_hidden_states=enc_view, past_key_values=cache,
                               use_cache=True)
            lps = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
            live = new_live
        return finished

    @staticmethod
    def _rank(item: tuple[str, tuple[int, ...], float], length_norm: bool) -> float:
        _, ids, score = item
        return score / (len(ids) + 1) if length_norm else score

    def _search_slow(self, enc1: Any, width: int, max_steps: int, lexicon: Lexicon | None,
                     charset: frozenset[str] | None) -> list[tuple[str, tuple[int, ...], float]]:
        """Ripiego senza cache (usato solo se l'API della cache cambia)."""
        torch = self.torch
        live = [_Hyp((), "", 0.0)]
        finished: list[tuple[str, tuple[int, ...], float]] = []
        static_ids = None if lexicon is not None else self._charset_for(charset)
        for step in range(max_steps + 1):
            seqs = torch.tensor([[self.start_id, *h.ids] for h in live], dtype=torch.long, device=self.device)
            out = self.decoder(input_ids=seqs, encoder_hidden_states=enc1.expand(len(live), -1, -1),
                               use_cache=False)
            lps = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
            cands: list[tuple[float, int, int]] = []
            for i, h in enumerate(live):
                if lexicon is not None:
                    allowed = self._allowed_for(lexicon, h.key)
                    eos_ok = lexicon.complete(h.key) is not None
                else:
                    allowed = static_ids
                    eos_ok = bool(h.key)
                if eos_ok:
                    finished.append((h.key, h.ids, h.score + float(lps[i, self.eos_id])))
                if allowed.numel() and step < max_steps:
                    s = lps[i, allowed] + h.score
                    k = min(width, int(s.numel()))
                    ts, ti = torch.topk(s, k)
                    cands += [(float(a), i, int(allowed[b])) for a, b in zip(ts, ti)]
            if not cands:
                break
            cands.sort(reverse=True)
            live = [_Hyp(live[i].ids + (t,), live[i].key + self.tok_key[t], s) for s, i, t in cands[:width]]
        return finished

    def _decode_ids(self, ids: Sequence[int]) -> str:
        text = self.tokenizer.decode(list(ids), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return " ".join(text.split())

    def _read_one(self, enc1: Any, req: ReadRequest) -> Reading:
        reading = Reading()
        start = None
        if self._fast:
            try:
                start = self._start(enc1)
            except Exception:  # noqa: BLE001 - API interna di transformers cambiata: ripiego lento
                log.warning("Decodifica rapida non disponibile, uso il ripiego lento", exc_info=True)
                self._fast = False
        lex = req.lexicon
        if lex is not None:
            steps = lex.max_chars
            if start is not None:
                fin = self._search(enc1, start, max(1, req.beams), steps, lex, None, False)
            else:
                fin = self._search_slow(enc1, max(1, req.beams), steps, lex, None)
            by_value: dict[Any, list[float]] = {}
            for key, _ids, score in fin:
                value = lex.complete(key)
                if value is not None:
                    by_value.setdefault(value, []).append(score)
            cands = [(v, _logsumexp(s)) for v, s in by_value.items()]
            cands.sort(key=lambda c: c[1], reverse=True)
            reading.candidates = cands
            best = max(fin, key=lambda f: f[2], default=None)
            if (best is not None and best[2] > math.log(0.5) and req.charset is None
                    and req.free_beams <= 1):
                # una sequenza con probabilita' > 1/2 e' per forza quella della lettura
                # libera "golosa": inutile ricalcolarla
                reading.free_text = self._decode_ids(best[1])
                reading.free_logprob = best[2]
                reading.free_tokens = len(best[1])
                return reading
        if req.free or lex is None:
            width = max(1, req.free_beams)
            if start is not None:
                fin = self._search(enc1, start, width, req.max_tokens, None, req.charset, True)
            else:
                fin = self._search_slow(enc1, width, req.max_tokens, None, req.charset)
            if fin:
                best = max(fin, key=lambda f: self._rank(f, True))
                reading.free_text = self._decode_ids(best[1])
                reading.free_logprob = best[2]
                reading.free_tokens = len(best[1])
        return reading

    def read(self, requests: Sequence[ReadRequest],
             progress: Callable[[int, int], None] | None = None) -> list[Reading]:
        torch = self.torch
        results: list[Reading] = []
        total = len(requests)
        with self.lock, torch.inference_mode():
            for b0 in range(0, total, self.ENCODER_BATCH):
                batch = requests[b0:b0 + self.ENCODER_BATCH]
                enc = self.encode([r.image for r in batch])
                for j, req in enumerate(batch):
                    results.append(self._read_one(enc[j:j + 1], req))
                    if progress is not None:
                        progress(b0 + j + 1, total)
        return results


def _to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ==========================================================================
# Gestione del modello (cache su disco, download, caricamento)
# ==========================================================================

_MODELS: dict[tuple[str, str, str], TrOCRRecognizer] = {}
_MODELS_LOCK = threading.Lock()
_DOWNLOAD_LOCKS: dict[str, threading.Lock] = {}
_IMPORT_STATE: dict[str, Any] = {}


def default_cache_dir() -> Path:
    from sirio.config import data_dir  # noqa: PLC0415

    return data_dir() / "modelli"


def _repo_folder(model_name: str) -> str:
    return "models--" + model_name.replace("/", "--")


def _snapshot_complete(path: Path) -> bool:
    try:
        names = {p.name for p in path.iterdir()}
    except OSError:
        return False
    return ("config.json" in names and "preprocessor_config.json" in names
            and any(w in names for w in _WEIGHT_FILES) and any(t in names for t in _TOKENIZER_FILES))


def local_model_path(model_name: str, cache_dir: Path | None) -> Path | None:
    """Cartella locale completa del modello, se gia' scaricato (o se ``model_name`` e' una cartella)."""
    direct = Path(model_name).expanduser()
    if direct.is_dir() and _snapshot_complete(direct):
        return direct
    if cache_dir is None:
        return None
    for base in (cache_dir, cache_dir / "hub"):
        repo = base / _repo_folder(model_name)
        snaps = repo / "snapshots"
        if not snaps.is_dir():
            continue
        preferred: list[Path] = []
        ref = repo / "refs" / "main"
        try:
            preferred.append(snaps / ref.read_text(encoding="utf-8").strip())
        except OSError:
            pass
        try:
            others = sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            others = []
        for snap in preferred + others:
            if snap.is_dir() and _snapshot_complete(snap):
                return snap
    return None


def _missing_components() -> str | None:
    """None se torch e transformers sono importabili, altrimenti il messaggio d'errore."""
    if "ok" in _IMPORT_STATE:
        return None if _IMPORT_STATE["ok"] else _MISSING_MSG
    for mod in ("torch", "transformers"):
        if importlib.util.find_spec(mod) is None:
            _IMPORT_STATE["ok"] = False
            return _MISSING_MSG
    try:
        importlib.import_module("torch")
        importlib.import_module("transformers")
    except Exception:  # noqa: BLE001 - libreria presente ma non funzionante
        log.warning("torch/transformers presenti ma non importabili", exc_info=True)
        _IMPORT_STATE["ok"] = False
        return _MISSING_MSG
    _IMPORT_STATE["ok"] = True
    return None


def _is_network_error(exc: BaseException) -> bool:
    names = {c.__name__ for c in type(exc).__mro__}
    network = {
        "ConnectError", "ConnectTimeout", "ReadTimeout", "TimeoutException", "NetworkError",
        "ConnectionError", "ProxyError", "SSLError", "Timeout", "TimeoutError", "OfflineModeIsEnabled",
        "LocalEntryNotFoundError", "RemoteProtocolError", "TransportError", "gaierror",
    }
    if names & network:
        return True
    cause = exc.__cause__ or exc.__context__
    return cause is not None and cause is not exc and _is_network_error(cause)


def _download_error(exc: BaseException, model_name: str, size_mb: float | None) -> EngineError:
    size = f" (circa {_fmt_size(size_mb)})" if size_mb else ""
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
        return EngineError(
            f"Spazio su disco insufficiente per scaricare il modello di riconoscimento offline{size}. "
            "Liberare spazio e riprovare."
        )
    names = {c.__name__ for c in type(exc).__mro__}
    if names & {"RepositoryNotFoundError", "RevisionNotFoundError", "GatedRepoError"}:
        return EngineError(
            f"Il modello «{model_name}» non è disponibile su huggingface.co: verificare il nome del "
            "modello locale nelle impostazioni."
        )
    if _is_network_error(exc):
        return EngineError(
            "Impossibile scaricare il modello per il riconoscimento offline: nessuna connessione a "
            f"Internet o huggingface.co non raggiungibile. Il download{size} serve solo al primo "
            "utilizzo: collegarsi a Internet e riprovare, oppure usare il motore Claude."
        )
    return EngineError(f"Download del modello di riconoscimento offline non riuscito: {exc}")


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _weight_patterns(files: Sequence[str]) -> list[str]:
    base = ["*.json", "*.txt", "*.model"]
    if any(f.endswith(".safetensors") for f in files):
        return base + ["*.safetensors"]
    return base + ["*.bin"]


def _silent_tqdm() -> Any:
    """Barre di avanzamento disattivate: l'avanzamento passa da ``progress`` e
    l'applicazione puo' girare senza console (stderr assente)."""
    from tqdm.auto import tqdm  # noqa: PLC0415

    class _Silent(tqdm):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

    return _Silent


def download_model(model_name: str, cache_dir: Path, progress: ProgressFn = no_progress) -> Path:
    """Scarica il modello da huggingface.co in ``cache_dir`` (solo i file necessari),
    riportando l'avanzamento. Restituisce la cartella del modello."""
    label = _model_label(model_name)
    size_mb = KNOWN_MODELS.get(model_name, {}).get("mb")
    try:
        from huggingface_hub import HfApi, snapshot_download  # noqa: PLC0415
    except ImportError as exc:
        raise EngineError(_MISSING_MSG) from exc
    progress(0.0, f"Connessione a huggingface.co per scaricare il modello {label}…")
    try:
        info = HfApi().model_info(model_name, files_metadata=True)
    except Exception as exc:  # noqa: BLE001
        raise _download_error(exc, model_name, size_mb) from exc
    siblings = [(s.rfilename, int(s.size or 0)) for s in (info.siblings or [])]
    patterns = _weight_patterns([f for f, _ in siblings])
    total = sum(sz for f, sz in siblings if any(fnmatch.fnmatch(f, p) for p in patterns)) or (
        (size_mb or 0) * 1e6)
    cache_dir.mkdir(parents=True, exist_ok=True)
    outcome: dict[str, Any] = {}

    def worker() -> None:
        try:
            outcome["path"] = snapshot_download(model_name, cache_dir=str(cache_dir), allow_patterns=patterns,
                                                revision=getattr(info, "sha", None),
                                                tqdm_class=_silent_tqdm())
        except BaseException as exc:  # noqa: BLE001 - riportata nel thread chiamante
            outcome["error"] = exc

    blobs = cache_dir / _repo_folder(model_name) / "blobs"
    start_size = _dir_size(blobs)
    thread = threading.Thread(target=worker, name="sirio-download-modello", daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(0.5)
        done = max(0, _dir_size(blobs) - start_size)
        frac = min(0.99, done / total) if total else 0.0
        msg = f"Download del modello {label} (solo al primo utilizzo): {_fmt_mb(done)}"
        if total:
            msg += f" di {_fmt_mb(total)}"
        progress(frac, msg + "…")
    if "error" in outcome:
        raise _download_error(outcome["error"], model_name, size_mb) from outcome["error"]
    path = local_model_path(model_name, cache_dir)
    if path is None:
        raise EngineError(f"Download del modello {label} incompleto: riprovare.")
    progress(1.0, f"Modello {label} scaricato.")
    return path


def _load_recognizer(path: Path, model_name: str, device: str, quantize: bool = True) -> TrOCRRecognizer:
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    try:
        # niente barre di avanzamento su stderr (l'applicazione puo' non avere una console)
        transformers.utils.logging.disable_progress_bar()
    except Exception:  # noqa: BLE001
        pass
    try:
        processor = transformers.TrOCRProcessor.from_pretrained(str(path), local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        text = str(exc).lower()
        if "sentencepiece" in text or "tiktoken" in text or "protobuf" in text:
            raise EngineError(
                f"Il modello {_model_label(model_name)} richiede i pacchetti «sentencepiece» e «protobuf», "
                "non installati: scegliere un altro modello locale nelle impostazioni."
            ) from exc
        raise EngineError(f"Modello di riconoscimento offline non valido o incompleto: {exc}") from exc
    try:
        model = transformers.VisionEncoderDecoderModel.from_pretrained(str(path), local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        raise EngineError(
            f"Impossibile caricare il modello di riconoscimento offline ({exc}). Se il download era stato "
            "interrotto, eliminare la cartella dei modelli e riprovare."
        ) from exc
    model.eval()
    model.to(torch.device(device))
    if quantize and torch.device(device).type == "cpu":
        _quantize_decoder(model)
    return TrOCRRecognizer(model, processor, model_name, device)


def _quantize_decoder(model: Any) -> None:
    """Quantizzazione dinamica int8 dei livelli lineari del *solo* decoder: circa
    2x piu' veloce su CPU con letture identiche (l'encoder resta in float32:
    quantizzato peggiora sensibilmente il riconoscimento)."""
    import warnings  # noqa: PLC0415

    torch = importlib.import_module("torch")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from torch.ao.quantization import quantize_dynamic  # noqa: PLC0415

            model.decoder = quantize_dynamic(model.decoder, {torch.nn.Linear}, dtype=torch.qint8)
    except Exception:  # noqa: BLE001 - facoltativa: senza quantizzazione funziona comunque
        log.info("Quantizzazione del decoder non disponibile, uso float32", exc_info=True)


# ==========================================================================
# Ritagli puliti
# ==========================================================================

def _paper_level(gray: np.ndarray) -> float:
    bg = float(np.percentile(gray, 90))
    if bg < 80:
        bg = max(bg, float(np.percentile(gray, 99)))
    return bg


def _ink_threshold(bg: float) -> float:
    return max(35.0, bg - max(45.0, 0.2 * bg))


def _line_mask(ink_mask: np.ndarray, unit: float, rows: Sequence[float], cols: Sequence[float],
               split: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Pixel delle linee stampate: tratti rettilinei molto lunghi ovunque, tratti
    lunghi vicino alle linee note della griglia. Con ``split`` restituisce
    separatamente (orizzontali, verticali)."""
    m = ink_mask.astype(np.uint8)
    h, w = m.shape
    far = max(9, int(round(1.6 * unit)))
    near_h = max(7, int(round(0.8 * unit)))
    near_v = max(7, int(round(0.9 * unit)))
    band = max(3, int(round(0.2 * unit)))
    hl = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (far, 1)))
    vl = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, far)))
    if rows:
        hn = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (near_h, 1)))
        keep = np.zeros(h, dtype=bool)
        for y in rows:
            keep[max(0, int(y) - band):max(0, int(y) + band + 1)] = True
        hn[~keep, :] = 0
        hl |= hn
    if cols:
        vn = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, near_v)))
        keep = np.zeros(w, dtype=bool)
        for x in cols:
            keep[max(0, int(x) - band):max(0, int(x) + band + 1)] = True
        vn[:, ~keep] = 0
        vl |= vn
    t = max(1, int(round(0.025 * unit)))
    hl = cv2.dilate(hl, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2 * t + 1)))
    vl = cv2.dilate(vl, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * t + 1, 3)))
    if split:
        return hl > 0, vl > 0
    return (hl | vl) > 0


def clean_crop(gray: np.ndarray, box: Box, unit: float, rows: Sequence[float] = (), cols: Sequence[float] = (),
               extend: tuple[float, float] = (0.5, 0.5), min_share: float = 0.5) -> np.ndarray | None:
    """Ritaglio della sola scrittura che appartiene a ``box``: inchiostro scuro su
    bianco, linee della griglia tolte, tratti per lo piu' esterni esclusi.

    ``unit`` e' l'altezza di una riga della tabella (px); ``rows``/``cols`` sono le
    ordinate/ascisse (immagine) delle linee della griglia vicine. Restituisce
    ``None`` se nel riquadro non c'e' scrittura.
    """
    H, W = gray.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    ex0 = max(0, int(x0 - extend[0] * unit))
    ex1 = min(W, int(x1 + extend[0] * unit))
    ey0 = max(0, int(y0 - extend[1] * unit))
    ey1 = min(H, int(y1 + extend[1] * unit))
    if ex1 - ex0 < 4 or ey1 - ey0 < 4:
        return None
    region = gray[ey0:ey1, ex0:ex1]
    bg = _paper_level(region)
    ink_mask = region < _ink_threshold(bg)
    if not ink_mask.any():
        return None
    h_lines, v_lines = _line_mask(ink_mask, unit, [y - ey0 for y in rows], [x - ex0 for x in cols], split=True)
    lines = h_lines | v_lines
    core = ink_mask & ~lines
    # Attribuzione dei tratti: le componenti vengono ricucite solo attraverso le
    # linee orizzontali (cifre o firme che scavalcano il bordo di riga), mai
    # attraverso quelle verticali (la scrittura appoggiata al bordo di colonna
    # si fonderebbe con quella delle righe vicine).
    r = max(2, int(round(0.1 * unit)))
    near = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2 * r + 1))) > 0
    conn = (core | (h_lines & ~v_lines & near)).astype(np.uint8)
    n, labels = cv2.connectedComponents(conn, connectivity=8)
    if n <= 1:
        return None
    lab = np.where(core, labels, 0)
    total = np.bincount(lab.ravel(), minlength=n)
    bx0, by0, bx1, by1 = x0 - ex0, y0 - ey0, x1 - ex0, y1 - ey0
    inside = np.bincount(lab[max(0, by0):max(0, by1), max(0, bx0):max(0, bx1)].ravel(), minlength=n)
    min_area = max(4.0, (0.035 * unit) ** 2)
    keep = np.zeros(n, dtype=bool)      # componente tenuta per intero
    clip = np.zeros(n, dtype=bool)      # componente condivisa: solo la parte nel riquadro
    for k in range(1, n):
        if total[k] < min_area or inside[k] == 0:
            continue
        share = inside[k] / float(total[k])
        if share >= min_share:
            keep[k] = True
        elif share >= 0.25 and inside[k] >= max(min_area, (0.12 * unit) ** 2):
            clip[k] = True
    sel = keep[lab]
    if clip.any():
        inner = np.zeros_like(sel)
        m = max(1, int(round(0.04 * unit)))
        inner[max(0, by0 + m):max(0, by1 - m), max(0, bx0):max(0, bx1)] = True
        sel |= clip[lab] & inner
    # frammenti di linea rimasti: sottili, allungati e addossati alle linee tolte
    if sel.any():
        line_near = cv2.dilate(lines.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(sel.astype(np.uint8), connectivity=8)
        thin = max(2.0, 0.07 * unit)
        for k in range(1, n2):
            _x, _y, w, h, area = (int(v) for v in stats2[k])
            comp = lab2 == k
            if area < min_area or (min(w, h) <= thin and max(w, h) >= 3 * min(w, h)
                                   and float((comp & line_near).sum()) > 0.5 * area):
                sel &= ~comp
    if not sel.any():
        return None
    ys, xs = np.nonzero(sel)
    cx0, cx1 = int(xs.min()), int(xs.max()) + 1
    cy0, cy1 = int(ys.min()), int(ys.max()) + 1
    # tratti antialiasing attorno all'inchiostro tenuto
    soft = (cv2.dilate(sel.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (region < bg - 15) & ~lines
    soft |= sel
    vals = region.astype(np.float32)
    lo = float(np.percentile(vals[sel], 5))
    span = max(20.0, bg - lo)
    stretched = np.clip(255.0 - (bg - vals) / span * 235.0, 0, 255)
    out = np.full(region.shape, 255, dtype=np.uint8)
    out[soft] = stretched[soft].astype(np.uint8)
    pad_x = max(2, int(round(0.22 * unit)))
    pad_y = max(2, int(round(0.15 * unit)))
    crop = out[cy0:cy1, cx0:cx1]
    return cv2.copyMakeBorder(crop, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=255)


def model_image(crop: np.ndarray, min_aspect: float = 1.0) -> np.ndarray:
    """Allarga con bianco i ritagli troppo stretti (TrOCR ridimensiona a quadrato)."""
    h, w = crop.shape[:2]
    if h <= 0 or w <= 0:
        return crop
    target = int(math.ceil(min_aspect * h))
    if w < target:
        extra = target - w
        crop = cv2.copyMakeBorder(crop, 0, 0, extra // 2, extra - extra // 2, cv2.BORDER_CONSTANT, value=255)
    return crop


# ==========================================================================
# Posizioni dei campi del modulo (misurate sul modulo reale)
# ==========================================================================

# Ascisse in frazione della larghezza della tabella (da col_x[0]); ordinate in
# altezze di riga rispetto al bordo superiore dell'intestazione della tabella
# (negative = sopra). Per ogni campo: (x0, x1, ordinata della riga di risposta).
HEADER_LAYOUT: dict[str, tuple[float, float, float]] = {
    "lotto": (0.262, 0.300, -3.42),
    "municipalita": (0.372, 0.745, -3.42),
    "ente": (0.250, 0.745, -2.65),
    "istituto": (0.340, 0.745, -1.88),
    "operatore": (0.405, 0.755, -1.13),
    "alunno": (0.383, 0.760, -0.38),
    "mese_anno": (0.772, 0.990, -2.71),
    "ore_pei": (0.848, 0.990, -1.19),
}
# Riquadri "Sostituzione: [SI] [NO]": ascisse (SI, divisorio, NO) e ordinate (alto, basso).
SOSTITUZIONE_LAYOUT = (0.880, 0.940, 1.000, -0.97, -0.22)
# Altezza della scrittura sopra la riga di risposta (in altezze di riga).
HEADER_TEXT_ABOVE = 0.62
HEADER_TEXT_BELOW = 0.12
# Fine dell'etichetta "Firma Coordinatore dell'Ente" (frazione della tabella).
COORD_LABEL_END = 0.62
REFERENT_LABEL_END = 0.62
DATE_LABEL_END = 0.105


@dataclass
class _HeaderSpot:
    box: Box
    baseline: float


def _header_lines(gray: np.ndarray, grid: TableGrid) -> list[tuple[float, float, float]]:
    """Linee orizzontali stampate sopra la tabella: (x0, x1, y) in pixel."""
    x0, y0, x1, y1 = grid.header_region()
    region = gray[y0:y1, x0:x1]
    if region.size == 0:
        return []
    bg = _paper_level(region)
    bw = (region < _ink_threshold(bg)).astype(np.uint8)
    tw = max(1, grid.col_x[-1] - grid.col_x[0])
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(0.045 * tw)), 1))
    hl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(hl, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = (int(v) for v in stats[i])
        if h <= max(3, 0.25 * grid.row_height):
            out.append((float(x + x0), float(x + w + x0), float(y + h / 2.0 + y0)))
    return out


def header_spots(gray: np.ndarray, grid: TableGrid) -> dict[str, _HeaderSpot]:
    """Riquadri della scrittura dei campi d'intestazione."""
    rh = grid.row_height
    tx0 = grid.col_x[0]
    tw = grid.col_x[-1] - grid.col_x[0]
    lines = _header_lines(gray, grid)
    spots: dict[str, _HeaderSpot] = {}
    for name, (fx0, fx1, fy) in HEADER_LAYOUT.items():
        ex0, ex1 = tx0 + fx0 * tw, tx0 + fx1 * tw
        ey = grid.header_top + fy * rh
        near = [ln for ln in lines
                if abs(ln[2] - ey) <= 0.3 * rh and min(ln[1], ex1) - max(ln[0], ex0) >= 0.3 * (ex1 - ex0)]
        if near:
            base = float(np.median([ln[2] for ln in near]))
            start = min(ln[0] for ln in near)
            end = max(ln[1] for ln in near)
            # la riga trovata delimita lo spazio della risposta (con un po' di tolleranza)
            if abs(start - ex0) <= 0.06 * tw:
                # per i campi con etichetta sulla stessa riga la risposta inizia dopo
                # l'etichetta, cioe' dove inizia la riga; mese/anno non ha etichetta
                # a sinistra e la scrittura puo' cominciare prima della riga
                ex0 = min(ex0, start - 0.03 * tw) if name == "mese_anno" else start - 0.002 * tw
            if abs(end - ex1) <= 0.06 * tw and name not in ("operatore", "alunno"):
                ex1 = max(ex1, end + 0.01 * tw)
        else:
            base = ey
        box = (int(round(ex0)), int(round(base - HEADER_TEXT_ABOVE * rh)),
               int(round(ex1)), int(round(base + HEADER_TEXT_BELOW * rh)))
        spots[name] = _HeaderSpot(box=box, baseline=base)
    return spots


def _clip_box(box: tuple[float, float, float, float], w: int, h: int) -> Box:
    x0, y0, x1, y1 = box
    xa = max(0, min(w - 1, int(round(min(x0, x1)))))
    ya = max(0, min(h - 1, int(round(min(y0, y1)))))
    xb = max(xa + 1, min(w, int(round(max(x0, x1)))))
    yb = max(ya + 1, min(h, int(round(max(y0, y1)))))
    return xa, ya, xb, yb


def _components(gray: np.ndarray, box: Box, unit: float, rows: Sequence[float] = (),
                cols: Sequence[float] = ()) -> tuple[float, list[tuple[int, int, int, int, int]]]:
    """Inchiostro (senza linee) nel riquadro: (frazione, componenti x, y, w, h, area)."""
    x0, y0, x1, y1 = box
    region = gray[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0, []
    bg = _paper_level(region)
    m = region < _ink_threshold(bg)
    m &= ~_line_mask(m, unit, [y - y0 for y in rows], [x - x0 for x in cols])
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    min_area = max(4.0, (0.04 * unit) ** 2)
    comps = [tuple(int(v) for v in stats[i]) for i in range(1, n) if stats[i][4] >= min_area]
    return float(m.mean()), comps  # type: ignore[return-value]


# ==========================================================================
# Analisi della pagina (senza OCR)
# ==========================================================================

TIME_FIELDS = ("prog_entrata", "prog_uscita", "eff_entrata", "eff_uscita")
TEXT_FIELDS = (*TIME_FIELDS, "ore_dichiarate")

# Stati delle celle dopo l'analisi dell'inchiostro.
VUOTA, TESTO, TRATTINO, CROCETTA, FIRMA, SEGNO, TIMBRO = (
    "vuota", "testo", "trattino", "crocetta", "firma", "segno", "timbro")


@dataclass
class PagePlan:
    """Esito dell'analisi d'immagine e ritagli da leggere."""

    cells: dict[int, dict[str, str]] = field(default_factory=dict)       # giorno -> campo -> stato
    requests: list[ReadRequest] = field(default_factory=list)
    header_ink: dict[str, bool] = field(default_factory=dict)            # campo -> scrittura presente
    sostituzione: str | None = None
    sostituzione_incerta: bool = False
    firma_coordinatore: bool = False
    timbro_referente: bool = False
    totale_ink: bool = False
    data_ink: bool = False
    notes: list[str] = field(default_factory=list)


def _cell_state_text(img: np.ndarray, box: Box, allow_dash: bool) -> str:
    if ink.is_blank(img, box):
        return VUOTA
    if allow_dash and ink.is_dash(img, box):
        return TRATTINO
    return TESTO


def _cell_state_cross(img: np.ndarray, box: Box) -> str:
    if ink.has_cross(img, box):
        return CROCETTA
    return VUOTA if ink.is_blank(img, box) else SEGNO


def _cell_state_signature(img: np.ndarray, box: Box) -> str:
    if ink.has_signature(img, box):
        return FIRMA
    return VUOTA if ink.is_blank(img, box) else SEGNO


def _text_request(key: str, crop: np.ndarray, lexicon: Lexicon | None, **kw: Any) -> ReadRequest:
    return ReadRequest(key=key, image=model_image(crop, kw.pop("min_aspect", 1.0)), lexicon=lexicon, **kw)


def _sostituzione(gray: np.ndarray, grid: TableGrid) -> tuple[str | None, bool]:
    """Crocetta nei riquadri "Sostituzione: [SI] [NO]": (valore, incerto).

    Le lettere stampate occupano la fascia centrale dei riquadri e "NO" ha
    circa 1,4-1,7 volte l'inchiostro di "SI": un segno si riconosce dall'inchiostro
    che esce dalla fascia delle lettere o dall'eccesso d'inchiostro in un riquadro
    rispetto all'altro. Nei casi dubbi il campo resta vuoto."""
    rh = grid.row_height
    tx0 = grid.col_x[0]
    tw = grid.col_x[-1] - grid.col_x[0]
    a, mid, b, top, bot = SOSTITUZIONE_LAYOUT
    ya, yb = grid.header_top + top * rh, grid.header_top + bot * rh
    H, W = gray.shape[:2]
    amount: dict[str, int] = {}
    outside: dict[str, float] = {}
    for label, (fx0, fx1) in (("SI", (a, mid)), ("NO", (mid, b))):
        bx0, bx1 = tx0 + fx0 * tw, tx0 + fx1 * tw
        box = _clip_box((bx0 + 0.06 * (bx1 - bx0), ya + 0.06 * (yb - ya),
                         bx1 - 0.06 * (bx1 - bx0), yb - 0.06 * (yb - ya)), W, H)
        x0, y0, x1, y1 = box
        region = gray[y0:y1, x0:x1]
        if region.size == 0:
            amount[label], outside[label] = 0, 0.0
            continue
        m = region < _ink_threshold(_paper_level(region))
        m &= ~_line_mask(m, rh, [ya - y0, yb - y0], [bx0 - x0, bx1 - x0])
        h, w = m.shape
        central = np.zeros_like(m)
        central[int(0.22 * h):int(0.78 * h), int(0.15 * w):int(0.85 * w)] = True
        amount[label] = int(m.sum())
        outside[label] = float((m & ~central).sum()) / max(1, amount[label])
    if not amount.get("SI") or not amount.get("NO"):
        return None, False
    # segno che esce dalla fascia delle lettere
    if max(outside.values()) >= 0.15 and min(outside.values()) < 0.05:
        value = max(outside, key=lambda k: outside[k])
        return value, max(outside.values()) < 0.25
    ratio = amount["NO"] / float(amount["SI"])
    if ratio < 1.25:
        return "SI", ratio > 1.1
    if ratio > 2.0:
        return "NO", ratio < 2.3
    return None, False


def stamp_boxes(gray: np.ndarray, grid: TableGrid) -> list[Box]:
    """Timbri (grandi macchie tondeggianti) nella parte bassa a destra del foglio:
    colonna "Note", margine destro e pie' di pagina."""
    H, W = gray.shape[:2]
    rh = grid.row_height
    zx0 = max(0, int(grid.col_x[9] - 0.3 * rh))
    zy0 = max(0, int(grid.row_y[15]))
    region = gray[zy0:H, zx0:W]
    if region.size == 0:
        return []
    bg = _paper_level(region)
    m = region < _ink_threshold(bg)
    m &= ~_line_mask(m, rh, [], [])
    k = max(3, int(round(0.3 * rh)))
    merged = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    out: list[Box] = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if w >= 1.5 * rh and h >= 1.5 * rh and 0.5 <= w / float(h) <= 2.0 and area >= 0.3 * w * h:
            out.append((x + zx0, y + zy0, x + w + zx0, y + h + zy0))
    return out


def _inside_stamp(gray: np.ndarray, box: Box, stamps: Sequence[Box]) -> bool:
    """True se l'inchiostro del riquadro sta per lo piu' dentro un timbro."""
    if not stamps:
        return False
    x0, y0, x1, y1 = box
    region = gray[y0:y1, x0:x1]
    if region.size == 0:
        return False
    m = region < _ink_threshold(_paper_level(region))
    total = int(m.sum())
    if total == 0:
        return False
    covered = np.zeros_like(m)
    grow = int(round(0.6 * (y1 - y0)))     # frammenti dell'anello attorno al timbro
    for sx0, sy0, sx1, sy1 in stamps:
        covered[max(0, sy0 - grow - y0):max(0, sy1 + grow - y0),
                max(0, sx0 - grow - x0):max(0, sx1 + grow - x0)] = True
    return float((m & covered).sum()) >= 0.6 * total


def _signature_like(comps: Sequence[tuple[int, int, int, int, int]], unit: float) -> bool:
    big = [c for c in comps if max(c[2], c[3]) >= 0.35 * unit]
    area = sum(c[4] for c in big)
    return bool(big) and area >= 0.04 * unit * unit


def _label_end(gray: np.ndarray, box: Box, unit: float) -> int:
    """Fine dell'etichetta stampata all'inizio del riquadro: primo spazio vuoto
    largo dopo la scritta (ascissa in pixel)."""
    x0, y0, x1, y1 = box
    region = gray[y0:y1, x0:x1]
    if region.size == 0:
        return x0
    bg = _paper_level(region)
    m = region < _ink_threshold(bg)
    m &= ~_line_mask(m, unit, [0, y1 - y0 - 1], [])
    cols = m.any(axis=0)
    idx = np.nonzero(cols)[0]
    if idx.size == 0:
        return x0
    gap = max(4, int(0.55 * unit))
    last = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - last > gap:
            return x0 + last + 1
        last = i
    return x0 + last + 1


def plan_page(img: np.ndarray, grid: TableGrid) -> PagePlan:
    """Analizza l'inchiostro della pagina e prepara i ritagli da leggere."""
    gray = to_gray(img)
    H, W = gray.shape[:2]
    rh = grid.row_height
    plan = PagePlan()
    t_lex, h_lex, n_lex = time_lexicon(), hours_lexicon(), notes_lexicon()
    stamps = stamp_boxes(gray, grid)
    for g in range(1, 32):
        states: dict[str, str] = {}
        y0, y1 = grid.row_y[g - 1], grid.row_y[g]
        for f in TEXT_FIELDS:
            box = grid.cell(g, f)
            st = _cell_state_text(img, box, allow_dash=True)
            states[f] = st
            if st != TESTO:
                continue
            c = grid.col_index(f)
            crop = clean_crop(gray, box, rh, rows=(y0, y1), cols=(grid.col_x[c], grid.col_x[c + 1]))
            if crop is None:
                states[f] = VUOTA
                continue
            if f == "ore_dichiarate":
                plan.requests.append(_text_request(f"rows.{g}.{f}", crop, h_lex, beams=5, max_tokens=6))
            else:
                plan.requests.append(_text_request(f"rows.{g}.{f}", crop, t_lex, beams=6, max_tokens=8))
        for f in ("assenza_alunno", "assenza_operatore"):
            states[f] = _cell_state_cross(img, grid.cell(g, f))
        states["firma"] = _cell_state_signature(img, grid.cell(g, "firma"))
        box = grid.cell(g, "note")
        st = _cell_state_text(img, box, allow_dash=True)
        if st == TESTO and _inside_stamp(gray, box, stamps):
            st = TIMBRO
        if st == TESTO:
            c = grid.col_index("note")
            crop = clean_crop(gray, box, rh, rows=(y0, y1), cols=(grid.col_x[c], grid.col_x[c + 1]),
                              extend=(0.3, 0.5))
            if crop is None:
                st = VUOTA
            else:
                plan.requests.append(_text_request(f"rows.{g}.note", crop, n_lex, beams=4, free_beams=3,
                                                   charset=NOTE_CHARS, max_tokens=16, min_aspect=2.0))
        states["note"] = st
        plan.cells[g] = states

    # --- intestazione
    spots = header_spots(gray, grid)
    years = _years_around(None)
    for name, spot in spots.items():
        box = _clip_box(spot.box, W, H)
        crop = clean_crop(gray, box, rh, rows=(spot.baseline,), cols=(), extend=(0.25, 0.45))
        plan.header_ink[name] = crop is not None
        if crop is None:
            continue
        key = f"header.{name}"
        if name in ("operatore", "alunno", "ente", "istituto"):
            plan.requests.append(_text_request(key, crop, None, free_beams=4, charset=NAME_CHARS,
                                               max_tokens=24, min_aspect=2.0))
        elif name == "lotto":
            plan.requests.append(_text_request(key, crop, small_int_lexicon(30), beams=4, max_tokens=4))
        elif name == "municipalita":
            plan.requests.append(_text_request(key, crop, small_int_lexicon(10, roman=True), beams=4,
                                               max_tokens=4))
        elif name == "mese_anno":
            plan.requests.append(_text_request(key, crop, month_year_lexicon(years), beams=6, max_tokens=8))
        elif name == "ore_pei":
            plan.requests.append(_text_request(key, crop, pei_lexicon(), beams=5, max_tokens=5))
    plan.sostituzione, plan.sostituzione_incerta = _sostituzione(gray, grid)

    # --- pie' di pagina: totale, firme, timbro, data
    tv = grid.total_value_box()
    if tv is not None and grid.total_row is not None:
        ty0, ty1 = grid.total_row
        c = grid.col_index("ore_dichiarate")
        crop = None
        if not ink.is_blank(img, tv):
            crop = clean_crop(gray, tv, rh, rows=(ty0, ty1), cols=(grid.col_x[c], grid.col_x[c + 1]))
        plan.totale_ink = crop is not None
        if crop is not None:
            plan.requests.append(_text_request("footer.totale", crop, total_lexicon(), beams=6, max_tokens=6))
        th = ty1 - ty0
        tw = grid.col_x[-1] - grid.col_x[0]
        # firma del coordinatore: a destra dell'etichetta, nella riga dei totali
        lab_box = _clip_box((grid.col_x[6], ty0 + 0.1 * th, grid.col_x[-1], ty1 - 0.1 * th), W, H)
        end = min(_label_end(gray, lab_box, rh), int(grid.col_x[0] + (COORD_LABEL_END + 0.08) * tw))
        end = max(end, int(grid.col_x[0] + (COORD_LABEL_END - 0.12) * tw))
        cbox = _clip_box((end + 0.15 * rh, ty0, grid.col_x[-1], ty1), W, H)
        _r, comps = _components(gray, cbox, rh, rows=(ty0, ty1), cols=(grid.col_x[-1],))
        plan.firma_coordinatore = _signature_like(comps, rh)
        # timbro e firma del referente: riquadro sotto la riga dei totali (o timbro che la sovrasta)
        rbox_full = _clip_box((grid.col_x[6], ty1 + 0.08 * th, grid.col_x[-1], ty1 + 1.05 * th), W, H)
        end = min(_label_end(gray, rbox_full, rh), int(grid.col_x[0] + (REFERENT_LABEL_END + 0.08) * tw))
        end = max(end, int(grid.col_x[0] + (REFERENT_LABEL_END - 0.12) * tw))
        rbox = _clip_box((end + 0.15 * rh, ty1, grid.col_x[-1] + 0.4 * rh, ty1 + 1.6 * th), W, H)
        _r, comps = _components(gray, rbox, rh, rows=(ty1, ty1 + 1.05 * th), cols=(grid.col_x[-1],))
        plan.timbro_referente = _signature_like(comps, rh) or bool(stamps)
        # data di compilazione "Napoli, __/__/____"
        dbox = _clip_box((grid.col_x[0] + DATE_LABEL_END * tw, ty1 + 0.05 * th, grid.col_x[5], ty1 + 1.1 * th), W, H)
        _r, comps = _components(gray, dbox, rh, rows=(), cols=())
        tall = [cc for cc in comps if cc[3] >= 0.25 * rh]
        if len(tall) >= 5:
            crop = clean_crop(gray, dbox, rh, extend=(0.1, 0.3))
            if crop is not None:
                plan.data_ink = True
                plan.requests.append(_text_request("footer.data", crop, date_lexicon(years), beams=5,
                                                   max_tokens=10))
    return plan


# ==========================================================================
# Post-elaborazione: evidenze, riconciliazione, incertezze
# ==========================================================================

P_INCERTO = 0.6          # probabilita' minima del valore scelto per non segnalarlo come incerto
P_LEGGIBILE = 0.05       # copertura minima: sotto, la scrittura non somiglia a nessun valore ammesso
SUPPORTO_CONTESTO = 1.0  # coerenza minima perche' un valore poco leggibile sia dedotto dalla riga
COPERTURA_DEBOLE = 0.3   # sotto questa copertura la lettura conta poco e si considerano i valori del contesto
P_CONTESTO = 0.02        # probabilita' iniziale di un valore suggerito dal contesto
TAU_MIN = 0.25           # attenuazione massima delle preferenze di una lettura debole
TOP_K = 4                # alternative per cella considerate nella riconciliazione
W_COLONNA = 1.0          # preferenza per i valori ricorrenti nella stessa colonna (orario scolastico)
QUOTA_COLONNA = 0.6      # quota oltre la quale un valore "domina" la colonna
SIMILE_NOTA = 0.8        # somiglianza per agganciare una nota a una formula ricorrente
SIMILE_NOTA_VICINA = 0.45   # ... o alla nota (sicura) del giorno precedente/successivo


def _time_minutes(v: str | None) -> int | None:
    p = parse_time(v) if v else None
    return None if p is None else p[0] * 60 + p[1]


def _time_prior(value: str) -> float:
    """Preferenza per gli orari "tondi" (8:00 molto piu' frequente di 8:30, 8:15)."""
    m = _time_minutes(value)
    if m is None:
        return 0.0
    return {0: 0.0, 30: -1.0}.get(m % 60, -2.0)


def _hours_prior(value: float) -> float:
    return 0.0 if abs(value - round(value)) < 1e-9 else -0.5


def free_value(text: str, lexicon: Lexicon | None, kind: str = "") -> Any:
    """Valore della lettura libera (``None`` se non interpretabile).

    Usa il lessico (con le sue tolleranze) e, se non basta, le funzioni di
    interpretazione di ``validation``; "3 3" (cifra ripetuta) vale "3"."""
    if not text:
        return None
    if lexicon is not None:
        v = lexicon.value(text)
        if v is not None:
            return v
        parts = text.split()
        if len(parts) > 1:
            vals = {lexicon.value(p) for p in parts}
            if len(vals) == 1 and None not in vals:
                return vals.pop()
    if kind == "time":
        return normalize_time(text)
    if kind == "hours":
        return parse_hours(text)
    return None


@dataclass
class Evidence:
    """Cosa dice la lettura di un campo con lessico."""

    options: list[tuple[Any, float]]   # (valore, log-prob relativa ai soli valori ammessi), decrescente
    coverage: float                    # 0..1: quanto la scrittura somiglia a un valore ammesso
    free_value: Any = None             # valore della lettura libera (None = non interpretabile)

    @property
    def top(self) -> Any:
        return self.options[0][0] if self.options else None

    def p(self, value: Any) -> float:
        """Probabilita' relativa di ``value`` fra i valori ammessi."""
        return next((math.exp(lp) for v, lp in self.options if v == value), 0.0)


def evidence(reading: Reading | None, lexicon: Lexicon | None, kind: str = "",
             prior: Callable[[Any], float] | None = None) -> Evidence:
    """Distribuzione dei valori ammessi e copertura della lettura.

    La copertura confronta la probabilita' dei valori ammessi con quella della
    lettura libera quando quest'ultima non corrisponde a nessuno di essi
    ("Atlas" letto in una cella degli orari -> copertura quasi nulla)."""
    if reading is None:
        return Evidence([], 0.0)
    fv = free_value(reading.free_text, lexicon, kind)
    cands = [(v, lp) for v, lp in reading.candidates if math.isfinite(lp)]
    if not cands:
        return Evidence([], 0.0, fv)
    z = _logsumexp(lp for _v, lp in cands)
    other = -math.inf
    if (reading.free_text and math.isfinite(reading.free_logprob)
            and (fv is None or all(fv != v for v, _ in cands))):
        other = reading.free_logprob
    coverage = math.exp(z - _logsumexp([z, other]))
    scored = [(v, lp + (prior(v) if prior else 0.0)) for v, lp in cands]
    zp = _logsumexp(lp for _v, lp in scored)
    options = sorted(((v, lp - zp) for v, lp in scored), key=lambda t: t[1], reverse=True)
    return Evidence(options, coverage, fv)


def _with_bonus(options: list[tuple[Any, float]], bonus: Callable[[Any], float]) -> list[tuple[Any, float]]:
    scored = [(v, lp + bonus(v)) for v, lp in options]
    z = _logsumexp(lp for _v, lp in scored)
    return sorted(((v, lp - z) for v, lp in scored), key=lambda t: t[1], reverse=True)


# pesi (log) della riconciliazione di riga
W_ORDER = -8.0           # uscita non successiva all'entrata
W_DURATION = -3.0        # durata < 30 min o > 8 h
W_SAME_PROG_EFF = 1.0    # effettivo uguale al programmato (caso piu' frequente)
W_HOURS_OK = 2.5         # ore dichiarate = uscita - entrata effettive
W_HOURS_BAD = -2.0
W_HOURS_PROG_MATCH = 0.5  # ore dichiarate = durata dell'orario programmato
W_HOURS_PROG_OK = 0.3    # senza orario effettivo: ore entro l'orario programmato
W_HOURS_PROG_BAD = -1.0


def _pair_score(e: int | None, u: int | None) -> float:
    if e is None or u is None:
        return 0.0
    if u <= e:
        return W_ORDER
    d = u - e
    if d < 30 or d > HOURS_MAX_DAY * 60:
        return W_DURATION
    return 0.0


def row_consistency(values: dict[str, Any], assenza_alunno: bool = False) -> float:
    """Punteggio (log) di coerenza di una combinazione di valori di una riga."""
    pe, pu = _time_minutes(values.get("prog_entrata")), _time_minutes(values.get("prog_uscita"))
    ee, eu = _time_minutes(values.get("eff_entrata")), _time_minutes(values.get("eff_uscita"))
    ore = values.get("ore_dichiarate")
    s = _pair_score(pe, pu) + _pair_score(ee, eu)
    if pe is not None and ee is not None and pe == ee:
        s += W_SAME_PROG_EFF
    if pu is not None and eu is not None and pu == eu:
        s += W_SAME_PROG_EFF
    if ore is not None:
        if ee is not None and eu is not None and eu > ee:
            d = (eu - ee) / 60.0
            if abs(d - ore) < 0.01:
                s += W_HOURS_OK
            elif not (assenza_alunno and ore < d):
                s += W_HOURS_BAD
        if pe is not None and pu is not None and pu > pe:
            d = (pu - pe) / 60.0
            if abs(d - ore) < 0.01:
                s += W_HOURS_PROG_MATCH
            elif ee is None or eu is None:
                s += W_HOURS_PROG_OK if ore <= d + 0.01 else W_HOURS_PROG_BAD
    return s


def reconcile_row(options: dict[str, list[tuple[Any, float]]], assenza_alunno: bool = False
                  ) -> dict[str, tuple[Any, float]]:
    """Sceglie la combinazione piu' probabile e coerente.

    ``options``: per campo, alternative (valore, log-prob) in ordine decrescente.
    Restituisce per campo (valore scelto, log-prob del valore)."""
    fields = [f for f, opts in options.items() if opts]
    if not fields:
        return {}
    choices = [options[f][:TOP_K] for f in fields]
    best: tuple[float, tuple[tuple[Any, float], ...]] | None = None
    for combo in itertools.product(*choices):
        values = {f: v for f, (v, _lp) in zip(fields, combo)}
        score = sum(lp for _v, lp in combo) + row_consistency(values, assenza_alunno)
        if best is None or score > best[0]:
            best = (score, combo)
    assert best is not None
    return {f: c for f, c in zip(fields, best[1])}


def _clean_name(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).upper()
    t = re.sub(r"[^A-ZÀÈÉÌÒÙ0-9' .\-/]", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" .-/',")
    return t


@dataclass
class _Field:
    value: Any = None
    confidence: float | None = None
    incerto: bool = False
    illeggibile: bool = False


def _choose(ev: Evidence) -> _Field:
    """Valore piu' probabile di un campo senza contesto."""
    if not ev.options or ev.coverage < P_LEGGIBILE:
        return _Field(illeggibile=True, confidence=0.0)
    v = ev.top
    p = ev.p(v) * ev.coverage
    incerto = p < P_INCERTO or (ev.free_value is not None and ev.free_value != v)
    return _Field(value=v, confidence=p, incerto=incerto)


def _similarity(a: str, b: str) -> float:
    import difflib  # noqa: PLC0415

    ka, kb = _key(a), _key(b)
    if not ka or not kb:
        return 0.0
    return difflib.SequenceMatcher(None, ka, kb).ratio()


def _text_field(reading: Reading | None, min_conf: float = 0.55, floor: float = 0.12) -> _Field:
    """Campo di testo libero (nomi): lettura libera ripulita."""
    if reading is None:
        return _Field(illeggibile=True)
    text = _clean_name(reading.free_text)
    conf = reading.free_confidence
    if not text or not re.search(r"[A-Z0-9ÀÈÉÌÒÙ]", text) or conf < floor:
        return _Field(illeggibile=True, confidence=conf)
    return _Field(value=text, confidence=conf, incerto=conf < min_conf)


def _note_field(reading: Reading | None) -> _Field:
    """Nota: formula ricorrente (lessico o somiglianza) oppure testo letto."""
    if reading is None:
        return _Field(illeggibile=True)
    text = _clean_name(reading.free_text)
    conf = reading.free_confidence
    n_tok = max(1, reading.free_tokens)
    if reading.candidates:
        lex_v, lex_lp = reading.candidates[0]
        if not text or lex_lp >= reading.free_logprob - 1.0:
            lconf = float(math.exp(min(0.0, lex_lp) / (n_tok + 1)))
            return _Field(value=str(lex_v), confidence=lconf, incerto=lconf < 0.55)
    if text:
        best, sim = None, 0.0
        for canon, forms in NOTE_FREQUENTI:
            for form in forms:
                r = _similarity(text, form)
                if r > sim:
                    best, sim = canon, r
        if best is not None and sim >= SIMILE_NOTA:
            return _Field(value=best, confidence=min(conf, sim), incerto=sim < 0.9 or conf < 0.55)
    return _text_field(reading)


_ETICHETTE = {
    "prog_entrata": "entrata programmata", "prog_uscita": "uscita programmata",
    "eff_entrata": "entrata effettiva", "eff_uscita": "uscita effettiva", "ore_dichiarate": "ore",
}


def etichetta(campo: str) -> str:
    return _ETICHETTE.get(campo, campo)


def _shift(t: Any, hours: Any, sign: int) -> str | None:
    m = _time_minutes(t) if isinstance(t, str) else None
    if m is None or not isinstance(hours, (int, float)):
        return None
    r = m + sign * int(round(float(hours) * 60))
    if not TIME_MIN <= r <= TIME_MAX or r % TIME_STEP:
        return None
    return f"{r // 60:02d}:{r % 60:02d}"


def _context_values(f: str, cells: dict[str, Evidence], column_values: list[Any]) -> list[Any]:
    """Valori suggeriti dal resto della riga e dalla colonna per una cella poco leggibile."""
    top = {k: ev.top for k, ev in cells.items() if k != f and ev.options}
    out: list[Any] = []
    twin = {"prog_entrata": "eff_entrata", "eff_entrata": "prog_entrata",
            "prog_uscita": "eff_uscita", "eff_uscita": "prog_uscita"}
    if f in twin and top.get(twin[f]) is not None:
        out.append(top[twin[f]])
    ore = top.get("ore_dichiarate")
    if f == "eff_uscita":
        out += [_shift(top.get("eff_entrata"), ore, 1), _shift(top.get("prog_entrata"), ore, 1)]
    elif f == "eff_entrata":
        out += [_shift(top.get("eff_uscita"), ore, -1), _shift(top.get("prog_uscita"), ore, -1)]
    elif f == "prog_uscita":
        out += [_shift(top.get("prog_entrata"), ore, 1), _shift(top.get("eff_entrata"), ore, 1)]
    elif f == "prog_entrata":
        out += [_shift(top.get("prog_uscita"), ore, -1), _shift(top.get("eff_uscita"), ore, -1)]
    elif f == "ore_dichiarate":
        for a, b in (("eff_entrata", "eff_uscita"), ("prog_entrata", "prog_uscita")):
            d = hours_between(top.get(a), top.get(b))
            if d is not None and abs(d * 2 - round(d * 2)) < 1e-9 and 0 < d <= HOURS_MAX_DAY:
                out.append(d)
    if column_values:
        out.append(max(set(column_values), key=column_values.count))
    return [v for v in dict.fromkeys(out) if v is not None]


def _weak_options(options: list[tuple[Any, float]], extra: list[Any], coverage: float
                  ) -> list[tuple[Any, float]]:
    """Lettura debole: differenze fra le alternative attenuate in proporzione alla
    copertura e valori suggeriti dal contesto aggiunti fra le alternative."""
    tau = min(1.0, max(TAU_MIN, coverage / COPERTURA_DEBOLE))
    scored = [(v, lp * tau) for v, lp in options]
    present = {v for v, _ in scored}
    floor = math.log(P_CONTESTO) * tau
    scored += [(v, floor) for v in extra if v not in present]
    if not scored:
        return []
    z = _logsumexp(lp for _v, lp in scored)
    return sorted(((v, lp - z) for v, lp in scored), key=lambda t: t[1], reverse=True)


def _column_bonus(values: list[Any], own: Any) -> Callable[[Any], float]:
    """Preferenza per i valori gia' letti con sicurezza negli altri giorni della colonna."""
    others = list(values)
    if own is not None and own in others:
        others.remove(own)
    n = len(others)
    if n < 3:
        return lambda _v: 0.0
    counts: dict[Any, int] = {}
    for v in others:
        counts[v] = counts.get(v, 0) + 1
    return lambda v: W_COLONNA * counts.get(v, 0) / n


def _context_confidence(support: float) -> float:
    """Fiducia data dalla coerenza della riga (es. ore = uscita - entrata)."""
    return min(0.95, max(0.0, 1.0 - math.exp(-max(0.0, support) / 1.5)))


def _dominant(values: list[Any], own: Any) -> Any:
    others = list(values)
    if own is not None and own in others:
        others.remove(own)
    if len(others) < 3:
        return None
    best = max(set(others), key=others.count)
    return best if others.count(best) >= QUOTA_COLONNA * len(others) else None


def assemble(plan: PagePlan, readings: dict[str, Reading], grid: TableGrid) -> ExtractionResult:
    """Costruisce il risultato finale dai riconoscimenti e dall'analisi d'inchiostro."""
    t_lex, h_lex = time_lexicon(), hours_lexicon()
    # 1) evidenze di tutte le celle con scrittura
    evs: dict[int, dict[str, Evidence]] = {}
    for g in range(1, 32):
        st = plan.cells.get(g, {})
        evs[g] = {}
        for f in TEXT_FIELDS:
            if st.get(f) != TESTO:
                continue
            reading = readings.get(f"rows.{g}.{f}")
            if f == "ore_dichiarate":
                evs[g][f] = evidence(reading, h_lex, "hours", _hours_prior)
            else:
                evs[g][f] = evidence(reading, t_lex, "time", _time_prior)
    # valori letti con sicurezza in ogni colonna (per la preferenza di colonna)
    column: dict[str, list[Any]] = {f: [] for f in TEXT_FIELDS}
    for g, cells in evs.items():
        for f, ev in cells.items():
            if ev.options and ev.coverage >= 0.3 and math.exp(ev.options[0][1]) >= 0.5:
                column[f].append(ev.top)

    rows: list[DayRow] = []
    confs: list[float] = []
    dedotti: list[str] = []
    for g in range(1, 32):
        st = plan.cells.get(g, {})
        row = DayRow(giorno=g)
        row.assenza_alunno = st.get("assenza_alunno") in (CROCETTA, SEGNO)
        row.assenza_operatore = st.get("assenza_operatore") in (CROCETTA, SEGNO)
        row.firma = st.get("firma") in (FIRMA, SEGNO)
        for f in ("assenza_alunno", "assenza_operatore", "firma"):
            if st.get(f) == SEGNO:
                row.incerti.append(f)
                row.confidenza[f] = 0.5
        row.trattino_effettivo = TRATTINO in (st.get("eff_entrata"), st.get("eff_uscita"))
        cells = evs[g]
        options = {f: _with_bonus(ev.options, _column_bonus(column[f], ev.top))
                   for f, ev in cells.items() if ev.options}
        for f, ev in cells.items():
            if ev.coverage < COPERTURA_DEBOLE:
                options[f] = _weak_options(options.get(f, []), _context_values(f, cells, column[f]),
                                           ev.coverage)
        chosen = reconcile_row(options, assenza_alunno=row.assenza_alunno)
        values = {f: v for f, (v, _lp) in chosen.items()}
        for f in TEXT_FIELDS:
            if f not in cells:
                continue
            ev = cells[f]
            if f not in chosen:
                row.illeggibili.append(f)
                confs.append(0.0)
                continue
            v = values[f]
            p = ev.p(v) * ev.coverage
            rest = {k: x for k, x in values.items() if k != f}
            support = (row_consistency(values, row.assenza_alunno)
                       - row_consistency(rest, row.assenza_alunno))
            deduced = ev.coverage < P_LEGGIBILE
            if deduced:
                if support < SUPPORTO_CONTESTO:
                    # scrittura presente ma non riconducibile a un valore ammesso
                    row.illeggibili.append(f)
                    values.pop(f)
                    confs.append(0.0)
                    continue
                dedotti.append(f"giorno {g} ({etichetta(f)})")
                p = min(p, 0.2)
            else:
                # la coerenza con il resto della riga rafforza una lettura gia' preferita
                p = 1.0 - (1.0 - p) * (1.0 - _context_confidence(support))
            setattr(row, f, v)
            row.confidenza[f] = round(p, 3)
            confs.append(p)
            dom = _dominant(column[f], ev.top)
            unusual = dom is not None and dom != v and ev.p(dom) >= 0.02
            if (deduced or v != ev.top or p < P_INCERTO or unusual
                    or (ev.free_value is not None and ev.free_value != v)):
                row.incerti.append(f)
        rows.append(row)

    # note: formule ricorrenti, testo letto, nota uguale a quella del giorno vicino
    notes: dict[int, _Field] = {}
    for g in range(1, 32):
        if plan.cells.get(g, {}).get("note") == TESTO:
            notes[g] = _note_field(readings.get(f"rows.{g}.note"))
    for g, fld in list(notes.items()):
        if fld.illeggibile or fld.incerto:
            reading = readings.get(f"rows.{g}.note")
            text = _clean_name(reading.free_text) if reading else ""
            for ng in (g - 1, g + 1):
                other = notes.get(ng)
                reliable = other is not None and bool(other.value) and not other.illeggibile and (
                    not other.incerto or other.value in _NOTE_CANONICHE)
                if reliable and text and _similarity(text, str(other.value)) >= SIMILE_NOTA_VICINA:
                    notes[g] = _Field(value=other.value, confidence=fld.confidence, incerto=True)
                    break
    for g, fld in notes.items():
        row = rows[g - 1]
        if fld.illeggibile:
            row.illeggibili.append("note")
        else:
            row.note = fld.value
            if fld.incerto:
                row.incerti.append("note")
        if fld.confidence is not None:
            row.confidenza["note"] = round(fld.confidence, 3)

    header = _assemble_header(plan, readings, rows, confs)

    n_incerti = sum(len(r.incerti) for r in rows) + len(header.incerti)
    n_illeggibili = sum(len(r.illeggibili) for r in rows) + len(header.illeggibili)
    detected = bool(grid.detected) or grid.score >= 0.5
    text_conf = float(np.mean(confs)) if confs else 1.0
    confidence = round(max(0.0, min(1.0, 0.85 * text_conf + 0.15 * float(grid.score))), 3)
    remarks = list(plan.notes)
    if not grid.detected:
        remarks.append("Tabella individuata solo con il modello proporzionale: verificare l'allineamento "
                       "delle righe.")
    if n_illeggibili:
        remarks.append(f"Campi illeggibili: {n_illeggibili}.")
    if dedotti:
        remarks.append("Valori poco leggibili dedotti dalla coerenza della riga (da verificare): "
                       + ", ".join(dedotti) + ".")
    if n_incerti:
        remarks.append(f"Campi da verificare (lettura incerta): {n_incerti}.")
    return ExtractionResult(
        is_foglio_firma=detected,
        header=header,
        rows=rows,
        ocr_notes=" ".join(remarks) or None,
        confidence=confidence,
        engine="locale",
    )


def _assemble_header(plan: PagePlan, readings: dict[str, Reading], rows: list[DayRow],
                     confs: list[float]) -> Header:
    header = Header()
    for name in ("operatore", "alunno", "ente", "istituto"):
        if not plan.header_ink.get(name):
            continue
        fld = _text_field(readings.get(f"header.{name}"))
        _apply_header(header, name, fld)
        if fld.confidence is not None:
            confs.append(fld.confidence)
    for name, lex in (("lotto", small_int_lexicon(30)), ("municipalita", small_int_lexicon(10, roman=True)),
                      ("ore_pei", pei_lexicon())):
        if not plan.header_ink.get(name):
            continue
        ev = evidence(readings.get(f"header.{name}"), lex, "hours" if name == "ore_pei" else "")
        fld = _choose(ev)
        if name == "ore_pei" and fld.value is not None:
            fld.value = float(fld.value)
        _apply_header(header, name, fld)
        if fld.confidence is not None:
            confs.append(fld.confidence)
    if plan.header_ink.get("mese_anno"):
        ev = evidence(readings.get("header.mese_anno"), None, "month")
        worked = [g for g, st in plan.cells.items()
                  if any(st.get(f) == TESTO for f in TEXT_FIELDS) or st.get("firma") == FIRMA]
        fld = _choose_month(ev, worked)
        if fld.illeggibile:
            header.illeggibili += ["mese", "anno"]
        elif fld.value is not None:
            header.mese, header.anno = fld.value
            if fld.incerto:
                header.incerti += ["mese", "anno"]
        if fld.confidence is not None:
            confs.append(fld.confidence)
    if header.anno and header.mese:
        start = header.anno if header.mese >= 9 else header.anno - 1
        header.anno_scolastico = f"{start}/{start + 1}"
    header.sostituzione = plan.sostituzione  # type: ignore[assignment]
    if plan.sostituzione_incerta:
        header.incerti.append("sostituzione")
    header.firma_coordinatore = plan.firma_coordinatore
    header.timbro_referente = plan.timbro_referente
    if plan.totale_ink:
        ev = evidence(readings.get("footer.totale"), total_lexicon(), "hours", _hours_prior)
        somma = sum(r.ore_dichiarate or 0.0 for r in rows)
        fld = _choose_total(ev, somma)
        _apply_header(header, "totale_mensile_dichiarato", fld)
        if fld.confidence is not None:
            confs.append(fld.confidence)
    if plan.data_ink:
        ev = evidence(readings.get("footer.data"), None, "date")
        _apply_header(header, "data_compilazione", _choose(ev))
    return header


W_CAL_SABATO = -0.5      # giorno lavorato di sabato (possibile)
W_CAL_FESTIVO = -1.5     # giorno lavorato di domenica, festivo o inesistente nel mese


def _month_prior(value: tuple[int, int], today: date | None = None) -> float:
    """Preferenza per i mesi recenti: i fogli si rendicontano dopo il mese di riferimento."""
    today = today or date.today()
    m, y = value
    ago = (today.year * 12 + today.month) - (y * 12 + m)
    if ago < 0:
        return -3.0          # mese futuro: i giorni non possono essere gia' stati lavorati
    return -0.04 * max(0, ago - 1)


def calendar_penalty(value: tuple[int, int], worked_days: Sequence[int]) -> float:
    """Incoerenza fra mese/anno e giorni compilati: chi lavora di domenica o in
    giorni inesistenti suggerisce una lettura sbagliata del mese o dell'anno."""
    from sirio.calendario import tipo_giorno  # noqa: PLC0415

    m, y = value
    pen = 0.0
    for g in worked_days:
        try:
            tipo = tipo_giorno(y, m, g)
        except (ValueError, TypeError):
            tipo = "inesistente"
        if tipo == "sabato":
            pen += W_CAL_SABATO
        elif tipo != "feriale":
            pen += W_CAL_FESTIVO
    return pen


def _choose_month(ev: Evidence, worked_days: Sequence[int], today: date | None = None) -> _Field:
    """Mese/anno: lettura di mese e anno (separatamente) combinata con la coerenza
    del calendario dei giorni compilati e con la preferenza per i mesi recenti.

    Esempio: "02/2026" letto male come "02/2025" viene corretto se nel 2025 i
    giorni compilati cadrebbero di domenica."""
    if not ev.options:
        return _Field(illeggibile=True, confidence=0.0)
    today = today or date.today()
    month_lp: dict[int, list[float]] = {}
    year_lp: dict[int, list[float]] = {}
    for (m, y), lp in ev.options:
        month_lp.setdefault(m, []).append(lp)
        year_lp.setdefault(y, []).append(lp)
    ml = {m: _logsumexp(v) for m, v in month_lp.items()}
    yl = {y: _logsumexp(v) for y, v in year_lp.items()}
    tau = min(1.0, max(0.05, ev.coverage / COPERTURA_DEBOLE))
    floor_m = min(ml.values()) - 3.0
    floor_y = min(yl.values()) - 3.0
    years = sorted(set(_years_around(today.year)) | set(yl))
    scored = []
    for y in years:
        for m in range(1, 13):
            lp = tau * (ml.get(m, floor_m) + yl.get(y, floor_y))
            scored.append(((m, y), lp + _month_prior((m, y), today) + calendar_penalty((m, y), worked_days)))
    z = _logsumexp(s_ for _v, s_ in scored)
    scored.sort(key=lambda t: t[1], reverse=True)
    v, best = scored[0]
    p_post = math.exp(best - z)
    if ev.coverage < P_LEGGIBILE and (calendar_penalty(v, worked_days) < 0 or len(worked_days) < 5):
        return _Field(illeggibile=True, confidence=0.0)
    p = min(ev.p(v) * ev.coverage, p_post) if v == ev.top else min(p_post, 0.5) * max(ev.coverage, 0.2)
    incerto = v != ev.top or p < P_INCERTO
    return _Field(value=v, confidence=p, incerto=incerto)


def _choose_total(ev: Evidence, somma: float) -> _Field:
    """Totale mensile: se fra le alternative plausibili c'e' la somma delle ore
    giornaliere la preferisce (segnalandola se non era la lettura migliore)."""
    if not ev.options:
        return _Field(illeggibile=True, confidence=0.0)
    for v, lp in ev.options[:3]:
        if abs(float(v) - somma) < 0.01 and math.exp(lp) >= 0.1:
            p = math.exp(lp) * ev.coverage
            incerto = v != ev.top or p < P_INCERTO
            return _Field(value=float(v), confidence=p, incerto=incerto)
    fld = _choose(ev)
    if fld.value is not None:
        fld.value = float(fld.value)
    return fld


def _apply_header(header: Header, name: str, fld: _Field) -> None:
    if fld.illeggibile:
        header.illeggibili.append(name)
        return
    setattr(header, name, fld.value)
    if fld.incerto:
        header.incerti.append(name)


# ==========================================================================
# Motore
# ==========================================================================

class LocalEngine:
    """Motore OCR locale: TrOCR + analisi dell'inchiostro, nessun dato inviato in rete."""

    name = "locale"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu", cache_dir: Path | None = None,
                 recognizer: Recognizer | None = None):
        self.model_name = (model_name or DEFAULT_MODEL).strip()
        self.device = device or "cpu"
        self._cache_dir = cache_dir
        self._recognizer = recognizer

    @property
    def cache_dir(self) -> Path:
        if self._cache_dir is None:
            self._cache_dir = default_cache_dir()
        return self._cache_dir

    # ------------------------------------------------------------ disponibilita'
    def is_available(self) -> tuple[bool, str]:
        if self._recognizer is not None:
            return True, "Motore locale pronto."
        missing = _missing_components()
        if missing:
            return False, missing
        label = _model_label(self.model_name)
        if local_model_path(self.model_name, self.cache_dir) is not None:
            return True, f"Motore locale pronto: modello {label} già scaricato (funziona senza Internet)."
        mb = KNOWN_MODELS.get(self.model_name, {}).get("mb")
        size = f" (circa {_fmt_size(mb)}, una sola volta)" if mb else " (una sola volta)"
        return True, (f"Motore locale disponibile: al primo utilizzo verrà scaricato il modello {label}"
                      f"{size}; poi funziona senza Internet.")

    # ---------------------------------------------------------------- modello
    def _ensure_recognizer(self, progress: ProgressFn) -> Recognizer:
        if self._recognizer is not None:
            return self._recognizer
        missing = _missing_components()
        if missing:
            raise EngineError(missing)
        key = (self.model_name, self.device, str(self.cache_dir))
        with _MODELS_LOCK:
            rec = _MODELS.get(key)
        if rec is not None:
            return rec
        with _MODELS_LOCK:
            dl_lock = _DOWNLOAD_LOCKS.setdefault(self.model_name, threading.Lock())
        with dl_lock:
            with _MODELS_LOCK:
                rec = _MODELS.get(key)
            if rec is not None:
                return rec
            path = local_model_path(self.model_name, self.cache_dir)
            if path is None:
                path = download_model(self.model_name, self.cache_dir,
                                      lambda f, m: progress(0.25 * f, m))
            progress(0.26, f"Caricamento del modello {_model_label(self.model_name)}…")
            rec = _load_recognizer(path, self.model_name, self.device)
            with _MODELS_LOCK:
                # un solo modello in memoria alla volta
                _MODELS.clear()
                _MODELS[key] = rec
        return rec

    # --------------------------------------------------------------- lettura
    def extract(self, page: PageInput, progress: ProgressFn) -> ExtractionResult:
        progress = progress or no_progress
        t0 = time.monotonic()
        img = page.image
        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            raise EngineError("Immagine della pagina non valida.")
        grid = page.grid
        h, w = img.shape[:2]
        if (grid.width, grid.height) != (w, h):
            grid = grid.scaled(w / float(grid.width), h / float(grid.height))
            grid.width, grid.height = w, h
        if not grid.detected and grid.score < 0.5:
            return ExtractionResult(
                is_foglio_firma=False,
                ocr_notes="La pagina non sembra un foglio firma: tabella giornaliera non riconosciuta.",
                confidence=round(float(grid.score), 3),
                engine="locale",
                model=self.model_name,
                usage=Usage(seconds=round(time.monotonic() - t0, 2)),
            )
        recognizer = self._ensure_recognizer(progress)
        progress(0.3, "Analisi della tabella e delle firme…")
        try:
            plan = plan_page(img, grid)
        except Exception as exc:  # noqa: BLE001
            log.exception("Analisi della pagina non riuscita")
            raise EngineError(f"Analisi della pagina non riuscita: {exc}") from exc
        total = len(plan.requests)

        def on_read(done: int, n: int) -> None:
            progress(0.33 + 0.62 * done / max(1, n), f"Lettura della scrittura a mano: {done} di {n} campi…")

        progress(0.33, f"Lettura della scrittura a mano: 0 di {total} campi…")
        try:
            results = recognizer.read(plan.requests, on_read) if total else []
        except EngineError:
            raise
        except MemoryError as exc:
            raise EngineError("Memoria insufficiente per il riconoscimento offline: chiudere altri programmi "
                              "e riprovare.") from exc
        except Exception as exc:  # noqa: BLE001
            log.exception("Errore del riconoscimento offline")
            raise EngineError(f"Errore del riconoscimento offline: {exc}") from exc
        readings = {req.key: res for req, res in zip(plan.requests, results)}
        progress(0.97, "Controllo di coerenza dei valori letti…")
        result = assemble(plan, readings, grid)
        result.model = self.model_name
        result.usage = Usage(seconds=round(time.monotonic() - t0, 2))
        progress(1.0, "Lettura completata.")
        return result


__all__ = [
    "DEFAULT_MODEL",
    "KNOWN_MODELS",
    "Lexicon",
    "LocalEngine",
    "PagePlan",
    "ReadRequest",
    "Reading",
    "Recognizer",
    "TrOCRRecognizer",
    "assemble",
    "clean_crop",
    "download_model",
    "evidence",
    "free_value",
    "hours_lexicon",
    "local_model_path",
    "plan_page",
    "reconcile_row",
    "row_consistency",
    "time_lexicon",
]
