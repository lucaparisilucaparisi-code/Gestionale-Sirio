"""Motore OCR locale (offline): OpenCV + TrOCR, solo CPU.

Funziona senza Internet (dopo il primo download del modello) e senza chiave API.

Principio guida: su fogli di operatori diversi (scansioni, foto, grafie e
abitudini di compilazione diverse) un valore sbagliato mostrato come sicuro e'
l'errore peggiore. Ogni regola che *sceglie* un valore diverso dalla lettura
migliore lo segnala; i controlli di coerenza e plausibilita' *segnalano*, non
riscrivono le letture chiare.

Pipeline di una pagina
----------------------
1. **Barrature e segni** (``sirio.engines.strokes``): le lunghe linee a mano
   (righe non usate annullate in diagonale, linea orizzontale sugli orari
   effettivi = "non svolta") vengono trovate e cancellate prima di ogni analisi;
   una "/" isolata in una riga senza orari non e' un valore.
   Anche le pieghe del foglio (``ink.find_creases``: lunghe linee quasi verticali
   continue per molte righe, non stampate) sono tolte in ogni colonna: la
   scrittura che attraversano (o che vi resta sotto) e' segnalata.
2. **Inchiostro** (``sirio.vision.ink``): ogni cella dei 31 giorni viene
   classificata come vuota, trattino, crocetta (assenze: X, x, A, spunte),
   firma o scrittura. Timbri sulle colonne Firma/Note, timbri e firme del pie'
   di pagina che sconfinano nelle ultime righe (``foreign_ink``), inizio di una
   firma che sconfina nella colonna delle assenze e celle attraversate da
   barrature sono segnalati; le code delle firme che sconfinano nella colonna
   "Note" vengono tolte prima di decidere se c'e' una nota.
3. **Ritagli puliti**: le celle con scrittura vengono ritagliate togliendo le
   linee della griglia e i tratti delle celle vicine; per orari e ore si
   aggiunge bianco sopra e sotto (TrOCR riduce ogni ritaglio a un quadrato); le
   note scritte su due righe nella cella sono lette una riga alla volta. I campi
   dell'intestazione vengono cercati sulle righe di risposta a destra delle
   etichette stampate.
4. **TrOCR** (``VisionEncoderDecoderModel``, pesi in float32): decodifica
   *vincolata* al lessico delle scritture reali (orari "8:00", "8.00", "10,30",
   "800", "1100", "8h30", "8:30h"; ore "3", "2,5", "2h", "2h30'", "1h15'", "0";
   mese/anno, numeri, frasi frequenti delle note) affiancata da una lettura
   libera; ritagli letti a lotti con il decoder che fa avanzare insieme tutte le
   ricerche del lotto.
5. **Scelta dei valori** per riga: probabilita' delle letture con preferenze
   *deboli* (minuti tondi adattati alla grafia della pagina, orario scolastico,
   valori ricorrenti in colonna, programmato = effettivo, uscita - entrata = ore)
   e alternative per le cifre che TrOCR confonde (1/4/7, 2/9, 0/6). Le
   preferenze decidono solo fra letture quasi equivalenti; un valore diverso
   dalla lettura migliore e' sempre segnalato.
6. **Segnalazioni** (``incerti``): lettura poco probabile o poco leggibile,
   lettura libera con cifre o forma diverse ("10/3" contro 10:00, sola ora
   "13" in un ritaglio largo), orario fuori dalla fascia scolastica, riga
   incoerente (ore diverse da uscita - entrata, uscita prima dell'entrata, un
   solo orario effettivo, ore oltre l'orario programmato), celle barrate, riga o
   pagina non allineata alla griglia (``TableGrid.reliable`` /
   ``row_reliable``). Se per rendere coerente una riga si scarta una lettura
   chiara, si segnala l'intera riga. Una cella incerta *solo* perche' poco
   leggibile torna sicura se una relazione esatta con celle sicure della stessa
   riga (o lo stesso orario programmato negli stessi giorni della settimana)
   conferma proprio il valore preferito dalla lettura. Scrittura presente ma non
   interpretabile -> ``illeggibili`` (valore ``None``).
   Firme e crocette in una riga senza orari, ore e note sono sempre da
   verificare (segno isolato). L'orario programmato e' confrontato con gli
   stessi giorni della settimana (valori e durata; basta una settimana letta con
   sicurezza) e, con l'alunno assente, le ore riconosciute devono essere una
   frazione tonda della durata programmata (di solito la meta').
7. **Note**: riportate come sicure solo se lette con buona fiducia e fatte di
   parole note (vocabolario generico della scuola, abbreviazioni puntate
   comprese) senza numeri; le letture brevi fuori vocabolario (code di firme,
   timbri, tratti) non diventano note. Una nota scritta per davvero (due righe,
   o almeno tre segni larghi e alti come una parola breve, es. "104") che non
   viene letta e' illeggibile, mai vuota in silenzio.

Precisione misurata (CPU a 4 core)
----------------------------------
Pagine reali di sviluppo di quattro operatori diversi (scansioni e una foto;
orari, ore, crocette/firme, note, intestazione): 89% dei campi giusti, nessun
errore non segnalato (prima: 83%, 27 errori mostrati come sicuri). Fogli
sintetici con orari con la virgola o senza separatore, righe barrate, firme
lunghe, foto: 98,8% (prima 96,4%, 10 errori non segnalati). I nomi in stampatello
sono letti solo in parte e vanno sempre verificati. Circa 60-90 secondi per
pagina: l'encoder in float32 da solo richiede circa 35 s per ~100 campi.
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
import shutil
import stat
import sys
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
from sirio.engines.strokes import erase, find_strikes, is_slash, stroke_shape
from sirio.models import DAY_FIELDS, HEADER_FIELDS, DayRow, ExtractionResult, Header, Usage
from sirio.validation import avviso_griglia_incerta, hours_between, normalize_time, parse_hours, parse_time
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
_DLL_MSG = (
    "PyTorch è installato ma Windows non riesce a caricarlo: installare «Microsoft Visual C++ "
    "Redistributable» (https://aka.ms/vs/17/release/vc_redist.x64.exe) e riavviare Sirio OCR."
)
OOM_MSG = (
    "Memoria insufficiente per il riconoscimento offline (servono circa 3 GB liberi): chiudere altri "
    "programmi e riprovare."
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
# minuti ammessi: multipli di 5. I quarti d'ora sono i piu' comuni, ma c'e' chi
# lavora "8:10 - 13:10" (orari scolastici sfalsati): con i soli quarti un 8:10
# scritto chiaramente diventava un 8:30 *sicuro*. I minuti non a quarti partono
# sfavoriti (``page_minute_prior``) e contano solo se la pagina li usa davvero.
TIME_STEP = 5
HOURS_MAX_DAY = 8.0

MESI_NOMI = (
    "GENNAIO", "FEBBRAIO", "MARZO", "APRILE", "MAGGIO", "GIUGNO",
    "LUGLIO", "AGOSTO", "SETTEMBRE", "OTTOBRE", "NOVEMBRE", "DICEMBRE",
)
_ROMANI = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")

# Note ricorrenti sui fogli firma: (valore, varianti scritte). Il valore e' la
# frase stessa (la nota si riporta come scritta); solo le varianti del permesso
# L. 104 si riportano a "104". Vocabolario generico della scuola, usato in modo
# "morbido": aiuta a leggere ma non trasforma una nota in un'altra.
NOTE_FREQUENTI: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("104", ("104", "L104", "L.104", "LEGGE 104", "PERMESSO 104", "PERM. 104")),
    *((p, (p,)) for p in (
        "PONTE DI CARNEVALE", "PONTE CARNEVALE", "CARNEVALE", "PONTE", "FESTIVO", "FESTA", "SCIOPERO",
        "ASSEMBLEA SINDACALE", "ASSEMBLEA DI ISTITUTO", "ASSEMBLEA", "MALATTIA", "FERIE", "PERMESSO",
        "PERMESSO RETRIBUITO", "CHIUSURA SCUOLA", "SCUOLA CHIUSA", "USCITA DIDATTICA", "USCITA ANTICIPATA",
        "ENTRATA POSTICIPATA", "GITA SCOLASTICA", "GITA", "ELEZIONI", "VACANZE", "VACANZE NATALIZIE",
        "VACANZE DI NATALE", "VACANZE PASQUALI", "VACANZE DI PASQUA", "ALUNNO ASSENTE", "ALUNNA ASSENTE",
        "ASSENTE", "RECUPERO", "SOSTITUZIONE", "FORMAZIONE", "CORSO DI FORMAZIONE", "GLO", "GLHO",
        "INCONTRO GLO", "RIUNIONE GLO", "RIUNIONE", "CERTIFICATO MEDICO", "VISITA MEDICA", "COLLEGIO DOCENTI",
        "CONSIGLIO DI CLASSE", "SCRUTINI", "DISINFESTAZIONE", "ALLERTA METEO", "LUTTO",
        "SOSPENSIONE ATTIVITA DIDATTICA",
    )),
)

_NOTE_FRASI = frozenset(v for v, _forms in NOTE_FREQUENTI)

# Parole delle note (generiche, per riconoscere una nota "in italiano" e
# correggere lievi errori di lettura); le abbreviazioni ("CARN.", "ASSEM.")
# valgono come la parola intera.
NOTE_PAROLE: frozenset[str] = frozenset(
    w for _v, forms in NOTE_FREQUENTI for f in forms for w in re.findall(r"[A-Z]+", f) if len(w) >= 2
) | frozenset((
    "ALUNNO", "ALUNNA", "NON", "ERA", "PRESENTE", "ASSENZA", "ORE", "ORA", "ORARIO", "ENTRATA", "USCITA",
    "ANTICIPO", "ANTICIPATA", "RITARDO", "POSTICIPATA", "ISTITUTO", "SCUOLA", "CLASSE", "DOCENTI", "MEDICO",
    "CERTIFICATO", "CERTIFICAZIONE", "PROTOCOLLO", "RICHIESTA", "SERVIZIO", "SOSPENSIONE",
    "ATTIVITA", "DIDATTICA", "DIDATTICHE", "MANIFESTAZIONE", "INCONTRO", "GENITORI", "COLLOQUIO", "SUL",
    "SULLA", "DEL", "DELLA", "DELL", "DI", "DA", "AL", "ALLA", "IL", "LA", "LO", "PER", "CON", "SU", "NEL",
    "MATTINA", "POMERIGGIO", "GIORNO", "GIORNATA", "FESTIVITA", "SANTO", "PATRONO", "COMUNE", "CHIUSO",
    "CHIUSA", "RETRIBUITO", "RETRIBUITA", "MALATO", "MALATA", "LEGGE", "EVENTO", "TEATRO",
))


# Parole che da sole non fanno una nota (articoli, preposizioni)
PAROLE_VUOTE = frozenset((
    "IL", "LA", "LO", "LE", "GLI", "I", "DI", "DA", "AL", "ALLA", "SUL", "SULLA", "DEL", "DELLA", "DELL",
    "PER", "CON", "SU", "NEL", "E", "A", "IN", "UN", "UNA",
))


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


_TIME_SEPARATORS = (":", ".", ",", ";", "'", "H", "")


@functools.cache
def time_lexicon() -> Lexicon:
    """Orari 06:00-20:00 a passi di 5 minuti nelle scritture reali: "8:00",
    "08.30", "8,30", "10,30", "800", "1100", "830", "8h30", "8:30h", "8"..."""
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
                            pairs.append((f"{hh}{sep}{mm}H", value))    # "8:30h"
                if m == 0:
                    pairs.append((hh, value))
                    pairs.append((f"{hh}H", value))
    return Lexicon.build("orari", pairs, numeric=True)


HOURS_STEP = 0.25          # ore a quarti d'ora


def _hours_forms(v: float) -> list[str]:
    """Scritture di un numero di ore: "3", "3h", "2,5", "2h30'", "2:30", "1h15'"...

    I quarti d'ora si scrivono di norma con le ore e i minuti ("1h15'", "1:45"):
    le forme decimali "1,25"/"2,75" non sono ammesse perche' TrOCR legge spesso
    "2,5" come "2.25" e la lettura sbagliata diventerebbe un valore ammesso."""
    whole = int(math.floor(v + 1e-9))
    minutes = int(round((v - whole) * 60))
    w = str(whole)
    if minutes == 0:
        forms = [w, f"{w}H", f"{w},0", f"{w}.0", f"{w}:00", f"{w}H00", f"{w}H00'"]
    elif minutes == 30:
        forms = [*_num_forms(v), f"{w},50", f"{w}.50", f"{w},30", f"{w}.30", f"{w}½", f"{w}1/2", f"{w}1⁄2"]
        if whole == 0:
            forms += ["½", "1/2"]
    else:
        forms = []
    if minutes:
        for sep in ("H", ":"):
            forms += [f"{w}{sep}{minutes:02d}", f"{w}{sep}{minutes:02d}'"]
        forms += [f"{w}H{minutes:02d}MIN", f"{w}H{minutes:02d}M"]
        if whole == 0:
            forms += [f"{minutes}'", f"{minutes}MIN"]
    return list(dict.fromkeys(forms))


@functools.cache
def hours_lexicon() -> Lexicon:
    """Ore giornaliere 0-8 a passi di un quarto d'ora ("3", "3h", "2,5",
    "1,25", "2h30'", "1h15", "0")."""
    pairs: list[tuple[str, float]] = []
    for q in range(int(HOURS_MAX_DAY / HOURS_STEP) + 1):
        v = q * HOURS_STEP
        for f in _hours_forms(v):
            for variant in _digit_variants(f):
                pairs.append((variant, v))
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
    forms: dict[Any, str] = field(default_factory=dict)   # valore -> scrittura vincolata piu' probabile

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


Finished = tuple[str, tuple[int, ...], float]


@dataclass(eq=False)
class _Search:
    """Una ricerca a fascio (vincolata a un lessico o libera) su un ritaglio del lotto."""

    crop: int                         # indice del ritaglio nel lotto
    width: int
    max_steps: int
    lexicon: Lexicon | None
    allowed: Any = None               # id ammessi nella ricerca libera
    length_norm: bool = False
    live: list[_Hyp] = field(default_factory=lambda: [_Hyp((), "", 0.0)])
    rows: list[int] = field(default_factory=list)   # riga del lotto di ogni ipotesi viva
    finished: list[Finished] = field(default_factory=list)
    done: bool = False


@dataclass
class _DecoderParts:
    """Moduli del decoder TrOCR usati dalla decodifica a lotti."""

    embed_tokens: Any
    embed_positions: Any
    layernorm_embedding: Any
    layers: list[Any]
    lm_head: Any
    heads: int
    head_dim: int


class TrOCRRecognizer:
    """TrOCR su CPU con beam search vincolato a un lessico.

    I ritagli sono elaborati a lotti di lunghezza simile: l'encoder gira una
    volta per ritaglio, le chiavi/valori della cross-attention sono calcolati una
    sola volta per ritaglio e il decoder fa avanzare *insieme* tutte le ricerche
    del lotto (vincolate e libere), con una sola chiamata per passo e le ipotesi
    di tutti i ritagli impilate: sulla CPU un passo costa soprattutto la lettura
    dei pesi, quasi indipendente dal numero di righe. La cross-attention di ogni
    ipotesi usa le chiavi/valori del proprio ritaglio senza duplicarle. Pesi in
    float32: il risultato di ogni ritaglio non dipende dagli altri del lotto.
    Se la struttura interna del modello non e' quella attesa si ripiega su una
    decodifica senza cache (lenta ma equivalente).
    """

    # Ritagli letti insieme (encoder e decoder). Con 8 ritagli le chiavi/valori della
    # cross-attention occupano circa 450 MB; lotti piu' grandi riducono i passi del
    # decoder ma non il tempo totale (misurato su CPU a 4 core) e usano piu' memoria.
    BATCH = 8

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
        self._parts = self._decoder_parts()
        self._fast = self._parts is not None
        self._verified = False
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

    # ------------------------------------------------- decoder a lotti
    def _decoder_parts(self) -> _DecoderParts | None:
        """Moduli interni del decoder TrOCR (``None`` se la struttura non e' quella attesa)."""
        try:
            core = self.decoder.model.decoder
            layers = list(core.layers)
            lm_head = self.decoder.output_projection
            first = layers[0]
            heads = int(first.self_attn.num_heads)
            head_dim = int(first.self_attn.head_dim)
            for layer in layers:
                for name in ("self_attn", "encoder_attn", "self_attn_layer_norm", "encoder_attn_layer_norm",
                             "fc1", "fc2", "final_layer_norm", "activation_fn"):
                    getattr(layer, name)
                for attn in (layer.self_attn, layer.encoder_attn):
                    for name in ("q_proj", "k_proj", "v_proj", "out_proj", "scaling"):
                        getattr(attn, name)
            return _DecoderParts(core.embed_tokens, core.embed_positions, getattr(core, "layernorm_embedding", None),
                                 layers, lm_head, heads, head_dim)
        except (AttributeError, IndexError, TypeError):
            log.info("Struttura del decoder non riconosciuta: decodifica senza lotti", exc_info=True)
            return None

    def _cross_kv(self, enc: Any) -> list[tuple[Any, Any]]:
        """Chiavi (trasposte) e valori della cross-attention di ogni livello per i
        ritagli del lotto, contigui nella disposizione usata a ogni passo."""
        p = self._parts
        assert p is not None
        n, s, _ = enc.shape
        out = []
        for layer in p.layers:
            att = layer.encoder_attn
            k_t = att.k_proj(enc).view(n, s, p.heads, p.head_dim).permute(0, 2, 3, 1).contiguous()
            v = att.v_proj(enc).view(n, s, p.heads, p.head_dim).transpose(1, 2).contiguous()
            out.append((k_t, v))
        return out

    def _step(self, tokens: Any, pos: int, cache: list[tuple[Any, Any]] | None, cross: list[tuple[Any, Any]],
              row_crop: Any) -> tuple[Any, list[tuple[Any, Any]]]:
        """Un passo del decoder per tutte le ipotesi del lotto.

        ``tokens``: un token per riga; ``cache``: chiavi/valori della self-attention
        di ogni livello (righe, teste, passi, dim) o ``None`` al primo passo;
        ``row_crop``: ritaglio di ogni riga (righe ordinate per ritaglio).
        Restituisce le log-probabilita' del token successivo e la cache aggiornata."""
        torch = self.torch
        p = self._parts
        assert p is not None
        rows = int(tokens.shape[0])
        n_crops = int(cross[0][0].shape[0])
        ids = tokens.view(rows, 1)
        x = p.embed_tokens(ids) + p.embed_positions(ids, past_key_values_length=pos).to(ids.device)
        if p.layernorm_embedding is not None:
            x = p.layernorm_embedding(x)
        # posto di ogni riga nella matrice (ritaglio, ipotesi) della cross-attention
        counts = torch.bincount(row_crop, minlength=n_crops)
        wmax = max(1, int(counts.max()))
        first = torch.cumsum(counts, 0) - counts
        flat = row_crop * wmax + (torch.arange(rows, device=row_crop.device) - first[row_crop])
        d = p.heads * p.head_dim
        new_cache: list[tuple[Any, Any]] = []
        for li, layer in enumerate(p.layers):
            sa = layer.self_attn
            residual = x
            q = (sa.q_proj(x) * sa.scaling).view(rows, 1, p.heads, p.head_dim).transpose(1, 2)
            k = sa.k_proj(x).view(rows, 1, p.heads, p.head_dim).transpose(1, 2)
            v = sa.v_proj(x).view(rows, 1, p.heads, p.head_dim).transpose(1, 2)
            if cache is not None:
                k = torch.cat([cache[li][0], k], dim=2)
                v = torch.cat([cache[li][1], v], dim=2)
            new_cache.append((k, v))
            w = torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)
            a = torch.matmul(w, v).transpose(1, 2).reshape(rows, 1, d)
            x = layer.self_attn_layer_norm(residual + sa.out_proj(a))

            ca = layer.encoder_attn
            residual = x
            q = (ca.q_proj(x) * ca.scaling).view(rows, p.heads, p.head_dim)
            qp = q.new_zeros(n_crops * wmax, p.heads, p.head_dim)
            qp[flat] = q
            qp = qp.view(n_crops, wmax, p.heads, p.head_dim).transpose(1, 2)
            ck_t, cv = cross[li]
            w = torch.softmax(torch.matmul(qp, ck_t), dim=-1)
            a = torch.matmul(w, cv).transpose(1, 2).reshape(n_crops * wmax, d)[flat]
            x = layer.encoder_attn_layer_norm(residual + ca.out_proj(a.view(rows, 1, d)))

            residual = x
            x = layer.final_layer_norm(residual + layer.fc2(layer.activation_fn(layer.fc1(x))))
        logits = p.lm_head(x[:, -1, :])
        return torch.log_softmax(logits.float(), dim=-1), new_cache

    VERIFY_STEPS = 4          # passi della decodifica confrontati con il modello originale

    def _verify_fast_path(self, enc: Any) -> bool:
        """Confronta i primi passi della decodifica a lotti (con la cache della
        self-attention) con il modello originale senza cache, seguendo la lettura
        "golosa": se differiscono (versione di transformers diversa, posizioni o
        cache gestite in altro modo) si usa il ripiego lento."""
        torch = self.torch
        diff = 0.0
        try:
            enc1 = enc[:1]
            cross = self._cross_kv(enc1)
            row_crop = torch.zeros(1, dtype=torch.long, device=self.device)
            ids = [self.start_id]
            lp, cache = self._step(torch.tensor([self.start_id], dtype=torch.long, device=self.device), 0, None,
                                   cross, row_crop)
            for step in range(self.VERIFY_STEPS):
                seq = torch.tensor([ids], dtype=torch.long, device=self.device)
                ref = self.decoder(input_ids=seq, encoder_hidden_states=enc1, use_cache=False)
                ref_lp = torch.log_softmax(ref.logits[:, -1, :].float(), dim=-1)
                diff = max(diff, float((lp - ref_lp).abs().max()))
                if not math.isfinite(diff) or diff > 1e-3:
                    break
                tok = int(ref_lp[0].argmax())
                if tok == self.eos_id or step == self.VERIFY_STEPS - 1:
                    break
                ids.append(tok)
                lp, cache = self._step(torch.tensor([tok], dtype=torch.long, device=self.device), step + 1, cache,
                                       cross, row_crop)
        except Exception:  # noqa: BLE001 - API interna di transformers cambiata
            log.warning("Decodifica a lotti non disponibile, uso il ripiego lento", exc_info=True)
            return False
        if not math.isfinite(diff) or diff > 1e-3:
            log.warning("Decodifica a lotti non coerente con il modello (scarto %.2g): uso il ripiego lento", diff)
            return False
        return True

    # ------------------------------------------------------- beam search
    def _advance(self, job: _Search, lps: Any, step: int) -> list[tuple[int, int, _Hyp]]:
        """Un passo della ricerca ``job``: aggiorna le ipotesi concluse e restituisce
        le nuove ipotesi vive come (riga madre, token, ipotesi)."""
        torch = self.torch
        lex = job.lexicon
        eos = self.eos_id
        cand_scores = []
        cand_parent = []
        cand_tok = []
        for i, h in enumerate(job.live):
            if not math.isfinite(h.score):
                continue
            row = job.rows[i]
            if lex is not None:
                allowed = self._allowed_for(lex, h.key)
                eos_ok = lex.complete(h.key) is not None
            else:
                allowed = job.allowed
                eos_ok = bool(h.key)
            if eos_ok:
                job.finished.append((h.key, h.ids, h.score + float(lps[row, eos])))
            if allowed.numel() and step < job.max_steps:
                cand_scores.append(lps[row, allowed] + h.score)
                cand_parent.append(torch.full((allowed.numel(),), i, dtype=torch.long))
                cand_tok.append(allowed)
        if not cand_scores:
            return []
        scores = torch.cat(cand_scores)
        parents = torch.cat(cand_parent)
        toks = torch.cat(cand_tok)
        k = min(job.width * 4 if lex is not None else job.width, int(scores.numel()))
        top_s, top_i = torch.topk(scores, k)
        best_live = float(top_s[0])
        if len(job.finished) >= job.width:
            if lex is not None:
                # conta i valori distinti gia' conclusi (le varianti di scrittura non contano)
                best_by_value: dict[Any, float] = {}
                for fk, _fi, fs in job.finished:
                    fv = lex.complete(fk)
                    best_by_value[fv] = max(fs, best_by_value.get(fv, -math.inf))
                ranked = sorted(best_by_value.values(), reverse=True)
            else:
                ranked = sorted((self._rank(f, job.length_norm) for f in job.finished), reverse=True)
            bound = best_live if not job.length_norm else best_live / (step + 2)
            if len(ranked) >= job.width and ranked[job.width - 1] >= bound:
                return []
        out: list[tuple[int, int, _Hyp]] = []
        seen: set[str] = set()
        for s, j in zip(top_s.tolist(), top_i.tolist()):
            if not math.isfinite(s):
                continue
            i = int(parents[j])
            t = int(toks[j])
            h = job.live[i]
            key = h.key + self.tok_key[t]
            if lex is not None:
                # una sola ipotesi per "contenuto": le varianti (8:00 / 8.00 / 8,00)
                # non devono occupare tutto il beam a scapito di valori diversi
                sig = lex.signature(key)
                if sig in seen:
                    continue
                seen.add(sig)
            out.append((job.rows[i], t, _Hyp(h.ids + (t,), key, s)))
            if len(out) >= job.width:
                break
        return out

    def _run(self, jobs: Sequence[_Search], start_cache: list[tuple[Any, Any]], start_lps: Any,
             cross: list[tuple[Any, Any]]) -> None:
        """Fa avanzare insieme tutte le ricerche (una chiamata al decoder per passo)."""
        torch = self.torch
        for job in jobs:
            job.rows = [job.crop]
        active = [j for j in jobs if not j.done]
        lps = start_lps
        cache = start_cache
        step = 0
        while active:
            parents: list[int] = []
            tokens: list[int] = []
            crops: list[int] = []
            still: list[_Search] = []
            # righe ordinate per ritaglio (richiesto dalla cross-attention a lotti)
            for job in sorted(active, key=lambda j: j.crop):
                nxt = self._advance(job, lps, step)
                if not nxt:
                    job.done = True
                    continue
                job.live = [h for _r, _t, h in nxt]
                job.rows = list(range(len(parents), len(parents) + len(nxt)))
                parents += [r for r, _t, _h in nxt]
                tokens += [t for _r, t, _h in nxt]
                crops += [job.crop] * len(nxt)
                still.append(job)
            active = still
            if not active:
                break
            idx = torch.tensor(parents, dtype=torch.long, device=self.device)
            cache = [(k.index_select(0, idx), v.index_select(0, idx)) for k, v in cache]
            tok = torch.tensor(tokens, dtype=torch.long, device=self.device)
            row_crop = torch.tensor(crops, dtype=torch.long, device=self.device)
            lps, cache = self._step(tok, step + 1, cache, cross, row_crop)
            step += 1

    @staticmethod
    def _rank(item: Finished, length_norm: bool) -> float:
        _, ids, score = item
        return score / (len(ids) + 1) if length_norm else score

    def _search_slow(self, enc1: Any, width: int, max_steps: int, lexicon: Lexicon | None,
                     charset: frozenset[str] | None) -> list[Finished]:
        """Decodifica di riferimento senza cache (ripiego se l'API interna del modello cambia)."""
        torch = self.torch
        live = [_Hyp((), "", 0.0)]
        finished: list[Finished] = []
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

    @staticmethod
    def _needs_free(req: ReadRequest, constrained: Sequence[Finished]) -> bool:
        """La lettura libera serve se richiesta e se non coincide gia' con la migliore
        sequenza vincolata (probabilita' > 1/2: e' per forza anche la lettura "golosa")."""
        if req.lexicon is None:
            return True
        if not req.free:
            return False
        best = max(constrained, key=lambda f: f[2], default=None)
        return not (best is not None and best[2] > math.log(0.5) and req.charset is None and req.free_beams <= 1)

    def _reading(self, req: ReadRequest, constrained: Sequence[Finished], free: Sequence[Finished] | None) -> Reading:
        reading = Reading()
        lex = req.lexicon
        if lex is not None:
            by_value: dict[Any, list[float]] = {}
            best_form: dict[Any, tuple[str, float]] = {}
            for key, _ids, score in constrained:
                value = lex.complete(key)
                if value is not None:
                    by_value.setdefault(value, []).append(score)
                    if score > best_form.get(value, ("", -math.inf))[1]:
                        best_form[value] = (key, score)
            cands = [(v, _logsumexp(s)) for v, s in by_value.items()]
            cands.sort(key=lambda c: c[1], reverse=True)
            reading.candidates = cands
            reading.forms = {v: k for v, (k, _s) in best_form.items()}
            if free is None and req.free:
                best = max(constrained, key=lambda f: f[2])
                reading.free_text = self._decode_ids(best[1])
                reading.free_logprob = best[2]
                reading.free_tokens = len(best[1])
                return reading
        if free:
            best = max(free, key=lambda f: self._rank(f, True))
            reading.free_text = self._decode_ids(best[1])
            reading.free_logprob = best[2]
            reading.free_tokens = len(best[1])
        return reading

    def _new_job(self, crop: int, req: ReadRequest, constrained: bool) -> _Search:
        if constrained:
            assert req.lexicon is not None
            return _Search(crop, max(1, req.beams), req.lexicon.max_chars, req.lexicon)
        return _Search(crop, max(1, req.free_beams), req.max_tokens, None, allowed=self._charset_for(req.charset),
                       length_norm=True)

    def _read_batch(self, batch: Sequence[ReadRequest], enc: Any) -> list[Reading]:
        """Letture dei ritagli di un lotto (``enc``: uscite dell'encoder, una per ritaglio)."""
        torch = self.torch
        n = len(batch)
        cross = self._cross_kv(enc)
        start = torch.full((n,), self.start_id, dtype=torch.long, device=self.device)
        crop_ids = torch.arange(n, dtype=torch.long, device=self.device)
        start_lps, start_cache = self._step(start, 0, None, cross, crop_ids)
        # Le ricerche vincolate e le letture libere avanzano insieme: la lettura
        # libera di un campo con lessico serve solo se la lettura vincolata non
        # basta, ma calcolarla in anticipo costa una riga in piu' per ritaglio e
        # nessun passo in piu' (i passi, non le righe, determinano il tempo).
        lex_jobs: dict[int, _Search] = {}
        free_jobs: dict[int, _Search] = {}
        for i, req in enumerate(batch):
            if req.lexicon is not None:
                lex_jobs[i] = self._new_job(i, req, True)
            if req.lexicon is None or req.free:
                free_jobs[i] = self._new_job(i, req, False)
        self._run([*lex_jobs.values(), *free_jobs.values()], start_cache, start_lps, cross)
        out = []
        for i, req in enumerate(batch):
            constrained = lex_jobs[i].finished if i in lex_jobs else []
            free = free_jobs[i].finished if i in free_jobs and self._needs_free(req, constrained) else None
            out.append(self._reading(req, constrained, free))
        return out

    def _read_slow(self, req: ReadRequest, enc1: Any) -> Reading:
        """Lettura di un ritaglio con la decodifica di riferimento (ripiego)."""
        constrained: list[Finished] = []
        if req.lexicon is not None:
            constrained = self._search_slow(enc1, max(1, req.beams), req.lexicon.max_chars, req.lexicon, None)
        free = None
        if self._needs_free(req, constrained):
            free = self._search_slow(enc1, max(1, req.free_beams), req.max_tokens, None, req.charset)
        return self._reading(req, constrained, free)

    @staticmethod
    def _decode_length(req: ReadRequest) -> int:
        """Passi di decodifica massimi di una richiesta (per raggruppare quelle simili)."""
        steps = req.max_tokens if (req.free or req.lexicon is None) else 0
        if req.lexicon is not None:
            steps = max(steps, req.lexicon.max_chars)
        return steps

    def read(self, requests: Sequence[ReadRequest],
             progress: Callable[[int, int], None] | None = None) -> list[Reading]:
        torch = self.torch
        total = len(requests)
        results: list[Reading | None] = [None] * total
        # richieste di lunghezza simile nello stesso gruppo: il numero di passi del
        # gruppo e' quello della ricerca piu' lunga
        order = sorted(range(total), key=lambda i: self._decode_length(requests[i]))
        done = 0
        with self.lock, torch.inference_mode():
            for g0 in range(0, total, self.BATCH):
                idx = order[g0:g0 + self.BATCH]
                batch = [requests[i] for i in idx]
                enc = self.encode([r.image for r in batch])
                if self._fast and not self._verified:
                    self._fast = self._verify_fast_path(enc)
                    self._verified = True
                if self._fast:
                    readings = self._read_batch(batch, enc)
                else:
                    readings = [self._read_slow(req, enc[j:j + 1]) for j, req in enumerate(batch)]
                for i, reading in zip(idx, readings):
                    results[i] = reading
                done += len(batch)
                if progress is not None:
                    progress(done, total)
        return [r if r is not None else Reading() for r in results]


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
_REPLACED: set[tuple[str, str]] = set()     # modelli danneggiati gia' riscaricati in questa sessione
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


def _repo_of(snapshot: Path, cache_dir: Path) -> Path | None:
    """Cartella ``models--...`` che contiene la revisione ``snapshot``, se e' nella cache."""
    if snapshot.parent.name != "snapshots":
        return None
    repo = snapshot.parent.parent
    try:
        repo.resolve().relative_to(cache_dir.resolve())
    except (OSError, ValueError):
        return None
    return repo if repo.name.startswith("models--") else None


def _discard_cached_model(repo: Path) -> bool:
    """Elimina (o, se non si puo', accantona) la copia locale danneggiata del modello."""

    def writable_retry(func: Callable[..., Any], target: str, _exc: BaseException) -> None:
        # su Windows i file della cache sono in sola lettura e rmtree non li elimina
        os.chmod(target, stat.S_IWRITE)
        func(target)

    try:
        if sys.version_info >= (3, 12):
            shutil.rmtree(repo, onexc=writable_retry)
        else:
            shutil.rmtree(repo, onerror=lambda f, p, info: writable_retry(f, p, info[1]))
        return True
    except OSError:
        log.warning("Impossibile eliminare il modello danneggiato in %s", repo, exc_info=True)
    try:
        repo.rename(repo.with_name(f"{repo.name}.corrotto-{int(time.time())}"))
        return True
    except OSError:
        log.warning("Impossibile accantonare il modello danneggiato in %s", repo, exc_info=True)
        return False


def _damaged_message(folder: Path) -> str:
    return (f"Il modello di riconoscimento offline è danneggiato e non è stato possibile sostituirlo "
            f"automaticamente: chiudere Sirio OCR, eliminare la cartella «{folder}» e riprovare "
            "(il modello verrà scaricato di nuovo).")


def _import_failure_message(exc: BaseException) -> str:
    """Messaggio per torch/transformers presenti ma non importabili."""
    text = str(exc)
    if isinstance(exc, OSError) and (".dll" in text.lower() or "winerror" in text.lower()
                                     or getattr(exc, "winerror", None) is not None):
        # tipicamente manca il runtime Visual C++: reinstallare i pacchetti non serve
        return _DLL_MSG
    return _MISSING_MSG


def _missing_components() -> str | None:
    """None se torch e transformers sono importabili, altrimenti il messaggio d'errore."""
    if "ok" in _IMPORT_STATE:
        return None if _IMPORT_STATE["ok"] else _IMPORT_STATE.get("msg", _MISSING_MSG)
    for mod in ("torch", "transformers"):
        if importlib.util.find_spec(mod) is None:
            _IMPORT_STATE.update(ok=False, msg=_MISSING_MSG)
            return _MISSING_MSG
    try:
        importlib.import_module("torch")
        importlib.import_module("transformers")
    except Exception as exc:  # noqa: BLE001 - libreria presente ma non funzionante
        log.warning("torch/transformers presenti ma non importabili", exc_info=True)
        msg = _import_failure_message(exc)
        _IMPORT_STATE.update(ok=False, msg=msg)
        return msg
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


def _exc_chain(exc: BaseException) -> Iterable[BaseException]:
    """L'eccezione e le sue cause (``__cause__``/``__context__``), senza cicli."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


_OOM_TEXTS = (
    "out of memory", "not enough memory", "can't allocate memory", "cannot allocate memory",
    "unable to mmap", "defaultcpuallocator", "bad_alloc", "paging file", "file di paging",
    "memoria insufficiente", "(os error 12)", "(os error 1455)",
)


def _is_oom(exc: BaseException) -> bool:
    """True se l'errore (o una sua causa) e' una mancanza di memoria: ``MemoryError``,
    ``ENOMEM`` o i messaggi di torch/safetensors/Windows (torch su CPU solleva
    ``RuntimeError``, safetensors un errore con "Cannot allocate memory")."""
    for e in _exc_chain(exc):
        if isinstance(e, MemoryError):
            return True
        if isinstance(e, OSError) and (e.errno == errno.ENOMEM or getattr(e, "winerror", None) in (8, 1455)):
            return True
        text = str(e).lower()
        if any(t in text for t in _OOM_TEXTS):
            return True
    return False


_DISK_FULL_TEXTS = ("no space left", "(os error 28)", "(os error 112)", "disk full", "not enough space",
                    "spazio insufficiente", "spazio su disco insufficiente")


def _free_bytes(path: Path) -> float | None:
    for p in (path, *path.parents):
        try:
            return float(shutil.disk_usage(p).free)
        except OSError:
            continue
    return None


def _disk_full_error(size_mb: float | None, needed: float | None = None) -> EngineError:
    if needed:
        size = f" (servono circa {_fmt_size(needed / 1e6)} liberi)"
    else:
        size = f" (circa {_fmt_size(size_mb)})" if size_mb else ""
    return EngineError(
        f"Spazio su disco insufficiente per scaricare il modello di riconoscimento offline{size}. "
        "Liberare spazio e riprovare."
    )


def _is_disk_full(exc: BaseException, cache_dir: Path | None, remaining: float) -> bool:
    for e in _exc_chain(exc):
        if isinstance(e, OSError) and (e.errno == errno.ENOSPC or getattr(e, "winerror", None) in (39, 112)):
            return True
        if any(t in str(e).lower() for t in _DISK_FULL_TEXTS):
            return True
    # il download (hf_xet) puo' fallire con un errore generico: conta lo spazio rimasto
    if cache_dir is not None and remaining > 0:
        free = _free_bytes(cache_dir)
        return free is not None and free < remaining
    return False


def _download_error(exc: BaseException, model_name: str, size_mb: float | None,
                    cache_dir: Path | None = None, remaining: float = 0.0) -> EngineError:
    size = f" (circa {_fmt_size(size_mb)})" if size_mb else ""
    if _is_disk_full(exc, cache_dir, remaining):
        return _disk_full_error(size_mb)
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


DISK_MARGIN = 200e6         # spazio libero da lasciare oltre al modello (byte)


def download_model(model_name: str, cache_dir: Path, progress: ProgressFn = no_progress,
                   message: str | None = None) -> Path:
    """Scarica il modello da huggingface.co in ``cache_dir`` (solo i file necessari),
    riportando l'avanzamento. Restituisce la cartella del modello. ``message``
    sostituisce l'inizio dei messaggi di avanzamento del download."""
    label = _model_label(model_name)
    size_mb = KNOWN_MODELS.get(model_name, {}).get("mb")
    try:
        from huggingface_hub import HfApi, snapshot_download  # noqa: PLC0415
    except ImportError as exc:
        raise EngineError(_MISSING_MSG) from exc
    head = message or f"Download del modello {label} (solo al primo utilizzo)"
    progress(0.0, f"{message}…" if message else f"Connessione a huggingface.co per scaricare il modello {label}…")
    try:
        info = HfApi().model_info(model_name, files_metadata=True)
    except Exception as exc:  # noqa: BLE001
        raise _download_error(exc, model_name, size_mb) from exc
    siblings = [(s.rfilename, int(s.size or 0)) for s in (info.siblings or [])]
    patterns = _weight_patterns([f for f, _ in siblings])
    total = sum(sz for f, sz in siblings if any(fnmatch.fnmatch(f, p) for p in patterns)) or (
        (size_mb or 0) * 1e6)
    cache_dir.mkdir(parents=True, exist_ok=True)
    blobs = cache_dir / _repo_folder(model_name) / "blobs"
    start_size = _dir_size(blobs)
    # spazio libero controllato prima: un disco pieno durante il download produce
    # solo errori generici (hf_xet) e un avviso che nessuno vede
    remaining = max(0.0, total - start_size)
    free = _free_bytes(cache_dir)
    if free is not None and remaining and free < remaining * 1.1 + DISK_MARGIN:
        raise _disk_full_error(size_mb, needed=remaining * 1.1 + DISK_MARGIN)
    outcome: dict[str, Any] = {}

    def worker() -> None:
        try:
            outcome["path"] = snapshot_download(model_name, cache_dir=str(cache_dir), allow_patterns=patterns,
                                                revision=getattr(info, "sha", None),
                                                tqdm_class=_silent_tqdm())
        except BaseException as exc:  # noqa: BLE001 - riportata nel thread chiamante
            outcome["error"] = exc

    thread = threading.Thread(target=worker, name="sirio-download-modello", daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(0.5)
        done = max(0, _dir_size(blobs) - start_size)
        frac = min(0.99, done / total) if total else 0.0
        msg = f"{head}: {_fmt_mb(done)}"
        if total:
            msg += f" di {_fmt_mb(total)}"
        progress(frac, msg + "…")
    if "error" in outcome:
        left = max(0.0, total - _dir_size(blobs))
        raise _download_error(outcome["error"], model_name, size_mb, cache_dir, left) from outcome["error"]
    path = local_model_path(model_name, cache_dir)
    if path is None:
        raise EngineError(f"Download del modello {label} incompleto: riprovare.")
    progress(1.0, f"Modello {label} scaricato.")
    return path


class ModelDamagedError(EngineError):
    """File del modello presenti ma illeggibili (download interrotto, disco danneggiato...)."""


def _load_recognizer(path: Path, model_name: str, device: str) -> TrOCRRecognizer:
    """Carica il modello. Solleva ``EngineError`` con ``OOM_MSG`` se manca memoria e
    ``ModelDamagedError`` se i file del modello non sono leggibili."""
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    try:
        # niente barre di avanzamento su stderr (l'applicazione puo' non avere una console)
        transformers.utils.logging.disable_progress_bar()
    except Exception:  # noqa: BLE001 - facoltativo
        log.debug("Impossibile disattivare le barre di avanzamento di transformers", exc_info=True)
    try:
        processor = transformers.TrOCRProcessor.from_pretrained(str(path), local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        text = str(exc).lower()
        if "sentencepiece" in text or "tiktoken" in text or "protobuf" in text:
            raise EngineError(
                f"Il modello {_model_label(model_name)} richiede i pacchetti «sentencepiece» e «protobuf», "
                "non installati: scegliere un altro modello locale nelle impostazioni."
            ) from exc
        if _is_oom(exc):
            raise EngineError(OOM_MSG) from exc
        log.warning("Componenti del modello in %s non leggibili", path, exc_info=True)
        raise ModelDamagedError("Modello di riconoscimento offline non valido o incompleto.") from exc
    try:
        model = transformers.VisionEncoderDecoderModel.from_pretrained(str(path), local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        if _is_oom(exc):
            raise EngineError(OOM_MSG) from exc
        log.warning("Pesi del modello in %s non leggibili", path, exc_info=True)
        raise ModelDamagedError("Impossibile caricare il modello di riconoscimento offline.") from exc
    try:
        model.eval()
        model.to(torch.device(device))
        # Pesi in float32: la quantizzazione dinamica int8 del decoder (scala unica per
        # tutto il lotto) renderebbe le letture dipendenti dai ritagli letti insieme e
        # con la decodifica a lotti non e' piu' necessaria per la velocita'.
        return TrOCRRecognizer(model, processor, model_name, device)
    except Exception as exc:
        if _is_oom(exc):
            raise EngineError(OOM_MSG) from exc
        raise


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
VUOTA, TESTO, TRATTINO, CROCETTA, FIRMA, SEGNO, TIMBRO, NULLO = (
    "vuota", "testo", "trattino", "crocetta", "firma", "segno", "timbro", "nullo")
# NULLO: barretta "/" isolata (riga non lavorata): nessun valore


@dataclass
class PagePlan:
    """Esito dell'analisi d'immagine e ritagli da leggere."""

    cells: dict[int, dict[str, str]] = field(default_factory=dict)       # giorno -> campo -> stato
    requests: list[ReadRequest] = field(default_factory=list)
    header_ink: dict[str, bool] = field(default_factory=dict)            # campo -> scrittura presente
    sostituzione: str | None = None
    sostituzione_incerta: bool = False
    firma_coordinatore: bool = False
    firma_coordinatore_incerta: bool = False   # inchiostro di altri segni (firma/timbro del referente)
    timbro_referente: bool = False
    totale_ink: bool = False
    data_ink: bool = False
    notes: list[str] = field(default_factory=list)
    aspect: dict[str, float] = field(default_factory=dict)              # chiave -> larghezza/altezza del ritaglio
    flags: dict[int, dict[str, str]] = field(default_factory=dict)      # giorno -> campo -> motivo di dubbio
    note_scritta: set[int] = field(default_factory=set)   # giorni con una nota scritta per davvero (vedi plan_page)
    firme_vere: set[int] = field(default_factory=set)     # giorni con una firma dalla forma di firma (non un tratto)


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


# Bianco aggiunto sopra e sotto i ritagli di orari e ore (frazione dell'altezza): TrOCR
# ridimensiona ogni ritaglio a un quadrato e le cifre di un ritaglio basso e largo
# risultano molto stirate in verticale. Misurato su 297 celle di quattro fogli reali di
# operatori diversi: lettura migliore corretta 240 -> 243, probabilita' media del valore
# vero 0,77 -> 0,80, a costo nullo.
VPAD_NUMERI = 0.3


def _text_request(key: str, crop: np.ndarray, lexicon: Lexicon | None, **kw: Any) -> ReadRequest:
    vpad = kw.pop("vpad", 0.0)
    img = model_image(crop, kw.pop("min_aspect", 1.0))
    if vpad > 0:
        p = int(round(vpad * img.shape[0]))
        img = cv2.copyMakeBorder(img, p, p, 0, 0, cv2.BORDER_CONSTANT, value=255)
    return ReadRequest(key=key, image=img, lexicon=lexicon, **kw)


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
    colonne "Firma operatore" e "Note", margine destro e pie' di pagina (il
    timbro del referente scolastico copre spesso le ultime righe)."""
    H, W = gray.shape[:2]
    rh = grid.row_height
    zx0 = max(0, int(grid.col_x[8] - 0.3 * rh))
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
        if not (w >= 1.5 * rh and h >= 1.5 * rh and 0.5 <= w / float(h) <= 2.0 and area >= 0.3 * w * h):
            continue
        if h > TIMBRO_MAX_RIGHE * rh:
            # macchia alta piu' di un timbro: fondino retinato della fotocopia o scrittura
            # fusa, non un timbro (sotto un "timbro" note e firme non si leggono)
            continue
        if y + h + zy0 < grid.row_y[-1] - TIMBRO_DAL_FONDO * rh:
            # lontano dal fondo della tabella: un blocco di note scritte su piu' righe
            # ("USCITO / PRIMA" in due giorni di fila), non il timbro della scuola
            continue
        if x + zx0 + w / 2.0 < grid.col_x[9]:
            # nella colonna delle firme: firme di righe vicine fuse insieme somigliano a
            # una macchia; un timbro vero e' piu' grande e scende fino al fondo della tabella
            if w < 2.2 * rh or h < 2.2 * rh or y + h + zy0 < grid.row_y[-1] - 0.3 * rh:
                continue
        out.append((x + zx0, y + zy0, x + w + zx0, y + h + zy0))
    return out


TIMBRO_DAL_FONDO = 3.0     # righe: un timbro arriva almeno fin qui dal fondo della tabella
TIMBRO_MAX_RIGHE = 8.0     # righe: altezza massima di un timbro (misurati fino a 6)


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


STRUTTURA_SOTTO = 0.3      # quota minima dell'inchiostro di una struttura sotto la tabella (pie' di pagina)...
STRUTTURA_SOTTO_RIGHE = 1.0   # ... oppure estensione sotto la tabella (righe)
QUOTA_ESTRANEA = 0.5       # quota dell'inchiostro della cella che appartiene a una struttura: non e' della riga
STRUTTURA_PUNTINO = 0.07   # lato (righe) sotto cui un frammento e' un puntino del retino
STRUTTURA_MAX_RIGHE = 10.0 # righe: oltre, la "struttura" che sale nella tabella non e' timbro o firma
QUOTA_DUBBIA = 0.15        # ... quota oltre la quale il segno della riga e' da verificare


@dataclass
class ForeignInk:
    """Inchiostro delle colonne di destra (assenze, firma, note) che appartiene a
    strutture piu' grandi di una riga: timbro tondo della scuola, firma del referente
    o del coordinatore appoggiate sull'angolo della tabella. ``mask``: pixel delle
    strutture; ``ink``: tutto l'inchiostro della zona (linee stampate escluse);
    ``origin``: angolo in alto a sinistra della zona nell'immagine."""

    mask: np.ndarray
    ink: np.ndarray
    origin: tuple[int, int]

    def share(self, box: Box, inner: float = 0.06) -> float:
        """Quota dell'inchiostro del riquadro che appartiene alle strutture."""
        ox, oy = self.origin
        x0, y0, x1, y1 = box
        mx, my = int(round(inner * (x1 - x0))), int(round(inner * (y1 - y0)))
        h, w = self.ink.shape
        a0, a1 = max(0, y0 + my - oy), min(h, y1 - my - oy)
        b0, b1 = max(0, x0 + mx - ox), min(w, x1 - mx - ox)
        if a1 <= a0 or b1 <= b0:
            return 0.0
        total = int(self.ink[a0:a1, b0:b1].sum())
        if total == 0:
            return 0.0
        return float(self.mask[a0:a1, b0:b1].sum()) / total

    def erase(self, gray: np.ndarray) -> np.ndarray:
        """Copia di ``gray`` senza l'inchiostro delle strutture."""
        out = gray.copy()
        ox, oy = self.origin
        h, w = self.mask.shape
        sel = cv2.dilate(self.mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        view = out[oy:oy + h, ox:ox + w]
        view[sel[:view.shape[0], :view.shape[1]]] = 255
        return out


def _drop_specks(m: np.ndarray, unit: float) -> np.ndarray:
    """Maschera senza i puntini isolati (retino grigio delle fotocopie, grana della carta)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return m
    specks = st[:, cv2.CC_STAT_AREA] < max(4.0, (STRUTTURA_PUNTINO * unit) ** 2)
    specks[0] = False
    return m & ~specks[lab]


def foreign_ink(gray: np.ndarray, grid: TableGrid) -> ForeignInk | None:
    """Strutture d'inchiostro che non appartengono a una riga (vedi ``ForeignInk``).

    Una struttura (tratti uniti, compresi i frammenti vicini di un timbro) entra
    nelle righe della tabella ma sta in buona parte sotto di essa (riga dei totali,
    pie' di pagina: timbro della scuola, firme del referente e del coordinatore):
    almeno ``STRUTTURA_SOTTO`` del suo inchiostro, oppure scende per almeno
    ``STRUTTURA_SOTTO_RIGHE`` righe. La coda di una firma dell'ultima riga che
    sfiora la riga dei totali non basta. Le strutture alte piu' righe dentro la
    tabella (firme di righe vicine che si toccano) restano delle righe: i timbri
    tondi interni alla tabella sono cercati da ``stamp_boxes``."""
    H, W = gray.shape[:2]
    rh = grid.row_height
    zx0 = max(0, int(grid.col_x[6] - 0.3 * rh))
    zy0 = max(0, int(grid.row_y[0] - 0.5 * rh))
    region = gray[zy0:H, zx0:W]
    if region.size == 0 or rh <= 4:
        return None
    m = region < _ink_threshold(_paper_level(region))
    rows = [y - zy0 for y in grid.row_y] + ([grid.total_row[1] - zy0] if grid.total_row else [])
    m &= ~_line_mask(m, rh, rows, [x - zx0 for x in grid.col_x[6:]])
    # puntini (retino grigio delle fotocopie, grana): non si ricuciono, altrimenti il
    # retino unisce tutta la pagina in una sola "struttura" che scende sotto la tabella
    m = _drop_specks(m, rh)
    r = max(1, int(round(0.08 * rh)))
    link = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(link, connectivity=8)
    bottom = int(round(grid.row_y[-1] - zy0))
    if n <= 1 or bottom >= lab.shape[0]:
        return None
    labm = np.where(m, lab, 0)
    total = np.bincount(labm.ravel(), minlength=n)
    under = np.bincount(labm[bottom:].ravel(), minlength=n)
    keep = np.zeros(n, dtype=bool)
    for k in range(1, n):
        x, y, w, h, _a = (int(v) for v in stats[k])
        if y >= bottom - 0.1 * rh or total[k] == 0:
            continue                    # tutta sotto la tabella: non tocca le righe
        if bottom - y > STRUTTURA_MAX_RIGHE * rh:
            continue                    # sale per mezza tabella: fondino o scrittura fusa, non un timbro
        keep[k] = (under[k] >= STRUTTURA_SOTTO * total[k]
                   or y + h - bottom >= STRUTTURA_SOTTO_RIGHE * rh)
    if not keep.any():
        return None
    return ForeignInk(mask=keep[lab] & m, ink=m, origin=(zx0, zy0))


def _overlap(box: Box, other: Box) -> float:
    """Frazione dell'area di ``box`` coperta da ``other``."""
    x0, y0, x1, y1 = box
    ix = max(0, min(x1, other[2]) - max(x0, other[0]))
    iy = max(0, min(y1, other[3]) - max(y0, other[1]))
    area = max(1, (x1 - x0) * (y1 - y0))
    return ix * iy / area


def _owned_components(gray: np.ndarray, box: Box, unit: float, band: float, rows: Sequence[float] = (),
                      cols: Sequence[float] = ()) -> tuple[list[tuple[int, int, int, int, int]],
                                                           list[tuple[int, int, int, int, int]]]:
    """Componenti d'inchiostro che toccano ``box``, divise fra proprie (per lo piu'
    dentro il riquadro) ed estranee (per lo piu' fuori: tratti di una firma o di un
    timbro vicini che sconfinano). Analisi su una zona estesa di ``band`` sopra e sotto;
    le statistiche (x, y, w, h, area) sono quelle della parte nel riquadro."""
    H, W = gray.shape[:2]
    x0, y0, x1, y1 = box
    ey0, ey1 = max(0, int(y0 - band)), min(H, int(y1 + band))
    region = gray[ey0:ey1, x0:x1]
    if region.size == 0 or y1 <= y0 or x1 <= x0:
        return [], []
    bg = _paper_level(region)
    ink_mask = region < _ink_threshold(bg)
    h_lines, v_lines = _line_mask(ink_mask, unit, [y - ey0 for y in rows], [x - x0 for x in cols], split=True)
    m = ink_mask & ~(h_lines | v_lines)
    # un tratto che attraversa una linea orizzontale resta un solo segno (la firma del
    # referente che sale nella riga del coordinatore non diventa un segno a se')
    r = max(2, int(round(0.1 * unit)))
    near = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2 * r + 1))) > 0
    conn = m | (h_lines & near)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(conn.astype(np.uint8), connectivity=8)
    lab = np.where(m, lab, 0)
    stats = stats.copy()
    stats[:, 4] = np.bincount(lab.ravel(), minlength=n)
    a, b = y0 - ey0, y1 - ey0
    inner = lab[a:b]
    inside = np.bincount(inner.ravel(), minlength=n)
    min_area = max(4.0, (0.04 * unit) ** 2)
    own: list[tuple[int, int, int, int, int]] = []
    foreign: list[tuple[int, int, int, int, int]] = []
    for k in range(1, n):
        if inside[k] < min_area:
            continue
        ys, xs = np.nonzero(inner == k)
        st = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1),
              int(inside[k]))
        (own if inside[k] >= 0.5 * stats[k][4] else foreign).append(st)
    return own, foreign


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


def _touches_signature(gray: np.ndarray, box: Box, firma_box: Box) -> bool:
    """Il segno nella cella (colonna di assenza a sinistra della firma) arriva al
    bordo destro e la firma comincia sul bordo sinistro della sua cella: e'
    probabilmente l'inizio della firma che sconfina, non una crocetta."""
    info = ink.analyze_cell(gray, box)
    if info.bbox is None:
        return False
    w = box[2] - box[0]
    if info.bbox[2] < box[2] - 0.08 * w:
        return False
    finfo = ink.analyze_cell(gray, firma_box)
    return finfo.bbox is not None and finfo.bbox[0] <= firma_box[0] + 0.06 * (firma_box[2] - firma_box[0])


def _remove_overflow(gray: np.ndarray, grid: TableGrid, g: int) -> int:
    """Cancella (in ``gray``, modificata sul posto) i tratti che entrano nella
    colonna "Note" dalla firma della stessa riga: code di firme lunghe che
    sconfinano oltre il bordo. Un tratto e' considerato uno sconfinamento se
    attraversa il bordo fra firma e note oppure se comincia sul bordo e
    dall'altra parte del bordo, alla stessa altezza, c'e' inchiostro della firma.
    Restituisce i pixel cancellati."""
    H, W = gray.shape[:2]
    rh = grid.row_height
    xb = grid.col_x[9]
    y0 = max(0, int(grid.row_y[g - 1] - 0.6 * rh))
    y1 = min(H, int(grid.row_y[g] + 0.6 * rh))
    x0 = max(0, int(grid.col_x[8]))
    x1 = min(W, int(grid.col_x[10] + 0.5 * rh))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0
    region = gray[y0:y1, x0:x1]
    m = region < _ink_threshold(_paper_level(region))
    rows = [grid.row_y[i] - y0 for i in range(max(0, g - 2), min(len(grid.row_y), g + 2))]
    m &= ~_line_mask(m, rh, rows, [grid.col_x[8] - x0, xb - x0, grid.col_x[10] - x0])
    m = _drop_specks(m, rh)      # il retino delle fotocopie unirebbe firma e nota
    lx = xb - x0
    # pezzi della stessa scrittura separati da piccoli vuoti (o dalla linea tolta) restano uniti
    r = max(1, int(round(0.08 * rh)))
    link = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(link, connectivity=8)
    gap = max(3, int(round(0.2 * rh)))
    erased = 0
    for k in range(1, n):
        bx, by, bw, bh, _a = (int(v) for v in stats[k])
        if bx + bw <= lx:           # tutto nella firma
            continue
        comp = (lab == k) & m
        crosses = bx < lx - 0.12 * rh and bx + bw > lx + 0.05 * rh
        if not crosses:
            if bx > lx + gap:
                continue
            # comincia sul bordo: inchiostro della firma subito a sinistra, alla stessa altezza?
            ys = np.nonzero(comp[:, max(0, lx):min(comp.shape[1], lx + gap)].any(axis=1))[0]
            if ys.size == 0:
                continue
            left = m[max(0, ys.min() - r):ys.max() + r + 1, max(0, lx - gap):max(0, lx - 1)]
            if not left.any():
                continue
            if bx + bw > lx + 3.0 * rh:
                # tratto lungo che continua nella cella: probabile scrittura vera
                continue
        sel = cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        region[sel] = 255
        erased += int(comp.sum())
    return erased


def split_text_lines(crop: np.ndarray, unit: float, min_width: float | None = None) -> list[np.ndarray]:
    """Divide un ritaglio (inchiostro scuro su bianco) in due righe di testo se la
    scrittura e' disposta su due righe separate da una valle netta del profilo
    orizzontale; altrimenti restituisce il ritaglio intero. ``min_width``: larghezza
    minima di ognuna delle due righe (altezze di riga; default ``LINEA_MIN_LARGHEZZA``)."""
    ink = crop < 160
    prof = ink.sum(axis=1).astype(np.float64)
    rows = np.nonzero(prof > 0)[0]
    if rows.size == 0:
        return [crop]
    top, bot = int(rows[0]), int(rows[-1]) + 1
    min_line = max(3, int(round(LINEA_MIN_ALTEZZA * unit)))
    if bot - top < 2 * min_line + 1:
        return [crop]
    smooth = np.convolve(prof, np.ones(3) / 3.0, mode="same")
    inner = smooth[top + min_line:bot - min_line]
    if inner.size == 0:
        return [crop]
    cut = top + min_line + int(np.argmin(inner))
    upper, lower = smooth[top:cut].max(), smooth[cut:bot].max()
    if smooth[cut] > LINEA_VALLE * min(upper, lower):
        return [crop]
    pad = max(2, int(round(0.15 * unit)))
    a = crop[max(0, top - pad):cut]
    b = crop[cut:min(crop.shape[0], bot + pad)]
    out = []
    widths = []
    for part in (a, b):
        cols = np.nonzero((part < 160).any(axis=0))[0]
        if cols.size == 0:
            return [crop]
        widths.append(int(cols[-1]) - int(cols[0]) + 1)
        x0, x1 = max(0, int(cols[0]) - pad), min(part.shape[1], int(cols[-1]) + 1 + pad)
        out.append(cv2.copyMakeBorder(part[:, x0:x1], pad // 2, pad // 2, 0, 0, cv2.BORDER_CONSTANT, value=255))
    # due righe vere: entrambe larghe (un frammento della riga vicina non e' una riga)
    lim = LINEA_MIN_LARGHEZZA if min_width is None else min_width
    if min(widths) < max(lim * unit, 0.3 * max(widths)):
        return [crop]
    return out


LINEA_MIN_ALTEZZA = 0.22    # altezza minima (altezze di riga) di ognuna delle due righe di una nota
LINEA_MIN_LARGHEZZA = 1.0   # larghezza minima (altezze di riga) di ognuna delle due righe
LINEA_VALLE = 0.15          # profilo nella valle fra le righe rispetto al picco della riga piu' debole
ORARI_DUE_RIGHE_LARGHEZZA = 0.6   # larghezza minima (righe) di ognuno dei due orari in una cella


def _note_reading(readings: dict[str, Reading], g: int) -> Reading | None:
    """Lettura della nota del giorno ``g`` (le note su due righe sono lette riga per riga e riunite)."""
    key = f"rows.{g}.note"
    if key in readings:
        return readings[key]
    parts = [readings[k] for k in (f"{key}#1", f"{key}#2") if k in readings]
    if not parts:
        return None
    texts = [r.free_text for r in parts if r.free_text]
    if not texts:
        return Reading()
    return Reading([], " ".join(texts), float(sum(r.free_logprob for r in parts if r.free_text)),
                   sum(r.free_tokens for r in parts if r.free_text))


NOTE_MIN_WIDTH = 0.6        # larghezza minima (altezze di riga) di una nota vera
NOTE_MIN_HEIGHT = 0.3
NOTA_SCRITTA_LARGHEZZA = 1.5   # inchiostro largo almeno cosi' (righe)...
NOTA_SCRITTA_ALTEZZA = 0.3     # ... e alto almeno cosi' (righe), in piu' tratti: e' scrittura
NOTA_SCRITTA_TRATTI = 3
NOTA_BREVE_LARGHEZZA = 0.8     # nota breve di tre segni ("104": circa 1-1,5 righe): scritta per davvero
                               # (nel resto lasciato da un timbro si chiede la larghezza piena)


NOTA_OLTRE_FIRMA = 2.5      # righe: scrittura nelle Note piu' lunga di qualsiasi coda di firma


def _ink_width(crop: np.ndarray) -> int:
    """Larghezza (pixel) dell'inchiostro di un ritaglio pulito."""
    cols = np.nonzero((crop < 160).any(axis=0))[0]
    return int(cols[-1] - cols[0] + 1) if cols.size else 0


def _substantial_note(crop: np.ndarray, unit: float, min_width: float = NOTA_SCRITTA_LARGHEZZA) -> bool:
    """Il ritaglio pulito della nota contiene scrittura vera: inchiostro largo (almeno
    ``min_width`` righe) e alto come una parola, fatto di piu' tratti (non una coda di
    firma o un trattino)."""
    m = crop < 160
    if not m.any():
        return False
    ys, xs = np.nonzero(m)
    if xs.max() - xs.min() + 1 < min_width * unit or ys.max() - ys.min() + 1 < NOTA_SCRITTA_ALTEZZA * unit:
        return False
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    big = int((stats[1:, cv2.CC_STAT_AREA] >= max(6.0, (0.06 * unit) ** 2)).sum()) if n > 1 else 0
    return big >= NOTA_SCRITTA_TRATTI


# Segni chiarissimi nelle colonne delle crocette e delle firme: ombre, aloni della
# scrittura sul retro del foglio, macchie. Misurato su 465 celle di 18 fogli: crocette
# e firme vere hanno l'inchiostro scuro almeno 132 livelli sulla carta, le ombre 20-60.
SEGNO_MIN_SCURO = 60.0      # sotto: non e' un segno a penna (cella vuota)
SEGNO_SCURO_SICURO = 110.0  # sotto: segno troppo chiaro per essere sicuro (matita, penna scarica)


# Forma: scarto dell'inchiostro da una linea (altezze di riga). Misurato su 18 fogli: firme
# vere almeno 0,18, crocette vere almeno 0,13; barre "/" e linee che annullano la riga quasi
# tutte sotto 0,05.
FIRMA_TRATTO = 0.12         # sotto: un tratto solo, non una firma (da verificare)
FIRMA_VERA = 0.17           # da qui: forma di firma...
FIRMA_SPESSORE = 0.11       # ... se anche fitta (spessore medio): le linee ondulate non lo sono (firme vere >= 0,117 nel 99%)
CROCETTA_TRATTO = 0.08      # sotto: una barra, non una crocetta (da verificare)


def _faint_mark(st: str, marks: PageMarks, g: int, f: str) -> str:
    """Crocetta o firma fatta di inchiostro troppo chiaro: vuota se e' solo un'ombra,
    da verificare se e' un segno chiaro. Crocetta o firma che e' un tratto solo (la
    barra di una riga annullata): da verificare."""
    if st not in (CROCETTA, FIRMA, SEGNO):
        return st
    dark = marks.darkness(g, f)
    if dark is not None and dark < SEGNO_MIN_SCURO:
        return VUOTA
    if dark is not None and dark < SEGNO_SCURO_SICURO:
        return SEGNO
    if st in (CROCETTA, FIRMA):
        spread = marks.spread(g, f)
        if spread is not None and spread < (FIRMA_TRATTO if st == FIRMA else CROCETTA_TRATTO):
            return SEGNO
    return st


class PageMarks:
    """Segni a mano che non sono valori (barrature, linee, barrette "/"), dall'unione
    di due rilevatori indipendenti: ``strokes.find_strikes`` (tratti lunghi e sottili,
    cancellati dall'immagine prima dell'analisi delle celle) e ``ink.PageInk``
    (inchiostro "proprio" di ogni cella, righe barrate per intero, linee orizzontali,
    barrette). Unire i due serve alla sicurezza: un segno visto da uno solo dei due
    basta a *segnalare* la cella; solo i segni visti dal primo (o da entrambi)
    possono togliere un valore senza segnalarlo, come prima."""

    def __init__(self, gray: np.ndarray, img: np.ndarray, grid: TableGrid):
        self.grid = grid
        self.img = img
        self.gray = gray
        # pieghe del foglio (lunghe linee quasi verticali che non sono la griglia): rumore
        # per crocette, firme e testo di ogni colonna, tolte come le barrature
        self.creases: list[Any] = []
        crease_mask: np.ndarray | None = None
        try:
            self.creases, crease_mask = ink.find_creases(img, grid)
        except Exception:  # noqa: BLE001 - analisi aggiuntiva
            log.warning("Ricerca delle pieghe del foglio non riuscita", exc_info=True)
        self.crease_mask = crease_mask
        self._dark: np.ndarray | None = None
        self.strikes = find_strikes(gray, grid, ignore=crease_mask)
        if crease_mask is not None and crease_mask.shape[:2] == gray.shape[:2]:
            self.strikes.mask = crease_mask if self.strikes.mask is None else (self.strikes.mask | crease_mask)
        self.page: Any = None
        try:
            self.page = ink.PageInk(img, grid, creases=(self.creases, crease_mask))
        except Exception:  # noqa: BLE001 - analisi aggiuntiva: senza, restano i segni del primo rilevatore
            log.warning("Analisi dell'inchiostro delle celle non riuscita", exc_info=True)
        self._own: dict[tuple[int, str], Any] = {}
        rows: dict[int, int] = {}
        for (g, _c), (_len, horizontal) in self.strikes.cells.items():
            if not horizontal:
                rows[g] = rows.get(g, 0) + 1
        # riga barrata: un tratto diagonale attraversa almeno due celle della riga
        self.struck_rows = {g for g, n in rows.items() if n >= 2}
        if self.page is not None:
            try:
                self.struck_rows |= set(self.page.struck_days)
            except Exception:  # noqa: BLE001
                log.debug("Righe barrate non disponibili", exc_info=True)

    @property
    def n_strokes(self) -> int:
        n = len(self.strikes.strokes)
        if self.page is not None and not n:
            n = len(getattr(self.page, "strokes", []) or [])
        return n

    def own(self, g: int, f: str) -> Any:
        """``ink.CellInkResult`` della cella (None se l'analisi non e' disponibile)."""
        if self.page is None:
            return None
        key = (g, f)
        if key not in self._own:
            try:
                self._own[key] = self.page.own_ink(g, f)
            except Exception:  # noqa: BLE001
                log.debug("Inchiostro proprio della cella %s non disponibile", key, exc_info=True)
                self._own[key] = None
        return self._own[key]

    def struck(self, g: int, f: str) -> bool:
        """Cella attraversata da una barratura (o in una riga barrata)."""
        if g in self.struck_rows or self.strikes.struck(g, self.grid.col_index(f)):
            return True
        r = self.own(g, f)
        return bool(r is not None and (r.struck or r.row_struck))

    def crease(self, g: int, f: str, on_ink: bool = False) -> bool:
        """Cella attraversata da una piega del foglio (tolta dall'immagine prima
        dell'analisi). Con ``on_ink`` solo se la piega passa sulla scrittura rimasta
        (entro 0,15 righe dai suoi bordi): una piega nella parte vuota della cella
        non ha tagliato nulla."""
        box = self.grid.cell(g, f)
        hits = [c for c in self.creases if c.crosses(box)]
        if not hits or not on_ink:
            return bool(hits)
        r = self.own(g, f)
        mask = None if r is None else np.asarray(r.mask, dtype=bool)
        if mask is None or not mask.any():
            return True
        xs = np.nonzero(mask.any(axis=0))[0]
        reach = 0.15 * self.grid.row_height
        lo, hi = box[0] + xs.min() - reach, box[0] + xs.max() + reach
        yc = (box[1] + box[3]) / 2.0
        return any(lo - c.half_width <= c.x_at(yc) <= hi + c.half_width for c in hits)

    def hidden_by_crease(self, g: int, f: str) -> bool:
        """Scrittura sotto la piega (piu' scura della piega), tolta con essa."""
        if not self.creases:
            return False
        try:
            if self._dark is None:      # grigio "canale piu' scuro" calcolato una volta per pagina
                self._dark = ink._darkness_gray(self.img)
            return ink.ink_under_crease(self._dark, self.grid.cell(g, f), self.creases, self.grid.row_height,
                                        self.crease_mask)
        except Exception:  # noqa: BLE001
            log.debug("Controllo della scrittura sotto la piega non riuscito", exc_info=True)
            return True

    def line_across(self, g: int, f: str) -> bool:
        if self.strikes.horizontal(g, self.grid.col_index(f)):
            return True
        r = self.own(g, f)
        return bool(r is not None and r.line_across)

    def line_only(self, g: int, f: str) -> bool:
        """Nella cella c'e' solo una linea tirata a mano (secondo ``PageInk``)."""
        r = self.own(g, f)
        return bool(r is not None and r.line_across and r.empty)

    def own_crop(self, g: int, f: str) -> np.ndarray | None:
        """Ritaglio (inchiostro scuro su bianco, con margine) dell'inchiostro proprio della
        cella secondo ``PageInk``; None se la cella e' vuota o contiene solo segni
        (linea attraverso, barratura, "/")."""
        r = self.own(g, f)
        if r is None or r.empty or r.line_across or r.struck or r.slash:
            return None
        mask = np.asarray(r.mask, dtype=bool)
        if mask.sum() < max(12.0, (0.06 * self.grid.row_height) ** 2):
            return None
        ys, xs = np.nonzero(mask)
        crop = np.where(mask, np.asarray(r.crop), 255).astype(np.uint8)[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        unit = self.grid.row_height
        px, py = max(2, int(round(0.22 * unit))), max(2, int(round(0.15 * unit)))
        return cv2.copyMakeBorder(crop, py, py, px, px, cv2.BORDER_CONSTANT, value=255)

    def slash(self, g: int, f: str) -> bool:
        """Barretta "/" isolata secondo ``PageInk`` (il primo rilevatore la cerca sul ritaglio)."""
        r = self.own(g, f)
        return bool(r is not None and r.slash)

    def empty(self, g: int, f: str) -> bool:
        """``PageInk`` non trova inchiostro proprio nella cella (tolte linee, barrature e
        tratti di celle vicine)."""
        r = self.own(g, f)
        return bool(r is not None and r.empty)

    def shape(self, g: int, f: str) -> tuple[float, float] | None:
        """Forma dell'inchiostro proprio della cella, in altezze di riga: (scarto da una
        linea al 90 percentile, spessore medio lungo la linea). Un tratto solo ("/", linea)
        ha lo scarto basso, una linea ondulata lo spessore basso, una firma nessuno dei
        due; None se non misurabile."""
        r = self.own(g, f)
        if r is None or r.empty:
            return None
        ys, xs = np.nonzero(np.asarray(r.mask, dtype=bool))
        if xs.size < 10:
            return None
        _length, thick, dev, _angle = stroke_shape(xs.astype(np.float64), ys.astype(np.float64))
        rh = max(1.0, self.grid.row_height)
        return dev / rh, thick / rh

    def spread(self, g: int, f: str) -> float | None:
        """Scarto dell'inchiostro proprio della cella da una linea (vedi ``shape``)."""
        sh = self.shape(g, f)
        return None if sh is None else sh[0]

    def darkness(self, g: int, f: str) -> float | None:
        """Quanto e' scuro l'inchiostro proprio della cella rispetto alla carta vicina
        (90 percentile, livelli di grigio); None se non misurabile."""
        r = self.own(g, f)
        if r is None or r.empty:
            return None
        m = np.asarray(r.mask, dtype=bool)
        x0, y0 = int(r.box[0]), int(r.box[1])
        sub = self.gray[y0:y0 + m.shape[0], x0:x0 + m.shape[1]].astype(np.float32)
        if sub.shape != m.shape or not m.any():
            return None
        paper = float(np.percentile(sub, 90))
        return float(np.percentile(paper - sub[m], 90))


def plan_page(img: np.ndarray, grid: TableGrid) -> PagePlan:
    """Analizza l'inchiostro della pagina e prepara i ritagli da leggere."""
    gray0 = to_gray(img)
    H, W = gray0.shape[:2]
    rh = grid.row_height
    plan = PagePlan()
    t_lex, h_lex, n_lex = time_lexicon(), hours_lexicon(), notes_lexicon()
    # barrature a mano (righe annullate, linee sugli orari effettivi): si cancellano
    # prima di ogni analisi delle celle e si ricorda quali celle attraversano
    marks = PageMarks(gray0, img, grid)
    gray = erase(gray0, marks.strikes.mask)
    if marks.n_strokes:
        plan.notes.append(f"Linee di barratura riconosciute e ignorate: {marks.n_strokes}.")
    if marks.creases:
        giorni = sorted({g for c in marks.creases for g in c.days})
        plan.notes.append(f"Piega del foglio (linea verticale lunga) riconosciuta e ignorata nei giorni "
                          f"{giorni[0]}-{giorni[-1]}: la scrittura che attraversa e' da verificare.")
    if marks.struck_rows:
        plan.notes.append("Righe barrate (non usate) nei giorni "
                          + ", ".join(str(g) for g in sorted(marks.struck_rows))
                          + ": l'eventuale scrittura rimasta e' da verificare.")
    notes_gray: np.ndarray | None = None
    stamps = stamp_boxes(gray, grid)
    # timbri e firme del pie' di pagina che sconfinano nelle ultime righe: il loro
    # inchiostro non e' un segno della riga
    foreign: ForeignInk | None = None
    try:
        foreign = foreign_ink(gray, grid)
    except Exception:  # noqa: BLE001 - analisi aggiuntiva
        log.warning("Analisi dei timbri e delle firme del pie' di pagina non riuscita", exc_info=True)
    foreign_gray: np.ndarray | None = None
    for g in range(1, 32):
        states: dict[str, str] = {}
        flags: dict[str, str] = {}
        slashes: list[str] = []
        y0, y1 = grid.row_y[g - 1], grid.row_y[g]
        for f in TEXT_FIELDS:
            box = grid.cell(g, f)
            c = grid.col_index(f)
            st = _cell_state_text(gray, box, allow_dash=True)
            crop = None
            if st == TESTO and f in ("eff_entrata", "eff_uscita") and marks.line_only(g, f):
                st = VUOTA          # solo la linea "non svolta" (vedi sotto)
            if st == TESTO:
                crop = clean_crop(gray, box, rh, rows=(y0, y1), cols=(grid.col_x[c], grid.col_x[c + 1]))
                if crop is None:
                    # il ritaglio pulito non trova scrittura propria, l'analisi dell'inchiostro
                    # della cella si': si legge il suo ritaglio e il valore resta da verificare
                    crop = marks.own_crop(g, f)
                    if crop is not None:
                        flags[f] = "ritaglio"
                if crop is None:
                    st = VUOTA
                elif is_slash(crop, rh):
                    slashes.append(f)
                elif marks.slash(g, f):
                    # barretta vista solo dall'analisi dell'inchiostro: si legge e si segnala
                    flags[f] = "barretta"
            if st != TESTO and marks.crease(g, f) and marks.hidden_by_crease(g, f):
                # scrittura finita sotto la piega e sparita togliendola: si legge il ritaglio
                # con la piega e il valore resta da verificare
                crop = clean_crop(gray0, box, rh, rows=(y0, y1), cols=(grid.col_x[c], grid.col_x[c + 1]))
                if crop is not None:
                    st = TESTO
                    flags[f] = "piega"
            if st == TESTO and marks.struck(g, f):
                flags[f] = "barrato"
            elif st == TESTO and marks.crease(g, f, on_ink=True):
                flags.setdefault(f, "piega")      # scrittura tagliata dalla piega tolta
            states[f] = st
            if st != TESTO or crop is None:
                continue
            key = f"rows.{g}.{f}"
            lines = split_text_lines(crop, rh, min_width=ORARI_DUE_RIGHE_LARGHEZZA)
            if len(lines) > 1:
                # due fasce orarie nella stessa cella ("9:00" sopra "13:00"): l'entrata e'
                # la prima, l'uscita l'ultima; le ore restano quelle scritte. Sempre da
                # verificare (la riga non torna con le ore: c'e' una pausa in mezzo)
                flags.setdefault(f, "due_orari")
                if f in TIME_FIELDS:
                    crop = lines[0] if f.endswith("entrata") else lines[-1]
            plan.aspect[key] = crop.shape[1] / max(1.0, float(crop.shape[0]))
            if f == "ore_dichiarate":
                plan.requests.append(_text_request(key, crop, h_lex, beams=5, max_tokens=6, vpad=VPAD_NUMERI))
            else:
                plan.requests.append(_text_request(key, crop, t_lex, beams=6, max_tokens=8, vpad=VPAD_NUMERI))
        if states.get("ore_dichiarate") == TESTO and any(flags.get(f) == "due_orari" for f in TIME_FIELDS):
            # con due fasce orarie le ore non si possono verificare con entrata e uscita
            flags.setdefault("ore_dichiarate", "due_orari")
        # "/" isolata in una riga senza altri orari (riga non lavorata): nessun valore;
        # altrimenti (o nella colonna delle ore, dove un "1" puo' somigliarle) si legge e si segnala
        others = [f for f in TIME_FIELDS if states[f] == TESTO and f not in slashes]
        for f in slashes:
            if f in TIME_FIELDS and not others:
                states[f] = NULLO
                plan.requests = [r for r in plan.requests if r.key != f"rows.{g}.{f}"]
                plan.aspect.pop(f"rows.{g}.{f}", None)
            else:
                flags[f] = "barretta"
        # linea orizzontale tirata sugli orari effettivi = prestazione non svolta
        if any(marks.line_across(g, f) for f in ("eff_entrata", "eff_uscita")):
            if all(states[f] in (VUOTA, NULLO, TRATTINO) for f in ("eff_entrata", "eff_uscita")):
                states["eff_entrata"] = states["eff_uscita"] = TRATTINO
        for f in ("assenza_alunno", "assenza_operatore"):
            box = grid.cell(g, f)
            st = _faint_mark(_cell_state_cross(gray, box), marks, g, f)
            if st == CROCETTA and (marks.struck(g, f) or marks.slash(g, f) or marks.empty(g, f)):
                # pezzo di barratura, "/" isolata o inchiostro che secondo l'analisi delle
                # celle appartiene a una cella vicina (es. l'inizio di una firma): forse
                # non e' una crocetta
                st = SEGNO
            elif st != SEGNO and marks.crease(g, f) and (st == CROCETTA or marks.hidden_by_crease(g, f)):
                st = SEGNO          # crocetta attraversata da una piega (tolta), o segno sotto la piega
            elif st == CROCETTA and foreign is not None and foreign.share(box) >= QUOTA_DUBBIA:
                st = SEGNO          # parte di un timbro o di una firma del pie' di pagina
            states[f] = st
        fbox = grid.cell(g, "firma")
        st = _faint_mark(_cell_state_signature(gray, fbox), marks, g, "firma")
        share = foreign.share(fbox) if foreign is not None and st != VUOTA else 0.0
        if st != VUOTA and _inside_stamp(gray, fbox, stamps):
            st = TIMBRO
        elif share >= QUOTA_ESTRANEA:
            # inchiostro per lo piu' di un timbro o di una firma del pie' di pagina: se
            # tolto resta una firma, e' da verificare; altrimenti non e' una firma della riga
            if foreign_gray is None:
                foreign_gray = foreign.erase(gray)
            st = SEGNO if _cell_state_signature(foreign_gray, fbox) == FIRMA else TIMBRO
        elif st == FIRMA and share >= QUOTA_DUBBIA:
            st = SEGNO
        elif st == FIRMA and marks.struck(g, "firma"):
            st = SEGNO
        elif st == VUOTA and marks.crease(g, "firma") and marks.hidden_by_crease(g, "firma"):
            st = SEGNO          # tratto sotto la piega: da verificare
        states["firma"] = st
        sh = marks.shape(g, "firma") if st == FIRMA else None
        if sh is not None and sh[0] >= FIRMA_VERA and sh[1] >= FIRMA_SPESSORE:
            plan.firme_vere.add(g)
        if st in (FIRMA, SEGNO) and states["assenza_operatore"] == CROCETTA:
            worked = any(states[f] == TESTO for f in ("eff_entrata", "eff_uscita"))
            if worked or _touches_signature(gray, grid.cell(g, "assenza_operatore"), fbox):
                # operatore assente ma riga firmata con l'orario effettivo (contraddizione),
                # oppure inizio della firma che sconfina: la "crocetta" va verificata
                states["assenza_operatore"] = SEGNO
        box = grid.cell(g, "note")
        st = _cell_state_text(gray, box, allow_dash=True)
        nshare = foreign.share(box) if foreign is not None and st == TESTO else 0.0
        source = gray
        if st == TESTO and _inside_stamp(gray, box, stamps):
            st = TIMBRO
        elif st == TESTO and nshare >= QUOTA_ESTRANEA:
            # per lo piu' timbro o firma del pie' di pagina: si guarda cio' che resta senza
            if foreign_gray is None:
                foreign_gray = foreign.erase(gray)
            c = grid.col_index("note")
            rest = None if ink.is_blank(foreign_gray, box) else clean_crop(
                foreign_gray, box, rh, rows=(y0, y1), cols=(grid.col_x[c], grid.col_x[c + 1]), extend=(0.3, 0.5))
            if rest is None or not (_substantial_note(rest, rh) or len(split_text_lines(rest, rh)) > 1):
                st = TIMBRO                 # solo frammenti del timbro o della firma
            else:
                source = foreign_gray       # scrittura vera rimasta: letta e segnalata
        elif st == TESTO and marks.empty(g, "note"):
            # solo code di firme, linee o tratti delle righe vicine; ma se cio' che e' stato
            # attribuito alle celle vicine e' una scrittura consistente (una parola staccata
            # di poco dalla firma, es. "FERIE" subito dopo la firma, presa per la sua
            # continuazione in corsivo) la nota si legge e si segnala, mai persa in silenzio
            c = grid.col_index("note")
            whole = clean_crop(gray, box, rh, rows=(y0, y1), cols=(grid.col_x[c], grid.col_x[c + 1]),
                               extend=(0.3, 0.5))
            if whole is not None and _substantial_note(whole, rh):
                flags["note"] = "attribuzione"
            else:
                st = VUOTA
        if st == TESTO and states["firma"] != VUOTA and source is gray:
            if notes_gray is None:
                notes_gray = gray.copy()
            if _remove_overflow(notes_gray, grid, g):
                source = notes_gray
                if ink.is_blank(source, box):
                    # tolta la firma che sconfina non resta nulla: ma se la nota attaccata
                    # alla firma e' una scrittura consistente si legge com'e' e si segnala
                    c = grid.col_index("note")
                    whole = clean_crop(gray, box, rh, rows=(y0, y1), cols=(grid.col_x[c], grid.col_x[c + 1]),
                                       extend=(0.3, 0.5))
                    if whole is not None and _substantial_note(whole, rh) and (
                            len(split_text_lines(whole, rh)) > 1 or _ink_width(whole) >= NOTA_OLTRE_FIRMA * rh):
                        source = gray
                        flags["note"] = "attribuzione"
                    else:
                        st = VUOTA
        if st == TESTO:
            c = grid.col_index("note")
            crop = clean_crop(source, box, rh, rows=(y0, y1), cols=(grid.col_x[c], grid.col_x[c + 1]),
                              extend=(0.3, 0.5))
            pad_x, pad_y = max(2, int(round(0.22 * rh))), max(2, int(round(0.15 * rh)))
            if crop is None or (crop.shape[1] - 2 * pad_x < NOTE_MIN_WIDTH * rh
                                and crop.shape[0] - 2 * pad_y < NOTE_MIN_HEIGHT * rh):
                st = VUOTA
            else:
                key = f"rows.{g}.note"
                plan.aspect[key] = crop.shape[1] / max(1.0, float(crop.shape[0]))
                lines = split_text_lines(crop, rh)
                if len(lines) > 1 or _substantial_note(crop, rh, NOTA_BREVE_LARGHEZZA):
                    # scrittura vera (due righe, o almeno tre segni larghi e alti come una
                    # parola breve, es. "104"): se nessuna lettura viene accettata la nota
                    # va segnalata, non lasciata vuota
                    plan.note_scritta.add(g)
                if len(lines) == 1:
                    plan.requests.append(_text_request(key, crop, n_lex, beams=4, free_beams=3,
                                                       charset=NOTE_CHARS, max_tokens=20, min_aspect=2.0))
                else:
                    # nota scritta piccola su due righe nella cella: TrOCR legge una riga alla volta
                    for i, line in enumerate(lines):
                        plan.requests.append(_text_request(f"{key}#{i + 1}", line, None, free_beams=3,
                                                           charset=NOTE_CHARS, max_tokens=16, min_aspect=2.0))
        if st == TESTO and marks.struck(g, "note"):
            flags["note"] = "barrato"
        elif st == TESTO and marks.crease(g, "note", on_ink=True):
            flags["note"] = "piega"
        elif st == TESTO and nshare >= QUOTA_DUBBIA:
            flags["note"] = "timbro"
        states["note"] = st
        plan.cells[g] = states
        if flags:
            plan.flags[g] = flags

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
        elif plan.cells.get(31, {}).get("ore_dichiarate") == TESTO:
            # casella del totale vuota e scrittura nelle ore del giorno 31: spesso e' il totale
            # mensile scritto una riga sopra (si decide in ``assemble`` dal valore letto)
            r31 = next((r for r in plan.requests if r.key == "rows.31.ore_dichiarate"), None)
            if r31 is not None:
                plan.requests.append(ReadRequest(TOTALE_RIGA_31, r31.image, total_lexicon(), beams=6, max_tokens=6))
        th = ty1 - ty0
        tw = grid.col_x[-1] - grid.col_x[0]
        # firma del coordinatore: a destra dell'etichetta, nella riga dei totali
        lab_box = _clip_box((grid.col_x[6], ty0 + 0.1 * th, grid.col_x[-1], ty1 - 0.1 * th), W, H)
        end = min(_label_end(gray, lab_box, rh), int(grid.col_x[0] + (COORD_LABEL_END + 0.08) * tw))
        end = max(end, int(grid.col_x[0] + (COORD_LABEL_END - 0.12) * tw))
        cbox = _clip_box((end + 0.15 * rh, ty0, grid.col_x[-1], ty1), W, H)
        own, foreign = _owned_components(gray, cbox, rh, th, rows=(ty0, ty1, ty1 + th), cols=(grid.col_x[-1],))
        plan.firma_coordinatore = _signature_like(own, rh)
        # tratti che vengono da fuori (la firma del referente che sale nella riga, il
        # timbro che la copre): non sono la firma del coordinatore ma la rendono dubbia
        plan.firma_coordinatore_incerta = _signature_like(foreign, rh) or any(
            _overlap(cbox, sb) > 0.15 for sb in stamps)
        if not plan.firma_coordinatore:
            # firma del coordinatore messa sopra la sua casella, nella riga del giorno 31:
            # presente, da verificare (puo' essere anche la firma della riga 31)
            above = _clip_box((cbox[0], ty0 - 1.0 * th, cbox[2], ty0 + 0.2 * th), W, H)
            up_own, up_foreign = _owned_components(gray, above, rh, 0.6 * th, rows=(ty0 - th, ty0),
                                                   cols=(grid.col_x[-1],))
            if _signature_like(up_own + up_foreign, rh):
                plan.firma_coordinatore = True
                plan.firma_coordinatore_incerta = True
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

P_INCERTO = 0.7          # probabilita' minima del valore scelto per non segnalarlo come incerto
P_LEGGIBILE = 0.05       # copertura minima: sotto, la scrittura non somiglia a nessun valore ammesso
SUPPORTO_CONTESTO = 0.5  # coerenza minima perche' un valore poco leggibile sia dedotto dalla riga (resta segnalato)
COPERTURA_DEBOLE = 0.3   # sotto questa copertura la lettura conta poco e si considerano i valori del contesto
P_CONTESTO = 0.02        # probabilita' iniziale di un valore suggerito dal contesto
TAU_MIN = 0.25           # attenuazione massima delle preferenze di una lettura debole
TOP_K = 4                # alternative per cella considerate nella riconciliazione
W_COLONNA = 0.5          # preferenza per i valori ricorrenti nella stessa colonna (solo fra letture incerte)
W_COLONNA_PROG = 1.0     # ... piu' forte per l'orario programmato (settimanale, si ripete nel mese)
QUOTA_COLONNA = 0.6      # quota oltre la quale un valore "domina" la colonna
SIMILE_NOTA = 0.8        # somiglianza per agganciare una nota a una formula ricorrente
SIMILE_NOTA_VICINA = 0.45   # ... o alla nota (sicura) del giorno precedente/successivo
P_CONFERMA = 0.8         # probabilita' (fra i valori ammessi) minima perche' la coerenza confermi una lettura
P_ALTERNATIVA = 0.02     # probabilita' oltre la quale un'alternativa letta conta come possibile scambio
P_RIVALE = 0.25          # probabilita' dell'alternativa che rende dubbio un valore
P_CHIARA = 0.8           # lettura "chiara": la riconciliazione che la scarta rende dubbia tutta la riga
COPERTURA_CONTRARIA = 0.1   # ... come quella che sostituisce una lettura leggibile con un valore che non suggeriva


# Orario scolastico plausibile: entrate e uscite fuori da questa fascia sono rare
# (una cifra letta male, es. "12,30" letto "19,30").
ORARIO_PIENO = (7 * 60 + 30, 17 * 60)     # nessuna penalita'
ORARIO_LIMITE = (7 * 60, 18 * 60)         # oltre: valore sempre segnalato
W_ORARIO_BORDO = -1.0                     # fra la fascia piena e il limite
W_ORARIO_FUORI = -2.5                     # fuori dal limite
ORE_PLAUSIBILI = 6.0                      # ore giornaliere oltre le quali il valore e' segnalato


def _time_minutes(v: str | None) -> int | None:
    p = parse_time(v) if v else None
    return None if p is None else p[0] * 60 + p[1]


def _time_plausible(m: int | None) -> bool:
    return m is not None and ORARIO_LIMITE[0] <= m <= ORARIO_LIMITE[1]


# Distribuzione dei minuti della pagina (00/30/15-45/altri): pseudo-conteggi della
# preferenza generica, aggiornati con i minuti letti con sicurezza sulla pagina
# (chi scrive solo orari tondi rende piu' sospetto un "8:30"; chi usa le mezz'ore no;
# chi lavora "8:10 - 13:10" rende normali i minuti 10).
MINUTI_PSEUDO = {0: 3.0, 30: 2.0, 15: 0.5, 45: 0.5}
MINUTI_PSEUDO_ALTRI = 0.02     # minuti non a quarti d'ora (05, 10, 20...)
MINUTI_MIN_LOG = -2.5
MINUTI_MIN_LOG_ALTRI = -3.5    # minuti non a quarti mai letti con sicurezza sulla pagina
MINUTI_IN_USO = 3              # letture sicure con gli stessi minuti per considerarli "in uso"


def page_minute_prior(minutes: Iterable[int]) -> Callable[[str], float]:
    """Preferenza (log, 0 per i minuti piu' probabili) per i minuti degli orari,
    adattata ai minuti letti con sicurezza sulla pagina."""
    counts = {m: MINUTI_PSEUDO.get(m, MINUTI_PSEUDO_ALTRI) for m in range(0, 60, TIME_STEP)}
    seen: dict[int, int] = {}
    for m in minutes:
        if m in counts:
            seen[m] = seen.get(m, 0) + 1
    for m, n in seen.items():
        # i minuti non a quarti contano solo se ricorrono (una lettura isolata di "8:10"
        # e' piu' spesso un 8:00 letto male che un orario davvero sfalsato)
        if m in MINUTI_PSEUDO or n >= MINUTI_IN_USO:
            counts[m] += float(n)
    # l'ora piena non e' mai penalizzata (e' la scrittura piu' comune in generale): sulle
    # pagine con molte mezz'ore le mezz'ore smettono di esserlo, su quelle senza lo diventano di piu'
    table = {m: 0.0 if m == 0 else max(MINUTI_MIN_LOG if (m in MINUTI_PSEUDO or counts[m] >= MINUTI_IN_USO)
                                       else MINUTI_MIN_LOG_ALTRI, min(0.0, math.log(c / counts[0])))
             for m, c in counts.items()}

    def prior(value: str) -> float:
        m = _time_minutes(value)
        if m is None:
            return 0.0
        s = table.get(m % 60, MINUTI_MIN_LOG_ALTRI)
        if not ORARIO_PIENO[0] <= m <= ORARIO_PIENO[1]:
            s += W_ORARIO_BORDO if _time_plausible(m) else W_ORARIO_FUORI
        return s

    # minuti che sulla pagina sono una scrittura normale (quarti d'ora e minuti ricorrenti)
    prior.in_uso = frozenset(m for m, v in table.items() if v > MINUTI_MIN_LOG_ALTRI)  # type: ignore[attr-defined]
    return prior


_GENERIC_TIME_PRIOR = page_minute_prior(())


def _time_prior(value: str) -> float:
    """Preferenza generica (senza dati della pagina) per minuti e orario scolastico.

    Serve solo a scegliere fra letture quasi equivalenti: una lettura chiara di
    10:30 non diventa 10:00; 19:30 contro 12:30 invece pesa (fuori orario)."""
    return _GENERIC_TIME_PRIOR(value)


def _hours_prior(value: float) -> float:
    frac = abs(value - round(value))
    s = 0.0 if frac < 1e-9 else (-0.4 if abs(frac - 0.5) < 1e-9 else -0.9)
    if value <= 0 or value > ORE_PLAUSIBILI:
        s -= 1.0
    return s


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
W_SAME_PROG_EFF = 1.0    # effettivo uguale al programmato (frequente: decide solo fra letture quasi equivalenti)
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


def _choose(ev: Evidence, rival: float = 1.0) -> _Field:
    """Valore piu' probabile di un campo senza contesto; con ``rival`` < 1 il campo
    e' segnalato anche se la seconda lettura ha almeno quella probabilita'."""
    if not ev.options or ev.coverage < P_LEGGIBILE:
        return _Field(illeggibile=True, confidence=0.0)
    v = ev.top
    p = ev.p(v) * ev.coverage
    second = math.exp(ev.options[1][1]) if len(ev.options) > 1 else 0.0
    incerto = p < P_INCERTO or (ev.free_value is not None and ev.free_value != v) or second >= rival
    return _Field(value=v, confidence=p, incerto=incerto)


def _similarity(a: str, b: str) -> float:
    import difflib  # noqa: PLC0415

    ka, kb = _key(a), _key(b)
    if not ka or not kb:
        return 0.0
    return difflib.SequenceMatcher(None, ka, kb).ratio()


NOME_CONF_SICURA = 0.85      # nomi scritti a mano: sicuri solo con lettura molto convinta
_RE_RIPETIZIONE = re.compile(r"(.{1,3}?)\1{3,}")


def _nome_degenere(text: str) -> bool:
    """Lettura "impazzita" del riconoscitore: un frammento ripetuto in loop
    ("R.R.R.R.R", "ANANANAN") o un nome fatto quasi solo di lettere isolate.
    Non e' un nome: va trattato come illeggibile, non mostrato come sicuro."""
    compatto = text.replace(" ", "")
    if _RE_RIPETIZIONE.search(compatto):
        return True
    parole = text.split()
    singole = sum(1 for t in parole if len(re.sub(r"[^A-ZÀÈÉÌÒÙ0-9]", "", t)) <= 1)
    return len(parole) >= 4 and singole / len(parole) > 0.5


def _text_field(reading: Reading | None, min_conf: float = NOME_CONF_SICURA, floor: float = 0.12) -> _Field:
    """Campo di testo libero (nomi): lettura libera ripulita. I nomi propri scritti
    a mano sono letti spesso in modo approssimativo: restano da verificare salvo
    lettura molto convinta (l'anagrafica li conferma dai mesi successivi)."""
    if reading is None:
        return _Field(illeggibile=True)
    text = _clean_name(reading.free_text)
    conf = reading.free_confidence
    if not text or not re.search(r"[A-Z0-9ÀÈÉÌÒÙ]", text) or conf < floor or _nome_degenere(text):
        return _Field(illeggibile=True, confidence=conf)
    return _Field(value=text, confidence=conf, incerto=conf < min_conf)


NOTA_CONF = 0.6          # fiducia minima (per token) di una nota riportata come sicura
NOTA_SIMILE_FRASE = 0.85    # somiglianza della lettura libera con la frase del lessico per riportarla sicura
NOTA_SIMILE_LESSICO = 0.7   # ... per preferire la frase del lessico alla lettura libera (da verificare)
NOTA_SIMILE_PAROLA = 0.8    # somiglianza per correggere una parola letta male
NOTA_MIN_CARATTERI = 4      # note piu' corte: solo se nel vocabolario (es. "104", "GLO")
_COME_CIFRE = str.maketrans("OQDILSB", "0001158")   # lettere che nelle note brevi sono spesso cifre
NOTA_CONF_BREVE = 0.2      # fiducia minima della nota breve del lessico proposta (da verificare)
NOTA_SIMILE_BREVE = 0.5    # ... e somiglianza minima con la lettura libera
NOTA_LARGA = 3.0            # ritaglio (larghezza/altezza) oltre il quale una nota corta illeggibile si segnala


def _note_words(text: str) -> tuple[str, int, int, int]:
    """Parole della nota confrontate con il vocabolario: (testo con le parole
    corrette, parole note, parole corrette, parole totali). Numeri e
    abbreviazioni puntate di parole note ("CARN.") contano come note."""
    out: list[str] = []
    known = fixed = total = 0
    for tok in text.split():
        core = re.sub(r"[^A-Z0-9ÀÈÉÌÒÙ]", "", tok)
        if not core:
            out.append(tok)
            continue
        total += 1
        plain = unicodedata.normalize("NFKD", core).encode("ascii", "ignore").decode()
        if plain.isdigit() or plain in NOTE_PAROLE:
            known += 1
            out.append(tok)
            continue
        if tok.endswith(".") and len(plain) >= 3 and any(w.startswith(plain) for w in NOTE_PAROLE):
            known += 1
            out.append(tok)
            continue
        if len(plain) >= 4:
            best = max(NOTE_PAROLE, key=lambda w: _similarity(plain, w))
            if _similarity(plain, best) >= NOTA_SIMILE_PAROLA and abs(len(best) - len(plain)) <= 2:
                fixed += 1
                out.append(tok.replace(core, best))
                continue
        out.append(tok)
    return " ".join(out), known, fixed, total


def _note_field(reading: Reading | None, aspect: float | None = None) -> _Field:
    """Nota: frase del lessico se la lettura libera la conferma, altrimenti il
    testo letto (parole lievemente corrette con il vocabolario).

    E' riportata come sicura solo se letta con buona fiducia e composta di parole
    note; le letture brevi fuori vocabolario ("DE", "49": code di firme, timbri,
    tratti) non diventano note: con un ritaglio largo la nota e' illeggibile,
    altrimenti si ignora il segno."""
    if reading is None:
        return _Field(illeggibile=True)
    text = _clean_name(reading.free_text)
    if text and reading.free_text.rstrip().endswith("."):
        text += "."                 # abbreviazione finale ("PONTE CARN.")
    conf = reading.free_confidence
    n_tok = max(1, reading.free_tokens)
    if reading.candidates:
        lex_v, lex_lp = reading.candidates[0]
        lconf = float(math.exp(min(0.0, lex_lp) / (n_tok + 1)))
        sim = _similarity(text, str(lex_v)) if text else 0.0
        if sim >= NOTA_SIMILE_LESSICO or (not text and lconf >= 0.3):
            sure = sim >= NOTA_SIMILE_FRASE and lconf >= NOTA_CONF and conf >= NOTA_CONF
            return _Field(value=str(lex_v), confidence=min(lconf, conf) if text else lconf, incerto=not sure)
    if not text:
        return _Field(illeggibile=True, confidence=conf)
    fixed_text, known, fixed, total = _note_words(text)
    exact = notes_lexicon().value(fixed_text)
    if exact is not None:
        return _Field(value=str(exact), confidence=conf, incerto=fixed > 0 or conf < NOTA_CONF)
    alnum = re.sub(r"[^A-Z0-9ÀÈÉÌÒÙ]", "", fixed_text)
    words = [re.sub(r"[^A-Z0-9]", "", w) for w in fixed_text.split()]
    # una nota di sole parole "vuote" ("IL", "DI LA"): una barra o una coda di firma letta
    # come parola, non una nota
    vuota = bool(words) and all(w in PAROLE_VUOTE for w in words)
    if len(alnum) < NOTA_MIN_CARATTERI and not (total and fixed == 0 and not vuota and all(
            w in NOTE_PAROLE for w in words)):
        # segno breve fuori vocabolario: non e' una nota leggibile; ma se la lettura vincolata
        # indica con una certa fiducia una nota breve del vocabolario ("104" letto "1oh") la
        # si propone, da verificare
        if reading.candidates:
            lex_v, lex_lp = reading.candidates[0]
            lconf = float(math.exp(min(0.0, lex_lp) / (n_tok + 1)))
            digits = re.sub(r"[^A-Z0-9]", "", text).translate(_COME_CIFRE)
            sim = max(_similarity(text, str(lex_v)), _similarity(digits, str(lex_v)))
            if lconf >= NOTA_CONF_BREVE and sim >= NOTA_SIMILE_BREVE:
                return _Field(value=str(lex_v), confidence=lconf, incerto=True)
        if aspect is not None and aspect >= NOTA_LARGA:
            return _Field(illeggibile=True, confidence=conf)
        return _Field()
    best, sim = None, 0.0
    n_key = len(_key(fixed_text))
    for canon, forms in NOTE_FREQUENTI:
        for form in forms:
            if abs(len(_key(form)) - n_key) > 0.2 * max(n_key, len(_key(form))):
                continue            # una frase piu' corta o piu' lunga non sostituisce la nota letta
            r = _similarity(fixed_text, form)
            if r > sim:
                best, sim = canon, r
    if best is not None and sim >= SIMILE_NOTA:
        return _Field(value=best, confidence=min(conf, sim), incerto=True)
    # i numeri (protocolli, date) si leggono male facilmente: sempre da verificare
    sure = (total > 0 and known == total and conf >= NOTA_CONF and not re.search(r"\d", fixed_text)
            and not vuota)
    return _Field(value=fixed_text, confidence=conf, incerto=not sure)


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
            if d is not None and abs(d / HOURS_STEP - round(d / HOURS_STEP)) < 1e-9 and 0 < d <= HOURS_MAX_DAY:
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


# Cifre scritte a mano che TrOCR confonde spesso: "1"/"4"/"7", "2"/"9", "0"/"6".
CONFUSIONI_CIFRE = {"1": "47", "4": "1", "7": "1", "2": "9", "9": "2", "0": "6", "6": "0"}
W_CONFUSIONE = -4.0      # log-probabilita' (relativa) di un orario ottenuto cambiando una cifra confondibile


def _with_confusions(options: list[tuple[Any, float]], prior: Callable[[str], float]) -> list[tuple[Any, float]]:
    """Aggiunge fra le alternative di un orario gli orari ottenuti cambiando una
    cifra confondibile delle due letture migliori (es. 19:30 -> 12:30). Partono
    molto sfavoriti: vincono solo se la plausibilita' (orario scolastico) e la
    coerenza della riga li sostengono, e in quel caso la cella e' segnalata
    (valore diverso dalla lettura migliore)."""
    if not options:
        return options
    have = {v for v, _ in options}
    extra: dict[Any, float] = {}
    for v, lp in options[:2]:
        if not isinstance(v, str):
            continue
        digits = v.replace(":", "")
        for i, d in enumerate(digits):
            for alt in CONFUSIONI_CIFRE.get(d, ""):
                w = digits[:i] + alt + digits[i + 1:]
                w = f"{w[:2]}:{w[2:]}"
                m = _time_minutes(w)
                if m is None or not TIME_MIN <= m <= TIME_MAX or m % TIME_STEP or w in have:
                    continue
                score = lp - prior(v) + prior(w) + W_CONFUSIONE
                extra[w] = max(extra.get(w, -math.inf), score)
    if not extra:
        return options
    scored = list(options) + list(extra.items())
    z = _logsumexp(lp for _v, lp in scored)
    return sorted(((v, lp - z) for v, lp in scored), key=lambda t: t[1], reverse=True)


def _column_bonus(values: list[Any], own: Any, weight: float = W_COLONNA) -> Callable[[Any], float]:
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
    return lambda v: weight * counts.get(v, 0) / n


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


_TWIN = {"prog_entrata": "eff_entrata", "eff_entrata": "prog_entrata",
         "prog_uscita": "eff_uscita", "eff_uscita": "prog_uscita"}
_HOURS_PAIRS = (("eff_entrata", "eff_uscita"), ("prog_entrata", "prog_uscita"))
CONF_CONFERMA = 0.85     # confidenza attribuita a un valore confermato dalla coerenza della riga


@dataclass
class _CellState:
    """Valore scelto per una cella di orari/ore e motivi di dubbio."""

    giorno: int
    field: str
    value: Any
    ev: Evidence
    reasons: set[str]
    assenza_alunno: bool = False


def _same_value(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-6
    return bool(a == b)


def _relations(f: str, values: dict[str, Any]) -> list[tuple[tuple[str, ...], Any, bool]]:
    """Relazioni esatte della riga che coinvolgono il campo ``f``.

    Per ogni relazione: (altre celle coinvolte, valore che imporrebbero a ``f``,
    True se e' la relazione orari effettivi/ore, la cui violazione e' un'incoerenza).
    Programmato e effettivo uguali e durata programmata = ore sono solo i casi
    piu' frequenti: se non valgono non e' un errore."""
    out: list[tuple[tuple[str, ...], Any, bool]] = []
    twin = _TWIN.get(f)
    if twin is not None and values.get(twin) is not None:
        out.append(((twin,), values[twin], False))
    ore = values.get("ore_dichiarate")
    for a, b in _HOURS_PAIRS:
        strict = a == "eff_entrata"
        if f == a and values.get(b) is not None and ore is not None:
            out.append(((b, "ore_dichiarate"), _shift(values[b], ore, -1), strict))
        elif f == b and values.get(a) is not None and ore is not None:
            out.append(((a, "ore_dichiarate"), _shift(values[a], ore, 1), strict))
        elif f == "ore_dichiarate" and values.get(a) is not None and values.get(b) is not None:
            out.append(((a, b), hours_between(values[a], values[b]), strict))
    return out


def _admissible(f: str, value: Any) -> bool:
    """True se ``value`` e' un valore ammesso dal modulo per il campo ``f``."""
    if f == "ore_dichiarate":
        return (isinstance(value, (int, float)) and 0 <= value <= HOURS_MAX_DAY
                and abs(value / HOURS_STEP - round(value / HOURS_STEP)) < 1e-9)
    m = _time_minutes(value) if isinstance(value, str) else None
    return m is not None and TIME_MIN <= m <= TIME_MAX and m % TIME_STEP == 0


def _doubt_reasons(f: str, v: Any, ev: Evidence, p: float, values: dict[str, Any],
                   column_values: list[Any], minuti: Iterable[int] | None = None) -> set[str]:
    """Motivi per cui il valore ``v`` scelto per il campo ``f`` va verificato.

    ``minuti``: minuti degli orari "normali" sulla pagina (default: i quarti d'ora)."""
    reasons: set[str] = set()
    if ev.coverage < P_LEGGIBILE:
        reasons.add("dedotto")            # la scrittura somiglia poco a qualsiasi valore ammesso
    if v != ev.top:
        reasons.add("alternativa")        # scelto dal contesto contro la lettura migliore
    if p < P_INCERTO:
        reasons.add("bassa_prob")
    rival = next((math.exp(lp) for u, lp in ev.options if not _same_value(u, v)), 0.0)
    if rival >= P_RIVALE:
        # un'altra lettura ammessa resta molto probabile: la coerenza della riga, da sola,
        # non basta a escluderla (es. "14:00" scritto come "16:00")
        reasons.add("rivale")
    dom = _dominant(column_values, ev.top)
    if dom is not None and dom != v and ev.p(dom) >= P_ALTERNATIVA:
        reasons.add("colonna")            # diverso dal valore abituale della colonna, che e' fra le letture
    if ev.free_value is not None and ev.free_value != v:
        # la lettura libera indica un altro valore: normale sulla pagina (es. 11:30 invece
        # di 11:00; 11:10 dove i minuti 10 ricorrono) oppure no (es. 11:10 dove nessuno
        # scrive i minuti 10: piu' probabilmente una cifra letta male)
        normale = _admissible(f, ev.free_value)
        if normale and f in TIME_FIELDS:
            m = _time_minutes(ev.free_value)
            normale = m is not None and m % 60 in (frozenset(minuti) if minuti is not None
                                                   else frozenset(MINUTI_PSEUDO))
        reasons.add("lettura_libera" if normale else "lettura_libera_anomala")
    twin = _TWIN.get(f)
    tv = values.get(twin) if twin else None
    if tv is not None and tv != v and ev.p(tv) >= P_ALTERNATIVA:
        # programmato e effettivo diversi, il valore dell'altro e' fra le letture di
        # questa cella e nessuna relazione *obbligatoria* giustifica la differenza: solo
        # uscita - entrata effettive = ore lo e' (un ritardo vero, 8:30 invece di 8:00 con
        # mezz'ora in meno); la durata programmata uguale alle ore e' solo frequente e non
        # distingue 9:30-13:30 da 9:00-13:00 (se la cella gemella e' stata a sua volta
        # dedotta, vedi la regola in ``assemble``)
        own = [imp for others, imp, strict in _relations(f, values) if others != (twin,) and strict]
        if not any(_same_value(imp, v) for imp in own):
            reasons.add("gemello")
    return reasons



# motivi di dubbio che la coerenza con celle sicure puo' superare (gli altri indicano
# che la lettura stessa propende per un valore diverso)
_SUPERABILI = frozenset({"dedotto", "bassa_prob", "colonna", "lettura_libera_anomala"})
SETTIMANA_MIN = 2        # celle sicure dello stesso giorno della settimana per confermare l'orario programmato


def _weekly_support(cs: _CellState, solid_values: dict[tuple[int, str], Any],
                    all_values: dict[tuple[int, str], Any] | None = None) -> bool:
    """L'orario programmato e' settimanale: stesso valore letto con sicurezza negli
    stessi giorni della settimana (giorno +/- 7, 14...) e nessun valore diverso
    (fra tutti quelli scelti, anche se dubbi: ``all_values``)."""
    if not cs.field.startswith("prog_"):
        return False

    def week(values: dict[tuple[int, str], Any]) -> list[Any]:
        return [val for (g, f), val in values.items()
                if f == cs.field and g != cs.giorno and (g - cs.giorno) % 7 == 0]

    same = week(solid_values)
    every = week(all_values) if all_values is not None else same
    return len(same) >= SETTIMANA_MIN and all(_same_value(x, cs.value) for x in same + every)


def _flag_weekly_outliers(states: dict[tuple[int, str], _CellState],
                          solid: set[tuple[int, str]] | None = None) -> None:
    """L'orario programmato e' settimanale: se negli stessi giorni della settimana
    (giorno +/- 7, 14...) le altre celle hanno tutte lo stesso valore e questa cella
    ne ha un altro, la cella va verificata (es. "8:00" letto "9:00" in un solo
    martedi'). Bastano ``SETTIMANA_MIN`` celle concordi, oppure una sola se letta
    con sicurezza (mesi con poche settimane utili: festivi, assenze, gite)."""
    # anche i valori segnalati delle altre settimane contano: servono solo a segnalare
    if solid is None:
        solid = {k for k, cs in states.items() if not cs.reasons}
    for (g, f), cs in states.items():
        if not f.startswith("prog_"):
            continue
        same = [k for k in states if k[1] == f and k[0] != g and (k[0] - g) % 7 == 0]
        if not same or not all(_same_value(states[k].value, states[same[0]].value) for k in same):
            continue
        if _same_value(states[same[0]].value, cs.value):
            continue
        if len(same) >= SETTIMANA_MIN or any(k in solid for k in same):
            cs.reasons.add("settimana")


ORE_ASSENZA_QUOTE = (0.25, 1 / 3, 0.5, 2 / 3, 0.75, 1.0)   # ore riconosciute / durata programmata (alunno assente)
ORE_ASSENZA_TOLLERANZA = 0.03
DURATA_SETTIMANA_DIVERSA = 1.0   # ore: durata programmata diversa da quella degli stessi giorni della settimana


def _prog_duration(values: dict[str, Any]) -> float | None:
    a, b = values.get("prog_entrata"), values.get("prog_uscita")
    if not isinstance(a, str) or not isinstance(b, str):
        return None
    return hours_between(a, b)


def _flag_schedule_plausibility(states: dict[tuple[int, str], _CellState],
                                solid: set[tuple[int, str]] | None = None) -> None:
    """Controlli di plausibilita' dell'orario programmato (solo segnalazioni, nessun
    valore cambiato):

    * con l'alunno assente le ore riconosciute sono di solito una frazione "tonda"
      della durata programmata (meta': 1 di 2 ore, 1,5 di 3, 2 di 4); un rapporto
      come 1,5 / 5 = 0,3 indica un orario programmato (o le ore) letto male;
    * durata programmata diversa di almeno ``DURATA_SETTIMANA_DIVERSA`` ore da
      quella (concorde) degli stessi giorni della settimana."""
    if solid is None:
        solid = {k for k, cs in states.items() if not cs.reasons}
    by_row: dict[int, dict[str, _CellState]] = {}
    for (g, f), cs in states.items():
        by_row.setdefault(g, {})[f] = cs
    values = {g: {f: cs.value for f, cs in cells.items()} for g, cells in by_row.items()}
    durations = {g: _prog_duration(v) for g, v in values.items()}
    for g, cells in by_row.items():
        d = durations[g]
        if d is None or d <= 0:
            continue
        prog = [cells[f] for f in ("prog_entrata", "prog_uscita")]
        ore = values[g].get("ore_dichiarate")
        absent = any(cs.assenza_alunno for cs in cells.values())
        if absent and isinstance(ore, (int, float)) and ore > 0:
            ratio = float(ore) / d
            if not any(abs(ratio - q) <= ORE_ASSENZA_TOLLERANZA for q in ORE_ASSENZA_QUOTE):
                for cs in prog + ([cells["ore_dichiarate"]] if "ore_dichiarate" in cells else []):
                    cs.reasons.add("assenza_ore")
        others = [(durations[g2], g2) for g2 in by_row if g2 != g and (g2 - g) % 7 == 0
                  and durations[g2] is not None]
        if any("settimana" in cs.reasons for cs in prog):
            continue                    # gia' segnalato dal confronto dei valori
        if others and all(abs(o - others[0][0]) < 1e-6 for o, _ in others) \
                and abs(d - others[0][0]) >= DURATA_SETTIMANA_DIVERSA - 1e-6 \
                and (len(others) >= SETTIMANA_MIN
                     or any((g2, "prog_entrata") in solid and (g2, "prog_uscita") in solid for _o, g2 in others)):
            # si segnala l'orario che differisce da quello prevalente nelle altre settimane;
            # se non si puo' dire quale, entrambi
            def odd_one(cs: _CellState) -> bool:
                vals = [by_row[g2][cs.field].value for _o, g2 in others]
                same = sum(1 for v in vals if _same_value(v, cs.value))
                return 2 * same <= len(vals)

            odd = [cs for cs in prog if odd_one(cs)]
            for cs in odd or prog:
                cs.reasons.add("settimana")


def _confirm_by_consensus(states: dict[tuple[int, str], _CellState]) -> set[tuple[int, str]]:
    """Celle incerte solo perche' poco leggibili il cui valore e' confermato.

    Una cella e' confermata se il valore scelto e' anche quello preferito dalla
    lettura (probabilita' >= ``P_CONFERMA`` fra i valori ammessi, lettura libera
    non contraria) e se almeno una relazione esatta con celle *sicure* della
    stessa riga lo impone (es. uscita effettiva = entrata effettiva + ore, oppure
    uguale all'uscita programmata), senza relazioni orari/ore violate; per
    l'orario programmato vale anche lo stesso valore sicuro negli stessi giorni
    della settimana. Le celle confermate diventano a loro volta sicure
    (ripetizione fino a stabilita').
    Il valore non viene mai cambiato: si toglie solo la segnalazione."""
    by_row: dict[int, dict[str, Any]] = {}
    for (g, f), cs in states.items():
        by_row.setdefault(g, {})[f] = cs.value
    solid = {k for k, cs in states.items() if not cs.reasons}
    confirmed: set[tuple[int, str]] = set()
    changed = True
    while changed:
        changed = False
        for key, cs in states.items():
            if key in solid or not cs.reasons <= _SUPERABILI:
                continue
            if not _same_value(cs.value, cs.ev.top) or cs.ev.p(cs.value) < P_CONFERMA:
                continue
            holds = violated = 0
            for others, implied, strict in _relations(cs.field, by_row[cs.giorno]):
                if not all((cs.giorno, o) in solid for o in others):
                    continue
                if _same_value(implied, cs.value):
                    holds += 1
                elif strict and not _partial_hours_ok(cs, by_row[cs.giorno]):
                    violated += 1
            if not holds and not violated:
                holds = int(_weekly_support(cs, {k: states[k].value for k in solid},
                                            {k: st.value for k, st in states.items()}))
            if holds and not violated:
                solid.add(key)
                confirmed.add(key)
                changed = True
    return confirmed


def _partial_hours_ok(cs: _CellState, values: dict[str, Any]) -> bool:
    """Con l'alunno assente le ore riconosciute possono essere meno dell'orario effettivo."""
    if not cs.assenza_alunno:
        return False
    d = hours_between(values.get("eff_entrata"), values.get("eff_uscita"))
    ore = values.get("ore_dichiarate")
    return d is not None and isinstance(ore, (int, float)) and ore < d


FORMA_CORTA_ASPETTO = 1.25  # ritaglio (larghezza/altezza) troppo largo per un orario di sola ora ("8", "13")


def _short_form(f: str, v: Any, reading: Reading | None, aspect: float | None) -> bool:
    """Orario letto come sola ora ("13") in un ritaglio largo quanto "13,30": le
    cifre dei minuti sono probabilmente sfuggite alla lettura."""
    if f not in TIME_FIELDS or reading is None or aspect is None:
        return False
    form = getattr(reading, "forms", {}).get(v)
    if not form:
        return False
    lex = time_lexicon()
    if len(lex.signature(form)) > 2 or aspect < FORMA_CORTA_ASPETTO:
        return False
    # anche la lettura libera ha visto solo l'ora (con piu' cifre decide ``_free_digits_conflict``)
    return len(lex.signature(_key(reading.free_text))) <= 2


def _value_digits(f: str, v: Any) -> set[str]:
    """Cifre con cui si scrive il valore (per confrontarle con la lettura libera)."""
    if f in TIME_FIELDS:
        m = _time_minutes(v) if isinstance(v, str) else None
        if m is None:
            return set()
        h, mm = divmod(m, 60)
        return {f"{h}{mm:02d}", f"{h:02d}{mm:02d}"}
    return set()


def _free_digits_conflict(f: str, v: Any, reading: Reading | None,
                          minuti: Iterable[int] | None = None) -> bool:
    """La lettura libera mostra almeno tre cifre incompatibili con il valore
    scelto (es. "10/3" -> 1 0 3 contro 10:00): le cifre lette indicano un altro
    orario, anche se la lettura libera nel suo insieme non e' interpretabile.

    ``minuti``: minuti "normali" sulla pagina (default: i quarti d'ora). Un "11.10"
    su una pagina dove nessuno scrive i minuti 10 e' un 11:00 letto male, non un
    altro orario; dove i minuti 10 ricorrono e' un orario diverso."""
    if reading is None or f not in TIME_FIELDS or not reading.free_text:
        return False
    digits = time_lexicon().signature(_key(reading.free_text))
    if len(digits) < 3:
        return False
    canon = _value_digits(f, v)
    if not canon or any(c.startswith(digits) or digits.startswith(c) for c in canon):
        return False
    # conta solo se le cifre lette corrispondono a un altro orario ammesso ("1660" non
    # e' un orario: e' solo "16:00" letto male e non indica un valore diverso)
    ammessi = frozenset(minuti) if minuti is not None else frozenset(MINUTI_PSEUDO)
    for minutes in range(TIME_MIN, TIME_MAX + 1, TIME_STEP):
        if minutes % 60 not in ammessi:
            continue
        other = f"{minutes // 60:02d}:{minutes % 60:02d}"
        if other != v and any(c.startswith(digits) or digits.startswith(c) for c in _value_digits(f, other)):
            return True
    return False


def _hour_conflict(f: str, v: Any, reading: Reading | None) -> bool:
    """L'ora letta (prime cifre della lettura libera) e' diversa da quella del valore
    scelto: es. "13 w." contro 11:00 dedotto dal resto della riga."""
    if reading is None or f not in TIME_FIELDS or not reading.free_text:
        return False
    m = _time_minutes(v) if isinstance(v, str) else None
    digits = time_lexicon().signature(_key(reading.free_text))
    if m is None or not digits:
        return False
    hour = None
    if len(digits) >= 2 and 6 <= int(digits[:2]) <= 20:
        hour = int(digits[:2])
    elif 6 <= int(digits[0]) <= 9:
        hour = int(digits[0])
    return hour is not None and hour != m // 60


def row_doubts(values: dict[str, Any], assenza_alunno: bool = False, assenza_operatore: bool = False,
               trattino: bool = False) -> dict[str, set[str]]:
    """Controlli di coerenza e plausibilita' della riga sui valori scelti.

    Non cambiano nessun valore: indicano quali celle verificare. Se le ore non
    tornano con gli orari effettivi almeno una delle tre letture e' sbagliata e
    non si puo' sapere quale: si segnalano tutte (e l'orario programmato uguale a
    quello effettivo coinvolto, letto probabilmente nello stesso modo)."""
    out: dict[str, set[str]] = {}

    def add(fields: Iterable[str], reason: str) -> None:
        for f in fields:
            if values.get(f) is not None:
                out.setdefault(f, set()).add(reason)

    mins = {f: _time_minutes(values.get(f)) if isinstance(values.get(f), str) else None for f in TIME_FIELDS}
    for f, m in mins.items():
        if m is not None and not _time_plausible(m):
            add([f], "fuori_orario")
    ore = values.get("ore_dichiarate")
    if isinstance(ore, (int, float)) and ore > ORE_PLAUSIBILI:
        add(["ore_dichiarate"], "fuori_orario")
    durations: dict[str, float | None] = {}
    for kind, (a, b) in (("eff", ("eff_entrata", "eff_uscita")), ("prog", ("prog_entrata", "prog_uscita"))):
        ma, mb = mins[a], mins[b]
        durations[kind] = None
        if ma is None or mb is None:
            continue
        if mb <= ma or mb - ma > ORE_PLAUSIBILI * 60:
            add([a, b], "incoerente")
            continue
        durations[kind] = (mb - ma) / 60.0
    d_eff, d_prog = durations["eff"], durations["prog"]
    if isinstance(ore, (int, float)):
        if d_eff is not None:
            if abs(d_eff - ore) > 0.01 and not (assenza_alunno and ore < d_eff):
                # non si sa quale lettura sia sbagliata: si verifica l'intera riga
                add([*TIME_FIELDS, "ore_dichiarate"], "incoerente")
        elif d_prog is not None and not trattino:
            if ore > d_prog + 0.01 or (ore < d_prog - 0.01 and not (assenza_alunno or assenza_operatore)):
                add(["prog_entrata", "prog_uscita", "ore_dichiarate"], "incoerente")
    # un solo orario effettivo scritto: l'altro manca o e' stato letto come vuoto
    if (mins["eff_entrata"] is None) != (mins["eff_uscita"] is None) and not trattino:
        add(["eff_entrata", "eff_uscita"], "incompleto")
    # lo stesso per l'orario programmato: senza l'altro orario nessuna relazione lo controlla
    if (mins["prog_entrata"] is None) != (mins["prog_uscita"] is None):
        add(["prog_entrata", "prog_uscita"], "incompleto")
    return out


def assemble(plan: PagePlan, readings: dict[str, Reading], grid: TableGrid,
             reasons_out: dict[tuple[int, str], set[str]] | None = None) -> ExtractionResult:
    """Costruisce il risultato finale dai riconoscimenti e dall'analisi d'inchiostro.

    ``reasons_out`` (facoltativo) riceve i motivi di dubbio di ogni cella di orari/ore
    segnalata: (giorno, campo) -> motivi (es. ``{"barrato"}``, ``{"bassa_prob", "colonna"}``)."""
    t_lex, h_lex = time_lexicon(), hours_lexicon()
    # 1) evidenze di tutte le celle con scrittura
    # minuti letti con sicurezza sulla pagina (preferenza per i minuti adattata alla grafia)
    sure_minutes: list[int] = []
    for g in range(1, 32):
        for f in TIME_FIELDS:
            if plan.cells.get(g, {}).get(f) == TESTO:
                ev = evidence(readings.get(f"rows.{g}.{f}"), t_lex, "time")
                m = _time_minutes(ev.top) if ev.options else None
                if m is not None and ev.coverage >= 0.5 and ev.p(ev.top) >= 0.9:
                    sure_minutes.append(m % 60)
    t_prior = page_minute_prior(sure_minutes)
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
                evs[g][f] = evidence(reading, t_lex, "time", t_prior)
    # valori letti con sicurezza in ogni colonna (per la preferenza di colonna)
    column: dict[str, list[Any]] = {f: [] for f in TEXT_FIELDS}
    for g, cells in evs.items():
        for f, ev in cells.items():
            if ev.options and ev.coverage >= 0.3 and math.exp(ev.options[0][1]) >= 0.5:
                column[f].append(ev.top)

    rows: list[DayRow] = []
    confs: list[float] = []
    dedotti: list[str] = []
    extra_notes: list[str] = []
    states: dict[tuple[int, str], _CellState] = {}
    for g in range(1, 32):
        st = plan.cells.get(g, {})
        row = DayRow(giorno=g)
        row.assenza_alunno = st.get("assenza_alunno") in (CROCETTA, SEGNO)
        row.assenza_operatore = st.get("assenza_operatore") in (CROCETTA, SEGNO)
        row.firma = st.get("firma") in (FIRMA, SEGNO)
        for f in ("assenza_alunno", "assenza_operatore", "firma"):
            if st.get(f) in (SEGNO, TIMBRO):
                # segno dubbio, oppure inchiostro di un timbro sulla cella della firma
                row.incerti.append(f)
                row.confidenza[f] = 0.5
        row.trattino_effettivo = TRATTINO in (st.get("eff_entrata"), st.get("eff_uscita"))
        cells = evs[g]
        options = {f: _with_bonus(ev.options, _column_bonus(
                       column[f], ev.top, W_COLONNA_PROG if f.startswith("prog_") else W_COLONNA))
                   for f, ev in cells.items() if ev.options}
        for f, ev in cells.items():
            if ev.coverage < COPERTURA_DEBOLE:
                options[f] = _weak_options(options.get(f, []), _context_values(f, cells, column[f]),
                                           ev.coverage)
        for f in list(options):
            if f in TIME_FIELDS:
                options[f] = _with_confusions(options[f], t_prior)
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
                p = min(p, 0.2)
            elif any(_same_value(imp, v) for _o, imp, _s in _relations(f, values)):
                # la coerenza con il resto della riga rafforza una lettura gia' preferita (solo
                # se una relazione esatta la impone: una durata soltanto plausibile non basta)
                p = 1.0 - (1.0 - p) * (1.0 - _context_confidence(support))
            setattr(row, f, v)
            row.confidenza[f] = round(p, 3)
            confs.append(p)
            reasons = _doubt_reasons(f, v, ev, p, values, column[f], getattr(t_prior, "in_uso", None))
            twin = _TWIN.get(f)
            if twin in cells and twin in values and cells[twin].options:
                # la cella gemella e' stata riportata a un valore diverso dalla sua lettura
                # migliore (es. orario effettivo poco leggibile allineato al programmato):
                # se quella lettura e' plausibile anche qui, l'uguaglianza non conferma nulla
                tt = cells[twin].top
                if (not _same_value(values[twin], tt) and not _same_value(tt, v)
                        and ev.p(tt) >= P_ALTERNATIVA):
                    reasons.add("gemello")
            key = f"rows.{g}.{f}"
            if _short_form(f, v, readings.get(key), plan.aspect.get(key)):
                reasons.add("forma_corta")
            if _free_digits_conflict(f, v, readings.get(key), getattr(t_prior, "in_uso", None)):
                reasons.add("cifre_diverse")
            states[(g, f)] = _CellState(g, f, v, ev, reasons, row.assenza_alunno)
        # coerenza e plausibilita' della riga, segni sul foglio (barrature, "/")
        extra = row_doubts(values, row.assenza_alunno, row.assenza_operatore, row.trattino_effettivo)
        # la stessa verifica sulle letture migliori: se sono incoerenti fra loro, una scelta
        # "coerente" fatta dalla riconciliazione non basta a renderle sicure
        # (solo se per renderle coerenti si e' scartata una lettura chiara: allora anche le
        # letture usate per correggerla possono essere quelle sbagliate)
        tops = {f: ev.top for f, ev in cells.items() if ev.options and f in values}
        forced = [f for f, v in values.items()
                  if not _same_value(v, cells[f].top) and (
                      cells[f].p(cells[f].top) * cells[f].coverage >= P_CHIARA
                      or (cells[f].coverage >= COPERTURA_CONTRARIA and cells[f].p(v) < P_ALTERNATIVA)
                      or _hour_conflict(f, v, readings.get(f"rows.{g}.{f}")))]
        if forced:
            for f2, why in row_doubts(tops, row.assenza_alunno, row.assenza_operatore,
                                      row.trattino_effettivo).items():
                if "incoerente" in why and f2 in values:
                    extra.setdefault(f2, set()).add("incoerente")
        for f, why in plan.flags.get(g, {}).items():
            extra.setdefault(f, set()).add(why)
        for f, why in extra.items():
            if (g, f) in states:
                states[(g, f)].reasons |= why
            elif f not in row.incerti and f in TEXT_FIELDS:
                row.incerti.append(f)
                if reasons_out is not None:
                    reasons_out[(g, f)] = set(why)
        rows.append(row)

    # conferma per coerenza: celle incerte solo per la scarsa leggibilita' il cui
    # valore e' quello preferito dalla lettura *e* coincide con quanto imposto
    # dalle celle sicure della stessa riga (es. uscita = entrata + ore)
    _flag_minute_lookalikes(plan, states)
    solid0 = {k for k, cs in states.items() if not cs.reasons}
    _flag_weekly_outliers(states, solid0)
    _flag_schedule_plausibility(states, solid0)
    confirmed = _confirm_by_consensus(states)
    for (g, f), cs in sorted(states.items()):
        row = rows[g - 1]
        if cs.reasons and (g, f) not in confirmed:
            log.debug("giorno %d %s=%s da verificare: %s", g, f, cs.value, ",".join(sorted(cs.reasons)))
            row.incerti.append(f)
            if reasons_out is not None:
                reasons_out[(g, f)] = set(cs.reasons)
            if "dedotto" in cs.reasons:
                dedotti.append(f"giorno {g} ({etichetta(f)})")
        elif (g, f) in confirmed:
            row.confidenza[f] = round(max(row.confidenza.get(f, 0.0), CONF_CONFERMA), 3)
    if confirmed:
        extra_notes.append(f"Valori poco leggibili confermati dalla coerenza della riga: {len(confirmed)}.")

    # note: formule ricorrenti, testo letto, nota uguale a quella del giorno vicino
    notes: dict[int, _Field] = {}
    for g in range(1, 32):
        if plan.cells.get(g, {}).get("note") == TESTO:
            fld = _note_field(_note_reading(readings, g), plan.aspect.get(f"rows.{g}.note"))
            if plan.flags.get(g, {}).get("note") and fld.value:
                fld.incerto = True          # scrittura attraversata da una barratura
            if not fld.value and not fld.illeggibile and (g in plan.note_scritta or plan.flags.get(g, {}).get("note")):
                # scrittura vera nella cella ma nessuna lettura accettata: mai una nota
                # vuota in silenzio
                fld.illeggibile = True
            if fld.value or fld.illeggibile:
                notes[g] = fld
    for g, fld in list(notes.items()):
        if fld.illeggibile or fld.incerto:
            reading = readings.get(f"rows.{g}.note")
            text = _clean_name(reading.free_text) if reading else ""
            for ng in (g - 1, g + 1):
                other = notes.get(ng)
                reliable = other is not None and bool(other.value) and not other.illeggibile and (
                    not other.incerto or other.value in _NOTE_FRASI)
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
    # orario fuori dall'intervallo di tutti gli altri orari sicuri del foglio: sospetto
    _flag_time_outliers(rows)
    for row in rows:
        st = plan.cells.get(row.giorno, {})
        # firma o crocetta in una riga senza orari, ore e note: segno isolato (timbro,
        # firma del pie' di pagina, piega, macchia): mai sicuro
        for f in _lonely_marks(row, st, row.giorno in plan.firme_vere, plan.flags.get(row.giorno)):
            if f not in row.incerti and f not in row.illeggibili:
                row.incerti.append(f)
                row.confidenza[f] = min(row.confidenza.get(f, 0.5), 0.5)
        if row.ore_dichiarate and "ore_dichiarate" not in row.incerti and not any(
                getattr(row, f) is not None and f not in row.incerti and f not in row.illeggibili
                for f in TIME_FIELDS):
            # ore (diverse da zero) senza alcun orario sicuro nella riga: nessun riscontro
            # possibile ("0" della riga annullata letto "3"; "1" letto "2" con orari tutti
            # dubbi che per caso tornano con il "2")
            row.incerti.append("ore_dichiarate")
    if _flag_difficult_page(rows):
        extra_notes.append("Scrittura difficile per il motore: quasi tutti gli orari e le ore della pagina "
                           "sono dubbi, quindi anche gli altri sono da verificare.")

    header = _assemble_header(plan, readings, rows, confs)
    flagged_rows = _flag_unreliable_grid(grid, rows, header)
    avvisi = []
    lead: list[str] = []        # avviso sulla griglia: prima di ogni altra nota
    if not grid.detected:
        avvisi.append(avviso_griglia_incerta())
        lead.append("Griglia incerta: tabella individuata solo con il modello proporzionale, "
                    "verificare tutti i valori della pagina.")
    elif not bool(getattr(grid, "reliable", True)):
        avvisi.append(avviso_griglia_incerta())
        lead.append("Griglia incerta: verificare tutti i valori della pagina.")
    elif flagged_rows:
        avvisi += [avviso_griglia_incerta(g) for g in flagged_rows]
        extra_notes.append("Righe della tabella non ben allineate alla griglia: valori da verificare nei giorni "
                           + ", ".join(str(g) for g in flagged_rows) + ".")

    n_incerti = sum(len(r.incerti) for r in rows) + len(header.incerti)
    n_illeggibili = sum(len(r.illeggibili) for r in rows) + len(header.illeggibili)
    detected = bool(grid.detected) or grid.score >= 0.5
    text_conf = float(np.mean(confs)) if confs else 1.0
    confidence = round(max(0.0, min(1.0, 0.85 * text_conf + 0.15 * float(grid.score))), 3)
    remarks = lead + list(plan.notes) + extra_notes
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
        avvisi=avvisi,
    )


def _lonely_marks(row: DayRow, states: dict[str, str], firma_vera: bool = False,
                  flags: dict[str, str] | None = None) -> list[str]:
    """Firme e crocette di una riga senza alcun valore sicuro (orari, ore, nota):
    un segno isolato e' sospetto.

    Vale anche per le righe con scrittura solo dubbia: le righe "annullate" con una
    barra in ogni cella (giorno non lavorato) hanno barre anche nella colonna della
    firma e delle assenze, che sembrano una firma o una crocetta; e una nota letta con
    incertezza spesso e' proprio un pezzo del timbro o della firma del pie' di pagina
    che ha prodotto anche il falso segno. In una riga con scrittura (dubbia) nelle celle
    di orari e ore, una firma che ha la forma di una firma (``firma_vera``, non un tratto
    solo) resta sicura: la scrittura difficile da leggere non la rende dubbia."""
    nota_sicura = bool(row.note) and "note" not in row.incerti and "note" not in row.illeggibili
    if nota_sicura:
        return []
    if any(getattr(row, f) is not None and f not in row.incerti and f not in row.illeggibili
           for f in TEXT_FIELDS):
        return []
    marks = [f for f in ("firma", "assenza_alunno", "assenza_operatore") if getattr(row, f)]
    # scrittura vera nelle celle di orari e ore: non barre ("/" di una riga annullata)
    flags = flags or {}
    scritta = any(states.get(f) == TESTO and flags.get(f) not in ("barretta", "barrato") for f in TEXT_FIELDS)
    if firma_vera and scritta:
        marks = [f for f in marks if f != "firma"]
    return marks


# Coerenza della grafia dei minuti: chi scrive "00" lo scrive sempre allo stesso modo.
# TrOCR legge a volte con sicurezza "13:30" uno "13:00" scritto con gli zeri a ricciolo
# ("13:ↄↄ") e la riga resta coerente (anche l'entrata diventa ":30"); la scrittura dei
# minuti di quella cella pero' somiglia a quella delle celle lette ":00".
GRAFIA_PARTE_MINUTI = 0.42      # parte destra della scrittura confrontata (frazione della larghezza)
GRAFIA_VICINI = 3               # celle piu' somiglianti (di altre righe) consultate
GRAFIA_MIN_CELLE = 8            # orari letti sulla pagina perche' il confronto abbia senso
GRAFIA_SIMILE = 0.5             # somiglianza minima (correlazione) per contare una cella come "uguale"


def _glyph(img: np.ndarray | None, right: float = 1.0) -> np.ndarray | None:
    """Impronta della scrittura di un ritaglio (con ``right`` < 1 solo la parte destra:
    i minuti di un orario)."""
    if img is None or img.size == 0:
        return None
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(gray < 160)
    if xs.size < 20:
        return None
    x1 = int(xs.max()) + 1
    w = x1 - int(xs.min())
    sub = gray[int(ys.min()):int(ys.max()) + 1, x1 - max(1, int(round(right * w))):x1]
    if sub.size == 0:
        return None
    v = cv2.resize(255 - sub, (32, 20), interpolation=cv2.INTER_AREA).astype(np.float32)
    v = cv2.GaussianBlur(v, (3, 3), 0).ravel()
    v -= v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else None


def _minute_glyph(img: np.ndarray | None) -> np.ndarray | None:
    return _glyph(img, GRAFIA_PARTE_MINUTI)


def _flag_lookalikes(cells: list[tuple[int, str, Any]], feats: list[np.ndarray]) -> list[int]:
    """Indici delle celle il cui valore e' diverso da quello su cui concorda la maggioranza
    delle celle (di altre righe) con la scrittura piu' simile."""
    if len(cells) < GRAFIA_MIN_CELLE:
        return []
    F = np.stack(feats)
    sim = F @ F.T
    out: list[int] = []
    for i, (g, _f, own) in enumerate(cells):
        near = [j for j in np.argsort(-sim[i]) if cells[j][0] != g][:GRAFIA_VICINI]
        if len(near) < GRAFIA_VICINI:
            continue
        vals = [cells[j][2] for j in near if sim[i][j] >= GRAFIA_SIMILE]
        if not vals:
            continue
        top = max(set(vals), key=vals.count)
        if top != own and vals.count(top) * 2 > GRAFIA_VICINI:
            out.append(i)
    return out


def _flag_minute_lookalikes(plan: PagePlan, states: dict[tuple[int, str], _CellState]) -> list[tuple[int, str]]:
    """Segnala (senza cambiarlo) l'orario i cui minuti sono diversi da quelli delle celle
    di altre righe con la scrittura piu' simile (la maggioranza delle piu' simili concorda
    su minuti diversi). Motivo non superabile dalla coerenza della riga: lo stesso errore
    di lettura sposta di solito insieme entrata e uscita.

    (Per le ore scritte lo stesso confronto non serve: cifre diverse si somigliano poco e
    i valori unici sulla pagina darebbero solo falsi allarmi.)"""
    images = {r.key: r.image for r in plan.requests}
    cells: list[tuple[int, str, Any]] = []
    feats: list[np.ndarray] = []
    for (g, f), cs in states.items():
        if f not in TIME_FIELDS:
            continue
        m = _time_minutes(cs.value)
        fv = _minute_glyph(images.get(f"rows.{g}.{f}")) if m is not None else None
        if fv is not None:
            cells.append((g, f, m % 60))
            feats.append(fv)
    out: list[tuple[int, str]] = []
    for i in _flag_lookalikes(cells, feats):
        g, f, _v = cells[i]
        states[(g, f)].reasons.add("grafia_minuti")
        out.append((g, f))
    return sorted(out)


# Pagina difficile: se quasi tutti gli orari e le ore letti sono dubbi, la scrittura e' fuori
# dalla portata del modello e la sua sicurezza sui pochi valori rimasti non e' affidabile
# (misurato: sulle pagine con oltre l'85% di valori dubbi i valori "sicuri" sbagliati sono
# molti di piu' che sulle altre). Si segnalano anche quelli.
PAGINA_DIFFICILE_QUOTA = 0.85
PAGINA_DIFFICILE_MIN_CELLE = 20


def _flag_difficult_page(rows: list[DayRow]) -> bool:
    """Segnala tutti gli orari e le ore di una pagina in cui quasi tutti sono gia' dubbi."""
    total = flagged = 0
    for row in rows:
        doubt = set(row.incerti) | set(row.illeggibili)
        for f in TEXT_FIELDS:
            if getattr(row, f) is not None or f in doubt:
                total += 1
                flagged += f in doubt
    if total < PAGINA_DIFFICILE_MIN_CELLE or flagged < PAGINA_DIFFICILE_QUOTA * total or flagged == total:
        return False
    for row in rows:
        for f in TEXT_FIELDS:
            if getattr(row, f) is not None and f not in row.incerti and f not in row.illeggibili:
                row.incerti.append(f)
    return True


FUORI_INTERVALLO_MIN = 90      # minuti oltre il minimo/massimo degli altri orari sicuri
FUORI_INTERVALLO_RIFERIMENTI = 6


def _flag_time_outliers(rows: list[DayRow]) -> list[tuple[int, str]]:
    """Segnala (senza cambiarlo) un orario letto con sicurezza che cade ben fuori
    dall'intervallo di tutti gli altri orari sicuri del foglio, di righe diverse:
    es. un'uscita "16:00" su un foglio in cui nessun altro orario supera le 14:00
    e' quasi sempre una cifra fraintesa ("14" letto "16"). Serve un numero minimo
    di riferimenti, per non segnalare i fogli con pochi orari."""
    sicuri: list[tuple[int, int]] = []
    for r in rows:
        for f in TIME_FIELDS:
            v = getattr(r, f)
            if v and f not in r.incerti and f not in r.illeggibili:
                t = parse_time(v)
                if t is not None:
                    sicuri.append((r.giorno, t[0] * 60 + t[1]))
    segnalati: list[tuple[int, str]] = []
    for r in rows:
        altri = [m for g, m in sicuri if g != r.giorno]
        if len(altri) < FUORI_INTERVALLO_RIFERIMENTI:
            continue
        lo, hi = min(altri), max(altri)
        for f in TIME_FIELDS:
            v = getattr(r, f)
            if not v or f in r.incerti or f in r.illeggibili:
                continue
            t = parse_time(v)
            if t is None:
                continue
            m = t[0] * 60 + t[1]
            if m > hi + FUORI_INTERVALLO_MIN or m < lo - FUORI_INTERVALLO_MIN:
                r.incerti.append(f)
                r.confidenza[f] = min(r.confidenza.get(f, 0.5), 0.5)
                segnalati.append((r.giorno, f))
    return segnalati


def _flag_unreliable_grid(grid: TableGrid, rows: list[DayRow], header: Header) -> list[int]:
    """Se l'autoverifica della griglia (``TableGrid.reliable`` / ``row_reliable``)
    dice che la tabella, o una sua riga, non e' allineata alle linee del modulo,
    ogni campo di quella zona e' segnalato, anche se vuoto: le celle potrebbero
    contenere la scrittura della riga vicina e quella della riga giusta potrebbe
    essere finita altrove. Restituisce i giorni non allineati (tutti se la pagina
    intera non e' affidabile)."""
    page_ok = bool(getattr(grid, "reliable", True))
    row_ok = getattr(grid, "row_reliable", None)
    out: list[int] = []
    for row in rows:
        ok = page_ok
        if ok and callable(row_ok):
            try:
                ok = bool(row_ok(row.giorno))
            except (ValueError, TypeError, IndexError):
                ok = True
        if ok:
            continue
        _flag_all(row, header=None)
        out.append(row.giorno)
    if not page_ok:
        _flag_all(None, header)
    return out


def _flag_all(row: DayRow | None, header: Header | None) -> None:
    """Segnala ogni campo (anche vuoto) della riga o dell'intestazione."""
    if row is not None:
        for f in DAY_FIELDS:
            if f not in row.incerti and f not in row.illeggibili:
                row.incerti.append(f)
    if header is not None:
        for f in HEADER_FIELDS:
            if f not in header.incerti and f not in header.illeggibili:
                header.incerti.append(f)


TOTALE_RIGA_31 = "footer.totale_riga31"


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
        # numeri brevi scritti a mano: "1" e "11", "8" e "3" si confondono facilmente;
        # nessun contesto li conferma, quindi basta un'alternativa plausibile per segnalarli
        fld = _choose(ev, rival=P_ALTERNATIVA)
        r = readings.get(f"header.{name}")
        if fld.value is not None and r is not None and re.search(r"\d\s+\d", r.free_text or ""):
            # cifre lette staccate ("1 2"): la lettura ha visto due segni separati, il
            # numero e' incerto (un "13" letto "1 2" diventava 12 sicuro)
            fld.incerto = True
        if name == "ore_pei" and fld.value is not None:
            fld.value = float(fld.value)
        _apply_header(header, name, fld)
        if fld.confidence is not None:
            confs.append(fld.confidence)
    if plan.header_ink.get("mese_anno"):
        ev = evidence(readings.get("header.mese_anno"), None, "month")
        worked = [g for g, st in plan.cells.items()
                  if any(st.get(f) == TESTO for f in TEXT_FIELDS) or st.get("firma") == FIRMA]
        r_month = readings.get("header.mese_anno")
        fld = _choose_month(ev, worked, free_text=r_month.free_text if r_month else "")
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
    if plan.firma_coordinatore_incerta:
        header.incerti.append("firma_coordinatore")
    header.timbro_referente = plan.timbro_referente
    if plan.totale_ink:
        ev = evidence(readings.get("footer.totale"), total_lexicon(), "hours", _hours_prior)
        somma = sum(r.ore_dichiarate or 0.0 for r in rows)
        # ore scritte ma illeggibili: il totale puo' superare la somma letta di quanto manca
        mancanti = sum(1 for r in rows if "ore_dichiarate" in r.illeggibili and r.ore_dichiarate is None)
        fld = _choose_total(ev, somma, (somma, somma + HOURS_MAX_DAY * mancanti) if mancanti else None)
        _apply_header(header, "totale_mensile_dichiarato", fld)
        if fld.confidence is not None:
            confs.append(fld.confidence)
    elif TOTALE_RIGA_31 in readings and header.totale_mensile_dichiarato is None:
        ev = evidence(readings.get(TOTALE_RIGA_31), total_lexicon(), "hours")
        top = ev.top if ev.options else None
        r31 = rows[30]
        if (isinstance(top, (int, float)) and top > HOURS_MAX_DAY and ev.coverage >= P_LEGGIBILE
                and r31.ore_dichiarate is None):
            # numero impossibile come ore di un giorno: e' il totale mensile scritto nella
            # riga del giorno 31; riportato come totale, da verificare
            header.totale_mensile_dichiarato = float(top)
            header.incerti.append("totale_mensile_dichiarato")
            r31.illeggibili = [f for f in r31.illeggibili if f != "ore_dichiarate"]
            r31.incerti = [f for f in r31.incerti if f != "ore_dichiarate"]
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


CAL_MIN_GIORNI = 8         # giorni compilati minimi perche' il calendario decida l'anno da solo
CAL_MARGINE = 3.0          # penalita' minima (domeniche/festivi: 1,5 l'uno) degli altri anni
MESE_SIMILE = 0.6          # somiglianza minima della lettura libera con il nome del mese


def _month_in_free_text(text: str, m: int) -> bool:
    """La lettura libera mostra il mese ``m``: il nome (anche storpiato: "Febbrano",
    "Fembraio") piu' simile a quello di ``m`` che a ogni altro mese, oppure il numero
    del mese all'inizio ("02/...", "2-...")."""
    t = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().upper()
    words = re.findall(r"[A-Z]{4,}", t)
    if words:
        w = max(words, key=len)
        sims = [_similarity(w, name) for name in MESI_NOMI]
        best = max(range(12), key=lambda i: sims[i])
        others = max(x for i, x in enumerate(sims) if i != best)
        return best + 1 == m and sims[best] >= MESE_SIMILE and sims[best] - others >= 0.15
    digits = re.match(r"\s*(\d)\s*(\d)?\s*[/.,\-|]", t)
    if not digits:
        return False
    num = int("".join(d for d in digits.groups() if d))
    return num == m


def _year_by_calendar(v: tuple[int, int], worked_days: Sequence[int], years: Iterable[int]) -> bool:
    """Il calendario dei giorni compilati indica l'anno senza ambiguita': con l'anno scelto
    nessun giorno compilato cade di domenica o in un festivo, con ogni altro anno vicino
    almeno due."""
    if len(worked_days) < CAL_MIN_GIORNI:
        return False
    m, y = v
    own = calendar_penalty(v, worked_days)
    if own < W_CAL_SABATO * 2:        # al piu' qualche sabato
        return False
    return all(calendar_penalty((m, yy), worked_days) <= own - CAL_MARGINE for yy in years if yy != y)


def _past_years(m: int, years: Iterable[int], today: date) -> list[int]:
    """Anni per cui il mese ``m`` non e' nel futuro (un mese futuro non si rendiconta)."""
    return [y for y in years if y * 12 + m <= today.year * 12 + today.month]


def _choose_month(ev: Evidence, worked_days: Sequence[int], today: date | None = None,
                  free_text: str = "") -> _Field:
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
    if incerto and _month_year_sure(ev, v, worked_days, _past_years(v[0], years, today), free_text):
        # mese letto chiaramente (lettura vincolata e lettura libera concordi) e anno
        # imposto dal calendario dei giorni compilati: l'anno scritto male ("'96", "202")
        # non lo rende dubbio
        return _Field(value=v, confidence=max(p, CONF_CALENDARIO), incerto=False)
    return _Field(value=v, confidence=p, incerto=incerto)


CONF_CALENDARIO = 0.8      # confidenza di mese/anno confermati da lettura del mese + calendario
P_MESE = 0.95              # quota minima del mese scelto fra le letture vincolate


def _month_year_sure(ev: Evidence, v: tuple[int, int], worked_days: Sequence[int], years: Iterable[int],
                     free_text: str) -> bool:
    m, y = v
    if y not in years:
        return False
    p_month = sum(math.exp(lp) for (mm, _yy), lp in ev.options if mm == m)
    if p_month < P_MESE or not _month_in_free_text(free_text, m):
        return False
    # un anno scritto per intero e diverso nella lettura libera: si lascia il dubbio
    for yy in re.findall(r"(?<!\d)(20\d\d)(?!\d)", free_text or ""):
        if int(yy) != y:
            return False
    return _year_by_calendar(v, worked_days, years)


def _choose_total(ev: Evidence, somma: float, possibile: tuple[float, float] | None = None) -> _Field:
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
        if abs(fld.value - somma) >= 0.01:
            # diverso dalla somma delle ore lette: o il totale o qualche ora giornaliera e'
            # letto male (o il foglio e' sbagliato): va guardato, salvo che la differenza
            # stia tutta nelle ore giornaliere gia' segnalate (``possibile``: intervallo
            # del totale compatibile con le ore sicure)
            if possibile is None or not possibile[0] - 0.01 <= fld.value <= possibile[1] + 0.01:
                fld.incerto = True
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

MIN_ROW_HEIGHT = 22.0     # altezza minima di una riga-giorno (pixel, circa 80 dpi)
LOW_RES_MSG = (
    "Risoluzione dell'immagine troppo bassa per leggere la scrittura a mano in modo affidabile: "
    "scansionare il foglio ad almeno 150 dpi oppure rifare la foto più da vicino."
)
TABLE_NOT_FOUND_MSG = (
    "Tabella giornaliera non riconosciuta: la pagina non sembra un foglio firma, oppure la foto è storta "
    "o tagliata. Se è un foglio firma, rifare la foto dall'alto inquadrando tutto il foglio, oppure usare "
    "lo scanner."
)


# Secondo parere (GLM-OCR) sulle celle dubbie: motivi di dubbio "strutturali" che il
# secondo parere non puo' mai togliere (la cella e' barrata, contiene una "/", l'orario
# e' fuori dalla fascia scolastica): rilette le stesse cifre, il dubbio resterebbe.
SP_BLOCCANTI = frozenset({"barrato", "barretta", "fuori_orario", "piega", "assenza_ore"})
SP_MARGINE = 0.2           # margine attorno alla cella grezza (altezze di riga), come nelle misure
_SP_FAILED: dict[str, str] = {}   # cache del modello -> motivo: non si riprova nella stessa sessione
_SP_LOCK = threading.Lock()


class LocalEngine:
    """Motore OCR locale: TrOCR + analisi dell'inchiostro, nessun dato inviato in rete.

    Con ``second_opinion=True`` (impostazione ``secondo_parere``) le celle di orari e ore
    rimaste dubbie vengono rilette da un secondo modello offline indipendente
    (``secondo_parere.SecondoParere``, GLM-OCR) con una politica prudente
    (``secondo_parere.decidi_riga``): un valore smette di essere segnalato solo se i due
    modelli concordano *e* una relazione esatta della riga lo conferma; se il secondo
    parere non e' disponibile (RAM, download, memoria) le celle restano semplicemente
    segnalate."""

    name = "locale"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu", cache_dir: Path | None = None,
                 recognizer: Recognizer | None = None, second_opinion: bool = False,
                 second_opinion_reader: Any = None):
        self.model_name = (model_name or DEFAULT_MODEL).strip()
        self.device = device or "cpu"
        self._cache_dir = cache_dir
        self._recognizer = recognizer
        self.second_opinion = bool(second_opinion or second_opinion_reader is not None)
        self._sp = second_opinion_reader

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
            rec = self._load(progress)
            with _MODELS_LOCK:
                # un solo modello in memoria alla volta
                _MODELS.clear()
                _MODELS[key] = rec
        return rec

    def _load(self, progress: ProgressFn) -> TrOCRRecognizer:
        """Scarica (se serve) e carica il modello; una copia locale danneggiata
        viene eliminata e scaricata di nuovo, una sola volta per sessione."""
        label = _model_label(self.model_name)
        path = local_model_path(self.model_name, self.cache_dir)
        if path is None:
            path = download_model(self.model_name, self.cache_dir, lambda f, m: progress(0.25 * f, m))
        progress(0.26, f"Caricamento del modello {label}…")
        try:
            return _load_recognizer(path, self.model_name, self.device)
        except ModelDamagedError as exc:
            path = self._replace_damaged(path, progress, exc)
        progress(0.26, f"Caricamento del modello {label}…")
        try:
            return _load_recognizer(path, self.model_name, self.device)
        except ModelDamagedError as exc:
            repo = _repo_of(path, self.cache_dir)
            raise EngineError(_damaged_message(repo or path)) from exc

    def _replace_damaged(self, path: Path, progress: ProgressFn, cause: BaseException) -> Path:
        repo = _repo_of(path, self.cache_dir)
        if repo is None:
            # cartella indicata dall'utente nelle impostazioni: non si tocca
            raise EngineError(
                f"Il modello nella cartella «{path}» è danneggiato o incompleto: sostituirlo oppure scegliere "
                "un altro modello locale nelle impostazioni."
            ) from cause
        key = (self.model_name, str(self.cache_dir))
        with _MODELS_LOCK:
            again = key in _REPLACED
            _REPLACED.add(key)
        if again or not _discard_cached_model(repo):
            # gia' riscaricato in questa sessione (il problema non e' il download) o non eliminabile
            raise EngineError(_damaged_message(repo)) from cause
        log.warning("Modello %s danneggiato in %s: eliminato, nuovo download", self.model_name, repo)
        return download_model(self.model_name, self.cache_dir, lambda f, m: progress(0.25 * f, m),
                              message="Modello danneggiato: nuovo download")

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
            # nessun valore letto: ogni campo resta da verificare (inserimento manuale)
            rows = [DayRow(giorno=g) for g in range(1, 32)]
            header = Header()
            for r in rows:
                _flag_all(r, None)
            _flag_all(None, header)
            return ExtractionResult(
                is_foglio_firma=False,
                header=header,
                rows=rows,
                ocr_notes=TABLE_NOT_FOUND_MSG,
                confidence=round(float(grid.score), 3),
                engine="locale",
                model=self.model_name,
                usage=Usage(seconds=round(time.monotonic() - t0, 2)),
            )
        if grid.row_height < MIN_ROW_HEIGHT:
            # sotto questa soglia TrOCR "legge" valori anche nelle celle vuote
            log.info("Righe alte %.1f pixel: risoluzione insufficiente", grid.row_height)
            raise EngineError(LOW_RES_MSG)
        recognizer = self._ensure_recognizer(progress)
        sp = self._second_opinion_reader(progress) if self.second_opinion else None
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
        except Exception as exc:  # noqa: BLE001
            # torch su CPU segnala la mancanza di memoria con un RuntimeError
            if _is_oom(exc):
                log.warning("Memoria insufficiente durante la lettura", exc_info=True)
                raise EngineError(OOM_MSG) from exc
            log.exception("Errore del riconoscimento offline")
            raise EngineError(f"Errore del riconoscimento offline: {exc}") from exc
        readings = {req.key: res for req, res in zip(plan.requests, results)}
        progress(0.95, "Controllo di coerenza dei valori letti…")
        reasons: dict[tuple[int, str], set[str]] = {}
        result = assemble(plan, readings, grid, reasons_out=reasons)
        if self.second_opinion:
            self._apply_second_opinion(sp, img, grid, plan, result, reasons, progress)
        result.model = self.model_name
        result.usage = Usage(seconds=round(time.monotonic() - t0, 2))
        progress(1.0, "Lettura completata.")
        return result

    # --------------------------------------------------------- secondo parere
    def _second_opinion_reader(self, progress: ProgressFn) -> Any:
        """Il secondo parere pronto (scaricato e caricato), oppure None con il motivo
        registrato per la sessione (le celle dubbie restano segnalate)."""
        if self._sp is not None:
            return self._sp
        from sirio.engines import secondo_parere  # noqa: PLC0415

        key = str(self.cache_dir)
        with _SP_LOCK:
            if key in _SP_FAILED:
                return None
            sp = secondo_parere.SecondoParere(cache_dir=self.cache_dir)
            ok, msg = sp.is_available()
            if not ok:
                log.info("Secondo parere non disponibile: %s", msg)
                _SP_FAILED[key] = msg
                return None
            try:
                sp.load(lambda f, m: progress(0.26 + 0.04 * max(0.0, min(1.0, f)), m))
            except EngineError as exc:
                log.warning("Secondo parere non disponibile: %s", exc)
                _SP_FAILED[key] = str(exc)
                return None
            except Exception as exc:  # noqa: BLE001 - mai bloccare la lettura per il secondo parere
                log.exception("Secondo parere non caricabile")
                _SP_FAILED[key] = f"Secondo parere offline non disponibile: {exc}"
                return None
        self._sp = sp
        return sp

    def _apply_second_opinion(self, sp: Any, img: np.ndarray, grid: TableGrid, plan: PagePlan,
                              result: ExtractionResult, reasons: dict[tuple[int, str], set[str]],
                              progress: ProgressFn) -> None:
        """Rilegge con il secondo parere le celle dubbie di orari/ore e applica
        ``decidi_riga``: conferme (segnalazione tolta) e sostituzioni (restano segnalate)."""
        from sirio.engines import secondo_parere  # noqa: PLC0415

        if sp is None:
            why = _SP_FAILED.get(str(self.cache_dir))
            if why:
                result.ocr_notes = _join_notes(result.ocr_notes, f"Secondo parere non usato: {why}")
            return
        H, W = img.shape[:2]
        m = int(round(SP_MARGINE * grid.row_height))
        crops: list[tuple[np.ndarray, str]] = []
        keys: list[tuple[int, str]] = []
        for row in result.rows:
            g = row.giorno
            if not _row_ok(grid, g):
                continue        # riga non allineata alla griglia: nessuna conferma possibile
            for f in secondo_parere.ROW_FIELDS:
                if f not in row.incerti and f not in row.illeggibili:
                    continue
                if plan.cells.get(g, {}).get(f) != TESTO:
                    continue
                x0, y0, x1, y1 = grid.cell(g, f)
                crop = img[max(0, y0 - m):min(H, y1 + m), max(0, x0 - m):min(W, x1 + m)]
                if crop.size:
                    crops.append((crop, secondo_parere.kind_of(f)))
                    keys.append((g, f))
        if not crops:
            return
        t0 = time.monotonic()
        try:
            pareri = sp.read_many(crops, lambda f, msg: progress(0.95 + 0.04 * max(0.0, min(1.0, f)), msg))
        except Exception as exc:  # noqa: BLE001 - mai perdere la lettura principale
            log.warning("Secondo parere non riuscito", exc_info=True)
            msg = str(exc) if isinstance(exc, EngineError) else "errore imprevisto"
            result.ocr_notes = _join_notes(result.ocr_notes, f"Secondo parere non riuscito ({msg}): "
                                                             "le celle dubbie restano segnalate.")
            return
        by_row: dict[int, dict[str, Any]] = {}
        for (g, f), pa in zip(keys, pareri):
            by_row.setdefault(g, {})[f] = pa
        n_conf = n_sost = 0
        for row in result.rows:
            g = row.giorno
            if g not in by_row:
                continue
            flagged = set(row.incerti) | set(row.illeggibili)
            blocked = {f for f in secondo_parere.ROW_FIELDS
                       if reasons.get((g, f), set()) & SP_BLOCCANTI or plan.flags.get(g, {}).get(f)}
            esito = secondo_parere.decidi_riga({f: getattr(row, f) for f in secondo_parere.ROW_FIELDS}, flagged,
                                               by_row[g], assenza_alunno=row.assenza_alunno, bloccati=blocked)
            for f, (_old, new) in esito.sostituiti.items():
                setattr(row, f, new)
                if f in row.illeggibili:
                    row.illeggibili.remove(f)
                if f not in row.incerti:
                    row.incerti.append(f)
                n_sost += 1
            for f in esito.confermati:
                row.incerti = [x for x in row.incerti if x != f]
                row.confidenza[f] = round(max(row.confidenza.get(f, 0.0), CONF_SECONDO_PARERE), 3)
                n_conf += 1
        log.info("Secondo parere: %d celle rilette in %.1f s, %d confermate, %d corrette",
                 len(crops), time.monotonic() - t0, n_conf, n_sost)
        parts = [f"Secondo parere offline ({secondo_parere.MODEL_LABEL}) sulle celle dubbie: {len(crops)} rilette"]
        if n_conf:
            parts.append(f"{n_conf} confermate")
        if n_sost:
            parts.append(f"{n_sost} corrette (da verificare)")
        result.ocr_notes = _join_notes(result.ocr_notes, ", ".join(parts) + ".")
        _refresh_counts(result)


CONF_SECONDO_PARERE = 0.85   # confidenza di un valore confermato dal secondo parere


def _row_ok(grid: TableGrid, g: int) -> bool:
    row_ok = getattr(grid, "row_reliable", None)
    if not bool(getattr(grid, "reliable", True)):
        return False
    if callable(row_ok):
        try:
            return bool(row_ok(g))
        except (ValueError, TypeError, IndexError):
            return True
    return True


def _join_notes(notes: str | None, extra: str) -> str:
    return f"{notes} {extra}".strip() if notes else extra


_COUNT_RE = re.compile(r"Campi da verificare \(lettura incerta\): \d+\.")


def _refresh_counts(result: ExtractionResult) -> None:
    """Aggiorna nelle note il numero dei campi da verificare (dopo il secondo parere)."""
    if not result.ocr_notes:
        return
    n = sum(len(r.incerti) for r in result.rows) + len(result.header.incerti)
    text = _COUNT_RE.sub(f"Campi da verificare (lettura incerta): {n}." if n else "", result.ocr_notes)
    result.ocr_notes = re.sub(r"\s{2,}", " ", text).strip() or None


__all__ = [
    "DEFAULT_MODEL",
    "KNOWN_MODELS",
    "Evidence",
    "Lexicon",
    "LocalEngine",
    "PagePlan",
    "ReadRequest",
    "Reading",
    "Recognizer",
    "TrOCRRecognizer",
    "assemble",
    "calendar_penalty",
    "clean_crop",
    "date_lexicon",
    "default_cache_dir",
    "download_model",
    "evidence",
    "free_value",
    "header_spots",
    "hours_lexicon",
    "local_model_path",
    "model_image",
    "month_year_lexicon",
    "notes_lexicon",
    "pei_lexicon",
    "plan_page",
    "reconcile_row",
    "row_consistency",
    "small_int_lexicon",
    "stamp_boxes",
    "time_lexicon",
    "total_lexicon",
]
