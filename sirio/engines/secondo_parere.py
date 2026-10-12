"""Secondo parere offline sulle celle dubbie del motore locale (GLM-OCR, solo CPU).

Il motore locale (TrOCR) segnala come ``incerti`` i campi che non riesce a
leggere con sicurezza: su fogli reali di operatori diversi sono 20-40 celle per
pagina, e la maggior parte e' in realta' letta bene (falsi allarmi). Questo
modulo rilegge *solo* quelle celle con un secondo riconoscitore indipendente,
`GLM-OCR <https://huggingface.co/zai-org/GLM-OCR>`_ (~1 miliardo di parametri,
licenza MIT, ~2,65 GB scaricati una sola volta), e offre una politica prudente
per usarne le letture. Richiede almeno ``RAM_MINIMA_GB`` (12 GB) di RAM: il modello
in float32 occupa ~4,5 GB (picco ~6,8 GB durante il caricamento).

Uso previsto (l'integrazione nel motore e' a carico di ``local_engine``)::

    sp = SecondoParere()                       # cache: config.data_dir()/"modelli"
    ok, msg = sp.is_available()
    sp.load(progress)                          # scarica (prima volta) e carica il modello
    pareri = sp.read_many([(ritaglio, "orario"), (ritaglio2, "ore"), ...])
    esito = decidi_riga(valori_riga, segnalati, {campo: parere, ...},
                        assenza_alunno=..., bloccati=...)

Ritagli: la regione *grezza* della cella (``grid.cell(giorno, campo)``) con un
margine di ~0,2 altezze di riga, in BGR o in grigi. GLM-OCR legge meglio la cella
originale (linee della griglia comprese) che il ritaglio "pulito" preparato per
TrOCR (misurato: 85 contro 77 letture giuste su 160 celle dubbie).

Politica (``decidi_riga``), pensata perche' il secondo parere non possa mai
rendere *sicuro* un valore sbagliato:

1. **Conferma** (toglie la segnalazione): solo orari e ore; il secondo parere
   legge *lo stesso valore* del motore con probabilita' minima per token
   >= ``SOGLIA_CONFERMA``; il valore e' plausibile (orario 7:00-18:00, ore a quarti
   d'ora fino a 10); e una relazione *esatta* della riga lo conferma: programmato =
   effettivo con la cella gemella letta sicura dal motore, oppure uscita - entrata
   = ore con celle sicure o confermate anch'esse dal secondo parere (cifre diverse:
   un errore correlato e' molto meno probabile che fra due celle gemelle, dove la
   stessa grafia puo' ingannare allo stesso modo entrambi i modelli: "14:00" letto
   "16:00" da tutti e due in programmato ed effettivo). Un errore dovrebbe quindi
   ripetersi identico in due modelli indipendenti *e* tornare con le celle collegate.
   Mai per i campi in ``bloccati`` (segnalati per motivi strutturali: riga non
   allineata alla griglia, cella barrata, orario fuori fascia, riga incoerente),
   ne' per le note (due letture "simili" possono essere entrambe sbagliate: es.
   PONTE EARN. letto da entrambi i modelli per PONTE CARN.).
2. **Sostituzione** (il campo *resta* segnalato): per le celle segnalate di una
   riga si sceglie tra la lettura del motore e quella del secondo parere la
   combinazione che soddisfa piu' relazioni esatte; a parita' vince il motore.

Misure sulle pagine reali di sviluppo (5 pagine, 145 celle dubbie di orari/ore
rilette): 29 segnalazioni tolte, tutte a valori giusti; 6 valori sbagliati
corretti per sostituzione (restano segnalati), nessun valore giusto peggiorato;
0 conferme in 1244 prove avversarie (valori del motore alterati in modo
plausibile). Circa 0,6 s per cella con 4 thread e 1,1 s con 2 thread (lotti da 8,
CPU limitata ad AVX2), cioe' 10-50 s per pagina.

Dipendenze: ``torch`` e ``transformers`` >= 5 con il modello ``glm_ocr`` (gli
stessi pacchetti del motore locale); ``torchvision`` NON serve (si usa il
preprocessore PIL). Se mancano, ``is_available()`` restituisce ``False`` con un
messaggio in italiano e nessuna funzione solleva ``ImportError``.
"""

from __future__ import annotations

import importlib
import importlib.util
import itertools
import logging
import math
import os
import re
import shutil
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sirio.engines.base import EngineError, ProgressFn, no_progress
from sirio.validation import hours_between, parse_hours, parse_time

log = logging.getLogger(__name__)

# ==========================================================================
# Modello
# ==========================================================================

MODEL_ID = "zai-org/GLM-OCR"
# revisione fissata: il preprocessing (prompt, token immagine) e' verificato su questa
MODEL_REVISION = "2e85a62840ccac27daa451df36c736c4636b8628"
MODEL_LABEL = "GLM-OCR"
DOWNLOAD_MB = 2650
PROMPT = "Text Recognition:"
_ALLOW_PATTERNS = ["*.json", "*.jinja", "*.safetensors", "*.txt"]
_REQUIRED_FILES = ("config.json", "preprocessor_config.json", "tokenizer.json", "chat_template.jinja")
_DISK_MARGIN = 200e6

KINDS = ("orario", "ore", "nota", "numero")
# token massimi generati per tipo di cella (le letture lunghe sono comunque scartate)
_MAX_TOKENS = {"orario": 10, "ore": 8, "numero": 8, "nota": 28}

_MISSING_MSG = (
    "Secondo parere offline non disponibile: servono i pacchetti «torch» e «transformers» "
    "(versione 5 o successiva) del motore locale."
)
_OLD_TRANSFORMERS_MSG = (
    "Secondo parere offline non disponibile: la versione installata di «transformers» non "
    "supporta il modello GLM-OCR (serve la versione 5 o successiva)."
)
RAM_MINIMA_GB = 12.0          # il modello in float32 occupa ~4,5 GB (picco ~6,8 GB al caricamento)
_RAM_MSG = (
    "Secondo parere offline non attivo: servono almeno {min:.0f} GB di RAM (questo computer ne ha "
    "{tot:.0f} GB); le celle dubbie restano segnalate."
)
_OOM_MSG = (
    "Memoria insufficiente per il secondo parere offline (servono circa 4 GB liberi): "
    "le celle dubbie restano segnalate."
)

SOGLIA_CONFERMA = 0.5        # probabilita' minima per token della lettura che conferma
SOGLIA_SOSTITUZIONE = 0.0    # la sostituzione lascia il campo segnalato: nessuna soglia
ORA_MIN, ORA_MAX = 7 * 60, 18 * 60
ORE_MAX = 10.0

TIME_FIELDS = ("prog_entrata", "prog_uscita", "eff_entrata", "eff_uscita")
ROW_FIELDS = TIME_FIELDS + ("ore_dichiarate",)


def kind_of(campo: str) -> str:
    """Tipo di lettura per un campo di ``DayRow`` / ``Header``."""
    if campo in TIME_FIELDS:
        return "orario"
    if campo == "ore_dichiarate":
        return "ore"
    if campo == "note":
        return "nota"
    return "numero"


# ==========================================================================
# Interpretazione delle letture
# ==========================================================================

@dataclass
class Parere:
    """Lettura del secondo riconoscitore per una cella."""

    testo: str = ""                 # trascrizione letterale (ripulita)
    valore: Any = None              # "HH:MM" | float | str | None
    stato: str = "illeggibile"      # "valore" | "vuota" | "illeggibile"
    confidenza: float = 0.0         # probabilita' minima per token (0..1)

    @property
    def ha_valore(self) -> bool:
        return self.stato == "valore" and self.valore is not None


_EMPTY = re.compile(r"^(empty|none|vuot[ao]|null|n/?a|markdown|[-–—_/\\|.,:;~\s]*)$", re.IGNORECASE)
_FENCE = re.compile(r"```[a-zA-Z]*")
# lettere che nei numeri scritti a mano sono quasi sempre cifre
_DIGIT_LIKE = str.maketrans({"O": "0", "o": "0", "D": "0", "l": "1", "I": "1", "|": "1", ";": ":"})


def clean_text(text: str) -> str:
    t = _FENCE.sub(" ", text or "")
    t = t.replace("$", " ").replace("\n", " ").strip().strip("`\"'").strip()
    return re.sub(r"\s+", " ", t)


def _time_value(text: str) -> str | None:
    s = text.translate(_DIGIT_LIKE)
    if re.search(r"[A-Za-z]", s.replace("h", "")):
        return None                     # lettere rimaste: non e' un orario leggibile
    s = re.sub(r"[^0-9:.,h]", "", s)
    if not s or len(re.sub(r"\D", "", s)) > 4:
        return None
    t = parse_time(s)
    if t is None:
        return None
    return f"{t[0]:02d}:{t[1]:02d}"


def _hours_value(text: str, massimo: float) -> float | None:
    s = text.translate(_DIGIT_LIKE).replace("’", "'").strip()
    if re.search(r"[A-Za-gi-z]", s):    # solo "h" ammessa (2h30')
        return None
    v = parse_hours(s, massimo=massimo)
    return None if v is None else float(v)


def interpreta(kind: str, testo: str) -> tuple[str, Any]:
    """(stato, valore) di una trascrizione per il tipo di cella.

    stato: ``"valore"`` (valore interpretato), ``"vuota"`` (niente, trattino, barra),
    ``"illeggibile"`` (testo presente ma non interpretabile per il tipo)."""
    t = clean_text(testo)
    if _EMPTY.match(t):
        return "vuota", None
    if kind == "orario":
        v = _time_value(t)
        return ("valore", v) if v is not None else ("illeggibile", None)
    if kind in ("ore", "numero"):
        v = _hours_value(t, 24.0 if kind == "ore" else 744.0)
        return ("valore", v) if v is not None else ("illeggibile", None)
    letters = sum(c.isalnum() for c in t)
    if letters < 2:
        return "vuota", None
    return "valore", t.upper()


def plausibile(kind: str, valore: Any) -> bool:
    """Valore compatibile con un foglio firma scolastico (vincoli generici)."""
    if valore is None:
        return False
    if kind == "orario":
        t = parse_time(str(valore))
        if t is None:
            return False
        m = t[0] * 60 + t[1]
        return ORA_MIN <= m <= ORA_MAX and t[1] % 5 == 0
    if kind == "ore":
        v = float(valore)
        return 0 < v <= ORE_MAX and abs(v * 4 - round(v * 4)) < 1e-6
    return True


def stesso_valore(kind: str, a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if kind in ("ore", "numero"):
        try:
            return abs(float(a) - float(b)) < 0.01
        except (TypeError, ValueError):
            return False
    if kind == "orario":
        ta, tb = parse_time(str(a)), parse_time(str(b))
        return ta is not None and ta == tb
    return re.sub(r"[^A-Z0-9]", "", str(a).upper()) == re.sub(r"[^A-Z0-9]", "", str(b).upper())


# ==========================================================================
# Politica di riga
# ==========================================================================

@dataclass
class EsitoRiga:
    """Risultato di ``decidi_riga``."""

    valori: dict[str, Any] = field(default_factory=dict)       # valori della riga dopo le sostituzioni
    segnalati: set[str] = field(default_factory=set)            # campi che restano incerti
    confermati: set[str] = field(default_factory=set)           # segnalazioni tolte
    sostituiti: dict[str, tuple[Any, Any]] = field(default_factory=dict)  # campo -> (motore, secondo parere)


def _shift(t: Any, hours: Any, sign: int) -> str | None:
    if t is None or hours is None:
        return None
    p = parse_time(str(t))
    if p is None:
        return None
    m = p[0] * 60 + p[1] + sign * round(float(hours) * 60)
    return f"{m // 60:02d}:{m % 60:02d}" if 0 <= m < 24 * 60 else None


def relazioni(campo: str, valori: Mapping[str, Any], assenza_alunno: bool = False,
              uguaglianze: bool = True, calcoli: bool = True) -> list[Any]:
    """Valori che le relazioni esatte della riga impongono a ``campo`` (solo quelle
    calcolabili con i valori presenti in ``valori``): ``uguaglianze`` = programmato
    uguale all'effettivo; ``calcoli`` = uscita - entrata = ore (non con assenza alunno)."""
    pe, pu, ee, eu, ore = (valori.get(k) for k in ROW_FIELDS)
    out: list[Any] = []
    calc = calcoli and not assenza_alunno
    if campo == "eff_entrata":
        out += [pe if uguaglianze else None, _shift(eu, ore, -1) if calc else None]
    elif campo == "eff_uscita":
        out += [pu if uguaglianze else None, _shift(ee, ore, +1) if calc else None]
    elif campo == "prog_entrata" and uguaglianze:
        out.append(ee)
    elif campo == "prog_uscita" and uguaglianze:
        out.append(eu)
    elif campo == "ore_dichiarate" and calc:
        out.append(hours_between(ee, eu))
    return [x for x in out if x is not None]


def _row_score(v: Mapping[str, Any], assenza_alunno: bool) -> int:
    """Quante relazioni esatte la riga soddisfa (penalita' per uscita <= entrata)."""
    pe, pu, ee, eu, ore = (v.get(k) for k in ROW_FIELDS)
    s = 0
    for a, b in ((pe, ee), (pu, eu)):
        if a and b and stesso_valore("orario", a, b):
            s += 1
    for a, b in ((pe, pu), (ee, eu)):
        if a and b and not hours_between(a, b):
            s -= 3
    if ore is not None and not assenza_alunno:
        if ee and eu:
            h = hours_between(ee, eu)
            s += 2 if h is not None and abs(h - float(ore)) < 0.01 else -1
        elif pe and pu:
            h = hours_between(pe, pu)
            s += 1 if h is not None and abs(h - float(ore)) < 0.01 else 0
    return s


def decidi_riga(valori: Mapping[str, Any], segnalati: Iterable[str], pareri: Mapping[str, Parere],
                assenza_alunno: bool = False, bloccati: Iterable[str] = ()) -> EsitoRiga:
    """Applica la politica del secondo parere a una riga del foglio.

    ``valori``: valori del motore per i campi della riga (``prog_entrata`` ...
    ``ore_dichiarate``; gli altri sono ignorati). ``segnalati``: campi incerti o
    illeggibili secondo il motore. ``pareri``: letture del secondo parere per (alcuni)
    campi segnalati. ``bloccati``: campi la cui segnalazione non va mai tolta (riga
    non affidabile, cella barrata, motivi strutturali). Vedi la docstring del modulo."""
    flagged = {f for f in segnalati if f in ROW_FIELDS}
    blocked = set(bloccati)
    base = {f: valori.get(f) for f in ROW_FIELDS}
    esito = EsitoRiga(valori=dict(base), segnalati=set(segnalati))
    # --- 1. sostituzione congiunta (il campo resta segnalato)
    fields = sorted(flagged)
    options: list[list[Any]] = []
    for f in fields:
        k, p = kind_of(f), pareri.get(f)
        opts = [base[f]]
        if (p is not None and p.ha_valore and p.confidenza >= SOGLIA_SOSTITUZIONE and plausibile(k, p.valore)
                and not stesso_valore(k, p.valore, base[f])):
            opts.append(p.valore)
        options.append(opts)
    if any(len(o) > 1 for o in options):
        best: tuple[tuple[int, int], tuple[int, ...]] | None = None
        for choice in itertools.product(*(range(len(o)) for o in options)):
            v = dict(base)
            for f, o, j in zip(fields, options, choice):
                v[f] = o[j]
            key = (_row_score(v, assenza_alunno), -sum(choice))
            if best is None or key > best[0]:
                best = (key, choice)
        assert best is not None
        for f, o, j in zip(fields, options, best[1]):
            if j:
                esito.sostituiti[f] = (base[f], o[j])
                esito.valori[f] = o[j]
    # --- 2. conferma: accordo + plausibilita' + relazione esatta con celle fidate
    agreed = set()
    for f in flagged:
        k, p = kind_of(f), pareri.get(f)
        if (f not in esito.sostituiti and p is not None and p.ha_valore and p.confidenza >= SOGLIA_CONFERMA
                and stesso_valore(k, p.valore, base[f]) and plausibile(k, base[f])):
            agreed.add(f)
    # Uguaglianze (programmato = effettivo) solo con celle che il motore ha letto sicure:
    # la stessa grafia letta male allo stesso modo da entrambi i modelli in due celle
    # gemelle si "confermerebbe" da sola (es. 14:00 letto 16:00 in entrambe le colonne).
    # I calcoli (uscita - entrata = ore) coinvolgono cifre diverse e ammettono anche
    # celle segnalate ma confermate dal secondo parere.
    solid = {f: (base[f] if f not in flagged else None) for f in ROW_FIELDS}
    trusted = {f: (base[f] if (f not in flagged or f in agreed) else None) for f in ROW_FIELDS}
    for f in sorted(agreed - blocked):
        k = kind_of(f)
        eq = relazioni(f, {**solid, f: None}, assenza_alunno, calcoli=False)
        calc = relazioni(f, {**trusted, f: None}, assenza_alunno, uguaglianze=False)
        if any(stesso_valore(k, x, base[f]) for x in eq + calc):
            esito.confermati.add(f)
    esito.segnalati -= esito.confermati
    return esito


# ==========================================================================
# Download e caricamento
# ==========================================================================

def default_cache_dir() -> Path:
    from sirio.config import data_dir  # noqa: PLC0415

    return data_dir() / "modelli"


def _repo_dir(cache_dir: Path) -> Path:
    return cache_dir / ("models--" + MODEL_ID.replace("/", "--"))


def _complete(path: Path) -> bool:
    try:
        names = {p.name for p in path.iterdir()}
    except OSError:
        return False
    return all(n in names for n in _REQUIRED_FILES) and any(n.endswith(".safetensors") for n in names)


def model_path(cache_dir: Path) -> Path | None:
    """Cartella del modello se gia' scaricato completo nella cache, altrimenti None."""
    for base in (cache_dir, cache_dir / "hub"):
        snap = _repo_dir(base) / "snapshots" / MODEL_REVISION
        if snap.is_dir() and _complete(snap):
            return snap
    return None


def _gb(mb: float) -> str:
    return f"{mb / 1000:.1f}".replace(".", ",")


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _free_bytes(path: Path) -> float | None:
    for p in (path, *path.parents):
        try:
            return float(shutil.disk_usage(p).free)
        except OSError:
            continue
    return None


def _is_oom(exc: BaseException) -> bool:
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, MemoryError):
            return True
        text = str(cur).lower()
        if any(t in text for t in ("out of memory", "not enough memory", "cannot allocate memory",
                                   "can't allocate memory", "defaultcpuallocator", "bad_alloc",
                                   "paging file", "(os error 12)", "(os error 1455)")):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _silent_tqdm() -> Any:
    from tqdm.auto import tqdm  # noqa: PLC0415

    class _Silent(tqdm):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

    return _Silent


def ram_totale_gb() -> float | None:
    """RAM fisica totale in GB (None se non determinabile). Senza dipendenze esterne."""
    try:
        if os.name == "nt":
            import ctypes  # noqa: PLC0415

            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):  # type: ignore[attr-defined]
                return None
            return st.ullTotalPhys / 1e9
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (AttributeError, OSError, ValueError):
        return None


def dependencies_missing() -> str | None:
    """None se torch/transformers (con GLM-OCR e il preprocessore PIL) sono utilizzabili,
    altrimenti il messaggio per l'utente. Non solleva eccezioni."""
    for mod in ("torch", "transformers", "PIL"):
        if importlib.util.find_spec(mod) is None:
            return _MISSING_MSG
    try:
        importlib.import_module("torch")
        auto = importlib.import_module("transformers.models.auto.configuration_auto")
    except Exception:  # noqa: BLE001 - libreria presente ma non funzionante
        log.warning("torch/transformers non importabili per il secondo parere", exc_info=True)
        return _MISSING_MSG
    if "glm_ocr" not in getattr(auto, "CONFIG_MAPPING_NAMES", {}):
        return _OLD_TRANSFORMERS_MSG
    if importlib.util.find_spec("transformers.models.glm46v.image_processing_pil_glm46v") is None:
        return _OLD_TRANSFORMERS_MSG
    return None


def _to_pil(crop: np.ndarray) -> Any:
    from PIL import Image  # noqa: PLC0415

    a = np.asarray(crop)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    elif a.ndim == 3 and a.shape[2] == 4:
        a = a[:, :, :3]
    if a.ndim == 3 and a.shape[2] == 3:
        a = a[:, :, ::-1]               # BGR (OpenCV) -> RGB
    h, w = a.shape[:2]
    if min(h, w) < 16:                  # ritagli minuscoli: il modello vuole almeno 2x2 patch
        pad_h, pad_w = max(0, 16 - h), max(0, 16 - w)
        a = np.pad(a, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
                   constant_values=255)
    return Image.fromarray(np.ascontiguousarray(a))


class _Reader:
    """Modello caricato: tokenizer, preprocessore PIL e rete in float32."""

    def __init__(self, path: Path):
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        ipm = importlib.import_module("transformers.models.glm46v.image_processing_pil_glm46v")
        try:
            transformers.utils.logging.disable_progress_bar()
        except Exception:  # noqa: BLE001 - facoltativo
            log.debug("Barre di avanzamento di transformers non disattivabili", exc_info=True)
        self.torch = torch
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.tokenizer.padding_side = "left"
        self.images = ipm.Glm46VImageProcessorPil.from_pretrained(str(path), local_files_only=True)
        template = (path / "chat_template.jinja").read_text(encoding="utf-8")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        self.prompt = self.tokenizer.apply_chat_template(msgs, chat_template=template,
                                                         add_generation_prompt=True, tokenize=False)
        if self.prompt.count("<|image|>") != 1:
            raise EngineError("Modello GLM-OCR non compatibile (modello di prompt inatteso).")
        self.model = transformers.AutoModelForImageTextToText.from_pretrained(
            str(path), dtype=torch.float32, local_files_only=True)
        self.model.eval()
        self.merge = int(getattr(self.images, "merge_size", 2) or 2)
        eos = self.model.generation_config.eos_token_id
        self.stop = set(eos if isinstance(eos, (list, tuple)) else [eos])
        if self.tokenizer.pad_token_id is not None:
            self.stop.add(self.tokenizer.pad_token_id)

    def read(self, crops: Sequence[np.ndarray], max_tokens: int) -> list[tuple[str, float]]:
        torch = self.torch
        ims = [_to_pil(c) for c in crops]
        enc_img = self.images(images=ims, return_tensors="pt")
        thw = enc_img["image_grid_thw"]
        texts = [self.prompt.replace("<|image|>", "<|image|>" * int(t.prod().item() // self.merge ** 2))
                 for t in thw]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True)
        with torch.inference_mode():
            out = self.model.generate(**enc, pixel_values=enc_img["pixel_values"], image_grid_thw=thw,
                                      max_new_tokens=max_tokens, do_sample=False, num_beams=1,
                                      output_scores=True, return_dict_in_generate=True)
        seq = out.sequences[:, enc["input_ids"].shape[1]:]
        scores = torch.stack(out.scores, 1).float().log_softmax(-1)
        res: list[tuple[str, float]] = []
        for b in range(seq.shape[0]):
            ids, lps = [], []
            for t, i in enumerate(seq[b].tolist()):
                if i in self.stop:
                    break
                ids.append(i)
                lps.append(float(scores[b, t, i]))
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            res.append((text, math.exp(min(lps)) if lps else 0.0))
        return res


_LOCK = threading.Lock()
_READERS: dict[str, _Reader] = {}


class SecondoParere:
    """Secondo riconoscitore offline per le celle dubbie (GLM-OCR su CPU).

    Il modello si scarica alla prima ``load`` in ``cache_dir`` (predefinita
    ``config.data_dir()/"modelli"``) e resta in memoria per il processo (condiviso
    tra istanze con la stessa cache). Thread-safe."""

    def __init__(self, cache_dir: Path | None = None, batch_size: int = 8, threads: int | None = None,
                 min_ram_gb: float = RAM_MINIMA_GB):
        self._cache_dir = cache_dir
        self.min_ram_gb = min_ram_gb
        self.batch_size = max(1, int(batch_size))
        self.threads = threads
        self._reader: _Reader | None = None

    @property
    def cache_dir(self) -> Path:
        if self._cache_dir is None:
            self._cache_dir = default_cache_dir()
        return self._cache_dir

    # ------------------------------------------------------------ disponibilita'
    def is_downloaded(self) -> bool:
        return model_path(self.cache_dir) is not None

    def is_available(self) -> tuple[bool, str]:
        if self._reader is not None:
            return True, "Secondo parere offline pronto."
        missing = dependencies_missing()
        if missing:
            return False, missing
        ram = ram_totale_gb()
        if ram is not None and ram < self.min_ram_gb:
            return False, _RAM_MSG.format(min=self.min_ram_gb, tot=ram)
        if self.is_downloaded():
            return True, f"Secondo parere offline pronto: modello {MODEL_LABEL} già scaricato."
        return True, (f"Secondo parere offline disponibile: al primo utilizzo verrà scaricato il modello "
                      f"{MODEL_LABEL} (circa {_gb(DOWNLOAD_MB)} GB, una sola volta).")

    # ---------------------------------------------------------------- modello
    def download(self, progress: ProgressFn = no_progress) -> Path:
        """Scarica il modello (se manca) riportando l'avanzamento; restituisce la cartella."""
        path = model_path(self.cache_dir)
        if path is not None:
            return path
        if dependencies_missing():
            raise EngineError(dependencies_missing() or _MISSING_MSG)
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415
        except ImportError as exc:
            raise EngineError(_MISSING_MSG) from exc
        cache = self.cache_dir
        cache.mkdir(parents=True, exist_ok=True)
        blobs = _repo_dir(cache) / "blobs"
        total = DOWNLOAD_MB * 1e6
        start = _dir_size(blobs) if blobs.exists() else 0
        free = _free_bytes(cache)
        if free is not None and free < (total - start) * 1.1 + _DISK_MARGIN:
            raise EngineError(
                f"Spazio su disco insufficiente per il modello {MODEL_LABEL} del secondo parere "
                f"(servono circa {_gb(DOWNLOAD_MB)} GB liberi).")
        outcome: dict[str, Any] = {}

        def worker() -> None:
            try:
                outcome["path"] = snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(cache),
                                                    allow_patterns=_ALLOW_PATTERNS, tqdm_class=_silent_tqdm())
            except BaseException as exc:  # noqa: BLE001 - riportata nel thread chiamante
                outcome["error"] = exc

        head = f"Download del modello {MODEL_LABEL} per il secondo parere (solo al primo utilizzo)"
        progress(0.0, head + "…")
        th = threading.Thread(target=worker, name="sirio-download-secondo-parere", daemon=True)
        th.start()
        while th.is_alive():
            th.join(0.5)
            done = max(0, (_dir_size(blobs) if blobs.exists() else 0) - start)
            progress(min(0.99, done / total), f"{head}: {done / 1e6:.0f} MB di circa {total / 1e6:.0f} MB…")
        if "error" in outcome:
            exc = outcome["error"]
            log.warning("Download del secondo parere non riuscito", exc_info=exc)
            raise EngineError(
                f"Download del modello {MODEL_LABEL} per il secondo parere non riuscito (connessione a "
                f"Internet assente o interrotta?): le celle dubbie restano segnalate.") from exc
        path = model_path(cache)
        if path is None:
            raise EngineError(f"Download del modello {MODEL_LABEL} incompleto: riprovare.")
        progress(1.0, f"Modello {MODEL_LABEL} scaricato.")
        return path

    def load(self, progress: ProgressFn = no_progress) -> None:
        """Scarica (se serve) e carica il modello. Solleva ``EngineError`` con messaggio italiano."""
        if self._reader is not None:
            return
        ok, msg = self.is_available()
        if not ok:
            raise EngineError(msg)
        key = str(self.cache_dir)
        with _LOCK:
            reader = _READERS.get(key)
            if reader is None:
                path = self.download(progress)
                progress(0.0, f"Caricamento del modello {MODEL_LABEL} (secondo parere)…")
                try:
                    reader = _Reader(path)
                except EngineError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    if _is_oom(exc):
                        raise EngineError(_OOM_MSG) from exc
                    log.warning("Modello del secondo parere non caricabile", exc_info=True)
                    raise EngineError(f"Impossibile caricare il modello {MODEL_LABEL} del secondo parere.") from exc
                _READERS[key] = reader
        self._reader = reader

    def close(self) -> None:
        """Libera il modello (per esempio per ridare memoria al motore principale)."""
        with _LOCK:
            _READERS.pop(str(self.cache_dir), None)
        self._reader = None

    # ---------------------------------------------------------------- lettura
    def read_many(self, requests: Sequence[tuple[np.ndarray, str]],
                  progress: ProgressFn = no_progress) -> list[Parere]:
        """Legge i ritagli ``(immagine, tipo)`` (tipo in ``KINDS``) a lotti; stesso ordine."""
        if self._reader is None:
            self.load(progress)
        reader = self._reader
        assert reader is not None
        for _img, kind in requests:
            if kind not in KINDS:
                raise ValueError(f"tipo di cella sconosciuto: {kind!r}")
        out: list[Parere | None] = [None] * len(requests)
        # lotti per tipo e per area simile (meno riempimento)
        order = sorted(range(len(requests)), key=lambda i: (requests[i][1], int(np.asarray(requests[i][0]).size)))
        groups = [list(g) for _k, g in itertools.groupby(order, key=lambda i: requests[i][1])]
        batches = [g[i:i + self.batch_size] for g in groups for i in range(0, len(g), self.batch_size)]
        torch = reader.torch
        old_threads = torch.get_num_threads()
        if self.threads:
            torch.set_num_threads(int(self.threads))
        try:
            for n, batch in enumerate(batches):
                kind = requests[batch[0]][1]
                try:
                    texts = reader.read([requests[i][0] for i in batch], _MAX_TOKENS[kind])
                except Exception as exc:  # noqa: BLE001
                    if _is_oom(exc):
                        raise EngineError(_OOM_MSG) from exc
                    raise
                for i, (text, conf) in zip(batch, texts):
                    stato, valore = interpreta(kind, text)
                    out[i] = Parere(testo=clean_text(text), valore=valore, stato=stato, confidenza=float(conf))
                progress((n + 1) / len(batches), f"Secondo parere sulle celle dubbie: {n + 1}/{len(batches)}…")
        finally:
            if self.threads:
                torch.set_num_threads(old_threads)
        return [p if p is not None else Parere() for p in out]

    def read(self, crop: np.ndarray, kind: str, context: Mapping[str, Any] | None = None) -> tuple[str | None, float]:
        """Legge un solo ritaglio. Restituisce (valore normalizzato come testo, confidenza 0..1):
        ``"HH:MM"`` per gli orari, ``"2.5"`` per ore/numeri, il testo per le note; ``None``
        se la cella e' vuota o non interpretabile. ``context`` e' accettato per
        compatibilita' (il modello attuale non usa contesto: la riga si valuta con
        ``decidi_riga``)."""
        del context
        p = self.read_many([(crop, kind)])[0]
        if not p.ha_valore:
            return None, p.confidenza
        v = p.valore
        return (f"{v:g}" if isinstance(v, float) else str(v)), p.confidenza

